"""Interview state machine: works through three modules (concept, expectation,
behaviour), adapts with follow-ups, writes the reality-check question from the
respondent's own expectations, and optionally asks a bridge question when what
they say democracy IS conflicts with what they would DO."""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from analysis.annotate import CATS, annotate_transcript
from analysis.context import load_context, signals_discomfort

from .agents import BridgeAgent, InterviewerAgent, ProbeAgent, SafetyAgent
from .llm import LLMClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONNAIRE = ROOT / "config" / "questionnaire.yaml"

MESSAGES = {
    "distress": ("Thank you for telling me. I am an AI and cannot offer support myself, but you do not "
                 "have to continue this interview. If you are in crisis, please reach out to someone you "
                 "trust or a local helpline. Would you like to stop here?"),
    "abuse": "Let's keep our conversation respectful. Shall we continue with the question?",
    "stopped": "Of course - we will stop here. Thank you for your time.",
}
STOP_WORDS = {"stop", "quit", "exit"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InterviewSession:
    def __init__(self, llm: LLMClient | None = None, questionnaire_path: Path | str = DEFAULT_QUESTIONNAIRE,
                 respondent_id: str | None = None, bridge: bool | None = None, seed: int | None = None,
                 context: str | None = None):
        self.llm = llm or LLMClient()
        self.q = yaml.safe_load(Path(questionnaire_path).read_text(encoding="utf-8"))
        self.topics = self.q["topics"]
        self.ctx = load_context(context)          # country context pack (wording, vocabulary, conduct)
        notes = " ".join(x for x in ((self.ctx or {}).get("notes_for_agents", ""), (self.ctx or {}).get("probing_style", "")) if x)
        self.safety, self.probe = SafetyAgent(self.llm), ProbeAgent(self.llm, notes)
        self.interviewer, self.bridger = InterviewerAgent(self.llm, notes), BridgeAgent(self.llm, notes)
        if bridge is None:   # random assignment to the bridge arm (an experiment, not a default)
            bridge = random.Random(seed).random() < float(self.q.get("bridge_probability", 0.5))
        self.bridge = bridge
        self.topic_idx = -1          # -1 = intro not yet answered
        self.followups = 0
        self.finished = False
        self.transcript: list[dict] = []
        self.meta = {
            "respondent_id": respondent_id or uuid.uuid4().hex[:8], "language": "en",
            "questionnaire_version": self.q.get("version"), "bridge_arm": bridge,
            "backend": self.llm.config.backend, "model": None if self.llm.is_mock else self.llm.config.model,
            "started_at": _now(), "synthetic": False, "context": context,
        }

    # ------------------------------------------------------------------ helpers
    @property
    def topic(self) -> dict | None:
        return self.topics[self.topic_idx] if 0 <= self.topic_idx < len(self.topics) else None

    def _say(self, text: str, kind: str, **extra) -> str:
        t = self.topic
        self.transcript.append({"role": "interviewer", "text": text, "type": kind,
                                "topic": t["id"] if t else None, "module": t.get("module") if t else None, **extra})
        return text

    def _answers(self, pred) -> list[str]:
        return [t["text"] for t in self.transcript if t["role"] == "respondent" and pred(t)]

    def _dynamic_question(self, topic: dict) -> str | None:
        """Return the question for a dynamic topic, or None to skip it."""
        ann = annotate_transcript(self.to_dict())
        if topic["dynamic"] == "reality":
            expected = [CATS[c]["label"] for c in ann["layers"]["expectation"]]
            return self.interviewer.reality_question(
                expected, self._answers(lambda t: t.get("topic") in {"expect", "tradeoff"}), topic["question"])
        if topic["dynamic"] == "bridge":
            action = [t for t in ann["tensions"] if t["kind"] == "action"]
            if not self.bridge or not action:
                return None
            return self.bridger.question(action[0], self._answers(lambda t: t.get("module") == "concept"),
                                         self._answers(lambda t: t.get("topic") in {"vote_scenario", "party_scenario"}))
        return topic["question"]

    def _next_topic(self) -> str:
        while True:
            self.topic_idx += 1
            self.followups = 0
            if self.topic_idx >= len(self.topics):
                self.finished = True
                self.meta["ended_at"] = _now()
                return self._say(self.q["outro"], "outro")
            t = self.topic
            q = self._dynamic_question(t) if t.get("dynamic") else (self.ctx or {}).get("vignettes", {}).get(t["id"], t["question"])
            if q is not None:
                return self._say(q, "dynamic_question" if t.get("dynamic") else "main_question")
            # dynamic topic skipped (e.g. no tension for the bridge) -> continue with the next one

    # --------------------------------------------------------------------- API
    def start(self) -> str:
        return self._say(self.q["intro"], "intro")

    def respond(self, text: str) -> str:
        if self.finished:
            return ""
        text = text.strip()
        t = self.topic
        turn = {"role": "respondent", "text": text, "topic": t["id"] if t else None, "module": t.get("module") if t else None}
        self.transcript.append(turn)

        if text.lower() in STOP_WORDS:
            self.finished = True
            self.meta.update(ended_at=_now(), stopped_early=True)
            return self._say(MESSAGES["stopped"], "outro")
        if t is None:                        # answer to the intro -> first question
            return self._next_topic()

        label = self.safety.check(text)
        turn["safety"] = label
        if label == "DISTRESS":
            return self._say(MESSAGES["distress"], "safety")
        if label == "ABUSE":
            return self._say(MESSAGES["abuse"], "safety")

        # sensitive topic + signs of discomfort: respect the refusal, reassure, move on (no probing)
        if self.ctx and t["id"] in self.ctx.get("sensitive_topics", []) and signals_discomfort(text, self.ctx):
            turn["probe"] = {"action": "next", "reason": "sensitive topic: refusal respected", "depth": None}
            self._say(self.ctx["reassurance"].strip(), "reassurance")
            return self.ctx["reassurance"].strip() + "\n\n" + self._next_topic()

        max_fu = t.get("max_followups", self.q.get("max_followups_default", 1))
        exchange = [x for x in self.transcript if x.get("topic") == t["id"]]
        decision = self.probe.decide(t["goal"], exchange, self.followups, max_fu)
        turn["probe"] = {"action": decision.action, "reason": decision.reason, "depth": decision.depth}
        if decision.action == "follow_up":
            q = self.interviewer.follow_up(t["goal"], exchange, self.followups)
            self.followups += 1
            return self._say(q, "follow_up")
        return self._next_topic()

    def to_dict(self) -> dict:
        return {"meta": self.meta, "turns": self.transcript}

    def save(self, folder: Path | str = ROOT / "data" / "transcripts") -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"interview_{self.meta['respondent_id']}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
