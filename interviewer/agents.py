"""Modular multi-agent interviewer.

Each step of the interview is handled by a small agent with a task-specific
prompt (cf. the modular, multi-persona design in Chopra & Haaland 2023 and
Wuttke et al. 2024):

    SafetyAgent      - screens each answer (distress, abuse, off-topic)
    ProbeAgent       - decides: ask a follow-up, or move to the next topic?
    InterviewerAgent - writes follow-ups and the reality-check question
    BridgeAgent      - when what someone says democracy IS conflicts with what
                       they would DO, invites them (neutrally) to relate the two

Scripted main questions are asked verbatim so every respondent receives the
same stimulus; the LLM only writes the adaptive parts. Every agent has a
deterministic mock so the pipeline runs offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMClient

INTERVIEW_GUIDELINES = """\
You follow best practice for semi-structured qualitative interviews:
- Ask ONE short, open question at a time (max ~30 words).
- Be neutral and non-leading: never suggest an answer, never agree or disagree,
  never introduce new arguments or facts, never judge the respondent.
- Use the respondent's own words when asking for elaboration.
- Ask for concrete examples, reasons ("why?"), or clarification of vague terms.
- A brief neutral acknowledgement ("Thank you.", "I see.") is fine; praise is not.
"""


def _convo(exchange: list[dict]) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in exchange)


@dataclass
class ProbeDecision:
    action: str          # "follow_up" | "next"
    reason: str
    depth: int           # 1 = superficial, 2 = some elaboration, 3 = well elaborated


class SafetyAgent:
    DISTRESS = re.compile(r"\b(kill myself|suicide|hurt myself|want to die)\b", re.I)
    ABUSE = re.compile(r"\b(fuck|idiot|stupid bot)\b", re.I)
    SYSTEM = (
        "You screen answers in a research interview about democracy. Classify the "
        "respondent's message into exactly one label: OK, DISTRESS (signs of personal "
        "crisis or self-harm), ABUSE (insults/harassment), or OFF_TOPIC (deliberately "
        "unrelated or nonsense). Political criticism, strong opinions and anger at "
        'politicians are OK. Return {"label": "..."}.'
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def check(self, text: str) -> str:
        if self.DISTRESS.search(text):          # hard rule first, never left to the model
            return "DISTRESS"
        if self.llm.is_mock:
            return "ABUSE" if self.ABUSE.search(text) else "OK"
        out = self.llm.chat_json(self.SYSTEM, f"Message: {text}", default={"label": "OK"})
        label = str(out.get("label", "OK")).upper() if isinstance(out, dict) else "OK"
        return label if label in {"OK", "DISTRESS", "ABUSE", "OFF_TOPIC"} else "OK"


def _with_context(system: str, notes: str) -> str:
    return system + (f"\nContext for this country (follow it): {notes}" if notes else "")


class ProbeAgent:
    SYSTEM = (
        "You are the supervisor of a semi-structured interview about democracy. "
        "Given the research goal of the current topic and the exchange so far, decide "
        "whether the interviewer should ask ONE follow-up question or move on.\n"
        "Ask a follow-up if the answer is vague, very short, uses an abstract term "
        "without explaining it, or gives no reason - and a follow-up would serve the goal. "
        "Move on if the goal is reasonably covered or the respondent signals they have "
        "nothing to add. Also rate how elaborated the answer is (depth 1-3).\n"
        'Return {"action": "follow_up" | "next", "reason": "<short>", "depth": 1|2|3}.'
    )

    def __init__(self, llm: LLMClient, context_notes: str = ""):
        self.llm, self.notes = llm, context_notes

    def decide(self, goal: str, exchange: list[dict], followups_used: int, max_followups: int) -> ProbeDecision:
        last = exchange[-1]["text"] if exchange else ""
        n_words = len(last.split())
        if followups_used >= max_followups:     # hard budget, enforced in code
            return ProbeDecision("next", "follow-up budget used", 2 if n_words > 25 else 1)
        if re.search(r"\b(nothing|no idea|don't know|pass|skip)\b", last, re.I) and n_words < 8:
            return ProbeDecision("next", "respondent has nothing to add", 1)
        if self.llm.is_mock:
            if n_words < 20:
                return ProbeDecision("follow_up", "short answer (mock rule)", 1)
            return ProbeDecision("next", "elaborated answer (mock rule)", 3)
        out = self.llm.chat_json(
            _with_context(self.SYSTEM, self.notes),
            f"Research goal: {goal}\nFollow-ups used: {followups_used}/{max_followups}\n\nExchange:\n{_convo(exchange)}",
            default={"action": "next", "reason": "parse error", "depth": 2},
        )
        out = out if isinstance(out, dict) else {}
        try:
            depth = int(out.get("depth", 2))
        except (TypeError, ValueError):
            depth = 2
        action = out.get("action", "next")
        return ProbeDecision(action if action in {"follow_up", "next"} else "next",
                             str(out.get("reason", "")), min(max(depth, 1), 3))


class InterviewerAgent:
    MOCK_FOLLOWUPS = ["Could you say a bit more about that?", "Why do you think so?",
                      "Could you give me a concrete example?"]

    def __init__(self, llm: LLMClient, context_notes: str = ""):
        self.llm, self.notes = llm, context_notes

    def _system(self) -> str:
        return _with_context("You are a skilled, friendly research interviewer conducting a semi-structured "
                             f"interview about democracy.\n{INTERVIEW_GUIDELINES}\nOutput only what you would say.", self.notes)

    def follow_up(self, goal: str, exchange: list[dict], n: int) -> str:
        if self.llm.is_mock:
            return self.MOCK_FOLLOWUPS[n % len(self.MOCK_FOLLOWUPS)]
        user = f"Research goal of this topic: {goal}\n\nExchange so far:\n{_convo(exchange)}\n\nWrite the next follow-up question."
        return self.llm.chat(self._system(), user).strip().strip('"')

    def reality_question(self, expected_labels: list[str], expectation_answers: list[str], fallback: str) -> str:
        """Ask how well the country delivers exactly the things this respondent expects."""
        if not expected_labels:
            return fallback
        if self.llm.is_mock:
            items = ", ".join(l.lower() for l in expected_labels[:3])
            return f"You mentioned things like {items}. How well does your country do on these today?"
        user = ("The respondent said democracy should deliver the following (their words):\n"
                + "\n".join(f"- {a}" for a in expectation_answers)
                + f"\nCodebook summary: {', '.join(expected_labels)}.\n"
                "Write ONE neutral question asking how well their country delivers these things today, "
                "naming two or three of them in the respondent's own words.")
        return self.llm.chat(self._system(), user).strip().strip('"')


class BridgeAgent:
    """Neutral 'bridge' probe across layers - a measure of attitude strength.

    Whether people reconcile, rationalise or revise when two of their own answers
    pull apart is informative. Because the probe may itself influence answers, it
    is meant for a randomised experimental arm (the session flag `bridge`).
    """
    MOCK = {
        "freedoms": ("Earlier you described press freedom as part of democracy, and in the voting example "
                     "you chose the candidate who brought roads and bridges. How do you see these two together?"),
        "elections": ("Earlier you named elections as part of democracy, and in the second example you said "
                      "you would stay with your party after it postponed the election. How do you see these two together?"),
    }

    def __init__(self, llm: LLMClient, context_notes: str = ""):
        self.llm, self.notes = llm, context_notes

    def question(self, tension: dict, concept_answers: list[str], behaviour_answers: list[str]) -> str:
        if self.llm.is_mock:
            return self.MOCK.get(tension["category"], self.MOCK["elections"])
        system = ("You write a single bridge question for a research interview about democracy. "
                  f"{INTERVIEW_GUIDELINES}\nThe question must quote or paraphrase the respondent's own two "
                  "statements side by side and ask how they see them together. Do NOT say they are inconsistent, "
                  "do NOT imply a right answer, do NOT argue. Output only the question.")
        user = (f"What the respondent said democracy is:\n{chr(10).join(concept_answers)}\n\n"
                f"What they said they would do:\n{chr(10).join(behaviour_answers)}\n\n"
                f"Observed tension (for you only): {tension['explanation']}")
        return self.llm.chat(_with_context(system, self.notes), user).strip().strip('"')
