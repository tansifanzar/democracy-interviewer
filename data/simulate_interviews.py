"""Run the REAL interview engine against SYNTHETIC respondents, to see how the agents behave.

Before fieldwork, simulated respondents are a cheap way to stress-test an AI
interviewer: does the probe agent follow up where it should, do follow-ups
add anything, how often does the bridge agent fire? Each simulated respondent
answers from a 'home' persona (data/make_synthetic.py), with some answers
borrowed from other personas, some respondents low-effort, and generic
elaborations when the engine asks a follow-up the persona has no answer for.

The agents run in offline mock mode (transparent rules); with an LLM backend
the same engine records the same kind of trace. NOT real data.

    python data/simulate_interviews.py     # -> data/simulated/*.json
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from make_population import BRIDGE_A, SHORT  # noqa: E402
from make_synthetic import PERSONAS  # noqa: E402

from interviewer.engine import InterviewSession  # noqa: E402
from interviewer.llm import LLMClient, LLMConfig  # noqa: E402

ELABORATE = {
    "concept_meaning": ["I mean that people can choose and remove those in power, because otherwise they stop listening.",
                        "For example, being able to criticise the government without fear of arrest.", "Just that, really."],
    "concept_boundary": ["Because without a free press people cannot know what the government is doing.",
                         "Newspapers matter less than jobs, to be honest.", "It depends, I'm not sure."],
    "expect": ["Mostly jobs and fair prices, because families are struggling.", "Honest officials who don't take bribes.",
               "I can't think of anything else."],
    "tradeoff": ["Because development is what people feel every day.",
                 "Because without fair elections nobody controls the government.", "Hard to say."],
    "reality": ["Prices are too high and corruption is everywhere.", "The roads have improved a lot.", "Not much else to say."],
    "vote_scenario": ["Because roads change lives.", "Because a leader who arrests journalists will abuse power.", "I just would."],
    "party_scenario": ["Because stability comes first for me.", "Because elections must happen, whoever is in power.", "I don't know."],
}


class SimulatedRespondent:
    def __init__(self, rng: random.Random, home: str, low_effort: bool):
        self.rng, self.home, self.low_effort = rng, home, low_effort
        self.source = {t: (home if rng.random() < 0.6 else rng.choice(list(PERSONAS))) for t in ELABORATE | {"past_behaviour": 0, "closing": 0}}
        self.count: dict[str, int] = {}

    def answer(self, topic: str | None) -> str:
        if topic is None:
            return "Yes, let's start."
        if topic == "bridge":
            reaction = {"S03": "revises", "S04": "rationalises"}.get(self.home) or self.rng.choices(
                ["revises", "rationalises", "reconciles"], [0.25, 0.45, 0.30])[0]
            return BRIDGE_A[reaction]
        k = self.count.get(topic, 0)
        self.count[topic] = k + 1
        if self.low_effort and self.rng.random() < 0.85:
            return self.rng.choice(SHORT.get(topic, ["Yes."]))
        scripted = PERSONAS[self.source[topic]][2].get(topic, ["No."])[0::2]   # respondent lines only
        if k < len(scripted):
            return scripted[k]
        return self.rng.choice(ELABORATE.get(topic, ["No, nothing more."]))


def simulate(n: int = 150, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    mock = LLMClient(LLMConfig(backend="mock"))
    out = []
    for k in range(n):
        home = rng.choice(list(PERSONAS))
        resp = SimulatedRespondent(rng, home, low_effort=rng.random() < 0.15)
        s = InterviewSession(llm=mock, respondent_id=f"R{k + 1:03d}", seed=seed * 1000 + k,
                             context="bangladesh" if home in ("S04", "S05") else None)
        s.start()
        while not s.finished:
            s.respond(resp.answer(s.topic["id"] if s.topic else None))
        d = s.to_dict()
        d["meta"].update(synthetic=True, simulated=True, home_persona=home,
                         persona=f"simulated, mostly {PERSONAS[home][0]}" + (" (low effort)" if resp.low_effort else ""))
        out.append(d)
    return out


if __name__ == "__main__":
    folder = ROOT / "data" / "simulated"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for d in simulate():
        (folder / f"{d['meta']['respondent_id']}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(list(folder.glob("*.json"))), "simulated interviews to", folder)
