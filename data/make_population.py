"""Generate a SYNTHETIC population of 150 interviews for the multiverse demo.

Each respondent is assembled from the hand-written personas: for every topic
the answer comes from a 'home' persona most of the time and from another
persona otherwise, and some respondents are made low-effort (very short
answers). The result is varied enough to show how analytical decisions move
the results. It is NOT real data and says nothing about real public opinion.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from make_synthetic import PERSONAS, build  # noqa: E402

sys.path.insert(0, str(ROOT))
from analysis.annotate import annotate_transcript  # noqa: E402

BRIDGE_Q = {
    "freedoms": "Earlier you described press freedom as part of democracy, and in the voting example you chose the candidate who had journalists arrested. How do you see these two together?",
    "elections": "Earlier you named elections as part of democracy, and in the second example you said you would stay with your party after it postponed the election. How do you see these two together?",
}
BRIDGE_A = {
    "revises": PERSONAS["S03"][2]["bridge"][0],
    "rationalises": PERSONAS["S04"][2]["bridge"][0],
    "reconciles": "Both matter to me. When I vote, I weigh them case by case.",
}

SHORT = {
    "concept_meaning": ["Voting.", "Freedom, I guess.", "Elections and freedom."],
    "concept_boundary": ["Yes, I think so.", "No.", "Not really."],
    "expect": ["A good life.", "Jobs.", "Fair elections."],
    "tradeoff": ["Development.", "Elections, I guess."],
    "reality": ["Not good.", "It's okay.", "Badly."],
    "vote_scenario": ["The first one.", "The honest one.", "The second one."],
    "party_scenario": ["I would still vote for them.", "I would protest.", "Don't know."],
    "past_behaviour": ["Yes.", "No."],
    "closing": ["No."],
}
TOPICS = ["concept_meaning", "concept_boundary", "expect", "tradeoff", "reality",
          "vote_scenario", "party_scenario", "past_behaviour", "closing"]


def make(n: int = 150, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    ids = list(PERSONAS)
    people = []
    for k in range(n):
        home = rng.choice(ids)
        low_effort = rng.random() < 0.15
        answers = {}
        for t in TOPICS:
            src = home if rng.random() < 0.6 else rng.choice(ids)
            a = PERSONAS[src][2][t]
            answers[t] = [rng.choice(SHORT[t])] if low_effort and rng.random() < 0.85 else a
        reality_src = PERSONAS[home][2]
        answers["_reality_q"] = reality_src["_reality_q"]
        arm = rng.random() < 0.5                      # randomised bridge arm
        tr = build(f"P{k + 1:03d}", f"mixed, mostly {PERSONAS[home][0]}" + (" (low effort)" if low_effort else ""), arm, answers)
        tr["meta"]["home_persona"] = home
        tr["meta"]["context"] = "bangladesh" if home in ("S04", "S05") else None
        action = [t for t in annotate_transcript(tr)["tensions"] if t["kind"] == "action"]
        if arm and action:                            # bridge agent would ask; add question + a reaction
            reaction = {"S03": "revises", "S04": "rationalises"}.get(home) or rng.choices(
                ["revises", "rationalises", "reconciles"], [0.25, 0.45, 0.30])[0]
            at = next(i for i, t in enumerate(tr["turns"]) if t.get("topic") == "closing")
            tr["turns"][at:at] = [
                {"role": "interviewer", "text": BRIDGE_Q[action[0]["category"]], "type": "dynamic_question", "topic": "bridge", "module": "behaviour"},
                {"role": "respondent", "text": BRIDGE_A[reaction], "topic": "bridge", "module": "behaviour"},
            ]
        people.append(tr)
    return people


if __name__ == "__main__":
    out = ROOT / "data" / "synthetic_population"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for tr in make():
        (out / f"{tr['meta']['respondent_id']}.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(list(out.glob("*.json"))), "transcripts to", out)
