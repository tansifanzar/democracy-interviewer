"""Layered annotation of interview transcripts.

The project asks how people CONCEPTUALISE democracy, what they EXPECT from it,
and how their attitudes translate into BEHAVIOUR. Every answer is coded with
the same codebook, then assigned to its layer:

    concept      codes raised in Module A (+ whether press freedom is essential)
    expectation  codes raised in Module B's expectation / trade-off questions
    reality      perceived delivery (+1 well / -1 badly) of each expected code
    behaviour    choices in the two scenarios and past voting

Tensions are the gaps between layers that the project calls paradoxes:
    action paradox        principle in the concept, contrary choice in behaviour
    dissatisfaction       expected, but seen as badly delivered

Two annotators share one output format: a transparent keyword baseline that
runs offline, and an LLM annotator (`llm=`). Both must be validated against
human coders before substantive use.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CODEBOOK = yaml.safe_load((ROOT / "config" / "codebook.yaml").read_text(encoding="utf-8"))
CATS = CODEBOOK["categories"]
QUESTIONNAIRE = yaml.safe_load((ROOT / "config" / "questionnaire.yaml").read_text(encoding="utf-8"))
MODULE_OF = {t["id"]: t["module"] for t in QUESTIONNAIRE["topics"]}
DECISIONS = yaml.safe_load((ROOT / "config" / "decisions.yaml").read_text(encoding="utf-8"))
MAIN_SPEC = {d["id"]: d["main"] for d in DECISIONS["decisions"]}   # the analytical choices used by default
SKIP_TOPICS = {None, "closing"}


# ----------------------------------------------------------------- keyword helpers
def _pos(text: str, kw: str) -> int:
    m = re.search(r"(?<![a-z])" + re.escape(kw.lower()), text.lower())
    return m.start() if m else -1


def has_any(text: str, kws: list[str]) -> bool:
    return any(_pos(text, k) >= 0 for k in kws)


def earliest(text: str, groups: dict[str, list[str]], default: str = "unclear") -> str:
    """Return the group whose first keyword occurs earliest in the text."""
    best, best_pos = default, 10 ** 9
    for name, kws in groups.items():
        for k in kws:
            p = _pos(f" {text} ", k)
            if 0 <= p < best_pos:
                best, best_pos = name, p
    return best


def _glossary(context: str | None) -> dict[str, list[str]]:
    """Local terms from a context pack, grouped by codebook category."""
    from .context import load_context
    ctx = load_context(context)
    out: dict[str, list[str]] = {}
    for term, cat in (ctx or {}).get("glossary", {}).items():
        out.setdefault(cat, []).append(term)
    return out


def codes_of(text: str, context: str | None = None) -> list[str]:
    extra = _glossary(context) if context else {}
    return [c for c, spec in CATS.items() if has_any(text, spec["keywords"] + extra.get(c, []))]


def justified(text: str) -> bool:
    return has_any(text, CODEBOOK["justification_markers"])


def clause_valence(text: str, aggregation: str = "any_badly", context: str | None = None) -> dict[str, int]:
    """Perceived delivery per code, judged clause by clause (+1 well, -1 badly).

    aggregation (a logged decision): "any_badly" - one negative verdict wins;
    "majority" - verdicts are summed and the sign is kept.
    """
    out: dict[str, int] = {}
    total: dict[str, int] = {}
    vm = CODEBOOK["valence_markers"]
    for clause in re.split(r"[.;!?]|,\s*but\s|\sbut\s", text):
        codes = codes_of(clause, context)
        if not codes:
            continue
        neg, pos = has_any(f" {clause} ", vm["negative"]), has_any(f" {clause} ", vm["positive"])
        v = -1 if neg and not pos else 1 if pos and not neg else (-1 if neg else 0)
        for c in codes:
            out[c] = v if c not in out else min(out[c], v)   # any "badly" wins
            total[c] = total.get(c, 0) + v
    if aggregation == "majority":
        return {c: (1 if t > 0 else -1 if t < 0 else 0) for c, t in total.items()}
    return out


# ----------------------------------------------------------------- LLM annotator
class LLMAnnotator:
    def __init__(self, llm, context: str | None = None):
        self.llm, self.name = llm, f"llm:{llm.config.model}"
        cb = "\n".join(f"- {c}: {s['label']}. {s['description']}" for c, s in CATS.items())
        local = _glossary(context)
        if local:   # local vocabulary from the context pack
            cb += "\nLocal terms in this country: " + "; ".join(f"{', '.join(v)} -> {c}" for c, v in local.items())
        self.system = (
            "You annotate answers from research interviews about democracy. Using ONLY this "
            f"codebook, list every category the answer substantively refers to:\n{cb}\n"
            "Also say whether the respondent gives at least one reason (justified). "
            'Return {"codes": [..], "justified": true|false}.'
        )

    def code(self, text: str) -> tuple[list[str], bool]:
        out = self.llm.chat_json(self.system, f"Answer: {text}", default={"codes": [], "justified": False})
        out = out if isinstance(out, dict) else {}
        return [c for c in out.get("codes", []) if c in CATS], bool(out.get("justified", False))


# ----------------------------------------------------------------- main entry
def annotate_transcript(transcript: dict, llm=None, spec: dict | None = None, context: str | None = None) -> dict:
    """`spec` overrides the analytical decisions logged in config/decisions.yaml.
    `context` (or meta["context"]) adds a context pack's local vocabulary to the coder."""
    spec = {**MAIN_SPEC, **(spec or {})}
    context = context or transcript.get("meta", {}).get("context")
    ann = LLMAnnotator(llm, context) if llm is not None else None
    turns = []
    for i, t in enumerate(transcript["turns"]):
        if t["role"] != "respondent" or t.get("topic") in SKIP_TOPICS:
            continue
        codes, just = ann.code(t["text"]) if ann else (codes_of(t["text"], context), justified(t["text"]))
        turns.append({"idx": i, "topic": t["topic"], "module": MODULE_OF.get(t["topic"]), "text": t["text"],
                      "codes": codes, "justified": just})

    def answers(topic):
        return [t for t in turns if t["topic"] == topic]

    def joined(topic):
        return " ".join(t["text"] for t in answers(topic))

    # --- concept layer: 1 mentioned, 2 justified, 3 essential (press freedom judged indispensable)
    concept: dict[str, int] = {}
    for t in turns:
        if t["module"] == "concept":
            for c in t["codes"]:
                concept[c] = max(concept.get(c, 0), 2 if t["justified"] else 1)
    boundary = earliest(joined("concept_boundary"), CODEBOOK["boundary_markers"]) if answers("concept_boundary") else "unclear"
    if boundary == "essential":
        concept["freedoms"] = 3

    # --- expectation layer: 1 expected, 2 chosen as priority in the trade-off
    expectation: dict[str, int] = {}
    for t in turns:
        if t["topic"] == "expect" or (t["topic"] == "tradeoff" and spec["tradeoff_expectation"] == "all_codes"):
            for c in t["codes"]:
                expectation[c] = max(expectation.get(c, 0), 1)
    trade = joined("tradeoff").lower()
    priority = None
    if trade:
        e, d = _pos(trade, "election"), min([p for p in (_pos(trade, "develop"), _pos(trade, "econom")) if p >= 0], default=-1)
        priority = "elections" if e >= 0 and (d < 0 or e < d) else "economy" if d >= 0 else None
        if priority:
            expectation[priority] = 2

    # --- reality: perceived delivery of expected things
    reality = clause_valence(joined("reality"), spec["reality_aggregation"], context) if answers("reality") else {}
    gaps = {c: reality.get(c, 0) for c in expectation}

    # --- behaviour layer
    vote = earliest(joined("vote_scenario"), CODEBOOK["vote_choice"]) if answers("vote_scenario") else "unclear"
    party = earliest(joined("party_scenario"), CODEBOOK["party_action"]) if answers("party_scenario") else "unclear"
    past = joined("past_behaviour").lower()
    voted = "unclear" if not past else "no" if re.search(r"\b(no|didn't|did not|never|boycott\w*)\b", past) else "yes"
    acted: dict[str, int] = {}         # +1 acted in line with the consideration, -1 acted against it
    if vote == "builder":
        acted.update({"economy": 1, "freedoms": -1})
    elif vote == "honest":
        acted.update({"freedoms": 1, "corruption": 1})
    if party == "loyal":
        acted.update({"elections": -1, "order": 1})
    elif party == "withdraws":
        acted.update({"elections": 1})

    # --- tensions between layers
    tensions = []
    # press freedom counts as endorsed if named unprompted (concept_meaning) or judged essential at the boundary;
    # merely mentioning newspapers while calling them optional is not an endorsement
    if spec["freedoms_endorsement"] == "any_mention":
        freedoms_endorsed = concept.get("freedoms", 0) >= 1
    else:
        freedoms_endorsed = boundary == "essential" or any("freedoms" in t["codes"] for t in answers("concept_meaning"))
    elections_endorsed = concept.get("elections", 0) >= (2 if spec["elections_endorsement"] == "justified" else 1)
    if vote == "builder" and freedoms_endorsed:
        tensions.append({"kind": "action", "between": ["concept", "behaviour"], "category": "freedoms",
                         "label": "votes for the candidate who arrests journalists",
                         "explanation": "Sees press freedom as part of democracy, but would vote for the candidate who had journalists arrested."})
    if party == "loyal" and elections_endorsed:
        tensions.append({"kind": "action", "between": ["concept", "behaviour"], "category": "elections",
                         "label": "stays loyal after a postponed election",
                         "explanation": "Names elections as part of democracy, but would stay with their party after it postponed an election."})
    for c, v in gaps.items():
        if v < 0:
            tensions.append({"kind": "dissatisfaction", "between": ["expectation", "reality"], "category": c,
                             "label": f"{CATS[c]['label'].lower()} expected but not delivered",
                             "explanation": f"Expects {CATS[c]['label'].lower()} from democracy, but sees it badly delivered."})

    bridge_text = joined("bridge")
    bridge = earliest(bridge_text, {k: v for k, v in CODEBOOK["bridge_reaction"].items()}, "reconciles") if bridge_text else None

    return {
        "meta": transcript["meta"], "method": ann.name if ann else "dictionary-baseline", "spec": spec, "context": context, "turns": turns,
        "layers": {"concept": concept, "expectation": expectation, "reality": reality, "acted": acted},
        "boundary": boundary, "priority": priority, "gaps": gaps,
        "behaviour": {"vote": vote, "party": party, "voted_last": voted},
        "bridge": bridge, "tensions": tensions,
        # kept for the belief-network view
        "contradictions": [{"node": t["label"], "principles": [t["category"]]} for t in tensions if t["kind"] == "action"],
    }


def needs_bridge(transcript: dict) -> list[dict]:
    """Action-paradox tensions so far (used by the Bridge Agent during the interview)."""
    return [t for t in annotate_transcript(transcript)["tensions"] if t["kind"] == "action"]
