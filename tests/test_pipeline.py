import json
from pathlib import Path

from analysis.annotate import annotate_transcript
from analysis.network import build_network, metrics
from interviewer.engine import InterviewSession
from interviewer.llm import LLMClient, LLMConfig, parse_json

ROOT = Path(__file__).resolve().parent.parent
MOCK = LLMClient(LLMConfig(backend="mock"))
LONG = "I think that " + "people should choose their government in free elections and keep it honest " * 2


def load(pid):
    return json.loads((ROOT / f"data/synthetic/synthetic_{pid}.json").read_text(encoding="utf-8"))


def run(answers, bridge=True):
    s = InterviewSession(llm=MOCK, bridge=bridge)
    s.start()
    for a in answers:
        if s.finished:
            break
        s.respond(a)
    return s


# ---------------------------------------------------------------- interview engine
def test_three_modules_in_order_and_reality_question_is_personalised():
    s = run(["yes"] + [LONG] * 20, bridge=False)
    assert s.finished
    modules = [t["module"] for t in s.transcript if t["role"] == "interviewer" and t.get("type") in {"main_question", "dynamic_question"}]
    order = [m for i, m in enumerate(modules) if m and (i == 0 or modules[i - 1] != m)]
    assert order == ["concept", "expectation", "behaviour"]
    reality = [t for t in s.transcript if t.get("topic") == "reality" and t["role"] == "interviewer"][0]
    assert reality["type"] == "dynamic_question" and "You mentioned" in reality["text"]


def test_followup_budget_is_enforced():
    s = run(["yes"] + ["short"] * 60, bridge=False)
    assert s.finished
    for topic in s.topics:
        n = sum(1 for t in s.transcript if t.get("topic") == topic["id"] and t.get("type") == "follow_up")
        assert n <= topic.get("max_followups", s.q["max_followups_default"])


PAD = " That is really what I honestly think about it, all things considered."   # keeps the mock from probing


def _paradox_answers():
    return ["yes"] + [a + PAD for a in [
            "Democracy means free elections and a free press that can criticise those in power, because power must be checked.",
            "No, that is not a democracy, because without a free press the elections mean nothing to anyone at all.",
            "Jobs and roads and fair prices, because that is what ordinary families need every single day of their lives.",
            "Fast development, because people cannot eat elections and we need jobs and roads before anything else at all.",
            "Prices are too high and jobs are hard to find for young people, though the new roads are certainly good.",
            "The first one, the builder, because roads and bridges change lives more than newspapers ever do for people.",
            "I would still vote for them, because stability matters and my party wants the best for the country anyway.",
            "Yes, I voted.",
            "But development is necessary in our country, so it is a realistic choice for now, I think.",
            "No."]]


def test_bridge_question_asked_only_in_bridge_arm_and_only_with_a_tension():
    with_bridge = run(_paradox_answers(), bridge=True)
    assert any(t.get("topic") == "bridge" and t["role"] == "interviewer" for t in with_bridge.transcript)
    control = run(_paradox_answers(), bridge=False)
    assert not any(t.get("topic") == "bridge" for t in control.transcript)
    consistent = run(["yes"] + [LONG] * 20, bridge=True)          # no action paradox -> no bridge
    assert not any(t.get("topic") == "bridge" for t in consistent.transcript)


def test_bridge_arm_is_randomised_and_recorded():
    arms = {InterviewSession(llm=MOCK, seed=i).meta["bridge_arm"] for i in range(20)}
    assert arms == {True, False}


def test_stop_word_and_distress_rule():
    assert run(["yes", "stop"]).meta.get("stopped_early")
    s = run(["yes", "honestly I want to die"])
    assert s.transcript[-1]["type"] == "safety"


# ---------------------------------------------------------------- layered annotation
def test_partisan_bangladesh_persona_shows_both_action_paradoxes_and_rationalises():
    a = annotate_transcript(load("S04"))
    assert a["boundary"] == "essential" and a["priority"] == "elections"
    assert a["behaviour"] == {"vote": "builder", "party": "loyal", "voted_last": "yes"}
    assert {t["category"] for t in a["tensions"] if t["kind"] == "action"} == {"freedoms", "elections"}
    assert a["bridge"] == "rationalises"


def test_principled_persona_is_consistent_but_dissatisfied():
    a = annotate_transcript(load("S01"))
    assert not [t for t in a["tensions"] if t["kind"] == "action"]
    assert a["layers"]["acted"] == {"freedoms": 1, "corruption": 1, "elections": 1}
    assert {t["category"] for t in a["tensions"] if t["kind"] == "dissatisfaction"} == {"participation", "corruption"}


def test_optional_press_freedom_is_not_an_endorsement():
    a = annotate_transcript(load("S05"))       # says newspapers matter less than rice, then votes for the builder
    assert a["boundary"] == "optional" and a["behaviour"]["vote"] == "builder"
    assert not [t for t in a["tensions"] if t["kind"] == "action"]


def test_tradeoff_counts_only_the_chosen_priority():
    a = annotate_transcript(load("S05"))       # "Elections do not fill stomachs" must not make elections an expectation
    assert a["priority"] == "economy" and "elections" not in a["layers"]["expectation"]


def test_network_metrics_still_work():
    a = annotate_transcript(load("S01"))
    m = metrics(build_network(a), a)
    assert m["repertoire"] >= 6 and m["contradictions"] == 0


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```', {}) == {"a": 1}
    assert parse_json('Sure! {"action": "next"} hope this helps', {}) == {"action": "next"}
    assert parse_json("not json", {"d": 0}) == {"d": 0}


# ---------------------------------------------------------------- logged decisions / multiverse
def test_decisions_change_results_as_documented():
    s05 = load("S05")      # mentions newspapers but calls them optional, then votes for the builder
    strict = annotate_transcript(s05)
    loose = annotate_transcript(s05, spec={"freedoms_endorsement": "any_mention"})
    assert not [t for t in strict["tensions"] if t["kind"] == "action"]
    assert [t["category"] for t in loose["tensions"] if t["kind"] == "action"] == ["freedoms"]
    assert "elections" in annotate_transcript(s05, spec={"tradeoff_expectation": "all_codes"})["layers"]["expectation"]


def test_majority_aggregation_can_hide_a_single_complaint():
    from analysis.annotate import clause_valence
    text = "The roads are better now. The new bridge is good. But jobs are hard to find."
    assert clause_valence(text, "any_badly")["economy"] == -1
    assert clause_valence(text, "majority")["economy"] == 1


def test_multiverse_covers_every_combination_and_flags_main():
    from analysis.multiverse import run, specifications
    specs = specifications(include_llm=False)
    assert len(specs) == 32
    res = [run([load(p) for p in ("S01", "S03", "S05")], s) for s in specs]
    assert sum(r["is_main"] for r in res) == 1


# ---------------------------------------------------------------- agent traces
def test_trace_orders_agents_and_uses_recorded_probe_decisions():
    from analysis.agent_trace import trace_transcript
    s = run(_paradox_answers(), bridge=True)
    ev = trace_transcript(s.to_dict())
    agents = [e["agent"] for e in ev]
    first = agents.index("respondent", 2)                       # first substantive answer
    assert agents[first:first + 4] == ["respondent", "safety", "coder", "probe"]
    probes = [e for e in ev if e["agent"] == "probe"]
    assert probes and all(p["detail"]["reconstructed"] is False for p in probes)
    assert any(e["action"] == "conflict found, will ask" for e in ev)
    assert any(e["action"] == "asks bridge question" for e in ev)


def test_reconstructed_trace_for_persona_and_probe_yield():
    from analysis.agent_trace import probe_yield, trace_transcript
    ev = trace_transcript(load("S04"))
    assert any(e["agent"] == "interviewer" and e["action"].startswith("writes reality check") for e in ev)
    assert any(e["agent"] == "bridge" and e["action"] == "conflict found, will ask" for e in ev)
    y = probe_yield(ev)
    assert [x["topic"] for x in y] == ["concept_meaning"]      # S04's one follow-up, which added rule of law
    assert y[0]["gained"] >= 1


# ---------------------------------------------------------------- context packs
def test_context_pack_localises_wording_but_keeps_the_construct():
    s = InterviewSession(llm=MOCK, bridge=False, context="bangladesh")
    s.start()
    for a in ["yes"] + [LONG] * 6:
        s.respond(a)
    q = [t["text"] for t in s.transcript if t.get("topic") == "vote_scenario" and t["role"] == "interviewer"][0]
    assert "constituency" in q and "journalists arrested" in q and s.meta["context"] == "bangladesh"


def test_refusal_on_sensitive_topic_is_respected_without_probing():
    s = InterviewSession(llm=MOCK, bridge=False, context="bangladesh")
    s.start()
    while s.topic is None or s.topic["id"] != "party_scenario":
        s.respond("yes" if s.topic is None else LONG)
    reply = s.respond("I would rather not say, it is not safe.")
    assert "do not have to answer" in reply and s.topic["id"] == "past_behaviour"
    refusal = [t for t in s.transcript if t["role"] == "respondent"][-1]
    assert refusal["probe"]["reason"] == "sensitive topic: refusal respected"
    assert not any(t.get("type") == "follow_up" and t.get("topic") == "party_scenario" for t in s.transcript)


def test_local_vocabulary_reaches_the_coder():
    from analysis.annotate import codes_of
    text = "Every few months there is a hartal and the whole city stops."
    assert "order" not in codes_of(text) and "order" in codes_of(text, "bangladesh")
    assert "participation" in codes_of("The gram sabha decides.", "india")
