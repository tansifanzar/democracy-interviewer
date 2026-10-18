"""What did each agent do, and when? Agent traces for one interview and across many.

A trace is a list of events, one per agent action, in order:

    {"seq", "agent", "action", "topic", "module", "text", "detail"}

agents:  respondent · script (verbatim main questions) · safety · probe ·
         interviewer (LLM-written follow-ups / reality check) · bridge · coder

Interviews run by `interviewer.engine` record the agents' decisions on each turn
(safety label, probe action, reason and depth), which the trace reuses.
For transcripts without one (e.g. the synthetic personas) the trace is
*reconstructed* from the turn sequence; reconstructed probe reasons are marked
as such. The coder events come from analysis/annotate.py.
"""
from __future__ import annotations

from .annotate import CATS, MAIN_SPEC, QUESTIONNAIRE, annotate_transcript

TOPICS = {t["id"]: t for t in QUESTIONNAIRE["topics"]}
LEVEL = {"concept": ["", "named", "with a reason", "essential"], "expectation": ["", "expected", "top priority"]}


def _max_fu(topic: str) -> int:
    return TOPICS.get(topic, {}).get("max_followups", QUESTIONNAIRE.get("max_followups_default", 1))


def _layer_delta(prev: dict, now: dict) -> list[str]:
    out = []
    for layer in ("concept", "expectation"):
        for c, v in now["layers"][layer].items():
            if v > prev["layers"][layer].get(c, 0):
                out.append(f"{'is' if layer == 'concept' else 'expects'}: {CATS[c]['label']} ({LEVEL[layer][v]})")
    for c, v in now["layers"]["reality"].items():
        if prev["layers"]["reality"].get(c) != v:
            out.append(f"reality: {CATS[c]['label']} {'well' if v > 0 else 'badly' if v < 0 else 'mixed'}")
    for c, v in now["layers"]["acted"].items():
        if prev["layers"]["acted"].get(c) != v:
            out.append(f"does: {'in line with' if v > 0 else 'against'} {CATS[c]['label'].lower()}")
    for t in now["tensions"]:
        if not any(p["kind"] == t["kind"] and p["category"] == t["category"] for p in prev["tensions"]):
            out.append(("PARADOX: " if t["kind"] == "action" else "gap: ") + t["label"])
    if now["bridge"] and not prev["bridge"]:
        out.append(f"bridge reaction: {now['bridge']}")
    return out


def trace_transcript(tr: dict, spec: dict | None = None) -> list[dict]:
    """Agent events for one interview (recorded if available, otherwise reconstructed)."""
    turns = tr["turns"]
    recorded = {e["turn"]: e for e in tr.get("trace", []) if "turn" in e}
    ev: list[dict] = []

    def add(agent, action, topic=None, text="", **detail):
        ev.append({"seq": len(ev), "agent": agent, "action": action, "topic": topic,
                   "module": TOPICS.get(topic, {}).get("module"), "text": text, "detail": detail})

    prev = annotate_transcript({"meta": tr["meta"], "turns": []}, spec=spec)
    followups = 0
    bridge_checked = False
    for i, t in enumerate(turns):
        topic = t.get("topic")
        if t["role"] == "interviewer":
            kind = t.get("type")
            if kind == "main_question":
                followups = 0
                add("script", "asks main question (verbatim)", topic, t["text"])
            elif kind == "dynamic_question" and topic == "reality":
                followups = 0
                add("interviewer", "writes reality check from their expectations", topic, t["text"])
            elif kind == "dynamic_question" and topic == "bridge":
                followups = 0
                add("bridge", "asks bridge question", topic, t["text"])
            elif kind == "follow_up":
                add("interviewer", "writes follow-up", topic, t["text"])
            elif kind in ("intro", "outro"):
                add("script", kind, None, t["text"])
            elif kind == "safety":
                add("safety", "responds to flagged answer", topic, t["text"])
            continue

        # ---- respondent answer -> safety -> coder -> probe (-> bridge check)
        add("respondent", "answers", topic, t["text"])
        if topic is None:
            continue
        rec = recorded.get(i, {})
        add("safety", "screens answer", topic, label=t.get("safety", rec.get("safety", "OK")))
        if topic != "closing":
            now = annotate_transcript({"meta": tr["meta"], "turns": turns[: i + 1]}, spec=spec)
            mine = now["turns"][-1]
            add("coder", "codes answer", topic, codes=[CATS[c]["label"] for c in mine["codes"]],
                justified=mine["justified"], changes=_layer_delta(prev, now))
            prev = now
        nxt = next((x for x in turns[i + 1:] if x["role"] == "interviewer"), None)
        n_words = len(t["text"].split())
        if "probe" in t:                                   # recorded live by the engine
            p = t["probe"]
            add("probe", "follow up" if p["action"] == "follow_up" else "move on", topic,
                reason=p.get("reason", ""), depth=p.get("depth"), reconstructed=False)
        elif nxt is not None and nxt.get("type") == "follow_up" and nxt.get("topic") == topic:
            reason = "short answer" if n_words < 20 else "no reason given" if not prev["turns"][-1]["justified"] else "vague term to unpack"
            add("probe", "follow up", topic, reason=reason, reconstructed=True)
        else:
            reason = ("follow-up budget used" if followups >= _max_fu(topic) and _max_fu(topic) > 0
                      else "no follow-ups planned here" if _max_fu(topic) == 0 else "goal covered")
            add("probe", "move on", topic, reason=reason, reconstructed=True)
        if nxt is not None and nxt.get("type") == "follow_up":
            followups += 1
        # the bridge agent checks once, after the last behaviour question
        if topic == "past_behaviour" and not bridge_checked:
            bridge_checked = True
            action = [x for x in prev["tensions"] if x["kind"] == "action"]
            arm = tr["meta"].get("bridge_arm")
            if not action:
                add("bridge", "checks for a principle-choice conflict: none", topic, arm=arm)
            elif not arm:
                add("bridge", "conflict found, skipped (control arm)", topic, arm=arm, conflict=action[0]["label"])
            else:
                add("bridge", "conflict found, will ask", topic, arm=arm, conflict=action[0]["label"])
    return ev


def probe_yield(ev: list[dict]) -> list[dict]:
    """For every follow-up: what did the answer to it add? New codes or layer changes, or a reason
    where the earlier answer on the same topic gave none."""
    out = []
    for k, e in enumerate(ev):
        if e["agent"] == "interviewer" and e["action"] == "writes follow-up":
            coder = next((x for x in ev[k + 1:] if x["agent"] == "coder"), None)
            before = [x for x in ev[:k] if x["agent"] == "coder" and x["topic"] == e["topic"]]
            if coder and coder["topic"] == e["topic"]:
                gained = [c for c in coder["detail"]["changes"] if not c.startswith("bridge")]
                if coder["detail"]["justified"] and before and not any(b["detail"]["justified"] for b in before):
                    gained.append("reason added")
                out.append({"topic": e["topic"], "gained": len(gained), "items": gained})
    return out


def summarise(transcripts: list[dict]) -> dict:
    """Aggregate agent behaviour across interviews (for the research overview)."""
    per_topic: dict[str, dict] = {t: {"answers": 0, "followups": 0, "productive": 0} for t in TOPICS if t not in ("bridge", "closing")}
    bridge = {"interviews": 0, "bridge_arm": 0, "conflict": 0, "conflict_in_arm": 0, "asked": 0, "reactions": {}}
    safety_flags = 0
    for tr in transcripts:
        ev = trace_transcript(tr)
        bridge["interviews"] += 1
        bridge["bridge_arm"] += bool(tr["meta"].get("bridge_arm"))
        for e in ev:
            if e["agent"] == "probe" and e["topic"] in per_topic:
                per_topic[e["topic"]]["answers"] += 1
            if e["agent"] == "safety" and e["action"] == "screens answer" and e["detail"].get("label") != "OK":
                safety_flags += 1
            if e["agent"] == "bridge" and e["action"].startswith("conflict found"):
                bridge["conflict"] += 1
                bridge["conflict_in_arm"] += e["action"].endswith("will ask")
            if e["agent"] == "bridge" and e["action"] == "asks bridge question":
                bridge["asked"] += 1
        for y in probe_yield(ev):
            if y["topic"] in per_topic:
                per_topic[y["topic"]]["followups"] += 1
                per_topic[y["topic"]]["productive"] += y["gained"] > 0
        r = annotate_transcript(tr)["bridge"]
        if r:
            bridge["reactions"][r] = bridge["reactions"].get(r, 0) + 1
    return {"per_topic": per_topic, "bridge": bridge, "safety_flags": safety_flags, "n": len(transcripts), "spec": MAIN_SPEC}
