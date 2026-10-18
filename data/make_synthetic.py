"""Generate the SYNTHETIC demo transcripts in data/synthetic/.

These are hand-written illustrative personas, NOT real respondents. They exist
only to demonstrate the pipeline end to end and say nothing about how real
people think. Two personas are set in Bangladesh (S04, S05).

Each persona maps topic ids to [answer, follow-up, answer, ...]. Dynamic topics
take their interviewer question from "_reality_q" / "_bridge_q"; a persona
without "_bridge_q" was in the control arm (or showed no tension).
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
Q = yaml.safe_load((ROOT / "config" / "questionnaire.yaml").read_text(encoding="utf-8"))

PERSONAS = {
    "S01": ("principled democrat", False, {
        "concept_meaning": ["Democracy means people choose and can remove their government in free elections. But it also means courts are independent and the press can criticise those in power, because otherwise elections become a show."],
        "concept_boundary": ["No, that is not a democracy. If newspapers cannot criticise the government, voters cannot know what they are choosing."],
        "expect": ["Equal treatment for everyone, honest officials, and citizens who have a say in decisions.",
                   "Which of these matters most to you?",
                   "Having a say, because then people can fix the other problems themselves."],
        "tradeoff": ["Fair elections. Development without accountability ends up in a few pockets."],
        "_reality_q": "You mentioned equal treatment, honest officials and having a say. How well does your country do on these today?",
        "reality": ["Elections work well and power changes hands peacefully. But corruption is still a problem, and citizens rarely have a say in decisions. Equal treatment is improving."],
        "vote_scenario": ["The honest one. Arresting journalists is a line I would not cross, even for new roads."],
        "party_scenario": ["I would not vote for them again. Postponing elections is how democracies end, whichever party does it."],
        "past_behaviour": ["Yes, I always vote. It is the least a citizen can do."],
        "closing": ["No, I think I've said everything."],
    }),
    "S02": ("dissatisfied, outcome-focused", False, {
        "concept_meaning": ["Democracy is supposed to mean the government works for ordinary people.",
                            "What would that look like in practice?",
                            "That people can afford food, that young people find jobs, and that hospitals actually work."],
        "concept_boundary": ["Yes, I suppose so. If people can vote, it is a democracy, even if the newspapers are controlled."],
        "expect": ["Jobs, fair prices and honest politicians who don't steal."],
        "tradeoff": ["Fast development. Elections every five years change nothing in my life."],
        "_reality_q": "You mentioned jobs, fair prices and honest politicians. How well does your country do on these today?",
        "reality": ["Badly. Prices keep rising and my son has no job. The politicians only fill their own pockets."],
        "vote_scenario": ["The first one. Roads help people; journalists do not feed my family."],
        "party_scenario": ["I don't really support any party. They are all the same."],
        "past_behaviour": ["No. What is the point? Nothing changes."],
        "closing": ["Just that politicians should listen to people like us."],
    }),
    "S03": ("superficial supporter", True, {
        "concept_meaning": ["Democracy is good. It means freedom.",
                            "What does freedom mean to you here?",
                            "That you are free, you can vote and say what you want.",
                            "Could you give me a concrete example?",
                            "I don't know, like voting in elections."],
        "concept_boundary": ["No, that is not a democracy. People need freedom of the press."],
        "expect": ["A good life, I guess. Jobs and roads."],
        "tradeoff": ["Development, probably. That is what people need."],
        "_reality_q": "You mentioned a good life, jobs and roads. How well does your country do on these today?",
        "reality": ["It's okay. The roads are better now, but jobs are hard to find."],
        "vote_scenario": ["The first one, the builder. Roads are more useful."],
        "party_scenario": ["I would accept it if it is for stability. That's fine."],
        "past_behaviour": ["Yes, I voted."],
        "_bridge_q": "Earlier you said a country is not a democracy without a free press, and in the voting example you chose the candidate who had journalists arrested. How do you see these two together?",
        "bridge": ["Hmm, I didn't think of that. Maybe I would reconsider my vote, I'm not sure."],
        "closing": ["No."],
    }),
    "S04": ("partisan, group-serving (Bangladesh)", True, {
        "concept_meaning": ["Democracy means the people choose the government with their vote, and people have the freedom to speak.",
                            "Could you say a bit more about that?",
                            "I mean fair elections, and the party that wins runs the country. The opposition can also say what it wants."],
        "concept_boundary": ["No, without a free press it is not a real democracy. Under the previous government the newspapers were afraid."],
        "expect": ["Development, jobs for young people, and an end to hartals and chaos."],
        "tradeoff": ["Both matter, but if I must choose, fair elections, because through the vote the people own the country."],
        "_reality_q": "You mentioned development, jobs and an end to hartals. How well does your country do on these today?",
        "reality": ["Development is going well: the Padma Bridge, the metro rail, new roads. But jobs are hard to find, and the opposition still calls hartals that create chaos."],
        "vote_scenario": ["The first one. He built the roads and bridges; those journalists were spreading lies anyway."],
        "party_scenario": ["I would still vote for them. My party wants the best for the country, and stability comes first."],
        "past_behaviour": ["Yes, I voted, for my party of course."],
        "_bridge_q": "Earlier you said there is no real democracy without a free press, and in the voting example you chose the candidate who had journalists arrested. How do you see these two together?",
        "bridge": ["But that is different. In our country some newspapers spread propaganda, so it is necessary to control them for stability."],
        "closing": ["No, thank you."],
    }),
    "S05": ("development first (Bangladesh)", False, {
        "concept_meaning": ["To me democracy means the development of the country and a better life for people.",
                            "Why do you think so?",
                            "Because for ordinary people, jobs, the price of rice and onions, and safety matter most."],
        "concept_boundary": ["Yes, if the government delivers development, people will accept it. Newspapers matter less than rice."],
        "expect": ["Good hospitals, good schools, electricity without load-shedding, and no corruption."],
        "tradeoff": ["Fast development, without a doubt. Elections do not fill stomachs."],
        "_reality_q": "You mentioned hospitals, schools, electricity and no corruption. How well does your country do on these today?",
        "reality": ["Electricity has improved and there are new bridges. But nothing moves in a government office without a bribe, and hospitals are in bad shape."],
        "vote_scenario": ["The first one. Roads and bridges are what people remember."],
        "party_scenario": ["I would wait and see. If they keep developing the country, I can accept it."],
        "past_behaviour": ["Yes, I voted."],
        "closing": ["Bring prices down, that is all we want."],
    }),
    "S06": ("participatory, deliberative", False, {
        "concept_meaning": ["To me democracy is about people having a say in the decisions that affect them, not only on election day: community meetings, petitions, citizens' assemblies."],
        "concept_boundary": ["No. Without a free press citizens cannot debate or hold anyone to account, so it is not a democracy."],
        "expect": ["Real participation, space for debate where people listen to each other, and an equal voice for rich and poor."],
        "tradeoff": ["Fair elections, but only as a start. Development decided without the people rarely serves them."],
        "_reality_q": "You mentioned participation, debate and an equal voice for rich and poor. How well does your country do on these today?",
        "reality": ["Elections are fair, but people rarely have a say in local decisions, and rich and poor are not heard equally."],
        "vote_scenario": ["The honest one. A leader who arrests journalists will not listen to anyone else either."],
        "party_scenario": ["I would protest, even against my own party. Stability bought by silencing voters just postpones the conflict."],
        "past_behaviour": ["Yes, and I volunteered as a polling agent."],
        "closing": ["I'd like to see more citizens' assemblies."],
    }),
}


def build(pid: str, persona: str, bridge_arm: bool, answers: dict) -> dict:
    turns = [{"role": "interviewer", "text": Q["intro"], "type": "intro", "topic": None, "module": None},
             {"role": "respondent", "text": "Yes, let's start.", "topic": None, "module": None}]
    for topic in Q["topics"]:
        tid, mod = topic["id"], topic.get("module")
        if tid not in answers:
            continue
        if topic.get("dynamic") == "reality":
            q, kind = answers["_reality_q"], "dynamic_question"
        elif topic.get("dynamic") == "bridge":
            q, kind = answers["_bridge_q"], "dynamic_question"
        else:
            q, kind = topic["question"], "main_question"
        turns.append({"role": "interviewer", "text": q, "type": kind, "topic": tid, "module": mod})
        for i, text in enumerate(answers[tid]):
            role = "respondent" if i % 2 == 0 else "interviewer"
            t = {"role": role, "text": text, "topic": tid, "module": mod}
            if role == "interviewer":
                t["type"] = "follow_up"
            turns.append(t)
    turns.append({"role": "interviewer", "text": Q["outro"], "type": "outro", "topic": None, "module": None})
    meta = {"respondent_id": pid, "language": "en", "persona": persona, "synthetic": True, "bridge_arm": bridge_arm,
            "context": "bangladesh" if "Bangladesh" in persona else None,
            "questionnaire_version": Q["version"], "backend": "hand-written persona", "model": None}
    return {"meta": meta, "turns": turns}


if __name__ == "__main__":
    out = ROOT / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for pid, (persona, arm, answers) in PERSONAS.items():
        path = out / f"synthetic_{pid}.json"
        path.write_text(json.dumps(build(pid, persona, arm, answers), ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", path)
