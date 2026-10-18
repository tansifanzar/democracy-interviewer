"""Streamlit front end:  streamlit run app.py"""
import json

import streamlit as st

from analysis.annotate import annotate_transcript
from analysis.layers_viz import draw_three_layers
from analysis.network import build_network, draw_network, metrics
import matplotlib.pyplot as plt
from interviewer.engine import InterviewSession
from interviewer.llm import LLMClient, LLMConfig

st.set_page_config(page_title="AI Conversational Interview: Democracy", page_icon="🗳️", layout="centered")

with st.sidebar:
    st.header("Settings")
    arm = st.radio("Bridge question", ["random", "on", "off"], help="The bridge question is an experimental arm; by default each interview is assigned at random.")
    cfg = LLMConfig()
    backend = st.selectbox("Backend", ["openai", "mock"], index=0 if cfg.backend != "mock" else 1,
                           help="'openai' = any OpenAI-compatible server, e.g. Ollama at localhost:11434/v1")
    cfg.backend = backend
    if backend != "mock":
        cfg.base_url = st.text_input("Base URL", cfg.base_url)
        cfg.model = st.text_input("Model", cfg.model)
    if st.button("Start new interview", type="primary") or "session" not in st.session_state:
        st.session_state.session = InterviewSession(llm=LLMClient(cfg), bridge={"random": None, "on": True, "off": False}[arm])
        st.session_state.session.start()
    st.caption("Research demo. Answers are stored only in this browser session unless you download them.")

s: InterviewSession = st.session_state.session
st.title("🗳️ How do you think about democracy?")
module = s.topic.get("module") if s.topic else None
st.caption(f"AI Conversational Interview · {s.q['modules'].get(module, '') if module else 'intro'} · backend: {s.llm.config.backend}")

for turn in s.transcript:
    with st.chat_message("assistant" if turn["role"] == "interviewer" else "user"):
        st.write(turn["text"])

if not s.finished:
    if answer := st.chat_input("Type your answer…"):
        with st.spinner("…"):
            s.respond(answer)
        st.rerun()
else:
    st.success("Interview complete.")
    data = json.dumps(s.to_dict(), ensure_ascii=False, indent=2)
    st.download_button("Download transcript (JSON)", data, file_name=f"interview_{s.meta['respondent_id']}.json")

    if st.button("Map my democratic belief system"):
        ann = annotate_transcript(s.to_dict(), llm=None if s.llm.is_mock else s.llm)
        G = build_network(ann)
        m = metrics(G, ann)
        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        draw_three_layers(ann, ax, "What democracy is to you, what you expect, what you would do")
        st.pyplot(fig)
        for t in ann["tensions"]:
            st.write("• " + t["explanation"])
        cols = st.columns(4)
        cols[0].metric("Distinct considerations", m["repertoire"])
        cols[1].metric("Abstract share", f"{m['abstract_share']:.0%}")
        cols[2].metric("Justified answers", f"{m['justification_rate']:.0%}")
        cols[3].metric("Action paradoxes", m["contradictions"])
