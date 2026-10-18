"""Exercise the real OpenAI-compatible code path against a tiny local stub server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from analysis.annotate import annotate_transcript
from interviewer.engine import InterviewSession
from interviewer.llm import LLMClient, LLMConfig


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = body["messages"][0]["content"]
        if "screen answers" in system:
            content = '{"label": "OK"}'
        elif "supervisor" in system:
            n_fu = body["messages"][1]["content"].split("Follow-ups used: ")[1][0]
            content = '```json\n{"action": "%s", "reason": "stub", "depth": 2}\n```' % ("follow_up" if n_fu == "0" else "next")
        elif "codebook" in system:
            content = '{"codes": ["elections", "not_a_code"], "justified": true}'
        elif "anti-democratic proposal" in system:
            content = '{"stance": "accept"}'
        else:
            content = "Could you tell me more about why you see it that way?"
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def llm():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield LLMClient(LLMConfig(backend="openai", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                              model="stub", api_key="x"))
    srv.shutdown()


def test_interview_via_openai_compatible_server(llm):
    s = InterviewSession(llm=llm, bridge=False)
    s.start()
    while not s.finished:
        s.respond("Democracy means free elections.")
    kinds = [t.get("type") for t in s.transcript if t["role"] == "interviewer"]
    # the stub asks one follow-up wherever the budget allows it
    with_budget = [t for t in s.topics if t.get("max_followups", s.q["max_followups_default"]) > 0]
    assert kinds.count("follow_up") == len(with_budget)
    assert kinds.count("dynamic_question") == 1          # the reality check (bridge arm is off)
    assert s.meta["model"] == "stub"


def test_llm_annotation_filters_unknown_codes(llm):
    tr = {"meta": {"language": "en", "respondent_id": "x"},
          "turns": [{"role": "respondent", "topic": "concept_meaning", "text": "Free elections."},
                    {"role": "respondent", "topic": "party_scenario", "text": "I would still vote for them."}]}
    ann = annotate_transcript(tr, llm=llm)
    assert ann["turns"][0]["codes"] == ["elections"]
    assert ann["behaviour"]["party"] == "loyal"
    assert [t["category"] for t in ann["tensions"] if t["kind"] == "action"] == ["elections"]
