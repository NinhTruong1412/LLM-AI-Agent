import json

from agentcore import FakeLLM, Tracer


def test_tracer_records_and_writes_jsonl(tmp_path):
    sink = tmp_path / "trace.jsonl"
    t = Tracer(sink)
    with t.span("step", k=1):
        pass
    assert len(t.durations("step")) == 1
    rec = json.loads(sink.read_text().strip())
    assert rec["name"] == "step" and rec["attributes"] == {"k": 1}


def test_fake_llm_records_calls():
    llm = FakeLLM(responder=lambda p: p.upper())
    assert llm.complete("hi").text == "HI"
    assert llm.calls == ["hi"]
