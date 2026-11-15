import json
import math
from pathlib import Path

import pytest

from portagent.llm import client
from portagent.llm.news_analyst import analyze_news, validate_batch

ITEMS = [{"id": "n1", "text": "Компания Apple сообщает о росте выручки за квартал.", "source": "test"}]
ROW = {"id": "n1", "tickers": ["US:AAPL"], "sentiment": .5, "confidence": .9,
       "quote": "Apple сообщает о росте выручки"}


def test_bad_json_is_not_cached(tmp_path):
    calls = []
    def chat(*a, **kw):
        calls.append(1)
        return ("broken" if len(calls) == 1 else json.dumps({"items": [ROW]})), {}
    first = analyze_news(ITEMS, universe=["US:AAPL"], chat=chat, model="x", cache_dir=tmp_path)
    assert first["stats"]["batch_errors"] == 1 and not list(tmp_path.glob("*.json"))
    second = analyze_news(ITEMS, universe=["US:AAPL"], chat=chat, model="x", cache_dir=tmp_path)
    assert len(calls) == 2 and len(second["labels"]) == 1


def test_duplicate_labels_do_not_inflate_coverage():
    row = {**ROW, "tickers": ["US:AAPL", "US:AAPL"]}
    labels, _ = validate_batch(json.dumps({"items": [row, row, row]}), ITEMS, ["US:AAPL"])
    assert len(labels) == 1
    labels, rejected = validate_batch(json.dumps({"items": [ROW, {**ROW, "sentiment": -.5}]}), ITEMS, ["US:AAPL"])
    assert labels == [] and rejected


def test_proxy_and_redirect_protection(monkeypatch):
    calls = []
    class Response:
        status_code = 302
        text = "redirect"
    class Session:
        trust_env = True
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kw):
            calls.append((self.trust_env, url, kw))
            return Response()
    monkeypatch.setattr(client.requests, "Session", Session)
    with pytest.raises(client.LocalLLMError, match="перенаправления"):
        client.probe("http://127.0.0.1:8080", "model")
    assert len(calls) == 1 and calls[0][0] is False
    assert calls[0][2]["allow_redirects"] is False


def test_vision_seed_cache_and_finite_confidence(tmp_path, monkeypatch):
    from portagent.collect.charts import analyze_charts, _parse_chart_response
    calls = []
    def chat(*a, **kw):
        calls.append(kw["seed"])
        return json.dumps({"asset": "US:AAPL", "trend": "up", "confidence": .8, "note": "Рост"}), {}
    monkeypatch.setattr("portagent.collect.charts.vision_chat", chat)
    image = tmp_path / "chart.png"
    image.write_bytes(b"image")
    args = dict(universe=["US:AAPL"], endpoint="http://127.0.0.1:8081", model="qwen", timeout=1,
                seed=7, cache_dir=tmp_path / "cache")
    first, _ = analyze_charts([("US:AAPL", image)], **args)
    second, status = analyze_charts([("US:AAPL", image)], **args)
    assert first == second and status[0]["cache_hit"] and calls == [7]
    args["seed"] = 8
    analyze_charts([("US:AAPL", image)], **args)
    assert calls == [7, 8]
    for confidence in (float("nan"), float("inf"), -1, 1.1):
        assert _parse_chart_response(json.dumps({"asset": "US:AAPL", "trend": "up", "confidence": confidence}),
                                     expected_asset="US:AAPL", universe=["US:AAPL"]) is None
    assert _parse_chart_response(json.dumps({"asset": "US:AAPL", "trend": "up",
        "confidence": .9, "note": "Поддержка 10000"}), expected_asset="US:AAPL", universe=["US:AAPL"]) is None


def test_critic_requires_own_summary():
    from portagent.llm.critic import parse_critique, SUMMARY_IDS
    sections = [{"id": key, "stance": "ok", "fact": "fact", "evidence_ids": [value]}
                for key, value in SUMMARY_IDS.items()]
    sections[0]["evidence_ids"] = ["EV-DATE"]
    with pytest.raises(ValueError, match="EV-CONC"):
        parse_critique(json.dumps({"summary": "s", "sections": sections, "findings": []}),
                       valid_ids=set(SUMMARY_IDS.values()) | {"EV-DATE"})


def test_connection_recovery_is_bounded_and_keeps_request(monkeypatch):
    import requests
    from portagent.llm import client
    from portagent.llm.news_analyst import make_chat
    calls, recovered = [], []
    def call(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) != 2:
            raise client.LocalLLMError("connection lost") from requests.ConnectionError("refused")
        return "{}", {}
    monkeypatch.setattr(client, "chat", call)
    chat = make_chat("http://127.0.0.1:8080", "test", timeout_s=5, max_tokens=100,
                     seed=7, recover=lambda: recovered.append(True))
    assert chat([], schema={})[1]["connection_recovered"]
    assert calls[0] == calls[1] and recovered == [True]
    with pytest.raises(client.LocalLLMError):
        chat([], schema={})
    assert recovered == [True]
