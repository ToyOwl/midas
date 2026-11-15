"""Контракт локальной панели: очередь, границы HTTP и неизменность отчёта."""
import io
import json
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from portagent.web.server import Application, Handler, clean
from portagent.web.store import Store, validate

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def app(tmp_path):
    config = tmp_path / "configs" / "default.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    app = Application(config)
    app.sessions["test"] = ("csrf-token", 9999999999)
    return app


def request(app, path, *, method="GET", data=None, headers=None):
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(app=app, server_port=8765)
    handler.path = path
    handler.headers = Message()
    body = json.dumps(data).encode() if data is not None else b""
    values = {"Host": "127.0.0.1:8765", "Cookie": "pa_session=test",
              "Origin": "http://127.0.0.1:8765", "X-CSRF-Token": "csrf-token",
              "Content-Type": "application/json", "Content-Length": str(len(body)),
              "Idempotency-Key": "test-request-0001", **(headers or {})}
    for key, value in values.items():
        if value is not None:
            handler.headers[key] = value
    handler.rfile = io.BytesIO(body)
    result = {}
    handler.send = lambda value, status=200, **kw: result.update(value=value, status=status, **kw)
    getattr(handler, f"do_{method}")()
    return result


def test_queue_idempotence_and_pending_dedup(tmp_path):
    store = Store(tmp_path)
    first = store.submit("daily", {}, "request-0001")
    assert store.submit("daily", {}, "request-0001")["id"] == first["id"]
    assert store.submit("daily", {}, "request-0002")["id"] == first["id"]
    with pytest.raises(ValueError, match="ключ"):
        store.submit("model", {"kind": "vision"}, "request-0001")
    assert store.cancel(first["id"])
    assert not store.cancel(first["id"])
    assert Store(tmp_path).get(first["id"])["status"] == "cancelled"


@pytest.mark.parametrize("kind,args", [("shell", {}), ("daily", {"command": "whoami"}),
    ("daily", {"use_llm": "yes"}), ("model", {"kind": "https://example.com"}),
    ("ask", {"date": "../../secret", "question": "test"}),
    ("ask", {"date": "2026-11-13", "question": "x" * 1600}),
    ("backtest", {"rebalance_days": 1.5}), ("backtest", {"commission_bps": float("nan")}),
    ("backtest", {"start": "2026-11-14", "end": "2026-11-13"})])
def test_only_bounded_actions_are_allowed(kind, args):
    with pytest.raises(ValueError):
        validate(kind, args)


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example:8765"}, {"Origin": "https://evil.example"},
    {"Sec-Fetch-Site": "cross-site"}, {"X-CSRF-Token": "wrong"}, {"Cookie": None}, {"Origin": None}])
def test_cross_site_commands_are_rejected(app, headers):
    result = request(app, "/api/jobs", method="POST", data={"kind": "daily", "args": {}}, headers=headers)
    assert result["status"] == 403
    assert app.store.list() == []


def test_http_submission_and_cancel(app):
    result = request(app, "/api/jobs", method="POST", data={"kind": "daily", "args": {}})
    assert result["status"] == 202
    job_id = result["value"]["id"]
    assert request(app, f"/api/jobs/{job_id}")["value"]["status"] == "queued"
    assert request(app, f"/api/jobs/{job_id}/cancel", method="POST", data={})["status"] == 200


def test_private_data_requires_local_session(app):
    assert request(app, "/api/state", headers={"Cookie": None})["status"] == 401
    assert request(app, "/api/session", headers={"Origin": "https://evil.example"})["status"] == 403
    assert request(app, "/api/state")["value"]["access"] == "local_only"
    assert request(app, "/api/market?date=2026-11-14", headers={"Cookie": None})["status"] == 401
    assert request(app, "/api/market?date=../../config")["status"] == 400


def test_archive_recovery_is_a_bounded_dated_action():
    assert validate("recover_news", {"date": "2026-11-14"}) == {"date": "2026-11-14"}
    for args in ({}, {"date": "../../x"}, {"date": "2026-11-14", "weights": {}}):
        with pytest.raises(ValueError):
            validate("recover_news", args)


def test_session_reused_across_browser_tabs(app):
    first = request(app, "/api/session")["value"]["csrf"]
    assert request(app, "/api/session")["value"]["csrf"] == first


def test_historical_llm_note_does_not_claim_current_model_is_off():
    from portagent.llm.assistant import snapshot
    report = {"date": "2026-11-13", "weights": {}, "llm": {"enabled": False,
              "note": "LLM отключена флагом --no-llm."}}
    note = snapshot(report)["coverage_note"]
    assert "2026-11-13" in note and "была пропущена" in note
    assert "а не текущего состояния модели" in note and "LLM отключена" not in note


def test_report_and_json_nan(app):
    directory = app.paths["reports"] / "2026-11-13"
    directory.mkdir(parents=True)
    (directory / "portfolio.json").write_text(json.dumps({"date": "2026-11-13", "weights": {"US:AAPL": .9},
        "portfolio": {"cash_share": .1}, "fusion": {"view_quality": {"sentiment": float("nan")}}}), encoding="utf-8")
    result = request(app, "/api/report?date=2026-11-13")
    assert result["status"] == 200 and result["value"]["evidence"]
    assert clean(result["value"])["report"]["fusion"]["view_quality"]["sentiment"] is None
    assert request(app, "/api/report?date=../../config")["status"] == 400


def test_assistant_does_not_mutate_report_and_invalid_citations_are_not_cached(tmp_path):
    import copy
    from portagent.config import load_config
    from portagent.llm.assistant import explain
    report = {"date": "2026-11-13", "weights": {"US:AAPL": .9}, "portfolio": {"cash_share": .1}}
    original = copy.deepcopy(report)
    calls = []
    def chat(messages, **kwargs):
        calls.append(messages)
        return json.dumps({"answer": "Кэш составляет 10%.", "evidence_ids": ["EV-BOOK"], "unknowns": []}), {}
    llm = load_config(CONFIG)["llm"]
    first = explain(report, "Какова доля кэша?", llm=llm, cache_dir=tmp_path, chat=chat)
    second = explain(report, "Какова доля кэша?", llm=llm, cache_dir=tmp_path, chat=chat)
    assert report == original and len(calls) == 1 and second["cache_hit"]
    assert first["report_sha"] == second["report_sha"]
    def bad(*a, **kw):
        return json.dumps({"answer": "Вывод.", "evidence_ids": ["EV-FAKE"], "unknowns": []}), {}
    with pytest.raises(ValueError):
        explain(report, "Другой вопрос", llm=llm, cache_dir=tmp_path, chat=bad)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_research_view_auth_and_frozen_report(app):
    day = "2026-11-14"
    target = app.paths["reports"] / day / "portfolio.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"date": day, "weights": {"US:AAPL": .9}}), encoding="utf-8")
    before = target.read_bytes()
    assert request(app, "/api/research?date=" + day, headers={"Cookie": None})["status"] == 401
    result = request(app, "/api/research?date=" + day)
    assert result["status"] == 200 and len(result["value"]["channels"]) == 3
    assert target.read_bytes() == before
    assert request(app, "/api/research?date=../../secret")["status"] != 200
    for kind in ("refresh_sources", "train_models"):
        assert validate(kind, {"date": day}) == {"date": day}
        with pytest.raises(ValueError):
            validate(kind, {"date": day, "command": "arbitrary"})
