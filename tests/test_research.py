"""No-lookahead boundaries and source provenance, independent of GPU libraries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portagent.research.dataset import make_dataset, split_fold
from portagent.research.models import CANDIDATES, calibrate
from portagent.research.benchmark import paired_intervals
from portagent.research.events import knowledge_date, summary_for
from portagent.research.source_features import features_for
from portagent.research.modalities import prepare_sources
from portagent.research.io import write_json
from portagent.llm.assistant import snapshot
from portagent.llm.critic import build_evidence


def prices():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=900).strftime("%Y-%m-%d")
    returns = rng.normal(0, .01, len(dates)); returns[::30] = .08
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(returns)), "volume": 1000.}, index=dates)


def test_features_do_not_see_future_prices_and_target_uses_next_observation():
    frame = prices(); frame = frame.drop(frame.index[701:704])
    a, features = make_dataset({"US:A": frame})
    changed = frame.copy(); changed.loc[changed.index[700]:, "close"] *= 4
    b, _ = make_dataset({"US:A": changed})
    cutoff = frame.index[700]
    pd.testing.assert_frame_equal(a.loc[a.date < cutoff, features], b.loc[b.date < cutoff, features])
    row = a[a.date == frame.index[700]].iloc[0]
    assert row.target_date == frame.index[701]
    assert row.target_return == pytest.approx(np.log(frame.close.iloc[701] / frame.close.iloc[700]))
    assert not set(features) & {"target_z", "event", "target_return", "target_date", "target_gap_days"}
    assert a.iloc[-1].event != a.iloc[-1].event


def test_all_fitting_outcomes_resolve_before_next_block():
    data, _ = make_dataset({"US:A": prices(), "US:B": prices()})
    train, valid, cal, test = split_fold(data, 2022)
    for previous, following in ((train, valid), (valid, cal), (cal, test)):
        assert previous.target_date.max() < following.date.min()
    assert set(train.date).isdisjoint(test.date)


def test_paired_identical_models_have_zero_interval():
    rows = pd.DataFrame({"date": np.repeat(pd.date_range("2025-01-01", periods=80).strftime("%Y-%m-%d"), 3),
                         "event": np.tile([0, 0, 1], 80)})
    pred = pd.DataFrame({n: np.full(len(rows), .2) for n in ["frequency", "current_top3", *CANDIDATES]})
    out = paired_intervals(rows, pred, reps=100)
    assert all(r["ci_low"] == r["ci_high"] == r["improvement"] == 0 and not r["confirmed"] for r in out)


def event(**overrides):
    return {"kind": "telegram", "ticker": "US:A", "news_id": "post-1", "id": "source-1", "source": "tg:test",
        "available_date": "2026-11-15", "analyzed_at": "2026-11-14T12:00:00Z", "sentiment": -.5,
        "confidence": .8, "quote": "Apple legal dispute", "published_at": "2020-01-01", **overrides}


def test_source_availability_includes_collection_and_classification():
    assert knowledge_date("2020-01-01", "2026-11-13T00:00:00Z", "2026-11-14T23:59:00Z") == "2026-11-15"
    rows = pd.DataFrame({"asset": ["US:A"] * 3, "date": ["2026-11-14", "2026-11-15", "2026-11-16"]})
    f = features_for(rows, [event(), event(id="edit", available_date="2026-11-16", analyzed_at="2026-11-15T12:00:00Z", sentiment=-1)])
    assert f.tg_count.tolist() == [0, 1, 1]
    assert f.tg_negative.tolist() == [0, .4, .8]
    assert f.court_missing.tolist() == [1, 1, 1]
    assert f.legal_news_missing.tolist() == [1, 1, 1]


def test_media_legal_mention_is_not_court_document():
    rows = pd.DataFrame({"asset": ["US:A"], "date": ["2026-11-15"]})
    f = features_for(rows, [event(kind="legal_news")])
    assert f.legal_news_pressure.iloc[0] > 0 and f.court_pressure.iloc[0] == 0
    assert f.court_missing.iloc[0] == 1
    context = snapshot({"date": "2026-11-14", "weights": {"US:A": .9}}, {"source_events": {"records": [event(kind="legal_news")]}})
    evidence = build_evidence(context)
    assert any("не проверенный судебный документ" in r["text"] for r in evidence)


def test_source_gate_does_not_backdate_or_count_rereads(tmp_path):
    write_json(tmp_path / "records" / "one.json", event())
    for n in range(3):
        write_json(tmp_path / "collections" / f"{n}.json", {"date": "2026-11-14", "groups": {"telegram": {"batch_errors": 0}}})
    data = pd.DataFrame({"date": ["2026-11-14"], "asset": ["US:A"], "event": [1.]})
    enriched, status = prepare_sources(data, tmp_path)
    assert status["status"] == "insufficient_history" and status["collection_days"] == 1
    assert status["resolved_events"] == 0 and enriched.tg_missing.iloc[0] == 1


def test_source_summary_never_uses_future_collection(tmp_path):
    write_json(tmp_path / "events" / "latest.json", {"date": "2026-11-15", "updated_at": "2026-11-15", "court_documents": 8})
    assert summary_for({"data": tmp_path}, "2026-11-14").get("court_documents") is None


def test_source_refresh_excludes_demo_and_is_idempotent(tmp_path, monkeypatch):
    from portagent.config import load_config, project_paths
    from portagent.research import events
    cfgpath = tmp_path / "configs" / "default.yaml"; cfgpath.parent.mkdir()
    cfgpath.write_bytes((Path(__file__).parents[1] / "configs/default.yaml").read_bytes())
    cfg = load_config(cfgpath); paths = project_paths(cfg, cfgpath)
    (paths["data"] / "raw" / "2026-11-14").mkdir(parents=True)
    (paths["data"] / "raw" / "2026-11-14" / "manifest_news.json").touch()
    write_json(paths["data"] / "manual/courts/2026-11-14/demo.json", [{"case": "not-real"}])
    item = {"id": "post-1", "title": "Apple legal dispute", "text": "Apple legal dispute", "source": "tg:test", "published_at": "2026-11-14", "link": "https://example.com"}
    monkeypatch.setattr(events, "verified_entries", lambda *a: ({"fetched_at": "2026-11-14", "meta": {"source": "net", "feeds": []}}, [("tg_test", b"x")]))
    monkeypatch.setattr(events, "parse_tg", lambda *a, **k: [item])
    monkeypatch.setattr(events, "archived_news", lambda *a: [])
    def chat(messages, **kw):
        items = json.loads(messages[-1]["content"])["items"]
        return json.dumps({"items": [{"id": r["id"], "tickers": ["US:AAPL"], "sentiment": -.5, "confidence": .8, "quote": "Apple legal dispute"} for r in items]}), {}
    first = events.refresh_events(cfg, cfgpath, "2026-11-14", chat=chat)
    before = {p: p.read_bytes() for p in (paths["data"] / "events/records").glob("*.json")}
    assert len(before) == 1 and first["excluded_demo_documents"] and first["court_documents"] == 0
    events.refresh_events(cfg, cfgpath, "2026-11-14", chat=chat)
    assert before == {p: p.read_bytes() for p in (paths["data"] / "events/records").glob("*.json")}


def test_human_risk_explanation_keeps_math_in_details():
    from portagent.report.diagnostics import explanations
    row = {"asset": "MOEX:TATN", "warned": True, "score": .928, "drawdown": .123,
        "bias": -1.6, "components": {"p_tail": 1.81, "sentiment": 1.2, "macro": .79}}
    result = explanations({"tracker": {"rows": [row]}})["MOEX:TATN"]
    assert "12,3%" in result["summary"]
    assert "позитивнее обычного" in result["caveat"]
    assert all(word not in result["summary"] for word in ("логист", "вклад", "0.928", "хвост"))
    assert any("логист" in f for f in result["technical_facts"])
    assert result["score"] == row["score"] and result["components"][0]["contribution"] == 1.81
    assert len(result["brief"]) < len(result["summary"])
    assert "12,3%" in result["brief"] and "логист" not in result["brief"]


def test_requested_asset_sources_survive_evidence_limit():
    records = [event(id=str(i), ticker="US:AAPL", sentiment=-1) for i in range(10)]
    records.append(event(id="target", ticker="MOEX:TATN", sentiment=.5))
    evidence = build_evidence({"date": "2026-11-14", "weights": {"MOEX:TATN": .1},
        "focus_assets": ["MOEX:TATN"], "source_events": {"records": records}})
    assert any(e["id"] == "EV-SOURCE-target" for e in evidence)


def test_brief_distinguishes_recovery_decline_and_news_artifact():
    from portagent.report.briefing import asset_brief
    row = {"drawdown": .25, "warned": True}
    rebound = asset_brief(row, {"p_tail": 2., "vol_of_vol": .05}, {"ret_5d": .037, "ret_1d": .004})
    decline = asset_brief(row, {"p_tail": .5, "vol_of_vol": 1.5}, {"ret_5d": -.031, "ret_1d": .002})
    news = asset_brief(row, {"p_tail": .7, "sentiment": 1.2}, {"ret_5d": .009, "ret_1d": .001})
    assert "Отскочила" in rebound["situation"] and "продавцы" in rebound["watch"]
    assert "потеряла" in decline["situation"] and "Прекратится ли снижение" in decline["watch"]
    assert "улучшившиеся" in news["importance"] and "не ухудшается" in news["watch"]
    assert "главный сигнал" not in str([rebound, decline, news]).lower()
    missing = asset_brief({}, {}, {})
    assert "среза нет" in missing["situation"] and "%" not in missing["situation"]


def test_model_selector_excludes_missing_or_failed_robustness_checks():
    from portagent.web.research import eligible_models
    def good(name, base="frequency"):
        return {"baseline": base, "model": name, "confirmed": True, "ci_low": .0001}
    report = {"confirmed_models": ["steady", "fragile"], "comparisons": [good(m, b) for m in ("steady", "fragile") for b in ("frequency", "current_top3")],
        "block_sensitivity": {str(k): [good(m, b) for m in ("steady", "fragile") for b in ("frequency", "current_top3")] for k in (20,60,120)},
        "additional_control": {"comparisons": [good(m,"frequency_cal") for m in ("steady", "fragile")]}}
    report["block_sensitivity"]["120"][2].update(confirmed=False, ci_low=-.0001)
    assert eligible_models(report) == ["steady"]
    del report["additional_control"]
    assert eligible_models(report) == []
