import copy
import json
from pathlib import Path

import pytest

from portagent.config import load_config, project_paths
from portagent.report import recovery
from portagent.llm.news_analyst import analyze_news
from portagent.llm.assistant import snapshot
from portagent.llm.critic import build_evidence
from portagent.web.jobs import load_report

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
DAY = "2026-11-14"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg_path = tmp_path / "configs" / "default.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_bytes(CONFIG.read_bytes())
    cfg = load_config(cfg_path)
    cfg["llm"]["analyst_batch_size"] = 1
    paths = project_paths(cfg, cfg_path)
    original = {"date": DAY, "assets": ["US:AAPL"], "weights": {"US:AAPL": .9},
                "news_total": 3, "llm": {"enabled": True, "batch_errors": 2},
                "fusion": {"note": "price_only"}, "tracker": {"rows": []}}
    target = paths["reports"] / DAY / "portfolio.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(original), encoding="utf-8")
    forecast = paths["data"] / "forecasts" / f"{DAY}.json"
    forecast.parent.mkdir(parents=True)
    forecast.write_text('{"weights":{"US:AAPL":0.9}}', encoding="utf-8")
    items = [{"id": str(i), "title": "Apple growth", "description": "Apple growth", "text": "Apple growth",
              "source": "fixture", "link": "https://example.com", "published_at": DAY}
             for i in range(3)]
    monkeypatch.setattr(recovery, "archived_news", lambda *a: items)
    monkeypatch.setattr(recovery, "explain", lambda *a, **kw: {"response": {"summary": "Готово"}})
    calls = []
    def chat(messages, **kwargs):
        # Payload submitted to the model is the source of IDs; no invented source quotes.
        content = json.loads(messages[-1]["content"])
        rows = content["items"]
        calls.extend(r["id"] for r in rows)
        return json.dumps({"items": [{"id": r["id"], "confidence": .9, "tickers": ["US:AAPL"],
            "sentiment": .2, "quote": "Apple growth"} for r in rows]}), {}
    return cfg, cfg_path, paths, original, target, forecast, items, chat, calls


def test_recovery_reuses_successful_batch_and_preserves_frozen_files(setup):
    cfg, cfg_path, paths, original, target, forecast, items, chat, calls = setup
    before = target.read_bytes(), forecast.read_bytes()
    cached = analyze_news(items[:1], universe=original["assets"], chat=chat,
        model=cfg["llm"]["model"], batch_size=1, cache_dir=paths["data"] / "cache" / "llm_labels",
        cache_salt=f"mt{cfg['llm']['max_tokens']}")
    assert cached["stats"]["batch_errors"] == 0
    calls.clear()
    result = recovery.recover_news(cfg, cfg_path, DAY, chat=chat)
    assert calls == ["1", "2"]
    assert result["llm"]["cache_hits"] == 1 and result["llm"]["batch_errors"] == 0
    assert (target.read_bytes(), forecast.read_bytes()) == before
    loaded = load_report(paths, DAY)
    assert loaded["weights"] == original["weights"] and loaded["news_labels"]
    assert loaded["fusion"] == original["fusion"]
    assert "Дополненный анализ" in snapshot(loaded)["coverage_note"]
    # A mismatched sidecar cannot silently change weights.
    value = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    value["weights"]["US:AAPL"] = .1
    Path(result["path"]).write_text(json.dumps(value), encoding="utf-8")
    assert load_report(paths, DAY) == original


def test_failed_critique_keeps_checkpoint_without_publishing(setup, monkeypatch):
    cfg, cfg_path, paths, original, target, forecast, items, chat, calls = setup
    before = target.read_bytes(), forecast.read_bytes()
    def failed(*args, **kwargs):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(recovery, "explain", failed)
    with pytest.raises(RuntimeError, match="unavailable"):
        recovery.recover_news(cfg, cfg_path, DAY, chat=chat)
    directory = target.parent / "analysis_updates"
    assert (directory / "recovery_checkpoint.json").exists()
    assert not (directory / "latest.json").exists()
    assert (target.read_bytes(), forecast.read_bytes()) == before


def test_raw_tampering_is_rejected_before_parsing(tmp_path):
    paths = {"data": tmp_path}
    directory = tmp_path / "raw" / DAY
    directory.mkdir(parents=True)
    name = "rss_0_fixture__abc.xml"
    (directory / name).write_bytes(b"altered")
    (directory / "manifest_news.json").write_text(json.dumps({"date": DAY, "kind": "news",
        "entries": [{"name": name, "sha256": recovery.digest(b"original")}]}))
    with pytest.raises(ValueError, match="контрольная сумма"):
        recovery.verified_entries(paths, DAY, "news")


def test_llm_receives_warning_causes_separately_from_new_news():
    report = {"date": DAY, "weights": {"A": 1}}
    diag = {"risk_explanations": {"A": {"warned": True, "facts": ["Вклад просадки +1.20."]}}}
    evidence = build_evidence(snapshot(report, diag))
    cause = next(r for r in evidence if r["id"] == "EV-CAUSE-A")
    assert "+1.20" in cause["text"] and "не меняли" in cause["text"]
    assert "risk_explanations" not in report
