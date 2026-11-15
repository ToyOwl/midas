import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from portagent import scheduler
from portagent.config import load_config
from portagent.llm.runtime import launch_lock

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    path = tmp_path / "configs" / "default.yaml"
    path.parent.mkdir()
    path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    calls = []

    def daily(*args, **kwargs):
        day = kwargs["date"]
        calls.append(day)
        directory = tmp_path / "reports" / day
        directory.mkdir(parents=True, exist_ok=True)
        report = {"date": day, "weights": {"A": 0.9}}
        (directory / "portfolio.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "portfolio.md").write_text("report", encoding="utf-8")
        (directory / "weights.csv").write_text("asset,weight\nA,0.9", encoding="utf-8")
        return {"report": report, "paths": {"markdown": directory / "portfolio.md"}}

    monkeypatch.setattr(scheduler, "run_daily", daily)
    monkeypatch.setattr(scheduler, "_health", lambda *a: [])
    return config, path, calls


def test_due_time_is_moscow_and_duplicate_is_skipped(setup):
    config, path, calls = setup
    early = datetime(2026, 11, 14, 3, 59, tzinfo=timezone.utc)
    due = datetime(2026, 11, 14, 4, 0, tzinfo=timezone.utc)
    assert scheduler.run_scheduled(config, path, now=early, use_llm=False)["status"] == "not_due"
    assert not calls
    assert scheduler.run_scheduled(config, path, now=due, use_llm=False)["status"] == "succeeded"
    assert scheduler.run_scheduled(config, path, now=due, use_llm=False)["status"] == "already_completed"
    assert calls == ["2026-11-14"]


def test_overlap_does_not_touch_existing_state(setup):
    config, path, calls = setup
    runtime = path.parent.parent / "data" / "runtime"
    with launch_lock(runtime / "daily.lock"):
        result = scheduler.run_scheduled(config, path, due_at="00:00", use_llm=False)
    assert result["status"] == "already_running"
    assert not calls
    assert not (runtime / "scheduler" / "state.json").exists()


def test_failure_is_logged_and_lock_released_for_retry(setup, monkeypatch):
    config, path, calls = setup
    original = scheduler.run_daily

    def fail(*a, **kw):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(scheduler, "run_daily", fail)
    first = scheduler.run_scheduled(config, path, due_at="00:00", use_llm=False)
    assert first["exit_code"] == 1 and first["status"] == "failed"
    assert "provider unavailable" in Path(first["log"]).read_text(encoding="utf-8")
    monkeypatch.setattr(scheduler, "run_daily", original)
    assert scheduler.run_scheduled(config, path, due_at="00:00", use_llm=False)["status"] == "succeeded"
    assert len(calls) == 1


def test_llm_start_failure_keeps_daily_fallback(setup, monkeypatch):
    config, path, calls = setup

    def fail(*a, **kw):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(scheduler.RuntimeManager, "ensure_ready", fail)
    result = scheduler.run_scheduled(config, path, due_at="00:00")
    assert result["status"] == "degraded" and result["exit_code"] == 0
    assert "model unavailable" in result["warnings"][0]
    assert len(calls) == 1


def test_catch_up_only_current_moscow_day(setup):
    config, path, calls = setup
    # Уже 15 ноября в Москве; не выдаём сегодняшние новости за вчерашний прогноз.
    now = datetime(2026, 11, 14, 22, 0, tzinfo=timezone.utc)
    scheduler.run_scheduled(config, path, due_at="00:00", now=now, use_llm=False)
    assert calls == ["2026-11-15"]


def test_health_detects_stale_assets_and_failed_sources(tmp_path, monkeypatch):
    import pandas as pd
    from portagent import history

    panel = pd.DataFrame({"A": [1.0, None], "B": [1.0, 2.0]},
                         index=["2026-11-01", "2026-11-13"])
    monkeypatch.setattr(history, "load_price_panel", lambda *a: panel)
    warnings = scheduler._health(
        {"provider_statuses": [{"provider": "moex", "failed": 1}],
         "feed_statuses": [{"ok": False}], "history_error": "disk full"},
        {"data": tmp_path}, "2026-11-14")
    assert "stale_prices:A" in warnings
    assert "forecast_missing" in warnings and "feeds_failed:1" in warnings
    assert "provider:moex:1" in warnings and "history:disk full" in warnings
