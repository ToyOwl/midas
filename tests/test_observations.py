"""Trading-observation horizons, path-dependent outcomes and saved contracts."""
import copy
import json

import pandas as pd
import pytest

from portagent.tracker.observations import Observations, evaluate, make_observation, price_series
from portagent.history import upsert_prices
from test_web import app, request


def series(values, start="2026-10-19"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)).strftime("%Y-%m-%d"))


def args(kind="hold_gain", **kw):
    return dict(asset="US:AAPL", kind=kind, sessions=5, threshold_pct=3, **kw)


def record(kind="hold_gain", values=None):
    prices = series(values or [100, 102, 104, 106, 108, 110])
    return make_observation(args(kind), {"US:AAPL": prices}, "2026-10-26"), prices


def test_weekends_missing_quotes_and_future_rows_do_not_consume_horizon():
    r, prices = record()
    prices.loc["2026-10-27"] = float("nan")
    prices.loc["2026-10-28"] = 111
    prices.loc["2026-11-02"] = 50
    result = evaluate(r, prices, "2026-10-29")
    assert result["status"] == "watching"
    assert result["points"] == [["2026-10-28", 111.0]]
    assert evaluate(r, prices, "2026-10-26")["points"] == []


def test_drawdown_breach_survives_rebound_and_is_terminal():
    r, _ = record()
    prices = series([100, 102, 104, 106, 108, 110, 120, 115, 125, 125, 125])
    result = evaluate(r, prices, "2026-11-30")
    assert result["status"] == "broken"  # 115 is above anchor, but >3% off 120.
    assert result["points"][-1][1] == 115
    assert len(result["points"]) == 2
    prices.iloc[-1] = 130
    assert evaluate(result, prices, "2026-12-01") == result
    assert r["status"] == "waiting" and r["points"] == []


def test_half_gain_boundary_and_horizon_are_frozen():
    r, _ = record()
    r["threshold"] = .5
    r["threshold_pct"] = 50
    result = evaluate(r, series([100, 102, 104, 106, 108, 110, 105, 105, 105, 105, 105, 80]), "2026-12-01")
    assert result["status"] == "held"  # equality allowed; sixth close out of scope
    assert len(result["points"]) == 5
    fail = evaluate(r, series([100, 102, 104, 106, 108, 110, 104.99, 110]), "2026-12-01")
    assert fail["status"] == "broken"


@pytest.mark.parametrize("future,status", [([91, 92, 93, 94, 95], "recovered"),
    ([95, 95, 95, 94, 94], "not_recovered"), ([92, 93, 89, 95, 96], "broken")])
def test_recovery_requires_target_at_end_and_no_breach(future, status):
    base = [100, 98, 96, 94, 92, 90]
    r, _ = record("recovery", base)
    assert evaluate(r, series(base + future), "2026-12-01")["status"] == status


@pytest.mark.parametrize("future,status", [([113.3], "triggered"), ([106.7], "triggered"),
    ([111, 112, 113, 114, 115], "quiet")])
def test_large_move_uses_single_session_both_directions(future, status):
    base = [100, 102, 104, 106, 108, 110]
    r, _ = record("large_move")
    assert evaluate(r, series(base + future), "2026-12-01")["status"] == status


def test_revised_prices_pause_without_mixing_bases():
    r, prices = record()
    result = evaluate(r, prices * .5, "2026-12-01")
    assert result["data_warning"] and result["status"] == "waiting" and result["points"] == []
    assert evaluate(result, prices, "2026-12-01")["data_warning"] is None
    assert evaluate(r, pd.Series(dtype=float), "2026-12-01")["data_warning"]


def test_backfill_before_frozen_observation_pauses():
    r, prices = record()
    prices.loc["2026-10-28"] = 111
    frozen = evaluate(r, prices, "2026-10-29")
    prices.loc["2026-10-27"] = 90
    revised = evaluate(frozen, prices, "2026-10-29")
    assert revised["data_warning"] and revised["points"] == frozen["points"]


def test_store_restart_dedup_and_stop(tmp_path):
    _, prices = record()
    store = Observations(tmp_path)
    first = store.create(args(), {"US:AAPL": prices}, "2026-10-26")
    assert Observations(tmp_path).create(args(), {"US:AAPL": prices}, "2026-10-26") == first
    prices.loc["2026-10-27"] = 111
    rows = store.refresh({"US:AAPL": prices}, "2026-10-27")
    assert rows[0]["completed"] == 1
    assert len(rows) == 1
    assert store.stop(first["id"])["status"] == "stopped"
    prices.loc["2026-10-28"] = 1
    assert Observations(tmp_path).refresh({"US:AAPL": prices}, "2026-10-28")[0]["completed"] == 1


@pytest.mark.parametrize("field,value", [("kind", "shell"), ("kind", []), ("asset", "../../file"),
    ("sessions", True), ("sessions", 4), ("threshold_pct", float("nan")),
    ("threshold_pct", 0), ("threshold_pct", 51), ("threshold_pct", "3")])
def test_strict_observation_parameters(field, value):
    _, prices = record()
    value_args = args()
    value_args[field] = value
    with pytest.raises(ValueError):
        make_observation(value_args, {"US:AAPL": prices}, "2026-10-26")


def test_wrong_direction_and_short_history_are_rejected():
    _, prices = record()
    with pytest.raises(ValueError):
        make_observation(args("recovery"), {"US:AAPL": prices}, "2026-10-26")
    with pytest.raises(ValueError):
        make_observation(args(), {"US:AAPL": prices.iloc[:3]}, "2026-10-26")


def test_verified_archive_does_not_merge_demo_rows(app, monkeypatch):
    from portagent.report import recovery
    _, prices = record()
    contaminated = prices.copy()
    contaminated.loc["2026-10-25"] = 20
    upsert_prices(app.paths["data"] / "history", {"US:AAPL": contaminated.to_frame("close")})
    raw = app.paths["data"] / "raw" / "2026-10-26"
    raw.mkdir(parents=True)
    (raw / "manifest_market.json").write_text('{"meta":{"source":"net"}}')
    monkeypatch.setattr(recovery, "archived_prices", lambda *a: {"US:AAPL": prices.to_frame("close")})
    actual = price_series(app.paths, app.config, "2026-10-29")
    assert actual["US:AAPL"].equals(prices)
    (raw / "manifest_market.json").write_text('{"meta":{"source":"fixture"}}')
    assert price_series(app.paths, app.config, "2026-10-29") == {}


def test_http_lifecycle_auth_frozen_files_and_latest_history(app, monkeypatch):
    from portagent.tracker import observations
    monkeypatch.setattr(observations, "today", lambda: "2026-10-29")
    _, prices = record()
    upsert_prices(app.paths["data"] / "history", {"US:AAPL": prices.to_frame("close")})
    target = app.paths["reports"] / "2026-10-26" / "portfolio.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"date": "2026-10-26", "weights": {"US:AAPL": 1}}))
    before = target.read_bytes()
    # Other universe assets are absent: use a config scoped to this fixture.
    app.config["universe"] = {"moex": [], "us": ["AAPL"]}
    assert request(app, "/api/observations", headers={"Cookie": None})["status"] == 401
    assert request(app, "/api/observations", method="POST", data=args(), headers={"X-CSRF-Token": "bad"})["status"] == 403
    created = request(app, "/api/observations", method="POST", data=args())
    assert created["status"] == 201
    first = request(app, "/api/observations")["value"]
    assert first["items"][0]["status"] == "waiting"
    prices.loc["2026-10-27"] = 111
    prices.loc["2026-10-28"] = 100
    prices.loc["2026-11-01"] = 120  # must not leak into candidate/observation
    upsert_prices(app.paths["data"] / "history", {"US:AAPL": prices.to_frame("close")})
    result = request(app, "/api/observations")["value"]
    assert result["items"][0]["status"] == "broken"
    assert result["candidates"][0]["date"] == "2026-10-28"
    assert request(app, "/api/observations/stop", method="POST", data={"id":created["value"]["id"]})["value"]["status"] == "broken"
    assert target.read_bytes() == before
