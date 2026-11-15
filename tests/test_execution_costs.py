"""Ручные контрольные расчёты: торговый оборот, самофинансирование и дрейф."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import portagent.backtest as backtest
from portagent.config import ConfigError, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_initial_entry_costs_include_cash_and_half_spread():
    trade = backtest._execution(pd.Series({"A": 0.0}), pd.Series({"A": 0.9}),
                               commission_bps=5, spread_bps=10)
    # NAV_after + 0.1% * (90% NAV_after) = 1
    post_nav = 1 / 1.0009
    assert trade["total"] == pytest.approx(1 - post_nav)
    assert trade["buys"] == pytest.approx(0.9 * post_nav)
    assert trade["sells"] == 0
    assert trade["commission"] == pytest.approx(trade["spread"])


def test_switch_charges_both_sides_and_no_cash_commission():
    trade = backtest._execution(pd.Series({"A": 1.0, "B": 0.0}),
                               pd.Series({"A": 0.0, "B": 1.0}),
                               commission_bps=5, spread_bps=10)
    post_nav = (1 - 0.001) / (1 + 0.001)
    assert trade["sells"] == pytest.approx(1)
    assert trade["buys"] == pytest.approx(post_nav)
    assert trade["total"] == pytest.approx(1 - post_nav)


def test_unchanged_positions_have_no_cost_and_sell_to_cash_is_charged():
    before = pd.Series({"A": 0.4, "B": 0.5})
    assert backtest._execution(before, before, commission_bps=5, spread_bps=10)["total"] == 0
    sold = backtest._execution(before, before * 0, commission_bps=5, spread_bps=10)
    assert sold["sells"] == pytest.approx(0.9)
    assert sold["total"] == pytest.approx(0.0009)


def test_holdings_drift_without_daily_rebalancing():
    changes = pd.DataFrame({"A": [1.0, -0.5], "B": [0.0, 0.0]})
    daily, end_weights = backtest._hold_period(changes, pd.Series({"A": 0.4, "B": 0.4}))
    assert daily.iloc[0] == pytest.approx(0.4)
    assert daily.iloc[1] == pytest.approx(-0.4 / 1.4)
    assert (1 + daily).prod() == pytest.approx(1.0)
    assert end_weights.to_dict() == pytest.approx({"A": 0.4, "B": 0.4})


def _walk(monkeypatch, *, rate=5, spread=10, shock=False):
    config = load_config(CONFIG)
    config["backtest"].update(commission_bps=rate, spread_bps=spread)
    config["hrp"].update(cash_share=0.0, max_weight=1.0)
    dates = pd.bdate_range("2025-01-01", periods=65).strftime("%Y-%m-%d")
    panel = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    if shock:
        panel.loc[dates[61]:, "A"] = 200.0
    monkeypatch.setattr(backtest, "_hrp", lambda *a, **kw: pd.Series({"A": 0.5, "B": 0.5}))
    monkeypatch.setattr(backtest, "_tracker_quality", lambda *a, **kw: {})
    return backtest.walk_forward(panel, config=config, rebalance_days=2, include_daily=True)


def test_flat_market_only_initial_entry_is_charged(monkeypatch):
    result = _walk(monkeypatch)
    for name, stats in result["strategies"].items():
        assert stats["total_return"] == pytest.approx(1 / 1.001 - 1)
        assert stats["max_drawdown"] == pytest.approx(stats["total_return"])
        assert stats["cost_paid"] == pytest.approx(-stats["total_return"])
        assert result["gross_strategies"][name]["total_return"] == 0
        assert result["executions"][name][-1]["total"] == 0
        assert np.prod(1 + np.array(result["daily_returns"][name])) - 1 == pytest.approx(stats["total_return"])


def test_drift_causes_trades_at_next_rebalance_and_cost_is_on_that_day(monkeypatch):
    result = _walk(monkeypatch, shock=True)
    trades = result["executions"]["equal_weight"]
    assert trades[1]["sells"] > 0 and trades[1]["buys"] > 0
    dates = result["daily_dates"]
    rebalance_index = dates.index(trades[1]["date"])
    net = result["daily_returns"]["equal_weight"]
    gross = result["gross_daily_returns"]["equal_weight"]
    assert net[rebalance_index] == pytest.approx((1 + gross[rebalance_index]) * (1 - trades[1]["total"]) - 1)
    assert net[rebalance_index + 1] == gross[rebalance_index + 1]
    assert result["strategies"]["equal_weight"]["total_return"] < 0.5
    assert result["gross_strategies"]["equal_weight"]["total_return"] == pytest.approx(0.5)


def test_zero_costs_equal_gross_and_do_not_change_weights(monkeypatch):
    free = _walk(monkeypatch, rate=0, spread=0, shock=True)
    paid = _walk(monkeypatch, shock=True)
    assert free["daily_returns"] == free["gross_daily_returns"]
    assert free["gross_daily_returns"] == paid["gross_daily_returns"]


@pytest.mark.parametrize("field,value", [("commission_bps", "-1"), ("spread_bps", ".nan"),
                                          ("spread_bps", "true")])
def test_cost_config_rejects_invalid_values(tmp_path, field, value):
    import yaml
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["backtest"][field] = yaml.safe_load(value)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)
