"""Трек B: дивиденды (total return), макро-стресс, флаг отчётностей."""
import json

import numpy as np
import pandas as pd
import pytest

from portagent.collect.dividends import load_dividend_drops
from portagent.collect.earnings import load_earnings_drops
from portagent.history import (load_dividends_panel, load_earnings,
                               load_macro_panel, upsert_dividends, upsert_earnings,
                               upsert_macro)
from portagent.tracker.rare_events import (_earnings_flag, _macro_stress,
                                           asset_scores)


# --- B5: дивиденды ---

def test_dividend_drops_parsing(tmp_path):
    (tmp_path / "sber.json").write_text(json.dumps({"dividends": [
        {"asset": "MOEX:SBER", "date": "2026-05-12", "value": 33.3},
        {"asset": "MOEX:SBER", "date": "2026-07-14", "value": 35.0}]}),
        encoding="utf-8")
    (tmp_path / "raw_list.json").write_text(json.dumps(
        [{"asset": "MOEX:LKOH", "date": "2026-06-01", "value": 496}]),
        encoding="utf-8")
    rows = load_dividend_drops(tmp_path)
    assert {(r["asset"], r["date"], r["value"]) for r in rows} == {
        ("MOEX:SBER", "2026-05-12", 33.3), ("MOEX:SBER", "2026-07-14", 35.0),
        ("MOEX:LKOH", "2026-06-01", 496.0)}


def test_dividends_panel_upsert_and_wide(tmp_path):
    upsert_dividends(tmp_path, [{"asset": "MOEX:SBER", "date": "2026-05-12", "value": 33.3}])
    upsert_dividends(tmp_path, [{"asset": "MOEX:SBER", "date": "2026-05-12", "value": 33.3},
                                {"asset": "MOEX:GAZP", "date": "2026-07-14", "value": 56.4}])
    panel = load_dividends_panel(tmp_path)
    assert panel.shape == (2, 2)
    assert panel.loc["2026-05-12", "MOEX:SBER"] == pytest.approx(33.3)
    assert load_dividends_panel(tmp_path, end="2026-06-01").shape[0] == 1


def test_walk_forward_total_return_adds_dividend_yield():
    from portagent.backtest import walk_forward
    import yaml
    from pathlib import Path

    config = yaml.safe_load((Path(__file__).parent.parent / "configs" / "default.yaml")
                            .read_text(encoding="utf-8"))
    config["hrp"]["max_weight"] = 0.8      # тестовая вселенная из 2 активов
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2025-06-02", periods=140).strftime("%Y-%m-%d")
    closes = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (140, 2)), axis=0)),
                          index=index, columns=["MOEX:SBER", "US:AAPL"])
    dividends = pd.DataFrame(0.0, index=index, columns=["MOEX:SBER"])
    ex_date = index[-60]
    dividends.loc[ex_date, "MOEX:SBER"] = 5.0  # 5% дивиденд на цене ~100

    base = walk_forward(closes, config=config, include_daily=True)
    total = walk_forward(closes, config=config, include_daily=True, dividends=dividends)
    assert total["strategies"]["fused"]["total_return"] > base["strategies"]["fused"]["total_return"]
    # ex-дата внутри окна: прирост существенный (>= 1 п.п. на полпериода)
    lift = (total["strategies"]["fused"]["total_return"]
            - base["strategies"]["fused"]["total_return"])
    assert lift > 0.01, f"дивиденд не начислен: {lift}"


# --- B6: макро-стресс ---

def _macro_frame(days: int, brent_shock_at: int | None, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-02", periods=days).strftime("%Y-%m-%d")
    brent = 90.0 * np.exp(np.cumsum(rng.normal(0, 0.015, days)))
    if brent_shock_at is not None:
        brent[brent_shock_at:] *= 0.6            # обвал нефти на 40%
    return pd.DataFrame({"brent": brent}, index=index)


def _asset_returns(macro: pd.DataFrame, oil_beta: float, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    brent_ret = np.log(macro["brent"] / macro["brent"].shift(1)).fillna(0).to_numpy()
    n = len(macro)
    oil = oil_beta * brent_ret + rng.normal(0, 0.01, n)
    tech = rng.normal(0, 0.01, n)
    return pd.DataFrame({"MOEX:LKOH": oil, "US:MSFT": tech}, index=macro.index)


def test_macro_stress_higher_for_oil_beta_asset_in_shock():
    days = 320
    macro = _macro_frame(days, brent_shock_at=days - 40)
    returns = _asset_returns(macro, oil_beta=1.0)
    closes = np.exp(np.cumsum(returns.to_numpy(), axis=0))
    closes = pd.DataFrame(closes, index=macro.index, columns=returns.columns)
    stress = _macro_stress(macro, returns, window=250)
    window_after = stress.iloc[-60:]                       # покрывает окно шока
    oil_peak = float(window_after["MOEX:LKOH"].max())
    tech_peak = float(window_after["US:MSFT"].max())
    assert oil_peak > 0.5, f"стресс в окно шума не поднят: {oil_peak}"
    assert oil_peak > tech_peak * 3, f"нефтяная бета не поймана: {oil_peak} vs {tech_peak}"
    # после шока стресс затухает (rolling-std впитывает событие)
    assert float(stress["MOEX:LKOH"].iloc[-1]) < oil_peak


def test_macro_stress_zero_when_calm_and_when_none():
    days = 320
    macro = _macro_frame(days, brent_shock_at=None)
    returns = _asset_returns(macro, oil_beta=1.0)
    stress = _macro_stress(macro, returns, window=250)
    assert float(stress["MOEX:LKOH"].iloc[-1]) < 0.5   # спокойно: |z| мал
    none_stress = _macro_stress(None, returns, window=250)
    assert (none_stress == 0).all().all()


# --- B7: флаг отчётностей ---

def test_earnings_flag_window():
    index = pd.bdate_range("2026-11-02", periods=12).strftime("%Y-%m-%d")
    returns = pd.DataFrame(0.0, index=index, columns=["US:AAPL", "US:MSFT"])
    flag = _earnings_flag([("2026-11-13", "US:AAPL")], returns)  # будущая дата
    aapl = flag["US:AAPL"]
    # календарное окно 7 дней до отчёта: 06..12 ноября
    assert aapl.loc["2026-11-06":"2026-11-12"].sum() == 5        # будни в окне
    assert aapl.loc["2026-11-02":"2026-11-05"].sum() == 0
    assert (flag["US:MSFT"] == 0).all()


def test_earnings_flag_future_date_beyond_index():
    """Дата отчёта позже конца ценового индекса — флаг всё равно работает."""
    index = pd.bdate_range("2026-11-02", periods=8).strftime("%Y-%m-%d")
    returns = pd.DataFrame(0.0, index=index, columns=["US:NVDA"])
    flag = _earnings_flag([("2026-11-17", "US:NVDA")], returns)
    assert flag["US:NVDA"].iloc[-1] == 1.0                      # 12 ноября: 5 дней до 17-го
    assert flag["US:NVDA"].iloc[0] == 0.0                       # 2 ноября: 15 дней, вне окна


def test_earnings_drops_and_history(tmp_path):
    drop = tmp_path / "manual" / "earnings" / "2026-11-12"
    drop.mkdir(parents=True)
    (drop / "ru.json").write_text(json.dumps(
        {"earnings": [{"asset": "MOEX:SBER", "date": "2026-11-20"}]}), encoding="utf-8")
    rows = load_earnings_drops(tmp_path / "manual" / "earnings", "2026-11-12")
    assert rows == [{"asset": "MOEX:SBER", "date": "2026-11-20"}]
    upsert_earnings(tmp_path, rows + [{"asset": "US:AAPL", "date": "2027-01-28"}])
    assert len(load_earnings(tmp_path)) == 2


def test_macro_history_roundtrip(tmp_path):
    upsert_macro(tmp_path, [{"date": "2026-11-12", "series": "usdrub", "value": 83.4},
                            {"date": "2026-11-12", "series": "brent", "value": 95.8}])
    upsert_macro(tmp_path, [{"date": "2026-11-12", "series": "usdrub", "value": 83.4},
                            {"date": "2026-11-11", "series": "usdrub", "value": 83.1}])
    panel = load_macro_panel(tmp_path)
    assert set(panel.columns) == {"usdrub", "brent"}
    assert panel.loc["2026-11-11", "usdrub"] == pytest.approx(83.1)


def test_asset_scores_accepts_macro_and_earnings():
    days = 200
    index = pd.bdate_range("2025-06-02", periods=days).strftime("%Y-%m-%d")
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0, 0.01, (days, 2)), index=index,
                           columns=["MOEX:SBER", "US:AAPL"])
    closes = np.exp(np.cumsum(returns.to_numpy(), axis=0))
    closes = pd.DataFrame(closes, index=index, columns=returns.columns)
    macro = pd.DataFrame({"brent": 90.0 * np.exp(
        np.cumsum(rng.normal(0, 0.015, days)))}, index=index)
    scores = asset_scores(returns, closes, window=120,
                          macro=macro, earnings_dates=[(index[-1], "US:AAPL")])
    assert scores.notna().all().all()
    assert ((scores >= 0) & (scores <= 1)).all().all()
    # без новых источников — прежнее поведение
    scores_plain = asset_scores(returns, closes, window=120)
    pd.testing.assert_frame_equal(scores_plain, asset_scores(returns, closes, window=120))
