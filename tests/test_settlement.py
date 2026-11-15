"""A1: заморозка прогноза, settlement без подглядывания, таблица качества."""
import json

import numpy as np
import pandas as pd
import pytest

from portagent.settlement import (PORTFOLIO_ROW, quality_table, settle_forecast,
                                  settle_pending, write_forecast)

HORIZON = 5


def _panel(days: int = 40, crash_day: int | None = None, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(end="2026-11-12", periods=days, freq="D").strftime("%Y-%m-%d")
    returns = rng.normal(0.0, 0.01, (days, 2))
    if crash_day is not None:
        returns[crash_day, 0] = -0.12          # обвал первого актива
    return pd.DataFrame(100.0 * np.cumprod(1 + returns, axis=0),
                        index=index, columns=["MOEX:SBER", "US:AAPL"])


def _forecast(date: str, weights=None, warned=None) -> dict:
    return {
        "date": date,
        "weights": weights or {"MOEX:SBER": 0.5, "US:AAPL": 0.4},
        "tracker_scores": {"MOEX:SBER": 0.8, "US:AAPL": 0.2},
        "tracker_warned": warned if warned is not None else {"MOEX:SBER": True, "US:AAPL": False},
    }


def test_write_forecast_immutable_and_idempotent(tmp_path):
    first = write_forecast(tmp_path, _forecast("2026-11-10"))
    same = write_forecast(tmp_path, _forecast("2026-11-10"))     # идентичный — no-op
    assert first == same
    with pytest.raises(FileExistsError, match="immutable"):
        write_forecast(tmp_path, _forecast("2026-11-10", weights={"MOEX:SBER": 0.9}))
    frozen = json.loads((tmp_path / "2026-11-10.json").read_text(encoding="utf-8"))
    assert frozen["snapshot_sha"]


def test_settle_pending_when_not_enough_future(tmp_path):
    panel = _panel()
    forecast_date = panel.index[-1]           # сегодня — будущего нет
    write_forecast(tmp_path / "forecasts", _forecast(str(forecast_date)))
    summary = settle_pending(tmp_path / "forecasts", tmp_path / "settlement",
                             tmp_path / "history", panel, threshold=2.5,
                             halflife=20.0, horizon=HORIZON)
    assert summary["settled"] == [] and summary["pending"] == [str(forecast_date)]


def test_settle_records_hit_on_crash(tmp_path):
    panel = _panel(crash_day=-3)              # обвал за 3 дня до конца
    forecast_date = str(panel.index[-7])      # горизонт 5 дней захватывает обвал
    write_forecast(tmp_path / "forecasts", _forecast(forecast_date))
    summary = settle_pending(tmp_path / "forecasts", tmp_path / "settlement",
                             tmp_path / "history", panel, threshold=2.5,
                             halflife=20.0, horizon=HORIZON)
    assert summary["settled"] == [forecast_date]
    rows = pd.read_parquet(tmp_path / "history" / "settlement.parquet")
    sber = rows[(rows["asset"] == "MOEX:SBER")].iloc[0]
    assert sber["warned"] is True or sber["warned"] == True  # noqa: E712
    assert bool(sber["event"]) is True                        # предупреждение сбылось
    assert sber["ret_5d"] < 0
    assert (rows["asset"] == PORTFOLIO_ROW).any()             # портфельная строка есть


def test_settle_sigma_uses_only_past(tmp_path):
    """σ для события считается по данным до даты прогноза: обвал в будущем
    окне не должен раздувать σ и «прятать» событие."""
    panel = _panel(crash_day=-3)
    forecast_date = str(panel.index[-7])
    result = settle_forecast(_forecast(forecast_date), panel, threshold=2.0,
                             halflife=20.0, horizon=HORIZON)
    assert result["status"] == "settled"
    sber = next(row for row in result["rows"] if row["asset"] == "MOEX:SBER")
    assert sber["event"] is True


def test_quality_table_after_settlement(tmp_path):
    panel = _panel(crash_day=-3, seed=9)
    for offset in (7, 12):                    # два прогноза в разных днях
        date = str(panel.index[-offset])
        write_forecast(tmp_path / "forecasts", _forecast(date))
    settle_pending(tmp_path / "forecasts", tmp_path / "settlement",
                   tmp_path / "history", panel, threshold=2.0,
                   halflife=20.0, horizon=HORIZON)
    quality = quality_table(tmp_path / "history")
    assert quality["n_rows"] > 0
    assert quality["overall"]["n"] >= 4                # 2 прогноза x 2 актива
    assert 0.0 <= quality["overall"]["brier"] <= 1.0
    assert quality["monthly"]                          # помесячная разбивка есть
    assert quality["best_calibrated"]                  # пер-активная калибровка


def test_settled_marker_prevents_resettlement(tmp_path):
    panel = _panel(crash_day=-3)
    date = str(panel.index[-7])
    write_forecast(tmp_path / "forecasts", _forecast(date))
    first = settle_pending(tmp_path / "forecasts", tmp_path / "settlement",
                           tmp_path / "history", panel, threshold=2.0,
                           halflife=20.0, horizon=HORIZON)
    second = settle_pending(tmp_path / "forecasts", tmp_path / "settlement",
                            tmp_path / "history", panel, threshold=2.0,
                            halflife=20.0, horizon=HORIZON)
    assert first["settled"] == [date] and second["settled"] == []  # повтор — no-op
