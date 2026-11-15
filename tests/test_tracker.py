"""Трекер: крэш поднимает скор; overlay режет потолок этого актива."""
import numpy as np
import pandas as pd
import pytest

from portagent.tracker.rare_events import asset_scores, overlay_caps


def test_crash_day_scores_above_calm_peers():
    rng = np.random.default_rng(0)
    days = 180
    index = pd.bdate_range("2025-01-01", periods=days).strftime("%Y-%m-%d")
    calm = rng.normal(0.0002, 0.01, size=(days, 3))
    calm[-1, 2] = -0.18
    closes = pd.DataFrame(100.0 * np.exp(np.cumsum(calm, axis=0)),
                          index=index, columns=["A", "B", "C"])
    returns = np.log(closes / closes.shift(1)).dropna()
    scores = asset_scores(returns, closes, window=60, halflife=10, threshold=2.5)
    last = scores.iloc[-1]
    assert last["C"] > last["A"]
    assert last["C"] > last["B"]


def test_overlay_caps_shrink_with_score():
    caps = overlay_caps(pd.Series({"A": 0.0, "B": 1.0}), max_weight=0.25, strength=0.5)
    assert caps["A"] == pytest.approx(0.25)
    assert caps["B"] == pytest.approx(0.125)
