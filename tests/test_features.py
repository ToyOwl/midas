"""Ценовая и сентимент-модальности: формулы расстояний, EWMA, покрытие."""
import numpy as np
import pandas as pd
import pytest

from portagent.features import prices, sentiment
from conftest import CLUSTERS, synthetic_frames


def test_build_panel_drops_thin_assets():
    frames = synthetic_frames()
    thin = synthetic_frames(days=250)
    thin["MOEX:SBER"] = thin["MOEX:SBER"].iloc[:10]  # почти пустая история
    panel, dropped = prices.build_panel({**frames, "MOEX:SBER": thin["MOEX:SBER"]})
    assert "MOEX:SBER" in dropped
    assert panel.shape[1] == 11


def test_distance_formula_and_diagonal():
    rng = np.random.default_rng(1)
    series = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    D = prices.distance_from_correlation(series)
    assert (np.diag(D.to_numpy()) == 0).all()
    assert D.shape == (3, 3)
    corr = np.corrcoef(series.T)
    expected = np.sqrt(0.5 * (1 - corr))
    np.testing.assert_allclose(D.to_numpy(), expected, atol=0.05)
    assert ((D.to_numpy() >= 0) & (D.to_numpy() <= 1)).all()


def test_intra_cluster_distance_smaller():
    panel, _ = prices.build_panel(synthetic_frames())
    D = prices.distance_from_correlation(prices.log_returns(panel))
    flat = D.to_numpy()[np.triu_indices_from(D.to_numpy(), k=1)]
    intra, inter = [], []
    names = list(D.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = any(names[i] in group and names[j] in group for group in CLUSTERS.values())
            (intra if same else inter).append(D.iloc[i, j])
    assert np.mean(intra) < np.mean(inter)  # кластеры различимы в ценах


def test_sentiment_series_alignment_and_decay():
    price_index = pd.date_range("2026-11-01", periods=10, freq="D").strftime("%Y-%m-%d")
    items_by_id = {"n1": {"published_at": "2026-11-02T09:00:00+00:00"}}
    labels = [{"news_id": "n1", "ticker": "MOEX:SBER", "sentiment": 1.0, "confidence": 1.0}]
    series = sentiment.daily_series(labels, items_by_id, price_index=price_index,
                                    halflife_days=2, fallback_date="2026-11-10")
    assert list(series.index) == list(price_index)
    column = series["MOEX:SBER"]
    assert column.iloc[0] == 0.0                                   # до новости — нейтрально
    assert column.iloc[1] == pytest.approx(0.5858, abs=0.01)       # EWMA-отклик (halflife=2, adjust)
    assert column.iloc[2] < column.iloc[1]                         # затухает к нейтрали
    assert column.iloc[9] < 0.05


def test_coverage_fraction():
    labels = [{"ticker": "A", "sentiment": 0.1}] * 3 + [{"ticker": "B", "sentiment": 0.1}]
    result = sentiment.coverage(labels, ["A", "B", "C"], min_news=3)
    assert result["covered"] == ["A"]
    assert result["coverage_frac"] == pytest.approx(1 / 3)


def test_daily_series_empty_labels():
    price_index = pd.Index(["2026-11-01", "2026-11-02"])
    series = sentiment.daily_series([], {}, price_index=price_index, halflife_days=5,
                                    fallback_date="2026-11-02")
    assert series.empty or (series == 0).all().all()
