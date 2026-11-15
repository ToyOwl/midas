"""HRP: корректность весов, потолки, кластерный баланс; fusion: режимы."""
import numpy as np
import pandas as pd
import pytest

from portagent.features import prices
from portagent.portfolio.fuse import fuse_distances
from portagent.portfolio.hrp import cap_weights, hrp_weights, recursive_bisection
from conftest import CLUSTERS, synthetic_frames


def _panel_returns():
    panel, _ = prices.build_panel(synthetic_frames())
    return prices.log_returns(panel)


def test_cap_weights_redistributes_overflow():
    weights = pd.Series({"a": 0.55, "b": 0.25, "c": 0.10, "d": 0.10})
    capped = cap_weights(weights, 0.40)
    assert capped.sum() == pytest.approx(1.0)
    assert (capped <= 0.40 + 1e-9).all()
    assert capped["a"] == pytest.approx(0.40)
    # излишок 0.15 раздаётся пропорционально запасу до потолка (b:0.15, c:0.30, d:0.30)
    assert capped["b"] == pytest.approx(0.25 + 0.15 * 0.15 / 0.75, abs=1e-6)
    assert capped["c"] == pytest.approx(0.10 + 0.15 * 0.30 / 0.75, abs=1e-6)


def test_cap_weights_noop_under_cap():
    weights = pd.Series({"a": 0.3, "b": 0.3, "c": 0.4})
    assert cap_weights(weights, 0.5).tolist() == weights.tolist()


def test_hrp_weights_sum_cash_and_cap():
    returns = _panel_returns()
    panel, _ = prices.build_panel(synthetic_frames())
    D = prices.distance_from_correlation(returns)
    result = hrp_weights(returns, D, max_weight=0.20, cash_share=0.10)
    weights = result["weights"]
    assert weights.sum() == pytest.approx(0.90, abs=1e-9)
    assert (weights <= 0.20 * 0.90 + 1e-9).all()
    assert set(weights.index) == set(returns.columns)


def test_hrp_balances_clusters():
    """Кластеры одного размера получают сопоставимые суммарные веса."""
    returns = _panel_returns()
    D = prices.distance_from_correlation(returns)
    result = hrp_weights(returns, D, max_weight=0.5, cash_share=0.0)
    cluster_sums = {
        name: sum(result["weights"][asset] for asset in assets)
        for name, assets in CLUSTERS.items()
    }
    for name, total in cluster_sums.items():
        assert total == pytest.approx(1 / 3, abs=0.15), f"кластер {name}: {total}"


def test_recursive_bisection_positive_normalized():
    returns = _panel_returns()
    cov = returns.cov()
    from portagent.portfolio.hrp import quasi_diag_order

    order = quasi_diag_order(prices.distance_from_correlation(returns))
    weights = recursive_bisection(order, cov)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_fuse_single_view_below_threshold():
    D_price = pd.DataFrame(0.1, index=list("abc"), columns=list("abc"))
    D_sent = pd.DataFrame(0.5, index=list("abc"), columns=list("abc"))
    result = fuse_distances({"price": D_price, "sentiment": D_sent},
                            coverage={"sentiment": 0.3}, min_coverage=0.6)
    assert result["mode"] == "single"
    assert result["D"] is D_price


def test_fuse_linear_blends_when_covered():
    D_price = pd.DataFrame(0.1, index=list("abc"), columns=list("abc"))
    D_sent = pd.DataFrame(0.5, index=list("abc"), columns=list("abc"))
    result = fuse_distances({"price": D_price, "sentiment": D_sent},
                            method="linear", coverage={"sentiment": 0.8},
                            w_price=0.7, min_coverage=0.6)
    assert result["mode"] == "linear"
    assert result["views_used"] == ["price", "sentiment"]
    off_diag = result["D"].to_numpy()[np.triu_indices(3, k=1)]
    assert off_diag[0] == pytest.approx(0.7 * 0.1 + 0.3 * 0.5)


def test_fuse_snf_mode():
    D_price = pd.DataFrame(0.1, index=list("abcd"), columns=list("abcd"))
    D_sent = pd.DataFrame(0.5, index=list("abcd"), columns=list("abcd"))
    result = fuse_distances({"price": D_price, "sentiment": D_sent},
                            method="snf", coverage={"sentiment": 0.8},
                            view_quality={"sentiment": 0.7}, min_coverage=0.6)
    assert result["mode"] == "snf"
    D = result["D"]
    assert (np.diag(D.to_numpy()) == 0).all()
    assert ((D.to_numpy() >= 0) & (D.to_numpy() <= 1)).all()


def test_fuse_pair_fallback_to_price_linear():
    D_price = pd.DataFrame([[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]],
                           index=list("abc"), columns=list("abc"))
    D_sent = pd.DataFrame([[0.0, np.nan, 0.8], [np.nan, 0.0, np.nan],
                           [0.8, np.nan, 0.0]], index=list("abc"), columns=list("abc"))
    result = fuse_distances({"price": D_price, "sentiment": D_sent},
                            method="linear", coverage={"sentiment": 0.9},
                            w_price=0.5, min_coverage=0.6)
    D = result["D"]
    assert D.loc["a", "b"] == pytest.approx(0.2)   # NaN в сентименте -> цена
    assert D.loc["a", "c"] == pytest.approx(0.5 * 0.3 + 0.5 * 0.8)


def test_fuse_without_aux_view():
    D_price = pd.DataFrame(0.1, index=list("ab"), columns=list("ab"))
    result = fuse_distances({"price": D_price, "sentiment": None},
                            coverage={"sentiment": 1.0}, min_coverage=0.6)
    assert result["mode"] == "single"


def test_cap_weights_vector_per_asset():
    weights = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    caps = pd.Series({"a": 0.30, "b": 0.40, "c": 0.45})
    capped = cap_weights(weights, caps)
    assert capped.sum() == pytest.approx(1.0)
    assert capped["a"] == pytest.approx(0.30)
    assert (capped <= caps + 1e-9).all()
    assert capped["b"] > capped["a"] and capped["c"] > capped["a"]  # излишок получил свободные


def test_cap_weights_vector_infeasible():
    weights = pd.Series({"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="неисполнимы"):
        cap_weights(weights, pd.Series({"a": 0.2, "b": 0.3}))
