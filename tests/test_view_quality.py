"""Гейт качества видов: consistency разделяет сигнал и шум; взвешенный SNF."""
import numpy as np
import pandas as pd
import pytest

from portagent.features.view_quality import consistency, gate_quality
from portagent.portfolio.fuse import fuse_distances
from portagent.portfolio.snf import snf

NAMES = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _structured_series(days: int = 240, seed: int = 5) -> pd.DataFrame:
    """Сентимент-подобные ряды с устойчивой кластерной структурой."""
    rng = np.random.default_rng(seed)
    factors = [rng.normal(0, 1, days) for _ in range(3)]
    columns = {}
    for i, name in enumerate(NAMES):
        columns[name] = factors[i % 3] + rng.normal(0, 0.4, days)
    return pd.DataFrame(columns, index=range(days))


def _noise_series(days: int = 240, seed: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (days, len(NAMES))),
                        columns=NAMES, index=range(days))


def test_consistency_separates_structure_from_noise():
    structured = consistency(_structured_series())
    noise = consistency(_noise_series())
    assert structured > 0.7, f"структурированный вид: {structured}"
    assert noise < 0.3, f"шумовой вид: {noise}"


def test_consistency_short_history_is_nan():
    assert np.isnan(consistency(_noise_series(days=20)))
    assert np.isnan(consistency(None))


def test_gate_quality_thresholding():
    assert gate_quality(0.71, threshold=0.3) == pytest.approx(0.71)
    assert gate_quality(0.07, threshold=0.3) == 0.0
    assert gate_quality(float("nan"), threshold=0.3) == 0.0
    assert gate_quality(1.7, threshold=0.3) == 1.0


def _distance_matrix(fill: float) -> pd.DataFrame:
    values = np.full((len(NAMES), len(NAMES)), fill, dtype=float)
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=NAMES, columns=NAMES)


def test_fuse_snf_gate_excludes_low_quality_view():
    D_price, D_sent = _distance_matrix(0.2), _distance_matrix(0.6)
    rejected = fuse_distances({"price": D_price, "sentiment": D_sent},
                              method="snf", coverage={"sentiment": 1.0},
                              view_quality={"sentiment": 0.07},
                              snf_min_quality=0.3)
    assert rejected["mode"] == "single"
    assert "отклонены" in rejected["note"]
    admitted = fuse_distances({"price": D_price, "sentiment": D_sent},
                              method="snf", coverage={"sentiment": 1.0},
                              view_quality={"sentiment": 0.71},
                              snf_min_quality=0.3)
    assert admitted["mode"] == "snf"
    assert admitted["views_used"] == ["price", "sentiment"]
    assert "0.71" in admitted["note"]


def test_fuse_linear_ignores_quality_gate():
    D_price, D_sent = _distance_matrix(0.2), _distance_matrix(0.6)
    result = fuse_distances({"price": D_price, "sentiment": D_sent},
                            method="linear", coverage={"sentiment": 1.0},
                            view_quality={"sentiment": 0.07},  # шум — linear робастен
                            w_price=0.5)
    assert result["mode"] == "linear"


def test_snf_equal_weights_match_classical():
    """Веса None/равные -> одно и то же слияние."""
    rng = np.random.default_rng(2)
    views = {}
    for name in ("v1", "v2"):
        values = rng.uniform(0.1, 0.9, (6, 6))
        values = (values + values.T) / 2
        np.fill_diagonal(values, 0.0)
        views[name] = pd.DataFrame(values, index=NAMES, columns=NAMES)
    classic = snf(views, k=3, t=5)
    weighted = snf(views, k=3, t=5, weights={"v1": 1.0, "v2": 1.0})
    pd.testing.assert_frame_equal(classic, weighted)


def test_snf_weight_zero_view_dropped():
    """Нулевой вес -> вид исключён целиком: результат идентичен одиночному."""
    rng = np.random.default_rng(3)
    strong = rng.uniform(0.1, 0.9, (6, 6))
    strong = (strong + strong.T) / 2
    np.fill_diagonal(strong, 0.0)
    view1 = pd.DataFrame(strong, index=NAMES, columns=NAMES)
    junk = pd.DataFrame(rng.uniform(0.1, 0.9, (6, 6)), index=NAMES, columns=NAMES)
    single = snf({"v1": view1}, k=3, t=5)
    with_junk = snf({"v1": view1, "junk": junk}, k=3, t=5,
                    weights={"v1": 1.0, "junk": 0.0})
    pd.testing.assert_frame_equal(single, with_junk)


def test_snf_low_weight_view_bounded_influence():
    """Малый (но ненулевой) вес даёт ограниченное влияние, не отравление."""
    rng = np.random.default_rng(4)
    strong = rng.uniform(0.05, 0.3, (6, 6))
    strong = (strong + strong.T) / 2
    np.fill_diagonal(strong, 0.0)
    view1 = pd.DataFrame(strong, index=NAMES, columns=NAMES)
    junk = pd.DataFrame(rng.uniform(0.1, 0.9, (6, 6)), index=NAMES, columns=NAMES)
    single = snf({"v1": view1}, k=3, t=5)
    with_junk = snf({"v1": view1, "junk": junk}, k=3, t=5,
                    weights={"v1": 1.0, "junk": 0.2})
    max_shift = float(np.max(np.abs(single.to_numpy() - with_junk.to_numpy())))
    assert max_shift < 0.35, f"влияние вида с весом 0.2 слишком велико: {max_shift}"
