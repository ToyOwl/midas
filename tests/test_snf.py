"""SNF: слияние kNN-ядер — корректность и превосходство над одиночным видом.

Синтетика: 6 активов, истинная структура {A1,A2}, {B1,B2}, {C1,C2}.
Вид-1 разделяет только кластеры A|B (внутри — шум), вид-2 — только B|C.
SNF должен восстановить все три кластера лучше любого одиночного вида.
"""
import numpy as np
import pandas as pd
import pytest

from portagent.portfolio.snf import snf

NAMES = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _distance(block_matrix):
    return pd.DataFrame(block_matrix, index=NAMES, columns=NAMES)


def _two_views():
    rng = np.random.default_rng(11)
    noise = lambda: rng.uniform(0.55, 0.95, (6, 6))

    view1 = noise()
    for i in range(6):
        view1[i, i] = 0.0
    # вид 1: A-пары и B-пары близки (C — шум)
    for a, b in [("A1", "A2"), ("B1", "B2")]:
        i, j = NAMES.index(a), NAMES.index(b)
        view1[i, j] = view1[j, i] = 0.05

    view2 = noise()
    for i in range(6):
        view2[i, i] = 0.0
    # вид 2: B-пары и C-пары близки (A — шум)
    for a, b in [("B1", "B2"), ("C1", "C2")]:
        i, j = NAMES.index(a), NAMES.index(b)
        view2[i, j] = view2[j, i] = 0.05

    for view in (view1, view2):
        view = (view + view.T) / 2
    return _distance((view1 + view1.T) / 2), _distance((view2 + view2.T) / 2)


TRUE_PAIRS = [("A1", "A2"), ("B1", "B2"), ("C1", "C2")]


def _mean_true_pair_distance(D):
    return float(np.mean([D.loc[a, b] for a, b in TRUE_PAIRS]))


def test_snf_fuses_complementary_views():
    view1, view2 = _two_views()
    fused = snf({"v1": view1, "v2": view2}, k=2, t=10)
    m_fused = _mean_true_pair_distance(fused)
    assert m_fused < min(_mean_true_pair_distance(view1), _mean_true_pair_distance(view2))
    assert (np.diag(fused.to_numpy()) == 0).all()
    np.testing.assert_allclose(fused.to_numpy(), fused.to_numpy().T, atol=1e-12)
    assert ((fused.to_numpy() >= 0) & (fused.to_numpy() <= 1)).all()


def test_snf_single_view_returns_unchanged():
    view1, _ = _two_views()
    assert snf({"only": view1}) is view1


def test_snf_more_views_stays_stable_with_noise_view():
    view1, view2 = _two_views()
    noise = pd.DataFrame(np.abs(np.random.default_rng(2).normal(0.5, 0.2, (6, 6))),
                         index=NAMES, columns=NAMES)
    for i in range(6):
        noise.iloc[i, i] = 0.0
    noise = (noise + noise.T) / 2
    fused = snf({"v1": view1, "v2": view2, "v3": noise}, k=2, t=10)
    # шумовой вид не разрушает структуру двух информативных видов
    assert _mean_true_pair_distance(fused) <= _mean_true_pair_distance(view1) + 0.05


def test_snf_requires_view():
    with pytest.raises(ValueError):
        snf({})
