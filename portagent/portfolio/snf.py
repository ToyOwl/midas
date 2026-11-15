"""Similarity Network Fusion (Wang et al., NMI 2014) — нелинейное слияние
модальностей: итеративная кросс-диффузия kNN-ядер похожестей.

Отличие от линейной смеси: локальная структура каждого вида (соседи) усиливает
консенсус, а глобальные различия затухают; несогласованные виды не «размывают»
структуру, а дополняют её. Чистый numpy, без обучаемых параметров.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _full_similarity(D: np.ndarray, eps: float) -> np.ndarray:
    """W = exp(-D^2 / (eps * mean(D))) — полная похожесть вида."""
    n = D.shape[0]
    if n < 2:
        return np.ones((n, n))
    off = D[~np.eye(n, dtype=bool)]
    scale = eps * float(off.mean()) + 1e-12
    return np.exp(-(D.astype(float) ** 2) / scale)


def _knn_kernel(W: np.ndarray, D: np.ndarray, k: int) -> np.ndarray:
    """Разреженное ядро: топ-k соседей по D (включая себя), построчная нормировка.

    k адаптивно ограничивается ~60% узлов: плотное ядро (k ~ n) вырождает
    кросс-диффузию в почти полный граф и растворяет структуру.
    """
    n = D.shape[0]
    k = min(k, max(3, int(round(0.6 * (n - 1)))))
    K = np.zeros_like(W)
    for i in range(n):
        neighbours = np.argsort(D[i], kind="stable")[: k + 1]  # k соседей + сам
        K[i, neighbours] = W[i, neighbours]
    row_sums = K.sum(axis=1, keepdims=True)
    return K / np.where(row_sums > 0, row_sums, 1.0)


def _diag_normalise(S: np.ndarray) -> np.ndarray:
    """Симметричная диагональная нормировка (косинусная): диагональ -> 1,
    значения остаются в [0, 1] для неотрицательных матриц."""
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return np.clip(S / np.outer(d, d), 0.0, 1.0)


def snf(views: dict[str, pd.DataFrame], *, k: int = 15, t: int = 20,
        eps: float = 0.5, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Слияние видов расстояний -> fused-расстояние на общем наборе активов.

    weights (0..1) взвешивают вклад видов в кросс-диффузию и финальное
    усреднение — весом служит качество вида (см. features.view_quality).
    При равных весах (или None) поведение совпадает с классическим SNF.
    Возвращает D_fused = sqrt(clip(1 - S_final, 0, 1)), симметричное, диагональ 0.
    """
    if len(views) == 0:
        raise ValueError("snf: нужен хотя бы один вид")
    names = list(views)
    first = views[names[0]]
    assets = list(first.index)
    matrices = [views[name].loc[assets, assets].to_numpy(dtype=float) for name in names]
    if len(matrices) == 1:
        return first

    raw = [(weights or {}).get(name, 1.0) for name in names]
    # виды с почти нулевым весом исключаем совсем: влияние шумового вида
    # обязано быть нулевым, а не «маленьким» (найдено историческим тестом)
    keep_mask = np.asarray([value >= 0.02 for value in raw])
    if keep_mask.sum() <= 1:
        strongest = int(np.argmax(keep_mask)) if keep_mask.any() else 0
        return views[names[strongest]]
    names = [name for name, keep in zip(names, keep_mask) if keep]
    matrices = [matrix for matrix, keep in zip(matrices, keep_mask) if keep]
    w = np.clip(np.asarray([raw[i] for i, keep in enumerate(keep_mask) if keep],
                           dtype=float), 0.02, 1.0)

    Ws = [_full_similarity(D, eps) for D in matrices]
    Ks = [_knn_kernel(W, D, k) for W, D in zip(Ws, matrices)]
    Ss = [W.copy() for W in Ws]

    # Взвешенный консенсус вместо классического leave-one-out: при двух видах
    # leave-one-out делает cross для базового вида равным второму виду ЦЕЛИКОМ
    # (шум отравляет даже с малым весом). Здесь каждый вид диффундирует к
    # взвешенному среднему всех видов — вклад вида пропорционален его весу.
    for _ in range(t):
        consensus = sum(w[i] * Ss[i] for i in range(len(Ss))) / w.sum()
        Ss = [_diag_normalise(Ks[i] @ consensus @ Ks[i].T) for i in range(len(Ss))]
    final = _diag_normalise(sum(w[i] * Ss[i] for i in range(len(Ss))) / w.sum())
    final = (final + final.T) / 2.0
    np.fill_diagonal(final, 1.0)
    distance = np.sqrt(np.clip(1.0 - final, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=assets, columns=assets)
