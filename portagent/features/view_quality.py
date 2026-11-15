"""Качество вида модальности: устойчивость корреляционной структуры во времени.

consistency(series) = Spearman-корреляция векторов расстояний, посчитанных
на двух половинах истории ряда. Осмысленный вид устойчив (структура
воспроизводится между половинами), шумовой — нет.

Калибровка на реальной панели 2019-2026 (20 активов): oracle-сентимент 0.71,
шумовой сентимент 0.07, цены 0.49 — порог по умолчанию 0.30 лежит в зазоре.
Используется как гейт (вид ниже порога не входит в SNF) и как вес вида
в кросс-диффузии. Детерминировано, без подглядывания (использует только
прошлое ряда).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .prices import distance_from_correlation


def consistency(series: pd.DataFrame | None, *, min_periods: int = 30) -> float:
    """Устойчивость структуры вида: Spearman(D_первая_половина, D_вторая).

    NaN при слишком короткой истории (гейт трактует как «не пускать»).
    """
    if series is None or series.empty or series.shape[1] < 3 or len(series) < 60:
        return float("nan")
    middle = len(series) // 2
    periods = max(10, min(min_periods, middle // 2))
    halves = (series.iloc[:middle], series.iloc[middle:])
    distances = [distance_from_correlation(half, min_periods=periods)
                 for half in halves]
    assets = [c for c in distances[0].columns if c in distances[1].columns]
    pairs = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a = distances[0].loc[assets[i], assets[j]]
            b = distances[1].loc[assets[i], assets[j]]
            if np.isfinite(a) and np.isfinite(b):
                pairs.append((float(a), float(b)))
    if len(pairs) < 10:
        return float("nan")
    left, right = zip(*pairs)
    return float(spearmanr(left, right).statistic)


def gate_quality(value: float, threshold: float) -> float:
    """Вес вида для SNF: качество ниже порога -> 0 (не участвует),
    выше -> само качество; NaN -> 0 (короткая история — не пускать)."""
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0)) if value >= threshold else 0.0
