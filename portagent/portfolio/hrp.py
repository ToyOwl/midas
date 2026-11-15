"""Hierarchical Risk Parity (Lopez de Prado): tree clustering -> quasi-
diagonalization -> recursive bisection, на кастомной fused-матрице расстояний.

Веса полностью детерминированы; LLM не участвует. Ограничения: потолок веса
на актив (итеративная раздача излишка свободным активам) и доля CASH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def _linkage_matrix(D: pd.DataFrame, method: str):
    condensed = squareform(D.values, checks=False)
    return linkage(condensed, method=method)


def quasi_diag_order(D: pd.DataFrame, *, method: str = "single") -> list[str]:
    """Порядок листьев дендрограммы — квазидиагонализация ковариационной матрицы."""
    order_indices = leaves_list(_linkage_matrix(D, method))
    return [D.index[i] for i in order_indices]


def cluster_groups(D: pd.DataFrame, *, method: str = "single", max_clusters: int = 3) -> dict[int, list[str]]:
    """Неглубокие кластеры для нарратива отчёта (не влияют на веса)."""
    matrix = _linkage_matrix(D, method)
    labels = fcluster(matrix, t=max_clusters, criterion="maxclust")
    groups: dict[int, list[str]] = {}
    for asset, label in zip(D.index, labels):
        groups.setdefault(int(label), []).append(asset)
    return groups


def _cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    sub = cov.loc[assets, assets].to_numpy()
    try:
        inverse = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(sub)
    ones = np.ones(len(assets))
    return float(1.0 / max(ones @ inverse @ ones, 1e-12))


def recursive_bisection(order: list[str], cov: pd.DataFrame) -> pd.Series:
    """Классический HRP (recBipart): веса перетекают сверху вниз по квазидиагонали.

    Бисекция применяется только к кластерам длины > 1; соседние половины одного
    родителя делят единичный вес обратно пропорционально дисперсии кластера.
    """
    if len(order) == 1:
        return pd.Series(1.0, index=order)
    weights = pd.Series(1.0, index=order)
    clusters: list[list[str]] = [list(order)]
    while clusters:
        halves: list[list[str]] = []
        for cluster in clusters:
            middle = len(cluster) // 2
            halves.extend((cluster[:middle], cluster[middle:]))
        for i in range(0, len(halves) - 1, 2):
            left, right = halves[i], halves[i + 1]  # половины одного родителя
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] = weights[left] * alpha
            weights[right] = weights[right] * (1.0 - alpha)
        clusters = [half for half in halves if len(half) > 1]
    return weights / weights.sum()


def cap_weights(weights: pd.Series, max_weight) -> pd.Series:
    """Потолок веса: скаляр или пер-тикерный вектор (pd.Series/Series-подобный).
    Масса свыше потолка передаётся активам под потолком пропорционально их
    текущим весам; итеративно, пока не стабилизируется."""
    caps = (pd.Series(max_weight, index=weights.index, dtype=float)
            if np.isscalar(max_weight) else pd.Series(max_weight).reindex(weights.index))
    if (caps <= 0).any():
        raise ValueError("потолок веса должен быть положительным для каждого актива")
    if caps.sum() < 1.0 - 1e-9:
        raise ValueError(f"сумма потолков {caps.sum():.3f} < 1 — ограничения неисполнимы")
    capped = weights.astype(float).copy()
    for _ in range(len(weights) + 1):
        over = capped > caps + 1e-12
        if not over.any():
            break
        overflow = float((capped[over] - caps[over]).sum())
        capped[over] = caps[over]
        free = capped[~over]
        free_caps = caps[~over]
        room = (free_caps - free).clip(lower=0.0)  # запас до потолка у свободных
        if room.sum() <= 0:
            break
        capped[~over] = free + overflow * (room / room.sum())
    return capped / capped.sum()


def hrp_weights(returns: pd.DataFrame, D: pd.DataFrame, *, linkage_method: str = "single",
                max_weight=0.25, cash_share: float = 0.0) -> dict:
    """Полный расчёт: порядок -> бисекция -> потолок (скаляр/вектор) -> CASH.

    returns задают ковариацию (масштаб риска), D — структуру кластеров
    (мультимодальное расстояние). Возвращает веса (сумма = 1 - cash_share),
    порядок квазидиагонали и кластеры для отчёта.
    """
    order = quasi_diag_order(D, method=linkage_method)
    cov = returns.cov()
    raw = recursive_bisection(order, cov)
    capped = cap_weights(raw, max_weight)
    invested = capped * (1.0 - cash_share)
    caps = (float(max_weight) if np.isscalar(max_weight)
            else pd.Series(max_weight).reindex(invested.index).round(6).to_dict())
    return {
        "weights": invested,
        "raw_weights": raw,
        "order": order,
        "clusters": cluster_groups(D, method=linkage_method),
        "linkage_method": linkage_method,
        "max_weight": caps,
        "cash_share": cash_share,
    }
