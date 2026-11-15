"""Калибровка метрики качества вида: gap-статистика квантилей расстояний."""
import numpy as np
import pandas as pd

from portagent.backtest import _items_by_id, _labels_until
from portagent.features import prices as pf
from portagent.features import sentiment as sf
from portagent.history import load_price_panel
from portagent.modality_test import build_labels_noise, build_labels_oracle


def gap_quality(D: pd.DataFrame) -> float:
    """Крупнейший разрыв между соседними квантилями (5..95%), нормированный на
    межквантильный размах. Бимодальный (структурный) вид -> 0.4..1.0;
    равномерный шум -> ~0.06; вырожденный (все расстояния равны) -> 0."""
    values = D.to_numpy(dtype=float).copy()
    n = values.shape[0]
    if n < 4:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    flat = values[mask]
    flat = flat[np.isfinite(flat)]
    if flat.size < 6:
        return 0.0
    quantiles = np.quantile(flat, np.linspace(0.05, 0.95, 19))
    spread = quantiles[-1] - quantiles[0]
    if spread < 1e-9:
        return 0.0
    return float(np.clip(np.max(np.diff(quantiles)) / spread, 0.0, 1.0))


panel = load_price_panel("data/history")
returns = pf.log_returns(panel)
D_price = pf.distance_from_correlation(returns, min_periods=40)
print(f"панель {panel.index[0]}..{panel.index[-1]}, {panel.shape[1]} активов")
print(f"price: quality={gap_quality(D_price):.2f}")

day = str(panel.index[-60])
oracle = build_labels_oracle(panel, horizon=5)
noise = build_labels_noise(panel, oracle)
for name, labels in (("oracle", oracle), ("noise", noise)):
    until = _labels_until(labels, day)
    position = panel.index.get_loc(day)
    sent = sf.daily_series(until, _items_by_id(until),
                           price_index=panel.index[: position + 1],
                           halflife_days=5.0, fallback_date=day)
    D_sent = pf.distance_from_correlation(sent, min_periods=10)
    covered = [c for c in D_sent.columns if D_sent[c].dropna().size > 5]
    if not covered:
        print(f"{name}: вид пуст")
        continue
    Dc = D_sent.loc[covered, covered]
    print(f"{name}: {len(covered)} активов, quality={gap_quality(Dc):.2f}")

rng = np.random.default_rng(7)
n = panel.shape[1]
rand = rng.uniform(0.05, 0.95, (n, n))
rand = (rand + rand.T) / 2
np.fill_diagonal(rand, 0.0)
print(f"uniform-random: quality={gap_quality(pd.DataFrame(rand)):.2f}")


def consistency(series: pd.DataFrame, min_periods: int = 30) -> float:
    """Spearman-корреляция расстояний, посчитанных на двух половинах истории
    ряда. Осмысленный вид устойчив во времени; шум — нет."""
    from scipy.stats import spearmanr

    middle = len(series) // 2
    halves = (series.iloc[:middle], series.iloc[middle:])
    distances = [pf.distance_from_correlation(half, min_periods=min_periods)
                 for half in halves]
    assets = [c for c in distances[0].columns if c in distances[1].columns]
    pairs = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = distances[0].loc[assets[i], assets[j]], distances[1].loc[assets[i], assets[j]]
            if np.isfinite(a) and np.isfinite(b):
                pairs.append((a, b))
    if len(pairs) < 10:
        return float("nan")
    left, right = zip(*pairs)
    return float(spearmanr(left, right).statistic)


print()
for name, labels in (("oracle", oracle), ("noise", noise)):
    until = _labels_until(labels, day)
    position = panel.index.get_loc(day)
    sent = sf.daily_series(until, _items_by_id(until),
                           price_index=panel.index[: position + 1],
                           halflife_days=5.0, fallback_date=day)
    print(f"{name}: consistency={consistency(sent):.2f}")
print(f"price: consistency={consistency(returns):.2f}")
# ряд без структуры: переставленные колонки oracle во времени
shuffled_series = sent.copy()
for column in shuffled_series.columns:
    shuffled_series[column] = shuffled_series[column].sample(frac=1.0, random_state=3).to_numpy()
print(f"oracle-shuffled-in-time: consistency={consistency(shuffled_series):.2f}")
