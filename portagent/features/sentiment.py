"""Сентимент-модальность: метки LLM -> пер-тикерные дневные ряды (EWMA).

Ряд строится на календаре цен: день без новостей даёт 0 (сентимент затухает
к нейтрали с периодом полураспада halflife_days), день с новостями —
средневзвешенная по confidence оценка.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _news_dates(labels: list[dict], items_by_id: dict[str, dict], fallback_date: str) -> list[str]:
    for label in labels:
        item = items_by_id.get(label["news_id"], {})
        published = item.get("published_at")
        label["date"] = (str(published)[:10] if published else fallback_date)
    return [label["date"] for label in labels]


def daily_series(labels: list[dict], items_by_id: dict[str, dict], *,
                 price_index: pd.Index, halflife_days: float,
                 fallback_date: str) -> pd.DataFrame:
    """Дневные сентимент-ряды на календаре цен (даты x активы)."""
    assets = sorted({label["ticker"] for label in labels})
    if not labels or len(price_index) == 0:
        return pd.DataFrame(0.0, index=price_index, columns=assets or [])
    _news_dates(labels, items_by_id, fallback_date)
    frame = pd.DataFrame(labels)
    weighted = frame.assign(score=frame["sentiment"] * frame["confidence"],
                            weight=frame["confidence"])
    grouped = weighted.groupby(["date", "ticker"])[["score", "weight"]].sum()
    daily_mean = (grouped["score"] / grouped["weight"].replace(0.0, np.nan)).fillna(0.0)
    pivot = daily_mean.unstack("ticker")
    union = price_index.union(pivot.index).sort_values()
    series = pivot.reindex(union).fillna(0.0)
    smoothed = series.ewm(halflife=halflife_days, min_periods=1).mean()
    return smoothed.reindex(price_index).fillna(0.0)


def coverage(labels: list[dict], assets: list[str], *, min_news: int) -> dict:
    """Доля активов юниверса с достаточным новостным покрытием."""
    counts = {ticker: 0 for ticker in assets}
    for label in labels:
        if label["ticker"] in counts:
            counts[label["ticker"]] += 1
    covered = [t for t, n in counts.items() if n >= min_news]
    frac = (len(covered) / len(assets)) if assets else 0.0
    return {"covered": covered, "counts": counts, "coverage_frac": frac,
            "min_news": min_news}
