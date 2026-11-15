"""Ценовая модальность: панель закрытий, доходности, матрица расстояний."""
from __future__ import annotations

import numpy as np
import pandas as pd

FFILL_LIMIT = 5  # дней: календари торгов MOEX и США не совпадают (праздники)


def build_panel(frames: dict[str, pd.DataFrame], *, min_history: int = 60) -> tuple[pd.DataFrame, list[str]]:
    """Панель закрытий даты x активы. Даты — outer join; гэпы календарей
    закрываем forward-fill (до 5 дней), активы без истории отбрасываем."""
    closes = pd.concat({asset: frame["close"] for asset, frame in frames.items()},
                       axis=1, join="outer").sort_index()
    closes = closes.ffill(limit=FFILL_LIMIT)
    closes = closes.dropna(how="all")
    keep = [c for c in closes.columns if closes[c].notna().sum() >= min_history]
    dropped = sorted(set(closes.columns) - set(keep))
    return closes[keep], dropped


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log(panel / panel.shift(1)).dropna(how="all").iloc[1:]


def distance_from_correlation(series: pd.DataFrame, *, min_periods: int = 60) -> pd.DataFrame:
    """D(i,j) = sqrt(0.5 * (1 - corr)); диагональ 0, значения в [0, 1].

    Неопределённые корреляции (мало данных) остаются NaN — потребитель
    (fusion) решает, чем их заполнить.
    """
    corr = series.corr(min_periods=min_periods)
    values = np.sqrt(0.5 * (1.0 - corr)).clip(lower=0.0, upper=1.0).to_numpy().copy()
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def movers(panel: pd.DataFrame, *, top: int = 5) -> list[dict]:
    """Движения за 1 и 5 торговых дней для рецензии критика."""
    rows: list[dict] = []
    if len(panel) < 6:
        return rows
    last = panel.iloc[-1]
    prev1, prev5 = panel.iloc[-2], panel.iloc[-6]
    for asset in panel.columns:
        if prev1[asset] > 0 and prev5[asset] > 0:
            rows.append({"asset": asset,
                         "ret_1d": float(last[asset] / prev1[asset] - 1),
                         "ret_5d": float(last[asset] / prev5[asset] - 1)})
    rows.sort(key=lambda r: abs(r["ret_5d"]), reverse=True)
    return rows[:top]
