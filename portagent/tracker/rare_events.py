"""Трекер редких событий по хвостам распределения и режимным признакам.

Редкое событие актива = дневная доходность <= -threshold * sigma (по умолчанию
-2.5 сигмы EWMA-волатильности). Трекер считает риск-скор каждого актива на
каждую дату по принципу «данные до t -> событие в t+1» (без подглядывания):

- p_tail: POT-оценка хвоста (Хилл/Парето по превышениям порога -2*sigma,
  fallback — историческая частота при малом числе превышений);
- vol_of_vol, drawdown — режимные признаки цен;
- sentiment z и legal-флаг — мультимодальные признаки (отрицательный новостной
  фон и судебное давление повышают скор).

score = logistic(bias + сумма весов * кросс-секциональных z-признаков) —
детерминированная эвристика; калибровку (hit-rate/Brier) показывает бэктест.
Overlay (opt-in): пер-тикерный потолок веса max_weight*(1-strength*score).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"p_tail": 1.0, "vol_of_vol": 0.5, "drawdown": 0.5,
                   "sentiment": 0.4, "legal": 0.8, "macro": 0.3,
                   "earnings": 0.5, "bias": -1.6}


def ewma_vol(returns: pd.DataFrame, *, halflife: float) -> pd.DataFrame:
    return returns.ewm(halflife=halflife, min_periods=20).std()


def tail_probability(asset_returns: pd.Series, *, sigma: pd.Series,
                     threshold: float, window: int,
                     u_multiplier: float = 2.0) -> pd.Series:
    """P(r_next <= -threshold*sigma_t) по окну `window` дней до t включительно.

    POT с хвостом Хилла/Парето: p = (n_u/n) * (u/x)^alpha, alpha из логарифмов
    превышений; при n_u < 5 — историческая частота событий (x=u -> p = n_u/n).
    """
    result = pd.Series(np.nan, index=asset_returns.index, dtype=float)
    values = asset_returns.to_numpy(dtype=float)
    sigma_values = sigma.to_numpy(dtype=float)
    for i in range(len(values)):
        if i + 1 < 20 or np.isnan(sigma_values[i]):
            continue
        start = max(0, i + 1 - window)
        history = values[start:i + 1]
        history = history[~np.isnan(history)]
        if len(history) < 20:
            continue
        u = u_multiplier * sigma_values[i]
        x = threshold * sigma_values[i]          # порог редкого события
        losses = -history[history < -u]          # превышения порога u
        n_u = len(losses)
        if n_u == 0:
            result.iloc[i] = 0.0
            continue
        freq = n_u / len(history)
        if x <= u:
            result.iloc[i] = min(freq, 1.0)      # событие не реже порога u
            continue
        if n_u >= 5:
            logs = np.log(losses / u)
            alpha = 1.0 / max(np.mean(logs), 1e-6)   # оценка Хилла
            p = freq * (u / x) ** alpha
        else:
            p = freq * (u / x)                   # консервативный Парето alpha=1
        result.iloc[i] = float(min(p, 1.0))
    return result


def vol_of_vol(asset_returns: pd.Series, *, sigma: pd.Series, window: int) -> pd.Series:
    """Волатильность волатильности: std(sigma) по окну."""
    return sigma.rolling(window, min_periods=20).std()


def drawdown_depth(closes: pd.Series, *, window: int) -> pd.Series:
    """Глубина текущей просадки внутри окна: 1 - close/max(close_window)."""
    rolling_max = closes.rolling(window, min_periods=20).max()
    return (1.0 - closes / rolling_max).clip(lower=0.0)


def _cross_sectional_z(frame: pd.DataFrame) -> pd.DataFrame:
    """Кросс-секциональные z-скор по строкам (датам); медиана/MAD устойчивы."""
    median = frame.median(axis=1)
    mad = (frame.sub(median, axis=0)).abs().median(axis=1)
    scaled = frame.sub(median, axis=0).div(mad.replace(0, np.nan) * 1.4826, axis=0)
    return scaled.fillna(0.0).clip(-3.0, 3.0)


def _sentiment_z(sentiment: pd.DataFrame | None, returns: pd.DataFrame) -> pd.DataFrame:
    """z последнего 5-дневного сентимента против 60-дневного окна; без новостей 0."""
    if sentiment is None or sentiment.empty:
        return pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    aligned = sentiment.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    short = aligned.rolling(5, min_periods=1).mean()
    long = aligned.rolling(60, min_periods=10).mean()
    spread = (short - long)
    std = spread.rolling(60, min_periods=10).std()
    z = (spread / std.replace(0, np.nan)).clip(-3, 3)
    return z.fillna(0.0)


def _legal_flag(legal: pd.DataFrame | None, returns: pd.DataFrame) -> pd.DataFrame:
    """1, если судебное давление (EWMA severity) отрицательно на дату."""
    if legal is None or legal.empty:
        return pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    aligned = legal.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    return (aligned < -0.1).astype(float)


def _macro_stress(macro: pd.DataFrame | None, returns: pd.DataFrame,
                  *, window: int) -> pd.DataFrame:
    """B6: стресс = mean_m(|z_m| * |beta_{a,m}|).

    z_m — rolling-z 5-дневного log-изменения макро-серии (окно window);
    beta — rolling-бета доходности актива к дневному изменению серии.
    Кросс-секционно осмысленно: в макро-турбулентности выше скор у активов
    с большой экспозицией к двигателю.
    """
    zero = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    if macro is None or macro.empty:
        return zero
    stress_terms: list[pd.DataFrame] = []
    for series_name in macro.columns:
        series = macro[series_name].dropna()
        if len(series) < 60:
            continue
        series = series.reindex(returns.index).ffill()
        change = np.log(series / series.shift(1))
        fwd5 = change.rolling(5).sum().shift(1)  # прошедшее 5-дневное движение
        mean, std = fwd5.rolling(window, min_periods=60).mean(), fwd5.rolling(window, min_periods=60).std()
        z = ((fwd5 - mean) / std.replace(0.0, np.nan)).clip(-4, 4)
        aligned_change = change.reindex(returns.index)
        var = aligned_change.rolling(window, min_periods=60).var()
        betas = {}
        for asset in returns.columns:
            cov = returns[asset].rolling(window, min_periods=60).cov(aligned_change)
            betas[asset] = (cov / var).abs()
        beta_frame = pd.DataFrame(betas)
        z_series = z.abs().reindex(returns.index).fillna(0.0)
        stress_terms.append(beta_frame.fillna(0.0).mul(z_series, axis=0))
    if not stress_terms:
        return zero
    return (sum(stress_terms) / len(stress_terms)).clip(0, 4)


def _earnings_flag(earnings_dates: list[tuple[str, str]] | None,
                   returns: pd.DataFrame, *, horizon_days: int = 7) -> pd.DataFrame:
    """B7: 1, если у актива отчётность в ближайшие календарные `horizon_days`
    дней после t (≈5 торговых). Календарное окно — чтобы будущие даты
    отчётностей работали и тогда, когда их нет в ценовом индексе.
    """
    from datetime import date as date_type
    from datetime import timedelta

    flag = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    if not earnings_dates:
        return flag
    by_asset: dict[str, list[date_type]] = {}
    for when, asset in earnings_dates:
        try:
            by_asset.setdefault(asset, []).append(date_type.fromisoformat(when[:10]))
        except ValueError:
            continue
    for asset, earnings_days in by_asset.items():
        if asset not in flag.columns:
            continue
        flag_dates = {(day - timedelta(days=back)).isoformat()
                      for day in earnings_days for back in range(1, horizon_days + 1)}
        for row_date in flag.index:
            if str(row_date)[:10] in flag_dates:
                flag.loc[row_date, asset] = 1.0
    return flag


def _features(returns: pd.DataFrame, closes: pd.DataFrame, *,
              halflife: float, window: int, threshold: float,
              sentiment: pd.DataFrame | None, legal: pd.DataFrame | None,
              macro: pd.DataFrame | None = None,
              earnings_dates: list[tuple[str, str]] | None = None) -> dict[str, pd.DataFrame]:
    sigma = ewma_vol(returns, halflife=halflife)
    p_tail = pd.DataFrame({asset: tail_probability(returns[asset], sigma=sigma[asset],
                                                    threshold=threshold, window=window)
                           for asset in returns.columns})
    vov = pd.DataFrame({asset: vol_of_vol(returns[asset], sigma=sigma[asset], window=window)
                        for asset in returns.columns})
    dd = pd.DataFrame({asset: drawdown_depth(closes[asset], window=window)
                       for asset in returns.columns})
    return {"p_tail": p_tail, "vol_of_vol": vov, "drawdown": dd,
            "sentiment_z": _sentiment_z(sentiment, returns),
            "legal_flag": _legal_flag(legal, returns),
            "macro_stress": _macro_stress(macro, returns, window=window),
            "earnings_flag": _earnings_flag(earnings_dates, returns)}


def asset_scores(returns: pd.DataFrame, closes: pd.DataFrame, *,
                 weights: dict[str, float] | None = None,
                 halflife: float = 20.0, window: int = 250,
                 threshold: float = 2.5,
                 sentiment: pd.DataFrame | None = None,
                 legal: pd.DataFrame | None = None,
                 macro: pd.DataFrame | None = None,
                 earnings_dates: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Риск-скоры (0..1) дат x активы; признаки на дату t используют данные до t."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts = _features(returns, closes, halflife=halflife, window=window,
                      threshold=threshold, sentiment=sentiment, legal=legal,
                      macro=macro, earnings_dates=earnings_dates)
    terms = contribution_frames(parts, weights)
    linear = sum(terms.values()) + weights["bias"]
    scores = 1.0 / (1.0 + np.exp(-linear.clip(-30, 30)))
    return scores.fillna(0.0)


def contribution_frames(parts, weights):
    """Точные слагаемые существующего скора; формула и её знаки сохранены."""
    return {"p_tail": weights["p_tail"] * _cross_sectional_z(parts["p_tail"]),
            "vol_of_vol": weights["vol_of_vol"] * _cross_sectional_z(parts["vol_of_vol"]),
            "drawdown": weights["drawdown"] * _cross_sectional_z(parts["drawdown"]),
            "sentiment": weights["sentiment"] * parts["sentiment_z"],
            "legal": weights["legal"] * parts["legal_flag"],
            "macro": weights["macro"] * _cross_sectional_z(parts["macro_stress"]),
            "earnings": weights["earnings"] * parts["earnings_flag"]}


def explain_last(returns: pd.DataFrame, closes: pd.DataFrame, *,
                 weights: dict[str, float] | None = None,
                 halflife: float = 20.0, window: int = 250,
                 threshold: float = 2.5,
                 sentiment: pd.DataFrame | None = None,
                 legal: pd.DataFrame | None = None,
                 macro: pd.DataFrame | None = None,
                 earnings_dates: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Компоненты скора на последнюю дату: DataFrame[актив x компонент]."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts = _features(returns, closes, halflife=halflife, window=window,
                      threshold=threshold, sentiment=sentiment, legal=legal,
                      macro=macro, earnings_dates=earnings_dates)
    scores = asset_scores(returns, closes, weights=weights, halflife=halflife,
                          window=window, threshold=threshold,
                          sentiment=sentiment, legal=legal, macro=macro,
                          earnings_dates=earnings_dates)
    result = pd.DataFrame({
        "score": scores.iloc[-1],
        "p_tail": parts["p_tail"].iloc[-1],
        "vol_of_vol": parts["vol_of_vol"].iloc[-1],
        "drawdown": parts["drawdown"].iloc[-1],
        "sentiment_z": parts["sentiment_z"].iloc[-1],
        "legal_flag": parts["legal_flag"].iloc[-1],
        "macro_stress": parts["macro_stress"].iloc[-1],
        "earnings_flag": parts["earnings_flag"].iloc[-1],
    })
    for key, frame in contribution_frames(parts, weights).items():
        result["contribution_" + key] = frame.iloc[-1]
    result["bias"] = weights["bias"]
    return result


def overlay_caps(scores_last: pd.Series, *, max_weight: float,
                 strength: float) -> pd.Series:
    """Пер-тикерные потолки для HRP при включённом overlay."""
    return (max_weight * (1.0 - strength * scores_last.clip(0.0, 1.0))).clip(lower=1e-4)
