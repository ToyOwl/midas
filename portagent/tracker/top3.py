"""Ранжирование кандидатов на резкие движения без изменения весов.

Ранжирование: память прошлых событий + сумма z; top3/count/selective.
Признаки, банк событий и логистическая калибровка строятся на ценах акций. Один шаг = следующая
торговая запись актива. Оценки исследовательские, не гарантия вероятности.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp, softmax

VERSION = "equity-top3-memory-count-v1"
STRENGTHS = (0., .25, .5, 1.)
GRID = np.linspace(-3, 3, 13)
GRID_WEIGHTS = np.exp(-GRID ** 2 / 2)
GRID_WEIGHTS /= GRID_WEIGHTS.sum()
DEPENDENCE_DECAY = .5 ** (1 / 30)
MIN_CALIBRATION_ROWS = 500
MIN_CALIBRATION_EVENTS = 20


def poisson_binomial(p):
    out = np.array([1.])
    for value in np.asarray(p, float):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Вероятность должна быть в [0, 1]")
        out = np.convolve(out, [1 - value, value])
    return out


def components(p):
    p = np.asarray(p, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("Требуются вероятности строго внутри (0, 1)")
    groups = []
    for strength in STRENGTHS:
        rows = []
        for target in p:
            shift = brentq(lambda a: GRID_WEIGHTS @ expit(a + strength * GRID) - target, -40, 40)
            rows.append(expit(shift + strength * GRID))
        groups.append(np.array(rows).T)
    return np.concatenate(groups)


def count_distribution(p, log_weights=None):
    lw = np.zeros(len(STRENGTHS)) if log_weights is None else np.asarray(log_weights, float)
    if lw.shape != (len(STRENGTHS),) or not np.isfinite(lw).all():
        raise ValueError("Некорректные веса зависимости")
    mix = np.exp(lw - logsumexp(lw))
    weights = (mix[:, None] * GRID_WEIGHTS).ravel()
    pp = components(p)
    return sum(w * poisson_binomial(row) for w, row in zip(weights, pp)), weights @ pp


def learn_dependence(log_weights, p, y):
    pp = components(p)
    y = np.asarray(y)
    if y.shape != (len(p),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Нужны исходы целой завершённой даты")
    ll = np.log(np.where(y[None, :], pp, 1 - pp)).sum(axis=1).reshape(len(STRENGTHS), len(GRID))
    updated = DEPENDENCE_DECAY * log_weights + logsumexp(ll + np.log(GRID_WEIGHTS), axis=1)
    return updated - logsumexp(updated)


def event_features(close, *, threshold, halflife, window):
    close = close.dropna().astype(float)
    close = close[close > 0]
    r = np.log(close / close.shift())
    sigma = r.ewm(halflife=halflife, min_periods=20).std()
    lagged = sigma.shift()
    z = r / lagged.replace(0, np.nan)
    y = (z.abs() >= threshold).astype(float).where(z.notna())
    valid = y.notna().rolling(window, min_periods=1).sum()
    # В прогнозе следующего наблюдения событие текущего дня уже известно.
    frequency = ((y.fillna(0).rolling(window, min_periods=1).sum() + .5) / (valid + 1)).where(valid >= 40)
    vfast = r.ewm(halflife=5, min_periods=10).std()
    ratio = (vfast / sigma.replace(0, np.nan)).clip(0, 4)
    dd = (1 - close / close.rolling(window, min_periods=20).max()).clip(0, 1)
    momentum = (r.rolling(5).sum().abs() / (sigma * np.sqrt(5)).replace(0, np.nan)).clip(0, 5)
    f = pd.DataFrame({"frequency": frequency, "volatility_ratio": ratio,
        "drawdown": dd, "shock_size": z.abs().clip(0, 6), "momentum": momentum})
    return {"features": f, "outcomes": y, "z": z, "sigma": lagged, "returns": r}


def zscore(values):
    values = np.asarray(values, float)
    std = np.std(values)
    return (values - np.mean(values)) / std if std > 1e-9 else np.zeros_like(values)


def signature(features):
    # Ручная безразмерная сигнатура, НЕ нейросетевой embedding TabM.
    x = np.array(features, float) * [10, 1, 5, .5, .5]
    return softmax(np.clip(x, -6, 6), axis=-1)


def memory_scores(features, bank):
    if not bank:
        return np.zeros(len(features))
    q = signature(features)
    reference = signature(np.asarray(bank[-1000:]))
    # Симметричный KL; все аналоги предшествуют прогнозируемым событиям.
    delta = q[:, None, :] - reference[None, :, :]
    distance = .5 * (delta * (np.log(q[:, None, :]) - np.log(reference[None, :, :]))).sum(axis=2)
    k = min(25, len(reference))
    return -np.sort(distance, axis=1)[:, :k].mean(axis=1)


def fit_calibration(records):
    if len(records) < MIN_CALIBRATION_ROWS or sum(r[3] for r in records) < MIN_CALIBRATION_EVENTS:
        return None
    a = np.asarray([r[1:] for r in records], float)
    dates = [r[0] for r in records]
    counts = {d: dates.count(d) for d in set(dates)}
    weights = np.array([1 / counts[d] for d in dates])
    x, baseline, y = a.T
    offset = logit(np.clip(baseline, 1e-6, 1 - 1e-6))
    def loss(beta):
        eta = offset + beta[0] + beta[1] * x
        return np.sum(weights * (np.logaddexp(0, eta) - y * eta)) + 5 * np.dot(beta, beta)
    fit = minimize(loss, [0., 0.], method="L-BFGS-B", bounds=[(-4, 4), (0, 4)])
    return fit.x if fit.success and np.isfinite(fit.x).all() else None


def attention_lists(assets, ratings, probabilities, baselines, past_maxima):
    order = sorted(range(len(assets)), key=lambda i: (-ratings[i], assets[i]))
    top = order[:3]
    count = min(3, int(np.floor(np.sum(probabilities) + .5)))
    gate = float(np.median(past_maxima)) if past_maxima else None
    admitted = len(assets) >= 4 and gate is not None and max(ratings) >= gate
    selective = [i for i in top if probabilities[i] > baselines[i]] if admitted else []
    return {"top3": [assets[i] for i in top], "count_top3": [assets[i] for i in order[:count]],
            "selective_top3": [assets[i] for i in selective], "gate": gate, "gate_passed": bool(admitted),
            "selective_note": "Первые три по рейтингу; максимум не ниже медианы прошлых максимумов; оценка события выше собственной исторической частоты. Без добора четвёртого."}


def detect(frames, *, day, threshold, halflife, window, return_history=False):
    panels = {asset: event_features(frame.loc[frame.index <= day, "close"],
              threshold=threshold, halflife=halflife, window=window) for asset, frame in sorted(frames.items())}
    assets = list(panels)
    dates = sorted({d for panel in panels.values() for d in panel["features"].index})
    previous, current = {}, {}
    records, bank, maxima, validation, full_dates = [], [], [], [], []
    forecast_rows = []
    lw = np.zeros(len(STRENGTHS))
    beta, last = None, None
    for date in dates:
        outcomes = {}
        for asset, panel in panels.items():
            if date not in panel["features"].index:
                continue
            y = panel["outcomes"].loc[date]
            old = previous.get(asset)
            if old is not None and np.isfinite(y):
                records.append((date, old["rank"], old["baseline"], float(y)))
                validation.append((date, old["p"], old["baseline"], float(y)))
                outcomes[asset] = (old["p"], int(y))
                if y:
                    bank.append(old["features"])
            f = panel["features"].loc[date].to_numpy(float)
            if np.isfinite(f).all():
                current[asset] = f
        if len(outcomes) == len(assets) and assets:
            lw = learn_dependence(lw, [outcomes[a][0] for a in assets], [outcomes[a][1] for a in assets])
            full_dates.append(date)
        if len(current) != len(assets) or not assets:
            continue
        records = [r for r in records if r[0] >= dates[max(0, dates.index(date) - window)]]
        if len(maxima) % 10 == 0:
            beta = fit_calibration(records)
        f = np.array([current[a] for a in assets])
        memory = memory_scores(f, bank)
        memory_z, frequency_z = zscore(memory), zscore(f[:, 0])
        regime_z = zscore(f[:, 1] + .5 * f[:, 3] + f[:, 4])
        ratings = memory_z + frequency_z + regime_z
        probabilities = f[:, 0] if beta is None else expit(logit(f[:, 0]) + beta[0] + beta[1] * ratings)
        last = {"date": date, "ratings": ratings, "probabilities": probabilities,
                "baseline": f[:, 0], "memory_z": memory_z, "frequency_z": frequency_z,
                "regime_z": regime_z, "memory": memory, "prior_maxima": maxima.copy()}
        # Не заменяем прогноз актива без новой торговой записи его цены.
        for i, asset in enumerate(assets):
            if date in panels[asset]["features"].index:
                previous[asset] = {"rank": ratings[i], "p": probabilities[i],
                                   "baseline": f[i, 0], "features": f[i].copy()}
                if return_history:
                    forecast_rows.append({"date": str(date), "asset": asset,
                        "current_top3": float(probabilities[i]), "current_rating": float(ratings[i])})
        maxima.append(float(max(ratings)))
    if last is None:
        return {"version": VERSION, "status": "insufficient_history", "rows": [],
                "note": "Недостаточно прошлых наблюдений для полного состава активов."}
    pmf, marginal = count_distribution(last["probabilities"], lw)
    policies = attention_lists(assets, last["ratings"], marginal, last["baseline"], last["prior_maxima"])
    rows = []
    for i, asset in enumerate(assets):
        panel = panels[asset]
        observed = panel["z"].dropna()
        last_z = float(observed.iloc[-1]) if len(observed) else None
        rows.append({"asset": asset, "probability": float(marginal[i]),
            "baseline": float(last["baseline"][i]), "rating": float(last["ratings"][i]),
            "memory_z": float(last["memory_z"][i]), "frequency_z": float(last["frequency_z"][i]),
            "regime_z": float(last["regime_z"][i]), "observed_z": last_z,
            "observed_date": str(observed.index[-1]) if len(observed) else None,
            "observed_event": bool(last_z is not None and abs(last_z) >= threshold),
            "reasons": [f"Сходство с прошлыми шоками: z-оценка {last['memory_z'][i]:+.2f}.",
                        f"Частота прошлых событий: {last['baseline'][i]:.1%}; относительная z-оценка {last['frequency_z'][i]:+.2f}.",
                        f"Текущая волатильность и движение: z-оценка {last['regime_z'][i]:+.2f}."]})
    valid = np.asarray([r[1:] for r in validation], float)
    result = {"version": VERSION, "status": "research" if beta is not None else "frequency_baseline",
        "date": last["date"], "threshold_sigma": threshold, "horizon": "Следующая торговая запись каждого актива",
        "event_definition": f"Абсолютная дневная log-доходность ≥ {threshold:g} × EWMA-волатильность предыдущего наблюдения",
        "rows": sorted(rows, key=lambda r: (-r["rating"], r["asset"])), "policies": policies,
        "count_pmf": pmf.tolist(), "expected_count": float(marginal.sum()),
        "p_three_or_more": float(pmf[3:].sum()), "independent_pmf": poisson_binomial(marginal).tolist(),
        "dependence": {"strengths": list(STRENGTHS), "weights": np.exp(lw - logsumexp(lw)).tolist(),
            "completed_dates": len(full_dates), "fit_end": full_dates[-1] if full_dates else None,
            "halflife_dates": 30, "grid_nodes": 13},
        "calibration": {"trained": beta is not None, "coefficients": beta.tolist() if beta is not None else None,
            "rows": len(records), "events": int(sum(r[3] for r in records)), "analogs": len(bank),
            "method": "Past-only logit offset + неотрицательный коэффициент суммы z; L2=10; переоценка каждые 10 дат"},
        "validation": {"observations": len(validation),
            "brier": float(np.mean((valid[:, 0] - valid[:, 2]) ** 2)) if len(valid) else None,
            "baseline_brier": float(np.mean((valid[:, 1] - valid[:, 2]) ** 2)) if len(valid) else None,
            "start": validation[0][0] if validation else None, "end": validation[-1][0] if validation else None},
        "source": "Ценовой детектор: память событий, сумма z, count/selective, conditional_draw; обучение на акциях",
        "note": "Исследовательская оценка. Спортивные модели и коэффициенты не используются. Список внимания не меняет веса HRP."}
    if return_history:
        result["forecast_rows"] = forecast_rows
    return result
