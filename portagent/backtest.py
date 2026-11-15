"""Walk-forward бэктест HRP и оценка трекера редких событий.

На каждую дату ребаланса веса считаются только по данным до этой даты
включительно. Удержание — до следующей ребалансировки; доходность дня
ребаланса ещё принадлежит предыдущим весам. Сентимент берётся из
исторических меток с published <= t. Overlay на веса здесь не применяется:
трекер оценивается отдельно (hit-rate и Brier).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import project_paths
from .features import prices as price_features
from .features import sentiment as sentiment_features
from .history import load_labels, load_price_panel, upsert_prices
from .portfolio.fuse import fuse_distances
from .portfolio.hrp import hrp_weights
from .tracker.rare_events import asset_scores, ewma_vol

MIN_HISTORY = 60
TRACKER_HORIZON = 5
ENGINE_VERSION = "holdings-costs-v1"


def _metrics(daily: pd.Series, turnovers: list[float]) -> dict[str, float]:
    clean = daily.fillna(0.0)
    if clean.empty:
        return {"total_return": 0.0, "cagr": 0.0, "ann_vol": 0.0,
                "max_drawdown": 0.0, "turnover": 0.0, "n_days": 0}
    equity = (1.0 + clean).cumprod()
    total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    n_days = int(len(clean))
    cagr = float(equity.iloc[-1] ** (252.0 / n_days) - 1.0) if n_days and equity.iloc[-1] > 0 else 0.0
    vol = float(clean.std(ddof=1) * np.sqrt(252.0)) if n_days > 1 else 0.0
    drawdown = float((equity / equity.cummax().clip(lower=1.0) - 1.0).min()) if len(equity) else 0.0
    turnover = float(np.mean(turnovers)) if turnovers else 0.0
    return {"total_return": total, "cagr": cagr, "ann_vol": vol,
            "max_drawdown": drawdown, "turnover": turnover, "n_days": n_days}


def _labels_until(labels: list[dict], day: str) -> list[dict]:
    kept = []
    for label in labels:
        stamp = str(label.get("date") or "")[:10]
        if not stamp or stamp <= day:
            kept.append(label)
    return kept


def _items_by_id(labels: list[dict]) -> dict[str, dict]:
    return {str(label["news_id"]): {"published_at": str(label.get("date", ""))[:10]}
            for label in labels if label.get("news_id")}


def _distance(panel_slice: pd.DataFrame, labels: list[dict] | None, fusion: dict) -> pd.DataFrame:
    from .features.view_quality import consistency

    returns = price_features.log_returns(panel_slice)
    D_price = price_features.distance_from_correlation(returns, min_periods=40)
    views: dict[str, pd.DataFrame | None] = {"price": D_price}
    coverage = {"sentiment": 0.0}
    quality: dict[str, float] = {}
    if labels:
        sent = sentiment_features.daily_series(
            labels, _items_by_id(labels), price_index=panel_slice.index,
            halflife_days=fusion["sentiment_halflife_days"],
            fallback_date=str(panel_slice.index[-1]))
        if not sent.empty:
            views["sentiment"] = price_features.distance_from_correlation(
                sent, min_periods=10)
            quality["sentiment"] = consistency(sent)
            coverage["sentiment"] = sentiment_features.coverage(
                labels, list(panel_slice.columns),
                min_news=fusion["min_news_per_asset"])["coverage_frac"]
    fused = fuse_distances(
        views, base="price", method=fusion["method"], coverage=coverage,
        view_quality=quality,
        w_price=fusion["w_price"], min_coverage=fusion["min_coverage"],
        snf_neighbors=fusion["snf_neighbors"], snf_iterations=fusion["snf_iterations"],
        snf_min_quality=fusion["snf_min_quality"])
    distance = fused["D"].where(fused["D"].notna(), D_price).fillna(0.0)
    values = np.array(distance.to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=distance.index, columns=distance.columns)


def _hrp(panel_slice: pd.DataFrame, labels: list[dict] | None, *,
         fusion: dict, hrp_cfg: dict) -> pd.Series:
    returns = price_features.log_returns(panel_slice)
    weights = hrp_weights(
        returns, _distance(panel_slice, labels, fusion),
        linkage_method=hrp_cfg["linkage"], max_weight=hrp_cfg["max_weight"],
        cash_share=hrp_cfg["cash_share"])["weights"]
    return weights.astype(float)


def _hold_period(simple: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Удержание позиций: веса дрейфуют; CASH имеет нулевую доходность.

    Total-return ряды предполагают реинвестирование дивидендов в тот же актив.
    Никакой бесплатной ежедневной ребалансировки к целевым весам нет.
    """
    aligned = simple.reindex(columns=weights.index).fillna(0.0)
    positions = (1.0 + aligned).cumprod().mul(weights, axis=1)
    wealth = positions.sum(axis=1) + (1.0 - float(weights.sum()))
    if (wealth <= 0).any():
        raise ValueError("стоимость портфеля стала неположительной")
    daily = wealth.div(wealth.shift(1).fillna(1.0)) - 1.0
    return daily, positions.iloc[-1] / wealth.iloc[-1]


def _execution(previous: pd.Series, target: pd.Series, *,
               commission_bps: float, spread_bps: float) -> dict[str, float]:
    """Сделки и затраты в долях NAV перед ребалансом, включая первый вход.

    Целевые доли относятся к NAV ПОСЛЕ издержек: решаем
    cost = rate * sum(abs(target * (1-cost) - previous)).
    Продажи и покупки облагаются отдельно; движение CASH не облагается.
    """
    commission = float(commission_bps) / 10000.0
    half_spread = float(spread_bps) / 20000.0
    if not (0 <= commission <= 0.05 and 0 <= half_spread <= 0.10):
        raise ValueError("некорректные commission_bps/spread_bps")
    union = previous.index.union(target.index)
    before = previous.reindex(union).fillna(0.0).to_numpy(dtype=float)
    desired = target.reindex(union).fillna(0.0).to_numpy(dtype=float)
    if (not np.isfinite(before).all() or not np.isfinite(desired).all()
            or (before < -1e-12).any() or (desired < 0).any()
            or before.sum() > 1.0 + 1e-9 or desired.sum() > 1.0 + 1e-9):
        raise ValueError("для исполнения нужны конечные long-only веса с суммой <= 1")
    rate = commission + half_spread
    low, high = 0.0, 1.0
    for _ in range(60):
        cost = (low + high) / 2.0
        payable = rate * np.abs(desired * (1.0 - cost) - before).sum()
        if cost < payable:
            low = cost
        else:
            high = cost
    cost = (low + high) / 2.0 if rate else 0.0
    delta = desired * (1.0 - cost) - before
    buys = float(np.maximum(delta, 0.0).sum())
    sells = float(np.maximum(-delta, 0.0).sum())
    traded = buys + sells
    return {"buys": buys, "sells": sells, "traded": traded,
            "commission": commission * traded, "spread": half_spread * traded,
            "total": rate * traded}


def _tracker_quality(panel: pd.DataFrame, *, tracker_cfg: dict, start: str,
                     end: str, labels: list[dict],
                     macro: pd.DataFrame | None = None,
                     earnings_pairs: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """score(t) против события «лог-доходность <= -threshold·σ в ближайшие 5 дней»."""
    returns = price_features.log_returns(panel)
    if returns.empty:
        return {"hit_rate": None, "brier": None, "n_alerts": 0, "n_scored": 0,
                "tau": tracker_cfg["warn_threshold"]}
    sent = None
    if labels:
        sent = sentiment_features.daily_series(
            labels, _items_by_id(labels), price_index=panel.index,
            halflife_days=5.0, fallback_date=str(panel.index[-1]))
    scores = asset_scores(
        returns, panel, weights=tracker_cfg["weights"],
        halflife=tracker_cfg["vol_halflife_days"], window=tracker_cfg["window_days"],
        threshold=tracker_cfg["sigma_threshold"], sentiment=sent,
        macro=macro, earnings_dates=earnings_pairs)
    sigma = ewma_vol(returns, halflife=tracker_cfg["vol_halflife_days"])
    threshold = tracker_cfg["sigma_threshold"]
    values = returns.to_numpy(dtype=float)
    sigma_values = sigma.reindex(index=returns.index, columns=returns.columns).to_numpy(dtype=float)
    score_values = scores.reindex(index=returns.index, columns=returns.columns).to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    events = []
    predicted = []
    for i in range(n_rows - TRACKER_HORIZON):
        day = str(returns.index[i])
        if day < start or day > end:
            continue
        future = values[i + 1:i + 1 + TRACKER_HORIZON]
        level = -threshold * sigma_values[i]
        for col in range(n_cols):
            score = score_values[i, col]
            if not np.isfinite(score) or not np.isfinite(level[col]):
                continue
            hit = bool(np.any(future[:, col] <= level[col]))
            events.append(1.0 if hit else 0.0)
            predicted.append(float(score))
    if not predicted:
        return {"hit_rate": None, "brier": None, "n_alerts": 0, "n_scored": 0,
                "tau": tracker_cfg["warn_threshold"]}
    pred = np.asarray(predicted)
    actual = np.asarray(events)
    tau = float(tracker_cfg["warn_threshold"])
    alerts = pred >= tau
    hit_rate = float(actual[alerts].mean()) if alerts.any() else None
    return {
        "hit_rate": hit_rate,
        "brier": float(np.mean((pred - actual) ** 2)),
        "n_alerts": int(alerts.sum()),
        "n_scored": int(len(pred)),
        "tau": tau,
    }


def walk_forward(panel: pd.DataFrame, *, config: dict, start: str | None = None,
                 end: str | None = None, rebalance_days: int | None = None,
                 labels: list[dict] | None = None,
                 include_daily: bool = False,
                 dividends: pd.DataFrame | None = None,
                 macro: pd.DataFrame | None = None,
                 earnings_pairs: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Панель close -> метрики fused / price-only / equal-weight и качество трекера.

    dividends (даты x активы, значения в валюте) добавляются к дневной
    доходности на ex-дату: total return. Только для явных колонок (MOEX):
    yfinance-котировки уже auto-adjusted и дивидендами не дублируются.
    """
    if panel.empty or panel.shape[1] < 2:
        raise ValueError("для бэктеста нужна панель минимум из 2 активов")
    frame = panel.sort_index().copy()
    frame.index = frame.index.astype(str).str[:10]
    frame = frame.loc[:, frame.notna().sum() >= MIN_HISTORY]
    if end:
        frame = frame.loc[frame.index <= end]
    if frame.shape[0] <= MIN_HISTORY + 1 or frame.shape[1] < 2:
        raise ValueError(
            f"после отсечения истории короче {MIN_HISTORY} наблюдений "
            f"({frame.shape[0]} дат, {frame.shape[1]} активов)")
    dates = list(frame.index)
    start = start or dates[0]
    end = end or dates[-1]
    step = int(rebalance_days or config["backtest"]["rebalance_days"])
    last_i = max(i for i, day in enumerate(dates) if day <= end)
    chosen = [i for i in range(MIN_HISTORY, last_i)
              if dates[i] >= start and (i - MIN_HISTORY) % step == 0]
    if not chosen:
        raise ValueError(f"в окне {start}..{end} нет даты ребаланса "
                         f"(нужно > {MIN_HISTORY} наблюдений до первой даты)")

    fusion = config["fusion"]
    hrp_cfg = config["hrp"]
    labels = labels or []
    simple = frame.ffill().pct_change(fill_method=None)
    if dividends is not None and not dividends.empty:
        aligned = dividends.reindex(index=frame.index,
                                    columns=[c for c in dividends.columns
                                             if c in frame.columns]).fillna(0.0)
        yield_daily = aligned / frame.shift(1).replace(0.0, np.nan)
        simple = simple + yield_daily.fillna(0.0)
    books: dict[str, dict[str, float]] = {name: {} for name in
                                         ("fused", "price_only", "equal_weight")}
    gross_books: dict[str, dict[str, float]] = {name: {} for name in books}
    turns: dict[str, list[float]] = {name: [] for name in books}
    previous = {name: pd.Series(0.0, index=frame.columns) for name in books}
    executions: dict[str, list[dict]] = {name: [] for name in books}
    nav = {name: 1.0 for name in books}
    commission_bps = float(config["backtest"]["commission_bps"])
    spread_bps = float(config["backtest"]["spread_bps"])
    cash = float(hrp_cfg["cash_share"])
    equal = pd.Series((1.0 - cash) / frame.shape[1], index=frame.columns)

    for position, i in enumerate(chosen):
        day = dates[i]
        nxt = chosen[position + 1] if position + 1 < len(chosen) else last_i
        if nxt <= i:
            continue
        window = simple.iloc[i + 1:nxt + 1]
        until = _labels_until(labels, day)
        weights = {
            "fused": _hrp(frame.iloc[:i + 1], until, fusion=fusion, hrp_cfg=hrp_cfg),
            "price_only": _hrp(frame.iloc[:i + 1], None, fusion=fusion, hrp_cfg=hrp_cfg),
            "equal_weight": equal,
        }
        for name, current in weights.items():
            execution = _execution(previous[name], current,
                                   commission_bps=commission_bps, spread_bps=spread_bps)
            if position:
                turns[name].append(0.5 * execution["traded"])
            executions[name].append({"date": day, **execution,
                                     "nav_before": nav[name],
                                     "commission_paid": nav[name] * execution["commission"],
                                     "spread_paid": nav[name] * execution["spread"]})
            # Исполнение на закрытии t после доходности старых позиций за t.
            # Первая дата — только вход из CASH; принудительной продажи в конце нет.
            books[name][day] = (1.0 + books[name].get(day, 0.0)) * (1.0 - execution["total"]) - 1.0
            gross_books[name].setdefault(day, 0.0)
            nav[name] *= 1.0 - execution["total"]
            series, previous[name] = _hold_period(window, current)
            rows = {str(stamp): float(value) for stamp, value in series.items()}
            books[name].update(rows)
            gross_books[name].update(rows)
            nav[name] *= float((1.0 + series).prod())

    strategies = {}
    gross_strategies = {}
    daily: dict[str, list[float]] = {}
    for name, rows in books.items():
        series = pd.Series(rows, dtype=float)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        strategies[name] = _metrics(series, turns[name])
        gross_strategies[name] = _metrics(pd.Series(gross_books[name], dtype=float), turns[name])
        commission_paid = sum(row["commission_paid"] for row in executions[name])
        spread_paid = sum(row["spread_paid"] for row in executions[name])
        strategies[name].update({
            "commission_paid": commission_paid, "spread_paid": spread_paid,
            "cost_paid": commission_paid + spread_paid,
            "cost_drag": gross_strategies[name]["total_return"] - strategies[name]["total_return"],
            "traded_notional": sum(row["traded"] * row["nav_before"] for row in executions[name]),
        })
        if include_daily:
            daily[name] = [float(v) for v in series.to_numpy()]
    tracker = _tracker_quality(
        frame, tracker_cfg=config["tracker"], start=start, end=end, labels=labels,
        macro=macro, earnings_pairs=earnings_pairs)
    result = {
        "start": dates[chosen[0]],
        "end": dates[min(last_i, len(dates) - 1)],
        "rebalance_days": step,
        "n_rebalances": len(chosen),
        "n_assets": int(frame.shape[1]),
        "strategies": strategies,
        "gross_strategies": gross_strategies,
        "engine_version": ENGINE_VERSION,
        "cost_model": {"commission_bps": commission_bps, "spread_bps": spread_bps,
                       "spread_is_full": True, "initial_entry": True,
                       "terminal_liquidation": False, "execution": "rebalance_close",
                       "dividends": "same_asset_reinvestment", "cash_return": 0.0},
        "executions": executions,
        "tracker": tracker,
    }
    if include_daily:
        result["daily_returns"] = daily
        result["daily_dates"] = list(series.index)
        result["gross_daily_returns"] = {name: pd.Series(rows, dtype=float).reindex(result["daily_dates"]).tolist()
                                          for name, rows in gross_books.items()}
    return result


def run_backtest(config: dict, config_path: str | Path, *, start: str | None = None,
                 end: str | None = None, rebalance_days: int | None = None,
                 fetch: bool = False) -> dict[str, Any]:
    """Прошлые данные -> walk_forward -> отчёт. fetch обновляет историю с бирж."""
    from datetime import date as date_type

    from .history import load_dividends_panel, load_earnings, load_macro_panel

    paths = project_paths(config, config_path)
    history_dir = paths["data"] / "history"
    fetch_statuses: list[dict] = []
    if fetch:
        from .pipeline.daily import _collect_market_net, collect_track_b

        run_day = end or date_type.today().isoformat()
        frames, _payloads, fetch_statuses = _collect_market_net(config, run_day)
        if frames:
            upsert_prices(history_dir, frames)
        collect_track_b(config, history_dir,
                        paths["data"] / "manual", run_day)
    panel = load_price_panel(history_dir, end=end)
    if panel.empty:
        raise RuntimeError("история пуста: запустите с --fetch или сделайте дневной прогон")
    dividends = load_dividends_panel(history_dir, end=end)
    macro = load_macro_panel(history_dir, end=end)
    if macro.empty:
        macro = None
    earnings_pairs = [(row["date"], row["asset"])
                      for row in load_earnings(history_dir)] or None
    result = walk_forward(panel, config=config, start=start, end=end,
                          rebalance_days=rebalance_days, labels=load_labels(history_dir),
                          dividends=dividends if not dividends.empty else None,
                          macro=macro, earnings_pairs=earnings_pairs, include_daily=True)
    result["fetch_statuses"] = fetch_statuses
    directory = paths["reports"] / "backtest"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result['start']}_{result['end']}"
    json_path = directory / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = directory / f"{stem}.md"
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    result["paths"] = {"markdown": str(markdown_path), "json": str(json_path)}
    return result


def _render_markdown(report: dict) -> str:
    from .report.markdown import render_backtest

    return render_backtest(report)
