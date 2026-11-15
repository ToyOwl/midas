"""Заморозка прогноза до события и проверка результата после.

Прогноз (data/forecasts/<date>.json) пишется в конце дневного прогона и
immutable: файл замораживает данные, доступные на дату — веса, скоры
трекера, метки LLM с цитатами, покрытия модальностей, матрицы расстояний
видов и дайджест конфига. Это позволяет переигрывать эксперименты без
повторных вызовов LLM.

Settlement (data/settlement/<date>.json + data/history/settlement.parquet)
закрывает прогноз через `horizon_days` торговых дней: событие редкого
движения (лог-доходность <= -threshold*sigma_D, sigma посчитана по данным
ДО даты прогноза — без подглядывания), реализованные доходности актива и
портфеля. До истечения горизонта статус pending.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features.prices import log_returns
from .history import _upsert  # тот же идемпотентный паттерн upsert
from .tracker.rare_events import ewma_vol

PORTFOLIO_ROW = "__portfolio__"


def _config_digest(config: dict) -> str:
    """SHA релевантных для весов секций конфига (юниверс/fusion/tracker/hrp)."""
    payload = json.dumps({key: config.get(key) for key in
                          ("universe", "fusion", "tracker", "hrp", "backtest")},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_forecast(forecasts_dir: Path, snapshot: dict[str, Any]) -> Path:
    """Замораживает прогноз дня. Immutable: существующий файл с другим
    содержимым — ошибка; идентичная повторная заморозка — no-op."""
    directory = Path(forecasts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{snapshot['date']}.json"
    incoming_sha = _snapshot_sha(snapshot)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("snapshot_sha") == incoming_sha:
            return target
        raise FileExistsError(
            f"прогноз {target.name} уже заморожен с другим содержимым; "
            f"новая заморозка того же дня запрещена (immutable)")
    frozen = {**snapshot, "snapshot_sha": incoming_sha}
    payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=1)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return target


def _snapshot_sha(snapshot: dict) -> str:
    payload = json.dumps({k: v for k, v in snapshot.items() if k != "snapshot_sha"},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_forecasts(forecasts_dir: Path) -> list[dict]:
    directory = Path(forecasts_dir)
    if not directory.is_dir():
        return []
    forecasts = []
    for path in sorted(directory.glob("*.json")):
        try:
            forecasts.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            continue  # битый файл — пропускаем, не роняем батч
    return forecasts


def settle_forecast(forecast: dict, panel: pd.DataFrame, *, threshold: float,
                    halflife: float, horizon: int) -> dict[str, Any]:
    """Закрывает один прогноз по панели цен.

    Возвращает {"date", "status": "settled"|"pending", "rows": [...]}.
    sigma для события — EWMA по лог-доходностям до даты прогноза включительно.
    """
    date = str(forecast.get("date"))
    index = [str(value)[:10] for value in panel.index]
    if date not in index:
        return {"date": date, "status": "pending",
                "reason": "дата прогноза отсутствует в панели истории"}
    position = len(index) - 1 - index[::-1].index(date)
    future = panel.iloc[position + 1: position + 1 + horizon]
    if len(future) < horizon:
        return {"date": date, "status": "pending",
                "reason": f"будущих дней {len(future)} < {horizon}"}

    history = panel.iloc[: position + 1]
    returns = log_returns(history)
    sigma = ewma_vol(returns, halflife=halflife).iloc[-1]
    future_returns = log_returns(pd.concat([history.tail(1), future])).iloc[1:]
    weights = forecast.get("weights", {})
    rows: list[dict] = []
    for asset, weight in weights.items():
        if asset not in panel.columns or asset not in future_returns.columns:
            continue
        level = -threshold * float(sigma.get(asset, np.nan))
        series = future_returns[asset].to_numpy(dtype=float)
        event = bool(np.any(series <= level)) if np.isfinite(level) else False
        closes = future[asset].to_numpy(dtype=float)
        base = float(history[asset].iloc[-1])
        rows.append({
            "forecast_date": date,
            "asset": asset,
            "weight": float(weight),
            "score": float(forecast.get("tracker_scores", {}).get(asset, np.nan)),
            "warned": bool(forecast.get("tracker_warned", {}).get(asset, False)),
            "event": event,
            "ret_1d": float(closes[0] / base - 1.0) if len(closes) else np.nan,
            "ret_5d": float(closes[-1] / base - 1.0) if len(closes) else np.nan,
        })
    if rows:
        portfolio_1d = sum(row["ret_1d"] * row["weight"] for row in rows
                           if np.isfinite(row["ret_1d"]))
        portfolio_5d = sum(row["ret_5d"] * row["weight"] for row in rows
                           if np.isfinite(row["ret_5d"]))
        rows.append({"forecast_date": date, "asset": PORTFOLIO_ROW,
                     "weight": float(sum(row["weight"] for row in rows)),
                     "score": np.nan, "warned": False, "event": False,
                     "ret_1d": float(portfolio_1d), "ret_5d": float(portfolio_5d)})
    return {"date": date, "status": "settled", "rows": rows}


def settle_pending(forecasts_dir: Path, settlement_dir: Path, history_dir: Path,
                   panel: pd.DataFrame, *, threshold: float, halflife: float,
                   horizon: int, only_date: str | None = None) -> dict[str, Any]:
    """Закрывает прогнозы без data/settlement/<date>.json (или pending).
    only_date ограничивает батч одним днём. Пишет settlement/<date>.json и
    upsert в data/history/settlement.parquet."""
    forecasts = load_forecasts(forecasts_dir)
    settled: list[str] = []
    still_pending: list[str] = []
    fresh_rows: list[dict] = []
    Path(settlement_dir).mkdir(parents=True, exist_ok=True)
    for forecast in forecasts:
        date = str(forecast.get("date"))
        if only_date is not None and date != only_date:
            continue
        marker = Path(settlement_dir) / f"{date}.json"
        if marker.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous.get("status") == "settled":
                continue  # уже закрыт окончательно
        result = settle_forecast(forecast, panel, threshold=threshold,
                                 halflife=halflife, horizon=horizon)
        marker.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        if result["status"] == "settled":
            settled.append(date)
            fresh_rows.extend(result["rows"])
        else:
            still_pending.append(date)
    if fresh_rows:
        frame = pd.DataFrame(fresh_rows)
        _upsert(Path(history_dir) / "settlement.parquet", frame,
                ["forecast_date", "asset"])
    return {"settled": settled, "pending": still_pending}


def load_settlement(history_dir: Path) -> pd.DataFrame:
    path = Path(history_dir) / "settlement.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _brier(scores: pd.Series, events: pd.Series) -> float:
    valid = scores.notna() & events.notna()
    if not valid.any():
        return float("nan")
    return float(np.mean((scores[valid] - events[valid].astype(float)) ** 2))


def quality_table(history_dir: Path) -> dict[str, Any]:
    """Качество трекера по закрытым прогнозам: общий, помесячно, по активам."""
    frame = load_settlement(history_dir)
    if frame.empty:
        return {"n_rows": 0}
    assets = frame[frame["asset"] != PORTFOLIO_ROW].copy()
    assets["month"] = assets["forecast_date"].str[:7]

    def block(sub: pd.DataFrame) -> dict:
        events = sub["event"].astype(bool)
        base = float(events.mean()) if len(sub) else float("nan")
        warned = sub["warned"].astype(bool)
        hits = events[warned]
        return {
            "n": int(len(sub)),
            "base_rate": round(base, 4),
            "brier": round(_brier(sub["score"], sub["event"]), 4),
            "brier_base": round(float(np.mean((base - events.astype(float)) ** 2)), 4)
            if len(sub) else float("nan"),
            "warnings": int(warned.sum()),
            "hit_rate": round(float(hits.mean()), 4) if len(hits) else None,
        }

    monthly = {month: block(group) for month, group in assets.groupby("month")}
    by_asset = {}
    for asset, group in assets.groupby("asset"):
        by_asset[asset] = {**block(group), "mean_weight": round(float(group["weight"].mean()), 4)}
    ranked = sorted(by_asset.items(), key=lambda kv: kv[1]["brier"])
    portfolio = frame[frame["asset"] == PORTFOLIO_ROW]
    return {
        "n_rows": int(len(frame)),
        "overall": block(assets),
        "monthly": monthly,
        "by_asset": by_asset,
        "best_calibrated": ranked[:3],
        "worst_calibrated": ranked[-3:][::-1],
        "portfolio_ret_1d_mean": round(float(portfolio["ret_1d"].mean()), 4)
        if len(portfolio) else None,
        "portfolio_ret_5d_mean": round(float(portfolio["ret_5d"].mean()), 4)
        if len(portfolio) else None,
    }


def render_quality_markdown(quality: dict) -> str:
    lines = ["# Качество трекера редких событий (settlement)", ""]
    if not quality.get("n_rows"):
        lines.append("Закрытых прогнозов пока нет: прогнозы замораживаются каждым "
                     "прогоном и закрываются через horizon_days торговых дней.")
        return "\n".join(lines) + "\n"
    overall = quality["overall"]
    hit = overall["hit_rate"]
    lines += [
        f"Активо-строк: {quality['n_rows']}; базовая частота события "
        f"{overall['base_rate']:.1%}; Brier {overall['brier']} "
        f"(у тривиального прогноза базовой частотой {overall['brier_base']}).",
        f"Предупреждений {overall['warnings']}; hit-rate "
        f"{'—' if hit is None else f'{hit:.1%}'}.",
        "",
        "## По месяцам",
        "",
        "| Месяц | n | База | Brier | Предупр. | Hit-rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for month in sorted(quality["monthly"]):
        row = quality["monthly"][month]
        hit = row["hit_rate"]
        lines.append(f"| {month} | {row['n']} | {row['base_rate']:.1%} | "
                     f"{row['brier']} | {row['warnings']} | "
                     f"{'—' if hit is None else f'{hit:.1%}'} |")
    lines += ["", "## Калибровка по активам (лучшие/худшие)", ""]
    for label, entries in (("лучшие", quality["best_calibrated"]),
                           ("худшие", quality["worst_calibrated"])):
        for asset, row in entries:
            lines.append(f"- {label}: **{asset}** — Brier {row['brier']}, n={row['n']}, "
                         f"средний вес {row['mean_weight']:.1%}.")
    p1, p5 = quality.get("portfolio_ret_1d_mean"), quality.get("portfolio_ret_5d_mean")
    if p1 is not None:
        lines += ["", f"Портфель (среднее по закрытым): 1д {p1:+.2%}, 5д {p5:+.2%}."]
    lines.append("")
    return "\n".join(lines)


def write_quality_report(reports_dir: Path, quality: dict) -> Path:
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "quality.md"
    path.write_text(render_quality_markdown(quality), encoding="utf-8")
    (directory / "quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
