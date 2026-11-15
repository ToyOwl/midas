"""Накопление истории прогонов: цены, метки LLM, скоры трекера.

Файлы parquet в data/history/. Это производное состояние, не raw-архив:
повторный прогон того же дня заменяет строки с тем же ключом (upsert),
веса от этого не зависят. Чтение и запись требуют pyarrow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PRICES = "prices.parquet"
LABELS = "labels.parquet"
TRACKER = "tracker.parquet"
SETTLEMENT = "settlement.parquet"
AGREEMENT = "agreement.parquet"
DIVIDENDS = "dividends.parquet"
MACRO = "macro.parquet"
EARNINGS = "earnings.parquet"


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "для истории нужен pyarrow: .venv\\Scripts\\pip install pyarrow") from error


def _upsert(path: Path, fresh: pd.DataFrame, keys: list[str]) -> int:
    """Склеивает fresh с уже записанным файлом. Совпадающие ключи — из fresh."""
    if fresh.empty:
        return 0
    _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_parquet(path)
        combined = pd.concat([previous, fresh], ignore_index=True)
        combined = combined.drop_duplicates(keys, keep="last")
    else:
        combined = fresh.copy()
    combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return int(len(combined))


def frames_to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{asset: DataFrame[close, volume]} -> длинная таблица date/asset/close/volume."""
    parts: list[pd.DataFrame] = []
    for asset, frame in sorted(frames.items()):
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        part = pd.DataFrame({
            "close": frame["close"].astype(float),
            "volume": (frame["volume"].astype(float) if "volume" in frame.columns
                       else float("nan")),
        })
        part.index = part.index.astype(str).str[:10]
        part.index.name = "date"
        part = part.reset_index()
        part["asset"] = asset
        parts.append(part[["date", "asset", "close", "volume"]])
    if not parts:
        return pd.DataFrame(columns=["date", "asset", "close", "volume"])
    return pd.concat(parts, ignore_index=True)


def upsert_prices(history_dir: Path, frames: dict[str, pd.DataFrame]) -> int:
    return _upsert(Path(history_dir) / PRICES, frames_to_long(frames), ["date", "asset"])


def upsert_labels(history_dir: Path, labels: list[dict], items_by_id: dict[str, dict], *,
                  fallback_date: str) -> int:
    rows = []
    for label in labels:
        item = items_by_id.get(label.get("news_id", ""), {})
        published = item.get("published_at")
        rows.append({
            "date": (str(published)[:10] if published else fallback_date),
            "news_id": str(label.get("news_id", "")),
            "ticker": str(label.get("ticker", "")),
            "sentiment": float(label.get("sentiment", 0.0)),
            "confidence": float(label.get("confidence", 0.0)),
            "source": str(item.get("source", "")),
        })
    frame = pd.DataFrame(rows, columns=["date", "news_id", "ticker", "sentiment",
                                        "confidence", "source"])
    return _upsert(Path(history_dir) / LABELS, frame, ["news_id", "ticker"])


def upsert_tracker(history_dir: Path, run_date: str, rows: list[dict]) -> int:
    records = []
    for row in rows:
        records.append({
            "date": run_date,
            "asset": row["asset"],
            "score": float(row["score"]),
            "p_tail": None if row.get("p_tail") is None else float(row["p_tail"]),
            "drawdown": float(row.get("drawdown") or 0.0),
            "sentiment_z": float(row.get("sentiment_z") or 0.0),
            "legal_flag": int(row.get("legal_flag") or 0),
            "macro_stress": float(row.get("macro_stress") or 0.0),
            "earnings_flag": int(row.get("earnings_flag") or 0),
            "warned": bool(row.get("warned")),
        })
    frame = pd.DataFrame(records, columns=["date", "asset", "score", "p_tail", "drawdown",
                                           "sentiment_z", "legal_flag", "macro_stress",
                                           "earnings_flag", "warned"])
    return _upsert(Path(history_dir) / TRACKER, frame, ["date", "asset"])


def load_price_panel(history_dir: Path, *, end: str | None = None) -> pd.DataFrame:
    """Панель close (даты x активы). Строки новее `end` отбрасываются."""
    path = Path(history_dir) / PRICES
    if not path.is_file():
        return pd.DataFrame()
    _require_pyarrow()
    frame = pd.read_parquet(path)
    frame["date"] = frame["date"].astype(str).str[:10]
    if end:
        frame = frame[frame["date"] <= end]
    panel = frame.pivot(index="date", columns="asset", values="close").sort_index()
    panel.columns.name = None
    return panel


def load_labels(history_dir: Path) -> list[dict[str, Any]]:
    path = Path(history_dir) / LABELS
    if not path.is_file():
        return []
    _require_pyarrow()
    frame = pd.read_parquet(path)
    frame["date"] = frame["date"].astype(str).str[:10]
    return frame.to_dict(orient="records")


def upsert_agreement(history_dir: Path, run_date: str, primary: list[dict],
                     secondary: list[dict], *, model_a: str, model_b: str) -> int:
    """A4: попарные метки двух движков по общим (news, ticker) — задел под
    калибровку A3. Ключ: news_id + ticker + model_b (смена второго движка
    не затирает историю предыдущего)."""
    index_a = {(label["news_id"], label["ticker"]): label for label in primary}
    index_b = {(label["news_id"], label["ticker"]): label for label in secondary}
    rows = []
    for key in sorted(set(index_a) & set(index_b)):
        a, b = index_a[key], index_b[key]
        rows.append({
            "date": run_date,
            "news_id": key[0],
            "ticker": key[1],
            "model_a": model_a,
            "model_b": model_b,
            "sentiment_a": float(a["sentiment"]),
            "sentiment_b": float(b["sentiment"]),
            "delta": round(float(b["sentiment"]) - float(a["sentiment"]), 3),
        })
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=["date", "news_id", "ticker", "model_a",
                                        "model_b", "sentiment_a", "sentiment_b", "delta"])
    return _upsert(Path(history_dir) / AGREEMENT, frame, ["news_id", "ticker", "model_b"])


def upsert_dividends(history_dir: Path, rows: list[dict]) -> int:
    """B5: дивиденды (long: date/asset/value), ключи [date, asset]."""
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=["date", "asset", "value"])
    frame["date"] = frame["date"].astype(str).str[:10]
    return _upsert(Path(history_dir) / DIVIDENDS, frame, ["date", "asset"])


def load_dividends_panel(history_dir: Path, *, end: str | None = None) -> pd.DataFrame:
    """Wide-панель дивидендов (даты x активы, NaN -> 0.0)."""
    path = Path(history_dir) / DIVIDENDS
    if not path.is_file():
        return pd.DataFrame()
    _require_pyarrow()
    frame = pd.read_parquet(path)
    frame["date"] = frame["date"].astype(str).str[:10]
    if end:
        frame = frame[frame["date"] <= end]
    panel = frame.pivot(index="date", columns="asset", values="value").sort_index()
    panel.columns.name = None
    return panel.fillna(0.0)


def upsert_macro(history_dir: Path, rows: list[dict]) -> int:
    """B6: макро-сериалы (long: date/series/value), ключи [date, series]."""
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=["date", "series", "value"])
    frame["date"] = frame["date"].astype(str).str[:10]
    return _upsert(Path(history_dir) / MACRO, frame, ["date", "series"])


def load_macro_panel(history_dir: Path, *, end: str | None = None) -> pd.DataFrame:
    """Wide-панель макро (даты x серии: usdrub/brent/ust10y)."""
    path = Path(history_dir) / MACRO
    if not path.is_file():
        return pd.DataFrame()
    _require_pyarrow()
    frame = pd.read_parquet(path)
    frame["date"] = frame["date"].astype(str).str[:10]
    if end:
        frame = frame[frame["date"] <= end]
    panel = frame.pivot(index="date", columns="series", values="value").sort_index()
    panel.columns.name = None
    return panel


def upsert_earnings(history_dir: Path, rows: list[dict]) -> int:
    """B7: календарь отчётностей (date = дата отчёта), ключи [date, asset]."""
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=["date", "asset"])
    frame["date"] = frame["date"].astype(str).str[:10]
    return _upsert(Path(history_dir) / EARNINGS, frame, ["date", "asset"])


def load_earnings(history_dir: Path) -> list[dict]:
    """-> [{asset, date}] — все известные даты отчётностей."""
    path = Path(history_dir) / EARNINGS
    if not path.is_file():
        return []
    _require_pyarrow()
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")
