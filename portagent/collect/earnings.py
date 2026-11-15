"""B7: календарь отчётностей — событийный риск для трекера.

US: yfinance Ticker.get_earnings_dates (прошлые + ~2 будущих квартала,
будущие distinguished NaN-значениями EPS — нам важны сами даты).
RU: ручная выкладка data/manual/earnings/<date>/*.json (e-disclosure API нет).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ProviderError


def fetch_us_earnings(ticker: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """Даты отчётностей -> [{asset, date}] (прошлые и будущие)."""
    try:
        import yfinance
    except ImportError as error:
        raise ProviderError("yfinance не установлен") from error
    try:
        frame = yfinance.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception as error:
        raise ProviderError(f"yfinance earnings {ticker}: {error}") from error
    if frame is None or len(frame) == 0:
        return []
    return [{"asset": f"US:{ticker}", "date": stamp.strftime("%Y-%m-%d")}
            for stamp in frame.index]


def load_earnings_drops(manual_dir: Path, run_date: str) -> list[dict[str, Any]]:
    """data/manual/earnings/<date>/*.json -> [{asset, date}].

    Формат: список или {earnings: [{asset, date}]}; asset — полный
    идентификатор (MOEX:SBER), date — YYYY-MM-DD (для RU удобна дата
    закрытия реестра/раскрытия).
    """
    directory = Path(manual_dir) / run_date
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ProviderError(f"{path.name}: некорректный JSON: {error}") from error
        items = document.get("earnings", document) if isinstance(document, dict) else document
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("asset") and item.get("date"):
                rows.append({"asset": str(item["asset"]),
                             "date": str(item["date"])[:10]})
    return rows
