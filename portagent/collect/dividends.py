"""B5: дивиденды.

MOEX: ISS-эндпоинта дивидендов в текущем деплое нет (проверено 2026-11-13:
/securities/{t}/dividends отдаёт generic-описание инструмента) — источник
ручная выкладка оператора data/manual/dividends/*.json.
US: yfinance Ticker.dividends — хранится для отчётности; в total return
НЕ подмешивается (yfinance уже отдаёт auto_adjust=True котировки).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ProviderError


def load_dividend_drops(manual_dir: Path) -> list[dict[str, Any]]:
    """data/manual/dividends/*.json -> [{asset, date, value}].

    Формат файла: список строк или {dividends: [...]}.
    """
    directory = Path(manual_dir)
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ProviderError(f"{path.name}: некорректный JSON: {error}") from error
        items = document.get("dividends", document) if isinstance(document, dict) else document
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            asset, date, value = item.get("asset"), item.get("date"), item.get("value")
            if asset and date and isinstance(value, (int, float)):
                rows.append({"asset": str(asset), "date": str(date)[:10],
                             "value": float(value)})
    return rows


def fetch_us_dividends(ticker: str) -> list[dict[str, Any]]:
    """yfinance-дивиденды US (для истории/отчёта; в доходность не мешаются)."""
    try:
        import yfinance
    except ImportError as error:
        raise ProviderError("yfinance не установлен") from error
    try:
        series = yfinance.Ticker(ticker).dividends
    except Exception as error:
        raise ProviderError(f"yfinance dividends {ticker}: {error}") from error
    if series is None or series.empty:
        return []
    return [{"asset": f"US:{ticker}", "date": stamp.strftime("%Y-%m-%d"),
             "value": float(value)} for stamp, value in series.items()]
