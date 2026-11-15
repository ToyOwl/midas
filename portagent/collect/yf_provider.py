"""Провайдер Yahoo Finance (акции США) через опциональный пакет yfinance."""
from __future__ import annotations

import pandas as pd

from .errors import ProviderError


def fetch_us_candles(ticker: str, *, history_days: int, timeout_s: float = 30.0) -> tuple[pd.DataFrame, str]:
    """Дневные скорректированные котировки -> DataFrame[close, volume] + исходный CSV-текст.

    Возвращает (frame, csv_payload): CSV нужен для raw-архива (yfinance не даёт
    сырых байтов ответа).
    """
    try:
        import yfinance
    except ImportError as error:
        raise ProviderError("пакет yfinance не установлен: pip install '.[us]'") from error
    try:
        raw = yfinance.Ticker(ticker).history(period=f"{history_days}d", interval="1d",
                                              auto_adjust=True, timeout=timeout_s)
    except Exception as error:  # yfinance бросает разнородные исключения
        raise ProviderError(f"yfinance {ticker}: {error}") from error
    if raw is None or raw.empty:
        raise ProviderError(f"yfinance {ticker}: пустой ответ")
    frame = pd.DataFrame({
        "date": [ts.strftime("%Y-%m-%d") for ts in raw.index],
        "close": raw["Close"].astype(float).values,
        "volume": raw["Volume"].astype(float).values,
    })
    frame = frame.groupby("date", as_index=True).agg({"close": "last", "volume": "sum"})
    return frame.sort_index(), frame.to_csv()
