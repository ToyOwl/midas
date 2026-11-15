"""B6: макро-сериалы для трекера — USDRUB (MOEX ISS SELT), Brent и US10Y
(yfinance). Не вид расстояний: из них строится признак macro_stress
(|z макро-движка| x |бета актива| к нему) в tracker/rare_events.
"""
from __future__ import annotations

import pandas as pd
import requests

from .errors import ProviderError
from .moex import ISS_BASE, _iss_get

USDRUB_CANDIDATES = ("USD000UTSTOM", "USDRUB_TOM")  # классический код и новый


def fetch_usdrub(*, from_date: str, till_date: str, secid: str = "USD000UTSTOM",
                 timeout: float = 30.0, retries: int = 2,
                 session: requests.Session | None = None,
                 page_size: int = 500) -> pd.DataFrame:
    """Дневные свечи USDRUB -> DataFrame[value] с датой в индексе.

    Пагинация по `start` (лимит ISS 500 строк на страницу — без него
    история обрывается). Пробует заданный secid, затем известные
    альтернативы (проверено 2026-11-13: USD000UTSTOM отдаёт данные).
    """
    import json as json_lib

    for candidate in (secid, *[c for c in USDRUB_CANDIDATES if c != secid]):
        url = (f"{ISS_BASE}/engines/currency/markets/selt/securities/"
               f"{candidate}/candles.json")
        rows: list[dict] = []
        start = 0
        try:
            while True:
                payload = _iss_get(url, {"from": from_date, "till": till_date,
                                         "interval": 24, "iss.meta": "off",
                                         "start": start},
                                   timeout, retries, session)
                block = json_lib.loads(payload.decode("utf-8"))["candles"]
                page = [dict(zip(block["columns"], row)) for row in block["data"]]
                rows.extend(page)
                if len(page) < page_size:
                    break
                start += page_size
        except (ProviderError, KeyError, ValueError):
            continue
        if not rows:
            continue
        frame = pd.DataFrame({
            "date": [str(row.get("begin", ""))[:10] for row in rows],
            "value": [float(row["close"]) for row in rows],
        })
        return frame.groupby("date", as_index=True)["value"].last().sort_index().to_frame()
    raise ProviderError("USDRUB: свечи недоступны ни по одному коду SELT")


def fetch_yf_series(symbol: str, *, history_days: int = 1825,
                    timeout_s: float = 30.0) -> pd.DataFrame:
    """Close-сериал yfinance (BZ=F, ^TNX, ...) -> DataFrame[value] по датам."""
    try:
        import yfinance
    except ImportError as error:
        raise ProviderError("yfinance не установлен") from error
    try:
        raw = yfinance.Ticker(symbol).history(period=f"{history_days}d", interval="1d",
                                              timeout=timeout_s)
    except Exception as error:
        raise ProviderError(f"yfinance {symbol}: {error}") from error
    if raw is None or raw.empty:
        raise ProviderError(f"yfinance {symbol}: пустой ответ")
    frame = pd.DataFrame({"date": [ts.strftime("%Y-%m-%d") for ts in raw.index],
                          "value": raw["Close"].astype(float).values})
    return frame.groupby("date", as_index=True)["value"].last().sort_index().to_frame()
