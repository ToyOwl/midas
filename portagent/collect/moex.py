"""MOEX ISS API: дневные свечи TQBR и новости площадки sitenews.

Формат ISS: {"<block>": {"columns": [...], "data": [[...], ...]}} — колонки
парсим динамически, не завязываясь на порядок полей.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from .errors import ProviderError

ISS_BASE = "https://iss.moex.com/iss"


def _iss_get(url: str, params: dict, timeout: float, retries: int, session: requests.Session | None = None) -> bytes:
    """GET с ограниченными повторами; возвращает сырые байты ответа."""
    client = session or requests.Session()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url, params=params, timeout=timeout,
                                  headers={"User-Agent": "midas/0.1 (research)"})
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:  # сеть/HTTP-статус
            last = error
    raise ProviderError(f"MOEX ISS {url}: {last}") from last


def _block(payload: bytes, block: str) -> list[dict[str, Any]]:
    import json

    try:
        document = json.loads(payload.decode("utf-8"))
        section = document[block]
        columns = section["columns"]
        return [dict(zip(columns, row)) for row in section["data"]]
    except (ValueError, KeyError, TypeError) as error:
        raise ProviderError(f"MOEX ISS: неожиданный формат блока {block!r}: {error}") from error


def fetch_candles_raw(ticker: str, *, board: str, from_date: str, till_date: str,
                      timeout: float = 30.0, retries: int = 2,
                      session: requests.Session | None = None,
                      page_size: int = 500) -> bytes:
    """Дневные свечи за период. ISS отдаёт максимум 500 строк на запрос —
    поэтому цикл пагинации по параметру `start` до короткой страницы.

    Возвращаемый payload — склейка страниц в исходном ISS-формате candles
    (единственный блок, колонки первой страницы), что сохраняет контракт
    parse_candles и raw-архива.
    """
    import json as json_lib

    url = f"{ISS_BASE}/engines/stock/markets/shares/boards/{board}/securities/{ticker}/candles.json"
    rows: list[list] = []
    columns: list[str] | None = None
    start = 0
    while True:
        payload = _iss_get(url, {
            "from": from_date, "till": till_date, "interval": 24, "iss.meta": "off",
            "candles.columns": "begin,open,high,low,close,volume", "start": start,
        }, timeout, retries, session)
        section = json_lib.loads(payload.decode("utf-8"))["candles"]
        columns = columns or section["columns"]
        rows.extend(section["data"])
        if len(section["data"]) < page_size:
            break
        start += page_size
    combined = {"candles": {"columns": columns, "data": rows}}
    return json_lib.dumps(combined, ensure_ascii=False).encode("utf-8")


def parse_candles(payload: bytes) -> pd.DataFrame:
    """Свечи -> DataFrame[close, volume] с датой (YYYY-MM-DD) в индексе."""
    rows = _block(payload, "candles")
    if not rows:
        return pd.DataFrame(columns=["close", "volume"])
    frame = pd.DataFrame({
        "date": [str(row["begin"])[:10] for row in rows],
        "close": [float(row["close"]) for row in rows],
        "volume": [float(row["volume"]) for row in rows],
    })
    frame = frame.groupby("date", as_index=True).agg({"close": "last", "volume": "sum"})
    return frame.sort_index()


def fetch_sitenews_raw(*, limit: int = 100, timeout: float = 30.0, retries: int = 2,
                       session: requests.Session | None = None) -> bytes:
    return _iss_get(f"{ISS_BASE}/sitenews.json",
                    {"iss.meta": "off", "limit": limit, "start": 0}, timeout, retries, session)


def parse_sitenews(payload: bytes, *, lookback_hours: float, now: datetime | None = None) -> list[dict]:
    """Новости площадки -> список {id, title, text, published_at, source}.

    Отбрасываем всё старше окна lookback_hours; поле body присутствует не во
    всех версиях ISS — берём, что есть.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    items: list[dict] = []
    for row in _block(payload, "sitenews"):
        published_raw = str(row.get("published_at", ""))
        try:
            published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            if published.tzinfo is None:  # ISS отдаёт даты без смещения — считаем UTC
                published = published.replace(tzinfo=timezone.utc)
        except ValueError:
            published = None
        if published is not None and published < cutoff:
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        body = str(row.get("body") or "").strip()
        items.append({
            "id": f"moex-{row.get('id', title)}",
            "title": title,
            "text": f"{title}. {body}".strip(),
            "published_at": published.isoformat() if published else None,
            "source": "moex-sitenews",
        })
    return items
