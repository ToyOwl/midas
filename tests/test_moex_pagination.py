"""Пагинация свечей MOEX ISS: лимит 500 строк на запрос -> склейка по start."""
import json

import pytest

from portagent.collect import moex
from portagent.collect.errors import ProviderError


def _page(rows: list) -> bytes:
    return json.dumps({"candles": {
        "columns": ["begin", "open", "high", "low", "close", "volume"],
        "data": rows}}).encode("utf-8")


def _rows(start_day: int, count: int):
    from datetime import date, timedelta

    base = date(2026, 1, 1)
    return [[(base + timedelta(days=start_day + i)).isoformat() + " 00:00:00",
             1, 1, 1, float(start_day + i), 10.0] for i in range(count)]


class PagedSession:
    """Отдаёт страницы по 500 строк; третья — короткая (конец выдачи)."""

    def __init__(self):
        self.pages = [_rows(0, 500), _rows(500, 500), _rows(1000, 37)]
        self.calls = []

    def get(self, url, params=None, **kwargs):
        self.calls.append(params.get("start", 0))
        return FakeResponse(self.pages[len(self.calls) - 1])


class FakeResponse:
    def __init__(self, rows):
        self.content = _page(rows)
        self.status_code = 200

    def raise_for_status(self):
        pass


def test_fetch_candles_paginates_and_stitches():
    session = PagedSession()
    payload = moex.fetch_candles_raw("SBER", board="TQBR", from_date="2026-01-01",
                                     till_date="2026-12-31", session=session,
                                     page_size=500)
    frame = moex.parse_candles(payload)
    assert session.calls == [0, 500, 1000]              # три страницы
    assert len(frame) == 1037                            # 500 + 500 + 37 без дублей


def test_fetch_candles_single_page_stops():
    class OnePage(PagedSession):
        def __init__(self):
            super().__init__()
            self.pages = [_rows(0, 120)]

    session = OnePage()
    payload = moex.fetch_candles_raw("SBER", board="TQBR", from_date="2026-01-01",
                                     till_date="2026-12-31", session=session,
                                     page_size=500)
    assert session.calls == [0]
    assert len(moex.parse_candles(payload)) == 120


def test_fetch_candles_network_error_still_wrapped():
    import requests

    class BadSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("нет сети")

    with pytest.raises(ProviderError):
        moex.fetch_candles_raw("SBER", board="TQBR", from_date="2026-01-01",
                               till_date="2026-12-31", session=BadSession())
