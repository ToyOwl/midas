"""Парсеры коллекторов на fixture-байтах (без сети)."""
import json
from datetime import datetime, timezone

import pytest

from portagent.collect import moex, news as news_collector
from portagent.collect.errors import ProviderError
from portagent.collect.moex import parse_candles, parse_sitenews
from portagent.collect.news import (
    dedupe, filter_recent, google_news_url, matched_assets, parse_rss,
    select_for_budget, ticker_query,
)

ISS_CANDLES = json.dumps({"candles": {
    "columns": ["begin", "open", "high", "low", "close", "volume"],
    "data": [
        ["2026-11-09 00:00:00", 100.0, 101.0, 99.5, 100.5, 1000.0],
        ["2026-11-10 00:00:00", 100.5, 102.0, 100.0, 101.5, 1500.0],
        ["2026-11-10 00:00:00", 101.5, 103.0, 101.0, 102.5, 500.0],  # вечерняя сессия -> агрегация
    ]}}).encode("utf-8")

SITENEWS = json.dumps({"sitenews": {
    "columns": ["id", "title", "published_at", "body"],
    "data": [
        [1, "Совет директоров поднял лимит", "2026-11-11T10:00:00+00:00", "Текст новости один."],
        [2, "Старая новость", "2026-10-01T10:00:00+00:00", "Должна отфильтроваться."],
        [3, "", "2026-11-11T11:00:00+00:00", "Без заголовка — отбрасывается."],
        [4, "Naive-дата без смещения", "2026-11-11 09:00:00", "ISS иногда отдаёт без зоны."],
    ]}}).encode("utf-8")

RSS_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>Test</title>
<item><title>Sber posts record profit</title><link>https://x/1</link>
<description>Record profit beats forecasts</description>
<pubDate>Wed, 11 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>Old item</title><link>https://x/2</link>
<description>Too old</description>
<pubDate>Thu, 01 Jan 2026 09:00:00 GMT</pubDate></item>
<item><title>Без даты</title><link>https://x/3</link><description>Kept</description></item>
</channel></rss>""".encode("utf-8")


def test_parse_candles_aggregates_same_day():
    frame = parse_candles(ISS_CANDLES)
    assert list(frame.index) == ["2026-11-09", "2026-11-10"]
    assert frame.loc["2026-11-10", "close"] == pytest.approx(102.5)
    assert frame.loc["2026-11-10", "volume"] == pytest.approx(2000.0)


def test_parse_candles_empty_block():
    assert parse_candles(json.dumps({"candles": {"columns": ["begin"], "data": []}}).encode()).empty


def test_parse_candles_bad_payload():
    with pytest.raises(ProviderError):
        parse_candles(b"not json at all")


def test_parse_sitenews_filters_by_window():
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    items = parse_sitenews(SITENEWS, lookback_hours=24, now=now)
    ids = [item["id"] for item in items]
    assert ids == ["moex-1", "moex-4"]  # свежие остаются, naive-дата трактуется как UTC
    assert "Текст новости один" in items[0]["text"]


def test_parse_rss_and_filter_recent():
    items = parse_rss(RSS_XML, source="test")
    assert len(items) == 3
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    kept = filter_recent(items, lookback_hours=48, now=now)
    titles = [item["title"] for item in kept]
    assert "Sber posts record profit" in titles
    assert "Без даты" in titles          # без даты — оставляем
    assert "Old item" not in titles      # старое отсекается


def test_dedupe_by_id():
    first = parse_rss(RSS_XML, source="test")
    assert len(dedupe(first + first)) == 3


def _item(title: str, text: str = "") -> dict:
    return {"id": title, "title": title, "text": text or title}


def test_select_keeps_ticker_mentions_ahead_of_flood():
    flood = [_item(f"макрос {i}") for i in range(30)]
    hit = _item("Сбербанк отчитался о прибыли")
    chosen = select_for_budget(
        flood + [hit], universe=["MOEX:SBER"],
        aliases={"SBER": ["Сбербанк"]}, limit=5, per_asset=5)
    assert hit in chosen
    assert len(chosen) == 5


def test_select_round_robin_before_extra_mentions():
    items = [_item(f"AAPL note {i}") for i in range(6)] + [_item("MSFT note 1"), _item("MSFT note 2")]
    chosen = select_for_budget(
        items, universe=["US:AAPL", "US:MSFT"], aliases={}, limit=5, per_asset=5)
    titles = [item["title"] for item in chosen]
    assert titles.count("MSFT note 1") + titles.count("MSFT note 2") == 2
    assert sum(title.startswith("AAPL") for title in titles) == 3
    assert titles[0].startswith("AAPL")
    assert titles[1].startswith("MSFT")


def test_select_without_matches_preserves_order():
    items = [_item("один"), _item("два"), _item("три")]
    chosen = select_for_budget(items, universe=["MOEX:SBER"], aliases={}, limit=2, per_asset=1)
    assert [item["title"] for item in chosen] == ["один", "два"]


def test_match_is_whole_word():
    universe = ["US:AAPL", "MOEX:SBER", "US:META"]
    aliases = {"AAPL": ["Apple"], "SBER": ["Сбербанк"], "META": ["Meta Platforms"]}
    assert matched_assets(_item("pineapple harvest"), universe, aliases) == []
    assert matched_assets(_item("SBERBANK rally"), universe, aliases) == []
    assert matched_assets(_item("Apple raises prices"), universe, aliases) == ["US:AAPL"]
    assert matched_assets(_item("бумага $SBER выросла"), universe, aliases) == ["MOEX:SBER"]


def test_rss_html_summary_does_not_match_google_alias():
    raw = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>gn</title>
<item><title>Сбербанк поднял прогноз</title><link>https://news.google.com/rss/articles/abc</link>
<description><![CDATA[<a href="https://news.google.com/rss/articles/abc">Сбербанк поднял прогноз</a>]]></description>
</item></channel></rss>""".encode("utf-8")
    item = parse_rss(raw, source="gn")[0]
    assert "news.google.com" not in item["text"]
    assert "<a" not in item["text"]
    assert matched_assets(item, ["US:GOOGL", "MOEX:SBER"],
                          {"GOOGL": ["Google", "Alphabet"], "SBER": ["Сбербанк"]}) == ["MOEX:SBER"]


def test_ticker_query_quotes_multiword_names():
    query = ticker_query("JNJ", ["Johnson & Johnson", "JNJ"])
    assert query == 'JNJ OR "Johnson & Johnson"'
    url = google_news_url(query, hl="en-US", gl="US", ceid="US:en")
    assert url.startswith("https://news.google.com/rss/search?")
    assert "Johnson+%26+Johnson" in url or "Johnson+%26+Johnson".lower() in url.lower()
    assert "hl=en-US" in url


def test_fetch_raw_network_error_wrapped(requests_mock=None):
    import requests

    class BadSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("нет сети")

    with pytest.raises(ProviderError):
        moex._iss_get("https://iss.moex.com/iss/x", {}, timeout=1, retries=1, session=BadSession())


# --- Telegram (t.me/s) ---

TG_HTML = """<html><body><div class="tgme_widget_wrap">
<div class="tgme_widget_message_wrap"><div class="tgme_widget_message text-not_supported" data-post="bcs_express/1001">
<div class="tgme_widget_message_bubble"><div class="tgme_widget_message_text js-message_text" dir="auto">Сбербанк отчитался о рекордной прибыли.<br/>Дивиденды повышены.</div>
<a class="tgme_widget_message_date"><time datetime="2026-11-12T07:00:00+00:00"></time></a></div></div></div>
<div class="tgme_widget_message_wrap"><div class="tgme_widget_message text-not_supported" data-post="bcs_express/1002">
<div class="tgme_widget_message_bubble"><div class="tgme_widget_message_text js-message_reply_text" dir="auto">Цитата: Нефть дешевеет.</div>
<div class="tgme_widget_message_text js-message_text" dir="auto">Газпром &amp; ЛУКОЙЛ снижают добычу из-за квот</div>
<a class="tgme_widget_message_date"><time datetime="2026-11-12T08:00:00+00:00"></time></a></div></div></div>
<div class="tgme_widget_message_wrap"><div class="tgme_widget_message text-not_supported" data-post="bcs_express/1003">
<div class="tgme_widget_message_bubble"><a class="tgme_widget_message_date"><time datetime="2026-11-12T09:00:00+00:00"></time></a></div></div></div>
</div></body></html>""".encode("utf-8")


def test_parse_tg_messages_text_and_time():
    from portagent.collect.tg import parse_tg

    items = parse_tg(TG_HTML, channel="bcs_express")
    assert len(items) == 2  # сообщение без текста отброшено
    first, second = items
    assert first["published_at"] == "2026-11-12T07:00:00+00:00"
    assert "рекордной прибыли" in first["text"]
    assert "Дивиденды" in first["text"]            # <br> -> перенос
    assert first["link"] == "https://t.me/bcs_express/1001"
    assert first["source"] == "tg:bcs_express"
    # цитата ответа + основной текст склеены, сущность &amp; разобрана
    assert "Цитата: Нефть дешевеет." in second["text"]
    assert "Газпром & ЛУКОЙЛ" in second["text"]


def test_parse_tg_ids_stable():
    from portagent.collect.tg import parse_tg

    ids = [item["id"] for item in parse_tg(TG_HTML, channel="bcs_express")]
    assert len(set(ids)) == 2


def test_fetch_tg_detects_disabled_preview():
    import requests

    from portagent.collect.tg import fetch_tg_raw

    class InviteSession:
        def get(self, *args, **kwargs):
            return requests.Response()

    response = requests.Response()
    response.status_code = 200
    response._content = b"<html>invite page without messages</html>"
    response.encoding = "utf-8"
    InviteSession.get = lambda self, *a, **k: response
    with pytest.raises(ProviderError, match="превью недоступно"):
        fetch_tg_raw("nope", session=InviteSession())
