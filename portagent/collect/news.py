"""RSS-коллектор: fetch байтов через requests (чтобы raw-архив хранил именно
ответ ленты) + разбор feedparser'ом в доменные элементы новостей.

Отбор в бюджет LLM — детерминированный: сначала round-robin по активам
юниверса (упоминание тикера или алиаса), затем прочие совпадения, и только
потом новости без привязки к юниверсу.
"""
from __future__ import annotations

import html as html_lib
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import feedparser
import requests

from ..storage.raw import short_id
from .errors import ProviderError


def fetch_rss_raw(url: str, *, timeout: float = 20.0, session: requests.Session | None = None) -> bytes:
    client = session or requests.Session()
    try:
        response = client.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Midas research RSS reader)"})
        response.raise_for_status()
        return response.content
    except requests.RequestException as error:
        raise ProviderError(f"RSS {url}: {error}") from error


def _plain_text(fragment: str) -> str:
    """Текст без HTML. Иначе URL вроде news.google.com попадает в текст и
    алиас «Google» совпадает с каждой карточкой Google News."""
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_rss(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    """Элементы ленты -> {id, title, text, link, published_at, source}.

    Новостной идентификатор — SHA(title|link): стабилен между прогонами и
    не зависит от порядка элементов.
    """
    parsed = feedparser.parse(payload)
    items: list[dict[str, Any]] = []
    for entry in parsed.entries:
        title = _plain_text(entry.get("title") or "")
        if not title:
            continue
        link = (entry.get("link") or "").strip()
        summary = _plain_text(entry.get("summary") or "")
        published = None
        for attribute in ("published_parsed", "updated_parsed"):
            stamp = entry.get(attribute)
            if stamp:
                published = datetime.fromtimestamp(
                    _time_to_epoch(stamp), tz=timezone.utc).isoformat()
                break
        items.append({
            "id": short_id(f"{title}|{link}"),
            "title": title,
            "text": f"{title}. {summary}".strip(),
            "link": link,
            "published_at": published,
            "source": source,
        })
    return items


def _time_to_epoch(stamp) -> float:
    import calendar

    return calendar.timegm(tuple(stamp)[:6])


def filter_recent(items: list[dict], *, lookback_hours: float,
                  now: datetime | None = None) -> list[dict]:
    """Оставляем свежие элементы; без даты публикации — оставляем (лучше
    лишний элемент на разметку, чем потерянная новость)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    kept: list[dict] = []
    for item in items:
        if item.get("published_at") is None:
            kept.append(item)
            continue
        try:
            published = datetime.fromisoformat(item["published_at"])
        except ValueError:
            kept.append(item)
            continue
        if published >= cutoff:
            kept.append(item)
    return kept


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        unique.append(item)
    return unique


def ticker_query(ticker: str, aliases: list[str]) -> str:
    """Поисковый запрос Google News: тикер и алиасы через OR.

    Многословные имена и имена с «&» берутся в кавычки, чтобы «Johnson &
    Johnson» не распадался на два независимых термина.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for term in [ticker, *aliases]:
        cleaned = term.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        if any(ch.isspace() or ch == "&" for ch in cleaned):
            terms.append(f'"{cleaned}"')
        else:
            terms.append(cleaned)
    return " OR ".join(terms)


def google_news_url(query: str, *, hl: str, gl: str, ceid: str) -> str:
    params = urlencode({"q": query, "hl": hl, "gl": gl, "ceid": ceid})
    return f"https://news.google.com/rss/search?{params}"


def _matchers(universe: list[str], aliases: dict[str, list[str]]) -> list[tuple[str, re.Pattern[str]]]:
    """Актив -> шаблон целого слова. Короткий тикер не совпадает с префиксом
    более длинного слова («SBER» не матчит «SBERBANK», «Apple» — «pineapple»)."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for asset in universe:
        ticker = asset.split(":", 1)[-1]
        terms = [ticker, *aliases.get(ticker, [])]
        seen: set[str] = set()
        for term in terms:
            cleaned = term.strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            pattern = re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)", re.IGNORECASE)
            compiled.append((asset, pattern))
    return compiled


def matched_assets(item: dict, universe: list[str], aliases: dict[str, list[str]],
                   *, matchers: list[tuple[str, re.Pattern[str]]] | None = None) -> list[str]:
    """Активы юниверса, упомянутые в заголовке или тексте. Порядок — порядок юниверса."""
    text = f"{item.get('title', '')} {item.get('text', '')}"
    compiled = matchers if matchers is not None else _matchers(universe, aliases)
    found: list[str] = []
    seen: set[str] = set()
    for asset, pattern in compiled:
        if asset in seen:
            continue
        if pattern.search(text):
            seen.add(asset)
            found.append(asset)
    return found


def select_for_budget(items: list[dict], *, universe: list[str],
                      aliases: dict[str, list[str]], limit: int,
                      per_asset: int) -> list[dict]:
    """Усекает ленту до `limit`, не отдавая бюджет общим новостям раньше тикеров.

    Фаза A — round-robin: каждый актив получает до quota своих новостей,
    прежде чем кто-то получит следующую. quota = min(per_asset, limit // N),
    чтобы хвост юниверса не остался без слотов при большом per_asset.
    Фаза B — остальные новости с упоминанием юниверса, в исходном порядке.
    Фаза C — новости без упоминания, в исходном порядке.
    """
    if limit <= 0 or not items:
        return []
    compiled = _matchers(universe, aliases)
    hits = [matched_assets(item, universe, aliases, matchers=compiled) for item in items]
    buckets: dict[str, list[int]] = {asset: [] for asset in universe}
    for index, item_hits in enumerate(hits):
        for asset in item_hits:
            buckets[asset].append(index)

    n_assets = len(universe)
    if n_assets and limit >= n_assets:
        quota = min(per_asset, limit // n_assets)
    else:
        quota = max(per_asset, 1)

    selected: list[int] = []
    used: set[int] = set()

    def _take(index: int) -> bool:
        if index in used or len(selected) >= limit:
            return False
        used.add(index)
        selected.append(index)
        return True

    for _round in range(quota):
        if len(selected) >= limit:
            break
        for asset in universe:
            for index in buckets.get(asset, []):
                if _take(index):
                    break
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        for index, item_hits in enumerate(hits):
            if item_hits and _take(index) and len(selected) >= limit:
                break
    if len(selected) < limit:
        for index in range(len(items)):
            if _take(index) and len(selected) >= limit:
                break
    return [items[index] for index in selected]
