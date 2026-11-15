"""Telegram-модальность: публичный веб-превью t.me/s/<channel> без API-ключей.

Разметка t.me/s стабильна: сообщения идут блоками `div.tgme_widget_message`
с атрибутом data-post="<channel>/<id>"; текст — в `tgme_widget_message_text`
(без вложенных div, только inline-теги); время — `<time datetime="ISO">`.
Пагинация — параметр `?before=<id>` (20 сообщений на страницу).

Парсинг регэкспами по «плоской» повторяющейся структуре: сегмент сообщения =
от одного data-post до следующего; внутри сегмента текстовые блоки и время.
"""
from __future__ import annotations

import html as html_lib
import re
from typing import Any

import requests

from ..storage.raw import short_id
from .errors import ProviderError

_POST_RE = re.compile(r'data-post="([^"]+/(\d+))"')
_TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')
_TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
                      re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_PREVIEW_MARK = "tgme_widget_message"


def fetch_tg_raw(channel: str, *, pages: int = 1, timeout: float = 20.0,
                 session: requests.Session | None = None) -> bytes:
    """Страницы превью канала (до `pages`), склеенные в один документ."""
    client = session or requests.Session()
    chunks: list[bytes] = []
    before: str | None = None
    for page in range(max(pages, 1)):
        url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
        try:
            response = client.get(url, timeout=timeout, allow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            response.raise_for_status()
        except requests.RequestException as error:
            raise ProviderError(f"t.me/s/{channel}: {error}") from error
        body = response.content
        if _PREVIEW_MARK not in response.text:
            raise ProviderError(
                f"t.me/s/{channel}: превью недоступно (канал скрыт или отключён веб-превью)")
        chunks.append(body)
        if page + 1 < pages:
            ids = [int(m.group(2)) for m in _POST_RE.finditer(response.text)]
            if not ids:
                break
            before = str(min(ids))
    return b"\n".join(chunks)


def _clean_text(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_tg(payload: bytes, *, channel: str) -> list[dict[str, Any]]:
    """Сообщения канала -> стандартные news-слоты {id, title, text, link,
    published_at, source}. Порядок блоков = порядок на странице."""
    document = payload.decode("utf-8", errors="replace")
    matches = list(_POST_RE.finditer(document))
    items: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        segment = document[match.start():end]
        text_parts = [_clean_text(fragment) for fragment in _TEXT_RE.findall(segment)]
        text = " | ".join(part for part in text_parts if part)
        if not text:
            continue
        time_match = _TIME_RE.search(segment)
        post_number = match.group(2)
        items.append({
            "id": short_id(f"tg:{channel}|{match.group(1)}"),
            "title": text[:120],
            "text": text,
            "link": f"https://t.me/{channel}/{post_number}",
            "published_at": time_match.group(1) if time_match else None,
            "source": f"tg:{channel}",
        })
    return items
