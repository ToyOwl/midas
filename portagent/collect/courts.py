"""Арбитражные суды (kad.arbitr.ru): модальность судебного давления.

Публичного API у картотеки нет: оператор выкладывает
экспорт поиска в data/manual/courts/<date>/*.json (инструкция в README).
Дела маппятся на тикеры юниверса по алиасам из конфига courts.aliases,
затем локальная LLM оценивает severity влияния на актив (с тем же грандингом:
quote — точная цитата из текста дела). Живой поиск — задокументированная
точка расширения.
"""
from __future__ import annotations

import json
from datetime import date as date_type
from pathlib import Path
from typing import Any

from ..storage.raw import short_id
from .errors import ProviderError

# Судебное давление затухает медленнее новостного шума
LEGAL_HALFLIFE_DAYS = 20.0

CASES_SYSTEM_PROMPT = (
    "Ты — юридический аналитик инвестиционного портфеля. Тебе дают список "
    "судебных дел (арбитражные суды РФ) и список идентификаторов активов. "
    "Для КАЖДОГО дела верни строгое JSON. Правила:\n"
    "1) tickers — только идентификаторы из предоставленного списка, которых "
    "касается дело; пустой список допустим.\n"
    "2) sentiment — влияние исхода/сущности дела на актив: от -1.0 (банкротство, "
    "отзыв лицензии, крупный проигрыш с серьёзными выплатами) до 1.0 (выигрыш, "
    "снятие претензий), 0.0 — незначимое/нейтральное дело.\n"
    "3) confidence — уверенность от 0.0 до 1.0.\n"
    "4) quote — точная цитата из текста дела (не короче 10 символов), "
    "подтверждающая оценку. Не выдумывай цитаты.\n"
    "Не добавляй активы вне списка, не давай юридических заключений. "
    "Вывод — только JSON без пояснений."
)


def fetch_courts_live() -> None:
    """Точка расширения: живой поиск kad.arbitr.ru (см. README).

    Картотека закрывается от автоматических клиентов капчей/403 без
    документированного API — поэтому основной канал это ручная выкладка.
    """
    raise ProviderError(
        "живой поиск kad.arbitr.ru не поддерживается — выложите экспорт в "
        "data/manual/courts/<date>/*.json (см. README)")


def load_court_drops(manual_dir: Path, run_date: str) -> list[dict[str, Any]]:
    """Читает data/manual/courts/<date>/*.json -> список кейсов.

    Формат файла: список кейсов или {"cases": [...]}; кейс:
    {case_number, date, type, plaintiff, defendant, summary} (summary опционален).
    """
    directory = Path(manual_dir) / run_date
    if not directory.is_dir():
        return []
    cases: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ProviderError(f"{path.name}: некорректный JSON: {error}") from error
        rows = document.get("cases", document) if isinstance(document, dict) else document
        if not isinstance(rows, list):
            raise ProviderError(f"{path.name}: ожидается список кейсов или {{cases: [...]}}")
        cases.extend(row for row in rows if isinstance(row, dict))
    return cases


def _match_tickers(haystack: str, aliases: dict[str, list[str]],
                   universe: list[str]) -> list[str]:
    lowered = haystack.lower()
    matched = []
    for asset in universe:
        ticker = asset.split(":", 1)[1]
        for alias in aliases.get(ticker, []):
            if alias.lower() in lowered:
                matched.append(asset)
                break
    return matched


def cases_to_items(cases: list[dict], *, aliases: dict[str, list[str]],
                   universe: list[str]) -> tuple[list[dict], dict[str, int]]:
    """Кейсы -> news-слоты для LLM (только дела, задевающие юниверс) + счётчики.

    Алиасное соответствие детерминировано и подсказывается LLM в тексте;
    итоговые tickers всё равно валидируются по юниверсу.
    """
    items: list[dict] = []
    matched_counts: dict[str, int] = {asset: 0 for asset in universe}
    for case in cases:
        number = str(case.get("case_number", "")).strip()
        if not number:
            continue
        plaintiff = str(case.get("plaintiff", "")).strip()
        defendant = str(case.get("defendant", "")).strip()
        summary = str(case.get("summary", "")).strip()
        kind = str(case.get("type", "Судебное дело")).strip()
        when = str(case.get("date", "")).strip()
        haystack = " ".join([plaintiff, defendant, summary, kind])
        tickers = _match_tickers(haystack, aliases, universe)
        if not tickers:
            continue
        for asset in tickers:
            matched_counts[asset] += 1
        text = (f"{kind}. Дело {number} от {when or 'дата не указана'}. "
                f"Истец: {plaintiff or 'не указан'}. Ответчик: {defendant or 'не указан'}. "
                f"{summary} Затронутые активы: {', '.join(tickers)}.")
        items.append({
            "id": short_id(f"court|{number}|{when}"),
            "title": f"{kind}: {number}",
            "text": text,
            "link": f"https://kad.arbitr.ru/Card?number={number}",
            "published_at": _iso_date(when),
            "source": "courts",
        })
    return items, matched_counts


def _iso_date(value: str) -> str | None:
    if not value:
        return None
    try:
        return date_type.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None
