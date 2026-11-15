"""Структурированный разбор портфеля.

Таблицы считаются из замороженного отчёта. Текст разделов и замечания пишет
локальный критик; этот модуль только раскладывает их по фиксированным главам.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SECTION_TITLES = {
    "concentration": "Концентрация",
    "sentiment": "Сентимент против веса",
    "price": "Цена",
    "rare": "Трекер редких событий",
    "coverage": "Покрытие модальностей",
}
STANCE_RU = {"ok": "спокойно", "watch": "наблюдать", "risk": "риск"}
NEG_SENTIMENT = -0.2
POS_SENTIMENT = 0.2


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _book(report: dict[str, Any]) -> dict[str, Any]:
    weights = {asset: float(value) for asset, value in report.get("weights", {}).items()}
    cash = float((report.get("portfolio") or {}).get("cash_share") or 0.0)
    count = len(weights)
    equal = ((1.0 - cash) / count) if count else 0.0
    ordered = sorted(weights.items(), key=lambda item: -item[1])
    top3 = sum(value for _, value in ordered[:3])
    hhi = sum(value * value for value in weights.values()) + cash * cash
    return {
        "weights": weights,
        "ordered": ordered,
        "cash": cash,
        "count": count,
        "equal": equal,
        "top3": top3,
        "top3_share": (top3 / (1.0 - cash)) if cash < 1 else 0.0,
        "hhi": hhi,
        "effective_n": (1.0 / hhi) if hhi else 0.0,
        "above_equal": [(asset, value) for asset, value in ordered if value > equal + 1e-12],
    }


def _sentiment_flag(mean: float | None, news: int, weight: float, equal: float) -> str:
    if mean is None or news <= 0:
        return "нет новостей"
    if mean <= NEG_SENTIMENT and weight + 1e-12 >= equal:
        return "негатив при весе выше равной доли"
    if mean <= NEG_SENTIMENT:
        return "негатив"
    if mean >= POS_SENTIMENT:
        return "позитив"
    return "нейтрально"


def _sentiment_rows(report: dict[str, Any], book: dict[str, Any]) -> list[dict[str, Any]]:
    stats = {row["asset"]: row for row in report.get("sentiment_stats") or []}
    rows = []
    for asset, weight in book["ordered"]:
        stat = stats.get(asset)
        mean = float(stat["mean_sentiment"]) if stat else None
        news = int(stat["n_news"]) if stat else 0
        rows.append({
            "asset": asset,
            "weight": weight,
            "mean": mean,
            "news": news,
            "flag": _sentiment_flag(mean, news, weight, book["equal"]),
        })
    return rows


def _findings_by_section(critique: dict[str, Any]) -> dict[str, list[dict]]:
    grouped = {section_id: [] for section_id in SECTION_TITLES}
    grouped["other"] = []
    for finding in critique.get("findings") or []:
        section_id = finding.get("section")
        if section_id in grouped:
            grouped[section_id].append(finding)
        else:
            grouped["other"].append(finding)
    return grouped


def _review_block(lines: list[str], critique: dict[str, Any], section_id: str,
                 grouped: dict[str, list[dict]]) -> None:
    sections = {row["id"]: row for row in critique.get("sections") or [] if isinstance(row, dict)}
    note = sections.get(section_id)
    lines.append("")
    lines.append("### Оценка локальной модели")
    lines.append("")
    if note:
        stance = STANCE_RU.get(note.get("stance"), note.get("stance"))
        lines.append(f"**{stance}.** {note.get('fact', '')} "
                     f"(evidence: {', '.join(note.get('evidence_ids') or [])})")
    elif critique.get("failed"):
        lines.append(critique.get("summary") or "Рецензия локальной модели не выполнена.")
    else:
        lines.append("Модель не вернула оценку этого раздела.")
    for finding in grouped.get(section_id) or []:
        hypothesis = f" Гипотеза: {finding['hypothesis']}" if finding.get("hypothesis") else ""
        lines.append("")
        lines.append(f"- **[{finding['severity'].upper()}] {finding['title']}.** "
                     f"{finding['fact']} (evidence: {', '.join(finding['evidence_ids'])})."
                     f"{hypothesis}")


def render_analysis(report: dict[str, Any]) -> str:
    """Текст разбора в том же порядке глав, что и устные разборы по отчёту."""
    book = _book(report)
    critique = report.get("critique") or {}
    grouped = _findings_by_section(critique)
    fusion = report.get("fusion") or {}
    views = ", ".join(fusion.get("views_used") or ["price"])
    lines = [
        f"# Разбор портфеля на {report.get('date', '—')}",
        "",
        f"Режим {fusion.get('mode', '—')}, метод {fusion.get('method', '—')}, "
        f"виды: {views}. Таблицы посчитаны из замороженных весов HRP. "
        f"Оценки разделов написал локальный критик и веса не менял.",
        "",
        "## Паспорт",
        "",
        "| | |",
        "|---|---|",
        f"| Слияние | {fusion.get('note', '—')} |",
        f"| Активов | {book['count']} |",
        f"| Инвестировано | {_pct(1.0 - book['cash'])}, кэш {_pct(book['cash'])} |",
        f"| Равная доля одной бумаги | {_pct(book['equal'])} |",
        f"| Потолок веса | {report.get('portfolio', {}).get('max_weight', '—')} |",
    ]
    llm = report.get("llm") or {}
    if llm:
        lines.append(f"| LLM | вызовов {llm.get('llm_calls', 0)}, "
                     f"кэш {llm.get('cache_hits', 0)}, меток {llm.get('labels_total', 0)} |")
    legal = report.get("legal") or {}
    vision = report.get("vision") or {}
    tracker = report.get("tracker") or {}
    lines.append(f"| Суды / скриншоты | дел {legal.get('cases_total', 0)}, "
                 f"графиков {vision.get('charts_found', 0)} |")
    lines.append(f"| Overlay трекера | {'включён' if tracker.get('overlay') else 'выключен'} |")

    lines.extend(["", "## Концентрация", ""])
    lines.append(
        f"Индекс Херфиндаля по весам вместе с кэшем — {book['hhi']:.3f}. "
        f"Эффективное число позиций — {book['effective_n']:.1f} из {book['count'] + (1 if book['cash'] else 0)}. "
        f"Три крупнейшие бумаги — {_pct(book['top3'])} портфеля "
        f"({_pct(book['top3_share'])} инвестированной части)."
    )
    lines.append("")
    if book["above_equal"]:
        lines.append("| Актив | Вес | К равной доле |")
        lines.append("|---|---:|---:|")
        for asset, weight in book["above_equal"]:
            lines.append(f"| {asset} | {_pct(weight)} | {(weight - book['equal']) * 100:+.1f} п.п. |")
    else:
        lines.append("Ни одна бумага не выше равной доли.")
    _review_block(lines, critique, "concentration", grouped)

    lines.extend(["", "## Сентимент против веса", ""])
    lines.append("Негатив — средний сентимент не выше −0.20. "
                 "«Выше равной доли» — вес не ниже равной доли одной бумаги.")
    lines.append("")
    lines.append("| Актив | Вес | Сентимент | Новостей | Фон |")
    lines.append("|---|---:|---:|---:|---|")
    for row in _sentiment_rows(report, book):
        mean = "—" if row["mean"] is None else f"{row['mean']:+.2f}"
        lines.append(f"| {row['asset']} | {_pct(row['weight'])} | {mean} | {row['news']} | {row['flag']} |")
    _review_block(lines, critique, "sentiment", grouped)

    clusters = (report.get("portfolio") or {}).get("clusters") or {}
    if clusters:
        lines.extend(["", "## Кластеры", ""])
        for number, members in sorted(clusters.items(), key=lambda item: str(item[0])):
            names = ", ".join(members) if isinstance(members, list) else str(members)
            lines.append(f"- Кластер {number}: {names}")

    lines.extend(["", "## Цена", ""])
    movers = report.get("movers") or []
    if movers:
        lines.append("| Актив | 1д | 5д |")
        lines.append("|---|---:|---:|")
        for mover in movers:
            lines.append(f"| {mover['asset']} | {mover['ret_1d']:+.2%} | {mover['ret_5d']:+.2%} |")
    else:
        lines.append("Движений в отчёте нет.")
    _review_block(lines, critique, "price", grouped)

    lines.extend(["", "## Трекер редких событий", ""])
    rows = tracker.get("rows") or []
    lines.append(f"Порог предупреждения {tracker.get('warn_threshold', '—')}. "
                 f"Overlay на потолки весов {'включён' if tracker.get('overlay') else 'выключен'}.")
    warned = [row for row in rows if row.get("warned")]
    shown = warned or sorted(rows, key=lambda row: -row.get("score", 0))[:5]
    if shown:
        lines.append("")
        lines.append("| Актив | Скор | p_tail | Просадка | Предупреждение |")
        lines.append("|---|---:|---:|---:|:---:|")
        for row in shown:
            tail = "—" if row.get("p_tail") is None else f"{row['p_tail']:.2%}"
            lines.append(f"| {row['asset']} | {row['score']:.2f} | {tail} | "
                         f"{row.get('drawdown', 0):.0%} | {'да' if row.get('warned') else 'нет'} |")
    else:
        lines.append("")
        lines.append("Скоров трекера нет.")
    _review_block(lines, critique, "rare", grouped)

    lines.extend(["", "## Покрытие модальностей", ""])
    coverage = report.get("coverage") or {}
    frac = coverage.get("coverage_frac", coverage.get("covered_frac", 0)) or 0
    lines.append(f"Новостное покрытие {frac * 100:.0f}%. "
                 f"Судебное покрытие {(legal.get('coverage_frac') or 0) * 100:.0f}%. "
                 f"Скриншотов {vision.get('charts_found', 0)}.")
    _review_block(lines, critique, "coverage", grouped)

    agreement = report.get("agreement") or {}
    if agreement.get("enabled"):
        lines.extend(["", "## Согласованность LLM", ""])
        if agreement.get("error"):
            lines.append(f"Второй движок недоступен: {agreement['error'][:160]}")
        else:
            jac = agreement.get("jaccard_mean")
            sign = agreement.get("sign_agreement")
            delta = agreement.get("delta_mean_abs")
            lines.append(
                f"Два локальных движка разметили один поток: общих пар "
                f"{agreement.get('common_pairs', 0)}; Jaccard тикеров "
                f"{'—' if jac is None else f'{jac:.0%}'}; совпадение знака "
                f"{'—' if sign is None else f'{sign:.0%}'}; средний |Δ| сентимента "
                f"{'—' if delta is None else f'{delta:.2f}'}.")
            disagreements = agreement.get("disagreements") or []
            if disagreements:
                lines.append(f"Сильные расхождения (|Δ| >= {agreement.get('delta_alert', 0.4)}): "
                             f"{agreement.get('n_disagreements', len(disagreements))}. Топ:")
                for row in disagreements[:5]:
                    lines.append(f"- {row['ticker']}: {row['sentiment_a']:+.2f} vs "
                                 f"{row['sentiment_b']:+.2f} (Δ {row['delta']:+.2f}) — "
                                 f"«{row['quote_a'][:80]}» / «{row['quote_b'][:80]}»")
            else:
                lines.append("Сильных расхождений нет.")

    earnings_calendar = report.get("earnings_calendar") or []
    flagged = [row for row in (report.get("tracker") or {}).get("rows", [])
               if row.get("earnings_flag")]
    if earnings_calendar or flagged:
        lines.extend(["", "## Отчётности (событийный риск)", ""])
        if flagged:
            lines.append("Флаг трекера (отчётность в ближайшие 5 дней): "
                         + ", ".join(row["asset"] for row in flagged) + ".")
        if earnings_calendar:
            lines.append("Ближайшие даты: " + "; ".join(
                f"{row['asset']} — {row['date']}" for row in earnings_calendar[:8]) + ".")

    materials = report.get("materials") or {}
    if materials.get("applied"):
        lines.extend(["", "## Утверждённые изображения", "",
            f"Материалов: {len(materials['used_materials'])}. Влияние на расчёт: {materials['influence']:.0%}."])
        for asset, reasons in materials.get("reasons", {}).items():
            lines.append(f"- {asset}: " + "; ".join(r["text"] for r in reasons))
    elif materials.get("error"):
        lines.extend(["", "## Изображения", "", "Материалы не учтены: " + materials["error"]])
    settlement = report.get("settlement") or {}
    if settlement:
        lines.extend(["", "## Прогноз и settlement", ""])
        if settlement.get("settled"):
            lines.append(f"Закрыто прогнозов в этом прогоне: "
                         f"{', '.join(settlement['settled'])}.")
        if settlement.get("pending"):
            lines.append(f"Ожидают горизонта: {', '.join(settlement['pending'][:8])}.")
        if settlement.get("forecast"):
            lines.append("Прогноз этого дня заморожен (immutable) в data/forecasts.")
        if settlement.get("error"):
            lines.append(f"Сбой settlement: {settlement['error'][:160]}")

    lines.extend(["", "## Сводка локальной модели", ""])
    lines.append(critique.get("summary") or "—")
    if critique.get("findings_truncated"):
        lines.append("")
        lines.append("Замечаний было больше шести: в разбор вошли первые шесть.")
    other = grouped.get("other") or []
    if other:
        lines.append("")
        lines.append("Замечания без раздела:")
        for finding in other:
            lines.append(f"- {finding.get('title')}: {finding.get('fact')}")
    lines.append("")
    return "\n".join(lines)


def write_analysis(directory: Path, report: dict[str, Any]) -> Path:
    path = Path(directory) / "analysis.md"
    path.write_text(render_analysis(report), encoding="utf-8")
    return path
