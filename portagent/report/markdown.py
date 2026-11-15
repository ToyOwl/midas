"""Человекочитаемый markdown-отчёт + машиночитаемые portfolio.json / weights.csv."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import render_analysis, write_analysis


def render_backtest(result: dict[str, Any]) -> str:
    """Markdown-отчёт walk-forward: три стратегии и калибровка трекера."""
    lines = [
        f"# Бэктест {result['start']} — {result['end']}",
        "",
        f"Ребаланс каждые {result['rebalance_days']} торговых дней, "
        f"дат ребаланса: {result['n_rebalances']}, активов: {result['n_assets']}.",
        "Веса на дату t используют только наблюдения до t. "
        "Equal-weight держит ту же долю кэша, что и HRP. Overlay трекера на веса не накладывается.",
        "",
        "| Стратегия | До издержек | После издержек | CAGR нетто | Волатильность | Просадка | Оборот |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    titles = {"fused": "HRP fused", "price_only": "HRP price-only", "equal_weight": "Equal-weight"}
    for key, title in titles.items():
        metrics = result["strategies"][key]
        gross = result.get("gross_strategies", result["strategies"])[key]
        lines.append(
            f"| {title} | {gross['total_return']:+.2%} | {metrics['total_return']:+.2%} | {metrics['cagr']:+.2%} | "
            f"{metrics['ann_vol']:.2%} | {metrics['max_drawdown']:.2%} | {metrics['turnover']:.2%} |")
    costs = result.get("cost_model")
    if costs:
        lines += ["", "## Издержки и исполнение", "",
                  f"Комиссия: {costs['commission_bps']:g} б.п. "
                  f"({costs['commission_bps'] / 100:.3f}%); полный спред: "
                  f"{costs['spread_bps']:g} б.п. ({costs['spread_bps'] / 100:.3f}%). "
                  "На каждую покупку и продажу списываются комиссия и половина спреда.",
                  "Исполнение на закрытии даты ребаланса. Включён первый вход из CASH; "
                  "принудительной ликвидации в конце нет. Между ребалансами позиции "
                  "удерживаются, веса меняются вслед за ценами. Дивиденды моделируются "
                  "реинвестированными в тот же актив, доходность CASH — нулевая.",
                  "Оборот — средняя половина суммы покупок и продаж относительно капитала "
                  "до сделки, без первого входа. Первая дата доходности содержит стоимость входа.",
                  "", "| Стратегия | Комиссии / начальный капитал | Спред / начальный капитал | Потеря доходности, п.п. |",
                  "|---|---:|---:|---:|"]
        for key, title in titles.items():
            m = result["strategies"][key]
            lines.append(f"| {title} | {m['commission_paid']:.3%} | {m['spread_paid']:.3%} | "
                         f"{100 * m['cost_drag']:.3f} |")
    tracker = result.get("tracker") or {}
    hit = tracker.get("hit_rate")
    hit_text = "н/д" if hit is None else f"{hit:.1%}"
    brier = tracker.get("brier")
    brier_text = "н/д" if brier is None else f"{brier:.3f}"
    lines.extend([
        "",
        "## Трекер редких событий",
        "",
        f"- Порог предупреждения τ = {tracker.get('tau')}.",
        f"- Наблюдений (актив × дата): {tracker.get('n_scored', 0)}; "
        f"срабатываний score≥τ: {tracker.get('n_alerts', 0)}.",
        f"- Hit-rate (среди срабатываний событие ≤ −threshold·σ в ближайшие 5 дней): {hit_text}.",
        f"- Brier score: {brier_text}.",
        "",
    ])
    return "\n".join(lines)


def _metadata_section(report: dict[str, Any]) -> list[str]:
    lines = ["## Метаданные запуска", ""]
    for provider in report.get("provider_statuses", []):
        status = "ok" if provider.get("failed") == 0 else f"ошибок: {provider.get('failed')}"
        lines.append(f"- Провайдер {provider['provider']}: активов {provider.get('ok', 0)} ({status}).")
    for feed in report.get("feed_statuses", []):
        mark = "ok" if feed.get("ok") else f"ошибка: {feed.get('error', '')[:120]}"
        lines.append(f"- Лента {feed['feed']}: {feed.get('items', 0)} элементов ({mark}).")
    llm = report.get("llm", {})
    if llm.get("enabled"):
        if "error" in llm:
            lines.append(f"- LLM: сбой — {llm['error'][:200]}")
        else:
            lines.append(f"- LLM: вызовов {llm.get('llm_calls', 0)}, кэш-попаданий "
                         f"{llm.get('cache_hits', 0)}, отклонено элементов — см. data/rejected/.")
    elif llm:
        lines.append("- LLM: отключена (--no-llm).")
    if report.get("news_total") is not None:
        lines.append(f"- Новостей в бюджете: {report['news_total']}.")
    materials = report.get("materials", {})
    if materials.get("applied"):
        lines.append(f"- Утверждённые изображения: {len(materials['used_materials'])}; влияние {materials['influence']:.0%}. "
                     f"План: {materials['plan']}.")
    if materials.get("error"):
        lines.append(f"- Изображения не учтены: {materials['error']}")
    for status in (report.get("vision") or {}).get("statuses") or []:
        if not status.get("ok"):
            lines.append(f"- Vision {status.get('asset')}: {str(status.get('error', ''))[:160]}")
    if report.get("dropped_assets"):
        lines.append(f"- Отброшены за недостатком истории: {', '.join(report['dropped_assets'])}.")
    if report.get("history_error"):
        lines.append(f"- История: не записана — {report['history_error'][:200]}")
    lines.append("")
    lines.append("*Веса вычислены детерминированно (HRP); LLM не имеет полномочий менять веса.*")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Структурированный отчёт: таблицы и оценки разделов, затем метаданные прогона."""
    body = render_analysis(report).rstrip("\n")
    return body + "\n\n" + "\n".join(_metadata_section(report))


def write_report(reports_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    directory = Path(reports_dir) / report["date"]
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / "portfolio.md"
    markdown.write_text(render_markdown(report), encoding="utf-8")

    structured = directory / "portfolio.json"
    structured.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    weights_csv = directory / "weights.csv"
    weights_csv.write_text("asset,weight\n" + "".join(
        f"{asset},{weight:.6f}\n" for asset, weight in sorted(report["weights"].items())),
        encoding="utf-8")
    analysis = write_analysis(directory, report)
    return {"markdown": markdown, "json": structured, "weights": weights_csv, "analysis": analysis}


def write_backtest(reports_dir: Path, result: dict[str, Any]) -> dict[str, Path]:
    directory = Path(reports_dir) / "backtest"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result['start']}_{result['end']}"
    markdown = directory / f"{stem}.md"
    structured = directory / f"{stem}.json"
    markdown.write_text(render_backtest(result), encoding="utf-8")
    structured.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": markdown, "json": structured}
