"""A4: согласованность двух локальных LLM на одном потоке новостей.

Оба движка проходят одинаковую валидацию грандинга (analyze_news); здесь
сравниваются их итоговые метки: пересечение тикеров на новость (Jaccard),
расхождение сентимента по общим (news, ticker) и список сильных
расхождений (|Δ| >= порога) с цитатами обеих моделей — кандидаты на
ручную проверку. Метрики попадают в отчёт и в history (задел под A3).
"""
from __future__ import annotations

from typing import Any

DELTA_ALERT = 0.4


def _index(labels: list[dict]) -> dict[tuple[str, str], dict]:
    return {(label["news_id"], label["ticker"]): label for label in labels}


def _sign(value: float) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)


def _news_tickers(labels: list[dict]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for label in labels:
        groups.setdefault(label["news_id"], set()).add(label["ticker"])
    return groups


def compare_labels(primary: list[dict], secondary: list[dict],
                   *, delta_alert: float = DELTA_ALERT) -> dict[str, Any]:
    """-> метрики согласованности + список расхождений (топ по |Δ|)."""
    index_a = _index(primary)
    index_b = _index(secondary)
    common = sorted(set(index_a) & set(index_b))
    deltas = []
    sign_matches = 0
    for key in common:
        a, b = index_a[key], index_b[key]
        deltas.append(float(b["sentiment"]) - float(a["sentiment"]))
        if _sign(float(a["sentiment"])) == _sign(float(b["sentiment"])):
            sign_matches += 1

    tickers_a = _news_tickers(primary)
    tickers_b = _news_tickers(secondary)
    jaccards = []
    for news_id in set(tickers_a) | set(tickers_b):
        set_a = tickers_a.get(news_id, set())
        set_b = tickers_b.get(news_id, set())
        union = set_a | set_b
        if union:
            jaccards.append(len(set_a & set_b) / len(union))

    disagreements = []
    for key in common:
        a, b = index_a[key], index_b[key]
        delta = float(b["sentiment"]) - float(a["sentiment"])
        if abs(delta) >= delta_alert:
            disagreements.append({
                "news_id": key[0], "ticker": key[1],
                "sentiment_a": round(float(a["sentiment"]), 3),
                "sentiment_b": round(float(b["sentiment"]), 3),
                "delta": round(delta, 3),
                "quote_a": a.get("quote", "")[:160],
                "quote_b": b.get("quote", "")[:160],
            })
    disagreements.sort(key=lambda row: -abs(row["delta"]))

    return {
        "labels_a": len(primary), "labels_b": len(secondary),
        "common_pairs": len(common),
        "jaccard_mean": round(sum(jaccards) / len(jaccards), 3) if jaccards else None,
        "delta_mean_abs": round(sum(abs(d) for d in deltas) / len(deltas), 3) if deltas else None,
        "sign_agreement": round(sign_matches / len(common), 3) if common else None,
        "delta_alert": delta_alert,
        "disagreements": disagreements[:10],
        "n_disagreements": len(disagreements),
    }
