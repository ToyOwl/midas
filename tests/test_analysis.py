"""Структурированный разбор: таблицы из весов, текст разделов — из критика."""
from portagent.report.analysis import render_analysis


def test_render_analysis_tables_and_review_sections():
    text = render_analysis({
        "date": "2026-11-12",
        "weights": {"MOEX:SBER": 0.40, "MOEX:MGNT": 0.35, "US:MSFT": 0.15},
        "portfolio": {"cash_share": 0.10, "max_weight": 0.25,
                      "clusters": {"1": ["MOEX:MGNT"], "2": ["MOEX:SBER", "US:MSFT"]}},
        "fusion": {"mode": "snf", "method": "snf", "views_used": ["price", "sentiment"],
                   "note": "SNF сливает виды."},
        "llm": {"llm_calls": 3, "cache_hits": 1, "labels_total": 36},
        "legal": {"cases_total": 0, "coverage_frac": 0.0},
        "vision": {"charts_found": 0},
        "tracker": {"overlay": False, "warn_threshold": 0.5, "rows": [
            {"asset": "US:NVDA", "score": 0.74, "p_tail": 0.01, "drawdown": 0.16, "warned": True}]},
        "coverage": {"coverage_frac": 1.0},
        "sentiment_stats": [
            {"asset": "MOEX:SBER", "mean_sentiment": 0.7, "n_news": 3},
            {"asset": "MOEX:MGNT", "mean_sentiment": -0.6, "n_news": 3},
            {"asset": "US:MSFT", "mean_sentiment": 0.2, "n_news": 3},
        ],
        "movers": [{"asset": "MOEX:SBER", "ret_1d": 0.01, "ret_5d": 0.05}],
        "critique": {
            "summary": "Концентрация заметна.",
            "sections": [
                {"id": "concentration", "stance": "risk", "fact": "Три бумаги держат основную долю.",
                 "evidence_ids": ["EV-W-MOEX:SBER"]},
                {"id": "sentiment", "stance": "watch", "fact": "У MGNT негатив при большом весе.",
                 "evidence_ids": ["EV-SENT-MOEX:MGNT"]},
                {"id": "price", "stance": "ok", "fact": "Движения умеренные.",
                 "evidence_ids": ["EV-MOV-MOEX:SBER"]},
                {"id": "rare", "stance": "watch", "fact": "NVDA выше порога трекера.",
                 "evidence_ids": ["EV-RARE-US:NVDA"]},
                {"id": "coverage", "stance": "ok", "fact": "Новости покрывают все бумаги.",
                 "evidence_ids": ["EV-COVERAGE"]},
            ],
            "findings": [
                {"section": "sentiment", "severity": "warning", "title": "MGNT",
                 "fact": "Сентимент −0.60.", "evidence_ids": ["EV-SENT-MOEX:MGNT"],
                 "hypothesis": "Новостной фон может давить на цену."},
            ],
        },
    })
    assert "## Паспорт" in text
    assert "## Концентрация" in text
    assert "Херфиндаля" in text
    assert "MOEX:MGNT" in text
    assert "негатив при весе выше равной доли" in text
    assert "**риск.** Три бумаги держат основную долю." in text
    assert "[WARNING] MGNT" in text
    assert "US:NVDA" in text
    assert "Концентрация заметна." in text
