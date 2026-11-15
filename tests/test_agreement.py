"""A4: сравнение меток двух LLM + upsert истории согласованности."""
import pandas as pd
import pytest

from portagent.history import upsert_agreement
from portagent.llm.agreement import compare_labels

PRIMARY = [
    {"news_id": "n1", "ticker": "MOEX:SBER", "sentiment": 0.8, "confidence": 0.9,
     "quote": "рекордная прибыль"},
    {"news_id": "n1", "ticker": "US:AAPL", "sentiment": -0.4, "confidence": 0.7,
     "quote": "снижение прогноза"},
    {"news_id": "n2", "ticker": "MOEX:GAZP", "sentiment": -0.6, "confidence": 0.8,
     "quote": "иск на 12 млрд"},
]
SECONDARY = [
    {"news_id": "n1", "ticker": "MOEX:SBER", "sentiment": 0.7, "confidence": 0.9,
     "quote": "рекордная прибыль"},
    {"news_id": "n1", "ticker": "US:AAPL", "sentiment": 0.3, "confidence": 0.6,
     "quote": "снижение прогноза"},
    {"news_id": "n3", "ticker": "US:MSFT", "sentiment": 0.5, "confidence": 0.5,
     "quote": "новый продукт"},
]


def test_compare_labels_metrics():
    result = compare_labels(PRIMARY, SECONDARY)
    assert result["labels_a"] == 3 and result["labels_b"] == 3
    assert result["common_pairs"] == 2                  # SBER и AAPL на n1
    # Jaccard: n1 — {SBER,AAPL} vs {SBER,AAPL} = 1.0; n2 — {GAZP} vs {} = 0; n3 — 0
    assert result["jaccard_mean"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    # расхождения: SBER Δ=-0.1 (ок), AAPL Δ=+0.7 (алерт)
    assert result["n_disagreements"] == 1
    row = result["disagreements"][0]
    assert row["ticker"] == "US:AAPL" and row["delta"] == 0.7
    assert row["quote_a"] == "снижение прогноза"
    # знак совпал у SBER; у AAPL знак разный -> 1/2
    assert result["sign_agreement"] == 0.5
    assert result["delta_mean_abs"] == pytest.approx(0.4, abs=1e-3)  # (0.1+0.7)/2


def test_compare_labels_empty():
    result = compare_labels([], [])
    assert result["common_pairs"] == 0 and result["disagreements"] == []


def test_upsert_agreement_rows_and_dedup(tmp_path):
    first = upsert_agreement(tmp_path, "2026-11-12", PRIMARY, SECONDARY,
                             model_a="bonsai", model_b="qwen")
    assert first >= 2
    frame = pd.read_parquet(tmp_path / "agreement.parquet")
    assert set(frame.columns) >= {"news_id", "ticker", "sentiment_a", "sentiment_b", "delta"}
    # повторная запись того же дня/моделей не дублирует строки
    upsert_agreement(tmp_path, "2026-11-12", PRIMARY, SECONDARY,
                     model_a="bonsai", model_b="qwen")
    frame2 = pd.read_parquet(tmp_path / "agreement.parquet")
    assert len(frame2) == len(frame)
