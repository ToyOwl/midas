"""Арбитражные суды: загрузка ручной выкладки, алиасы, грандинг LLM."""
import json

import pytest

from portagent.collect.courts import (CASES_SYSTEM_PROMPT, cases_to_items,
                                      load_court_drops)
from portagent.llm.news_analyst import analyze_news

UNIVERSE = ["MOEX:SBER", "MOEX:GAZP", "US:AAPL"]
ALIASES = {"SBER": ["Сбербанк"], "GAZP": ["Газпром"], "AAPL": ["Apple"]}

CASES = [
    {"case_number": "А40-123/26", "date": "2026-11-10", "type": "Исковое производство",
     "plaintiff": "ИФНС № 12", "defendant": "ПАО Сбербанк",
     "summary": "Налоговые претензии к ПАО Сбербанк на 12 млрд руб."},
    {"case_number": "А56-77/26", "date": "2026-11-08", "type": "Банкротство",
     "plaintiff": "Кредитор", "defendant": "Дочернее общество Газпрома",
     "summary": "Заявление о банкротстве подрядчика Газпрома."},
    {"case_number": "А77-99/26", "date": "2026-11-01", "type": "Иное",
     "plaintiff": "ООО Ромашка", "defendant": "ООО Лютик",
     "summary": "Спор о поставке мебели, юниверса не касается."},
]


def test_load_court_drops(tmp_path):
    drop = tmp_path / "2026-11-12"
    drop.mkdir(parents=True)
    (drop / "kad_export.json").write_text(
        json.dumps(CASES, ensure_ascii=False), encoding="utf-8")
    (drop / "wrapped.json").write_text(
        json.dumps({"cases": [CASES[0]]}, ensure_ascii=False), encoding="utf-8")
    cases = load_court_drops(tmp_path, "2026-11-12")
    assert len(cases) == 4
    assert load_court_drops(tmp_path, "2026-11-11") == []


def test_alias_mapping_filters_universe(tmp_path):
    items, counts = cases_to_items(CASES, aliases=ALIASES, universe=UNIVERSE)
    assert len(items) == 2  # третий кейс юниверса не касается
    assert counts == {"MOEX:SBER": 1, "MOEX:GAZP": 1, "US:AAPL": 0}
    sber_item = items[0]
    assert "MOEX:SBER" in sber_item["text"]          # подсказка для LLM
    assert sber_item["link"].endswith("А40-123/26".replace("А40-123/26", "number=А40-123/26")) or \
           "number=" in sber_item["link"]
    assert sber_item["published_at"] == "2026-11-10"


def test_court_llm_labeling_with_grounding():
    items, _ = cases_to_items(CASES, aliases=ALIASES, universe=UNIVERSE)

    class CourtChat:
        calls = 0

        def __call__(self, messages, *, schema=None):
            import json as json_lib

            CourtChat.calls += 1
            assert "юридический аналитик" in messages[0]["content"]
            request = json_lib.loads(messages[-1]["content"])
            rows = []
            for i, item in enumerate(request["items"]):
                tickers = ["MOEX:SBER"] if "Сбербанк" in item["text"] else ["MOEX:GAZP"]
                rows.append({"id": item["id"], "tickers": tickers,
                             "sentiment": -0.9 if i == 0 else -0.4,
                             "confidence": 0.8, "quote": "Налоговые претензии"})
            return json_lib.dumps({"items": rows}, ensure_ascii=False), {"latency_s": 0.0}

    result = analyze_news(items, universe=UNIVERSE, chat=CourtChat(), model="test",
                          batch_size=10, system_prompt=CASES_SYSTEM_PROMPT)
    labels = result["labels"]
    # quote «Налоговые претензии» есть только в первом кейсе — второй отброшен грандингом
    assert [label["ticker"] for label in labels] == ["MOEX:SBER"]
    assert labels[0]["sentiment"] == pytest.approx(-0.9)
    assert any(r["reason"] == "quote не является подстрокой текста"
               for r in result["rejected"])


def test_cases_prompt_contract():
    assert "quote" in CASES_SYSTEM_PROMPT and "не выдумывай" in CASES_SYSTEM_PROMPT.lower()
