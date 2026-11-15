"""Валидация грандинга LLM-аналитика: цитаты, тикеры, диапазоны, кэш."""
import json

from portagent.llm.news_analyst import RESPONSE_SCHEMA, analyze_news, validate_batch

UNIVERSE = ["MOEX:SBER", "MOEX:GAZP", "US:AAPL"]
ITEMS = [
    {"id": "n1", "title": "t1", "text": "Сбер отчитался о рекордной прибыли за квартал",
     "source": "demo"},
    {"id": "n2", "title": "t2", "text": "Газпром снизил прогноз добычи на текущий год",
     "source": "demo"},
]


def _response(items):
    return json.dumps({"items": items}, ensure_ascii=False)


def test_valid_labels_accepted():
    raw = _response([
        {"id": "n1", "tickers": ["MOEX:SBER"], "sentiment": 0.8, "confidence": 0.9,
         "quote": "рекордной прибыли"},
        {"id": "n2", "tickers": ["MOEX:GAZP"], "sentiment": -0.6, "confidence": 0.7,
         "quote": "Газпром снизил прогноз добычи"},
    ])
    labels, rejected = validate_batch(raw, ITEMS, UNIVERSE)
    assert len(labels) == 2 and not rejected
    assert labels[0]["ticker"] == "MOEX:SBER"
    assert labels[0]["sentiment"] == 0.8


def test_quote_must_be_substring():
    raw = _response([{"id": "n1", "tickers": ["MOEX:SBER"], "sentiment": 0.5,
                      "confidence": 0.9, "quote": "выдуманная цитата которой нет"}])
    labels, rejected = validate_batch(raw, ITEMS, UNIVERSE)
    assert not labels
    assert rejected[0]["reason"] == "quote не является подстрокой текста"


def test_unknown_ticker_dropped():
    raw = _response([{"id": "n1", "tickers": ["XX:UNKNOWN", "MOEX:SBER"], "sentiment": 0.5,
                      "confidence": 0.9, "quote": "рекордной прибыли"}])
    labels, _ = validate_batch(raw, ITEMS, UNIVERSE)
    assert [label["ticker"] for label in labels] == ["MOEX:SBER"]


def test_empty_tickers_rejected():
    raw = _response([{"id": "n1", "tickers": [], "sentiment": 0.5, "confidence": 0.9,
                      "quote": "рекордной прибыли"}])
    labels, rejected = validate_batch(raw, ITEMS, UNIVERSE)
    assert not labels and "нет тикеров" in rejected[0]["reason"]


def test_out_of_range_rejected():
    raw = _response([{"id": "n1", "tickers": ["MOEX:SBER"], "sentiment": 2.0,
                      "confidence": 0.9, "quote": "рекордной прибыли"}])
    labels, rejected = validate_batch(raw, ITEMS, UNIVERSE)
    assert not labels and "вне диапазона" in rejected[0]["reason"]


def test_missing_item_marked_rejected():
    raw = _response([{"id": "n1", "tickers": ["MOEX:SBER"], "sentiment": 0.5,
                      "confidence": 0.9, "quote": "рекордной прибыли"}])
    _, rejected = validate_batch(raw, ITEMS, UNIVERSE)
    assert any(r["news_id"] == "n2" and "не вернула" in r["reason"] for r in rejected)


def test_invalid_json_rejects_whole_batch():
    labels, rejected = validate_batch("совсем не JSON", ITEMS, UNIVERSE)
    assert not labels and len(rejected) == 2


def test_think_tags_stripped():
    raw = "<think>рассуждения</think>" + _response(
        [{"id": "n1", "tickers": ["MOEX:SBER"], "sentiment": 0.5, "confidence": 0.9,
          "quote": "рекордной прибыли"}])
    labels, _ = validate_batch(raw, ITEMS, UNIVERSE)
    assert len(labels) == 1


def test_analyze_news_cache(tmp_path, fake_chat):
    result = analyze_news(ITEMS, universe=UNIVERSE, chat=fake_chat, model="test",
                          batch_size=10, cache_dir=tmp_path)
    assert fake_chat.calls == 1
    assert result["stats"]["llm_calls"] == 1
    assert result["stats"]["cache_hits"] == 0
    second = analyze_news(ITEMS, universe=UNIVERSE, chat=fake_chat, model="test",
                          batch_size=10, cache_dir=tmp_path)
    assert fake_chat.calls == 1  # повторный прогон не дёргает LLM
    assert second["stats"]["cache_hits"] == 1
    assert second["labels"] == result["labels"]


def test_analyze_news_llm_error_does_not_crash(tmp_path):
    from conftest import FakeChat

    failing = FakeChat(fail=True)
    result = analyze_news(ITEMS, universe=UNIVERSE, chat=failing, model="test",
                          batch_size=10, cache_dir=tmp_path)
    assert result["stats"]["batch_errors"] == 1
    assert not result["labels"] and len(result["rejected"]) == 2
    # транзиентный сбой не кэшируется: следующий прогон с рабочим чатом делает реальный вызов
    working = FakeChat()
    retry = analyze_news(ITEMS, universe=UNIVERSE, chat=working, model="test",
                         batch_size=10, cache_dir=tmp_path)
    assert retry["stats"]["llm_calls"] == 1
    assert len(retry["labels"]) == 2


def test_response_schema_shape():
    assert RESPONSE_SCHEMA["required"] == ["items"]
    item = RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert set(item["required"]) == {"id", "tickers", "sentiment", "confidence", "quote"}
