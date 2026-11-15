"""Критик: строгая валидация evidence-ID, graceful degradation при сбое LLM."""
import json

import pytest

from portagent.llm.client import LocalLLMError
from portagent.llm.critic import build_evidence, critique_portfolio, llm_failure_note, parse_critique

SNAPSHOT = {
    "date": "2026-11-11",
    "weights": {"MOEX:SBER": 0.30, "US:AAPL": 0.25, "MOEX:GAZP": 0.25, "US:MSFT": 0.10},
    "cash_share": 0.10,
    "fusion_note": "fused 70/30",
    "coverage_note": "3/4 активов покрыты новостями.",
    "movers": [{"asset": "MOEX:SBER", "ret_1d": 0.01, "ret_5d": 0.05}],
    "sentiment_stats": [{"asset": "MOEX:SBER", "mean_sentiment": 0.4, "n_news": 5}],
    "top_news": [{"id": "abc", "title": "Прибыль выросла", "tickers": ["MOEX:SBER"],
                  "sentiment": 0.7}],
}


def _sections():
    return [
        {"id": section_id, "stance": "ok", "fact": "Факт из evidence.",
         "evidence_ids": ["EV-DATE"]}
        for section_id in ("concentration", "sentiment", "price", "rare", "coverage")
    ]


def _valid_raw():
    from portagent.llm.critic import SUMMARY_IDS
    sections = _sections()
    for section in sections:
        section["evidence_ids"] = [SUMMARY_IDS[section["id"]]]
    return json.dumps({
        "summary": "Портфель сбалансирован.",
        "sections": sections,
        "findings": [
            {"section": "concentration", "severity": "warning", "title": "Концентрация в MOEX",
             "fact": "Вес MOEX:SBER = 0.30 — потолок.", "evidence_ids": ["EV-W-MOEX:SBER"],
             "hypothesis": "Возможна повышенная чувствительность к локальному рынку."},
            {"section": "sentiment", "severity": "info", "title": "Позитивный сентимент",
             "fact": "Средний сентимент 0.40 по 5 новостям.", "evidence_ids": ["EV-SENT-MOEX:SBER"]},
        ]}, ensure_ascii=False)


def test_build_evidence_ids_deterministic():
    evidence = build_evidence(SNAPSHOT)
    ids = {row["id"] for row in evidence}
    assert {"EV-DATE", "EV-W-MOEX:SBER", "EV-MOV-MOEX:SBER",
            "EV-SENT-MOEX:SBER", "EV-NEWS-abc",
            "EV-BOOK", "EV-CONC", "EV-SENT-HEAVY", "EV-PRICE", "EV-RARE-SUM"} <= ids


def test_comparison_notes_name_concentration_and_heavy_negative():
    evidence = {row["id"]: row["text"] for row in build_evidence(SNAPSHOT)}
    assert "0.235" in evidence["EV-CONC"]
    assert "MOEX:SBER 30.00%" in evidence["EV-CONC"]
    assert "US:MSFT" not in evidence["EV-CONC"]
    assert evidence["EV-RARE-SUM"] == "Предупреждений трекера нет."
    heavy = {
        **SNAPSHOT,
        "sentiment_stats": [
            {"asset": "MOEX:SBER", "mean_sentiment": -0.5, "n_news": 3},
            {"asset": "US:MSFT", "mean_sentiment": -0.4, "n_news": 3},
        ],
        "rare_events": [{"asset": "MOEX:GAZP", "score": 0.7, "warned": True,
                         "p_tail": 0.01, "drawdown": 0.16, "legal_flag": 0}],
        "clusters": {"1": ["MOEX:SBER", "MOEX:GAZP"]},
    }
    notes = {row["id"]: row["text"] for row in build_evidence(heavy)}
    above, below = notes["EV-SENT-HEAVY"].split("Негатив ниже равной доли: ")
    assert "MOEX:SBER" in above
    assert "US:MSFT" in below
    assert "MOEX:GAZP скор 0.70" in notes["EV-RARE-SUM"]
    assert "кластер 1: MOEX:SBER, MOEX:GAZP" in notes["EV-CLUSTERS"]


def test_parse_critique_valid():
    valid = parse_critique(_valid_raw(), valid_ids={row["id"] for row in build_evidence(SNAPSHOT)})
    assert valid["summary"].startswith("Портфель")
    assert [row["id"] for row in valid["sections"]] == [
        "concentration", "sentiment", "price", "rare", "coverage"]
    assert len(valid["findings"]) == 2
    assert valid["findings"][0]["severity"] == "warning"
    assert valid["findings_truncated"] is False


def test_parse_critique_unknown_evidence_rejected():
    raw = json.dumps({"summary": "s", "sections": _sections(), "findings": [
        {"section": "price", "severity": "info", "title": "t", "fact": "f",
         "evidence_ids": ["EV-NOPE"]}]})
    with pytest.raises(ValueError, match="несуществующие"):
        parse_critique(raw, valid_ids={"EV-W-MOEX:SBER", "EV-DATE"})


def test_parse_critique_bad_severity_rejected():
    raw = json.dumps({"summary": "s", "sections": _sections(), "findings": [
        {"section": "price", "severity": "critical", "title": "t", "fact": "f",
         "evidence_ids": ["EV-DATE"]}]})
    with pytest.raises(ValueError, match="severity"):
        parse_critique(raw, valid_ids={"EV-DATE"})


def test_parse_critique_requires_five_sections():
    raw = json.dumps({"summary": "s", "sections": _sections()[:4], "findings": []})
    with pytest.raises(ValueError, match="нет разделов"):
        parse_critique(raw, valid_ids={"EV-DATE"})


def test_parse_critique_keeps_six_findings():
    findings = [
        {"section": "price", "severity": "info", "title": f"t{i}", "fact": "f",
         "evidence_ids": ["EV-DATE"]}
        for i in range(8)
    ]
    raw = json.dumps({"summary": "s", "sections": _sections(), "findings": findings})
    parsed = parse_critique(raw, valid_ids={"EV-DATE"})
    assert len(parsed["findings"]) == 6
    assert parsed["findings_truncated"] is True


def test_critique_portfolio_ok(fake_chat):
    class CriticChat:
        def __call__(self, messages, *, schema=None):
            payload = json.loads(messages[-1]["content"])
            ids = {row["id"] for row in payload["evidence"]}
            from portagent.llm.critic import SUMMARY_IDS
            assert set(SUMMARY_IDS.values()) <= ids
            assert "EV-W-MOEX:SBER" not in ids
            response = json.loads(_valid_raw())
            response["findings"][0]["evidence_ids"] = ["EV-CONC"]
            response["findings"][1]["evidence_ids"] = ["EV-SENT-HEAVY"]
            return json.dumps(response, ensure_ascii=False), {"latency_s": 0.0}

    result = critique_portfolio(SNAPSHOT, chat=CriticChat())
    assert not result.get("failed")
    assert len(result["findings"]) == 2


def test_critique_portfolio_llm_failure_degrades():
    class FailingChat:
        def __call__(self, messages, *, schema=None):
            raise LocalLLMError("endpoint down")

    result = critique_portfolio(SNAPSHOT, chat=FailingChat())
    assert result["failed"]
    assert "остаётся действительным" in result["summary"]
