"""Общие помощники тестов: синтетические кластеры активов, фейковый LLM-чат,
мини-fixture в tmp для e2e-прогона без сети."""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

CLUSTERS = {
    "A": ["MOEX:SBER", "MOEX:GAZP", "MOEX:LKOH", "MOEX:ROSN"],
    "B": ["MOEX:GMKN", "MOEX:PLZL", "MOEX:MGNT", "MOEX:MTSS"],
    "C": ["US:AAPL", "US:MSFT", "US:NVDA", "US:GOOGL"],
}


def synthetic_frames(days: int = 250, seed: int = 7) -> dict[str, pd.DataFrame]:
    """3 кластера x 4 актива: общий фактор кластера + идиосинкратический шум."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(end=date.today(), periods=days, freq="D").strftime("%Y-%m-%d")
    frames = {}
    for assets in CLUSTERS.values():
        factor = rng.normal(0.0, 0.01, days)
        for asset in assets:
            returns = factor + rng.normal(0.0, 0.004, days)
            frames[asset] = pd.DataFrame(
                {"close": pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index),
                 "volume": 1e6})
    return frames


class FakeChat:
    """Детерминированная замена LLM: шаблонный ответ + счётчик вызований."""

    def __init__(self, responses: list[str] | None = None, fail: bool = False):
        self.responses = responses or []
        self.fail = fail
        self.calls = 0

    def __call__(self, messages, *, schema=None):
        self.calls += 1
        if self.fail:
            from portagent.llm.client import LocalLLMError

            raise LocalLLMError("fake endpoint недоступен")
        user = messages[-1]["content"]
        request = json.loads(user)
        items = request["items"]
        universe = request["assets"]
        response = {"items": [
            {"id": item["id"], "tickers": [universe[i % len(universe)]],
             "sentiment": 0.5, "confidence": 0.8,
             "quote": item["text"][:40]} for i, item in enumerate(items)]}
        return json.dumps(response, ensure_ascii=False), {"latency_s": 0.0}


@pytest.fixture
def fake_chat():
    return FakeChat()


@pytest.fixture
def mini_fixture(tmp_path):
    """Мини-fixture в формате demo: 12 активов + 6 новостей."""
    rng = np.random.default_rng(3)
    directory = tmp_path / "fixture"
    directory.mkdir()
    days = 120
    anchor = date.today()
    dates = [(anchor - timedelta(days=days - 1 - i)).isoformat() for i in range(days)]
    for assets in CLUSTERS.values():
        factor = rng.normal(0.0, 0.01, days)
        for asset in assets:
            closes = 100.0 * np.exp(np.cumsum(factor + rng.normal(0.0, 0.005, days)))
            payload = {"candles": {"columns": ["begin", "open", "high", "low", "close", "volume"],
                                   "data": [[d, 0, 0, 0, round(float(c), 2), 1e6]
                                            for d, c in zip(dates, closes)]}}
            (directory / ("candles_" + asset.replace(":", "__") + ".json")).write_text(
                json.dumps(payload), encoding="utf-8")
    items = []
    for group, assets in enumerate(CLUSTERS.values()):
        for k in range(2):
            name = assets[k].split(":")[1]
            items.append({
                "id": f"news-{group}-{k}",
                "title": f"Новость о {name}",
                "text": f"Компания {name} отчиталась о рекордной выручке за квартал.",
                "link": "", "published_at": f"{dates[-1]}T09:00:00+00:00", "source": "demo",
            })
    (directory / "news.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return directory
