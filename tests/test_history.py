"""История parquet: upsert цен и меток не плодит дубликаты ключей."""
import pandas as pd

from portagent.history import load_labels, load_price_panel, upsert_labels, upsert_prices


def test_price_upsert_replaces_same_day(tmp_path):
    index = pd.Index(["2026-01-01", "2026-01-02"])
    frames = {"US:AAPL": pd.DataFrame({"close": [1.0, 2.0], "volume": [10.0, 11.0]}, index=index)}
    upsert_prices(tmp_path, frames)
    frames["US:AAPL"].loc["2026-01-02", "close"] = 3.0
    upsert_prices(tmp_path, frames)
    panel = load_price_panel(tmp_path)
    assert panel.loc["2026-01-02", "US:AAPL"] == 3.0
    assert list(panel.index) == ["2026-01-01", "2026-01-02"]
    assert load_price_panel(tmp_path, end="2026-01-01").shape[0] == 1


def test_label_upsert_keeps_one_row_per_news_and_ticker(tmp_path):
    labels = [{"news_id": "n1", "ticker": "MOEX:SBER", "sentiment": 0.2, "confidence": 0.5}]
    items = {"n1": {"published_at": "2026-01-02T10:00:00+00:00", "source": "rss"}}
    upsert_labels(tmp_path, labels, items, fallback_date="2026-01-02")
    labels[0]["sentiment"] = -0.4
    upsert_labels(tmp_path, labels, items, fallback_date="2026-01-02")
    stored = load_labels(tmp_path)
    assert len(stored) == 1
    assert stored[0]["sentiment"] == -0.4
    assert stored[0]["date"] == "2026-01-02"
