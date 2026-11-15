import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from portagent.config import load_config, project_paths
from portagent.collect.errors import ProviderError
from portagent.research import events, telegram
from portagent.research.io import write_json
from portagent.web.research import view
from portagent.web.store import validate


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "configs/default.yaml"
    path.parent.mkdir()
    path.write_bytes((Path(__file__).parents[1] / "configs/default.yaml").read_bytes())
    cfg = load_config(path)
    cfg["news"]["telegram"]["channels"] = ["one", "two"]
    paths = project_paths(cfg, path)
    monkeypatch.setattr(events, "today", lambda: "2026-11-14")
    monkeypatch.setattr(telegram, "today", lambda: "2026-11-14")
    return cfg, path, paths


def payload(channel):
    stamp = datetime.now(timezone.utc).isoformat()
    return f'<div data-post="{channel}/123"><div class="tgme_widget_message_text">Apple legal dispute confirmed.</div><time datetime="{stamp}"></time></div>'.encode()


def chat(messages, **kwargs):
    items = json.loads(messages[-1]["content"])["items"]
    return json.dumps({"items": [{"id": i["id"], "tickers": ["US:AAPL"], "sentiment": -.5,
                                  "confidence": .8, "quote": "Apple legal dispute confirmed."} for i in items]}), {}


def test_fresh_collection_without_daily_archive_and_no_backdating(setup, monkeypatch):
    cfg, path, paths = setup
    frozen = paths["reports"] / "2026-11-12/portfolio.json"
    write_json(frozen, {"date": "2026-11-12", "weights": {"US:AAPL": .9}})
    before = frozen.read_bytes()
    calls = []
    def fetch(channel, **kwargs):
        calls.append(channel)
        return payload(channel)
    monkeypatch.setattr(telegram, "fetch_tg_raw", fetch)
    result = events.refresh_events(cfg, path, "2026-11-14", fetch_current=True, chat=chat)
    assert calls == ["one", "two"]
    assert result["groups"]["telegram"]["labels_total"] == 2
    assert frozen.read_bytes() == before and not (paths["data"] / "raw").exists()
    records = events.load_events(paths)
    assert all(r["available_date"] > "2026-11-14" for r in records)
    rows = view(paths, json.loads(before), cfg)
    assert rows["telegram_collection"]["date"] == "2026-11-14"
    assert all(r["received"] == 1 and len(r["posts"]) == 1 and r["independent_labeled_posts"] == 1 for r in rows["channels"])
    assert rows["events"]["record_count"] == 0  # Fresh evidence not backdated into old portfolio context.
    with pytest.raises(ValueError, match="текущей датой"):
        events.refresh_events(cfg, path, "2026-11-12", fetch_current=True, chat=chat)


def test_repeated_fetch_keeps_raw_records_and_first_observed_time(setup, monkeypatch):
    cfg, path, paths = setup
    data = {c: payload(c) for c in cfg["news"]["telegram"]["channels"]}
    monkeypatch.setattr(telegram, "fetch_tg_raw", lambda c, **k: data[c])
    events.refresh_events(cfg, path, "2026-11-14", fetch_current=True, chat=chat)
    originals = {p: p.read_bytes() for p in (paths["data"] / "events").rglob("*") if p.is_file() and p.name != "latest.json"}
    records = events.load_events(paths)
    result = events.refresh_events(cfg, path, "2026-11-14", fetch_current=True, chat=chat)
    assert result["groups"]["telegram"]["cache_hits"] == 1
    assert events.load_events(paths) == records
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert len(telegram.snapshots(paths)) == 2
    snapshot, directory = telegram.snapshots(paths)[0]
    entry = snapshot["entries"][0]
    raw = directory / "raw" / entry["name"]
    raw.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="контрольная сумма"):
        telegram.snapshot_items(snapshot, directory)


def test_channel_failure_does_not_hide_successful_fetch_or_model_failure(setup, monkeypatch):
    cfg, path, paths = setup
    calls = []
    def fetch(channel, **kwargs):
        calls.append(channel)
        if channel == "two":
            raise ProviderError("connection failed")
        return payload(channel)
    monkeypatch.setattr(telegram, "fetch_tg_raw", fetch)
    class FailedModel:
        def ensure_ready(self, **kwargs):
            raise RuntimeError("model unavailable")
    monkeypatch.setattr(events, "manager_for", lambda *a: FailedModel())
    result = events.refresh_events(cfg, path, "2026-11-14", fetch_current=True)
    assert calls == ["one", "two", "two"]
    assert [f["ok"] for f in result["collection"]["feeds"]] == [True, False]
    assert result["model_error"] == "model unavailable"
    assert result["groups"]["telegram"]["batch_errors"] == 1
    snapshot, directory = telegram.snapshots(paths)[0]
    assert len(telegram.snapshot_items(snapshot, directory)) == 1


def test_archived_reread_does_not_replace_newer_day_summary(setup):
    _, _, paths = setup
    root = paths["data"] / "events/collections"
    write_json(root / "new.json", {"date": "2026-11-14", "updated_at": "2026-11-14T18:00:00Z", "marker": "current"})
    write_json(root / "old.json", {"date": "2026-11-12", "updated_at": "2026-11-14T23:00:00Z", "marker": "old"})
    assert events.summary_for(paths, "2026-11-14")["marker"] == "current"
    assert events.summary_for(paths, "2026-11-12")["marker"] == "old"


def test_collection_job_has_no_historical_date_or_custom_url():
    assert validate("collect_telegram", {}) == {}
    for args in ({"date": "2020-01-01"}, {"url": "http://localhost/"}):
        with pytest.raises(ValueError):
            validate("collect_telegram", args)


def test_new_daily_collection_replaces_older_manual_status(setup, monkeypatch):
    from portagent.storage.raw import archive
    cfg, _, paths = setup
    monkeypatch.setattr(telegram, "fetch_tg_raw", lambda c, **k: payload(c))
    telegram.collect(cfg, paths)
    entry = archive(paths["data"] / "raw", "2026-11-14", "tg_one.html", payload("one"))
    newer_fetch = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    write_json(paths["data"] / "raw/2026-11-14/manifest_news.json", {
        "date": "2026-11-14", "fetched_at": newer_fetch, "entries": [entry],
        "meta": {"source": "net", "feeds": [{"feed": "tg:one", "ok": True, "items": 1}]}})
    latest, directory = telegram.snapshots(paths, daily_lookback_hours=72)[-1]
    assert latest["daily_archive"] and len(telegram.snapshot_items(latest, directory)) == 1


def test_wrong_issuer_cached_label_is_excluded_without_rewriting_archive(setup, monkeypatch):
    import pandas as pd
    from portagent.research.modalities import prepare_sources
    cfg, path, paths = setup
    monkeypatch.setattr(telegram, "fetch_tg_raw", lambda c, **k: payload(c))
    guard = events.entity_supported
    assert not guard("МГКЛ допустила дефолт. Газпром выпустил облигации.", "MOEX:MGNT", cfg["courts"]["aliases"])
    assert guard("Газпром выпустил облигации.", "MOEX:GAZP", cfg["courts"]["aliases"])
    def wrong(messages, **kwargs):
        answer, meta = chat(messages, **kwargs)
        return answer.replace("US:AAPL", "MOEX:MGNT"), meta
    monkeypatch.setattr(events, "entity_supported", lambda *a: True)  # Historical bug.
    events.refresh_events(cfg, path, "2026-11-14", fetch_current=True, chat=wrong)
    root = paths["data"] / "events"
    original = {p: p.read_bytes() for p in (root / "records").glob("*.json")}
    assert len(original) == 2
    monkeypatch.setattr(events, "entity_supported", guard)
    result = events.refresh_events(cfg, path, "2026-11-14", chat=wrong)
    assert result["groups"]["telegram"]["cache_hits"] == 1
    assert result["groups"]["telegram"]["rejected_entity_labels"] == 2
    assert events.load_events(paths) == []
    assert all(p.read_bytes() == b for p, b in original.items())
    _, status = prepare_sources(pd.DataFrame({"date": ["2026-11-15"], "asset": ["MOEX:MGNT"], "event": [1.]}), root)
    assert status["records"] == 0
