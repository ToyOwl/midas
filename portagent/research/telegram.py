"""Fresh Telegram snapshots, independent of immutable daily report archives."""
from datetime import datetime, timezone
import hashlib
import json
import uuid
from zoneinfo import ZoneInfo

from ..collect.errors import ProviderError
from ..collect.news import filter_recent
from ..collect.tg import fetch_tg_raw, parse_tg
from ..progress import emit
from ..storage.raw import archive
from .io import write_json


def today():
    return datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat()


def collect(config, paths):
    settings = config["news"]["telegram"]
    if not settings["enabled"]:
        raise ValueError("Сбор Telegram выключен в настройках")
    day = today()
    started = datetime.now(timezone.utc).isoformat()
    root = paths["data"] / "events" / "telegram"
    directory = root / day / uuid.uuid4().hex
    entries, feeds = [], []
    for channel in settings["channels"]:
        emit("telegram", f"Получение свежих постов @{channel}")
        error = None
        for _ in range(2):
            try:
                payload = fetch_tg_raw(channel, pages=settings["pages"], timeout=config["news"]["fetch_timeout_s"])
                break
            except ProviderError as exc:
                error = str(exc)
        else:
            feeds.append({"feed": "tg:" + channel, "ok": False, "items": 0, "error": error,
                          "fetched_at": datetime.now(timezone.utc).isoformat()})
            continue
        fetched = datetime.now(timezone.utc).isoformat()
        posts = filter_recent(parse_tg(payload, channel=channel),
                              lookback_hours=config["news"]["lookback_hours"], now=datetime.fromisoformat(fetched))
        entry = archive(directory, "raw", f"tg_{channel}.html", payload)
        entries.append({"name": entry["name"], "sha256": entry["sha256"], "channel": channel, "fetched_at": fetched})
        feeds.append({"feed": "tg:" + channel, "ok": True, "items": len(posts), "error": None,
                      "fetched_at": fetched, "latest_post_at": max((p["published_at"] or "" for p in posts), default=None)})
    snapshot = {"date": day, "fetched_at": started, "entries": entries, "feeds": feeds,
                "lookback_hours": config["news"]["lookback_hours"]}
    # Unique collection directory: retries never replace a previous snapshot or daily manifest.
    write_json(directory / "manifest.json", snapshot)
    return snapshot


def snapshots(paths, day=None, *, daily_lookback_hours=None):
    rows = []
    for path in (paths["data"] / "events" / "telegram").glob("????-??-??/*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if day is None or value["date"] <= day:
            rows.append((value, path.parent))
    if daily_lookback_hours is not None:
        for path in (paths["data"] / "raw").glob("????-??-??/manifest_news.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            feeds = [f for f in value.get("meta", {}).get("feeds", []) if f.get("feed", "").startswith("tg:")]
            if not feeds or value.get("meta", {}).get("source") != "net" or (day and value["date"] > day):
                continue
            entries = [{**e, "channel": e["name"].rsplit("__", 1)[0][3:], "fetched_at": value["fetched_at"]}
                       for e in value["entries"] if e["name"].startswith("tg_")]
            rows.append(({"date": value["date"], "fetched_at": value["fetched_at"], "feeds": feeds,
                          "entries": entries, "lookback_hours": daily_lookback_hours, "daily_archive": True}, path.parent))
    return sorted(rows, key=lambda r: datetime.fromisoformat(r[0]["fetched_at"]))


def snapshot_items(snapshot, directory):
    items = []
    root = (directory if snapshot.get("daily_archive") else directory / "raw").resolve()
    for entry in snapshot["entries"]:
        path = (root / entry["name"]).resolve()
        if path.parent != root:
            raise ValueError("Путь Telegram выходит за пределы архива")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Нарушена контрольная сумма Telegram-архива")
        rows = filter_recent(parse_tg(payload, channel=entry["channel"]),
                             lookback_hours=snapshot["lookback_hours"], now=datetime.fromisoformat(entry["fetched_at"]))
        items.extend({**row, "first_seen": entry["fetched_at"]} for row in rows)
    return items
