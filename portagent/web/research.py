"""Read-only research/source views for the local dashboard."""
import json
from pathlib import Path
from ..research.events import summary_for
from ..research.telegram import snapshots, snapshot_items, today


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def eligible_models(benchmark):
    """Fail closed: only candidates passing every saved robustness check."""
    selected = []
    for model in benchmark.get("confirmed_models", []):
        groups = [benchmark.get("comparisons", [])]
        groups += [benchmark.get("block_sensitivity", {}).get(str(block), []) for block in (20, 60, 120)]
        groups += [benchmark.get("additional_control", {}).get("comparisons", [])]
        checks = [[r for r in group if r["model"] == model] for group in groups]
        expected = [{"frequency", "current_top3"}] * 4 + [{"frequency_cal"}]
        if all({r.get("baseline") for r in rows} >= required and all(r.get("confirmed") and r.get("ci_low", 0) > 0 for r in rows)
               for rows, required in zip(checks, expected)):
            selected.append(model)
    return selected


def view(paths, report, config, store=None):
    root = paths["data"] / "research" / "top3"
    pointer = read_json(root / "latest.json", {})
    benchmark = None
    if pointer.get("report"):
        target = Path(pointer["report"]).resolve()
        if target.is_relative_to(root.resolve()):
            benchmark = read_json(target)
            if benchmark:
                benchmark["eligible_models"] = eligible_models(benchmark)
    capital = read_json(root / "capital.json")
    if store is not None:
        for job in store.list():
            if job["kind"] == "backtest" and job["status"] == "succeeded":
                candidate = store.get(job["id"]).get("result")
                if candidate and candidate.get("daily_dates"):
                    capital = candidate
                    break
    events = summary_for(paths, report["date"])
    source_status = read_json(root / "sources" / "status.json")
    if benchmark and source_status:
        benchmark["modalities"] = source_status
    feeds = report.get("feed_statuses", report.get("news_sources", []))
    manifest = read_json(paths["data"] / "raw" / report["date"] / "manifest_news.json", {})
    if not feeds:
        feeds = manifest.get("meta", {}).get("feeds", [])
    items = report.get("news_items", [])
    labels = report.get("news_labels", [])
    fresh = snapshots(paths, today(), daily_lookback_hours=config["news"]["lookback_hours"])
    latest, directory = fresh[-1] if fresh else ({}, None)
    live_items = snapshot_items(latest, directory) if directory else []
    channel_events = summary_for(paths, today()) if latest else events
    live_summary = channel_events if latest and (channel_events.get("collection") or {}).get("fetched_at") == latest.get("fetched_at") else {}
    channel_rows = []
    for channel in config["news"]["telegram"]["channels"]:
        source = "tg:" + channel
        state = next((f for f in feeds if f.get("feed") == source), {})
        if latest:
            state = next((f for f in latest["feeds"] if f["feed"] == source), {})
        ids = {r["id"] for r in items if r.get("source") == source}
        tagged = {r["news_id"] for r in labels if r["news_id"] in ids}
        posts = [r for r in live_items if r["source"] == source]
        post_ids = {r["id"] for r in posts}
        records = [r for r in channel_events.get("records", []) if r["kind"] == "telegram" and r["source"] == source
                   and (not latest or r["news_id"] in post_ids)]
        channel_rows.append({"name": channel, "url": "https://t.me/" + channel,
            "fetched_at": state.get("fetched_at", latest.get("fetched_at", manifest.get("fetched_at"))), "status": state.get("ok"), "error": state.get("error"),
            "latest_post_at": state.get("latest_post_at") or max((r["published_at"] or "" for r in posts), default=None),
            "posts": sorted(posts, key=lambda r: r.get("published_at") or "", reverse=True)[:5],
            "received": state.get("items", 0), "budget_selected": len(ids), "budget_labeled_posts": len(tagged),
            "independent_labeled_posts": len({r["news_id"] for r in records}),
            "records": sorted(records, key=lambda r: r.get("published_at") or "", reverse=True)[:6]})
    return {"benchmark": benchmark, "capital": capital, "events": events,
            "telegram_collection": {**latest, "analysis": live_summary.get("groups", {}).get("telegram"),
                                    "model_error": live_summary.get("model_error")} if latest else None,
            "channels": channel_rows, "feeds": feeds, "date": report["date"]}
