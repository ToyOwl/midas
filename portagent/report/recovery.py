"""Дополнительная разметка архива без пересчёта или перезаписи портфеля."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..collect import moex, news
from ..collect.tg import parse_tg
from ..config import project_paths
from ..features import sentiment
from ..llm.assistant import explain
from ..llm.models import manager_for
from ..llm.news_analyst import analyze_news, make_chat
from ..llm.runtime import launch_lock
from ..progress import emit
from ..scheduler import _write_json


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verified_entries(paths, day, kind):
    directory = paths["data"] / "raw" / day
    manifest = json.loads((directory / f"manifest_{kind}.json").read_text(encoding="utf-8"))
    if manifest["date"] != day or manifest["kind"] != kind:
        raise ValueError("Дата или тип архива не совпадает")
    entries = []
    for entry in manifest["entries"]:
        path = (directory / entry["name"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Путь выходит за пределы архива")
        payload = path.read_bytes()
        if digest(payload) != entry["sha256"]:
            raise ValueError(f"Нарушена контрольная сумма архива: {path.name}")
        entries.append((entry["name"].rsplit("__", 1)[0], payload))
    return manifest, entries


def archived_prices(paths, day):
    _, entries = verified_entries(paths, day, "market")
    frames = {}
    for name, payload in entries:
        if name.startswith("moex_"):
            frames["MOEX:" + name[5:]] = moex.parse_candles(payload)
        elif name.startswith("yf_"):
            frames["US:" + name[3:]] = pd.read_csv(io.BytesIO(payload), index_col="date")
    for frame in frames.values():
        frame.index = frame.index.astype(str).str[:10]
    return {asset: frame.loc[frame.index <= day].copy() for asset, frame in frames.items()}


def archived_news(config, paths, day):
    manifest, entries = verified_entries(paths, day, "news")
    anchor = datetime.fromisoformat(manifest["fetched_at"])
    items = []
    for name, payload in entries:
        if name.startswith("rss_"):
            rows = news.parse_rss(payload, source=name.split("_", 2)[2])
        elif name.startswith("gn_"):
            _, locale, ticker = name.split("_", 2)
            rows = news.parse_rss(payload, source=f"google-news {locale} {ticker}")
        elif name.startswith("tg_"):
            rows = parse_tg(payload, channel=name[3:])
        elif name == "moex_sitenews":
            items.extend(moex.parse_sitenews(payload,
                lookback_hours=config["news"]["moex_sitenews"]["lookback_hours"], now=anchor))
            continue
        else:
            raise ValueError(f"Неизвестный источник в архиве: {name}")
        rows = news.filter_recent(rows, lookback_hours=config["news"]["lookback_hours"], now=anchor)
        if name.startswith("gn_"):
            rows = rows[:config["news"]["google_news"]["per_query_limit"]]
        items.extend(rows)
    universe = (["MOEX:" + t for t in config["universe"]["moex"]]
                + ["US:" + t for t in config["universe"]["us"]])
    return news.select_for_budget(news.dedupe(items), universe=universe,
        aliases=config["courts"]["aliases"], limit=config["news"]["max_items_per_run"],
        per_asset=config["news"]["per_asset_priority"])


def load_supplement(paths, day, original):
    directory = paths["reports"] / day / "analysis_updates"
    pointer = directory / "latest.json"
    if not pointer.is_file():
        return original
    try:
        filename = json.loads(pointer.read_text(encoding="utf-8"))["file"]
        path = (directory / filename).resolve()
        if path.parent != directory.resolve():
            return original
        value = json.loads(path.read_text(encoding="utf-8"))
        original_sha = digest((paths["reports"] / day / "portfolio.json").read_bytes())
        if (value["analysis_update"]["source_report_sha"] != original_sha
                or value["weights"] != original["weights"] or value["date"] != day):
            return original
        return value
    except (ValueError, OSError, KeyError, TypeError):
        return original


def recover_news(config, config_path, day, *, chat=None):
    from ..pipeline.daily import _sentiment_stats, _top_news
    paths = project_paths(config, config_path)
    source = paths["reports"] / day / "portfolio.json"
    before = source.read_bytes()
    original = json.loads(before)
    frozen = paths["data"] / "forecasts" / f"{day}.json"
    forecast_before = frozen.read_bytes() if frozen.is_file() else None
    directory = paths["reports"] / day / "analysis_updates"
    with launch_lock(paths["data"] / "runtime" / f"news_recovery_{day}.lock"):
        items = archived_news(config, paths, day)
        if len(items) != original["news_total"]:
            raise ValueError("Исходный бюджет новостей не восстановлен: количество не совпало")
        rejected_file = paths["data"] / "rejected" / f"{day}.json"
        failed_ids = {r["news_id"] for r in json.loads(rejected_file.read_text(encoding="utf-8"))
                      if r["reason"].startswith("ошибка LLM:")} if rejected_file.is_file() else set()
        if not failed_ids <= {i["id"] for i in items}:
            raise ValueError("В архиве найдены не все исходные сбойные новости")
        llm = config["llm"]
        if chat is None:
            manager = manager_for(config, config_path)
            manager.ensure_ready(timeout=180)
            chat = make_chat(llm["endpoint"], llm["model"], timeout_s=llm["timeout_s"],
                max_tokens=llm["max_tokens"], seed=llm["seed"],
                recover=lambda: manager.ensure_ready(timeout=180))
        attempts = []
        for attempt in range(3):
            analysis = analyze_news(items, universe=original["assets"], chat=chat, model=llm["model"],
                batch_size=llm["analyst_batch_size"], cache_dir=paths["data"] / "cache" / "llm_labels",
                cache_salt=f"mt{llm['max_tokens']}")
            attempts.append(analysis["stats"])
            _write_json(directory / "recovery_checkpoint.json", {"source_report_sha": digest(before),
                "items": items, "analysis": analysis, "attempts": attempts})
            if not analysis["stats"]["batch_errors"]:
                break
        result = copy.deepcopy(original)
        labels = analysis["labels"]
        by_id = {i["id"]: i for i in items}
        coverage = sentiment.coverage(labels, original["assets"], min_news=config["fusion"]["min_news_per_asset"])
        result["coverage_counts"] = coverage.pop("counts")
        result["coverage"] = coverage
        result["llm"] = {"enabled": True, **analysis["stats"],
                         "llm_calls": sum(a["llm_calls"] for a in attempts)}
        result["sentiment_stats"] = _sentiment_stats(labels)
        result["top_news"] = _top_news(labels, by_id)
        result["news_labels"] = labels
        result["news_items"] = items
        result["analysis_update"] = {"source_report_sha": digest(before),
            "updated_at": datetime.now(timezone.utc).isoformat(), "weights_unchanged": True,
            "original_batch_errors": original["llm"].get("batch_errors", 0),
            "attempts": attempts, "failed_news_requested": len(failed_ids),
            "method": "archived-news-recovery-v1",
            "note": "Дополнительная разметка архива. Веса, прогноз и режим первоначального HRP сохранены."}
        emit("critique", "Рецензия обновлённых данных при сохранённых весах")
        review = explain(result, "", llm=llm, cache_dir=paths["data"] / "cache" / "assistant",
                         review=True, chat=chat)
        result["critique"] = review["response"]
        if source.read_bytes() != before or (forecast_before is not None and frozen.read_bytes() != forecast_before):
            raise RuntimeError("Исходный отчёт изменился во время разметки; обновление не опубликовано")
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True).encode()
        target = directory / (digest(payload)[:20] + ".json")
        _write_json(target, result)
        _write_json(directory / "latest.json", {"file": target.name, "updated_at": result["analysis_update"]["updated_at"]})
        return {"date": day, "llm": result["llm"], "coverage": coverage,
            "analysis_update": result["analysis_update"], "path": str(target),
            "review": review, "rejected": analysis["rejected"]}
