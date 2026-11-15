"""Paired price/source ablation, gated on genuinely observed source history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import load_snapshot, make_dataset, split_fold
from .source_features import features_for
from .io import write_json
from .events import load_records


def prepare_sources(data, event_root):
    root = Path(event_root)
    records = load_records(root)
    collections = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "collections").glob("*.json")]
    source = features_for(data, records)
    enriched = pd.concat([data.reset_index(drop=True), source.reset_index(drop=True)], axis=1)
    # Re-reading one archive is not a new day of collection.
    days = sorted({c["date"] for c in collections if c.get("groups", {}).get("telegram", {}).get("batch_errors", 1) == 0})
    start = min((r["available_date"] for r in records), default="9999-12-31")
    usable = enriched.loc[(enriched.date >= start) & enriched.event.notna()]
    status = {"status": "insufficient_history", "collection_days": len(days), "minimum_days": 180,
        "minimum_events": 30, "resolved_events": int(usable.event.sum()), "records": len(records),
        "court_records": sum(r["kind"] == "court" for r in records), "available_from": None if not records else start,
        "features": list(source.columns), "eligible_years": [],
        "note": "Датированной истории Telegram и судебных событий пока недостаточно; их добавочная польза не измерена."}
    if len(days) >= 180 and status["resolved_events"] >= 30:
        for year in sorted(pd.to_datetime(usable.date).dt.year.unique()):
            try:
                blocks = split_fold(usable, int(year))
                if all(int(b.event.sum()) >= 10 for b in blocks):
                    status["eligible_years"].append(int(year))
            except ValueError:
                pass
        if status["eligible_years"]:
            status.update(status="ready", note="Данные готовы к парному сравнению ценовых и дополненных моделей.")
    return enriched, status


def run(market_root, event_root, output):
    from .models import CANDIDATES, fit_families, raw_scores, calibrate, SEED
    from .benchmark import metric_table
    frames, _ = load_snapshot(market_root)
    data, price_features = make_dataset(frames)
    data, status = prepare_sources(data, event_root)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output / "features.parquet", index=False)
    write_json(output / "status.json", status)
    if status["status"] != "ready":
        return status
    # Both arms use exactly the same rows, folds and calibration dates.
    available = data[data.date >= status["available_from"]]
    signature = hashlib.sha256(pd.util.hash_pandas_object(available, index=False).values.tobytes()
        + Path(__file__).read_bytes()).hexdigest()[:16]
    campaign = output / signature; combined = []
    for year in status["eligible_years"]:
        train, valid, cal, test = split_fold(available, year)
        result = test[["date", "asset", "event"]].reset_index(drop=True)
        for arm, features in (("price", price_features), ("sources", price_features + status["features"])):
            target = campaign / str(year) / arm
            state, audit = fit_families(train, valid, features, target)
            predictions, _ = calibrate(raw_scores(state, cal), cal, raw_scores(state, test), saved=target / "calibration.json")
            write_json(target / "audit.json", audit)
            for name in CANDIDATES:
                result[arm + "_" + name] = predictions[name].to_numpy()
        combined.append(result)
    combined = pd.concat(combined, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    names = [a + "_" + m for a in ("price", "sources") for m in CANDIDATES]
    dates = sorted(combined.date.unique()); rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(dates), (5000, int(np.ceil(len(dates) / 20))))
    indices = ((starts[:, :, None] + np.arange(20)) % len(dates)).reshape(5000, -1)[:, :len(dates)]
    counts = combined.groupby("date").size().reindex(dates).to_numpy()
    comparisons = []
    for name in CANDIDATES:
        delta = (combined["price_" + name] - combined.event)**2 - (combined["sources_" + name] - combined.event)**2
        daily = delta.groupby(combined.date).sum().reindex(dates).to_numpy()
        draws = daily[indices].sum(1) / counts[indices].sum(1)
        low, high = np.quantile(draws, [.05 / 12, 1 - .05 / 12])
        comparisons.append(dict(model=name, improvement=float(delta.mean()), ci_low=float(low), ci_high=float(high), confirmed=bool(low > 0)))
    combined.to_parquet(campaign / "predictions.parquet", index=False)
    report = {**status, "status": "completed", "metrics": metric_table(combined, combined[names]), "comparisons": comparisons,
        "note": "Парное сравнение тех же ценовых моделей с дополнительными признаками на одинаковых доступных датах; 6 одновременных интервалов."}
    write_json(campaign / "report.json", report); write_json(output / "status.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-root", required=True)
    parser.add_argument("--event-root", default="data/events")
    parser.add_argument("--output", default="data/research/top3/sources")
    args = parser.parse_args()
    print(json.dumps(run(args.market_root, args.event_root, args.output), ensure_ascii=False))
