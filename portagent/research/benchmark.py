"""Frozen, date-purged stock benchmark. Never writes portfolio/history files."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, project_paths
from .io import write_json as _write_json
from ..tracker import top3
from .dataset import load_snapshot, make_dataset, split_fold
from .models import CANDIDATES, SEED, calibrate, fit_families, raw_scores

PROTOCOL = {"version": "stock-top3-frozen-v1", "first_test_year": 2022, "last_test_year": 2026,
    "target": "abs(next_real_log_return / current_EWMA_sigma) >= 2.5",
    "train_min_history": 250, "validation_calendar_days": 90, "calibration_calendar_days": 180,
    "tabm": {"k": 8, "blocks": 2, "width": 64, "dropout": .1, "epochs_max": 40, "patience": 7},
    "trees": {"rounds_max": 80, "max_depth": 3, "min_leaf": 30, "learning_rate": .05, "patience": 12},
    "primary_metric": "Brier", "comparators": ["frequency", "current_top3"],
    "candidates": list(CANDIDATES), "bootstrap_reps": 5000, "bootstrap_block_dates": 20,
    "multiplicity": "Bonferroni simultaneous 95% CI over 6 candidates x 2 comparators",
    "promotion": "none; research results do not alter HRP or active detector", "seed": SEED}


def metric_table(data, predictions):
    from sklearn.metrics import average_precision_score, log_loss
    results = []
    y = data.event.to_numpy(float)
    for name in predictions:
        p = np.clip(predictions[name].to_numpy(float), 1e-6, 1 - 1e-6)
        joined = data[["date", "asset", "event"]].copy()
        joined["p"] = p
        # The active selector ranks by rating, independently of probability calibration.
        joined["rank_score"] = data.current_rating.to_numpy() if name == "current_top3" and "current_rating" in data else p
        selected = joined.sort_values(["date", "rank_score", "asset"], ascending=[True, False, True]).groupby("date").head(3)
        unranked = joined.groupby("date").rank_score.nunique().eq(1).mean()
        bins = []
        for low, high in zip([0, .01, .025, .05, .1, .2, .5], [.01, .025, .05, .1, .2, .5, 1.001]):
            mask = (p >= low) & (p < high)
            if mask.any():
                bins.append({"low": low, "high": min(high, 1), "rows": int(mask.sum()),
                    "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
        results.append({"model": name, "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "pr_auc": float(average_precision_score(y, p)), "predicted_rate": float(p.mean()),
            "event_rate": float(y.mean()), "precision_top3": float(selected.event.mean()),
            "recall_top3": float(selected.event.sum() / max(y.sum(), 1)), "alerts_top3": len(selected),
            "unranked_days_fraction": float(unranked),
            "calibration": bins})
    return results


def paired_intervals(data, predictions, *, reps=5000, block=20, baselines=("frequency", "current_top3")):
    dates = sorted(data.date.unique())
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(dates), (reps, int(np.ceil(len(dates) / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % len(dates)).reshape(reps, -1)[:, :len(dates)]
    counts = data.groupby("date").size().reindex(dates).to_numpy()
    denominator = counts[indices].sum(1)
    y = data.event.to_numpy()
    alpha = .05 / (len(CANDIDATES) * len(baselines))
    out = []
    for model in CANDIDATES:
        for base in baselines:
            delta = (predictions[base].to_numpy() - y) ** 2 - (predictions[model].to_numpy() - y) ** 2
            daily = pd.Series(delta).groupby(data.date.reset_index(drop=True)).sum().reindex(dates).to_numpy()
            draws = daily[indices].sum(1) / denominator
            low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
            out.append({"model": model, "baseline": base, "improvement": float(delta.mean()),
                "ci_low": float(low), "ci_high": float(high), "confirmed": bool(low > 0)})
    return out


def audit_block(block):
    return {"rows": len(block), "events": int(block.event.sum()), "start": block.date.min(),
            "end": block.date.max(), "last_target": block.target_date.max()}


def run(config, config_path, market_root, *, epochs=40):
    paths = project_paths(config, config_path)
    frames, manifest = load_snapshot(market_root)
    protocol = {**PROTOCOL, "tabm": {**PROTOCOL["tabm"], "epochs_max": epochs}}
    source_files = [Path(__file__), Path(__file__).with_name("models.py"), Path(__file__).with_name("dataset.py"), Path(top3.__file__)]
    fingerprints = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    version = {"protocol": protocol, "source": manifest, "code": fingerprints,
               "libraries": {name: importlib.metadata.version(name) for name in ("torch", "tabm", "ngboost", "xgboost", "xgboostlss", "scikit-learn")}}
    key = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()[:16]
    root = paths["data"] / "research" / "top3" / "runs" / key
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "protocol.json", version)  # frozen before evaluation
    data, features = make_dataset(frames)
    data.to_parquet(root / "dataset.parquet", index=False)
    print("DATA", len(data), "features", len(features), "campaign", root, flush=True)
    baseline_file = Path(market_root) / ("baseline_" + fingerprints["top3.py"][:12] + ".parquet")
    if baseline_file.exists():
        baseline = pd.read_parquet(baseline_file)
    else:
        print("REPLAY current detector", flush=True)
        replay = top3.detect(frames, day=manifest["day"], threshold=2.5, halflife=20, window=250, return_history=True)
        baseline = pd.DataFrame(replay["forecast_rows"])
        baseline.to_parquet(baseline_file, index=False)
    all_results, audits = [], []
    for year in range(protocol["first_test_year"], protocol["last_test_year"] + 1):
        directory = root / str(year); directory.mkdir(exist_ok=True)
        if (directory / "predictions.parquet").exists() and (directory / "audit.json").exists():
            all_results.append(pd.read_parquet(directory / "predictions.parquet"))
            audits.append(json.loads((directory / "audit.json").read_text(encoding="utf-8")))
            print("RESUME", year, flush=True); continue
        train, valid, cal, test = split_fold(data, year)
        print("FIT", year, *(len(x) for x in (train, valid, cal, test)), flush=True)
        state, training = fit_families(train, valid, features, directory, epochs=epochs)
        raw_cal, raw_test = raw_scores(state, cal), raw_scores(state, test)
        predictions, _ = calibrate(raw_cal, cal, raw_test, saved=directory / "calibration.json")
        result = test[["date", "asset", "target_date", "event", "target_gap_days", "target_return", "frequency"]].reset_index(drop=True)
        for name in predictions:
            result[name] = predictions[name].to_numpy()
        result = result.merge(baseline, on=["date", "asset"], how="inner", validate="one_to_one")
        if len(result) != len(test):
            raise RuntimeError("Сравниваемые модели получили разные тестовые строки")
        result["fold"] = year
        audit = {"year": year, "train": audit_block(train), "validation": audit_block(valid),
                 "calibration": audit_block(cal), "test": audit_block(test), "training": training}
        _write_json(directory / "audit.json", audit)
        result.to_parquet(directory / "predictions.parquet", index=False)
        all_results.append(result); audits.append(audit)
        print("DONE", year, "test events", int(test.event.sum()), flush=True)
    combined = pd.concat(all_results, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    names = ["frequency", "current_top3", *CANDIDATES]
    if not np.isfinite(combined[names].to_numpy()).all():
        raise ValueError("Недопустимые вероятности в прогнозах")
    metrics = metric_table(combined, combined[names])
    comparisons = paired_intervals(combined, combined[names], reps=protocol["bootstrap_reps"])
    groups = {}
    for label, mask in [("MOEX", combined.asset.str.startswith("MOEX:")), ("US", combined.asset.str.startswith("US:")),
                        ("gap_le_4d_and_abs_return_le_30pct", (combined.target_gap_days <= 4) & (combined.target_return.abs() <= np.log(1.3)))]:
        part = combined.loc[mask].reset_index(drop=True)
        groups[label] = {"rows": len(part), "metrics": metric_table(part, part[names])}
    confirmed = [m for m in CANDIDATES if all(r["confirmed"] for r in comparisons if r["model"] == m)]
    report = {"schema": "stock_top3_benchmark_v1", "campaign": key, "date": manifest["day"],
        "protocol": protocol, "period": {"start": combined.date.min(), "end": combined.date.max(),
            "rows": len(combined), "dates": int(combined.date.nunique()), "events": int(combined.event.sum())},
        "metrics": metrics, "comparisons": comparisons, "confirmed_models": confirmed,
        "groups": groups, "folds": audits,
        "per_fold": {str(y): metric_table(p.reset_index(drop=True), p[names].reset_index(drop=True)) for y, p in combined.groupby("fold")},
        "modalities": {"status": "insufficient_history", "minimum_days": 180, "minimum_events": 30,
            "note": "История Telegram и реальных судебных событий пока не покрывает тестовые периоды; их добавочная польза не измерена."},
        "limitations": manifest["limitations"] + ["Годовой refit; гиперпараметры фиксированы до теста", "Это качество событий, не инвестиционная доходность", "Модели остаются исследовательскими; активный детектор и веса не заменены"],
        "paths": {"root": str(root), "predictions": str(root / "predictions.parquet")}}
    combined.to_parquet(root / "predictions.parquet", index=False)
    _write_json(root / "report.json", report)
    lines = ["# Проверка моделей редких событий", "", f"Период {report['period']['start']} — {report['period']['end']}; {len(combined)} наблюдений, {report['period']['events']} событий.",
        "", "| Модель | Brier ↓ | Log loss ↓ | PR-AUC ↑ | Precision TOP-3 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['brier']:.6f} | {r['logloss']:.6f} | {r['pr_auc']:.4f} | {r['precision_top3']:.2%} |" for r in metrics]
    lines += ["", "Подтверждённое преимущество над обеими базами (одновременные интервалы): " + (", ".join(confirmed) or "не установлено"),
        "", report["modalities"]["note"], "", "\n".join("- " + x for x in report["limitations"])]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(paths["data"] / "research" / "top3" / "latest.json", {"report": str(root / "report.json"), "campaign": key})
    print("COMPLETE", root, "confirmed", confirmed, flush=True)
    return report


def refresh_evaluation(report_path):
    """Re-score saved out-of-sample predictions; never retrain or select parameters."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data = pd.read_parquet(report_path.parent / "predictions.parquet")
    names = ["frequency", "current_top3", *CANDIDATES]
    report["metrics"] = metric_table(data, data[names])
    report["per_fold"] = {str(y): metric_table(p.reset_index(drop=True), p[names].reset_index(drop=True)) for y, p in data.groupby("fold")}
    report["block_sensitivity"] = {str(block): paired_intervals(data, data[names], reps=20000, block=block) for block in (20, 60, 120)}
    cal_rates = {a["year"]: a["calibration"]["events"] / a["calibration"]["rows"] for a in report["folds"]}
    data["frequency_cal"] = data.fold.map(cal_rates)
    report["additional_control"] = {"type": "exploratory_not_primary", "name": "Общая частота прошлого периода калибровки",
        "metrics": metric_table(data, data[["frequency_cal"]]),
        "comparisons": paired_intervals(data, data[names + ["frequency_cal"]], reps=20000, baselines=("frequency_cal",)),
        "note": "Дополнительная простая база без ранжирования бумаг; не использована для выбора моделей или параметров."}
    for label, mask in [("MOEX", data.asset.str.startswith("MOEX:")), ("US", data.asset.str.startswith("US:")),
        ("gap_le_4d_and_abs_return_le_30pct", (data.target_gap_days <= 4) & (data.target_return.abs() <= np.log(1.3)))]:
        part = data.loc[mask].reset_index(drop=True)
        report["groups"][label] = {"rows": len(part), "metrics": metric_table(part, part[names])}
    report["evaluation_revision"] = {"code_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Precision текущего детектора использует его rating; равные вероятности разбираются по тикеру. Повторная оценка сохранённых прогнозов без переобучения и выбора параметров."}
    report["limitations"] = list(dict.fromkeys(report["limitations"] + ["TabM пограничен: при блоке 120 дат интервал против частоты включает ноль; оба ансамбля сохраняют положительную границу при блоках 20/60/120.",
        "Преимущество в общем тесте не означает преимущество в каждом году или отдельно на MOEX.",
        "При равных оценках TOP-3 выбирается по тикеру; в таблице отмечена доля дней без ранжирования.",
        "Дополнительная проверка против общей частоты калибровочного периода: преимущество TabM не подтверждено, у двух ансамблей сохраняется."]))
    _write_json(report_path, report)
    lines = ["# Проверка моделей на истории акций", "", f"{report['period']['start']} — {report['period']['end']}; {len(data)} наблюдений, {report['period']['events']} событий.",
        "", "| Модель | Brier ↓ | PR-AUC ↑ | Precision TOP-3 | Дни без ранжирования |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['brier']:.6f} | {r['pr_auc']:.4f} | {r['precision_top3']:.2%} | {r['unranked_days_fraction']:.1%} |" for r in report["metrics"]]
    lines += ["", "Основной протокол, преимущество против обеих баз: " + ", ".join(report["confirmed_models"]), "", "\n".join("- " + s for s in report["limitations"]), "", report["modalities"]["note"]]
    report_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--market-root")
    parser.add_argument("--reevaluate")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.reevaluate:
        refresh_evaluation(args.reevaluate)
    elif args.market_root:
        run(load_config(args.config), args.config, args.market_root, epochs=args.epochs)
    else:
        parser.error("--market-root or --reevaluate required")
