"""Deterministic visual scenarios, explicit activation and daily input freezing."""
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import digest, immutable, packed, read, now, today
from ..llm.runtime import launch_lock
from ..scheduler import _write_json
from ..portfolio.hrp import hrp_weights

VERSION = "visual-hrp-v1"


def transform(returns, distances, records, influence):
    """Visual similarity changes clustering; grounded adverse facts scale covariance.

    Only pairs with same-type material and sufficiently confident observations
    have a visual distance. Missing coverage retains the original distance.
    No language model output is interpreted as a portfolio weight.
    """
    if not 0 < influence <= .3:
        raise ValueError("Влияние материалов должно быть от 0 до 30%")
    assets = list(distances.index)
    vectors, risks, reasons, seen = {}, {}, {}, set()
    model_keys = {(r["analysis"]["embedding"]["model"], r["analysis"]["embedding"]["revision"])
                  for r in records}
    if len(model_keys) > 1:
        raise ValueError("Нельзя объединить эмбеддинги разных моделей")
    for record in sorted(records, key=lambda r: r["id"]):
        a = record["analysis"]
        if not a["relevant"]:
            continue
        v = np.asarray(a["embedding"]["vector"], dtype=float)
        if v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
            raise ValueError("Повреждён вектор материала")
        for signal in a["signals"]:
            asset = signal["asset"]
            key = (record["original_sha"], asset)
            if asset not in assets or signal["confidence"] < .6 or key in seen:
                continue
            seen.add(key)
            vectors.setdefault((asset, a["content_type"]), []).append(v / np.linalg.norm(v))
            risk = signal["risk"] * signal["confidence"]
            risks[asset] = max(risks.get(asset, 0), risk)
            reasons.setdefault(asset, []).append({"material": record["id"], "text": signal["reason"]})
    centroids = {k: np.mean(v, axis=0) for k, v in vectors.items()}
    centroids = {k: v / np.linalg.norm(v) for k, v in centroids.items() if np.linalg.norm(v) > 1e-9}
    output = distances.copy()
    pairs = []
    for i, left in enumerate(assets):
        for right in assets[i+1:]:
            kinds = {kind for asset, kind in centroids if asset == left and (right, kind) in centroids}
            if not kinds:
                continue
            visual = float(np.mean([np.sqrt(np.clip((1 - centroids[(left, k)] @ centroids[(right, k)]) / 2, 0, 1))
                                    for k in sorted(kinds)]))
            value = (1-influence) * float(distances.loc[left, right]) + influence * visual
            output.loc[left, right] = output.loc[right, left] = value
            pairs.append({"left": left, "right": right, "visual_distance": visual})
    multipliers = {a: 1 + influence * risks.get(a, 0) for a in assets}
    scaled = returns.mul(pd.Series(multipliers), axis="columns")
    return scaled, output, {"version": VERSION, "influence": influence, "covered_assets": sorted(reasons),
        "risk_multipliers": multipliers, "reasons": reasons, "pairs": pairs,
        "used_materials": sorted({r["material"] for rs in reasons.values() for r in rs})}


def selected_records(root, ids):
    if not isinstance(ids, list) or not 1 <= len(ids) <= 20 or len(set(ids)) != len(ids):
        raise ValueError("Выберите от 1 до 20 разных материалов")
    return [{**read(root, "records", item), "analysis": read(root, "analyses", item)} for item in sorted(ids)]


def preview(paths, report, config, ids, influence):
    from ..report.diagnostics import archived_prices
    from ..features.prices import build_panel, log_returns
    from ..portfolio.fuse import fuse_distances
    root = paths["data"] / "materials"
    records = selected_records(root, ids)
    frames = archived_prices(paths, report["date"])
    panel, _ = build_panel(frames)
    assets = list(report["weights"])
    returns = log_returns(panel[assets])
    frozen = json.loads((paths["data"] / "forecasts" / (report["date"] + ".json")).read_text(encoding="utf-8"))
    used = report["fusion"]["views_used"]
    views = {k: pd.DataFrame(v) for k, v in frozen["views"].items() if k in used and v is not None}
    cfg = config["fusion"]
    fused = fuse_distances(views, method=report["fusion"]["method"],
        coverage={k: 1 for k in views}, view_quality=report["fusion"].get("view_quality"),
        w_price=report["fusion"]["w_price"], min_coverage=cfg["min_coverage"],
        snf_neighbors=cfg["snf_neighbors"], snf_iterations=cfg["snf_iterations"], snf_min_quality=cfg["snf_min_quality"])
    D = fused["D"].reindex(index=assets, columns=assets)
    settings = report["portfolio"]
    kwargs = dict(linkage_method=settings["linkage_method"], max_weight=settings["max_weight"], cash_share=settings["cash_share"])
    baseline_returns, baseline_D = returns, D
    if frozen.get("materials", {}).get("applied"):
        fixed = root / "daily" / (report["date"] + ".json")
        prior = json.loads(fixed.read_text(encoding="utf-8"))
        if prior.get("plan") != frozen["materials"]["plan"]:
            raise ValueError("Сохранённый набор изображений не совпадает с прогнозом")
        baseline_returns, baseline_D, _ = transform(returns, D, prior["records"], prior["influence"])
    baseline = hrp_weights(baseline_returns, baseline_D, **kwargs)["weights"]
    drift = max(abs(float(baseline[a]) - report["weights"][a]) for a in assets)
    if drift > .00001:
        raise ValueError("Не удалось точно воспроизвести исходный HRP; сравнение остановлено")
    changed_returns, changed_D, details = transform(returns, D, records, influence)
    result = hrp_weights(changed_returns, changed_D, **kwargs)["weights"]
    rows = [{"asset": a, "before": float(baseline[a]), "after": float(result[a]),
             "delta": float(result[a] - baseline[a]), "reasons": details["reasons"].get(a, [])} for a in assets]
    rows.sort(key=lambda r: -abs(r["delta"]))
    base_sha = digest((paths["reports"] / report["date"] / "portfolio.json").read_bytes())
    scenario_id = digest(packed([VERSION, base_sha, records, influence, kwargs, cfg]))
    target = root / "scenarios" / (scenario_id + ".json")
    if target.is_file():
        return read(root, "scenarios", scenario_id)
    result = {"id": scenario_id, "date": report["date"], "created_at": now(), "rows": rows,
              "report_sha": base_sha, "baseline_max_difference": drift, "cash_share": kwargs["cash_share"],
              "turnover": sum(abs(r["delta"]) for r in rows) / 2, "details": details, "records": records,
              "note": "Сравнение на сохранённых ценах. Визуальное сходство меняет группировку бумаг, факты риска — масштаб риска в HRP. Доходность этого метода ещё не проверена."}
    immutable(target, packed(result))
    return result


def activate(root, scenario_id, valid_days):
    root = Path(root)
    scenario = read(root, "scenarios", scenario_id)
    if not scenario["details"]["used_materials"]:
        raise ValueError("В материалах нет достаточно уверенных наблюдений для расчёта")
    start = max(today() + timedelta(days=1), date.fromisoformat(scenario["date"]) + timedelta(days=1))
    plan = {"scenario": scenario_id, "approved_at": now(), "effective_from": start.isoformat(),
            "expires_on": (start + timedelta(days=valid_days-1)).isoformat(),
            "influence": scenario["details"]["influence"], "version": VERSION,
            "records": scenario["records"]}
    plan_id = digest(packed(plan))
    with launch_lock(root / "plan.lock"):
        immutable(root / "plans" / (plan_id + ".json"), packed(plan))
        _write_json(root / "active.json", {"plan": plan_id})
    return {"plan": plan_id, **{k: v for k, v in plan.items() if k != "records"}}


def active_plan(root):
    root = Path(root)
    pointer = root / "active.json"
    if not pointer.is_file():
        return None
    item = json.loads(pointer.read_text(encoding="utf-8")).get("plan")
    if not item:
        return None
    plan = read(root, "plans", item)
    return {"plan": item, **{k: v for k, v in plan.items() if k != "records"}}


def deactivate(root):
    root = Path(root)
    with launch_lock(root / "plan.lock"):
        _write_json(root / "active.json", {"plan": None, "disabled_at": now()})


def daily_inputs(root, day):
    """Freeze both presence and absence. Later approvals cannot change this day."""
    root = Path(root)
    date.fromisoformat(day)
    target = root / "daily" / (day + ".json")
    with launch_lock(root / "plan.lock"):
        if target.is_file():
            return json.loads(target.read_text(encoding="utf-8"))
        active = active_plan(root)
        value = {"date": day, "records": [], "plan": None}
        if active and active["effective_from"] <= day <= active["expires_on"]:
            plan = read(root, "plans", active["plan"])
            value = {**value, **plan, "plan": active["plan"]}
        immutable(target, packed(value))
        return value


def apply_daily(root, day, returns, D):
    inputs = daily_inputs(root, day)
    if not inputs["records"]:
        return returns, D, {"applied": False, "plan": None}
    scaled, changed, details = transform(returns, D, inputs["records"], inputs["influence"])
    return scaled, changed, {**details, "applied": bool(details["used_materials"]), "plan": inputs["plan"]}
