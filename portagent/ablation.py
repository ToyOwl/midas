"""A2: абляционная матрица кампаний на walk-forward истории.

Перебирает конфигурации (мутации копий конфига; отключение сентимента —
передача labels=None в walk_forward), считает метрики, строит block-bootstrap
CI и парную вероятность «лучше базовой линии» на общих ресемпл-индексах.

Журнал кампании: data/ablation/<stem>/journal.jsonl, строка
на вариант {name, status: running|succeeded|failed, duration_s, metrics};
`--resume` пропускает succeeded; сбой варианта не останавливает матрицу.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import ENGINE_VERSION, walk_forward

BASELINE = "sent_snf_k15"
STRATEGY = "fused"


def variant_grid() -> list[tuple[str, dict]]:
    """(имя, мутация конфига). Мутация применяется к deepcopy базового конфига."""
    return [
        ("price_only", {"_labels": None}),
        ("sent_linear_w50", {"fusion": {"method": "linear", "w_price": 0.5}}),
        ("sent_linear_w70", {"fusion": {"method": "linear", "w_price": 0.7}}),
        ("sent_linear_w90", {"fusion": {"method": "linear", "w_price": 0.9}}),
        ("sent_snf_k10", {"fusion": {"method": "snf", "snf_neighbors": 10}}),
        ("sent_snf_k15", {}),  # базовая линия
        ("sent_snf_k25", {"fusion": {"method": "snf", "snf_neighbors": 25}}),
        ("sent_snf_T10", {"fusion": {"method": "snf", "snf_iterations": 10}}),
        ("mw15", {"hrp": {"max_weight": 0.15}}),
        ("mw35", {"hrp": {"max_weight": 0.35}}),
    ]


def apply_variant(config: dict, mutation: dict) -> dict:
    """Мутация секций целиком (deep merge верхнего уровня не нужен: секции
    маленькие и самодостаточные); "_labels" — служебный ключ отключения."""
    mutated = copy.deepcopy(config)
    for key, value in mutation.items():
        if key.startswith("_"):
            mutated[key] = value
        else:
            section = dict(mutated.get(key, {}))
            section.update(value)
            mutated[key] = section
    return mutated


def _metrics_from_daily(daily: list[float]) -> dict[str, float]:
    returns = np.asarray(daily, dtype=float)
    equity = np.cumprod(1.0 + returns)
    total = float(equity[-1] - 1.0) if len(equity) else 0.0
    n = max(len(returns), 1)
    cagr = float(equity[-1] ** (252.0 / n) - 1.0) if len(equity) else 0.0
    running_max = np.maximum(np.maximum.accumulate(equity), 1.0)
    drawdown = float(np.max(1.0 - equity / running_max)) if len(equity) else 0.0
    return {"total_return": total, "cagr": cagr, "max_drawdown": drawdown}


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Индексы блочного ресемплинга: случайные неперекрывающиеся по выбору блоки."""
    starts = np.arange(0, n, block)
    picked = rng.choice(starts, size=len(starts), replace=True)
    indices = [start + offset for start in picked
               for offset in range(block) if start + offset < n]
    return np.asarray(indices, dtype=int)


def bootstrap_ci(daily: list[float], *, reps: int, block: int,
                 seed: int = 20261112) -> dict[str, dict[str, float]]:
    """CI total/CAGR/MaxDD через блочный бутстрап дневных доходностей."""
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("total_return", "cagr", "max_drawdown")}
    for _ in range(reps):
        indices = _block_indices(len(daily), block, rng)
        metrics = _metrics_from_daily([daily[i] for i in indices])
        for key, value in metrics.items():
            samples[key].append(value)
    result = {}
    for key, values in samples.items():
        array = np.asarray(values, dtype=float)
        result[key] = {"mean": float(array.mean()),
                       "p05": float(np.percentile(array, 5)),
                       "p95": float(np.percentile(array, 95))}
    return result


def paired_better(daily_a: list[float], daily_b: list[float], *, reps: int,
                  block: int, metric: str = "cagr", seed: int = 20261112) -> float:
    """P(вариант A лучше базовой линии B) на общих ресемпл-индексах."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(reps):
        indices = _block_indices(len(daily_a), block, rng)
        resampled = [daily_a[i] for i in indices]
        base = [daily_b[i] for i in indices]
        if _metrics_from_daily(resampled)[metric] > _metrics_from_daily(base)[metric]:
            wins += 1
    return wins / max(reps, 1)


class CampaignJournal:
    """Атомарный append-only журнал кампании (строка = состояние варианта)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
        return entries

    def succeeded(self, name: str) -> bool:
        latest = [e for e in self._entries() if e.get("name") == name]
        return bool(latest) and latest[-1].get("status") == "succeeded"

    def append(self, entry: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def results(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for entry in self._entries():
            if entry.get("status") == "succeeded":
                latest[entry["name"]] = entry
        return latest


def run_ablation(config: dict, config_path: str | Path, panel: pd.DataFrame,
                 labels: list[dict], *, start: str | None = None,
                 end: str | None = None, resume: bool = False,
                 only: str | None = None) -> dict[str, Any]:
    """Матрица вариантов -> журнал + отчёт reports/ablation/<range>.md."""
    from .config import project_paths

    paths = project_paths(config, config_path)
    reps = int(config["ablation"]["bootstrap_reps"])
    block = int(config["ablation"]["block_days"])
    grid = [(name, mutation) for name, mutation in variant_grid()
            if only is None or only.lower() in name.lower()]
    stem = f"{start or 'full'}_{end or 'full'}"
    from .history import load_dividends_panel, load_earnings, load_macro_panel
    dividends = load_dividends_panel(paths["data"] / "history", end=end)
    macro = load_macro_panel(paths["data"] / "history", end=end)
    earnings_pairs = [(row["date"], row["asset"])
                      for row in load_earnings(paths["data"] / "history")] or None

    # Resume не смешивает старые бесплатные расчёты с новым учётом позиций/издержек.
    import pandas as pd

    digest = hashlib.sha256(json.dumps(
        {"engine": ENGINE_VERSION, "config": config, "labels": labels,
         "earnings": earnings_pairs, "start": start, "end": end, "grid": grid},
        sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    for data in (panel, dividends, macro):
        digest.update(str(list(data.columns)).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    campaign_id = digest.hexdigest()[:16]
    journal = CampaignJournal(paths["data"] / "ablation" / stem / campaign_id / "journal.jsonl")

    for name, mutation in grid:
        if resume and journal.succeeded(name):
            continue
        started = time.monotonic()
        journal.append({"name": name, "status": "running",
                        "ts": datetime.now(timezone.utc).isoformat()})
        try:
            mutated = apply_variant(config, mutation)
            run_labels = mutated.pop("_labels", labels)
            if run_labels is not None:
                run_labels = labels
            result = walk_forward(panel, config=mutated, start=start, end=end,
                                  labels=run_labels, include_daily=True,
                                  dividends=dividends if not dividends.empty else None,
                                  macro=None if macro.empty else macro,
                                  earnings_pairs=earnings_pairs)
            entry = {
                "name": name, "status": "succeeded",
                "duration_s": round(time.monotonic() - started, 2),
                "metrics": result["strategies"][STRATEGY],
                "metrics_price_only": result["strategies"]["price_only"],
                "tracker_brier": result["tracker"].get("brier"),
                "daily": result["daily_returns"][STRATEGY],
            }
        except Exception as error:  # сбой варианта не останавливает матрицу
            journal.append({"name": name, "status": "failed",
                            "duration_s": round(time.monotonic() - started, 2),
                            "error": str(error)[:300]})
            continue
        journal.append({k: v for k, v in entry.items() if k != "daily"})
        # daily-серии пишем рядом отдельным файлом: журнал остаётся компактным
        daily_path = journal.path.parent / f"daily_{name}.json"
        daily_path.write_text(json.dumps(entry["daily"]), encoding="utf-8")

    results = journal.results()
    baseline_daily = None
    baseline_path = journal.path.parent / f"daily_{BASELINE}.json"
    if baseline_path.exists():
        baseline_daily = json.loads(baseline_path.read_text(encoding="utf-8"))

    table: dict[str, dict[str, Any]] = {}
    for name, entry in results.items():
        daily_path = journal.path.parent / f"daily_{name}.json"
        daily = json.loads(daily_path.read_text(encoding="utf-8")) if daily_path.exists() else None
        row: dict[str, Any] = {"metrics": entry.get("metrics"),
                               "tracker_brier": entry.get("tracker_brier"),
                               "duration_s": entry.get("duration_s")}
        if daily:
            row["ci"] = bootstrap_ci(daily, reps=reps, block=block)
            if baseline_daily and name != BASELINE:
                row["p_better_baseline"] = paired_better(
                    daily, baseline_daily, reps=reps, block=block)
        table[name] = row

    report = {"start": start, "end": end, "variants": table,
              "campaign_id": campaign_id, "engine_version": ENGINE_VERSION,
              "cost_model": {key: config["backtest"][key]
                             for key in ("commission_bps", "spread_bps")},
              "baseline": BASELINE, "reps": reps, "block": block,
              "n_succeeded": len(results), "n_total": len(grid)}
    directory = paths["reports"] / "ablation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / f"{stem}.md").write_text(render_ablation_markdown(report),
                                          encoding="utf-8")
    report["paths"] = {"markdown": str(directory / f"{stem}.md"),
                       "json": str(directory / f"{stem}.json"),
                       "journal": str(journal.path)}
    return report


def render_ablation_markdown(report: dict) -> str:
    lines = [f"# Абляции walk-forward ({report['start'] or 'вся'} — {report['end'] or 'вся'})",
             "",
             f"Вариантов закрыто: {report['n_succeeded']}/{report['n_total']}; "
             f"block-bootstrap: {report['reps']} реплик, блок {report['block']} дн.; "
             f"базовая линия — `{report['baseline']}`.",
             "Метрики после комиссии и половины полного спреда; "
             f"комиссия {report.get('cost_model', {}).get('commission_bps', 0):g} б.п., "
             f"полный спред {report.get('cost_model', {}).get('spread_bps', 0):g} б.п.",
             "",
             "| Вариант | CAGR | CI95 CAGR | MaxDD | P(лучше базы) | Brier трекера |",
             "|---|---:|---:|---:|---:|---:|"]
    ranked = sorted(report["variants"].items(),
                    key=lambda kv: -(kv[1].get("metrics") or {}).get("cagr", -9))
    for name, row in ranked:
        metrics = row.get("metrics") or {}
        ci = (row.get("ci") or {}).get("cagr") or {}
        p_better = row.get("p_better_baseline")
        brier = row.get("tracker_brier")
        ci_text = (f"[{ci.get('p05', 0):+.1%}; {ci.get('p95', 0):+.1%}]"
                   if ci else "—")
        lines.append(
            f"| {'**' + name + '**' if name == report['baseline'] else name} "
            f"| {metrics.get('cagr', 0):+.1%} | {ci_text} "
            f"| {metrics.get('max_drawdown', 0):.1%} "
            f"| {'—' if p_better is None else f'{p_better:.0%}'} "
            f"| {'—' if brier is None else round(brier, 3)} |")
    lines += ["", "P(лучше базы) — доля бутстрап-реплик, где CAGR варианта выше "
              "базовой линии на общих ресемпл-индексах; ~50% — неотличимо.",
              ""]
    return "\n".join(lines)
