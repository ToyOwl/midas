"""Исторический тест использования сентимент-модальности (контроль качества
самого механизма fusion на реальных ценах).

Положительный контроль (oracle): сентимент на дату t = нормированная будущая
horizon-дневная доходность актива. Look-ahead здесь — НАМЕРЕННО: если
пайплайн «метки -> EWMA-ряд -> D_sent -> SNF/linear -> HRP» умеет пользоваться
модальностью, oracle обязан обходить price-only. HRP не видит направления
доходностей — модальность даёт ему структуру будущей совместной динамики.

Отрицательный контроль (noise): те же значения сентимента, перемешанные по
парам (дата, актив) — распределение сохранено, информация уничтожена; должен
быть неотличим от price-only (P ≈ 50%).

Схема теста: price-only / linear+oracle / snf+oracle / linear+noise /
snf+noise на одной панели, парный block-bootstrap по CAGR (выше — лучше) и
MaxDD (ниже — лучше) против price-only на общих ресемпл-индексах.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import bootstrap_ci, paired_better
from .backtest import walk_forward

STRATEGY = "fused"


def build_labels_oracle(panel: pd.DataFrame, *, horizon: int = 5,
                        confidence: float = 0.8,
                        sigma_scale: float = 2.0) -> list[dict]:
    """Сентимент t = clip(fwd_horizon_return / (sigma_scale * sigma_t)).

    Метки датированы днём t и содержат будущую информацию — это управление
    для проверки механизма, НЕ торговый сигнал.
    """
    log_returns = np.log(panel / panel.shift(1))
    forward = np.log(panel.shift(-horizon) / panel)
    sigma = log_returns.rolling(252, min_periods=60).std()
    scaled = (forward / (sigma_scale * sigma)).clip(-1.0, 1.0)
    labels: list[dict] = []
    for date, row in scaled.iterrows():
        day = str(date)[:10]
        for asset, value in row.items():
            if np.isfinite(value):
                labels.append({"date": day, "news_id": f"oracle-{day}-{asset}",
                               "ticker": asset, "sentiment": round(float(value), 4),
                               "confidence": confidence})
    return labels


def build_labels_noise(panel: pd.DataFrame, labels: list[dict], *,
                       seed: int = 20261112) -> list[dict]:
    """Перемешиваем sentiment по парам (date, ticker): маргинальное
    распределение то же, привязка к активу и дню уничтожена."""
    rng = np.random.default_rng(seed)
    values = [label["sentiment"] for label in labels]
    shuffled = list(rng.permutation(values))
    return [{**label, "sentiment": float(value)}
            for label, value in zip(labels, shuffled)]


def _variants(config: dict, oracle: list[dict], noise: list[dict]):
    import copy

    base = copy.deepcopy(config)
    linear = copy.deepcopy(config); linear["fusion"]["method"] = "linear"
    snf = copy.deepcopy(config); snf["fusion"]["method"] = "snf"
    return [
        ("price_only", base, None),
        ("linear_oracle", linear, oracle),
        ("snf_oracle", snf, oracle),
        ("linear_noise", linear, noise),
        ("snf_noise", snf, noise),
    ]


def run_modality_test(config: dict, config_path: str | Path, panel: pd.DataFrame,
                      *, start: str | None = None, end: str | None = None,
                      horizon: int | None = None) -> dict[str, Any]:
    from .config import project_paths

    paths = project_paths(config, config_path)
    horizon = horizon or int(config["settlement"]["horizon_days"])
    oracle = build_labels_oracle(panel, horizon=horizon)
    noise = build_labels_noise(panel, oracle)
    reps = int(config["ablation"]["bootstrap_reps"])
    block = int(config["ablation"]["block_days"])

    # Дивиденды не меняют ОТНОСИТЕЛЬНОЕ сравнение вариантов одинаково
    # начисляясь всем; макро/отчётности влияют только на трекер — тоже общие
    # для всех вариантов, поэтому контроли остаются чистыми. Прокидываем для
    # полноты (симметрично ablation), метрики сравнимы между собой.
    from .history import load_dividends_panel, load_earnings, load_macro_panel
    dividends = load_dividends_panel(paths["data"] / "history", end=end)
    macro = load_macro_panel(paths["data"] / "history", end=end)
    earnings_pairs = [(row["date"], row["asset"])
                      for row in load_earnings(paths["data"] / "history")] or None

    results: dict[str, dict[str, Any]] = {}
    baseline_daily: list[float] | None = None
    for name, mutated, labels in _variants(config, oracle, noise):
        outcome = walk_forward(panel, config=mutated, start=start, end=end,
                               labels=labels, include_daily=True,
                               dividends=dividends if not dividends.empty else None,
                               macro=None if macro.empty else macro,
                               earnings_pairs=earnings_pairs)
        metrics = outcome["strategies"][STRATEGY]
        daily = outcome["daily_returns"][STRATEGY]
        row: dict[str, Any] = {"metrics": metrics, "n_labels": len(labels or [])}
        if baseline_daily is not None and name != "price_only":
            row["p_cagr_better"] = paired_better(daily, baseline_daily,
                                                 reps=reps, block=block,
                                                 metric="cagr")
            # MaxDD: лучше = ниже
            row["p_maxdd_better"] = 1.0 - paired_better(
                daily, baseline_daily, reps=reps, block=block, metric="max_drawdown")
        if name == "price_only":
            baseline_daily = daily
        results[name] = row

    report = {
        "start": start, "end": end, "horizon": horizon,
        "cost_model": {key: config["backtest"][key]
                       for key in ("commission_bps", "spread_bps")},
        "n_assets": int(panel.shape[1]),
        "date_range": [str(panel.index[0])[:10], str(panel.index[-1])[:10]],
        "variants": results,
        "reps": reps, "block": block,
        "verdict": _verdict(results),
    }
    directory = paths["reports"] / "modality"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['date_range'][0]}_{report['date_range'][1]}"
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / f"{stem}.md").write_text(render_modality_markdown(report),
                                          encoding="utf-8")
    report["paths"] = {"markdown": str(directory / f"{stem}.md"),
                       "json": str(directory / f"{stem}.json")}
    return report


def _verdict(results: dict[str, dict]) -> dict[str, str]:
    oracle_p = [results[name].get("p_cagr_better") for name in
                ("linear_oracle", "snf_oracle") if name in results]
    noise_p = [results[name].get("p_cagr_better") for name in
               ("linear_noise", "snf_noise") if name in results]

    def judge(values, low, high):
        if not values or any(v is None for v in values):
            return "недостаточно данных"
        mean = sum(values) / len(values)
        if mean >= high:
            return "ПРОШЁЛ"
        if mean <= low:
            return "ПРОВАЛЕН"
        return "НЕОДНОЗНАЧНО"

    return {
        "oracle (механизм использует модальность)": judge(oracle_p, 0.45, 0.60),
        # noise «не помогает» = P(CAGR>price) не выше ~55%; 0% (гейт отсёк
        # шум, вариант тождественен базе) — идеальный исход, не провал
        "noise (ложная модальность не помогает)": (
            "ПРОШЁЛ" if noise_p and all(v is not None for v in noise_p)
            and sum(noise_p) / len(noise_p) <= 0.55 else "НЕОДНОЗНАЧНО"),
    }


def render_modality_markdown(report: dict) -> str:
    lines = [f"# Исторический тест сентимент-модальности ({report['date_range'][0]} — "
             f"{report['date_range'][1]})",
             "",
             f"Активов: {report['n_assets']}; горизонт oracle-меток: {report['horizon']} дн.; "
             f"парный block-bootstrap: {report['reps']} реплик, блок {report['block']} дн.",
             "Метрики после издержек: комиссия "
             f"{report.get('cost_model', {}).get('commission_bps', 0):g} б.п.; полный спред "
             f"{report.get('cost_model', {}).get('spread_bps', 0):g} б.п. (на сделку половина).",
             "",
             "| Вариант | CAGR | Vol | MaxDD | P(CAGR>price) | P(MaxDD<price) | Меток |",
             "|---|---:|---:|---:|---:|---:|---:|"]

    def fmt(value):
        return "—" if value is None else f"{value:.0%}"

    for name, row in report["variants"].items():
        m = row["metrics"]
        lines.append(f"| {name} | {m['cagr']:+.1%} | {m['ann_vol']:.1%} | "
                     f"{m['max_drawdown']:.1%} | {fmt(row.get('p_cagr_better'))} | "
                     f"{fmt(row.get('p_maxdd_better'))} | {row['n_labels']} |")
    lines += ["", "## Вердикт", ""]
    for check, verdict in report["verdict"].items():
        lines.append(f"- **{check}**: {verdict}")
    lines += ["", "Oracle-метки содержат будущую доходность (look-ahead намеренно): "
              "это проверка того, что механизм fusion действительно проводит "
              "информацию модальности в веса. Noise — то же распределение без "
              "привязки к активу: ложная модальность помогать не должна.",
              ""]
    return "\n".join(lines)
