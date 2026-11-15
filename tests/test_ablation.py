"""A2: сетка вариантов, журнал кампании, bootstrap-CI, resume."""
import json

import pytest
import yaml

from portagent.ablation import (BASELINE, apply_variant, bootstrap_ci,
                                run_ablation, variant_grid)
from portagent.config import load_config

from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_variant_grid_contains_baseline_and_price_only():
    names = [name for name, _ in variant_grid()]
    assert BASELINE in names and "price_only" in names and "mw15" in names


def test_apply_variant_merges_sections():
    config = {"fusion": {"method": "snf", "snf_neighbors": 15, "w_price": 0.7},
              "hrp": {"max_weight": 0.25}}
    mutated = apply_variant(config, {"fusion": {"snf_neighbors": 25},
                                     "hrp": {"max_weight": 0.35}})
    assert mutated["fusion"]["snf_neighbors"] == 25
    assert mutated["fusion"]["method"] == "snf"          # не затёрт
    assert mutated["hrp"]["max_weight"] == 0.35
    assert config["fusion"]["snf_neighbors"] == 15       # оригинал не тронут


def test_bootstrap_ci_reasonable_range():
    rng_returns = [0.001] * 200
    ci = bootstrap_ci(rng_returns, reps=100, block=5)
    assert ci["total_return"]["p05"] <= ci["total_return"]["mean"] + 1e-9
    assert ci["total_return"]["mean"] <= ci["total_return"]["p95"] + 1e-9
    # постоянная дневная 0.1% -> CAGR в районе 0.001*252 с геометрической поправкой
    assert 0.15 <= ci["cagr"]["mean"] <= 0.40


def _config_with_small_reps(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["ablation"]["bootstrap_reps"] = 60
    target = tmp_path / "configs" / "default.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return load_config(target), target


def test_run_ablation_journal_and_resume(tmp_path, mini_fixture):
    from portagent.report.markdown import write_report  # noqa: F401  (прогрев импортов)
    from conftest import synthetic_frames
    from portagent.features.prices import build_panel
    from portagent.collect import moex
    import numpy as np
    import pandas as pd

    # удлиняем мини-fixture до 200 дней через синтетику
    frames = synthetic_frames(days=200)
    panel, _ = build_panel(frames, min_history=60)

    config, config_path = _config_with_small_reps(tmp_path)
    report = run_ablation(config, config_path, panel, labels=[],
                          start=None, end=None, only="snf")
    assert report["n_succeeded"] >= 3                    # k10/k15/k25 (+T10)
    journal = json.loads(Path(report["paths"]["journal"]).read_text(encoding="utf-8").splitlines()[0])
    assert journal["status"] == "running"
    markdown = Path(report["paths"]["markdown"]).read_text(encoding="utf-8")
    assert BASELINE in markdown and "P(лучше базы)" in markdown

    # resume: все succeeded — повторный проход не добавляет running-строк
    lines_before = len(Path(report["paths"]["journal"]).read_text(encoding="utf-8").splitlines())
    run_ablation(config, config_path, panel, labels=[], resume=True, only="snf")
    lines_after = len(Path(report["paths"]["journal"]).read_text(encoding="utf-8").splitlines())
    assert lines_before == lines_after
    # Смена комиссии создаёт новую кампанию, даже при --resume.
    config["backtest"]["commission_bps"] = 15
    changed = run_ablation(config, config_path, panel, labels=[], resume=True, only="snf")
    assert changed["campaign_id"] != report["campaign_id"]
    assert changed["paths"]["journal"] != report["paths"]["journal"]
    assert changed["cost_model"]["commission_bps"] == 15


def test_walk_forward_include_daily():
    from portagent.backtest import walk_forward
    from conftest import synthetic_frames
    from portagent.features.prices import build_panel

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    panel, _ = build_panel(synthetic_frames(days=200), min_history=60)
    with_daily = walk_forward(panel, config=config, include_daily=True)
    without_daily = walk_forward(panel, config=config)
    assert "daily_returns" in with_daily and "daily_returns" not in without_daily
    for name, series in with_daily["daily_returns"].items():
        assert len(series) == with_daily["strategies"][name]["n_days"]
