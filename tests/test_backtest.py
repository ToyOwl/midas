"""Walk-forward: без меток fused совпадает с price-only; отчёт пишется."""
from pathlib import Path

import pytest

from portagent.backtest import walk_forward
from portagent.config import load_config
from portagent.features.prices import build_panel
from portagent.report.markdown import render_markdown, write_backtest
from conftest import synthetic_frames

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_fused_matches_price_only_without_labels():
    config = load_config(CONFIG)
    config["tracker"]["window_days"] = 60
    panel, _dropped = build_panel(synthetic_frames(days=160, seed=1))
    result = walk_forward(panel, config=config, rebalance_days=21)
    assert result["n_rebalances"] >= 2
    assert result["strategies"]["fused"]["total_return"] == pytest.approx(
        result["strategies"]["price_only"]["total_return"], abs=1e-9)
    assert result["strategies"]["fused"]["n_days"] > 0
    brier = result["tracker"]["brier"]
    assert brier is not None and 0.0 <= brier <= 1.0


def test_backtest_report_files(tmp_path):
    config = load_config(CONFIG)
    config["tracker"]["window_days"] = 60
    panel, _dropped = build_panel(synthetic_frames(days=160, seed=2))
    result = walk_forward(panel, config=config, rebalance_days=21)
    paths = write_backtest(tmp_path, result)
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "HRP price-only" in text
    assert "Brier" in text
    assert paths["json"].is_file()


def test_daily_report_shows_modalities_and_tracker():
    text = render_markdown({
        "date": "2026-11-12",
        "fusion": {"mode": "snf", "method": "snf", "views_used": ["price", "sentiment"],
                   "note": "SNF сливает виды."},
        "weights": {"MOEX:SBER": 0.5},
        "portfolio": {"cash_share": 0.1, "max_weight": 0.25, "linkage_method": "single",
                      "clusters": {}},
        "coverage": {"coverage_frac": 0.8},
        "news_total": 10,
        "llm": {"labels_total": 4},
        "legal": {"cases_total": 1, "coverage_frac": 0.1, "labels_total": 1},
        "vision": {"charts_found": 0, "labels": [], "statuses": []},
        "tracker": {"enabled": True, "overlay": False, "warn_threshold": 0.5,
                    "rows": [{"asset": "MOEX:SBER", "score": 0.72, "p_tail": 0.04,
                              "drawdown": 0.12, "sentiment_z": -1.1, "legal_flag": 0,
                              "warned": True}]},
        "critique": {"summary": "ok", "findings": []},
        "provider_statuses": [],
        "feed_statuses": [],
    })
    assert "## Паспорт" in text
    assert "## Покрытие модальностей" in text
    assert "sentiment" in text
    assert "## Трекер редких событий" in text
    assert "MOEX:SBER" in text
    assert "0.72" in text
