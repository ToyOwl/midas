"""Исторический тест модальности: oracle обязан помогать, noise — нет.

Синтетика с известной будущей структурой: 3 кластера, в середине серии
кластеры «меняются ролями» (факторы перетасовываются) — oracle-метки это
предвидят, price-only видит только смесь двух режимов.
"""
import numpy as np
import pandas as pd
import pytest

from portagent.modality_test import (build_labels_noise, build_labels_oracle,
                                     run_modality_test)

CONFIG_PATH = None  # заполняется фикстурой


def _panel(days: int = 320, regime_at: int | None = 160, seed: int = 21) -> pd.DataFrame:
    """regime_at — день смены кластерных ролей; None — стабильная структура
    (используется для отрицательного контроля: сегмент шума без сигнала)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(end="2026-11-12", periods=days, freq="B").strftime("%Y-%m-%d")
    assets = ["A1", "A2", "B1", "B2", "C1", "C2"]
    factors = [rng.normal(0, 0.012, days) for _ in range(3)]
    order = {"A1": 0, "A2": 0, "B1": 1, "B2": 1, "C1": 2, "C2": 2}
    permuted = {"A1": 1, "A2": 1, "B1": 2, "B2": 2, "C1": 0, "C2": 0}
    closes = {}
    for asset in assets:
        series = []
        for t in range(days):
            factor_index = order[asset]
            if regime_at is not None and t >= regime_at:
                factor_index = permuted[asset]
            series.append(factors[factor_index][t] + rng.normal(0, 0.004))
        closes[asset] = 100.0 * np.exp(np.cumsum(series))
    return pd.DataFrame(closes, index=index)


def _config(tmp_path):
    import yaml
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["ablation"]["bootstrap_reps"] = 80          # быстро в тестах
    target = tmp_path / "configs" / "default.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config, target


def test_oracle_labels_lookahead_by_design():
    panel = _panel(days=80)
    labels = build_labels_oracle(panel, horizon=5)
    assert labels, "метки должны строиться"
    by_key = {(l["date"], l["ticker"]): l["sentiment"] for l in labels}
    # сентимент коррелирует с фактической будущей доходностью
    log_ret = np.log(panel / panel.shift(1))
    fwd = np.log(panel.shift(-5) / panel)
    pairs = []
    for (date, asset), sentiment in by_key.items():
        if asset in fwd.columns and date in fwd.index:
            value = fwd.loc[date, asset]
            if np.isfinite(value):
                pairs.append((value, sentiment))
    assert len(pairs) > 80
    fwd_values, sentiments = zip(*pairs)
    correlation = np.corrcoef(fwd_values, sentiments)[0, 1]
    assert correlation > 0.5


def test_oracle_test_segment_works():
    """Позитивный контроль обязан проходить на своей территории:
    oracle-метки, привязанные к дате в сегменте смены режима."""
    panel = _panel()                      # 320 дней, смена режима в середине
    oracle = build_labels_oracle(panel, horizon=5)
    noise = build_labels_noise(panel, oracle, seed=3)
    # корреляция oracle с будущим выше, чем у noise
    log_ret = np.log(panel / panel.shift(1))
    fwd = np.log(panel.shift(-5) / panel)

    def alignment(labels):
        pairs = [(fwd.loc[l["date"], l["ticker"]], l["sentiment"]) for l in labels
                 if l["ticker"] in fwd.columns and l["date"] in fwd.index
                 and np.isfinite(fwd.loc[l["date"], l["ticker"]])]
        a, b = zip(*pairs)
        return float(np.corrcoef(a, b)[0, 1])

    assert alignment(oracle) > 0.5
    assert abs(alignment(noise)) < 0.2


def test_noise_labels_preserve_distribution_destroy_alignment():
    panel = _panel(days=80)
    oracle = build_labels_oracle(panel, horizon=5)
    noise = build_labels_noise(panel, oracle, seed=3)
    assert sorted(l["sentiment"] for l in oracle) == pytest.approx(
        sorted(l["sentiment"] for l in noise), abs=1e-12)
    # привязка к (дата, актив) разрушена: хотя бы половина пар изменила значение
    changed = sum(1 for a, b in zip(oracle, noise) if a["sentiment"] != b["sentiment"])
    assert changed > len(oracle) * 0.9


def test_modality_test_mechanism_on_synthetic(tmp_path):
    """Механизм: oracle-структура проходит в веса — вердикт не ПРОВАЛЕН;
    noise не даёт устойчивого преимущества (P около 50%)."""
    config, config_path = _config(tmp_path)
    panel = _panel()
    report = run_modality_test(config, config_path, panel)
    variants = report["variants"]
    assert set(variants) >= {"price_only", "linear_oracle", "snf_oracle",
                             "linear_noise", "snf_noise"}
    from pathlib import Path
    assert Path(report["paths"]["markdown"]).exists()

    oracle_p = [variants[n]["p_cagr_better"] for n in ("linear_oracle", "snf_oracle")]
    noise_p = [variants[n]["p_cagr_better"] for n in ("linear_noise", "snf_noise")]
    # oracle со знанием будущей структуры не должен проигрывать price-only
    assert all(p is not None for p in oracle_p + noise_p)
    assert max(oracle_p) >= 0.5, f"oracle не помогает: {oracle_p}"
    # шум не должен стабильно помогать
    assert max(noise_p) <= 0.75, f"шум «помогает» — утечка: {noise_p}"
