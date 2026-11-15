"""Одноразовая проверка: чувствительность SNF к kNN-ядру на реальной панели."""
import copy

from portagent.ablation import paired_better
from portagent.backtest import walk_forward
from portagent.config import load_config
from portagent.history import load_price_panel
from portagent.modality_test import _variants, build_labels_noise, build_labels_oracle

config = load_config("configs/default.yaml")
panel = load_price_panel("data/history")
oracle = build_labels_oracle(panel, horizon=5)
noise = build_labels_noise(panel, oracle)
base_daily = None
print(f"панель {panel.index[0]}..{panel.index[-1]}, {panel.shape[1]} активов")
for k in (5, 8, 12):
    cfg = copy.deepcopy(config)
    cfg["fusion"]["snf_neighbors"] = k
    for name, mutated, labels in _variants(cfg, oracle, noise):
        out = walk_forward(panel, config=mutated, include_daily=True, labels=labels)
        metrics = out["strategies"]["fused"]
        daily = out["daily_returns"]["fused"]
        if name == "price_only":
            base_daily = daily
            p = "—"
        else:
            p = f"{paired_better(daily, base_daily, reps=200, block=5):.0%}"
        print(f"k={k:<3} {name:<14} CAGR {metrics['cagr']:+8.2%}  "
              f"MaxDD {metrics['max_drawdown']:6.1%}  P {p}")
