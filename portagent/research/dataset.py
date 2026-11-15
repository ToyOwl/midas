"""Causal features and next-real-observation targets; no filled-price targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..tracker.top3 import event_features


def load_snapshot(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for row in manifest["assets"]:
        path = root / row["file"]
        if path.parent.resolve() != root.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Снимок цен изменён")
        frames[row["asset"]] = pd.read_parquet(path)
    common_start = max(f.index.min() for f in frames.values())
    return {a: f.loc[f.index >= common_start].copy() for a, f in frames.items()}, manifest


def make_dataset(frames, *, threshold=2.5, halflife=20, window=250):
    rows = []
    assets = sorted(frames)
    for asset, original in sorted(frames.items()):
        frame = original.copy().sort_index()
        if frame.index.has_duplicates:
            raise ValueError("Повторные торговые даты")
        frame = frame.loc[np.isfinite(frame.close) & (frame.close > 0)]
        close = frame.close.astype(float)
        panel = event_features(close, threshold=threshold, halflife=halflife, window=window)
        f = panel["features"].copy()
        returns = panel["returns"]
        sigma = returns.ewm(halflife=halflife, min_periods=20).std().replace(0, np.nan)
        for n in (1, 5, 20, 60):
            f[f"momentum_{n}"] = returns.rolling(n).sum() / (sigma * np.sqrt(n))
        f["sigma"] = sigma
        f["downside_frequency"] = (panel["z"] < -threshold).astype(float).where(panel["z"].notna()).rolling(window, min_periods=60).mean()
        f["drawdown_60"] = 1 - close / close.rolling(60, min_periods=60).max()
        volume = frame.get("volume", pd.Series(np.nan, index=frame.index)).replace(0, np.nan)
        f["volume_ratio"] = np.log(volume / volume.rolling(20, min_periods=10).median())
        f["volume_missing"] = volume.isna().astype(float)
        f["moex"] = float(asset.startswith("MOEX:"))
        for name in assets:
            f["asset_" + name.replace(":", "_")] = float(asset == name)
        f["date"], f["asset"] = f.index.astype(str), asset
        dates = pd.Series(f.index.astype(str), index=f.index)
        f["target_date"] = dates.shift(-1)
        f["target_z"] = returns.shift(-1) / sigma
        f["target_return"] = returns.shift(-1)
        f["event"] = (f.target_z.abs() >= threshold).astype(float).where(f.target_z.notna())
        f["target_gap_days"] = (pd.to_datetime(f.target_date) - pd.to_datetime(f.date)).dt.days
        f["rows_seen"] = np.arange(len(f))
        rows.append(f.loc[f.rows_seen >= 250])
    data = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["frequency", "sigma"]).sort_values(["date", "asset"]).reset_index(drop=True)
    excluded = {"date", "asset", "target_date", "target_z", "target_return", "event", "target_gap_days", "rows_seen"}
    features = [c for c in data if c not in excluded]
    return data, features


def split_fold(data, year, *, validation_days=90, calibration_days=180):
    start = pd.Timestamp(f"{year}-01-01")
    cal = (start - pd.Timedelta(days=calibration_days)).strftime("%Y-%m-%d")
    valid = (pd.Timestamp(cal) - pd.Timedelta(days=validation_days)).strftime("%Y-%m-%d")
    beginning, ending = start.strftime("%Y-%m-%d"), f"{year + 1}-01-01"
    complete = data[data.event.notna() & data.target_date.notna()]
    # Every outcome in a fitting block must have resolved strictly before the next block.
    train = complete[(complete.date < valid) & (complete.target_date < valid)]
    val = complete[(complete.date >= valid) & (complete.date < cal) & (complete.target_date < cal)]
    calibration = complete[(complete.date >= cal) & (complete.date < beginning) & (complete.target_date < beginning)]
    test = complete[(complete.date >= beginning) & (complete.date < ending)]
    for name, block in (("train", train), ("validation", val), ("calibration", calibration), ("test", test)):
        if len(block) < 100 or block.event.nunique() < 2:
            raise ValueError(f"{year}: недостаточно данных в {name}")
    return train, val, calibration, test
