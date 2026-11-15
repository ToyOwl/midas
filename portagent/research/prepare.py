"""Отдельный воспроизводимый снимок цен непосредственно от провайдеров."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..collect import moex
from ..collect.yf_provider import fetch_us_candles
from ..config import load_config, project_paths
from ..scheduler import _write_json
from ..storage.raw import archive


def prepare(config, config_path, day, days=2700):
    paths = project_paths(config, config_path)
    root = paths["data"] / "research" / "top3" / f"market-{day}-{days}"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return root
    assets = ["MOEX:" + a for a in config["universe"]["moex"]] + ["US:" + a for a in config["universe"]["us"]]
    def collect(asset):
        ticker = asset.split(":")[1]
        target = root / (asset.replace(":", "_") + ".parquet")
        metadata = target.with_suffix(".json")
        if metadata.exists():
            row = json.loads(metadata.read_text(encoding="utf-8"))
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
            return row
        if asset.startswith("MOEX:"):
            cfg = config["providers"]["moex"]
            payload = moex.fetch_candles_raw(ticker, board=cfg["board"],
                from_date=(pd.Timestamp(day) - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                till_date=f"{day} 23:59:59", timeout=cfg["timeout_s"], retries=cfg["retries"])
            frame, ext = moex.parse_candles(payload), "json"
        else:
            frame, csv = fetch_us_candles(ticker, history_days=days)
            payload, ext = csv.encode(), "csv"
        frame = frame.loc[frame.index <= day]
        if len(frame) < 500:
            raise ValueError(f"{asset}: история слишком короткая: {len(frame)}")
        entry = archive(root / "raw", day, asset.replace(":", "_") + "." + ext, payload)
        frame.to_parquet(target)
        row = {"asset": asset, "file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "raw": entry, "start": str(frame.index.min()), "end": str(frame.index.max()), "rows": len(frame)}
        _write_json(metadata, row)
        return row
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(collect, asset): asset for asset in assets}
        for future in as_completed(futures):
            asset = futures[future]
            try:
                row = future.result(); rows.append(row)
                print(asset, row["rows"], row["start"], row["end"], flush=True)
            except Exception as error:
                errors.append({"asset": asset, "error": str(error)})
                print(asset, "ERROR", str(error)[:200], flush=True)
    if errors:
        _write_json(root / "collection_errors.json", errors)
        raise RuntimeError(f"Не получен полный юниверс: {len(errors)} ошибок; повтор использует сохранённые серии")
    _write_json(manifest_path, {"day": day, "days": days, "source": "fresh_provider_snapshot",
        "assets": sorted(rows, key=lambda r: r["asset"]),
        "limitations": ["Фиксированный текущий юниверс: survivorship bias", "MOEX: close без корректировки корпоративных действий", "Пересмотр исторических данных провайдером возможен"]})
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--date", required=True)
    parser.add_argument("--days", type=int, default=2700)
    args = parser.parse_args()
    print(prepare(load_config(args.config), args.config, args.date, args.days), flush=True)
