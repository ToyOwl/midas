"""Журнал решений: append-only jsonl с SHA весов."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def append_ledger(data_dir: Path, record: dict) -> Path:
    ledger = Path(data_dir) / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    weights_blob = json.dumps(record.get("weights", {}), ensure_ascii=False, sort_keys=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **record,
        "weights_sha": hashlib.sha256(weights_blob.encode("utf-8")).hexdigest()[:16],
    }
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return ledger
