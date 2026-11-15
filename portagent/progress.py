"""Небольшие атомарные снимки прогресса для фоновых заданий панели."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def emit(stage: str, message: str, **details) -> None:
    target = os.environ.get("PORTAGENT_PROGRESS_PATH")
    if not target:
        return
    path = Path(target)
    value = {"stage": stage, "message": message, "at": datetime.now(timezone.utc).isoformat(), **details}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass  # Наблюдаемость не влияет на расчёт.
