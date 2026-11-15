"""Модальность изображений: скриншоты графиков, выложенные оператором.

Каталог data/manual/charts/<date>/; имя файла кодирует актив:
`MOEX__SBER.png`, `US_AAPL_daily.jpg` (`:` -> `__`, дальше произвольный тег).
VLM возвращает техническую картину {trend: up|side|down, confidence, note};
структурная валидация: тикер обязан совпасть с именем файла и быть в юниверсе,
trend — из enum. Грандинг-цитаты по изображению детерминированно проверить
нельзя — модальность честно помечена как VLM-мнение.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any

from ..llm.vision import MIME_BY_EXT, vision_chat

CHART_PROMPT = (
    "Перед тобой график цены (и, возможно, объёмов) актива {asset}. Определи "
    "техническую картину на конце графика. Правила: trend — up (восходящий "
    "тренд/пробой), side (боковик/неопределённо) или down (нисходящий тренд/пробой "
    "поддержки); confidence — уверенность 0..1; note — одна фраза с качественным "
    "наблюдением (направление, паттерн). В note не пиши чисел, ценовых уровней, "
    "процентов, дат и целей: их достоверность здесь не проверяется. Не выдумывай "
    "данные вне графика. Вывод — только JSON."
)

CHART_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "asset": {"type": "string"},
        "trend": {"type": "string"},
        "confidence": {"type": "number"},
        "note": {"type": "string"},
    },
    "required": ["asset", "trend", "confidence", "note"],
}

STANCE = {"up": 0.5, "side": 0.0, "down": -0.5}
_NAME_RE = re.compile(r"^(?P<provider>[A-Z]+)__(?P<ticker>[A-Z0-9.\-]+?)(?:_[A-Za-z0-9\-]+)?$")


def scan_chart_drops(manual_dir: Path, run_date: str) -> list[tuple[str, Path]]:
    """-> [(asset_id, путь)] из data/manual/charts/<date>/."""
    directory = Path(manual_dir) / run_date
    if not directory.is_dir():
        return []
    drops: list[tuple[str, Path]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in MIME_BY_EXT or not path.is_file():
            continue
        match = _NAME_RE.match(path.stem)
        if not match:
            continue
        drops.append((f"{match.group('provider')}:{match.group('ticker')}", path))
    return drops


def _parse_chart_response(raw: str, *, expected_asset: str,
                          universe: list[str]) -> dict[str, Any] | None:
    try:
        document = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(document, dict):
        return None
    asset = str(document.get("asset", ""))
    trend = str(document.get("trend", "")).lower()
    if asset != expected_asset or expected_asset not in universe:
        return None
    if trend not in STANCE:
        return None
    try:
        confidence = float(document.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    note = str(document.get("note", "")).strip()[:300]
    if re.search(r"\d", note):
        return None  # VLM не является источником числовых котировок/уровней.
    return {"asset": expected_asset, "trend": trend, "confidence": confidence,
            "stance": STANCE[trend] * confidence, "note": note}


def analyze_charts(drops: list[tuple[str, Path]], *, universe: list[str],
                   endpoint: str, model: str, timeout: float, seed: int = 20261112,
                   cache_dir: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Разбор каждого скриншота VLM -> (labels, statuses); сбой одного
    изображения не роняет остальные."""
    labels: list[dict] = []
    statuses: list[dict] = []
    for asset, path in drops:
        try:
            image = path.read_bytes()
            key = hashlib.sha256(image + json.dumps([asset, model, seed, 1024, CHART_PROMPT,
                                                   CHART_SCHEMA], sort_keys=True).encode()).hexdigest()
            cached = Path(cache_dir) / f"{key}.json" if cache_dir else None
            if cached and cached.is_file():
                try:
                    label = json.loads(cached.read_text(encoding="utf-8"))
                    if _parse_chart_response(json.dumps(label), expected_asset=asset, universe=universe):
                        label["file"] = path.name
                        labels.append(label)
                        statuses.append({"asset": asset, "file": path.name, "ok": True, "cache_hit": True})
                        continue
                except (ValueError, OSError):
                    pass
            raw, meta = vision_chat(endpoint, model, image,
                                    MIME_BY_EXT[path.suffix.lower()],
                                    CHART_PROMPT.format(asset=asset),
                                    schema=CHART_SCHEMA, timeout=timeout, seed=seed)
            label = _parse_chart_response(raw, expected_asset=asset, universe=universe)
            if label is None:
                statuses.append({"asset": asset, "file": path.name, "ok": False,
                                 "error": "ответ VLM не прошёл структурную валидацию"})
                continue
            label["file"] = path.name
            label["meta"] = {"latency_s": meta.get("latency_s"),
                             "image_sha": meta.get("image_sha")}
            labels.append(label)
            if cached:
                from ..scheduler import _write_json
                _write_json(cached, label)
            statuses.append({"asset": asset, "file": path.name, "ok": True,
                             "error": None})
        except Exception as error:  # LocalLLMError и прочее — модальность деградирует
            statuses.append({"asset": asset, "file": path.name, "ok": False,
                             "error": str(error)[:200]})
    return labels, statuses
