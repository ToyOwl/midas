"""Immutable uploads and metadata. No user supplied filesystem paths."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import warnings
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from PIL import Image, ImageOps

KINDS = {"auto", "chart", "document", "table", "news", "photo", "other"}
MSK = timezone(timedelta(hours=3))


def now():
    return datetime.now(timezone.utc).isoformat()


def today():
    return datetime.now(MSK).date()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def packed(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def immutable(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("Сохранённый материал нельзя перезаписать другим содержимым")


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Некорректный идентификатор материала")
    return value


def read(root, folder, item_id):
    return json.loads((Path(root) / folder / (valid_id(item_id) + ".json")).read_text(encoding="utf-8"))


def universe(config):
    return ["MOEX:" + a for a in config["universe"]["moex"]] + ["US:" + a for a in config["universe"]["us"]]


def upload(root, data, config):
    required = {"name", "content", "assets", "kind", "source_date", "note"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("Нужны файл, бумаги, тип, дата и заметка")
    if (not isinstance(data["assets"], list) or not data["assets"] or len(data["assets"]) > 20
            or any(a not in universe(config) for a in data["assets"])):
        raise ValueError("Выберите связанные с изображением бумаги")
    if data["kind"] not in KINDS:
        raise ValueError("Неизвестный тип материала")
    day = date.fromisoformat(data["source_date"])
    if day > today():
        raise ValueError("Дата материала не может быть в будущем")
    if not isinstance(data["note"], str) or len(data["note"]) > 800:
        raise ValueError("Заметка — до 800 символов")
    if not isinstance(data["name"], str) or len(data["name"]) > 250:
        raise ValueError("Слишком длинное имя файла")
    try:
        original = base64.b64decode(data["content"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Не удалось прочитать изображение") from error
    cfg = config["materials"]
    if not 0 < len(original) <= cfg["max_upload_mb"] * 1024 * 1024:
        raise ValueError(f"Файл должен быть не больше {cfg['max_upload_mb']} МБ")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(original)) as probe:
                if probe.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Поддерживаются PNG, JPEG и WebP")
                if probe.width * probe.height > cfg["max_pixels"] or min(probe.size) < 16:
                    raise ValueError("Размер изображения: от 16 пикселей, до 20 мегапикселей")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ValueError("Загрузите неподвижное изображение")
                probe.verify()
            with Image.open(io.BytesIO(original)) as raw:
                im = ImageOps.exif_transpose(raw).convert("RGB")
                original_size = list(im.size)
                im.thumbnail((1536, 1536))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                prepared = buf.getvalue()
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Повреждённое или слишком большое изображение") from error
    sha = digest(original)
    metadata = {"original_sha": sha, "assets": sorted(set(data["assets"])), "kind": data["kind"],
                "source_date": day.isoformat(), "note": data["note"].strip()}
    item_id = digest(packed(metadata))
    root = Path(root)
    from ..llm.runtime import launch_lock
    with launch_lock(root / "upload.lock"):
        path = root / "records" / (item_id + ".json")
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        immutable(root / "originals" / (sha + ".bin"), original)
        immutable(root / "images" / (item_id + ".png"), prepared)
        record = {**metadata, "id": item_id, "name": data["name"], "uploaded_at": now(),
                  "image_sha": digest(prepared), "original_size": original_size, "size": list(im.size)}
        immutable(path, packed(record))
    return record


def list_materials(root):
    root = Path(root)
    result = []
    for path in sorted((root / "records").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        record = json.loads(path.read_text(encoding="utf-8"))
        analysis = root / "analyses" / (record["id"] + ".json")
        if analysis.is_file():
            value = json.loads(analysis.read_text(encoding="utf-8"))
            record["analysis"] = {k: v for k, v in value.items() if k != "embedding"}
            record["analysis"]["embedding_info"] = {k: v for k, v in value["embedding"].items() if k != "vector"}
        result.append(record)
    return result
