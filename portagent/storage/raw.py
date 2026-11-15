"""Immutable raw-архив с SHA-256 манифестами.

Каждый сбор пишется в data/raw/<date>/<name>; файлы никогда не перезаписываются
(повтор того же имени с другим содержимым — ошибка). Манифест фиксирует
fetched_at, источник и SHA каждого файла.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def short_id(text: str) -> str:
    """Короткий стабильный идентификатор строки (для id новостей)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def archive(root: Path, date: str, name: str, payload: bytes) -> dict:
    """Сохраняет payload в raw-архив; возвращает запись манифеста.

    Имя контент-адресуемое: `<stem>__<sha8>.<ext>`. Идентичные байты ->
    тот же файл (reused=True); изменившийся intraday ответ провайдера ->
    новый файл рядом со старым, ничего не перезаписывается.
    """
    digest = sha256_bytes(payload)
    stem, _, extension = name.partition(".")
    suffix = f".{extension}" if extension else ""
    target = Path(root) / date / f"{stem}__{digest[:8]}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    reused = target.exists()
    if not reused:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(target)
    return {"name": target.name, "sha256": digest, "path": str(target), "reused": reused}


def write_manifest(root: Path, date: str, kind: str, entries: list[dict], meta: dict | None = None) -> Path:
    directory = Path(root) / date
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": kind,
        "date": date,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "entries": entries,
    }
    path = directory / f"manifest_{kind}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
