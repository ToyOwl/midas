"""Долговечная очередь: SQLite, разрешённые действия, идемпотентность."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

TERMINAL = {"succeeded", "degraded", "failed", "cancelled", "interrupted"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate(kind: str, args: dict) -> dict:
    fields = {"daily": {"use_llm"}, "backtest": {"start", "end", "rebalance_days", "commission_bps", "spread_bps"},
              "ask": {"date", "question"}, "review": {"date"}, "recover_news": {"date"},
              "refresh_sources": {"date"}, "collect_telegram": set(), "train_models": {"date"}, "model": {"kind"}}
    fields.update(analyze_material={"id"}, preview_materials={"date", "ids", "influence"})
    if kind not in fields or not isinstance(args, dict) or set(args) - fields[kind]:
        raise ValueError("неизвестное действие или параметр")
    if kind == "daily":
        value = args.get("use_llm", True)
        if not isinstance(value, bool):
            raise ValueError("use_llm должен быть логическим значением")
        return {"use_llm": value}
    if kind == "model":
        if args.get("kind") not in {"text", "vision"}:
            raise ValueError("выберите text или vision")
    if kind == "analyze_material":
        from ..materials.storage import valid_id
        valid_id(args.get("id"))
    if kind == "preview_materials":
        from ..materials.storage import valid_id
        ids = args.get("ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 20 or len(set(ids)) != len(ids):
            raise ValueError("Выберите от 1 до 20 материалов")
        for item in ids:
            valid_id(item)
        value = args.get("influence")
        if type(value) not in (int, float) or not 0 < value <= .3:
            raise ValueError("Влияние материалов — от 0 до 30%")
    for field in ("date", "start", "end"):
        if args.get(field):
            if not isinstance(args[field], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args[field]):
                raise ValueError("дата должна иметь вид YYYY-MM-DD")
            date.fromisoformat(args[field])
    if kind in {"ask", "review", "recover_news", "refresh_sources", "train_models", "preview_materials"} and not args.get("date"):
        raise ValueError("выберите дату портфеля")
    if kind == "ask" and (not isinstance(args.get("question"), str) or not 3 <= len(args["question"].strip()) <= 1500):
        raise ValueError("вопрос должен содержать от 3 до 1500 символов")
    if kind == "backtest":
        for field, limit in (("rebalance_days", 260), ("commission_bps", 500), ("spread_bps", 2000)):
            if field in args:
                v = args[field]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= limit:
                    raise ValueError(f"некорректное значение {field}")
                if field == "rebalance_days" and (not isinstance(v, int) or v < 1):
                    raise ValueError("период ребаланса — целое число от 1 до 260")
        if args.get("start") and args.get("end") and args["start"] > args["end"]:
            raise ValueError("начало периода позже конца")
    return args


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, args TEXT NOT NULL,
                idempotency TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
                created TEXT NOT NULL, started TEXT, finished TEXT, pid INTEGER,
                result TEXT, error TEXT)""")

    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def unpack(row):
        if row is None:
            return None
        value = dict(row)
        for key in ("args", "result"):
            value[key] = json.loads(value[key]) if value.get(key) else None
        value.pop("idempotency", None)
        return value

    def submit(self, kind: str, args: dict, key: str) -> dict:
        args = validate(kind, args)
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,100}", key):
            raise ValueError("нужен корректный ключ идемпотентности")
        packed = json.dumps(args, ensure_ascii=False, sort_keys=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE idempotency=?", (key,)).fetchone()
            if old:
                if old["kind"] != kind or old["args"] != packed:
                    raise ValueError("ключ уже использован для другого задания")
                return self.unpack(old)
            # Повторный клик/другая вкладка не размножает одинаковое активное задание.
            same = db.execute("SELECT * FROM jobs WHERE kind=? AND args=? AND status IN ('queued','running')",
                              (kind, packed)).fetchone()
            if same:
                return self.unpack(same)
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 8:
                raise ValueError("очередь заполнена; дождитесь выполнения текущих заданий")
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,kind,args,idempotency,status,created) VALUES(?,?,?,?,?,?)",
                       (job_id, kind, packed, key, "queued", now()))
            return self.unpack(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def get(self, job_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("некорректный идентификатор задания")
        with self.connect() as db:
            return self.unpack(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit=40):
        with self.connect() as db:
            return [self.unpack(row) for row in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]

    def update(self, job_id: str, **values):
        allowed = {"status", "started", "finished", "pid", "result", "error"}
        if set(values) - allowed:
            raise ValueError("неизвестные поля состояния")
        if "result" in values:
            values["result"] = json.dumps(values["result"], ensure_ascii=False)
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?",
                       (*values.values(), job_id))

    def cancel(self, job_id: str) -> bool:
        self.get(job_id)
        with self.connect() as db:
            return bool(db.execute("UPDATE jobs SET status='cancelled', finished=? WHERE id=? AND status='queued'",
                                   (now(), job_id)).rowcount)
