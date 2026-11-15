"""Saved, deterministic price observations. No forecasts or portfolio mutations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..history import load_price_panel

KINDS = {"hold_gain": "Удержится ли рост", "recovery": "Продолжится ли восстановление",
         "large_move": "Будет ли резкое движение"}
TERMINAL = {"held", "recovered", "not_recovered", "broken", "triggered", "quiet", "stopped"}
MSK = timezone(timedelta(hours=3))


def today():
    return datetime.now(MSK).date().isoformat()


def clean_series(series, end):
    values = pd.to_numeric(series, errors="coerce").copy()
    values.index = values.index.astype(str).str[:10]
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values[(values.index <= end) & values.map(lambda x: pd.notna(x) and math.isfinite(x) and x > 0)]


def price_series(paths, config, end):
    """Prefer verified daily snapshots: accumulated history may include demo rows.

    Never mix a complete archived series with extra dates from a different source.
    A failed asset keeps its previous snapshot and an explicit old last date.
    """
    assets = [f"{market}:{a}" for market, key in (("MOEX", "moex"), ("US", "us"))
              for a in config["universe"][key]]
    manifests = sorted((paths["data"] / "raw").glob("????-??-??/manifest_market.json"), reverse=True)
    manifests = [p for p in manifests if p.parent.name <= end]
    if manifests:
        from ..report.recovery import archived_prices
        result = {}
        for manifest in manifests:
            if json.loads(manifest.read_text(encoding="utf-8")).get("meta", {}).get("source") == "fixture":
                continue
            frames = archived_prices(paths, manifest.parent.name)
            for asset in assets:
                if asset not in result and asset in frames:
                    close = clean_series(frames[asset]["close"], end)
                    if not close.empty:
                        result[asset] = close
            if len(result) == len(assets):
                break
        return result
    panel = load_price_panel(paths["data"] / "history", end=end)
    result = {a: clean_series(panel[a], end) for a in assets if a in panel}
    return {a: s for a, s in result.items() if not s.empty}


def make_observation(args, series, end):
    if set(args) != {"asset", "kind", "sessions", "threshold_pct"}:
        raise ValueError("Нужны бумага, условие, срок и порог")
    asset, kind = args["asset"], args["kind"]
    if not isinstance(asset, str) or asset not in series:
        raise ValueError("Нет сохранённых котировок для этой бумаги")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("Неизвестное условие наблюдения")
    if type(args["sessions"]) is not int or args["sessions"] not in (3, 5, 10):
        raise ValueError("Срок: 3, 5 или 10 торговых наблюдений")
    threshold = args["threshold_pct"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not .1 <= threshold <= 50:
        raise ValueError("Порог должен быть от 0,1 до 50 процентов")
    close = clean_series(series[asset], end)
    if len(close) < 6:
        raise ValueError("Для условия нужны минимум шесть закрытий")
    anchor, before = float(close.iloc[-1]), float(close.iloc[-6])
    if kind == "hold_gain" and anchor <= before:
        raise ValueError("За последние пять наблюдений роста нет; выберите другое условие")
    if kind == "recovery" and anchor >= before:
        raise ValueError("За последние пять наблюдений снижения нет; выберите другое условие")
    key = {**args, "anchor_date": str(close.index[-1]), "anchor": anchor, "base": before, "version": 1}
    ident = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
    return {**key, "id": ident, "created_at": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold / 100, "target": (anchor + before) / 2 if kind != "large_move" else None,
            "points": [], "status": "waiting", "data_warning": None}


def evaluate(record, close, end):
    """Freeze first-seen closes; final outcomes cannot change after the horizon."""
    r = copy.deepcopy(record)
    if r["status"] in TERMINAL:
        return r
    close = clean_series(close, end)
    r["data_warning"] = None
    if close.empty:
        r["data_warning"] = "Источник пока не дал котировок этой бумаги. Результат не определён."
        return r
    # Adjusted history may change after splits/dividends. Mixing price bases would
    # create artificial breaks: require compatible overlapping observations.
    frozen = [[r["anchor_date"], r["anchor"]], *r["points"]]
    overlap = [(d, v) for d, v in frozen if d in close.index]
    if not overlap or any(not math.isclose(float(close.loc[d]), v, rel_tol=1e-6, abs_tol=1e-8) for d, v in overlap):
        r["data_warning"] = "История цен пересмотрена или нет общей контрольной даты. Проверка приостановлена; создайте новое наблюдение после проверки источника."
        return r
    last_date = frozen[-1][0]
    # A newly backfilled session before an already frozen close changes the
    # meaning of the horizon; do not silently splice it into an existing result.
    known = {d for d, _ in frozen}
    if any(d not in known for d in close.index if r["anchor_date"] < d <= last_date):
        r["data_warning"] = "Источник добавил пропущенную торговую запись. Проверка приостановлена; начните новое наблюдение по полной истории."
        return r
    remaining = r["sessions"] - len(r["points"])
    r["points"] += [[str(d), float(v)] for d, v in close.loc[close.index > last_date].iloc[:remaining].items()]
    peak, prev = r["anchor"], r["anchor"]
    for i, (day, price) in enumerate(r["points"]):
        peak = max(peak, price)
        boundary = peak * (1 - r["threshold"])
        if r["kind"] == "large_move":
            hit = abs(price / prev - 1) >= r["threshold"] - 1e-12
        else:
            hit = price < boundary - 1e-10 or (r["kind"] == "hold_gain" and price < r["target"] - 1e-10)
        if hit:
            r.update(status="triggered" if r["kind"] == "large_move" else "broken", resolved_date=day)
            r["points"] = r["points"][:i + 1]
            return r
        prev = price
    if len(r["points"]) == r["sessions"]:
        status = ("held" if r["kind"] == "hold_gain" else "quiet" if r["kind"] == "large_move"
                  else "recovered" if prev >= r["target"] else "not_recovered")
        r.update(status=status, resolved_date=r["points"][-1][0])
    else:
        r["status"] = "watching" if r["points"] else "waiting"
    return r


def describe(r, end):
    value = copy.deepcopy(r)
    points = [[r["anchor_date"], r["anchor"]], *r["points"]]
    value.update(title=KINDS[r["kind"]], completed=len(r["points"]),
                 last_date=points[-1][0], last_price=points[-1][1],
                 change=points[-1][1] / r["anchor"] - 1,
                 boundary=max(v for _, v in points) * (1 - r["threshold"]),
                 stale_days=(datetime.fromisoformat(end) - datetime.fromisoformat(points[-1][0])).days,
                 currency="RUB" if r["asset"].startswith("MOEX:") else "USD")
    return value


class Observations:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "observations.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, body TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def write(db, r):
        db.execute("INSERT OR REPLACE INTO observations VALUES (?, ?)",
                   (r["id"], json.dumps(r, ensure_ascii=False, allow_nan=False)))

    def create(self, args, series, end):
        record = make_observation(args, series, end)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT body FROM observations WHERE id=?", (record["id"],)).fetchone()
            if old:
                return json.loads(old[0])
            self.write(db, record)
        return record

    def refresh(self, series, end):
        items = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for (body,) in db.execute("SELECT body FROM observations").fetchall():
                old = json.loads(body)
                fresh = evaluate(old, series.get(old["asset"], pd.Series(dtype=float)), end)
                if fresh != old:
                    self.write(db, fresh)
                items.append(describe(fresh, end))
        return sorted(items, key=lambda r: r["created_at"], reverse=True)

    def stop(self, ident):
        if not isinstance(ident, str):
            raise ValueError("Некорректное наблюдение")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT body FROM observations WHERE id=?", (ident,)).fetchone()
            if not old:
                raise ValueError("Наблюдение не найдено")
            r = json.loads(old[0])
            if r["status"] not in TERMINAL:
                r.update(status="stopped", stopped_at=datetime.now(timezone.utc).isoformat())
                self.write(db, r)
        return r


def view(paths, config):
    end = today()
    series = price_series(paths, config, end)
    store = Observations(paths["data"] / "runtime" / "web")
    candidates = []
    for asset, close in sorted(series.items()):
        if len(close) < 6:
            continue
        change = float(close.iloc[-1] / close.iloc[-6] - 1)
        candidates.append({"asset": asset, "date": str(close.index[-1]), "last": float(close.iloc[-1]),
                           "change_5": change, "kinds": (["hold_gain"] if change > 0 else ["recovery"] if change < 0 else []) + ["large_move"]})
    return {"as_of": end, "items": store.refresh(series, end), "candidates": candidates,
            "source": "Архив дневных котировок с проверкой целостности" if any((paths["data"] / "raw").glob("????-??-??/manifest_market.json")) else "Накопленная история котировок"}
