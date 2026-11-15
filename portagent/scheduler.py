"""Однократный unattended-запуск для Windows Task Scheduler.

Расписание живёт в ОС; здесь — московская дата, общий с CLI lock,
проверка готового результата, запуск модели, журнал и состояние для будущего UI.
"""
from __future__ import annotations

import json
import os
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import project_paths
from .llm.runtime import RuntimeManager, launch_lock
from .pipeline.daily import run_daily

MOSCOW = timezone(timedelta(hours=3), "Europe/Moscow")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _completed(paths: dict, day: str) -> dict | None:
    directory = paths["reports"] / day
    if not all((directory / name).is_file() for name in
               ("portfolio.json", "portfolio.md", "weights.csv")):
        return None
    try:
        report = json.loads((directory / "portfolio.json").read_text(encoding="utf-8"))
        return report if report.get("date") == day and report.get("weights") else None
    except (OSError, ValueError, AttributeError):
        return None


def _health(report: dict, paths: dict, day: str) -> list[str]:
    """Деградация наблюдаема, но готовый портфель не пересчитывается повторно."""
    warnings = []
    for row in report.get("provider_statuses", []):
        if row.get("failed"):
            warnings.append(f"provider:{row.get('provider')}:{row['failed']}")
    failed_feeds = sum(not row.get("ok", False) for row in report.get("feed_statuses", []))
    if failed_feeds:
        warnings.append(f"feeds_failed:{failed_feeds}")
    for key in ("llm", "settlement", "materials"):
        if report.get(key, {}).get("error"):
            warnings.append(f"{key}:{report[key]['error']}")
    if report.get("critique", {}).get("failed"):
        warnings.append("critique_failed")
    if report.get("llm", {}).get("batch_errors"):
        warnings.append(f"llm_batch_errors:{report['llm']['batch_errors']}")
    if report.get("history_error"):
        warnings.append(f"history:{report['history_error']}")
    if not (paths["data"] / "forecasts" / f"{day}.json").is_file():
        warnings.append("forecast_missing")
    from .history import load_price_panel

    try:
        panel = load_price_panel(paths["data"] / "history")
        if panel.empty:
            warnings.append("price_history_missing")
        else:
            # Порог 7 календарных дней оставляет обычные выходные и короткие праздники.
            cutoff = datetime.fromisoformat(day).date() - timedelta(days=7)
            stale = [name for name in panel if panel[name].dropna().empty
                     or str(panel[name].last_valid_index())[:10] < cutoff.isoformat()]
            if stale:
                warnings.append("stale_prices:" + ",".join(stale))
    except Exception as error:
        warnings.append(f"price_history_check:{error}")
    return warnings


def run_scheduled(config: dict, config_path: str | Path, *, due_at: str = "07:00",
                  now: datetime | None = None, use_llm: bool = True,
                  source: str = "net", fixture_dir: str | Path | None = None) -> dict:
    """Не восстанавливает пропущенные дни задним числом: считает только сегодня.

    Повтор после сбоя возможен, готовый (даже degraded) отчёт сохраняется.
    OS-lock освобождается и после аварийного завершения процесса.
    """
    moment = now or datetime.now(MOSCOW)
    if moment.tzinfo is None:
        raise ValueError("now должен содержать часовой пояс")
    moment = moment.astimezone(MOSCOW)
    due = datetime.strptime(due_at, "%H:%M").time()
    day = moment.date().isoformat()
    if moment.time().replace(tzinfo=None) < due:
        return {"status": "not_due", "date": day, "due_at": due_at, "exit_code": 0}
    paths = project_paths(config, config_path)
    runtime = paths["data"] / "runtime"
    directory = runtime / "scheduler"
    directory.mkdir(parents=True, exist_ok=True)

    def save(state: dict) -> dict:
        _write_json(directory / "state.json", state)
        with (directory / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(state, ensure_ascii=False) + "\n")
        return state

    try:
        with launch_lock(runtime / "daily.lock"):
            existing = _completed(paths, day)
            base = {"date": day, "timezone": "Europe/Moscow", "due_at": due_at,
                    "started_at": moment.isoformat(), "pid": os.getpid(),
                    "config": str(Path(config_path).resolve())}
            if existing:
                return save({**base, "status": "already_completed", "exit_code": 0,
                             "warnings": _health(existing, paths, day),
                             "report": str(paths["reports"] / day / "portfolio.md")})
            log = directory / f"{day}_{moment:%H%M%S}_{uuid.uuid4().hex[:8]}.log"
            state = {**base, "status": "running", "log": str(log), "warnings": []}
            save(state)
            with log.open("a", encoding="utf-8", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    try:
                        print(f"START {moment.isoformat()} config={config_path}", flush=True)
                        if use_llm:
                            llm = config["llm"]
                            try:
                                manager = RuntimeManager(runtime, runtime_root=Path(llm["runtime_root"]),
                                                         endpoint=llm["endpoint"], model=llm["model"])
                                manager.ensure_ready(timeout=180)
                            except Exception as error:
                                state["warnings"].append(f"llm_start:{error}")
                                print(f"LLM startup failed; daily will record fallback: {error}", flush=True)
                            # Второй сервер нужен только для agreement или предоставленных картинок.
                            from .collect.charts import scan_chart_drops
                            if (llm["agreement"]["enabled"] or
                                    scan_chart_drops(paths["data"] / "manual" / "charts", day)):
                                from .llm.models import manager_for
                                try:
                                    manager_for(config, config_path, "vision").ensure_ready(timeout=180)
                                except Exception as error:
                                    state["warnings"].append(f"vision_start:{error}")
                        result = run_daily(config, config_path, date=day, use_llm=use_llm,
                                           source=source, fixture_dir=fixture_dir)
                        state["warnings"].extend(_health(result["report"], paths, day))
                        state.update(status="degraded" if state["warnings"] else "succeeded",
                                     exit_code=0, report=str(result["paths"]["markdown"]))
                        print(f"DONE {state['status']}: {state['report']}", flush=True)
                    except Exception as error:
                        traceback.print_exc()
                        state.update(status="failed", exit_code=1, error=str(error)[:2000])
                    state["finished_at"] = datetime.now(MOSCOW).isoformat()
                    return save(state)
    except BlockingIOError:
        # Не затираем состояние другого активного запуска.
        return {"status": "already_running", "date": day, "exit_code": 0}
