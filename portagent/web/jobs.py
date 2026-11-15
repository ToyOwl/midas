"""Исполнение разрешённых заданий в отдельном процессе: веб-сервер не блокируется."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

from ..config import load_config, project_paths
from ..llm.runtime import launch_lock
from ..progress import emit
from ..scheduler import run_scheduled, _write_json
from .store import Store, TERMINAL, now


def load_report(paths, day):
    from datetime import date
    date.fromisoformat(day)
    if len(day) != 10:
        raise ValueError("некорректная дата")
    report = json.loads((paths["reports"] / day / "portfolio.json").read_text(encoding="utf-8"))
    if report.get("date") != day or not report.get("weights"):
        raise ValueError("отчёт ещё не готов")
    from ..report.recovery import load_supplement
    return load_supplement(paths, day, report)


def execute(config, config_path, job):
    paths = project_paths(config, config_path)
    kind, args = job["kind"], job["args"]
    if kind == "analyze_material":
        from ..materials.analysis import analyze
        from ..llm.models import manager_for
        emit("models", "Подготовка Qwen для разбора изображения")
        manager_for(config, config_path, "vision").ensure_ready(timeout=180)
        emit("image", "Вычисление эмбеддинга и разбор содержимого изображения")
        result = analyze(paths["data"] / "materials", args["id"], config)
        return {k: v for k, v in result.items() if k != "embedding"}, "succeeded"
    if kind == "preview_materials":
        from ..materials.portfolio import preview
        emit("preview", "Сравнение исходного HRP и варианта с материалами")
        result = preview(paths, load_report(paths, args["date"]), config, args["ids"], args["influence"])
        return {k: v for k, v in result.items() if k != "records"}, "succeeded"
    if kind in {"refresh_sources", "collect_telegram"}:
        from ..research.events import refresh_events
        from ..research.telegram import today
        emit("events", "Разбор Telegram и судебных сообщений из архива")
        result = refresh_events(config, config_path, today() if kind == "collect_telegram" else args["date"],
                                fetch_current=kind == "collect_telegram")
        failed = any(v.get("batch_errors") for v in result["groups"].values())
        failed |= any(not f["ok"] for f in (result.get("collection") or {}).get("feeds", []))
        return result, "degraded" if failed else "succeeded"
    if kind == "train_models":
        import subprocess
        interpreter = paths["root"] / ".venv-research" / "Scripts" / "python.exe"
        if not interpreter.exists():
            interpreter = Path(config["llm"]["runtime_root"]) / ".venv" / "Scripts" / "python.exe"
        if not interpreter.exists():
            raise RuntimeError("Не найдена среда обучения с TabM/NGBoost/XGBoostLSS")
        market_root = paths["data"] / "research" / "top3" / ("market-" + args["date"] + "-2700")
        if not (market_root / "manifest.json").exists():
            raise RuntimeError("Сначала подготовьте снимок цен для этой даты: scripts/TRAIN_TOP3.ps1")
        emit("training", "Обучение и проверка моделей на пяти отложенных годах")
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", PYTHONUTF8="1")
        subprocess.run([str(interpreter), "-u", "-m", "portagent.research.benchmark", "--config", str(config_path),
                        "--market-root", str(market_root)], cwd=paths["root"], env=env, check=True,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        pointer = json.loads((paths["data"] / "research" / "top3" / "latest.json").read_text(encoding="utf-8"))
        return json.loads(Path(pointer["report"]).read_text(encoding="utf-8")), "succeeded"
    if kind == "recover_news":
        from ..report.recovery import recover_news
        result = recover_news(config, config_path, args["date"])
        return result, "degraded" if result["llm"]["batch_errors"] else "succeeded"
    if kind == "daily":
        emit("models", "Подготовка дневного расчёта")
        result = run_scheduled(config, config_path, due_at="00:00", use_llm=args["use_llm"])
        if result["status"] == "already_running":
            raise RuntimeError("Дневной расчёт уже выполняется вне панели. Его статус доступен в разделе запусков.")
        if result["exit_code"]:
            raise RuntimeError(result.get("error", "ошибка дневного расчёта"))
        return result, "degraded" if result.get("warnings") else "succeeded"
    if kind == "model":
        from ..llm.models import manager_for
        emit("models", "Запуск локальной модели; ожидание готовности")
        result = manager_for(config, config_path, args["kind"]).ensure_ready(timeout=180)
        return {k: result.get(k) for k in ("state", "model", "endpoint", "error")}, "succeeded"
    if kind in {"ask", "review"}:
        from ..llm.models import manager_for
        from ..llm.assistant import explain
        report = load_report(paths, args["date"])
        from ..report.diagnostics import get_diagnostics
        emit("diagnostics", "Подготовка причин предупреждений и детектора TOP-3")
        diagnostics = get_diagnostics(paths, report, config)
        from .research import view
        research = view(paths, report, config)
        diagnostics = {**diagnostics, "source_events": research["events"], "trained_benchmark": research["benchmark"]}
        from ..materials.storage import list_materials
        diagnostics["uploaded_materials"] = list_materials(paths["data"] / "materials")
        emit("models", "Подготовка локального аналитика")
        manager_for(config, config_path).ensure_ready(timeout=180)
        emit("answer", "Рецензия портфеля" if kind == "review" else "Ответ по фактам зафиксированного отчёта")
        result = explain(report, args.get("question", ""), llm=config["llm"],
                         cache_dir=paths["data"] / "cache" / "assistant", review=kind == "review",
                         diagnostics=diagnostics)
        return result, "succeeded"
    if kind == "backtest":
        from ..backtest import run_backtest
        working = copy.deepcopy(config)
        for field in ("commission_bps", "spread_bps"):
            if field in args:
                working["backtest"][field] = args[field]
        # Каждый эксперимент получает собственные файлы, история общая и только читается.
        working["paths"]["reports_dir"] = str(paths["reports"] / "web" / job["id"])
        emit("backtest", "Исторический тест с комиссиями и спредом")
        result = run_backtest(working, config_path, start=args.get("start") or None,
                              end=args.get("end") or None, rebalance_days=args.get("rebalance_days"))
        return result, "succeeded"
    raise ValueError("действие не разрешено")


def run(config_path: str, job_id: str) -> int:
    config = load_config(config_path)
    root = project_paths(config, config_path)["data"] / "runtime" / "web"
    store = Store(root)
    with launch_lock(root / f"{job_id}.lock"):
        job = store.get(job_id)
        if job is None or job["status"] in TERMINAL:
            return 0
        store.update(job_id, status="running", started=now(), pid=os.getpid())
        os.environ["PORTAGENT_PROGRESS_PATH"] = str(root / f"{job_id}.progress.json")
        try:
            result, status = execute(config, config_path, job)
            _write_json(root / f"{job_id}.result.json", result)
            store.update(job_id, status=status, finished=now(), result=result)
            emit("done", "Задание завершено")
            return 0
        except Exception as error:
            traceback.print_exc()
            store.update(job_id, status="failed", finished=now(), error=str(error)[:2000])
            emit("failed", str(error)[:300])
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        sys.exit(run(args.config, args.job))
    except BlockingIOError:
        sys.exit(0)
