"""CLI: python -m midas <команда>.

daily  — полный дневной прогон (сбор -> LLM -> HRP -> отчёт)
probe  — проверка доступности локальной LLM
llm    — запуск/статус llama-server (локальный рантайм)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config, project_paths
from .llm.runtime import RuntimeManager

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _load(path: str | None) -> tuple[dict, Path]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    return load_config(config_path), config_path


def cmd_daily(args: argparse.Namespace) -> int:
    from .pipeline.daily import run_daily
    from .llm.runtime import launch_lock

    config, config_path = _load(args.config)
    lock = project_paths(config, config_path)["data"] / "runtime" / "daily.lock"
    try:
        with launch_lock(lock):
            result = run_daily(config, config_path, date=args.date,
                               use_llm=not args.no_llm, source=args.source,
                               fixture_dir=args.fixture_dir)
    except BlockingIOError as error:
        raise RuntimeError("Дневной прогон уже выполняется") from error
    report = result["report"]
    weights = report["weights"]
    print(f"Портфель на {report['date']} ({report['fusion']['mode']}):")
    for asset, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {asset:<14} {weight:7.2%}")
    if report["portfolio"].get("cash_share"):
        print(f"  {'CASH':<14} {report['portfolio']['cash_share']:7.2%}")
    print(f"Отчёт: {result['paths']['markdown']}")
    if result["paths"].get("analysis"):
        print(f"Разбор: {result['paths']['analysis']}")
    critique = report.get("critique", {})
    if critique.get("failed"):
        print(f"LLM-критик: {critique.get('summary', '')}")
    return 0


def cmd_scheduled(args: argparse.Namespace) -> int:
    from .scheduler import run_scheduled

    config, config_path = _load(args.config)
    result = run_scheduled(config, config_path, due_at=args.due_at,
                           use_llm=not args.no_llm, source=args.source,
                           fixture_dir=args.fixture_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["exit_code"]


def cmd_probe(args: argparse.Namespace) -> int:
    from .llm.client import LocalLLMError, probe

    config, _config_path = _load(args.config)
    try:
        state = probe(config["llm"]["endpoint"], config["llm"]["model"])
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except LocalLLMError as error:
        print(f"LLM недоступна: {error}", file=sys.stderr)
        print("Поднимите сервер: scripts/START_LLM.cmd или midas llm start --wait",
              file=sys.stderr)
        return 1


def cmd_llm(args: argparse.Namespace) -> int:
    config, config_path = _load(args.config)
    paths = project_paths(config, config_path)
    llm_cfg = config["llm"]
    manager = RuntimeManager(paths["data"] / "runtime",
                             runtime_root=Path(llm_cfg["runtime_root"]),
                             endpoint=llm_cfg["endpoint"], model=llm_cfg["model"])
    result = manager.status() if args.action == "status" else manager.start()
    if args.wait and result.get("state") == "loading":
        result = manager.wait_ready()
    print(json.dumps({k: v for k, v in result.items() if k != "command"}, ensure_ascii=False, indent=2))
    return 0 if result.get("state") in {"ready", "loading"} else 1


def _panel_from_fixture(directory: Path):
    from .collect import moex
    from .features.prices import build_panel

    frames = {}
    for path in sorted(Path(directory).glob("candles_*.json")):
        asset = path.stem.replace("candles_", "").replace("__", ":")
        frame = moex.parse_candles(path.read_bytes())
        if not frame.empty:
            frames[asset] = frame
    if not frames:
        raise FileNotFoundError(f"в {directory} нет candles_*.json")
    panel, _dropped = build_panel(frames, min_history=60)
    return panel


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest import walk_forward
    from .history import load_labels, load_price_panel, upsert_prices
    from .report.markdown import write_backtest

    if args.fetch and args.fixture_dir:
        raise ValueError("укажите либо --fetch, либо --fixture-dir")
    config, config_path = _load(args.config)
    if args.commission_bps is not None:
        config["backtest"]["commission_bps"] = args.commission_bps
    if args.spread_bps is not None:
        config["backtest"]["spread_bps"] = args.spread_bps
    paths = project_paths(config, config_path)
    history_dir = paths["data"] / "history"
    labels: list = []
    if args.fixture_dir:
        panel = _panel_from_fixture(args.fixture_dir)
    else:
        if args.fetch:
            from datetime import date as date_type

            from .pipeline.daily import _collect_market_net

            frames, _payloads, statuses = _collect_market_net(
                config, date_type.today().isoformat())
            upsert_prices(history_dir, frames)
            failed = [row for row in statuses if row.get("failed")]
            if failed:
                print("Провайдеры с ошибками: " + "; ".join(
                    f"{row['provider']} ({row['failed']})" for row in failed))
        panel = load_price_panel(history_dir, end=args.end)
        labels = load_labels(history_dir)
        if panel.empty:
            raise FileNotFoundError(
                "история цен пуста: запустите daily или backtest --fetch "
                "или --fixture-dir fixtures/demo")
    from .history import load_dividends_panel, load_earnings, load_macro_panel
    dividends = load_dividends_panel(paths["data"] / "history", end=args.end)
    macro = load_macro_panel(paths["data"] / "history", end=args.end)
    earnings_pairs = [(row["date"], row["asset"])
                      for row in load_earnings(paths["data"] / "history")] or None
    result = walk_forward(panel, config=config, start=args.start, end=args.end,
                          rebalance_days=args.rebalance_days, labels=labels,
                          dividends=dividends if not dividends.empty else None,
                          macro=None if macro.empty else macro,
                          earnings_pairs=earnings_pairs)
    written = write_backtest(paths["reports"], result)
    print(f"Бэктест {result['start']} — {result['end']}, "
          f"ребалансов: {result['n_rebalances']}")
    for name, metrics in result["strategies"].items():
        print(f"  {name:<14} доходность {metrics['total_return']:+.2%}  "
              f"CAGR {metrics['cagr']:+.2%}  просадка {metrics['max_drawdown']:.2%}  "
              f"издержки {metrics['cost_paid']:.3%} начального капитала")
    tracker = result["tracker"]
    hit = "н/д" if tracker["hit_rate"] is None else f"{tracker['hit_rate']:.1%}"
    brier = "н/д" if tracker["brier"] is None else f"{tracker['brier']:.3f}"
    print(f"Трекер: hit-rate {hit}, Brier {brier}, "
          f"срабатываний {tracker['n_alerts']}/{tracker['n_scored']}")
    print(f"Отчёт: {written['markdown']}")
    return 0


def cmd_vision(args: argparse.Namespace) -> int:
    config, config_path = _load(args.config)
    paths = project_paths(config, config_path)
    llm_cfg = config["llm"]
    runtime_dir = paths["root"] / "llm_runtime_vision"
    ggufs = sorted(runtime_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    mmprojs = list(runtime_dir.glob("mmproj*.gguf"))
    if not ggufs or not (runtime_dir / "llama-server.exe").is_file():
        print(f"vision-рантайм не установлен: {runtime_dir}\n"
              f"Выполните: python scripts/install_vision_model.py", file=sys.stderr)
        return 1
    extra = ["--mmproj", str(mmprojs[0])] if mmprojs else []
    manager = RuntimeManager(paths["data"] / "runtime", runtime_dir=runtime_dir,
                             weights=ggufs[0], endpoint=llm_cfg["vision_endpoint"],
                             model=llm_cfg["vision_model"], extra_args=extra)
    result = manager.status() if args.action == "status" else manager.start()
    if args.wait and result.get("state") == "loading":
        result = manager.wait_ready()
    print(json.dumps({k: v for k, v in result.items() if k != "command"}, ensure_ascii=False, indent=2))
    return 0 if result.get("state") in {"ready", "loading"} else 1


def cmd_settle(args: argparse.Namespace) -> int:
    from .history import load_price_panel
    from .settlement import quality_table, settle_pending, write_quality_report

    config, config_path = _load(args.config)
    paths = project_paths(config, config_path)
    panel = load_price_panel(paths["data"] / "history")
    if panel.empty:
        print("история цен пуста: выполните daily или backtest --fetch",
              file=sys.stderr)
        return 1
    summary = settle_pending(
        paths["data"] / "forecasts", paths["data"] / "settlement",
        paths["data"] / "history", panel,
        threshold=config["tracker"]["sigma_threshold"],
        halflife=config["tracker"]["vol_halflife_days"],
        horizon=int(config["settlement"]["horizon_days"]),
        only_date=args.date)
    print(f"Закрыто прогнозов: {len(summary['settled'])}"
          + (f" ({', '.join(summary['settled'])})" if summary["settled"] else ""))
    print(f"Ожидают горизонта: {len(summary['pending'])}"
          + (f" ({', '.join(summary['pending'])})" if summary["pending"] else ""))
    quality_path = write_quality_report(paths["reports"],
                                        quality_table(paths["data"] / "history"))
    print(f"Качество: {quality_path}")
    return 0


def cmd_modality(args: argparse.Namespace) -> int:
    from .history import load_price_panel, upsert_prices
    from .modality_test import run_modality_test

    if args.fetch and args.fixture_dir:
        raise ValueError("укажите либо --fetch, либо --fixture-dir")
    config, config_path = _load(args.config)
    paths = project_paths(config, config_path)
    if args.fixture_dir:
        panel = _panel_from_fixture(args.fixture_dir)
    else:
        if args.fetch:
            import copy

            fetch_config = copy.deepcopy(config)
            days = int(config["backtest"]["fetch_days"])
            fetch_config["providers"]["moex"]["history_days"] = days
            fetch_config["providers"]["yfinance"]["history_days"] = days
            from datetime import date as date_type

            from .pipeline.daily import _collect_market_net, collect_track_b

            run_day = args.end or date_type.today().isoformat()
            frames, _payloads, statuses = _collect_market_net(fetch_config, run_day)
            upsert_prices(paths["data"] / "history", frames)
            collect_track_b(fetch_config, paths["data"] / "history",
                            paths["data"] / "manual", run_day)
            failed = [row for row in statuses if row.get("failed")]
            if failed:
                print("Провайдеры с ошибками: " + "; ".join(
                    f"{row['provider']} ({row['failed']})" for row in failed))
        panel = load_price_panel(paths["data"] / "history", end=args.end)
        if panel.empty:
            print("история пуста: --fetch или --fixture-dir, либо daily",
                  file=sys.stderr)
            return 1
    report = run_modality_test(config, config_path, panel,
                               start=args.start, end=args.end)
    print(f"Панель: {report['date_range'][0]} — {report['date_range'][1]}, "
          f"{report['n_assets']} активов")
    for name, row in report["variants"].items():
        p = row.get("p_cagr_better")
        print(f"  {name:<14} CAGR {row['metrics']['cagr']:+7.2%}  "
              f"MaxDD {row['metrics']['max_drawdown']:6.1%}  "
              f"P(CAGR>price) {'—' if p is None else f'{p:.0%}'}")
    for check, verdict in report["verdict"].items():
        print(f"  {check}: {verdict}")
    print(f"Отчёт: {report['paths']['markdown']}")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    from .settlement import quality_table, render_quality_markdown
    config, config_path = _load(args.config)
    paths = project_paths(config, config_path)
    quality = quality_table(paths["data"] / "history")
    if not quality.get("n_rows"):
        print("Закрытых прогнозов пока нет (прогнозы замораживает daily, "
              "закрывает settle --all).")
        return 0
    print(render_quality_markdown(quality))
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from .ablation import run_ablation
    from .history import load_labels, load_price_panel

    if args.fetch and args.fixture_dir:
        raise ValueError("укажите либо --fetch, либо --fixture-dir")
    config, config_path = _load(args.config)
    paths = project_paths(config, config_path)
    if args.fixture_dir:
        panel = _panel_from_fixture(args.fixture_dir)
        # метки берём из накопленной истории (daily их пишет) — на пустой
        # истории все sent-варианты вырождаются в price_only
        labels = load_labels(paths["data"] / "history")
    else:
        if args.fetch:
            fetch_config = json.loads(json.dumps(config))  # deepcopy без copy-импорта
            days = int(config["backtest"]["fetch_days"])
            fetch_config["providers"]["moex"]["history_days"] = days
            fetch_config["providers"]["yfinance"]["history_days"] = days
            from datetime import date as date_type

            from .history import upsert_prices
            from .pipeline.daily import _collect_market_net, collect_track_b

            run_day = args.end or date_type.today().isoformat()
            frames, _payloads, statuses = _collect_market_net(fetch_config, run_day)
            upsert_prices(paths["data"] / "history", frames)
            collect_track_b(fetch_config, paths["data"] / "history",
                            paths["data"] / "manual", run_day)
            failed = [row for row in statuses if row.get("failed")]
            if failed:
                print("Провайдеры с ошибками: " + "; ".join(
                    f"{row['provider']} ({row['failed']})" for row in failed))
        panel = load_price_panel(paths["data"] / "history", end=args.end)
        labels = load_labels(paths["data"] / "history")
        if panel.empty:
            print("история пуста: --fetch или --fixture-dir, либо daily",
                  file=sys.stderr)
            return 1
    report = run_ablation(config, config_path, panel, labels,
                          start=args.start, end=args.end,
                          resume=args.resume, only=args.only)
    print(f"Абляции: {report['n_succeeded']}/{report['n_total']} вариантов; "
          f"отчёт: {report['paths']['markdown']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="midas",
                                     description="Midas (Мидас): портфельная аналитика, мультимодальный HRP и локальная LLM")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("web", help="локальная мобильная панель")
    web.add_argument("--port", type=int, default=8765)
    web.set_defaults(func=cmd_web)

    daily = sub.add_parser("daily", help="дневной прогон")
    daily.add_argument("--date", default=None, help="YYYY-MM-DD (по умолчанию сегодня)")
    daily.add_argument("--no-llm", action="store_true", help="без LLM: price-only HRP")
    daily.add_argument("--source", choices=["net", "fixture"], default="net",
                       help="net — биржи/RSS; fixture — офлайн-демо из fixtures/demo")
    daily.add_argument("--fixture-dir", default=None, help="каталог fixture-данных")
    daily.set_defaults(func=cmd_daily)

    scheduled = sub.add_parser("scheduled-run", help="однократный запуск по расписанию с журналом")
    scheduled.add_argument("--due-at", default="07:00", help="время HH:MM, Europe/Moscow")
    scheduled.add_argument("--no-llm", action="store_true")
    scheduled.add_argument("--source", choices=["net", "fixture"], default="net")
    scheduled.add_argument("--fixture-dir", default=None)
    scheduled.set_defaults(func=cmd_scheduled)

    probe = sub.add_parser("probe", help="проверить LLM endpoint")
    probe.set_defaults(func=cmd_probe)

    llm = sub.add_parser("llm", help="управление llama-server")
    llm.add_argument("action", choices=["start", "status"])
    llm.add_argument("--wait", action="store_true", help="дождаться готовности")
    llm.set_defaults(func=cmd_llm)

    backtest = sub.add_parser("backtest", help="walk-forward бэктест по истории цен")
    backtest.add_argument("--start", default=None, help="YYYY-MM-DD, первая дата ребаланса")
    backtest.add_argument("--end", default=None, help="YYYY-MM-DD, конец окна")
    backtest.add_argument("--rebalance-days", type=int, default=None,
                          help="шаг ребаланса в торговых днях (иначе backtest.rebalance_days)")
    backtest.add_argument("--fetch", action="store_true",
                          help="скачать свечи в data/history перед расчётом")
    backtest.add_argument("--fixture-dir", default=None,
                          help="офлайн-панель из candles_*.json, без истории и сети")
    backtest.add_argument("--commission-bps", type=float, default=None,
                          help="комиссия каждой сделки в б.п. (5 = 0.05%%)")
    backtest.add_argument("--spread-bps", type=float, default=None,
                          help="полный спред в б.п.; на сделку половина (10 = 0.10%%)")
    backtest.set_defaults(func=cmd_backtest)

    settle = sub.add_parser("settle", help="закрыть прогнозы прошлых дней фактом")
    settle.add_argument("--date", default=None, help="конкретный день (иначе все pending)")
    settle.set_defaults(func=cmd_settle)

    quality = sub.add_parser("quality", help="таблица качества трекера (settlement)")
    quality.set_defaults(func=cmd_quality)

    ablation = sub.add_parser("ablation", help="матрица конфигураций walk-forward с CI")
    ablation.add_argument("--start", default=None, help="YYYY-MM-DD, первая дата ребаланса")
    ablation.add_argument("--end", default=None, help="YYYY-MM-DD, конец окна")
    ablation.add_argument("--fetch", action="store_true",
                          help="глубокий fetch (backtest.fetch_days) перед расчётом")
    ablation.add_argument("--fixture-dir", default=None,
                          help="офлайн-панель из candles_*.json")
    ablation.add_argument("--resume", action="store_true",
                          help="пропустить succeeded-варианты из журнала")
    ablation.add_argument("--only", default=None,
                          help="подстрока фильтра вариантов (например snf)")
    ablation.set_defaults(func=cmd_ablation)

    modality = sub.add_parser("modality-test",
                              help="исторический тест сентимент-модальности "
                                   "(oracle/noise контроли)")
    modality.add_argument("--start", default=None, help="YYYY-MM-DD")
    modality.add_argument("--end", default=None, help="YYYY-MM-DD")
    modality.add_argument("--fetch", action="store_true",
                          help="глубокий fetch (backtest.fetch_days) перед тестом")
    modality.add_argument("--fixture-dir", default=None,
                          help="офлайн-панель из candles_*.json")
    modality.set_defaults(func=cmd_modality)

    vision = sub.add_parser("vision", help="управление vision-сервером (порт 8081)")
    vision.add_argument("action", choices=["start", "status"])
    vision.add_argument("--wait", action="store_true", help="дождаться готовности")
    vision.set_defaults(func=cmd_vision)
    return parser


def cmd_web(args):
    from .web.server import serve
    _config, config_path = _load(args.config)
    if not 1024 <= args.port <= 65535:
        raise ValueError("порт должен быть от 1024 до 65535")
    serve(config_path, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as error:
        print(f"Ошибка конфигурации: {error}", file=sys.stderr)
        return 2
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"Ошибка прогона: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
