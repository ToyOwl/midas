"""Оркестрация дневного прогона: сбор -> LLM-разметка -> мультимодальные
расстояния -> HRP -> рецензия критика -> отчёт + ledger.

Правила устойчивости:
- сбой отдельного провайдера/ленты не роняет прогон — фиксируется статусом;
- LLM опциональна (--no-llm -> price-only режим) и не влияет на веса;
- прогон идемпотентен по дате: raw-архив переиспользует совпадающие файлы,
  LLM-разметка кэшируется по SHA батча.
"""
from __future__ import annotations

import json
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import requests

from ..collect import moex, news as news_collector
from ..collect.charts import analyze_charts, scan_chart_drops
from ..collect.courts import CASES_SYSTEM_PROMPT, LEGAL_HALFLIFE_DAYS, cases_to_items, load_court_drops
from ..collect.dividends import fetch_us_dividends, load_dividend_drops
from ..collect.earnings import fetch_us_earnings, load_earnings_drops
from ..collect.errors import ProviderError
from ..collect.macro import fetch_usdrub, fetch_yf_series
from ..collect.tg import fetch_tg_raw, parse_tg
from ..collect.yf_provider import fetch_us_candles
from ..config import project_paths
from ..history import (load_earnings, load_macro_panel, upsert_agreement,
                       upsert_dividends, upsert_earnings, upsert_labels,
                       upsert_macro, upsert_prices, upsert_tracker)
from ..features import prices as price_features
from ..features import sentiment as sentiment_features
from ..features import view_quality as view_quality_mod
from ..llm.client import LocalLLMError, probe
from ..llm.agreement import compare_labels
from ..llm.critic import critique_portfolio
from ..llm.news_analyst import analyze_news, make_chat
from ..portfolio.fuse import fuse_distances
from ..portfolio.hrp import hrp_weights
from ..portfolio.ledger import append_ledger
from ..report.markdown import write_report
from ..settlement import _config_digest, settle_pending, write_forecast
from ..tracker import rare_events
from ..storage.raw import archive, write_manifest

MIN_ASSETS = 5
SENTIMENT_MIN_PERIODS = 10


def _resolve_date(value: str | None) -> str:
    if value is None:
        return date_type.today().isoformat()
    try:
        return date_type.fromisoformat(value).isoformat()
    except ValueError as error:
        raise ValueError(f"--date ожидает YYYY-MM-DD, получено {value!r}") from error


def _asset_id(provider: str, ticker: str) -> str:
    return f"{provider}:{ticker}"


def _collect_market_net(config: dict, run_date: str) -> tuple[dict[str, pd.DataFrame], list[tuple[str, bytes]], list[dict]]:
    """Свечи MOEX ISS + yfinance -> ({asset_id: DataFrame}, сырые байты, статусы)."""
    frames: dict[str, pd.DataFrame] = {}
    payloads: list[tuple[str, bytes]] = []
    statuses: list[dict] = []

    moex_cfg = config["providers"]["moex"]
    if moex_cfg["enabled"]:
        start = (pd.Timestamp(run_date) - pd.Timedelta(days=moex_cfg["history_days"])).strftime("%Y-%m-%d")
        till = f"{run_date} 23:59:59"
        ok_count, errors = 0, []
        for ticker in config["universe"]["moex"]:
            asset = _asset_id("MOEX", ticker)
            try:
                payload = moex.fetch_candles_raw(
                    ticker, board=moex_cfg["board"], from_date=start, till_date=till,
                    timeout=moex_cfg["timeout_s"], retries=moex_cfg["retries"])
                frame = moex.parse_candles(payload)
                if frame.empty:
                    raise ProviderError("пустой ответ")
                frames[asset] = frame
                payloads.append((f"moex_{ticker}.json", payload))
                ok_count += 1
            except ProviderError as error:
                errors.append(f"{ticker}: {error}")
        statuses.append({"provider": "moex", "ok": ok_count,
                         "failed": len(errors), "error": "; ".join(errors)[:500] or None})

    yf_cfg = config["providers"]["yfinance"]
    if yf_cfg["enabled"]:
        ok_count, errors = 0, []
        for ticker in config["universe"]["us"]:
            asset = _asset_id("US", ticker)
            try:
                frame, csv_payload = fetch_us_candles(ticker, history_days=yf_cfg["history_days"],
                                                      timeout_s=yf_cfg["timeout_s"])
                frames[asset] = frame
                payloads.append((f"yf_{ticker}.csv", csv_payload.encode("utf-8")))
                ok_count += 1
            except ProviderError as error:
                errors.append(f"{ticker}: {error}")
        statuses.append({"provider": "yfinance", "ok": ok_count,
                         "failed": len(errors), "error": "; ".join(errors)[:500] or None})
    return frames, payloads, statuses


def _universe_ids(config: dict) -> list[str]:
    return ([_asset_id("MOEX", ticker) for ticker in config["universe"]["moex"]]
            + [_asset_id("US", ticker) for ticker in config["universe"]["us"]])


def _collect_news_net(config: dict, run_date: str) -> tuple[list[dict], list[tuple[str, bytes]], list[dict]]:
    """RSS + MOEX sitenews + Telegram + Google News -> (элементы, сырые байты, статусы)."""
    news_cfg = config["news"]
    items: list[dict] = []
    payloads: list[tuple[str, bytes]] = []
    statuses: list[dict] = []
    session = requests.Session()
    for index, url in enumerate(news_cfg["rss"]):
        host = urlsplit(url).netloc.replace(".", "_")
        try:
            payload = news_collector.fetch_rss_raw(url, timeout=news_cfg["fetch_timeout_s"],
                                                   session=session)
            feed_items = news_collector.parse_rss(payload, source=host)
            feed_items = news_collector.filter_recent(feed_items,
                                                      lookback_hours=news_cfg["lookback_hours"])
            items.extend(feed_items)
            payloads.append((f"rss_{index}_{host}.xml", payload))
            statuses.append({"feed": url, "ok": True, "items": len(feed_items), "error": None})
        except ProviderError as error:
            statuses.append({"feed": url, "ok": False, "items": 0, "error": str(error)[:300]})
    sitenews_cfg = news_cfg["moex_sitenews"]
    if sitenews_cfg["enabled"]:
        try:
            payload = moex.fetch_sitenews_raw(limit=sitenews_cfg["limit"],
                                              timeout=news_cfg["fetch_timeout_s"])
            sitenews_items = moex.parse_sitenews(payload,
                                                 lookback_hours=sitenews_cfg["lookback_hours"])
            items.extend(sitenews_items)
            payloads.append(("moex_sitenews.json", payload))
            statuses.append({"feed": "moex-sitenews", "ok": True,
                             "items": len(sitenews_items), "error": None})
        except ProviderError as error:
            statuses.append({"feed": "moex-sitenews", "ok": False, "items": 0,
                             "error": str(error)[:300]})
    tg_cfg = news_cfg.get("telegram", {})
    for channel in tg_cfg.get("channels", []) if tg_cfg.get("enabled") else []:
        try:
            payload = fetch_tg_raw(channel, pages=int(tg_cfg.get("pages", 1)),
                                   timeout=news_cfg["fetch_timeout_s"], session=session)
            tg_items = parse_tg(payload, channel=channel)
            tg_items = news_collector.filter_recent(tg_items,
                                                    lookback_hours=news_cfg["lookback_hours"])
            items.extend(tg_items)
            payloads.append((f"tg_{channel}.html", payload))
            statuses.append({"feed": f"tg:{channel}", "ok": True,
                             "items": len(tg_items), "error": None})
        except ProviderError as error:
            statuses.append({"feed": f"tg:{channel}", "ok": False, "items": 0,
                             "error": str(error)[:300]})
    google_cfg = news_cfg["google_news"]
    if google_cfg["enabled"]:
        aliases = config["courts"]["aliases"]
        per_query = int(google_cfg["per_query_limit"])
        for ticker_asset in _universe_ids(config):
            ticker = ticker_asset.split(":", 1)[1]
            query = news_collector.ticker_query(ticker, aliases.get(ticker, []))
            for locale in google_cfg["locales"]:
                feed_name = f"google-news {locale['gl']} {ticker}"
                url = news_collector.google_news_url(
                    query, hl=locale["hl"], gl=locale["gl"], ceid=locale["ceid"])
                try:
                    payload = news_collector.fetch_rss_raw(
                        url, timeout=news_cfg["fetch_timeout_s"], session=session)
                    feed_items = news_collector.parse_rss(payload, source=feed_name)
                    feed_items = news_collector.filter_recent(
                        feed_items, lookback_hours=news_cfg["lookback_hours"])[:per_query]
                    items.extend(feed_items)
                    payloads.append((f"gn_{locale['gl']}_{ticker}.xml", payload))
                    statuses.append({"feed": feed_name, "ok": True,
                                     "items": len(feed_items), "error": None})
                except ProviderError as error:
                    statuses.append({"feed": feed_name, "ok": False, "items": 0,
                                     "error": str(error)[:300]})
    return items, payloads, statuses


def _load_fixture(fixture_dir: Path, run_date: str) -> tuple[dict[str, pd.DataFrame], list[dict], list[dict], list[dict]]:
    """Демо-данные офлайн: candles_*.json (формат ISS) + news.json."""
    directory = Path(fixture_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"каталог fixture не найден: {directory}")
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("candles_*.json")):
        asset = path.stem.replace("candles_", "").replace("__", ":")
        frame = moex.parse_candles(path.read_bytes())
        if not frame.empty:
            frames[asset] = frame
    news_path = directory / "news.json"
    items = json.loads(news_path.read_text(encoding="utf-8")) if news_path.exists() else []
    statuses = [{"provider": "fixture", "ok": len(frames), "failed": 0, "error": None}]
    feed_statuses = [{"feed": str(directory), "ok": True, "items": len(items), "error": None}]
    return frames, statuses, items, feed_statuses


def collect_track_b(config: dict, history_dir: Path, manual_dir: Path,
                    run_date: str) -> tuple[pd.DataFrame | None, list[tuple[str, str]], dict]:
    """B6+B7: макро-сериалы и календарь отчётностей — сбор -> история -> панели.

    Возвращает (macro_panel, earnings_pairs, statuses). Сбой любого
    источника пишется в statuses и не роняет вызывающий прогон.
    """
    statuses: dict[str, str] = {}
    macro_panel = None
    earnings_pairs: list[tuple[str, str]] = []
    try:
        # B5: дивиденды — ручная выкладка MOEX + yfinance US (справочно)
        if config["dividends"]["enabled"]:
            dividend_rows = load_dividend_drops(manual_dir / "dividends")
            for ticker in config["universe"]["us"]:
                try:
                    dividend_rows += fetch_us_dividends(ticker)
                except ProviderError:
                    pass  # справочный источник, деградация тихая
            if dividend_rows:
                upsert_dividends(history_dir, dividend_rows)
    except (OSError, RuntimeError, ValueError) as error:
        statuses["dividends"] = str(error)[:160]
    try:
        if config["macro"]["enabled"]:
            rows: list[dict] = []
            start = (pd.Timestamp(run_date) - pd.Timedelta(
                days=config["backtest"]["fetch_days"])).strftime("%Y-%m-%d")
            try:
                usdrub = fetch_usdrub(from_date=start, till_date=run_date,
                                      secid=config["macro"]["usdrub_secid"])
                rows += [{"date": day, "series": "usdrub", "value": float(value)}
                         for day, value in usdrub["value"].items()]
            except ProviderError as error:
                statuses["usdrub"] = str(error)[:160]
            for spec in config["macro"]["yf_series"]:
                try:
                    series_frame = fetch_yf_series(spec["symbol"])
                    rows += [{"date": day, "series": spec["name"], "value": float(value)}
                             for day, value in series_frame["value"].items()]
                except ProviderError as error:
                    statuses[spec["name"]] = str(error)[:160]
            if rows:
                upsert_macro(history_dir, rows)
                panel = load_macro_panel(history_dir)
                if not panel.empty:
                    macro_panel = panel
    except (OSError, RuntimeError, ValueError) as error:
        statuses["macro"] = str(error)[:160]
    try:
        if config["earnings"]["enabled"]:
            earnings_rows: list[dict] = []
            for ticker in config["universe"]["us"]:
                try:
                    earnings_rows += fetch_us_earnings(ticker)
                except ProviderError as error:
                    statuses[f"earnings:{ticker}"] = str(error)[:160]
            earnings_rows += load_earnings_drops(manual_dir / "earnings", run_date)
            if earnings_rows:
                upsert_earnings(history_dir, earnings_rows)
            earnings_pairs = [(row["date"], row["asset"])
                              for row in load_earnings(history_dir)]
    except (OSError, RuntimeError, ValueError) as error:
        statuses["earnings"] = str(error)[:160]
    return macro_panel, earnings_pairs, statuses


def _sentiment_stats(labels: list[dict]) -> list[dict]:
    if not labels:
        return []
    frame = pd.DataFrame(labels)
    stats = frame.groupby("ticker").agg(mean_sentiment=("sentiment", "mean"),
                                        n_news=("sentiment", "count")).reset_index()
    return [{"asset": row.ticker, "mean_sentiment": round(float(row.mean_sentiment), 3),
             "n_news": int(row.n_news)} for row in stats.itertuples()]


def _upcoming_earnings(earnings_pairs: list[tuple[str, str]], assets: list[str],
                       run_date: str, *, limit: int = 10) -> list[dict]:
    """Ближайшие отчётности юниверса: актив, дата, торговых дней до неё."""
    universe = set(assets)
    upcoming = sorted((when, asset) for when, asset in earnings_pairs
                      if asset in universe and when > run_date)[:limit]
    dates = [row[0] for row in upcoming]
    return [{"asset": asset, "date": when} for when, asset in upcoming] if dates else []


def _top_news(labels: list[dict], items_by_id: dict[str, dict], *, limit: int = 10) -> list[dict]:
    by_news: dict[str, dict] = {}
    for label in labels:
        entry = by_news.setdefault(label["news_id"], {"sentiments": [], "tickers": []})
        entry["sentiments"].append(label["sentiment"] * label["confidence"])
        entry["tickers"].append(label["ticker"])
    ranked = sorted(by_news.items(),
                    key=lambda kv: max(abs(s) for s in kv[1]["sentiments"]), reverse=True)[:limit]
    result = []
    for news_id, entry in ranked:
        item = items_by_id.get(news_id, {})
        result.append({"id": news_id, "title": item.get("title", news_id),
                       "tickers": sorted(set(entry["tickers"])),
                       "sentiment": round(sum(entry["sentiments"]) / max(len(entry["sentiments"]), 1), 3)})
    return result


def run_daily(config: dict, config_path: str | Path, *, date: str | None = None,
              use_llm: bool = True, source: str = "net",
              fixture_dir: str | Path | None = None) -> dict[str, Any]:
    from ..progress import emit
    run_date = _resolve_date(date)
    emit("collect", "Сбор цен и новостей")
    paths = project_paths(config, config_path)
    raw_dir = paths["data"] / "raw"
    cache_dir = paths["data"] / "cache" / "llm_labels"

    # 1. Рыночные данные
    if source == "fixture":
        frames, provider_statuses, items, feed_statuses = _load_fixture(
            Path(fixture_dir or (paths["root"] / "fixtures" / "demo")), run_date)
    else:
        frames, market_payloads, provider_statuses = _collect_market_net(config, run_date)
        items, news_payloads, feed_statuses = _collect_news_net(config, run_date)
        market_entries = [archive(raw_dir, run_date, name, payload)
                          for name, payload in market_payloads]
        write_manifest(raw_dir, run_date, "market", market_entries,
                       {"source": "net", "providers": provider_statuses})
        news_entries = [archive(raw_dir, run_date, name, payload)
                        for name, payload in news_payloads]
        write_manifest(raw_dir, run_date, "news", news_entries,
                       {"source": "net", "feeds": feed_statuses})

    panel, dropped = price_features.build_panel(frames)
    if panel.shape[1] < MIN_ASSETS:
        raise RuntimeError(f"достаточно данных только по {panel.shape[1]} активам "
                           f"(нужно >= {MIN_ASSETS}); статусы провайдеров: {provider_statuses}")
    assets = panel.columns.tolist()
    returns = price_features.log_returns(panel)

    emit("labels", "Подготовка новостей к разметке")
    # 2. Новости -> LLM-разметка (вторая модальность)
    items = news_collector.select_for_budget(
        news_collector.dedupe(items),
        universe=_universe_ids(config),
        aliases=config["courts"]["aliases"],
        limit=config["news"]["max_items_per_run"],
        per_asset=config["news"]["per_asset_priority"])
    items_by_id = {item["id"]: item for item in items}
    llm_stats: dict[str, Any] = {"enabled": use_llm}
    llm_cfg = config["llm"]
    labels: list[dict] = []
    chat = None
    case_items: list[dict] = []
    if use_llm:
        llm_cfg = config["llm"]
        try:
            probe(llm_cfg["endpoint"], llm_cfg["model"])
            from ..llm.models import manager_for
            chat = make_chat(llm_cfg["endpoint"], llm_cfg["model"],
                             timeout_s=int(llm_cfg["timeout_s"]),
                             max_tokens=int(llm_cfg["max_tokens"]), seed=int(llm_cfg["seed"]),
                             recover=lambda: manager_for(config, config_path).ensure_ready(timeout=180))
            analysis = analyze_news(items, universe=assets, chat=chat, model=llm_cfg["model"],
                                    batch_size=int(llm_cfg["analyst_batch_size"]),
                                    cache_dir=cache_dir,
                                    cache_salt=f"mt{int(llm_cfg['max_tokens'])}")
            labels = analysis["labels"]
            llm_stats.update(analysis["stats"])
            rejected_path = paths["data"] / "rejected" / f"{run_date}.json"
            rejected_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_path.write_text(json.dumps(analysis["rejected"], ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        except LocalLLMError as error:
            llm_stats.update({"error": str(error), "note":
                              "LLM недоступна — прогон продолжен на ценовой модальности."})
    else:
        llm_stats["note"] = "LLM отключена флагом --no-llm."

    # 3. Мультимодальные расстояния
    D_price = price_features.distance_from_correlation(returns)
    sent_series = sentiment_features.daily_series(
        labels, items_by_id, price_index=panel.index,
        halflife_days=config["fusion"]["sentiment_halflife_days"], fallback_date=run_date)
    coverage = sentiment_features.coverage(labels, assets,
                                           min_news=config["fusion"]["min_news_per_asset"])
    D_sent = (price_features.distance_from_correlation(sent_series, min_periods=SENTIMENT_MIN_PERIODS)
              if not sent_series.empty else None)

    # 3b. Арбитражные суды: ручная выкладка -> LLM severity -> legal-вид
    legal_labels: list[dict] = []
    legal_matched: dict[str, int] = {}
    legal_cases_total = 0
    if config["courts"]["enabled"]:
        try:
            cases = load_court_drops(paths["data"] / "manual" / "courts", run_date)
            case_items, legal_matched = cases_to_items(
                cases, aliases=config["courts"]["aliases"], universe=assets)
            legal_cases_total = len(case_items)
            if case_items and chat is not None:
                courts_analysis = analyze_news(
                    case_items, universe=assets, chat=chat, model=llm_cfg["model"],
                    batch_size=int(llm_cfg["analyst_batch_size"]), cache_dir=cache_dir,
                    cache_salt="courts", system_prompt=CASES_SYSTEM_PROMPT)
                legal_labels = courts_analysis["labels"]
        except ProviderError as error:
            llm_stats.setdefault("courts_error", str(error)[:300])
    legal_items_by_id = {item["id"]: item for item in case_items} if config["courts"]["enabled"] else {}
    legal_series = (sentiment_features.daily_series(
        legal_labels, legal_items_by_id, price_index=panel.index,
        halflife_days=LEGAL_HALFLIFE_DAYS, fallback_date=run_date)
        if legal_labels else None)
    legal_coverage_frac = (sum(1 for a in assets if legal_matched.get(a, 0) >= 1) / len(assets)
                           if legal_matched else 0.0)
    D_legal = (price_features.distance_from_correlation(legal_series, min_periods=5)
               if legal_series is not None and not legal_series.empty else None)

    # 2c. A4: согласованность двух локальных LLM на одном потоке новостей.
    agreement: dict[str, Any] = {"enabled": False}
    agr_cfg = llm_cfg.get("agreement", {})
    if use_llm and agr_cfg.get("enabled"):
        agreement = {"enabled": True}
        try:
            probe(agr_cfg["endpoint"], agr_cfg["model"])
            chat_b = make_chat(agr_cfg["endpoint"], agr_cfg["model"],
                               timeout_s=int(llm_cfg["timeout_s"]),
                               max_tokens=int(llm_cfg["max_tokens"]),
                               seed=int(llm_cfg["seed"]))
            analysis_b = analyze_news(
                items, universe=assets, chat=chat_b, model=agr_cfg["model"],
                batch_size=int(llm_cfg["analyst_batch_size"]),
                cache_dir=cache_dir, cache_salt="agreement")
            agreement.update(compare_labels(labels, analysis_b["labels"]))
            agreement["stats_b"] = {k: v for k, v in analysis_b["stats"].items()
                                    if k in ("llm_calls", "cache_hits", "labels_total")}
            try:
                upsert_agreement(paths["data"] / "history", run_date, labels,
                                 analysis_b["labels"], model_a=llm_cfg["model"],
                                 model_b=agr_cfg["model"])
            except (OSError, RuntimeError, ValueError):
                pass  # история согласованности необязательна для прогона
        except LocalLLMError as error:
            agreement["error"] = str(error)[:300]

    # 3c. Vision: скриншоты графиков -> VLM -> tech-вид (отладочная модальность)
    chart_labels: list[dict] = []
    chart_statuses: list[dict] = []
    chart_drops = scan_chart_drops(paths["data"] / "manual" / "charts", run_date)
    if chart_drops:
        if use_llm:
            try:
                probe(llm_cfg["vision_endpoint"], llm_cfg["vision_model"])
                chart_labels, chart_statuses = analyze_charts(
                    chart_drops, universe=assets, endpoint=llm_cfg["vision_endpoint"],
                    model=llm_cfg["vision_model"], timeout=float(llm_cfg["vision_timeout_s"]),
                    seed=int(llm_cfg["seed"]), cache_dir=paths["data"] / "cache" / "vision")
            except LocalLLMError as error:
                chart_statuses = [{"asset": asset, "file": path.name, "ok": False,
                                   "error": f"vision endpoint: {str(error)[:160]}"}
                                  for asset, path in chart_drops]
        else:
            chart_statuses = [{"asset": asset, "file": path.name, "ok": False,
                               "error": "LLM отключена (--no-llm)"}
                              for asset, path in chart_drops]
    D_tech = None
    tech_coverage = 0.0
    if chart_labels:
        stances = {label["asset"]: label["stance"] for label in chart_labels}
        tech_frame = pd.DataFrame(0.0, index=panel.index,
                                  columns=[asset for asset in assets if asset in stances])
        tech_frame.iloc[-1] = [stances[asset] for asset in tech_frame.columns]
        D_tech = price_features.distance_from_correlation(tech_frame, min_periods=30)
        tech_coverage = len(stances) / len(assets)

    emit("portfolio", "Расчёт расстояний, рисков и весов HRP")
    distance_views = {"price": D_price, "sentiment": D_sent,
                      "legal": D_legal, "tech": D_tech}
    view_quality = {
        "sentiment": view_quality_mod.consistency(sent_series),
        "legal": view_quality_mod.consistency(legal_series),
        "tech": view_quality_mod.consistency(tech_frame) if chart_labels else float("nan"),
    }
    fused = fuse_distances(
        distance_views,
        base="price", method=config["fusion"]["method"],
        coverage={"sentiment": coverage["coverage_frac"], "legal": legal_coverage_frac,
                  "tech": tech_coverage},
        view_quality=view_quality,
        w_price=config["fusion"]["w_price"],
        min_coverage=config["fusion"]["min_coverage"],
        snf_neighbors=config["fusion"]["snf_neighbors"],
        snf_iterations=config["fusion"]["snf_iterations"],
        snf_min_quality=config["fusion"]["snf_min_quality"])

    # 4. Трекер редких событий: скор-таблица (и опциональный overlay потолков)
    tracker_cfg = config["tracker"]
    # B6/B7: макро-панель и календарь отчётностей для трекера
    macro_panel = None
    earnings_pairs: list[tuple[str, str]] = []
    b_statuses: dict[str, str] = {}
    if source == "net":
        macro_panel, earnings_pairs, b_statuses = collect_track_b(
            config, paths["data"] / "history", paths["data"] / "manual", run_date)

    tracker_rows: list[dict] = []
    caps: float | pd.Series = config["hrp"]["max_weight"]
    if tracker_cfg["enabled"]:
        tracker_kwargs = dict(weights=tracker_cfg["weights"],
                              halflife=tracker_cfg["vol_halflife_days"],
                              window=tracker_cfg["window_days"],
                              threshold=tracker_cfg["sigma_threshold"],
                              sentiment=sent_series if sent_series is not None and not sent_series.empty else None,
                              legal=legal_series,
                              macro=macro_panel if macro_panel is not None and not macro_panel.empty else None,
                              earnings_dates=earnings_pairs or None)
        explained = rare_events.explain_last(returns, panel, **tracker_kwargs)
        scores_last = explained["score"]
        warned = scores_last >= tracker_cfg["warn_threshold"]
        tracker_rows = [
            {"asset": asset,
             "score": round(float(row["score"]), 3),
             "p_tail": round(float(row["p_tail"]), 4) if pd.notna(row["p_tail"]) else None,
             "drawdown": round(float(row["drawdown"]), 3),
             "sentiment_z": round(float(row["sentiment_z"]), 2),
             "legal_flag": int(row["legal_flag"]),
             "macro_stress": round(float(row["macro_stress"]), 3),
             "earnings_flag": int(row["earnings_flag"]),
             "components": {key: float(row["contribution_" + key]) for key in
                            ("p_tail", "vol_of_vol", "drawdown", "sentiment", "legal", "macro", "earnings")},
             "bias": float(row["bias"]), "score_raw": float(row["score"]),
             "formula_version": "heuristic-v1-explained",
             "warned": bool(warned[asset])}
            for asset, row in explained.iterrows()]
        if tracker_cfg["overlay"]:
            caps = rare_events.overlay_caps(
                scores_last, max_weight=config["hrp"]["max_weight"],
                strength=tracker_cfg["overlay_strength"])

    # 4b. HRP: детерминированные веса
    base_cap = (caps if isinstance(caps, float) else float(config["hrp"]["max_weight"]))
    if base_cap < 1.0 / len(assets):
        raise RuntimeError(f"hrp.max_weight={base_cap} меньше 1/N "
                           f"({1.0 / len(assets):.3f}) — потолок неисполним")
    portfolio_returns, portfolio_distance = returns, fused["D"]
    image_materials = {"applied": False, "plan": None}
    if use_llm:
        try:
            from ..materials.portfolio import apply_daily
            portfolio_returns, portfolio_distance, image_materials = apply_daily(
                paths["data"] / "materials", run_date, returns, fused["D"])
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            image_materials = {"applied": False, "error": str(error)[:300]}
    portfolio = hrp_weights(portfolio_returns, portfolio_distance,
                            linkage_method=config["hrp"]["linkage"],
                            max_weight=caps,
                            cash_share=config["hrp"]["cash_share"])

    # 5. LLM-критик замороженного портфеля (веса не меняет)
    legal_stats = _sentiment_stats(legal_labels)
    rare_for_critic = sorted(tracker_rows, key=lambda row: -row["score"])[:8]
    critique: dict[str, Any] = {"failed": True, "summary": (
        "LLM отключена (--no-llm)." if not use_llm else llm_stats.get("error", "LLM недоступна.")), "findings": []}
    if use_llm and chat is not None:
        snapshot = {
            "date": run_date,
            "weights": {asset: float(weight) for asset, weight in portfolio["weights"].items()},
            "cash_share": portfolio["cash_share"],
            "movers": price_features.movers(panel),
            "sentiment_stats": _sentiment_stats(labels),
            "legal_stats": legal_stats,
            "rare_events": rare_for_critic,
            "clusters": portfolio.get("clusters") or {},
            "top_news": _top_news(labels, items_by_id),
            "fusion_note": fused["note"],
            "material_effects": image_materials,
            "coverage_note": (f"Новостное покрытие: {len(coverage['covered'])}/{len(assets)} "
                              f"активов (порог {config['fusion']['min_news_per_asset']} новостей). "
                              f"Судебные дела: {legal_cases_total} затрагивают юниверс, "
                              f"покрытие {legal_coverage_frac:.0%}. "
                              f"Трекер редких событий: предупреждений "
                              f"{sum(1 for row in tracker_rows if row['warned'])}/{len(assets)}."),
        }
        emit("critique", "Локальная модель рецензирует замороженный портфель")
        critique = critique_portfolio(snapshot, chat=chat)

    emit("save", "Сохранение прогноза, истории и отчёта")
    # 6. Отчёт + ledger
    report = {
        "date": run_date,
        "assets": assets,
        "dropped_assets": dropped,
        "provider_statuses": provider_statuses,
        "feed_statuses": feed_statuses,
        "news_total": len(items),
        "llm": llm_stats,
        "coverage": {k: v for k, v in coverage.items() if k != "counts"},
        "coverage_counts": coverage["counts"],
        "legal": {"cases_total": legal_cases_total,
                  "matched": legal_matched,
                  "coverage_frac": legal_coverage_frac,
                  "stats": legal_stats,
                  "labels_total": len(legal_labels)},
        "vision": {"charts_found": len(chart_drops),
                   "labels": [{k: v for k, v in label.items() if k != "meta"}
                              for label in chart_labels],
                   "statuses": chart_statuses},
        "materials": image_materials,
        "earnings_calendar": _upcoming_earnings(earnings_pairs, assets, run_date),
        "tracker": {"enabled": tracker_cfg["enabled"],
                    "overlay": bool(tracker_cfg["overlay"]),
                    "warn_threshold": tracker_cfg["warn_threshold"],
                    "rows": tracker_rows},
        "fusion": {"mode": fused["mode"], "note": fused["note"],
                   "method": config["fusion"]["method"],
                   "views_used": fused["views_used"],
                   "w_price": config["fusion"]["w_price"],
                   "view_quality": {k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in view_quality.items()}},
        "portfolio": {k: (v.tolist() if isinstance(v, pd.Series) else v)
                      for k, v in portfolio.items() if k != "weights"},
        "weights": {asset: float(weight) for asset, weight in portfolio["weights"].items()},
        "movers": price_features.movers(panel),
        "sentiment_stats": _sentiment_stats(labels),
        "top_news": _top_news(labels, items_by_id),
        "agreement": agreement,
        "critique": critique,
    }
    settlement_cfg = config["settlement"]
    try:
        history_dir = paths["data"] / "history"
        upsert_prices(history_dir, frames)
        upsert_labels(history_dir, labels, items_by_id, fallback_date=run_date)
        upsert_tracker(history_dir, run_date, tracker_rows)
    except (OSError, RuntimeError, ValueError) as error:
        report["history_error"] = str(error)[:300]
    if b_statuses:
        report["track_b_sources"] = b_statuses

    if source == "net" and (paths["data"] / "runtime" / "web" / "observations.sqlite3").is_file():
        try:
            from ..tracker.observations import view as observation_view
            observation_view(paths, config)
        except Exception as error:
            # User observations are independent; failure must not stop daily.
            report["observations_error"] = str(error)[:300]

    # A1: авто-сеттмент pending-прогнозов (история только что пополнилась),
    # затем заморозка прогноза этого дня. Сбои не роняют прогон.
    settlement_summary: dict[str, Any] = {}
    try:
        if settlement_cfg["auto"]:
            settlement_summary = settle_pending(
                paths["data"] / "forecasts", paths["data"] / "settlement",
                history_dir, panel,
                threshold=tracker_cfg["sigma_threshold"],
                halflife=tracker_cfg["vol_halflife_days"],
                horizon=int(settlement_cfg["horizon_days"]))
        forecast_path = write_forecast(paths["data"] / "forecasts", {
            "date": run_date,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "weights": report["weights"],
            "cash_share": float(report["portfolio"].get("cash_share", 0.0)),
            "fusion": report["fusion"],
            "materials": image_materials,
            "coverage": report["coverage"],
            "sentiment_stats": report["sentiment_stats"],
            "legal_stats": legal_stats,
            "labels": [{k: v for k, v in label.items()}
                       for label in labels],
            "tracker_scores": {row["asset"]: row["score"] for row in tracker_rows},
            "tracker_warned": {row["asset"]: row["warned"] for row in tracker_rows},
            "views": {name: (D.round(6).to_dict() if D is not None else None)
                      for name, D in distance_views.items()},
            "config_digest": _config_digest(config),
            "universe": assets,
        })
        settlement_summary["forecast"] = str(forecast_path)
    except (OSError, RuntimeError, ValueError) as error:
        settlement_summary["error"] = str(error)[:300]
    report["settlement"] = settlement_summary
    if source == "net" and use_llm and (config["news"]["telegram"]["enabled"] or config["courts"]["enabled"]):
        try:
            from ..research.events import refresh_events
            emit("events", "Отдельная история Telegram и судебных событий")
            report["event_sources"] = refresh_events(config, config_path, run_date, chat=chat)
        except (OSError, RuntimeError, ValueError, ProviderError) as error:
            report["event_sources"] = {"error": str(error)[:300]}
    report_paths = write_report(paths["reports"], report)
    ledger_path = append_ledger(paths["data"], {
        "date": run_date,
        "mode": report["fusion"]["mode"],
        "n_assets": len(assets),
        "n_news": len(items),
        "n_labels": len(labels),
        "llm_enabled": use_llm,
        "weights": report["weights"],
        "report": str(report_paths["markdown"].relative_to(paths["root"])),
    })
    return {"report": report, "paths": {**{k: str(v) for k, v in report_paths.items()},
                                        "ledger": str(ledger_path)}}
