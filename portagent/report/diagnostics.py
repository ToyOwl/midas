"""Цены, наблюдённые шоки и объяснения; исходные портфели только читаются."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import prices, sentiment
from ..history import load_price_panel, load_macro_panel, load_earnings
from ..scheduler import _write_json
from ..tracker import rare_events, top3
from .recovery import archived_prices, digest
from .briefing import asset_brief

NAMES = {"p_tail": "Хвостовая оценка относительно других активов",
         "vol_of_vol": "Нестабильность волатильности", "drawdown": "Глубина просадки относительно других активов",
         "sentiment": "Отклонение новостного тона", "legal": "Судебное давление",
         "macro": "Чувствительность к макрострессу", "earnings": "Близкая отчётность"}

PLAIN_REASONS = {
    "p_tail": "По истории цен оценка риска резкого падения выше, чем у большинства бумаг в портфеле.",
    "vol_of_vol": "Размер ежедневных колебаний стал менее устойчивым по сравнению с другими бумагами портфеля.",
    "drawdown": "Падение от недавнего максимума глубже, чем у большинства бумаг портфеля.",
    "macro": "В расчёте эта бумага сильнее других реагирует на изменения внешних рыночных факторов.",
    "legal": "В исходных данных есть сообщения о судебном давлении на компанию.",
    "earnings": "Приближается публикация отчётности — она может сопровождаться заметным движением цены.",
}


def explanations(report, reconstructed=None, moves=None):
    threshold = report.get("tracker", {}).get("warn_threshold", .5)
    news_rows = report.get("news_labels", [])
    items = {i["id"]: i for i in report.get("news_items", [])}
    movements = {m["asset"]: m for m in (moves if moves is not None else report.get("movers", []))}
    out = {}
    for row in report.get("tracker", {}).get("rows", []):
        asset = row["asset"]
        components = row.get("components")
        bias = row.get("bias")
        exact = bool(components)
        if not exact and reconstructed is not None and asset in reconstructed.index:
            source = reconstructed.loc[asset]
            if abs(float(source["score"]) - row["score"]) <= .00051:
                components = {k: float(source["contribution_" + k]) for k in NAMES}
                bias = float(source["bias"])
                exact = True
        facts = [f"Сохранённый скор {row['score']:.3f} {'≥' if row.get('warned') else '<'} порога {threshold:g}.",
                 f"Наблюдаемая просадка от максимума окна: {row.get('drawdown', 0):.1%}."]
        if exact:
            ranked = sorted(components.items(), key=lambda kv: -kv[1])
            reasons = [f"{NAMES[k]}: вклад {v:+.2f} в сумму перед логистическим преобразованием."
                       for k, v in ranked if v > .01][:3]
            facts += reasons
            if components.get("sentiment", 0) > .01:
                facts.append("В исходной формуле положительное отклонение сентимента повышает скор. Это особенность эвристики, а не свидетельство негативных новостей.")
        else:
            if row.get("p_tail") is not None:
                facts.append(f"Однодневная хвостовая оценка: {row['p_tail']:.1%}; это отдельный показатель, не весь скор.")
            if row.get("legal_flag"):
                facts.append("В исходном расчёте отмечено судебное давление.")
            if row.get("earnings_flag"):
                facts.append("В исходном расчёте отмечена близкая отчётность.")
            facts.append("Полного разложения старого скора нет; перечислены сохранённые факторы, их точные вклады не установлены.")
        linked = []
        for label in sorted([r for r in news_rows if r["ticker"] == asset], key=lambda r: r["sentiment"]):
            if label["sentiment"] >= 0 or len(linked) >= 3:
                continue
            item = items.get(label["news_id"], {})
            linked.append({"title": item.get("title", label["news_id"]), "quote": label["quote"],
                           "link": item.get("link", ""), "sentiment": label["sentiment"]})
        human = []
        if row.get("drawdown") is not None:
            dd = row["drawdown"]
            human.append(f"Цена на {dd:.1%} ниже максимума за рассматриваемый период.".replace(".", ",", 1) if dd >= .0005
                         else "Цена находится вблизи максимума за рассматриваемый период.")
        caveat = None
        if exact:
            positive = [k for k, v in sorted(components.items(), key=lambda kv: -kv[1]) if v > .01 and k in PLAIN_REASONS]
            human += [PLAIN_REASONS[k] for k in positive[:3]]
            if components.get("sentiment", 0) > .01:
                caveat = ("Новостной фон в исходном расчёте стал позитивнее обычного. Старая формула тоже усиливает "
                          "предупреждение при таком улучшении. Это особенность расчёта, а не признак плохих новостей.")
        else:
            if row.get("legal_flag"):
                human.append(PLAIN_REASONS["legal"])
            if row.get("earnings_flag"):
                human.append(PLAIN_REASONS["earnings"])
            caveat = "В архиве не хватает данных, чтобы точно объяснить все причины этого предупреждения."
        if not human:
            human = ["Индикатор пересёк порог предупреждения." if row.get("warned") else "Индикатор ниже порога предупреждения."]
        briefing = asset_brief(row, components, movements.get(asset))
        brief = briefing["situation"] + " " + briefing["importance"]
        # Keep exact arithmetic available, while the primary explanation uses ordinary language.
        out[asset] = {"asset": asset, "warned": row.get("warned", False), "facts": human + ([caveat] if caveat else []),
            "summary": " ".join(human), "short_reason": next((PLAIN_REASONS[k] for k, v in sorted((components or {}).items(), key=lambda kv: -kv[1])
                if v > .01 and k in PLAIN_REASONS), human[0]), "caveat": caveat, "technical_facts": facts,
            "brief": brief, "briefing": briefing,
            "exact": exact, "components": [{"id": k, "name": NAMES[k], "contribution": float(v)}
                for k, v in (components or {}).items()], "bias": bias, "news": linked,
            "score": row["score"], "threshold": threshold}
    return out


def reconstruct_scores(paths, report, config, frames):
    """Разложение публикуется только если восстановленная сумма совпала со старым скором."""
    try:
        panel, _ = prices.build_panel(frames)
        returns = prices.log_returns(panel)
        forecast = json.loads((paths["data"] / "forecasts" / f"{report['date']}.json").read_text(encoding="utf-8"))
        labels = copy.deepcopy(forecast.get("snapshot", forecast).get("labels", []))
        by_id = {r["news_id"]: {"published_at": r.get("date", report["date"])} for r in labels}
        sent = sentiment.daily_series(labels, by_id, price_index=panel.index,
            halflife_days=config["fusion"]["sentiment_halflife_days"], fallback_date=report["date"])
        macro = load_macro_panel(paths["data"] / "history")
        macro = macro.loc[macro.index <= report["date"]] if not macro.empty else None
        earnings = [(r["date"], r["asset"]) for r in load_earnings(paths["data"] / "history")]
        cfg = config["tracker"]
        return rare_events.explain_last(returns, panel, weights=cfg["weights"],
            halflife=cfg["vol_halflife_days"], window=cfg["window_days"], threshold=cfg["sigma_threshold"],
            sentiment=sent, macro=macro, earnings_dates=earnings)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build(paths, report, config):
    day = report["date"]
    try:
        frames = archived_prices(paths, day)
        origin = "Котировки из архива выбранного расчёта"
    except FileNotFoundError:
        panel = load_price_panel(paths["data"] / "history", end=day)
        frames = {a: pd.DataFrame({"close": panel[a].dropna()}) for a in report["weights"] if a in panel}
        origin = "Накопленная история, ограниченная датой отчёта"
    frames = {a: f for a, f in frames.items() if a in report["weights"]}
    cfg = config["tracker"]
    series, moves = {}, []
    for asset, frame in frames.items():
        close = frame["close"].dropna().astype(float)
        close = close[(close > 0) & np.isfinite(close)]
        if close.empty:
            continue
        event = top3.event_features(close, threshold=cfg["sigma_threshold"],
                                    halflife=cfg["vol_halflife_days"], window=cfg["window_days"])
        data = [[str(d), float(v)] for d, v in close.items()]
        observed = [{"date": str(d), "z": float(z), "direction": "up" if z > 0 else "down"}
                    for d, z in event["z"].dropna().items() if abs(z) >= cfg["sigma_threshold"]]
        one = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else None
        five = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else None
        series[asset] = {"asset": asset, "currency": "RUB" if asset.startswith("MOEX:") else "USD",
            "adjustment": "MOEX: биржевая цена" if asset.startswith("MOEX:") else "Yahoo: скорректированная цена",
            "points": data, "events": observed, "last": data[-1], "ret_1d": one, "ret_5d": five}
        moves.append({"asset": asset, "ret_1d": one, "ret_5d": five})
    detector = top3.detect(frames, day=day, threshold=cfg["sigma_threshold"],
                           halflife=cfg["vol_halflife_days"], window=cfg["window_days"])
    return {"date": day, "origin": origin, "series": series, "movers": moves, "detector": detector,
            "risk_explanations": explanations(report, reconstruct_scores(paths, report, config, frames), moves)}


def get_diagnostics(paths, report, config):
    # Ключ включает архив, обновлённый анализ и параметры. Кэш не меняет исходные файлы.
    manifests = []
    for kind in ("market",):
        path = paths["data"] / "raw" / report["date"] / f"manifest_{kind}.json"
        manifests.append(digest(path.read_bytes()) if path.is_file() else "history")
    # Разложение старого скора использует замороженные метки и историю B-признаков.
    dependencies = [paths["data"] / "forecasts" / f"{report['date']}.json"]
    dependencies += [paths["data"] / "history" / name for name in ("macro.parquet", "earnings.parquet")]
    if "history" in manifests:
        dependencies.append(paths["data"] / "history" / "prices.parquet")
    manifests += [digest(p.read_bytes()) if p.is_file() else "missing" for p in dependencies]
    briefing_version = digest(Path(__file__).with_name("briefing.py").read_bytes())
    key = digest(json.dumps([8, briefing_version, top3.VERSION, report, config["tracker"], config["fusion"], manifests],
                            sort_keys=True, ensure_ascii=False).encode())
    target = paths["data"] / "cache" / "diagnostics" / (key + ".json")
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    result = build(paths, report, config)
    _write_json(target, result)
    return result
