"""Генератор демо-fixture: 3 кластера по 4 актива (коррелированные доходности,
520 дней) + 36 новостных элементов с кластерным сентиментальным уклоном.

Дополнительно (флагами или всегда в корень проекта):
- courts_sample.json: демо-выкладка арбитражных дел (копируется в
  .demo/data/manual/courts/<anchor>/) — активирует legal-модальность в демо;
- скриншоты графиков PNG (.demo/data/manual/charts/<anchor>/) — для vision-модальности
  (чистый PNG-энкодер на zlib, без зависимостей).

Запуск: python scripts/make_demo_fixture.py [каталог fixture]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np

CLUSTERS = {
    "ru-oil": ["MOEX:SBER", "MOEX:GAZP", "MOEX:LKOH", "MOEX:ROSN"],
    "ru-metal": ["MOEX:GMKN", "MOEX:PLZL", "MOEX:MGNT", "MOEX:MTSS"],
    "us-tech": ["US:AAPL", "US:MSFT", "US:NVDA", "US:GOOGL"],
}
SENTIMENT_BIAS = {"ru-oil": 0.45, "ru-metal": -0.25, "us-tech": 0.15}
DAYS = 520
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEMO_CASES = [
    # (alias в тексте, тип дела, дней назад)
    ("ПАО Сбербанк", "Исковое производство: налоговые претензии на 12 млрд руб.", 28),
    ("Газпром", "Спор о тарифах на транспортировку", 24),
    ("ЛУКОЙЛ", "Иск о взыскании неустойки по контракту", 20),
    ("Норникель", "Дело о возмещении экологического ущерба", 16),
    ("Роснефть", "Оспаривание кадастровой стоимости", 12),
    ("МТС", "Иск потребителя о защите прав", 8),
    ("Магнит", "Спор с поставщиком о качестве продукции", 6),
    ("Полюс", "Банкротство дочернего подрядчика", 3),
    ("Apple", "Антимонопольное дело о платёжной системе", 2),
]


def write_png(path: Path, width: int, height: int, rgb: np.ndarray) -> None:
    """Минимальный PNG-энкодер (RGB, без фильтрации) на stdlib."""
    raw = b"".join(b"\x00" + row.tobytes() for row in rgb.reshape(height, width * 3))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def render_chart_png(path: Path, closes: list[float]) -> None:
    """Линейный график цены (белый фон, чёрная линия, ось) — заготовка для VLM."""
    width, height = 640, 320
    margin = 24
    values = np.asarray(closes[-180:], dtype=float)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    rgb[height - margin, :, :] = 120                      # ось X
    rgb[:, margin, :] = 120                                # ось Y
    for x in range(width - 2 * margin):
        t = x / max(width - 2 * margin - 1, 1)
        y0 = values[int(t * (len(values) - 1))]
        y = int((height - margin) - (y0 - lo) / span * (height - 2 * margin))
        y = max(min(y, height - margin - 1), margin)
        rgb[y - 1:y + 2, margin + x, :] = (20, 40, 200)   # линия цены
    write_png(path, width, height, rgb)


def main(target: Path, data_dir: Path | None = None) -> None:
    data_dir = data_dir if data_dir is not None else PROJECT_ROOT / ".demo" / "data"
    rng = np.random.default_rng(SEED)
    target.mkdir(parents=True, exist_ok=True)
    anchor = date.today()
    dates = [(anchor - timedelta(days=DAYS - 1 - i)).isoformat() for i in range(DAYS)]

    candles_by_asset: dict[str, list[list]] = {}
    for cluster_name, assets in CLUSTERS.items():
        factor = rng.normal(0.0, 0.010, DAYS)  # общий фактор кластера
        for asset in assets:
            idio = rng.normal(0.0, 0.006, DAYS)
            returns = factor + idio
            closes = 100.0 * np.exp(np.cumsum(returns))
            volumes = rng.integers(10**5, 10**7, DAYS).astype(float)
            candles = []
            for i, day in enumerate(dates):
                close = float(closes[i])
                open_ = close * float(1 + rng.normal(0, 0.002))
                high = max(open_, close) * 1.002
                low = min(open_, close) * 0.998
                candles.append([day, round(open_, 2), round(high, 2),
                                round(low, 2), round(close, 2), float(volumes[i])])
            candles_by_asset[asset] = candles

    for asset, candles in candles_by_asset.items():
        payload = {"candles": {"columns": ["begin", "open", "high", "low", "close", "volume"],
                               "data": candles}}
        name = "candles_" + asset.replace(":", "__") + ".json"
        (target / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Новости: каждому активу >= 3 элементов за последние 10 дней, сентимент с
    # кластерным уклоном — сентимент-модальность коррелирует внутри кластеров.
    templates = [
        ("«{name}» отчиталась о рекордной выручке за квартал", +0.8,
         "Компания «{name}» отчиталась о рекордной выручке за квартал и превысила прогнозы аналитиков."),
        ("Совет директоров «{name}» повысил дивиденды", +0.6,
         "Совет директоров «{name}» повысил дивиденды и подтвердил программу обратного выкупа акций."),
        ("«{name}» проиграла суд по налоговым претензиям", -0.7,
         "«{name}» проиграла суд по налоговым претензиям и обязана выплатить штраф в бюджет."),
        ("«{name}» снизила прогноз продаж на год", -0.5,
         "«{name}» снизила прогноз продаж на год из-за ослабления спроса на ключевом рынке."),
        ("«{name}» открыла новое производство", +0.3,
         "«{name}» открыла новое производство и расширяет присутствие на региональных рынках."),
    ]
    items = []
    index = 0
    for cluster_name, assets in CLUSTERS.items():
        for asset in assets:
            for k in range(3):
                title_tpl, base_sent, text_tpl = templates[(index + k) % len(templates)]
                name = asset.split(":", 1)[1]
                bias = SENTIMENT_BIAS[cluster_name]
                # смещение выбора шаблона под уклон кластера
                if bias > 0 and base_sent < 0:
                    title_tpl, base_sent, text_tpl = templates[0]
                if bias < 0 and base_sent > 0:
                    title_tpl, base_sent, text_tpl = templates[2]
                # распределяем по ~120 дням: сентимент-вид живёт на нескольких
                # ребалансировках бэктеста, а не только на последней
                day = date.today() - timedelta(days=5 + k * 40)
                items.append({
                    "id": f"demo-{asset.lower().replace(':', '-')}-{k}",
                    "title": title_tpl.format(name=name),
                    "text": text_tpl.format(name=name),
                    "link": "",
                    "published_at": f"{day.isoformat()}T09:00:00+00:00",
                    "source": "demo",
                })
                index += 1
    (target / "news.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (target / "meta.json").write_text(json.dumps({
        "generated_for": anchor.isoformat(), "assets": sorted(candles_by_asset),
        "clusters": CLUSTERS, "news_items": len(items)}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # Демо-выкладка арбитражных дел (legal-модальность): 9 дел за 28 дней,
    # затрагивают 9 из 12 активов через алиасы courts.aliases.
    cases = [{
        "case_number": f"А40-{1000 + i * 7}/26",
        "date": (anchor - timedelta(days=days_ago)).isoformat(),
        "type": "Исковое производство",
        "plaintiff": "ООО Истец",
        "defendant": alias,
        "summary": f"{kind} в отношении {alias}.",
    } for i, (alias, kind, days_ago) in enumerate(DEMO_CASES)]
    (target / "courts_sample.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    courts_dir = data_dir / "manual" / "courts" / anchor.isoformat()
    courts_dir.mkdir(parents=True, exist_ok=True)
    (courts_dir / "demo.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")

    # Демо-скриншоты графиков (vision-модальность): все активы юниверса —
    # покрытие 100% >= fusion.min_coverage, вид D_tech участвует в SNF.
    charts_dir = data_dir / "manual" / "charts" / anchor.isoformat()
    charts_dir.mkdir(parents=True, exist_ok=True)
    for asset, candles in candles_by_asset.items():
        closes = [row[4] for row in candles]
        render_chart_png(charts_dir / (asset.replace(":", "__") + ".png"), closes)

    print(f"fixture готов: {target} ({len(candles_by_asset)} активов, {len(items)} новостей, "
          f"{len(cases)} судебных дел, {len(candles_by_asset)} скриншотов)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate isolated synthetic demo data")
    parser.add_argument("target", nargs="?", type=Path, default=PROJECT_ROOT / "fixtures" / "demo")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / ".demo" / "data")
    args = parser.parse_args()
    main(args.target, args.data_dir)
