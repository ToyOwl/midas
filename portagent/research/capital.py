"""Daily capital curve from the separately collected price snapshot."""
from pathlib import Path
from .dataset import load_snapshot
from .io import write_json
from ..backtest import walk_forward
from ..features.prices import build_panel


def calculate(config, market_root, target):
    frames, manifest = load_snapshot(market_root)
    panel, _ = build_panel(frames)
    # Historical news and manual demo dividends are deliberately not backfilled.
    result = walk_forward(panel, config=config, include_daily=True)
    result["provenance"] = {"source": "fresh_provider_snapshot", "manifest": str(Path(market_root) / "manifest.json"),
        "asof": manifest["day"], "currency": "local_returns_no_fx_conversion", "dividends": "MOEX not included; Yahoo adjusted close",
        "note": "Исторический расчёт; индекс стартует со 100. MOEX без полной поправки на корпоративные действия; Yahoo скорректирован. Валютная переоценка отсутствует. Исторические новости и демонстрационные дивиденды не используются."}
    write_json(target, result)
    return result
