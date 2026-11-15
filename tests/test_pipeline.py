"""E2E: run_daily на fixture-данных без сети (LLM отключена и недоступна)."""
import json
from pathlib import Path

import pytest
import yaml

from portagent.pipeline.daily import run_daily

CONFIG_SOURCE = (Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text(
    encoding="utf-8")


def _config_in(tmp_path):
    """Копия дефолтного конфига в tmp — данные/отчёты не трогают проект."""
    target = tmp_path / "configs" / "default.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(CONFIG_SOURCE, encoding="utf-8")
    return target


def test_daily_no_llm(tmp_path, mini_fixture):
    result = run_daily(yaml.safe_load(CONFIG_SOURCE), _config_in(tmp_path),
                       date="2026-11-11", use_llm=False,
                       source="fixture", fixture_dir=mini_fixture)
    report = result["report"]
    weights = report["weights"]
    assert len(weights) == 12
    assert sum(weights.values()) == pytest.approx(0.90, abs=1e-9)  # cash_share 0.10
    assert report["fusion"]["mode"] == "single"  # LLM не размечала новости
    report_dir = tmp_path / "reports" / "2026-11-11"
    markdown = (report_dir / "portfolio.md").read_text(encoding="utf-8")
    analysis = (report_dir / "analysis.md").read_text(encoding="utf-8")
    assert "## Паспорт" in markdown
    assert "## Концентрация" in markdown
    assert "## Трекер редких событий" in markdown
    assert "## Метаданные запуска" in markdown
    assert "## Паспорт" in analysis
    assert "## Метаданные запуска" not in analysis
    assert (tmp_path / "data" / "history" / "prices.parquet").is_file()
    assert (tmp_path / "reports" / "2026-11-11" / "weights.csv").exists()
    ledger = (tmp_path / "data" / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger) == 1
    assert json.loads(ledger[0])["mode"] == "single"


def test_daily_llm_unavailable_degrades(tmp_path, mini_fixture):
    """LLM включена, но endpoint недоступен -> прогон продолжает price-only."""
    config = yaml.safe_load(CONFIG_SOURCE)
    config["llm"]["endpoint"] = "http://127.0.0.1:59999"  # гарантированно пустой порт
    result = run_daily(config, _config_in(tmp_path), date="2026-11-11",
                       use_llm=True, source="fixture", fixture_dir=mini_fixture)
    report = result["report"]
    assert report["llm"]["enabled"] is True
    assert "error" in report["llm"]
    assert report["fusion"]["mode"] == "single"
    assert len(report["weights"]) == 12  # портфель всё равно построен
    assert report["critique"]["failed"]


def test_daily_rejects_too_few_assets(tmp_path, mini_fixture):
    config = yaml.safe_load(CONFIG_SOURCE)
    # оставляем в fixture только 3 актива
    for path in list(mini_fixture.glob("candles_*.json"))[3:]:
        path.unlink()
    with pytest.raises(RuntimeError, match="активам"):
        run_daily(config, _config_in(tmp_path), date="2026-11-11",
                  use_llm=False, source="fixture", fixture_dir=mini_fixture)
