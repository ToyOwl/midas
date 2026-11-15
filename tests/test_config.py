import pytest

from portagent.config import ConfigError, load_config

from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_default_config_loads():
    config = load_config(CONFIG)
    assert config["hrp"]["linkage"] == "single"
    assert config["fusion"]["w_price"] + (1 - config["fusion"]["w_price"]) == 1.0


def test_unknown_key_rejected(tmp_path):
    bad = CONFIG.read_text(encoding="utf-8") + "\nunknown_section: 1\n"
    target = tmp_path / "bad.yaml"
    target.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        load_config(target)


def test_missing_key_rejected(tmp_path):
    bad = CONFIG.read_text(encoding="utf-8").replace("  cash_share: 0.10\n", "")
    target = tmp_path / "bad.yaml"
    target.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="cash_share"):
        load_config(target)


def test_bad_value_rejected(tmp_path):
    bad = CONFIG.read_text(encoding="utf-8").replace("max_weight: 0.25", "max_weight: 1.7")
    target = tmp_path / "bad.yaml"
    target.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(target)


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTAGENT_LLM_ENDPOINT", "http://127.0.0.1:9999")
    monkeypatch.setenv("PORTAGENT_LLM_MODEL", "other-model")
    config = load_config(CONFIG)
    assert config["llm"]["endpoint"] == "http://127.0.0.1:9999"
    assert config["llm"]["model"] == "other-model"


def test_midas_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("PORTAGENT_LLM_MODEL", "legacy")
    monkeypatch.setenv("MIDAS_LLM_MODEL", "current")
    monkeypatch.setenv("MIDAS_LLM_ENDPOINT", "http://127.0.0.1:9998")
    config = load_config(CONFIG.with_name("example.yaml"))
    assert config["llm"]["model"] == "current"
    assert config["llm"]["endpoint"] == "http://127.0.0.1:9998"
