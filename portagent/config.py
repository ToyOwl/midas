"""Строгий контракт YAML-конфига.

Неизвестный ключ или неверный тип — ошибка конфигурации, тихих значений по
умолчанию для опечаток нет. env-переменные PORTAGENT_LLM_ENDPOINT и
PORTAGENT_LLM_MODEL переопределяют llm-секцию без правки файла.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _google_locales(value: object) -> bool:
    """Список локалей Google News: у каждой ровно hl/gl/ceid, без лишних ключей."""
    if not isinstance(value, list) or not value:
        return False
    required = {"hl", "gl", "ceid"}
    for item in value:
        if not isinstance(item, dict) or set(item) != required:
            return False
        if not all(isinstance(item[key], str) and item[key] for key in required):
            return False
    return True


# Схема: ключ -> (тип, обязательный, валидатор значения или None)
_LLM_SCHEMA = {
    "endpoint": (str, True, None),
    "model": (str, True, None),
    "runtime_root": (str, True, None),
    "timeout_s": ((int, float), True, lambda v: v > 0),
    "max_tokens": (int, True, lambda v: v > 0),
    "seed": (int, True, None),
    "analyst_batch_size": (int, True, lambda v: 1 <= v <= 50),
    "vision_endpoint": (str, True, None),
    "vision_model": (str, True, None),
    "vision_timeout_s": ((int, float), True, lambda v: v > 0),
    "agreement": {
        "enabled": (bool, True, None),
        "endpoint": (str, True, None),
        "model": (str, True, None),
    },
}

_TRACKER_WEIGHTS = {
    key: ((int, float), True, lambda v: -10.0 <= v <= 10.0)
    for key in ("p_tail", "vol_of_vol", "drawdown", "sentiment", "legal",
                "macro", "earnings", "bias")
}

_FULL_SCHEMA: dict[str, Any] = {
    "universe": {
        "moex": (list, True, lambda v: v and all(isinstance(x, str) and x for x in v)),
        "us": (list, True, lambda v: v and all(isinstance(x, str) and x for x in v)),
    },
    "providers": {
        "moex": {
            "enabled": (bool, True, None),
            "board": (str, True, None),
            "history_days": (int, True, lambda v: v >= 60),
            "timeout_s": ((int, float), True, lambda v: v > 0),
            "retries": (int, True, lambda v: 0 <= v <= 5),
        },
        "yfinance": {
            "enabled": (bool, True, None),
            "history_days": (int, True, lambda v: v >= 60),
            "timeout_s": ((int, float), True, lambda v: v > 0),
        },
    },
    "news": {
        "rss": (list, True, lambda v: all(isinstance(x, str) and x.startswith("http") for x in v)),
        "telegram": {
            "enabled": (bool, True, None),
            "channels": (list, True, lambda v: all(isinstance(x, str) and x and "/" not in x for x in v)),
            "pages": (int, True, lambda v: 1 <= v <= 10),
        },
        "moex_sitenews": {
            "enabled": (bool, True, None),
            "lookback_hours": ((int, float), True, lambda v: v > 0),
            "limit": (int, True, lambda v: 1 <= v <= 500),
        },
        "lookback_hours": ((int, float), True, lambda v: v > 0),
        "max_items_per_run": (int, True, lambda v: 1 <= v <= 1000),
        "per_asset_priority": (int, True, lambda v: 1 <= v <= 50),
        "fetch_timeout_s": ((int, float), True, lambda v: v > 0),
        "google_news": {
            "enabled": (bool, True, None),
            "per_query_limit": (int, True, lambda v: 1 <= v <= 50),
            "locales": (list, True, _google_locales),
        },
    },
    "courts": {
        "enabled": (bool, True, None),
        # свободный словарь тикер -> список алиасов (проверяем значения)
        "aliases": (dict, True, lambda v: v and all(
            isinstance(k, str) and k and isinstance(x, list) and x
            and all(isinstance(a, str) and a for a in x) for k, x in v.items())),
    },
    "llm": _LLM_SCHEMA,
    "materials": {
        "embedding_python": (str, True, None),
        "embedding_model": (str, True, None),
        "embedding_revision": (str, True, None),
        "max_upload_mb": (int, True, lambda v: 1 <= v <= 20),
        "max_pixels": (int, True, lambda v: 1000000 <= v <= 40000000),
        "influence": ((int, float), True, lambda v: 0 < v <= 0.3),
        "valid_days": (int, True, lambda v: 1 <= v <= 30),
    },
    "fusion": {
        "method": (str, True, lambda v: v in {"snf", "linear"}),
        "w_price": ((int, float), True, lambda v: 0.0 <= v <= 1.0),
        "snf_neighbors": (int, True, lambda v: 3 <= v <= 100),
        "snf_iterations": (int, True, lambda v: 1 <= v <= 200),
        "snf_min_quality": ((int, float), True, lambda v: 0.0 <= v < 1.0),
        "min_coverage": ((int, float), True, lambda v: 0.0 < v <= 1.0),
        "min_news_per_asset": (int, True, lambda v: 1 <= v <= 100),
        "sentiment_halflife_days": ((int, float), True, lambda v: v > 0),
    },
    "tracker": {
        "enabled": (bool, True, None),
        "sigma_threshold": ((int, float), True, lambda v: 1.0 <= v <= 5.0),
        "window_days": (int, True, lambda v: 60 <= v <= 1000),
        "vol_halflife_days": ((int, float), True, lambda v: v > 0),
        "warn_threshold": ((int, float), True, lambda v: 0.0 < v < 1.0),
        "overlay": (bool, True, None),
        "overlay_strength": ((int, float), True, lambda v: 0.0 <= v <= 1.0),
        "weights": _TRACKER_WEIGHTS,
    },
    "backtest": {
        "rebalance_days": (int, True, lambda v: 1 <= v <= 260),
        "fetch_days": (int, True, lambda v: 365 <= v <= 5000),
        "commission_bps": ((int, float), True, lambda v: 0 <= v <= 500),
        "spread_bps": ((int, float), True, lambda v: 0 <= v <= 2000),
    },
    "dividends": {
        "enabled": (bool, True, None),
    },
    "macro": {
        "enabled": (bool, True, None),
        "usdrub_secid": (str, True, None),
        "yf_series": (list, True, lambda v: bool(v) and all(
            isinstance(x, dict) and set(x) == {"symbol", "name"}
            and isinstance(x["symbol"], str) and isinstance(x["name"], str)
            for x in v)),
    },
    "earnings": {
        "enabled": (bool, True, None),
    },
    "settlement": {
        "auto": (bool, True, None),
        "horizon_days": (int, True, lambda v: 1 <= v <= 30),
    },
    "ablation": {
        "bootstrap_reps": (int, True, lambda v: 50 <= v <= 5000),
        "block_days": (int, True, lambda v: 1 <= v <= 30),
    },
    "hrp": {
        "linkage": (str, True, lambda v: v in {"single", "complete", "average", "ward"}),
        "max_weight": ((int, float), True, lambda v: 0.0 < v <= 1.0),
        "cash_share": ((int, float), True, lambda v: 0.0 <= v < 1.0),
    },
    "paths": {
        "data_dir": (str, True, None),
        "reports_dir": (str, True, None),
    },
}


def _check(section: dict, schema: dict[str, Any], where: str) -> None:
    if not isinstance(section, dict):
        raise ConfigError(f"{where}: ожидался словарь, получен {type(section).__name__}")
    for key, rule in schema.items():
        nested = isinstance(rule, dict)
        if key not in section:
            if nested or rule[1]:  # вложние секции всегда обязательны
                raise ConfigError(f"{where}.{key}: обязательный ключ отсутствует")
            continue
        value = section[key]
        if nested:
            _check(value, rule, f"{where}.{key}")
            continue
        kind, _required, validator = rule
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if isinstance(value, bool) and bool not in kinds:
            names = "/".join(k.__name__ for k in kinds)
            raise ConfigError(f"{where}.{key}: ожидался {names}, получен bool")
        if not isinstance(value, kind):
            names = "/".join(k.__name__ for k in kinds)
            raise ConfigError(f"{where}.{key}: ожидался {names}, получен {type(value).__name__}")
        if validator is not None and not validator(value):
            raise ConfigError(f"{where}.{key}: некорректное значение {value!r}")
    unknown = sorted(set(section) - set(schema))
    if unknown:
        raise ConfigError(f"{where}: неизвестные ключи {unknown}")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"конфиг не найден: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: некорректный YAML: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: корень конфига должен быть словарём")
    _check(data, _FULL_SCHEMA, "config")

    # env-переопределения LLM имеют приоритет над файлом
    endpoint = os.environ.get("MIDAS_LLM_ENDPOINT") or os.environ.get("PORTAGENT_LLM_ENDPOINT")
    model = os.environ.get("MIDAS_LLM_MODEL") or os.environ.get("PORTAGENT_LLM_MODEL")
    if endpoint:
        data["llm"]["endpoint"] = endpoint
    if model:
        data["llm"]["model"] = model
    return data


def project_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Path]:
    """Абсолютные пути данных/отчётов — относительно каталога конфига (корень проекта)."""
    root = Path(config_path).resolve().parent.parent
    return {
        "root": root,
        "data": root / config["paths"]["data_dir"],
        "reports": root / config["paths"]["reports_dir"],
    }
