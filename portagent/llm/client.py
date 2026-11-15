"""Явный адаптер локального llama-server.

Жёсткие правила: только HTTP на loopback, temperature 0, вывод ограничен
JSON-схемой; ответ сопровождается метаданными (latency, usage, SHA) для
аудита. Никаких облачных эндпоинтов by design.
"""
from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests


class LocalLLMError(RuntimeError):
    pass


def api_url(endpoint: str, resource: str) -> str:
    url = urlsplit(endpoint)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise LocalLLMError("локальная LLM требует HTTP endpoint на loopback")
    if url.username or url.password or url.query or url.fragment or url.path.rstrip("/") not in {"", "/v1"}:
        raise LocalLLMError("endpoint должен иметь вид http://127.0.0.1:8080[/v1]")
    base = endpoint.rstrip("/")
    return f"{base if base.endswith('/v1') else base + '/v1'}/{resource}"


def probe(endpoint: str, model: str) -> dict[str, Any]:
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(api_url(endpoint, "models"), timeout=3, allow_redirects=False)
        if 300 <= response.status_code < 400:
            raise LocalLLMError("перенаправления локальной LLM запрещены")
        response.raise_for_status()
        models = [row["id"] for row in response.json()["data"]]
        if model not in models:
            raise LocalLLMError(f"модель {model!r} не загружена; доступны: {models}")
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        raise LocalLLMError(f"локальная LLM недоступна: {error}") from error
    return {"state": "ready", "endpoint": endpoint, "model": model, "models": models}


def strip_reasoning(text: str) -> str:
    """Срез <think>-блоков (Qwen-стиль рассуждений) и markdown-ограждений."""
    clean = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", clean)


@contextmanager
def inference_lock(endpoint: str, timeout: float):
    # Общий для CLI, расписания и панели, включая изолированные demo-конфиги.
    from .runtime import launch_lock
    api_url(endpoint, "models")
    port = urlsplit(endpoint).port or 80
    path = Path(__file__).resolve().parents[2] / "data" / "runtime" / f"inference_{port}.lock"
    deadline = time.monotonic() + timeout
    while True:
        lock = launch_lock(path)
        try:
            lock.__enter__()
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise LocalLLMError("локальная модель занята другим заданием")
            time.sleep(0.2)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def local_post(endpoint: str, payload: dict, timeout: float):
    with inference_lock(endpoint, timeout), requests.Session() as session:
        session.trust_env = False
        return session.post(api_url(endpoint, "chat/completions"), json=payload,
                            timeout=timeout, allow_redirects=False)


def chat(endpoint: str, model: str, messages: list[dict[str, str]], *,
         timeout: int = 300, max_tokens: int = 3072,
         schema: dict | None = None, seed: int | None = None) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    payload = {
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "response_format": {"type": "json_object"},
    }
    if schema is not None:
        payload["response_format"]["schema"] = schema
    if seed is not None:
        payload["seed"] = seed
    try:
        response = local_post(endpoint, payload, timeout)
        if response.status_code != 200:
            raise LocalLLMError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            raise LocalLLMError("ответ LLM обрезан: увеличьте max_tokens")
        if not isinstance(content, str) or not content.strip():
            raise LocalLLMError("LLM вернула пустой content")
    except requests.RequestException as error:
        raise LocalLLMError(f"endpoint {endpoint} недоступен: {error}") from error
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LocalLLMError(f"LLM вернула неверный формат ответа: {error}") from error
    return content, {
        "model": model, "endpoint": endpoint, "temperature": 0,
        "seed": seed, "schema_constrained": schema is not None,
        "latency_s": round(time.monotonic() - started, 2), "usage": data.get("usage"),
        "response_sha": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
    }
