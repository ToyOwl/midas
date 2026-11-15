"""Vision-запросы к локальному OpenAI-совместимому VLM (llama-server + mmproj,
LM Studio): контент-парты с data-URI изображением. Правила те же: loopback-only,
temperature 0, JSON-схема. Грандинг по картинке невозможен — валидация только
структурная (enum тренда, тикер из юниверса), это честно фиксируется в отчёте.
"""
from __future__ import annotations

import base64
import hashlib
import time
from typing import Any

import requests

from .client import LocalLLMError, local_post, strip_reasoning

MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp"}


def vision_chat(endpoint: str, model: str, image_bytes: bytes, mime: str, prompt: str, *,
                schema: dict | None = None, timeout: float = 120.0,
                max_tokens: int = 1024, seed: int = 20261112) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]}],
        "temperature": 0, "max_tokens": max_tokens, "seed": seed,
        "response_format": {"type": "json_object"},
    }
    if schema is not None:
        payload["response_format"]["schema"] = schema
    try:
        response = local_post(endpoint, payload, timeout)
        if response.status_code != 200:
            raise LocalLLMError(f"VLM HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            raise LocalLLMError("ответ VLM обрезан: увеличьте max_tokens")
        if not isinstance(content, str) or not content.strip():
            raise LocalLLMError("VLM вернула пустый content")
    except requests.RequestException as error:
        raise LocalLLMError(f"vision endpoint {endpoint} недоступен: {error}") from error
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LocalLLMError(f"VLM вернула неверный формат ответа: {error}") from error
    return strip_reasoning(content), {
        "model": model, "endpoint": endpoint, "temperature": 0, "seed": seed,
        "latency_s": round(time.monotonic() - started, 2),
        "image_sha": hashlib.sha256(image_bytes).hexdigest()[:12],
        "usage": data.get("usage"),
    }
