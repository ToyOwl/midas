"""Idempotent-лаунчер llama-server.

Бинарник и веса по умолчанию берутся из каталога llm.runtime_root,
чтобы не дублировать ~6 ГБ модели. Состояние и журнал — в data/runtime/.
"""
from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import LocalLLMError, probe

DEFAULT_PORT = 8080


def process_alive(pid, executable: Path) -> bool:
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        size = wintypes.DWORD(32768)
        name = ctypes.create_unicode_buffer(size.value)
        expected = {Path(executable).resolve()}
        if Path(executable).resolve() == Path(sys.executable).resolve():
            expected.add(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259
                    and kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size))
                    and Path(name.value).resolve() in expected)
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def launch_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError("Запуск уже заблокирован другим процессом") from exc
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _find_weights(runtime_dir: Path) -> Path | None:
    candidates = sorted(runtime_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


class RuntimeManager:
    """Управляемый запуск llama-server на loopback.

    Параметризуется для второго сервера (vision, порт 8081): можно задать
    собственный каталог рантайма, файл весов, порт и дополнительные аргументы
    запуска (например, --mmproj для VLM).
    """

    def __init__(self, state_root: Path, *, runtime_root: Path | None = None,
                 runtime_dir: Path | None = None, weights: Path | None = None,
                 endpoint: str = "http://127.0.0.1:8080", model: str = "bonsai-2-27b",
                 extra_args: list[str] | None = None):
        from urllib.parse import urlsplit

        from .client import api_url
        api_url(endpoint, "models")
        self.state_root = Path(state_root)
        if runtime_dir is None and runtime_root is not None:
            runtime_dir = Path(runtime_root) / "llm_runtime"
        self.runtime_dir = Path(runtime_dir) if runtime_dir else None
        self.binary = self.runtime_dir / "llama-server.exe" if self.runtime_dir else None
        self.weights = (Path(weights) if weights
                        else (_find_weights(self.runtime_dir) if self.runtime_dir and self.runtime_dir.is_dir() else None))
        self.endpoint = endpoint
        self.model = model
        self.port = urlsplit(endpoint).port or DEFAULT_PORT
        self.extra_args = list(extra_args or [])
        self.state = self.state_root / f"llm_runtime_{self.port}.json"
        self.log = self.state_root / f"llm_start_{self.port}.log"
        self.process: subprocess.Popen | None = None
        self.lock = threading.RLock()

    def saved(self) -> dict[str, Any]:
        try:
            saved = json.loads(self.state.read_text(encoding="utf-8")) if self.state.exists() else {}
            return saved if isinstance(saved, dict) else {}
        except (ValueError, OSError):
            return {}

    def _record(self, value: dict[str, Any]) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state)

    def _alive(self, pid) -> bool:
        return process_alive(pid, self.binary) if self.binary else False

    def status(self) -> dict[str, Any]:
        saved = self.saved()
        try:
            return {**saved, **probe(self.endpoint, self.model), "error": None,
                    "process_alive": self._alive(saved.get("pid")),
                    "weights": str(self.weights) if self.weights else None,
                    "log": str(self.log)}
        except LocalLLMError as error:
            alive = self._alive(saved.get("pid"))
            age = time.time() - saved.get("started_epoch", time.time())
            if alive:
                state = "loading" if age < 180 else "failed"
                message = ("Модель загружается; дождитесь готовности." if age < 180
                           else "Процесс запущен, но модель не стала доступна за 180 с. См. журнал.")
            else:
                state = "failed" if saved.get("pid") else "stopped"
                message = ("Процесс LLM завершился. Можно повторить запуск." if saved.get("pid")
                           else "Локальная модель остановлена.")
            return {**saved, "state": state, "process_alive": alive, "endpoint": self.endpoint,
                    "model": self.model, "error": message, "connection_error": str(error),
                    "weights": str(self.weights) if self.weights else None, "log": str(self.log)}

    def start(self) -> dict[str, Any]:
        with self.lock:
            try:
                with launch_lock(self.state.with_suffix(".lock")):
                    current = self.status()
                    if current["state"] == "ready" or current.get("process_alive"):
                        return current
                    if not self.binary or not self.binary.is_file() or not self.weights or not self.weights.is_file():
                        missing = "llama-server.exe/*.gguf"
                        raise FileNotFoundError(
                            f"нужны {missing} в {self.runtime_dir} "
                            f"(настройте llm.runtime_root или vision-рантайм в конфиге)")
                    if "--mmproj" in self.extra_args:
                        projector = Path(self.extra_args[self.extra_args.index("--mmproj") + 1])
                        if not projector.is_file():
                            raise FileNotFoundError(f"Не найден проектор изображений: {projector}")
                    with socket.socket() as connection:
                        connection.settimeout(0.4)
                        if connection.connect_ex(("127.0.0.1", self.port)) == 0:
                            raise RuntimeError(
                                f"Порт {self.port} занят другим сервером; новый процесс не запущен")
                    command = [str(self.binary), "--model", str(self.weights), "--alias", self.model,
                               "--host", "127.0.0.1", "--port", str(self.port),
                               "--ctx-size", "8192", "--parallel", "1", "--threads", "4",
                               "--n-gpu-layers", "99"]
                    if self.port == DEFAULT_PORT:
                        command += ["--reasoning-budget", "512"]  # Bonsai: разумный бюджет <think>
                    command += self.extra_args
                    self.log.parent.mkdir(parents=True, exist_ok=True)
                    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                               MKL_NUM_THREADS="1")
                    with self.log.open("ab") as stream:
                        stream.write((f"\n--- START {datetime.now(timezone.utc).isoformat()} ---\n").encode())
                        stream.flush()
                        self.process = subprocess.Popen(
                            command, cwd=str(self.runtime_dir), env=env, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=stream,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    self._record({"state": "loading", "pid": self.process.pid,
                                  "started_epoch": time.time(),
                                  "started_at": datetime.now(timezone.utc).isoformat(),
                                  "command": command, "endpoint": self.endpoint,
                                  "model": self.model})
            except BlockingIOError:
                return {"state": "loading",
                        "error": "Другой запуск уже выполняется; проверьте статус."}
        watcher = threading.Thread(target=self.wait_ready, daemon=True, name="llm-startup")
        watcher.start()
        return self.status()

    def wait_ready(self, timeout: int = 180) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.status()
            if current["state"] in {"ready", "failed", "stopped"}:
                return current
            time.sleep(2)
        return {**self.status(), "state": "failed",
                "error": "Истекло время запуска LLM; см. журнал."}

    def ensure_ready(self, timeout: int = 180) -> dict[str, Any]:
        """Один ограниченный старт настроенной модели, никогда не подменная."""
        current = self.status()
        if current["state"] == "ready":
            return current
        current = self.start()
        if current["state"] == "loading":
            current = self.wait_ready(timeout)
        if current["state"] != "ready":
            raise LocalLLMError(current.get("error") or "Локальная модель не готова")
        return current
