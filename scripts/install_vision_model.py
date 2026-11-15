"""Установка отладочного vision-рантайма: llama.cpp (Windows) + Qwen2.5-VL-3B.

Кладёт в <проект>/llm_runtime_vision/:
  - llama-server.exe + ggml-DLL — копия из каталога llama.cpp (его сборка build 10754
    поддерживает --mmproj; проверяется при установке);
  - Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf + mmproj-...-F16.gguf (HuggingFace, ~2.7 ГБ).

Запуск: python scripts/install_vision_model.py [--runtime-dir ./llm_runtime]
Повторный запуск пропускает уже скачанные файлы. Запуск сервера:
  python -m midas vision start --wait
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT / "llm_runtime_vision"
DEFAULT_RUNTIME = ROOT / "llm_runtime"

MODEL_FILES = [
    ("https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/"
     "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"),
    ("https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/"
     "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"),
]


def _download(url: str, target: Path, expected_min_mb: float = 1.0,
              attempts: int = 5) -> None:
    if target.exists() and target.stat().st_size > expected_min_mb * 1e6:
        print(f"  [skip] {target.name} уже скачан")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "midas-installer"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        print(f"  [get ] {url}" + (f" (докачка с {resume_from / 1e6:.0f} МБ)"
                                  if resume_from else ""))
        try:
            with requests.get(url, stream=True, timeout=120, allow_redirects=True,
                              headers=headers) as response:
                if resume_from and response.status_code != 206:
                    resume_from = 0          # сервер не поддержал Range — качаем заново
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) + resume_from
                done = resume_from
                with tmp.open("ab" if resume_from else "wb") as stream:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        stream.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} МБ",
                                  end="", flush=True)
            print()
            if target.exists() or done > expected_min_mb * 1e6:
                tmp.replace(target)
                return
        except requests.RequestException as error:
            print(f"\n  [попытка {attempt}/{attempts}] обрыв: {error}")
    tmp.unlink(missing_ok=True)
    raise SystemExit(f"не удалось скачать {url} за {attempts} попыток")


def _install_binaries(source_runtime: Path) -> None:
    server = RUNTIME_DIR / "llama-server.exe"
    if server.exists():
        print("[skip] llama-server.exe уже установлен")
        return
    source_server = source_runtime / "llama-server.exe"
    if not source_server.is_file():
        raise SystemExit(f"llama-server не найден в {source_runtime}; укажите путь флагом "
                         f"--runtime-dir (или скачайте llama.cpp вручную)")
    help_text = subprocess.run([str(source_server), "--help"], capture_output=True,
                               text=True, timeout=30).stdout
    if "--mmproj" not in help_text:
        raise SystemExit(f"{source_server} не поддерживает --mmproj — нужен более новый "
                         f"llama.cpp (ggml-org releases)")
    print(f"[1/2] копирую бинарники llama.cpp из {source_runtime} (mmproj поддерживается)")
    copied = 0
    for path in source_runtime.iterdir():
        if path.is_file() and (path.name == "llama-server.exe" or path.suffix == ".dll"):
            shutil.copy2(path, RUNTIME_DIR / path.name)
            copied += 1
    print(f"  скопировано файлов: {copied}")


def _install_model() -> None:
    print("[2/2] Qwen2.5-VL-3B-Instruct GGUF + mmproj (HuggingFace, ~2.7 ГБ)")
    for url, name in MODEL_FILES:
        _download(url, RUNTIME_DIR / name, expected_min_mb=50.0)


def main() -> int:
    source_runtime = DEFAULT_RUNTIME
    if "--runtime-dir" in sys.argv:
        source_runtime = Path(sys.argv[sys.argv.index("--runtime-dir") + 1])
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _install_binaries(source_runtime)
    _install_model()
    print(json.dumps({"runtime_dir": str(RUNTIME_DIR),
                      "files": sorted(p.name for p in RUNTIME_DIR.iterdir())},
                     ensure_ascii=False, indent=2))
    print("Готово. Запуск: .venv\\Scripts\\python.exe -m midas vision start --wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
