@echo off
rem Поднимает локальную llama-server из указанного каталога (см. llm.runtime_root в конфиге).
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo venv не найден. Выполните: python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev,us]"
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m midas llm start --wait
pause
