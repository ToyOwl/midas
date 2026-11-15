@echo off
rem Демо локальной LLM на синтетике fixtures\demo (36 новостей, 12 активов).
rem Отчёт, кэш разметки и история пишутся в .demo\ и не затирают боевые data\ и reports\.
rem Флаг --config стоит до подкоманды: «portagent daily --config …» argparse не принимает.
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo venv не найден. Выполните: python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev,us]"
  exit /b 1
)
if not exist ".demo\configs" mkdir ".demo\configs"
if not exist ".demo\configs\demo.yaml" copy /Y "configs\default.yaml" ".demo\configs\demo.yaml" >nul
if not exist "fixtures\demo\news.json" (
  ".venv\Scripts\python.exe" scripts\make_demo_fixture.py fixtures\demo
  if errorlevel 1 exit /b 1
)
echo Поднимаю локальную LLM, если она ещё не отвечает...
".venv\Scripts\python.exe" -m midas llm start --wait
if errorlevel 1 exit /b 1
echo Демо: fixture -^> разметка -^> HRP -^> критик. Отчёт будет в .demo\reports\
".venv\Scripts\python.exe" -m midas --config .demo\configs\demo.yaml daily --source fixture --fixture-dir fixtures\demo %*
exit /b %ERRORLEVEL%
