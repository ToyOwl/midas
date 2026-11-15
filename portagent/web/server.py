"""Loopback-only HTTP-панель, защищённая от DNS rebinding и межсайтовых команд.

VPN пока не подключён: разрешены только локальные Host/Origin. Сессия выдаётся
локальному браузеру; все изменения требуют CSRF. Произвольных shell-команд нет.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import secrets
import subprocess
import sys
import threading
import time
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ..config import load_config, project_paths
from ..llm.runtime import launch_lock, process_alive
from ..scheduler import _write_json
from .jobs import load_report
from .store import Store, now

STATIC = Path(__file__).with_name("static")


def ui_version():
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(STATIC.iterdir()) if p.suffix in {".html", ".js", ".css"})).hexdigest()[:12]


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


class Application:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        self.paths = project_paths(self.config, self.config_path)
        self.store = Store(self.paths["data"] / "runtime" / "web")
        self.stop = threading.Event()
        self.sessions: dict[str, tuple[str, float]] = {}
        self.session_lock = threading.Lock()
        self.models = {"text": {"state": "checking"}, "vision": {"state": "checking"}}
        self.threads = []

    def start(self):
        for target, name in ((self.dispatch, "job-dispatcher"), (self.monitor, "model-monitor"),
                             (self.monitor_observations, "price-observations")):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()
            self.threads.append(thread)

    def monitor_observations(self):
        from ..tracker.observations import view
        while not self.stop.is_set():
            if (self.store.root / "observations.sqlite3").is_file():
                try:
                    view(self.paths, self.config)
                except Exception as error:
                    _write_json(self.store.root / "observations_error.json", {"at": now(), "error": str(error)})
            self.stop.wait(30)

    def monitor(self):
        from ..llm.models import manager_for
        while not self.stop.is_set():
            models = {}
            for kind in ("text", "vision"):
                try:
                    manager = manager_for(self.config, self.config_path, kind)
                    result = manager.status()
                    models[kind] = {k: result.get(k) for k in ("state", "model", "endpoint", "error")}
                    models[kind].update(model=manager.model, endpoint=manager.endpoint,
                                        installed=bool(manager.binary and manager.binary.is_file()
                                                       and manager.weights and manager.weights.is_file()))
                except Exception as error:
                    models[kind] = {"state": "failed", "error": str(error)}
            self.models = models
            self.stop.wait(12)

    def dispatch(self):
        while not self.stop.is_set():
            try:
                jobs = self.store.list(200)
                running = [j for j in jobs if j["status"] == "running"]
                for job in running:
                    if not process_alive(job["pid"], Path(sys.executable)):
                        self.store.update(job["id"], status="interrupted", finished=now(),
                                          error="Процесс задания прерван. Можно создать новый запуск.")
                if any(process_alive(j["pid"], Path(sys.executable)) for j in running):
                    self.stop.wait(1)
                    continue
                queued = [j for j in reversed(jobs) if j["status"] == "queued"]
                if queued:
                    job = queued[0]
                    log = self.store.root / f"{job['id']}.log"
                    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
                    with log.open("ab") as stream:
                        process = subprocess.Popen([sys.executable, "-u", "-m", "portagent.web.jobs",
                            "--config", str(self.config_path), "--job", job["id"]],
                            cwd=str(Path(__file__).resolve().parents[2]), env=env,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        began = time.monotonic()
                        while process.poll() is None and not self.stop.wait(0.5):
                            if time.monotonic() - began > 4 * 3600:
                                process.kill()
                        if self.stop.is_set():
                            return  # процесс задания завершает запись самостоятельно
                    current = self.store.get(job["id"])
                    if current["status"] in {"queued", "running"}:
                        self.store.update(job["id"], status="failed", finished=now(),
                                          error=f"Процесс завершился без результата (код {process.returncode})")
            except Exception as error:
                _write_json(self.store.root / "dispatcher_error.json", {"at": now(), "error": str(error)})
            self.stop.wait(0.5)

    def dates(self):
        return sorted([p.parent.name for p in self.paths["reports"].glob("????-??-??/portfolio.json")], reverse=True)

    def job(self, job, detail=False):
        value = {k: v for k, v in job.items() if detail or k != "result"}
        value["progress"] = read_json(self.store.root / f"{job['id']}.progress.json", {})
        if detail:
            try:
                with (self.store.root / f"{job['id']}.log").open("rb") as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 8000))
                    value["log"] = stream.read().decode("utf-8", errors="replace")
            except OSError:
                value["log"] = ""
        return value

    def state(self):
        dates = self.dates()
        revisions = {}
        for day in dates:
            pointer = read_json(self.paths["reports"] / day / "analysis_updates" / "latest.json", {})
            revisions[day] = pointer.get("file", "")
        return {"now": now(), "dates": dates, "models": self.models, "ui_version": ui_version(),
                "analysis_revisions": revisions,
                "jobs": [self.job(j) for j in self.store.list()],
                "scheduler": read_json(self.paths["data"] / "runtime" / "scheduler" / "state.json", {}),
                "backtest_defaults": self.config["backtest"],
                "llm_policy": {"local_only": True, "temperature": 0, "seed": self.config["llm"]["seed"],
                               "agreement_enabled": self.config["llm"]["agreement"]["enabled"]},
                "access": "local_only"}


class Handler(BaseHTTPRequestHandler):
    server_version = "Midas"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        pass  # вопросы, cookie и токены не попадают в HTTP-журнал

    def send(self, value, status=200, *, mime="application/json; charset=utf-8", cookie=None, attachment=None):
        data = (json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
                if mime.startswith("application/json") else value if isinstance(value, bytes) else value.encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def trusted(self):
        host = self.headers.get("Host", "")
        port = self.server.server_port
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return False
        origin = self.headers.get("Origin")
        return (origin is None or origin == f"http://{host}") and self.headers.get("Sec-Fetch-Site") != "cross-site"

    def session(self):
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            sid = cookies["pa_session"].value
            with self.app.session_lock:
                token, expiry = self.app.sessions.get(sid, (None, 0))
            return token if expiry > time.time() else None
        except (KeyError, ValueError, CookieError):
            return None

    def do_GET(self):
        if not self.trusted():
            return self.send({"error": "доступ разрешён только с локального адреса панели"}, 403)
        url = urlsplit(self.path)
        path = url.path
        if path == "/api/session":
            current = self.session()
            if current:
                return self.send({"csrf": current})
            sid, token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            with self.app.session_lock:
                self.app.sessions = {k: v for k, v in self.app.sessions.items() if v[1] > time.time()}
                self.app.sessions[sid] = (token, time.time() + 12 * 3600)
            return self.send({"csrf": token}, cookie=f"pa_session={sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200")
        static = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/charts.js": ("charts.js", "text/javascript; charset=utf-8"),
                  "/research.js": ("research.js", "text/javascript; charset=utf-8"),
                  "/briefing.js": ("briefing.js", "text/javascript; charset=utf-8"),
                  "/observations.js": ("observations.js", "text/javascript; charset=utf-8"),
                  "/materials.js": ("materials.js", "text/javascript; charset=utf-8"),
                  "/app.css": ("app.css", "text/css; charset=utf-8"),
                  "/icon.svg": ("icon.svg", "image/svg+xml"),
                  "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json")}
        if path in static:
            file, mime = static[path]
            content = (STATIC / file).read_bytes()
            if file == "index.html":
                content = content.replace(b"__UI_VERSION__", ui_version().encode())
            return self.send(content, mime=mime)
        if not self.session():
            return self.send({"error": "сессия истекла; обновите страницу"}, 401)
        try:
            if path == "/api/observations":
                from ..tracker.observations import view
                return self.send(view(self.app.paths, self.app.config))
            if path == "/api/materials":
                from ..materials.storage import list_materials, universe
                from ..materials.portfolio import active_plan
                root = self.app.paths["data"] / "materials"
                return self.send({"items": list_materials(root), "active": active_plan(root),
                                  "assets": universe(self.app.config), "settings": self.app.config["materials"]})
            if path.startswith("/api/materials/"):
                from ..materials.storage import valid_id, read
                parts = path.strip("/").split("/")
                if len(parts) != 4 or parts[3] not in {"image", "embedding"}:
                    raise ValueError("Неизвестный адрес материала")
                item = valid_id(parts[2])
                root = self.app.paths["data"] / "materials"
                if parts[3] == "image":
                    return self.send((root / "images" / (item + ".png")).read_bytes(), mime="image/png")
                return self.send(read(root, "analyses", item)["embedding"], attachment="embedding.json")
            if path == "/api/state":
                return self.send(self.app.state())
            if path == "/api/report":
                day = parse_qs(url.query).get("date", [""])[0]
                report = load_report(self.app.paths, day)
                from ..llm.assistant import snapshot
                from ..llm.critic import build_evidence
                return self.send({"report": report, "evidence": build_evidence(snapshot(report))})
            if path == "/api/market":
                day = parse_qs(url.query).get("date", [""])[0]
                report = load_report(self.app.paths, day)
                from ..report.diagnostics import get_diagnostics
                return self.send(get_diagnostics(self.app.paths, report, self.app.config))
            if path == "/api/research":
                day = parse_qs(url.query).get("date", [""])[0]
                report = load_report(self.app.paths, day)
                from .research import view
                return self.send(view(self.app.paths, report, self.app.config, self.app.store))
            if path.startswith("/api/jobs/"):
                parts = path.strip("/").split("/")
                job = self.app.store.get(parts[2])
                if job is None:
                    return self.send({"error": "задание не найдено"}, 404)
                if len(parts) == 4 and parts[3] == "report":
                    result = job.get("result") or {}
                    target = Path(result.get("paths", {}).get("markdown", ""))
                    if not target.is_file() or not target.resolve().is_relative_to(self.app.paths["reports"].resolve()):
                        raise ValueError("файл отчёта недоступен")
                    return self.send(target.read_bytes(), mime="text/markdown; charset=utf-8", attachment="backtest.md")
                return self.send(self.app.job(job, detail=True))
            self.send({"error": "не найдено"}, 404)
        except (ValueError, OSError, KeyError) as error:
            self.send({"error": str(error)}, 400)

    def do_POST(self):
        token = self.session()
        if (not self.trusted() or not token
                or not secrets.compare_digest(token, self.headers.get("X-CSRF-Token", ""))
                or self.headers.get("Origin") != f"http://{self.headers.get('Host')}"):
            return self.send({"error": "запрос не прошёл проверку сессии"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = (self.app.config["materials"]["max_upload_mb"] * 1024 * 1024 * 4 // 3 + 16384
                     if self.path == "/api/materials/upload" else 8192)
            if not 0 < length <= limit or self.headers.get_content_type() != "application/json":
                raise ValueError("Неверный формат или слишком большой запрос")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("ожидается JSON-объект")
            if self.path in {"/api/observations", "/api/observations/stop"}:
                from ..tracker.observations import Observations, price_series, today
                store = Observations(self.app.store.root)
                if self.path.endswith("/stop"):
                    if set(data) != {"id"}:
                        raise ValueError("Нужен идентификатор наблюдения")
                    return self.send(store.stop(data["id"]))
                end = today()
                return self.send(store.create(data, price_series(self.app.paths, self.app.config, end), end), 201)
            if self.path == "/api/materials/upload":
                from ..materials.storage import upload
                record = upload(self.app.paths["data"] / "materials", data, self.app.config)
                return self.send(record, 201)
            if self.path == "/api/materials/activate":
                from ..materials.portfolio import activate
                if set(data) != {"scenario"}:
                    raise ValueError("Нужен идентификатор готового сравнения")
                return self.send(activate(self.app.paths["data"] / "materials", data["scenario"], self.app.config["materials"]["valid_days"]))
            if self.path == "/api/materials/deactivate":
                from ..materials.portfolio import deactivate
                if data:
                    raise ValueError("Параметры не нужны")
                deactivate(self.app.paths["data"] / "materials")
                return self.send({"disabled": True})
            if self.path == "/api/jobs":
                if set(data) != {"kind", "args"}:
                    raise ValueError("ожидаются kind и args")
                job = self.app.store.submit(data["kind"], data["args"], self.headers.get("Idempotency-Key", ""))
                return self.send(self.app.job(job), 202)
            if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path.split("/")[3]
                if not self.app.store.cancel(job_id):
                    return self.send({"error": "отменить можно только ожидающее задание"}, 409)
                return self.send({"status": "cancelled"})
            self.send({"error": "действие не найдено"}, 404)
        except (ValueError, KeyError, TypeError, FileNotFoundError, BlockingIOError) as error:
            self.send({"error": str(error)}, 400)


def serve(config_path, port=8765):
    app = Application(config_path)
    with launch_lock(app.store.root / "server.lock"):
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        server.app = app
        app.start()
        _write_json(app.store.root / "server.json", {"pid": os.getpid(), "url": f"http://127.0.0.1:{port}", "started": now()})
        print(f"Midas: http://127.0.0.1:{port}", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            app.stop.set()
            server.server_close()
