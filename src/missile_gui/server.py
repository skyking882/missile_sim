"""Loopback-only HTTP server for the local missile trajectory GUI."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aim120_model.public_api import (
    SimulationInputError,
    UnsupportedPhysicsError,
    simulate,
)
from missile_gui.library import public_profile, scan_library


LOGGER = logging.getLogger("missile_gui")
PROJECT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MISSILES_DIR = PROJECT_DIR / "missiles"
MAX_REQUEST_BYTES = 1_000_000


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "MissileGUI/1.0.1"

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"ok": False, "error": {"code": code, "message": message}})

    def _profiles(self) -> tuple[list[dict[str, Any]], list[str]]:
        return scan_library(MISSILES_DIR, PROJECT_DIR)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/missiles":
                profiles, errors = self._profiles()
                self._json(HTTPStatus.OK, {
                    "ok": True,
                    "missiles": [public_profile(profile) for profile in profiles],
                    "library_errors": errors,
                })
                return
            if path in ("/", "/index.html"):
                self._static("index.html")
                return
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                if "/" in name or "\\" in name or name.startswith("."):
                    self._error(HTTPStatus.NOT_FOUND, "not_found", "资源不存在。")
                    return
                self._static(name)
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "页面不存在。")
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("客户端在响应完成前断开连接。")
        except Exception:
            LOGGER.exception("处理 GET %s 时发生未预期错误", path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "本地服务发生错误，请查看终端日志。")

    def _static(self, name: str) -> None:
        path = STATIC_DIR / name
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not_found", "资源不存在。")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path != "/api/simulate":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "接口不存在。")
            return
        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                self._error(HTTPStatus.LENGTH_REQUIRED, "missing_length", "请求缺少长度信息。")
                return
            length = int(raw_length)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "场景数据过大。")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "请求不是合法 JSON。")
                return
            if not isinstance(payload, dict):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "请求顶层必须是对象。")
                return
            missile_id = payload.get("missile_id")
            profiles, library_errors = self._profiles()
            profile = next((
                item for item in profiles
                if item.get("missile_id", item.get("id")) == missile_id
            ), None)
            if profile is None:
                detail = "；".join(library_errors[:3])
                message = "找不到所选导弹。" + (f" 导弹库错误：{detail}" if detail else "")
                self._error(HTTPStatus.BAD_REQUEST, "unknown_missile", message)
                return
            result = simulate(profile, payload.get("scenario"))
            self._json(HTTPStatus.OK, {"ok": True, "result": result})
        except SimulationInputError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_scenario", str(exc))
        except UnsupportedPhysicsError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "unsupported_physics", str(exc))
        except (ArithmeticError, OverflowError, FloatingPointError) as exc:
            LOGGER.exception("模拟数值失败")
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "numerical_failure", f"模拟数值失败：{exc}")
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("客户端在模拟响应完成前断开连接。")
        except Exception:
            LOGGER.exception("运行模拟时发生未预期错误")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "计算失败，请查看终端中的详细错误。")


def _open_browser_when_ready(url: str) -> None:
    health = url.rstrip("/") + "/api/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(health, timeout=0.5) as response:
                if response.status == HTTPStatus.OK:
                    webbrowser.open(url, new=2)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIM-120 local missile GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MISSILE_GUI_PORT", "8765")))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("GUI v1 只允许绑定 127.0.0.1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    url = f"http://{args.host}:{args.port}/"
    try:
        server = GuiServer((args.host, args.port), GuiHandler)
    except OSError as exc:
        print(f"无法启动 GUI：端口 {args.port} 已被占用或不可用。", flush=True)
        print("可关闭占用该端口的程序，或设置 MISSILE_GUI_PORT 后重试。", flush=True)
        traceback.print_exc()
        return 2
    LOGGER.info("本地 GUI 已启动：%s", url)
    LOGGER.info("只监听 127.0.0.1；按 Ctrl+C 关闭。")
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        LOGGER.info("收到关闭请求。")
    finally:
        server.server_close()
        LOGGER.info("GUI 已关闭，端口 %s 已释放。", args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
