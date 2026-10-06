from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEB_ROOT = ROOT / "demo"
SAMPLE_ROOT = ROOT / "data" / "synthetic" / "v0.1" / "images" / "test"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_benchmark.adapters.paddleocr_vl_adapter import PaddleOCRVLAdapter
from ocr_benchmark.adapters.tesseract_adapter import TesseractAdapter


ENGINE_INFO = {
    "tesseract": {
        "label": "Tesseract (fast)",
        "description": "Quick Bengali and English CPU OCR for live demonstrations.",
    },
    "paddleocr-vl": {
        "label": "PaddleOCR-VL 1.5 (structured)",
        "description": "Higher-quality document parsing; expect several minutes per page on CPU.",
    },
}


def sample_catalog() -> list[dict[str, str]]:
    labels = {
        "bn_policy_01_clean.png": "Bangla policy — clean",
        "bn_policy_01_scan.jpg": "Bangla policy — scanned",
        "en_policy_01_clean.png": "English policy — clean",
        "en_policy_01_scan.jpg": "English policy — scanned",
        "mixed_policy_01_clean.png": "Mixed Bangla/English — clean",
        "mixed_policy_01_scan.jpg": "Mixed Bangla/English — scanned",
    }
    return [
        {"id": name, "label": label, "url": f"/samples/{name}"}
        for name, label in labels.items()
        if (SAMPLE_ROOT / name).is_file()
    ]


def decode_upload(payload: dict[str, Any]) -> tuple[bytes, str]:
    filename = Path(str(payload.get("filename", "upload.png"))).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Use a PNG, JPEG, WebP, TIFF, or BMP image.")
    encoded = payload.get("content_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("No image data was received.")
    if encoded.startswith("data:"):
        encoded = encoded.partition(",")[2]
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The uploaded image data is invalid.") from exc
    if not content:
        raise ValueError("The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The image exceeds the 15 MB local-demo limit.")
    return content, suffix


class OCRDemoServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]):
        super().__init__(address, handler)
        self.adapters: dict[str, Any] = {}
        self.adapter_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def get_adapter(self, engine: str) -> Any:
        factories: dict[str, Callable[[], Any]] = {
            "tesseract": lambda: TesseractAdapter(
                tessdata_directory=ROOT / "models" / "tesseract"
            ),
            "paddleocr-vl": lambda: PaddleOCRVLAdapter(
                cache_directory=ROOT / "models" / "paddlex"
            ),
        }
        if engine not in factories:
            raise ValueError("Unknown OCR engine.")
        with self.adapter_lock:
            if engine not in self.adapters:
                self.adapters[engine] = factories[engine]()
            return self.adapters[engine]


class DemoHandler(BaseHTTPRequestHandler):
    server: OCRDemoServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/config":
            self._json({"engines": ENGINE_INFO, "samples": sample_catalog()})
            return
        if path.startswith("/samples/"):
            name = Path(path.removeprefix("/samples/")).name
            self._file(SAMPLE_ROOT / name)
            return
        static_files = {"/": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}
        if path in static_files:
            self._file(WEB_ROOT / static_files[path])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/ocr":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
                raise ValueError("The request is empty or too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            engine = str(payload.get("engine", "tesseract"))
            sample_id = payload.get("sample_id")
            if sample_id:
                name = Path(str(sample_id)).name
                image_path = SAMPLE_ROOT / name
                if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_SUFFIXES:
                    raise ValueError("Unknown sample image.")
                result = self._recognize(engine, image_path)
            else:
                content, suffix = decode_upload(payload)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as upload:
                    upload.write(content)
                    image_path = Path(upload.name)
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    result = self._recognize(engine, image_path)
                    result["image_path"] = Path(str(payload.get("filename", "upload"))).name
                finally:
                    image_path.unlink(missing_ok=True)
            self._json({"result": result})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, UnidentifiedImageError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"OCR error: {type(exc).__name__}: {exc}")
            self._json(
                {"error": f"OCR failed: {type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _recognize(self, engine: str, image_path: Path) -> dict[str, Any]:
        adapter = self.server.get_adapter(engine)
        with self.server.inference_lock:
            return adapter.recognize(image_path).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local OCR client demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    server = OCRDemoServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Local OCR demo: {url}")
    print("Press Ctrl+C to stop. Uploaded files are deleted after inference.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local OCR demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
