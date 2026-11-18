"""TEST INFRASTRUCTURE ONLY: In-memory Mock GPU Worker HTTP Server.

DO NOT USE IN PRODUCTION.
This module provides a lightweight, pure-Python HTTP mock worker using the standard
library http.server to test the client, remote backend, error handling, protocol parsing,
and Streamlit UI integration without requiring an external Google Colab GPU instance.
"""
from __future__ import annotations

import base64
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

from .protocol import (
    DetectionItem,
    InferenceRequest,
    InferenceResponse,
    WorkerHealth,
)


class _WorkerHTTPServer(HTTPServer):
    """Subclass of HTTPServer that holds a reference to the MockGPUWorkerServer."""

    def __init__(self, server_address: Any, RequestHandlerClass: Any, worker: "MockGPUWorkerServer") -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.worker = worker


class _MockHandler(BaseHTTPRequestHandler):
    """Internal HTTP request handler for the mock GPU worker."""

    server: _WorkerHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server request logging during tests."""
        pass

    @property
    def worker(self) -> "MockGPUWorkerServer":
        return self.server.worker

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def do_POST(self) -> None:
        if self.path == "/detect":
            self._handle_detect()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def _handle_health(self) -> None:
        if self.worker.should_timeout:
            time.sleep(2.0)
            return

        if self.worker.should_error_500:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "error", "message": "Simulated 500 error"}')
            return

        if self.worker.should_malform_json:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"invalid_json": true, incomplete')
            return

        health = WorkerHealth(
            status="ok",
            cuda_available=self.worker.cuda_available,
            gpu_name=self.worker.gpu_name,
            gpu_memory_total_mb=15109.0,
            gpu_memory_allocated_mb=128.5,
            yolo_loaded=self.worker.yolo_loaded,
            model_name="yolov8n.pt",
            torch_version="2.1.0+cu121 (mock)",
            device="cuda" if self.worker.cuda_available else "cpu",
        )

        body = json.dumps(health.to_dict()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_detect(self) -> None:
        if self.worker.should_timeout:
            time.sleep(2.0)
            return

        if self.worker.should_error_500:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "error", "error": "Internal worker fault"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        if self.worker.should_malform_json:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detections": "not-a-list"')
            return

        try:
            req_dict = json.loads(post_data.decode("utf-8"))
            req = InferenceRequest.from_dict(req_dict)
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid request body: {e}"}).encode("utf-8"))
            return

        dets = list(self.worker.mock_detections)
        resp = InferenceResponse(
            status="ok",
            detections=dets,
            inference_time_ms=self.worker.latency_ms,
            device="cuda" if self.worker.cuda_available else "cpu",
            gpu_name=self.worker.gpu_name,
            frame_index=req.frame_index,
        )

        body = json.dumps(resp.to_dict()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockGPUWorkerServer:
    """Mock HTTP server simulating a Google Colab GPU worker for testing."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        cuda_available: bool = True,
        gpu_name: str = "NVIDIA T4 (Mock Test Worker)",
        yolo_loaded: bool = True,
        latency_ms: float = 14.5,
    ) -> None:
        self.host = host
        self.port = port
        self.cuda_available = cuda_available
        self.gpu_name = gpu_name
        self.yolo_loaded = yolo_loaded
        self.latency_ms = latency_ms

        self.should_timeout: bool = False
        self.should_error_500: bool = False
        self.should_malform_json: bool = False

        # Default mock detections: 1 car and 1 pedestrian
        self.mock_detections: List[DetectionItem] = [
            DetectionItem(x1=50.0, y1=100.0, x2=150.0, y2=250.0, confidence=0.88, class_id=2, class_name="car"),
            DetectionItem(x1=400.0, y1=300.0, x2=450.0, y2=420.0, confidence=0.79, class_id=0, class_name="person"),
        ]

        self._server: Optional[_WorkerHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._url: Optional[str] = None

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("Mock server is not running. Call start() first.")
        return self._url

    @property
    def base_url(self) -> str:
        return self.url

    def start(self) -> str:
        """Start the mock HTTP worker on a background thread."""
        server_address = (self.host, self.port)
        self._server = _WorkerHTTPServer(server_address, _MockHandler, self)
        actual_port = self._server.server_address[1]
        self._url = f"http://{self.host}:{actual_port}"

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._url

    def shutdown(self) -> None:
        """Stop the mock server and terminate the background thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._url = None

    def __enter__(self) -> "MockGPUWorkerServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()
