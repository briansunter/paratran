import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError

import pytest

from paratran.cli import _upload_file


class UploadHandler(BaseHTTPRequestHandler):
    status = 200
    response = {"text": "ok"}
    body = b""

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).body = self.rfile.read(length)
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).response).encode())

    def log_message(self, *_args):
        return


@pytest.fixture
def upload_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), UploadHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_upload_streams_multipart_fields_and_file(tmp_path: Path, upload_server):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio-bytes")

    result = _upload_file(
        f"http://127.0.0.1:{upload_server.server_port}/transcribe",
        audio,
        {"response_format": "verbose_json"},
        timeout=5,
    )

    assert result == {"text": "ok"}
    assert b'name="response_format"' in UploadHandler.body
    assert b"audio-bytes" in UploadHandler.body


def test_upload_exposes_http_errors_as_http_error(tmp_path: Path, upload_server):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    UploadHandler.status = 413
    try:
        with pytest.raises(HTTPError) as error:
            _upload_file(
                f"http://127.0.0.1:{upload_server.server_port}/transcribe",
                audio,
                timeout=5,
            )
        assert error.value.code == 413
    finally:
        UploadHandler.status = 200
