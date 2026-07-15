from pathlib import Path

from fastapi.testclient import TestClient

import paratran.server as server
from paratran.contracts import TranscriptionResult


def fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="ok",
        duration=2.0,
        processing_time=0.1,
        sentences=(),
    )


def run_client(monkeypatch, fake_transcribe):
    monkeypatch.setattr(server, "_load_model", lambda: None)
    monkeypatch.setattr(server, "_model_status", lambda: {"model": "test", "model_dir": None})
    monkeypatch.setattr(server, "_transcribe_file", fake_transcribe)
    return TestClient(server.app)


def test_transcription_route_uses_shared_defaults_and_response_contract(monkeypatch):
    calls = []

    def fake_transcribe(path: str, options):
        calls.append((Path(path).exists(), options))
        return fake_result()

    with run_client(monkeypatch, fake_transcribe) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"response_format": "verbose_json"},
        )

    assert response.status_code == 200
    assert response.json()["processing_time"] == 0.1
    assert calls[0][0] is True
    assert calls[0][1].chunk_duration == 120.0
    assert calls[0][1].length_penalty == 0.013


def test_invalid_numeric_parameters_are_rejected(monkeypatch):
    with run_client(monkeypatch, lambda *_args: fake_result()) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"beam_size": "0", "duration_reward": "9"},
        )

    assert response.status_code == 422


def test_api_key_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("PARATRAN_API_KEY", "secret")
    with run_client(monkeypatch, lambda *_args: fake_result()) as client:
        request = {"files": {"file": ("sample.wav", b"audio", "audio/wav")}}
        assert client.post("/v1/audio/transcriptions", **request).status_code == 401
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer secret"},
            **request,
        )

    assert response.status_code == 200


def test_upload_limit_is_enforced(monkeypatch):
    monkeypatch.setenv("PARATRAN_MAX_UPLOAD_MB", "1")
    with run_client(monkeypatch, lambda *_args: fake_result()) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"x" * (1024 * 1024 + 1), "audio/wav")},
        )

    assert response.status_code == 413
