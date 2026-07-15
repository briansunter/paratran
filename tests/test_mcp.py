import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import paratran.mcp_server as mcp_server
from paratran.contracts import TranscriptionResult


def test_mcp_tool_requires_absolute_path_and_respects_allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "audio"
    root.mkdir()
    audio = root / "sample.wav"
    audio.write_bytes(b"audio")

    fake_module = types.ModuleType("paratran.transcribe")
    fake_module.transcribe_file = lambda path, options: TranscriptionResult(
        text=Path(path).name,
        duration=1.0,
        processing_time=0.1,
        sentences=(),
    )
    monkeypatch.setitem(sys.modules, "paratran.transcribe", fake_module)

    mcp = mcp_server.create_mcp(allowed_root=str(root))
    tool = mcp._tool_manager._tools["transcribe"].fn

    assert '"text": "sample.wav"' in tool(str(audio))
    with pytest.raises(ValueError, match="absolute"):
        tool("sample.wav")
    with pytest.raises(ValueError, match="allowed root"):
        tool(str(tmp_path / "outside.wav"))


def test_streamable_http_mcp_requires_bearer_token(tmp_path):
    mcp = mcp_server.create_mcp(
        host="127.0.0.1",
        port=8123,
        allowed_root=str(tmp_path),
        api_key="secret",
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Host": "127.0.0.1:8123",
    }
    initialize = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize",'
        '"params":{"protocolVersion":"2025-03-26","capabilities":{},'
        '"clientInfo":{"name":"test","version":"1"}}}'
    )

    with TestClient(mcp.streamable_http_app()) as client:
        assert client.post("/mcp", headers=headers, content=initialize).status_code == 401
        response = client.post(
            "/mcp",
            headers={**headers, "Authorization": "Bearer secret"},
            content=initialize,
        )

    assert response.status_code == 200
    assert "serverInfo" in response.text
