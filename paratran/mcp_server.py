"""MCP adapter for Paratran transcription."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

from paratran.contracts import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_DECODING,
    DEFAULT_DURATION_REWARD,
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP_DURATION,
    DEFAULT_PATIENCE,
    TranscriptionOptions,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("paratran-mcp")


class _StaticTokenVerifier:
    """Verify a configured API key presented as an MCP bearer token."""

    def __init__(self, expected_token: str):
        self._expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self._expected_token):
            return None
        return AccessToken(
            token=token,
            client_id="paratran-static-api-key",
            scopes=["transcribe"],
        )


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _server_url(host: str, port: int) -> str:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{display_host}:{port}"


def create_mcp(
    host: str = "127.0.0.1",
    port: int = 8000,
    allowed_root: str | None = None,
    api_key: str | None = None,
) -> FastMCP:
    root = Path(allowed_root).expanduser().resolve() if allowed_root else None
    if root is not None and not root.is_dir():
        raise ValueError(f"Allowed root is not a directory: {root}")
    if api_key == "":
        raise ValueError("api_key must not be empty")

    auth_kwargs = {}
    if api_key:
        # FastMCP requires an issuer URL whenever its bearer-token verifier is
        # enabled. Static API-key auth does not expose an OAuth issuer, so keep
        # metadata disabled while still using FastMCP's standards-compliant
        # Authorization: Bearer enforcement.
        auth_kwargs = {
            "token_verifier": _StaticTokenVerifier(api_key),
            "auth": AuthSettings(
                issuer_url=_server_url(host, port),
                resource_server_url=None,
                required_scopes=["transcribe"],
            ),
        }

    mcp = FastMCP("paratran", host=host, port=port, **auth_kwargs)

    @mcp.tool()
    def transcribe(
        file_path: str,
        decoding: str = DEFAULT_DECODING,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        patience: float = DEFAULT_PATIENCE,
        duration_reward: float = DEFAULT_DURATION_REWARD,
        max_words: int | None = None,
        silence_gap: float | None = None,
        max_duration: float | None = None,
        chunk_duration: float | None = DEFAULT_CHUNK_DURATION,
        overlap_duration: float = DEFAULT_OVERLAP_DURATION,
        fp32: bool = False,
    ) -> str:
        """Transcribe an audio file to JSON with aligned word timestamps.

        Args:
            file_path: Absolute path to an audio file. If --allowed-root is set,
                the path must be inside that directory.
            decoding: Decoding method - 'greedy' or 'beam'.
            beam_size: Beam size (beam decoding only).
            length_penalty: Length penalty (beam decoding only).
            patience: Patience (beam decoding only).
            duration_reward: Duration reward from 0.0 to 1.0 (beam decoding only).
            max_words: Max words per sentence.
            silence_gap: Split sentence on silence gap (seconds).
            max_duration: Max sentence duration (seconds).
            chunk_duration: Chunk duration in seconds; 0 disables chunking.
            overlap_duration: Overlap between chunks (seconds).
            fp32: Use float32 instead of bfloat16.
        """

        path = Path(file_path).expanduser()
        if not path.is_absolute():
            raise ValueError("file_path must be an absolute path")
        resolved_path = path.resolve()
        if root is not None and not resolved_path.is_relative_to(root):
            raise ValueError(f"file_path must be inside the allowed root: {root}")

        options = TranscriptionOptions(
            decoding=decoding,
            beam_size=beam_size,
            length_penalty=length_penalty,
            patience=patience,
            duration_reward=duration_reward,
            max_words=max_words,
            silence_gap=silence_gap,
            max_duration=max_duration,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            fp32=fp32,
        )
        from paratran.transcribe import transcribe_file

        logger.info("Transcribing: %s", resolved_path)
        result = transcribe_file(str(resolved_path), options=options)
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="paratran-mcp",
        description="Paratran MCP server for audio transcription.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for HTTP (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port for HTTP (default: 8000)")
    parser.add_argument(
        "--model",
        default=os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL),
        help=f"HF model ID or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("PARATRAN_MODEL_DIR"),
        help="Directory for HuggingFace model cache",
    )
    parser.add_argument(
        "--allowed-root",
        default=os.environ.get("PARATRAN_ALLOWED_ROOT"),
        help="Restrict MCP file access to this directory",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PARATRAN_API_KEY"),
        help="Require this value as an Authorization bearer token for HTTP",
    )
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        parser.error("port must be between 1 and 65535")
    if args.transport == "streamable-http" and not _is_loopback(args.host):
        if not args.allowed_root:
            parser.error("--allowed-root is required for non-loopback HTTP MCP servers")
        if not args.api_key:
            parser.error("--api-key is required for non-loopback HTTP MCP servers")

    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir
    if args.allowed_root:
        os.environ["PARATRAN_ALLOWED_ROOT"] = args.allowed_root
    if args.api_key:
        os.environ["PARATRAN_API_KEY"] = args.api_key

    mcp = create_mcp(
        host=args.host,
        port=args.port,
        allowed_root=args.allowed_root,
        api_key=args.api_key if args.transport == "streamable-http" else None,
    )
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
