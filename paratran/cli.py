"""Command-line adapter for local and server-backed transcription."""

from __future__ import annotations

import argparse
import http.client
import io
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit

from paratran.contracts import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_DECODING,
    DEFAULT_DURATION_REWARD,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP_DURATION,
    DEFAULT_PATIENCE,
    OUTPUT_FORMATS,
    TranscriptionOptions,
)
from paratran.serializers import from_openai_verbose_json, write_outputs


def _add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--output-dir",
        default=".",
        help="Directory to save transcriptions (default: .)",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--decoding",
        default=DEFAULT_DECODING,
        choices=["greedy", "beam"],
        help=f"Decoding method (default: {DEFAULT_DECODING})",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Chunk duration in seconds, 0 to disable (default: {DEFAULT_CHUNK_DURATION:g})",
    )
    parser.add_argument(
        "--overlap-duration",
        type=float,
        default=DEFAULT_OVERLAP_DURATION,
        help=f"Overlap duration in seconds (default: {DEFAULT_OVERLAP_DURATION:g})",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size (default: {DEFAULT_BEAM_SIZE})",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=DEFAULT_LENGTH_PENALTY,
        help=f"Length penalty (default: {DEFAULT_LENGTH_PENALTY})",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=DEFAULT_PATIENCE,
        help=f"Patience (default: {DEFAULT_PATIENCE})",
    )
    parser.add_argument(
        "--duration-reward",
        type=float,
        default=DEFAULT_DURATION_REWARD,
        help=f"Duration reward (default: {DEFAULT_DURATION_REWARD})",
    )
    parser.add_argument("--max-words", type=int, default=None, help="Max words per sentence")
    parser.add_argument(
        "--silence-gap",
        type=float,
        default=None,
        help="Split at silence gaps (seconds)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Max sentence duration (seconds)",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision instead of BF16",
    )


def _options_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> TranscriptionOptions:
    try:
        return TranscriptionOptions(
            decoding=args.decoding,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            patience=args.patience,
            duration_reward=args.duration_reward,
            max_words=args.max_words,
            silence_gap=args.silence_gap,
            max_duration=args.max_duration,
            chunk_duration=args.chunk_duration if args.chunk_duration > 0 else None,
            overlap_duration=args.overlap_duration,
            fp32=args.fp32,
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error should have exited")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return _serve(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Parakeet MLX models.",
        usage=(
            "paratran [OPTIONS] AUDIOS...\n"
            "       paratran serve [--host HOST] [--port PORT] [--model MODEL] [--cache-dir DIR]"
        ),
    )
    parser.add_argument(
        "audios",
        nargs="*",
        metavar="AUDIOS",
        help="Audio files to transcribe",
    )
    parser.add_argument(
        "-s",
        "--server",
        default=os.environ.get("PARATRAN_SERVER"),
        help="URL of a running paratran server (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PARATRAN_API_KEY"),
        help="Bearer token for an authenticated server",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_HTTP_TIMEOUT,
        help=f"Server request timeout in seconds (default: {DEFAULT_HTTP_TIMEOUT:g})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress")
    _add_transcription_arguments(parser)
    args = parser.parse_args()

    if not args.audios:
        parser.print_help()
        return 1
    if args.timeout <= 0:
        parser.error("timeout must be greater than 0")

    options = _options_from_args(args, parser)
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"Could not create output directory '{output_dir}': {exc}")

    formats = ["txt", "srt", "vtt", "json"] if args.output_format == "all" else [args.output_format]
    if args.server:
        if args.model != DEFAULT_MODEL or args.cache_dir:
            print(
                "Warning: --model and --cache-dir configure local/server mode only; "
                "configure them on 'paratran serve'.",
                file=sys.stderr,
            )
        return _transcribe_via_server(args, options, output_dir, formats)
    return _transcribe_local(args, options, output_dir, formats)


def _transcribe_local(
    args: argparse.Namespace,
    options: TranscriptionOptions,
    output_dir: Path,
    formats: list[str],
) -> int:
    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir

    from paratran.transcribe import transcribe_file

    failures = 0
    for audio_path in args.audios:
        path = Path(audio_path)
        if not path.is_file():
            print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
            failures += 1
            continue

        if args.verbose:
            print(f"Transcribing: {path.name}", file=sys.stderr)

        try:
            result = transcribe_file(str(path), options=options)
            if args.verbose:
                print(
                    f"  Duration: {result.duration:.2f}s, "
                    f"Processing: {result.processing_time:.3f}s",
                    file=sys.stderr,
                )
            _write_output(result, path.stem, output_dir, formats, args.verbose)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Error: Could not transcribe {audio_path}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def _transcribe_via_server(
    args: argparse.Namespace,
    options: TranscriptionOptions,
    output_dir: Path,
    formats: list[str],
) -> int:
    server_url = args.server.rstrip("/")
    url = f"{server_url}/v1/audio/transcriptions"
    fields: dict[str, str] = {
        "response_format": "verbose_json",
        **{key: str(value) for key, value in options.to_dict().items() if value is not None},
    }
    if options.chunk_duration is None:
        fields["chunk_duration"] = "0"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    failures = 0
    for audio_path in args.audios:
        path = Path(audio_path)
        if not path.is_file():
            print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
            failures += 1
            continue

        if args.verbose:
            print(f"Uploading to server: {path.name}", file=sys.stderr)

        try:
            response = _upload_file(
                url,
                path,
                fields,
                headers=headers,
                timeout=args.timeout,
            )
            result = _openai_to_internal(response)
            if args.verbose:
                print(
                    f"  Duration: {result.duration:.2f}s, "
                    f"Processing: {result.processing_time:.3f}s",
                    file=sys.stderr,
                )
            _write_output(result, path.stem, output_dir, formats, args.verbose)
        except HTTPError as exc:
            body = exc.read().decode(errors="replace") if exc.fp else ""
            print(f"Error: Server returned {exc.code}: {body}", file=sys.stderr)
            failures += 1
        except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error: Could not transcribe via {server_url}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def _multipart_field(boundary: str, name: str, value: str) -> bytes:
    return (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
    ).encode()


def _upload_file(
    url: str,
    path: Path,
    fields: dict[str, str] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Stream a multipart upload without duplicating the audio in memory."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid server URL: {url}")

    file_size = path.stat().st_size
    boundary = f"----ParatranBoundary{uuid.uuid4().hex}"
    field_parts = [
        _multipart_field(boundary, name, value) for name, value in (fields or {}).items()
    ]
    safe_filename = path.name.replace("\\", "_").replace('"', "'")
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        "\r\n"
    ).encode()
    closing = f"\r\n--{boundary}--\r\n".encode()
    content_length = (
        sum(len(part) for part in field_parts) + len(file_header) + file_size + len(closing)
    )

    target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    connection_class = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_class(parsed.netloc, timeout=timeout)
    request_headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(content_length),
        **(headers or {}),
    }

    try:
        connection.putrequest("POST", target)
        for name, value in request_headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        for part in field_parts:
            connection.send(part)
        connection.send(file_header)
        with path.open("rb") as audio_file:
            while chunk := audio_file.read(1024 * 1024):
                connection.send(chunk)
        connection.send(closing)
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            response_headers = {name: value for name, value in response.getheaders()}
            raise HTTPError(
                url,
                response.status,
                response.reason,
                response_headers,
                io.BytesIO(body),
            )
        return json.loads(body)
    except HTTPError:
        raise
    except OSError as exc:
        raise URLError(exc) from exc
    finally:
        connection.close()


def _openai_to_internal(response: dict[str, Any]):
    return from_openai_verbose_json(response)


def _write_output(result, stem: str, output_dir: Path, formats: list[str], verbose: bool) -> None:
    for path in write_outputs(result, stem, output_dir, formats):
        if verbose:
            print(f"  Saved: {path}", file=sys.stderr)


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="paratran serve",
        description="Start the Paratran REST interface.",
    )
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
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PARATRAN_API_KEY"),
        help="Optional bearer token; required for non-loopback hosts",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=int(os.environ.get("PARATRAN_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)),
        help=f"Maximum upload size in MB (default: {DEFAULT_MAX_UPLOAD_MB})",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("PARATRAN_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)),
        help=f"Maximum concurrent transcriptions (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    args = parser.parse_args(argv)

    if args.port < 1 or args.port > 65535:
        parser.error("port must be between 1 and 65535")
    if args.max_upload_mb < 1:
        parser.error("max-upload-mb must be at least 1")
    if args.max_concurrency < 1:
        parser.error("max-concurrency must be at least 1")
    if not _is_loopback(args.host) and not args.api_key:
        parser.error("--api-key is required when binding a non-loopback host")

    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir
    if args.api_key:
        os.environ["PARATRAN_API_KEY"] = args.api_key
    else:
        os.environ.pop("PARATRAN_API_KEY", None)
    os.environ["PARATRAN_MAX_UPLOAD_MB"] = str(args.max_upload_mb)
    os.environ["PARATRAN_MAX_CONCURRENCY"] = str(args.max_concurrency)

    import uvicorn

    from paratran.server import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
