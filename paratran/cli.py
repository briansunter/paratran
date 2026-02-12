import argparse
import json
import mimetypes
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlencode

from paratran.transcribe import DEFAULT_MODEL


def main():
    # Check if first arg is "serve" — handle as subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return _serve(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Parakeet MLX models.",
        usage="paratran [OPTIONS] AUDIOS...\n       paratran serve [--host HOST] [--port PORT] [--model MODEL] [--cache-dir DIR]",
    )
    parser.add_argument("audios", nargs="*", metavar="AUDIOS", help="Audio files to transcribe")
    parser.add_argument("-s", "--server", default=os.environ.get("PARATRAN_SERVER"),
                        help="URL of a running paratran server (e.g. http://localhost:8000)")
    parser.add_argument("--model", default=os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL),
                        help=f"HF model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--cache-dir", default=os.environ.get("PARATRAN_MODEL_DIR"),
                        help="Directory for HuggingFace model cache")
    parser.add_argument("--output-dir", default=".", help="Directory to save transcriptions (default: .)")
    parser.add_argument("--output-format", default="txt",
                        help="Output format: txt, json, srt, vtt, all (default: txt)")
    parser.add_argument("--decoding", default="greedy", choices=["greedy", "beam"],
                        help="Decoding method (default: greedy)")
    parser.add_argument("--chunk-duration", type=float, default=120,
                        help="Chunk duration in seconds, 0 to disable (default: 120)")
    parser.add_argument("--overlap-duration", type=float, default=15.0,
                        help="Overlap duration in seconds (default: 15)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--length-penalty", type=float, default=0.013, help="Length penalty (default: 0.013)")
    parser.add_argument("--patience", type=float, default=3.5, help="Patience (default: 3.5)")
    parser.add_argument("--duration-reward", type=float, default=0.67, help="Duration reward (default: 0.67)")
    parser.add_argument("--max-words", type=int, default=None, help="Max words per sentence")
    parser.add_argument("--silence-gap", type=float, default=None, help="Split at silence gaps (seconds)")
    parser.add_argument("--max-duration", type=float, default=None, help="Max sentence duration (seconds)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision instead of BF16")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    if not args.audios:
        parser.print_help()
        sys.exit(1)

    chunk = args.chunk_duration if args.chunk_duration > 0 else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = ["txt", "srt", "vtt", "json"] if args.output_format == "all" else [args.output_format]

    if args.server:
        _transcribe_via_server(args, chunk, output_dir, formats)
    else:
        _transcribe_local(args, chunk, output_dir, formats)


def _transcribe_local(args, chunk, output_dir, formats):
    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir

    from paratran.transcribe import transcribe_file

    for audio_path in args.audios:
        path = Path(audio_path)
        if not path.exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"Transcribing: {path.name}", file=sys.stderr)

        result = transcribe_file(
            str(path),
            decoding=args.decoding,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            patience=args.patience,
            duration_reward=args.duration_reward,
            max_words=args.max_words,
            silence_gap=args.silence_gap,
            max_duration=args.max_duration,
            chunk_duration=chunk,
            overlap_duration=args.overlap_duration,
            fp32=args.fp32,
        )

        if args.verbose:
            print(f"  Duration: {result['duration']:.2f}s, Processing: {result['processing_time']:.3f}s", file=sys.stderr)

        _write_output(result, path.stem, output_dir, formats, args.verbose)


def _transcribe_via_server(args, chunk, output_dir, formats):
    server_url = args.server.rstrip("/")

    params = {"decoding": args.decoding, "beam_size": args.beam_size,
              "length_penalty": args.length_penalty, "patience": args.patience,
              "duration_reward": args.duration_reward,
              "overlap_duration": args.overlap_duration}
    if args.max_words is not None:
        params["max_words"] = args.max_words
    if args.silence_gap is not None:
        params["silence_gap"] = args.silence_gap
    if args.max_duration is not None:
        params["max_duration"] = args.max_duration
    if chunk is not None:
        params["chunk_duration"] = chunk
    if args.fp32:
        params["fp32"] = "true"

    url = f"{server_url}/transcribe?{urlencode(params)}"

    for audio_path in args.audios:
        path = Path(audio_path)
        if not path.exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"Uploading to server: {path.name}", file=sys.stderr)

        try:
            result = _upload_file(url, path)
        except urllib.error.URLError as e:
            print(f"Error: Could not connect to server at {server_url}: {e.reason}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"Error: Server returned {e.code}: {body}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"  Duration: {result['duration']:.2f}s, Processing: {result['processing_time']:.3f}s", file=sys.stderr)

        _write_output(result, path.stem, output_dir, formats, args.verbose)


def _upload_file(url: str, path: Path) -> dict:
    """Upload an audio file to the server using multipart/form-data."""
    boundary = "----ParatranBoundary"
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_data = path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _write_output(result: dict, stem: str, output_dir: Path, formats: list, verbose: bool):
    for fmt in formats:
        if fmt == "txt":
            out = output_dir / f"{stem}.txt"
            out.write_text(result["text"] + "\n")
        elif fmt == "json":
            out = output_dir / f"{stem}.json"
            out.write_text(json.dumps(result, indent=2) + "\n")
        elif fmt == "srt":
            out = output_dir / f"{stem}.srt"
            out.write_text(_to_srt(result["sentences"]))
        elif fmt == "vtt":
            out = output_dir / f"{stem}.vtt"
            out.write_text(_to_vtt(result["sentences"]))

        if verbose:
            print(f"  Saved: {out}", file=sys.stderr)


def _serve(argv):
    parser = argparse.ArgumentParser(
        prog="paratran serve",
        description="Start the Paratran REST API server.",
    )
    parser.add_argument("--model", default=os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL),
                        help=f"HF model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--cache-dir", default=os.environ.get("PARATRAN_MODEL_DIR"),
                        help="Directory for HuggingFace model cache")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args(argv)

    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir

    from paratran.server import app
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


def _format_ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _to_srt(sentences: list) -> str:
    lines = []
    for i, sent in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append(f"{_format_ts_srt(sent['start'])} --> {_format_ts_srt(sent['end'])}")
        lines.append(sent["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _to_vtt(sentences: list) -> str:
    lines = ["WEBVTT", ""]
    for sent in sentences:
        lines.append(f"{_format_ts_vtt(sent['start'])} --> {_format_ts_vtt(sent['end'])}")
        lines.append(sent["text"].strip())
        lines.append("")
    return "\n".join(lines)
