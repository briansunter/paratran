import argparse
import json
import os
import sys
from pathlib import Path

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

    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir

    from paratran.transcribe import transcribe_file

    chunk = args.chunk_duration if args.chunk_duration > 0 else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = ["txt", "srt", "vtt", "json"] if args.output_format == "all" else [args.output_format]

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

        stem = path.stem

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

            if args.verbose:
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
