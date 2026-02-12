import argparse
import json
import logging
import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("paratran-mcp")


def create_mcp(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("paratran", host=host, port=port)

    @mcp.tool()
    def transcribe(
        file_path: str,
        decoding: str = "greedy",
        beam_size: int = 5,
        length_penalty: float = 1.0,
        patience: float = 1.0,
        duration_reward: float = 0.7,
        max_words: Optional[int] = None,
        silence_gap: Optional[float] = None,
        max_duration: Optional[float] = None,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        fp32: bool = False,
    ) -> str:
        """Transcribe an audio file to text with word-level timestamps.

        Returns JSON with full text, duration, processing time, and sentences
        with word-level timestamps.

        Args:
            file_path: Absolute path to the audio file (wav, mp3, flac, m4a, ogg, webm).
            decoding: Decoding method - 'greedy' or 'beam'.
            beam_size: Beam size (beam decoding only).
            length_penalty: Length penalty (beam decoding only).
            patience: Patience (beam decoding only).
            duration_reward: Duration reward 0.0-1.0 (beam decoding only).
            max_words: Max words per sentence.
            silence_gap: Split sentence on silence gap (seconds).
            max_duration: Max sentence duration (seconds).
            chunk_duration: Chunk duration in seconds for long audio (None = no chunking).
            overlap_duration: Overlap between chunks (seconds).
            fp32: Use float32 instead of bfloat16.
        """
        from paratran.transcribe import transcribe_file

        logger.info(f"Transcribing: {file_path}")
        result = transcribe_file(
            file_path,
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
        return json.dumps(result, indent=2)

    return mcp


def main():
    from paratran.transcribe import DEFAULT_MODEL

    parser = argparse.ArgumentParser(
        prog="paratran-mcp",
        description="Paratran MCP server for audio transcription.",
    )
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "streamable-http"],
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for HTTP (default: 8000)")
    parser.add_argument(
        "--model", default=os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL),
        help=f"HF model ID or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("PARATRAN_MODEL_DIR"),
                        help="Directory for HuggingFace model cache")
    args = parser.parse_args()

    os.environ["PARATRAN_MODEL"] = args.model
    if args.cache_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.cache_dir

    mcp = create_mcp(host=args.host, port=args.port)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
