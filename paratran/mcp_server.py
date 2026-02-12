import json
import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("paratran-mcp")

mcp = FastMCP("paratran")


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


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
