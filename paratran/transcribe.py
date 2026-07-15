"""Model lifecycle and transcription implementation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from paratran.contracts import (
    ALLOWED_EXTENSIONS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_DECODING,
    DEFAULT_DURATION_REWARD,
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP_DURATION,
    DEFAULT_PATIENCE,
    Sentence,
    Token,
    TranscriptionOptions,
    TranscriptionResult,
)

_model: Any = None
_model_name: str | None = None
_model_dir: str | None = None
_model_lock = threading.Lock()


def get_model(
    model_name: str | None = None,
    model_dir: str | None = None,
):
    """Load and cache one model, serializing concurrent first loads."""

    global _model, _model_name, _model_dir
    name = model_name or os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL)
    cache = model_dir or os.environ.get("PARATRAN_MODEL_DIR")

    with _model_lock:
        if _model is not None and _model_name == name and _model_dir == cache:
            return _model

        import parakeet_mlx

        kwargs = {"cache_dir": cache} if cache else {}
        loaded_model = parakeet_mlx.from_pretrained(name, **kwargs)
        _model = loaded_model
        _model_name = name
        _model_dir = cache
        return loaded_model


def model_status() -> dict[str, str | None]:
    return {"model": _model_name, "model_dir": _model_dir}


def _build_options(
    *,
    decoding: str,
    beam_size: int,
    length_penalty: float,
    patience: float,
    duration_reward: float,
    max_words: int | None,
    silence_gap: float | None,
    max_duration: float | None,
    chunk_duration: float | None,
    overlap_duration: float,
    fp32: bool,
) -> TranscriptionOptions:
    return TranscriptionOptions(
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


def _audio_duration(path: Path, fallback: float) -> float:
    """Return the input duration when ffprobe is available, otherwise speech end."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return fallback

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        duration = float(completed.stdout.strip())
        return max(duration, 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return fallback


def transcribe_file(
    file_path: str,
    *,
    options: TranscriptionOptions | None = None,
    model_name: str | None = None,
    model_dir: str | None = None,
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
) -> TranscriptionResult:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if options is None:
        options = _build_options(
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

    model = get_model(model_name, model_dir)

    from parakeet_mlx import Beam, DecodingConfig, Greedy, SentenceConfig

    decoding_method = (
        Beam(
            beam_size=options.beam_size,
            length_penalty=options.length_penalty,
            patience=options.patience,
            duration_reward=options.duration_reward,
        )
        if options.decoding == "beam"
        else Greedy()
    )
    config = DecodingConfig(
        decoding=decoding_method,
        sentence=SentenceConfig(
            max_words=options.max_words,
            silence_gap=options.silence_gap,
            max_duration=options.max_duration,
        ),
    )

    import mlx.core as mx

    dtype = mx.float32 if options.fp32 else mx.bfloat16
    start = time.perf_counter()
    result = model.transcribe(
        str(path),
        dtype=dtype,
        decoding_config=config,
        chunk_duration=options.chunk_duration,
        overlap_duration=options.overlap_duration,
    )
    elapsed = time.perf_counter() - start

    sentences: list[Sentence] = []
    for segment in result.sentences:
        tokens = tuple(
            Token(
                text=token.text,
                start=token.start,
                end=token.end,
                duration=getattr(token, "duration", None),
                confidence=getattr(token, "confidence", None),
            )
            for token in segment.tokens
        )
        sentences.append(
            Sentence(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                tokens=tokens,
            )
        )

    speech_end = sentences[-1].end if sentences else 0.0
    return TranscriptionResult(
        text=result.text,
        duration=_audio_duration(path, speech_end),
        processing_time=round(elapsed, 3),
        sentences=tuple(sentences),
    )


def transcribe_file_json(file_path: str, **kwargs: Any) -> str:
    """Transcribe and return as a formatted JSON string."""

    return json.dumps(
        transcribe_file(file_path, **kwargs).to_dict(),
        indent=2,
        ensure_ascii=False,
    )
