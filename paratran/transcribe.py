import json
import os
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import parakeet_mlx
from parakeet_mlx import Beam, DecodingConfig, Greedy, SentenceConfig

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}

_model = None
_model_name = None
_model_dir = None


def get_model(
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
):
    global _model, _model_name, _model_dir
    name = model_name or os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL)
    cache = model_dir or os.environ.get("PARATRAN_MODEL_DIR", None)

    if _model is not None and _model_name == name and _model_dir == cache:
        return _model

    kwargs = {}
    if cache:
        kwargs["cache_dir"] = cache
    _model = parakeet_mlx.from_pretrained(name, **kwargs)
    _model_name = name
    _model_dir = cache
    return _model


def transcribe_file(
    file_path: str,
    *,
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
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
) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    model = get_model(model_name, model_dir)

    if decoding == "beam":
        decoding_method = Beam(
            beam_size=beam_size,
            length_penalty=length_penalty,
            patience=patience,
            duration_reward=duration_reward,
        )
    else:
        decoding_method = Greedy()

    config = DecodingConfig(
        decoding=decoding_method,
        sentence=SentenceConfig(
            max_words=max_words,
            silence_gap=silence_gap,
            max_duration=max_duration,
        ),
    )

    dtype = mx.float32 if fp32 else mx.bfloat16

    start = time.perf_counter()
    result = model.transcribe(
        str(path),
        dtype=dtype,
        decoding_config=config,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )
    elapsed = time.perf_counter() - start

    sentences = []
    for seg in result.sentences:
        tokens = [
            {"text": t.text, "start": t.start, "end": t.end}
            for t in seg.tokens
        ]
        sentences.append({
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
            "tokens": tokens,
        })

    duration = result.sentences[-1].end if result.sentences else 0.0

    return {
        "text": result.text,
        "duration": duration,
        "processing_time": round(elapsed, 3),
        "sentences": sentences,
    }


def transcribe_file_json(file_path: str, **kwargs) -> str:
    """Transcribe and return as formatted JSON string."""
    return json.dumps(transcribe_file(file_path, **kwargs), indent=2)
