import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import parakeet_mlx
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from parakeet_mlx import Beam, DecodingConfig, Greedy, SentenceConfig

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
MODEL_NAME = os.environ.get("PARATRAN_MODEL", DEFAULT_MODEL)
MODEL_DIR = os.environ.get("PARATRAN_MODEL_DIR", None)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}

app = FastAPI(title="Paratran", description="Audio transcription API powered by parakeet-mlx")
model = None


@app.on_event("startup")
def load_model():
    global model
    kwargs = {}
    if MODEL_DIR:
        kwargs["cache_dir"] = MODEL_DIR
    model = parakeet_mlx.from_pretrained(MODEL_NAME, **kwargs)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "model_dir": MODEL_DIR}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    # Decoding
    decoding: str = Query("greedy", description="Decoding method: 'greedy' or 'beam'"),
    beam_size: int = Query(5, description="Beam size (beam decoding only)"),
    length_penalty: float = Query(1.0, description="Length penalty (beam decoding only)"),
    patience: float = Query(1.0, description="Patience (beam decoding only)"),
    duration_reward: float = Query(0.7, description="Duration reward 0.0-1.0 (beam decoding only)"),
    # Sentence splitting
    max_words: Optional[int] = Query(None, description="Max words per sentence"),
    silence_gap: Optional[float] = Query(None, description="Split sentence on silence gap (seconds)"),
    max_duration: Optional[float] = Query(None, description="Max sentence duration (seconds)"),
    # Chunking
    chunk_duration: Optional[float] = Query(None, description="Chunk duration in seconds for long audio (None = no chunking)"),
    overlap_duration: float = Query(15.0, description="Overlap between chunks (seconds)"),
    # Precision
    fp32: bool = Query(False, description="Use float32 instead of bfloat16"),
):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"},
        )

    if decoding not in ("greedy", "beam"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid decoding method '{decoding}'. Must be 'greedy' or 'beam'."},
        )

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(await file.read())
        tmp.close()

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
            tmp.name,
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
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)
