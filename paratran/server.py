import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from paratran.transcribe import ALLOWED_EXTENSIONS, get_model, transcribe_file

app = FastAPI(title="Paratran", description="Audio transcription API powered by parakeet-mlx")


@app.on_event("startup")
def load_model():
    get_model()


@app.get("/health")
def health():
    from paratran.transcribe import _model_dir, _model_name
    return {"status": "ok", "model": _model_name, "model_dir": _model_dir}


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

        return transcribe_file(
            tmp.name,
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
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)
