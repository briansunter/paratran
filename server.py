import argparse
import os
import tempfile
import time
from pathlib import Path

import parakeet_mlx
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

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
):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"},
        )

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(await file.read())
        tmp.close()

        start = time.perf_counter()
        result = model.transcribe(tmp.name)
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


def main():
    global MODEL_NAME, MODEL_DIR

    parser = argparse.ArgumentParser(description="Paratran transcription server")
    parser.add_argument("--model", default=MODEL_NAME, help=f"HF model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory to download/cache models")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    MODEL_NAME = args.model
    MODEL_DIR = args.model_dir

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
