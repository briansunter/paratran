"""OpenAI-compatible REST interface for Paratran."""

from __future__ import annotations

import asyncio
import hmac
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from paratran.contracts import (
    ALLOWED_EXTENSIONS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_DECODING,
    DEFAULT_DURATION_REWARD,
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_OVERLAP_DURATION,
    DEFAULT_PATIENCE,
    RESPONSE_FORMATS,
    OptionValidationError,
    TranscriptionOptions,
    TranscriptionResult,
)
from paratran.serializers import to_openai_response


def _load_model() -> None:
    from paratran.transcribe import get_model

    get_model()


def _transcribe_file(path: str, options: TranscriptionOptions) -> TranscriptionResult:
    from paratran.transcribe import transcribe_file

    return transcribe_file(path, options=options)


def _model_status() -> dict[str, str | None]:
    from paratran.transcribe import model_status

    return model_status()


def _max_upload_bytes() -> int:
    try:
        megabytes = int(os.environ.get("PARATRAN_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
    except ValueError:
        megabytes = DEFAULT_MAX_UPLOAD_MB
    return max(megabytes, 1) * 1024 * 1024


def _max_concurrency() -> int:
    try:
        return max(
            int(os.environ.get("PARATRAN_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)),
            1,
        )
    except ValueError:
        return DEFAULT_MAX_CONCURRENCY


def _provided_api_key(
    x_api_key: str | None,
    authorization: str | None,
) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    expected = os.environ.get("PARATRAN_API_KEY")
    if not expected:
        return
    provided = _provided_api_key(x_api_key, authorization)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="A valid API key is required",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup runs before HTTP requests are accepted. Loading here keeps model
    # initialization on the process thread, while request-time inference below
    # is moved off the event loop.
    _load_model()
    application.state.transcription_semaphore = asyncio.Semaphore(_max_concurrency())
    yield


app = FastAPI(
    title="Paratran",
    description="OpenAI-compatible audio transcription interface powered by parakeet-mlx",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    status = _model_status()
    return {
        "status": "ok" if status["model"] else "starting",
        **status,
    }


def _response_for(result: TranscriptionResult, response_format: str):
    rendered = to_openai_response(result, response_format)
    if response_format == "text":
        return PlainTextResponse(rendered, media_type="text/plain")
    if response_format == "srt":
        return PlainTextResponse(rendered, media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(rendered, media_type="text/vtt")
    return JSONResponse(rendered)


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_api_key)])
async def transcribe(
    file: UploadFile = File(...),
    # OpenAI-compatible parameters. These are accepted for request compatibility;
    # model, language, prompt, and temperature are currently configured/auto-detected.
    model: str | None = Form(
        None,
        description="Accepted for compatibility; server model is configured at startup",
    ),
    response_format: str = Form("json", description="json, text, srt, vtt, or verbose_json"),
    language: str | None = Form(
        None,
        description="Accepted for compatibility; language is auto-detected",
    ),
    prompt: str | None = Form(
        None,
        description="Accepted for compatibility; prompts are not applied",
    ),
    temperature: float | None = Form(
        None,
        description="Accepted for compatibility; temperature is not applied",
    ),
    # Paratran-specific parameters
    decoding: str = Form(DEFAULT_DECODING),
    beam_size: int = Form(DEFAULT_BEAM_SIZE, ge=1),
    length_penalty: float = Form(DEFAULT_LENGTH_PENALTY, ge=0),
    patience: float = Form(DEFAULT_PATIENCE, gt=0),
    duration_reward: float = Form(DEFAULT_DURATION_REWARD, ge=0, le=1),
    max_words: int | None = Form(None, gt=0),
    silence_gap: float | None = Form(None, gt=0),
    max_duration: float | None = Form(None, gt=0),
    chunk_duration: float | None = Form(DEFAULT_CHUNK_DURATION, ge=0),
    overlap_duration: float = Form(DEFAULT_OVERLAP_DURATION, ge=0),
    fp32: bool = Form(False),
):
    del model, language, prompt, temperature

    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Unsupported file type '{suffix}'. Allowed: "
                    f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )
            },
        )
    if response_format not in RESPONSE_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Invalid response_format '{response_format}'. Must be one of: "
                    f"{', '.join(RESPONSE_FORMATS)}"
                )
            },
        )

    try:
        options = TranscriptionOptions(
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
    except OptionValidationError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temp_path = Path(temporary_file.name)
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > _max_upload_bytes():
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file exceeds the "
                            f"{_max_upload_bytes() // (1024 * 1024)} MB limit"
                        ),
                    )
                temporary_file.write(chunk)

        semaphore = getattr(app.state, "transcription_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(_max_concurrency())
            app.state.transcription_semaphore = semaphore
        async with semaphore:
            result = await asyncio.to_thread(_transcribe_file, str(temp_path), options)
        return _response_for(result, response_format)
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Transcription failed", "detail": str(exc)},
        )
    finally:
        await file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)
