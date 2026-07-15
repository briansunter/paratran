"""Shared transcription contracts used by every Paratran interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
ALLOWED_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"})
RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
OUTPUT_FORMATS = ("txt", "json", "srt", "vtt", "all")

DEFAULT_DECODING = "greedy"
DEFAULT_BEAM_SIZE = 5
DEFAULT_LENGTH_PENALTY = 0.013
DEFAULT_PATIENCE = 3.5
DEFAULT_DURATION_REWARD = 0.67
DEFAULT_CHUNK_DURATION = 120.0
DEFAULT_OVERLAP_DURATION = 15.0
DEFAULT_HTTP_TIMEOUT = 60.0
DEFAULT_MAX_UPLOAD_MB = 512
DEFAULT_MAX_CONCURRENCY = 1


class OptionValidationError(ValueError):
    """Raised when a transcription option violates the shared contract."""


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    """Validated options shared by the CLI, REST interface, and MCP tool."""

    decoding: str = DEFAULT_DECODING
    beam_size: int = DEFAULT_BEAM_SIZE
    length_penalty: float = DEFAULT_LENGTH_PENALTY
    patience: float = DEFAULT_PATIENCE
    duration_reward: float = DEFAULT_DURATION_REWARD
    max_words: int | None = None
    silence_gap: float | None = None
    max_duration: float | None = None
    chunk_duration: float | None = DEFAULT_CHUNK_DURATION
    overlap_duration: float = DEFAULT_OVERLAP_DURATION
    fp32: bool = False

    def __post_init__(self) -> None:
        if self.decoding not in ("greedy", "beam"):
            raise OptionValidationError(
                f"Invalid decoding method '{self.decoding}'. Must be 'greedy' or 'beam'."
            )
        if self.beam_size < 1:
            raise OptionValidationError("beam_size must be at least 1")
        for name, value in (
            ("length_penalty", self.length_penalty),
            ("patience", self.patience),
            ("duration_reward", self.duration_reward),
            ("silence_gap", self.silence_gap),
            ("max_duration", self.max_duration),
            ("chunk_duration", self.chunk_duration),
            ("overlap_duration", self.overlap_duration),
        ):
            if value is not None and not math.isfinite(value):
                raise OptionValidationError(f"{name} must be finite")
        if self.length_penalty < 0:
            raise OptionValidationError("length_penalty must be non-negative")
        if self.patience <= 0:
            raise OptionValidationError("patience must be greater than 0")
        if not 0 <= self.duration_reward <= 1:
            raise OptionValidationError("duration_reward must be between 0 and 1")
        if self.max_words is not None and self.max_words < 1:
            raise OptionValidationError("max_words must be at least 1")
        if self.silence_gap is not None and self.silence_gap <= 0:
            raise OptionValidationError("silence_gap must be greater than 0")
        if self.max_duration is not None and self.max_duration <= 0:
            raise OptionValidationError("max_duration must be greater than 0")
        if self.chunk_duration is not None:
            if self.chunk_duration == 0:
                object.__setattr__(self, "chunk_duration", None)
            elif self.chunk_duration < 0:
                raise OptionValidationError("chunk_duration must be 0 or greater")
        if self.overlap_duration < 0:
            raise OptionValidationError("overlap_duration must be non-negative")
        if self.chunk_duration is not None and self.overlap_duration >= self.chunk_duration:
            raise OptionValidationError("overlap_duration must be less than chunk_duration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoding": self.decoding,
            "beam_size": self.beam_size,
            "length_penalty": self.length_penalty,
            "patience": self.patience,
            "duration_reward": self.duration_reward,
            "max_words": self.max_words,
            "silence_gap": self.silence_gap,
            "max_duration": self.max_duration,
            "chunk_duration": self.chunk_duration,
            "overlap_duration": self.overlap_duration,
            "fp32": self.fp32,
        }


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: float
    end: float
    duration: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }
        if self.duration is not None:
            value["duration"] = self.duration
        if self.confidence is not None:
            value["confidence"] = self.confidence
        return value


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    start: float
    end: float
    tokens: tuple[Token, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "tokens": [token.to_dict() for token in self.tokens],
        }


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    duration: float
    processing_time: float
    sentences: tuple[Sentence, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }
