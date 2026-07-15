"""Serialization adapters for CLI files and the OpenAI-compatible REST interface."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from paratran.contracts import (
    OUTPUT_FORMATS,
    RESPONSE_FORMATS,
    Sentence,
    Token,
    TranscriptionResult,
)


def format_timestamp(seconds: float, separator: str) -> str:
    if seconds < 0:
        raise ValueError("timestamps must be non-negative")
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


def to_srt(result: TranscriptionResult) -> str:
    lines: list[str] = []
    for index, sentence in enumerate(result.sentences, 1):
        lines.extend(
            [
                str(index),
                f"{format_timestamp(sentence.start, ',')} --> "
                f"{format_timestamp(sentence.end, ',')}",
                sentence.text.strip(),
                "",
            ]
        )
    return "\n".join(lines)


def to_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for sentence in result.sentences:
        lines.extend(
            [
                f"{format_timestamp(sentence.start, '.')} --> "
                f"{format_timestamp(sentence.end, '.')}",
                sentence.text.strip(),
                "",
            ]
        )
    return "\n".join(lines)


def render_cli(result: TranscriptionResult, output_format: str) -> str:
    if output_format == "txt":
        return result.text + "\n"
    if output_format == "json":
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if output_format == "srt":
        return to_srt(result)
    if output_format == "vtt":
        return to_vtt(result)
    raise ValueError(
        f"Invalid output format '{output_format}'. Choose from {', '.join(OUTPUT_FORMATS)}."
    )


def write_outputs(
    result: TranscriptionResult,
    stem: str,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    output_paths: list[Path] = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        path.write_text(render_cli(result, output_format), encoding="utf-8")
        output_paths.append(path)
    return output_paths


def to_openai_response(result: TranscriptionResult, response_format: str) -> dict[str, Any] | str:
    if response_format == "text":
        return result.text
    if response_format == "srt":
        return to_srt(result)
    if response_format == "vtt":
        return to_vtt(result)
    if response_format == "verbose_json":
        segments = []
        words = []
        for index, sentence in enumerate(result.sentences):
            segments.append(
                {
                    "id": index,
                    "start": sentence.start,
                    "end": sentence.end,
                    "text": sentence.text,
                }
            )
            words.extend(
                {
                    "word": token.text,
                    "start": token.start,
                    "end": token.end,
                    **({"duration": token.duration} if token.duration is not None else {}),
                    **({"confidence": token.confidence} if token.confidence is not None else {}),
                }
                for token in sentence.tokens
            )
        return {
            "task": "transcribe",
            "duration": result.duration,
            "processing_time": result.processing_time,
            "text": result.text,
            "segments": segments,
            "words": words,
        }
    if response_format == "json":
        return {"text": result.text}
    raise ValueError(
        f"Invalid response format '{response_format}'. Choose from {', '.join(RESPONSE_FORMATS)}."
    )


def from_openai_verbose_json(response: dict[str, Any]) -> TranscriptionResult:
    words = response.get("words", [])
    sentences: list[Sentence] = []
    used_word_indexes: set[int] = set()

    for segment in response.get("segments", []):
        start = float(segment["start"])
        end = float(segment["end"])
        tokens: list[Token] = []
        for index, word in enumerate(words):
            if index in used_word_indexes:
                continue
            word_start = float(word["start"])
            word_end = float(word["end"])
            if word_start >= start - 0.001 and word_end <= end + 0.001:
                tokens.append(
                    Token(
                        text=word["word"],
                        start=word_start,
                        end=word_end,
                        duration=word.get("duration"),
                        confidence=word.get("confidence"),
                    )
                )
                used_word_indexes.add(index)
        sentences.append(
            Sentence(
                text=segment["text"],
                start=start,
                end=end,
                tokens=tuple(tokens),
            )
        )

    return TranscriptionResult(
        text=response["text"],
        duration=float(response.get("duration", 0.0)),
        processing_time=float(response.get("processing_time", 0.0)),
        sentences=tuple(sentences),
    )
