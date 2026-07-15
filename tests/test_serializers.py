import json

import pytest

from paratran.contracts import Sentence, Token, TranscriptionResult
from paratran.serializers import (
    format_timestamp,
    from_openai_verbose_json,
    render_cli,
    to_openai_response,
    to_srt,
    to_vtt,
    write_outputs,
)


def sample_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="Hello world.",
        duration=4.0,
        processing_time=0.25,
        sentences=(
            Sentence(
                text="Hello world.",
                start=0.25,
                end=1.5,
                tokens=(
                    Token("Hello", 0.25, 0.75, duration=0.5, confidence=0.9),
                    Token(" world.", 0.75, 1.5),
                ),
            ),
        ),
    )


def test_timestamp_rounding_avoids_float_boundary_errors():
    assert format_timestamp(1.9996, ",") == "00:00:02,000"
    assert format_timestamp(1.9996, ".") == "00:00:02.000"


def test_subtitle_serializers_emit_valid_headers_and_cues():
    result = sample_result()

    assert "1\n00:00:00,250 --> 00:00:01,500" in to_srt(result)
    assert to_vtt(result).startswith("WEBVTT\n\n")


def test_openai_round_trip_preserves_processing_and_word_metadata():
    result = sample_result()
    payload = to_openai_response(result, "verbose_json")
    restored = from_openai_verbose_json(payload)

    assert restored.text == result.text
    assert restored.duration == result.duration
    assert restored.processing_time == result.processing_time
    assert restored.sentences[0].tokens[0].confidence == 0.9


def test_write_outputs_uses_one_canonical_renderer(tmp_path):
    paths = write_outputs(sample_result(), "recording", tmp_path, ["txt", "json", "srt", "vtt"])

    assert [path.name for path in paths] == [
        "recording.txt",
        "recording.json",
        "recording.srt",
        "recording.vtt",
    ]
    assert json.loads((tmp_path / "recording.json").read_text())["text"] == "Hello world."


def test_invalid_output_format_fails_before_writing(tmp_path):
    with pytest.raises(ValueError, match="Invalid output format"):
        render_cli(sample_result(), "bogus")
