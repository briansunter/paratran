import pytest

from paratran.contracts import (
    DEFAULT_CHUNK_DURATION,
    DEFAULT_DURATION_REWARD,
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_PATIENCE,
    OptionValidationError,
    TranscriptionOptions,
)


def test_defaults_match_the_shared_contract():
    options = TranscriptionOptions()

    assert options.chunk_duration == DEFAULT_CHUNK_DURATION
    assert options.length_penalty == DEFAULT_LENGTH_PENALTY
    assert options.patience == DEFAULT_PATIENCE
    assert options.duration_reward == DEFAULT_DURATION_REWARD


def test_zero_chunk_duration_disables_chunking():
    assert TranscriptionOptions(chunk_duration=0).chunk_duration is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("beam_size", 0),
        ("length_penalty", -1),
        ("patience", 0),
        ("duration_reward", 2),
        ("max_words", 0),
        ("silence_gap", 0),
        ("max_duration", 0),
        ("overlap_duration", -1),
        ("patience", float("nan")),
        ("length_penalty", float("inf")),
        ("overlap_duration", float("nan")),
    ],
)
def test_invalid_options_are_rejected(field, value):
    with pytest.raises(OptionValidationError):
        TranscriptionOptions(**{field: value})


def test_overlap_must_fit_inside_a_chunk():
    with pytest.raises(OptionValidationError, match="overlap_duration"):
        TranscriptionOptions(chunk_duration=10, overlap_duration=10)
