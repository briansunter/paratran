from pathlib import Path

import pytest

import paratran.transcribe as transcribe


def test_audio_duration_falls_back_without_ffprobe(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(transcribe.shutil, "which", lambda _name: None)

    assert transcribe._audio_duration(audio, 4.5) == 4.5


def test_directory_is_not_accepted_as_audio(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        transcribe.transcribe_file(str(tmp_path / "sample.wav"))
