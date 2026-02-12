---
name: Paratran Transcription
description: Transcribe audio files using the Paratran REST API server powered by parakeet-mlx on Apple Silicon. Use when transcribing audio, converting speech to text, getting word-level timestamps, or working with audio files (wav, mp3, flac, m4a, ogg, webm).
keywords: [transcription, asr, speech-to-text, audio, mlx, apple-silicon, parakeet]
topics: [audio-processing, machine-learning, rest-api]
---

# Paratran Transcription

Audio transcription REST API for Apple Silicon using parakeet-mlx. #1 on Open ASR Leaderboard, ~30x faster than Whisper.

## Setup

### Quick run (no install)

```bash
uvx paratran --model-dir /Volumes/Storage/models
```

### Persistent install

```bash
uv tool install paratran
paratran --model-dir /Volumes/Storage/models
```

### From source

```bash
git clone https://github.com/briansunter/paratran.git
cd paratran
uv sync
uv run paratran
```

## CLI Options

| Flag | Env Var | Default |
|------|---------|---------|
| `--model` | `PARATRAN_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` |
| `--model-dir` | `PARATRAN_MODEL_DIR` | `~/.cache/huggingface` |
| `--host` | | `0.0.0.0` |
| `--port` | | `8000` |

## API Endpoints

### Health check

```bash
curl http://localhost:8000/health
```

### Transcribe audio

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@recording.m4a"
```

Supported formats: wav, mp3, flac, m4a, ogg, webm.

### Response format

```json
{
  "text": "Full transcription text.",
  "duration": 3.52,
  "processing_time": 0.176,
  "sentences": [
    {
      "text": "Full transcription text.",
      "start": 0.0,
      "end": 3.52,
      "tokens": [
        { "text": "Full", "start": 0.0, "end": 0.24 },
        { "text": " transcription", "start": 0.24, "end": 0.8 }
      ]
    }
  ]
}
```

## Common Tasks

### Start server with custom model cache

```bash
paratran --model-dir /Volumes/Storage/models --port 8000
```

### Transcribe and extract just text

```bash
curl -s -X POST http://localhost:8000/transcribe -F "file=@audio.m4a" | jq -r '.text'
```

### Transcribe and save full JSON output

```bash
curl -s -X POST http://localhost:8000/transcribe -F "file=@audio.m4a" | jq . > transcript.json
```

### Use a different model

```bash
paratran --model mlx-community/parakeet-tdt-1.1b-v2 --model-dir /Volumes/Storage/models
```

### Interactive API docs

Open `http://localhost:8000/docs` in a browser for the Swagger UI.
