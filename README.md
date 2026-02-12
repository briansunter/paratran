# Paratran

REST API for audio transcription on Apple Silicon, powered by [parakeet-mlx](https://github.com/senstella/parakeet-mlx).

Parakeet is #1 on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) and runs ~30x faster than Whisper on Apple Silicon via MLX.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~2 GB memory for the default model

## Quick Start

```bash
git clone https://github.com/briansunter/paratran.git
cd paratran
./run.sh
```

This creates a virtual environment, installs dependencies, downloads the model, and starts the server at `http://localhost:8000`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run:

```bash
paratran
```

## Usage

```bash
# Default settings
paratran

# Custom model cache location
paratran --model-dir /Volumes/Storage/models

# Custom host and port
paratran --host 127.0.0.1 --port 9000

# Different model
paratran --model mlx-community/parakeet-tdt-1.1b-v2
```

All options can also be set via environment variables:

| CLI Flag       | Env Var              | Default                                  |
|----------------|----------------------|------------------------------------------|
| `--model`      | `PARATRAN_MODEL`     | `mlx-community/parakeet-tdt-0.6b-v3`    |
| `--model-dir`  | `PARATRAN_MODEL_DIR` | HuggingFace default (`~/.cache/huggingface`) |
| `--host`       |                      | `0.0.0.0`                                |
| `--port`       |                      | `8000`                                   |

## API

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "mlx-community/parakeet-tdt-0.6b-v3",
  "model_dir": "/Volumes/Storage/models"
}
```

### `POST /transcribe`

Upload an audio file (wav, mp3, flac, m4a, ogg, webm):

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@recording.m4a"
```

```json
{
  "text": "Hello world, this is a test.",
  "duration": 3.52,
  "processing_time": 0.176,
  "sentences": [
    {
      "text": "Hello world, this is a test.",
      "start": 0.0,
      "end": 3.52,
      "tokens": [
        { "text": "Hello", "start": 0.0, "end": 0.48 },
        { "text": " world", "start": 0.48, "end": 0.8 }
      ]
    }
  ]
}
```

Interactive API docs are available at `http://localhost:8000/docs`.

## License

MIT
