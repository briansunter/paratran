# Paratran

CLI, REST API, and MCP server for audio transcription on Apple Silicon, powered by [parakeet-mlx](https://github.com/senstella/parakeet-mlx).

The default model ([parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) achieves 6.34% average WER across 8 English benchmarks and supports 25 languages. Runs ~30x faster than Whisper on Apple Silicon via MLX.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~2 GB memory for the default model

## Quick Start

Transcribe audio files directly:

```bash
uvx paratran recording.wav
```

Or start the REST API server:

```bash
uvx paratran serve
```

## Install

### uv (recommended)

```bash
uv tool install paratran
```

### pip

```bash
pip install paratran
```

### From source

```bash
git clone https://github.com/briansunter/paratran.git
cd paratran
uv sync
uv run paratran
```

## CLI Usage

```bash
# Transcribe a single file
paratran recording.wav

# Transcribe multiple files with verbose output
paratran -v file1.wav file2.mp3 file3.m4a

# Output as SRT subtitles
paratran --output-format srt recording.wav

# Output all formats (txt, json, srt, vtt)
paratran --output-format all --output-dir ./output recording.wav

# Use beam search decoding
paratran --decoding beam recording.wav

# Custom model and cache directory
paratran --model mlx-community/parakeet-tdt-1.1b-v2 --cache-dir /Volumes/Storage/models recording.wav
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/parakeet-tdt-0.6b-v3` | HF model ID or local path |
| `--cache-dir` | HuggingFace default | Model cache directory |
| `--output-dir` | `.` | Output directory |
| `--output-format` | `txt` | `txt`, `json`, `srt`, `vtt`, or `all` |
| `--decoding` | `greedy` | `greedy` or `beam` |
| `--chunk-duration` | `120` | Chunk duration in seconds (0 to disable) |
| `--overlap-duration` | `15` | Overlap between chunks |
| `--beam-size` | `5` | Beam size (beam decoding) |
| `--length-penalty` | `0.013` | Length penalty (beam decoding) |
| `--patience` | `3.5` | Patience (beam decoding) |
| `--duration-reward` | `0.67` | Duration reward (beam decoding) |
| `--max-words` | | Max words per sentence |
| `--silence-gap` | | Split at silence gaps (seconds) |
| `--max-duration` | | Max sentence duration (seconds) |
| `--fp32` | | Use FP32 precision instead of BF16 |
| `-v` | | Verbose output |

Environment variables: `PARATRAN_MODEL`, `PARATRAN_MODEL_DIR`.

## REST API Server

```bash
# Start server with default settings
paratran serve

# Custom host, port, and model cache
paratran serve --host 127.0.0.1 --port 9000 --cache-dir /Volumes/Storage/models
```

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

Optional query parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decoding` | `greedy` | `greedy` or `beam` |
| `beam_size` | `5` | Beam size (beam decoding) |
| `length_penalty` | `1.0` | Length penalty (beam decoding) |
| `patience` | `1.0` | Patience (beam decoding) |
| `duration_reward` | `0.7` | Duration reward (beam decoding) |
| `max_words` | | Max words per sentence |
| `silence_gap` | | Split at silence gaps (seconds) |
| `max_duration` | | Max sentence duration (seconds) |
| `chunk_duration` | | Chunk duration for long audio (seconds) |
| `overlap_duration` | `15.0` | Overlap between chunks (seconds) |
| `fp32` | `false` | Use FP32 instead of BF16 |

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

## MCP Server

Paratran includes an MCP server so Claude Code, Claude Desktop, or any MCP client can transcribe audio files directly.

### Claude Code

Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "paratran": {
      "command": "uvx",
      "args": ["--from", "paratran", "paratran-mcp"]
    }
  }
}
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "paratran": {
      "command": "uvx",
      "args": ["--from", "paratran", "paratran-mcp"]
    }
  }
}
```

Optionally set `PARATRAN_MODEL_DIR` in the `env` block to customize the model cache location.

### MCP Tool

The `transcribe` tool accepts a file path and all the same options as the REST API (decoding, beam search, sentence splitting, chunking, precision).

## License

MIT
