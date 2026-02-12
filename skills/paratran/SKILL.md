---
name: Paratran Transcription
description: Transcribe audio files using Paratran CLI, REST API, or MCP server, powered by parakeet-mlx on Apple Silicon. Use when transcribing audio, converting speech to text, generating subtitles (SRT/VTT), getting word-level timestamps, or working with audio files (wav, mp3, flac, m4a, ogg, webm).
keywords: [transcription, asr, speech-to-text, audio, mlx, apple-silicon, parakeet, subtitles, srt, vtt]
topics: [audio-processing, machine-learning, rest-api, mcp]
---

# Paratran Transcription

Audio transcription for Apple Silicon using parakeet-mlx. Default model achieves 6.34% average WER, supports 25 languages, ~30x faster than Whisper via MLX.

Three interfaces: CLI, REST API, and MCP server.

## Setup

### Quick run (no install)

```bash
uvx paratran recording.wav
```

### Persistent install

```bash
uv tool install paratran
```

### From source

```bash
git clone https://github.com/briansunter/paratran.git
cd paratran
uv sync
uv run paratran recording.wav
```

## CLI Transcription

```bash
# Transcribe to text (default)
paratran recording.wav

# Multiple files with verbose output
paratran -v file1.wav file2.mp3 file3.m4a

# Output as SRT subtitles
paratran --output-format srt recording.wav

# All formats (txt, json, srt, vtt) to a directory
paratran --output-format all --output-dir ./output recording.wav

# Beam search decoding
paratran --decoding beam recording.wav

# Custom model and cache directory
paratran --model mlx-community/parakeet-tdt-1.1b-v2 --cache-dir /path/to/models recording.wav
```

### Client Mode

Use `--server` / `-s` to send files to a running server instead of transcribing locally. Avoids model loading on every call.

```bash
# Start server once
paratran serve

# Transcribe via server (instant, no model loading)
paratran -s http://localhost:8000 recording.wav

# All options work in client mode
paratran -s http://localhost:8000 --output-format all --output-dir ./output -v recording.wav

# Or set via environment variable
export PARATRAN_SERVER=http://localhost:8000
paratran recording.wav
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-s`, `--server` | | URL of a running paratran server |
| `--model` | `mlx-community/parakeet-tdt-0.6b-v3` | HF model ID or local path |
| `--cache-dir` | HuggingFace default | Model cache directory |
| `--output-dir` | `.` | Output directory |
| `--output-format` | `txt` | `txt`, `json`, `srt`, `vtt`, or `all` |
| `--decoding` | `greedy` | `greedy` or `beam` |
| `--chunk-duration` | `120` | Chunk duration in seconds (0 to disable) |
| `--overlap-duration` | `15` | Overlap between chunks |
| `--beam-size` | `5` | Beam size (beam decoding) |
| `--fp32` | | Use FP32 precision instead of BF16 |
| `-v` | | Verbose output |

Environment variables: `PARATRAN_MODEL`, `PARATRAN_MODEL_DIR`, `PARATRAN_SERVER`.

## REST API Server

```bash
# Start server
paratran serve

# Custom host, port, and model cache
paratran serve --host 127.0.0.1 --port 9000 --cache-dir /path/to/models
```

### Endpoints

**`GET /health`** — Returns model name, status, and cache directory.

**`POST /transcribe`** — Upload audio file, returns transcription JSON.

```bash
# Basic transcription
curl -X POST http://localhost:8000/transcribe -F "file=@recording.m4a"

# With beam search and sentence splitting
curl -X POST "http://localhost:8000/transcribe?decoding=beam&max_words=20" -F "file=@recording.m4a"

# Extract just text
curl -s -X POST http://localhost:8000/transcribe -F "file=@audio.m4a" | jq -r '.text'
```

Query parameters: `decoding`, `beam_size`, `length_penalty`, `patience`, `duration_reward`, `max_words`, `silence_gap`, `max_duration`, `chunk_duration`, `overlap_duration`, `fp32`.

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

Interactive API docs at `http://localhost:8000/docs`.

## MCP Server

Supports stdio (for Claude Code/Desktop) and streamable HTTP (for remote/multi-client access).

### Claude Code (stdio)

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

### Claude Desktop (stdio)

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

### Streamable HTTP

```bash
paratran-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

MCP endpoint at `http://localhost:8000/mcp`.

### MCP Tool

The `transcribe` tool accepts:
- `file_path` (required) — absolute path to audio file
- All transcription options: `decoding`, `beam_size`, `length_penalty`, `patience`, `duration_reward`, `max_words`, `silence_gap`, `max_duration`, `chunk_duration`, `overlap_duration`, `fp32`

Returns JSON string with full text, duration, processing time, and sentences with word-level timestamps.
