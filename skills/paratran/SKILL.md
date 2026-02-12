---
name: Paratran Transcription
description: Transcribe audio files using Paratran CLI, REST API, or MCP server, powered by parakeet-mlx on Apple Silicon. Use when transcribing audio, converting speech to text, generating subtitles (SRT/VTT), getting word-level timestamps, or working with audio files (wav, mp3, flac, m4a, ogg, webm).
keywords: [transcription, asr, speech-to-text, audio, mlx, apple-silicon, parakeet, subtitles, srt, vtt]
topics: [audio-processing, machine-learning, rest-api, mcp]
---

# Paratran Transcription

Audio transcription for Apple Silicon using parakeet-mlx. #1 on Open ASR Leaderboard, ~30x faster than Whisper via MLX.

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
| `--fp32` | | Use FP32 precision instead of BF16 |
| `-v` | | Verbose output |

Environment variables: `PARATRAN_MODEL`, `PARATRAN_MODEL_DIR`.

## REST API Server

```bash
# Start server
paratran serve

# Custom host, port, and model cache
paratran serve --host 127.0.0.1 --port 9000 --cache-dir /path/to/models
```

### Transcribe via API

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@recording.m4a"
```

### Extract just text

```bash
curl -s -X POST http://localhost:8000/transcribe -F "file=@audio.m4a" | jq -r '.text'
```

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

For Claude Code, add to `.claude/settings.json`:

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

The MCP `transcribe` tool accepts a file path and all transcription options (decoding, beam search, sentence splitting, chunking, precision).
