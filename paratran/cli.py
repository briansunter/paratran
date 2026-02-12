import argparse

from paratran.server import DEFAULT_MODEL, MODEL_DIR, MODEL_NAME


def main():
    from paratran import server

    parser = argparse.ArgumentParser(description="Paratran transcription server")
    parser.add_argument("--model", default=MODEL_NAME, help=f"HF model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory to download/cache models")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    server.MODEL_NAME = args.model
    server.MODEL_DIR = args.model_dir

    import uvicorn
    uvicorn.run(server.app, host=args.host, port=args.port)
