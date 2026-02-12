import argparse

from paratran.transcribe import DEFAULT_MODEL


def main():
    import paratran.transcribe as t

    model_name = t._model_name or DEFAULT_MODEL
    model_dir = t._model_dir

    parser = argparse.ArgumentParser(description="Paratran transcription server")
    parser.add_argument("--model", default=model_name, help=f"HF model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-dir", default=model_dir, help="Directory to download/cache models")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    # Pre-configure the model before FastAPI startup
    import os
    os.environ["PARATRAN_MODEL"] = args.model
    if args.model_dir:
        os.environ["PARATRAN_MODEL_DIR"] = args.model_dir

    from paratran.server import app
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
