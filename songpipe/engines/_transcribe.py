"""Runs INSIDE the ASR venv (paths.ASR_PY — any venv with mlx-whisper installed).
Transcribes an audio file, prints the transcript."""
import sys, mlx_whisper
def main():
    r = mlx_whisper.transcribe(sys.argv[1], path_or_hf_repo="mlx-community/whisper-small-mlx")
    print(r["text"].strip())
if __name__ == "__main__":
    main()
