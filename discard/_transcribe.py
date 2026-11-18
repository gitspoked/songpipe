"""Runs INSIDE Dia .venv. Transcribes an audio file, prints transcript."""
import sys, mlx_whisper
def main():
    r = mlx_whisper.transcribe(sys.argv[1], path_or_hf_repo="mlx-community/whisper-small-mlx")
    print(r["text"].strip())
if __name__ == "__main__":
    main()
