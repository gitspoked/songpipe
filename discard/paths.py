from pathlib import Path

WORKSPACE = Path("/Users/mfk/LocalMusic")
ACE_REPO = WORKSPACE / "DLMusic" / "ACE-Step-1.5-main"
ACE_PY = ACE_REPO / ".venv-ace" / "bin" / "python"
ACE_CHECKPOINTS = ACE_REPO / "checkpoints"
DIA_REPO = Path("/Users/mfk/LocalVoice/dia")
DIA_PY = DIA_REPO / ".venv" / "bin" / "python"
DIA_CONFIG = DIA_REPO / "model-weights" / "config.json"
DIA_CKPT = DIA_REPO / "model-weights" / "dia-v0_1.pth"
SONGS_DIR = WORKSPACE / "mkMusic" / "songs"
ENV_FILE = WORKSPACE / "mkMusic" / ".env"

def _has_weights(d: Path) -> bool:
    # A usable LM checkpoint has actual tensor shards, not just an index json.
    # (The 4B dir on disk has only model.safetensors.index.json — a partial download.)
    return any(p.suffix == ".safetensors" and not p.name.endswith(".index.json")
               for p in d.iterdir())

ON_DISK_LMS = {
    p.name for p in ACE_CHECKPOINTS.iterdir()
    if p.is_dir() and p.name.startswith("acestep-5Hz-lm") and _has_weights(p)
} if ACE_CHECKPOINTS.exists() else set()
