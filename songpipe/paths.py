"""Machine-specific locations, resolved from SONGPIPE_* environment variables.

Defaults derive from SONGPIPE_WORKSPACE so one variable relocates the whole
layout; any individual path can still be overridden directly:

    SONGPIPE_WORKSPACE        root for output and .env   (default: current dir)
    SONGPIPE_ACE_REPO         ACE-Step checkout          (default: <ws>/ACE-Step)
    SONGPIPE_ACE_PY           python in the ACE venv     (default: <ace>/.venv/bin/python)
    SONGPIPE_ACE_CHECKPOINTS  ACE model checkpoints      (default: <ace>/checkpoints)
    SONGPIPE_ASR_PY           python of a venv with mlx-whisper (default: ACE python)
    SONGPIPE_SONGS_DIR        where finished songs land  (default: <ws>/songs)
    SONGPIPE_ENV_FILE         API-key env file           (default: <ws>/.env)
"""
import os
from pathlib import Path


def _p(var: str, default) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else Path(default)


WORKSPACE = _p("SONGPIPE_WORKSPACE", Path.cwd())
ACE_REPO = _p("SONGPIPE_ACE_REPO", WORKSPACE / "ACE-Step")
ACE_PY = _p("SONGPIPE_ACE_PY", ACE_REPO / ".venv" / "bin" / "python")
ACE_CHECKPOINTS = _p("SONGPIPE_ACE_CHECKPOINTS", ACE_REPO / "checkpoints")
ASR_PY = _p("SONGPIPE_ASR_PY", ACE_PY)
SONGS_DIR = _p("SONGPIPE_SONGS_DIR", WORKSPACE / "songs")
ENV_FILE = _p("SONGPIPE_ENV_FILE", WORKSPACE / ".env")


def _has_weights(d: Path) -> bool:
    # A usable LM checkpoint has actual tensor shards, not just an index json
    # (a partial download leaves only model.safetensors.index.json behind).
    return any(p.suffix == ".safetensors" and not p.name.endswith(".index.json")
               for p in d.iterdir())


def on_disk_lms(checkpoints: Path = None) -> set:
    """Names of fully-downloaded ACE LM checkpoints. Scans lazily, on call."""
    root = ACE_CHECKPOINTS if checkpoints is None else checkpoints
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("acestep-5Hz-lm") and _has_weights(p)}
