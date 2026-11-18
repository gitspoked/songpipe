from pathlib import Path
from songpipe.roll import Roll
from songpipe.compose import Chosen
from songpipe import paths, safety

def _first_hook(lyrics: str) -> str:
    for line in lyrics.splitlines():
        s = line.strip()
        if s and not s.startswith("["):
            return s
    return "yeah"

def spoken_text(r: Roll, chosen: Chosen):
    policy = r.preset.dia
    if policy == "none":
        return None
    hook = _first_hook(chosen.lyrics)
    if policy == "adlib":
        return f"[S1] (hypes) {hook} (laughs) let's go."
    if policy == "bridge":
        return f"[S1] (sighs) {hook}"
    return None

def build_dia_cmd(text: str, out: Path, seed: int) -> list:
    return [
        str(paths.DIA_PY), "cli.py", text,
        "--local-paths", "--config", str(paths.DIA_CONFIG), "--checkpoint", str(paths.DIA_CKPT),
        "--device", "mps", "--seed", str(seed), "--max-tokens", "1200",
        "--output", str(out),
    ]

def speak(r: Roll, chosen: Chosen, out: Path, *, floor_gb: float = safety.WATCHDOG_FLOOR_GB):
    text = spoken_text(r, chosen)
    if text is None:
        return None
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    # Dia is MPS/Metal — same wired-memory class that kernel-panicked long ACE runs.
    res = safety.run_guarded(build_dia_cmd(text, out, r.seed),
                             cwd=str(paths.DIA_REPO), floor_gb=floor_gb)
    if res.killed_by_watchdog:
        print("[songpipe] WARN dia: killed by memory watchdog — skipping spoken layer")
        return None
    if res.returncode != 0:
        raise RuntimeError(f"Dia failed:\n{res.stderr[-2000:]}")
    return out
