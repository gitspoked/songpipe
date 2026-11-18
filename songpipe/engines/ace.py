import json
from pathlib import Path
from songpipe.roll import Roll
from songpipe.compose import Chosen
from songpipe import paths, safety

_RUNNER = Path(__file__).parent / "_ace_runner.py"

def build_ace_job(r: Roll, chosen: Chosen, save_dir: Path, duration: int = 180, seed: int = None) -> dict:
    seed = r.seed if seed is None else seed
    caption = (r.caption or r.preset.style)
    return {
        "ace_repo": str(paths.ACE_REPO),
        "checkpoint_dir": str(paths.ACE_CHECKPOINTS),
        "dit": "acestep-v15-turbo",
        "lm": r.preset.lm,
        "save_dir": str(save_dir),
        # 18GB soft ceiling: max observed active set is ~12.6GB, so this only
        # throttles the cache spike, never honest work (see _ace_runner.py)
        "mlx": {"cache_limit_gb": 0, "memory_limit_gb": 18},
        "params": {
            "caption": caption, "lyrics": chosen.lyrics,
            "bpm": r.bpm, "keyscale": r.keyscale, "timesignature": "4",
            "vocal_language": "en", "duration": duration,
            "inference_steps": r.preset.inference_steps,
            "guidance_scale": r.preset.guidance_scale, "seed": seed,
        },
    }

def generate_song(r: Roll, chosen: Chosen, save_dir: Path, duration: int = 180,
                  seed: int = None, floor_gb: float = safety.WATCHDOG_FLOOR_GB) -> Path:
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    job_file = save_dir / "_ace_job.json"
    job_file.write_text(json.dumps(build_ace_job(r, chosen, save_dir, duration, seed)))
    res = safety.run_guarded([str(paths.ACE_PY), str(_RUNNER), str(job_file)],
                             floor_gb=floor_gb)
    if res.killed_by_watchdog:
        raise safety.SafetyError(
            f"ACE-Step killed by memory watchdog (available RAM fell below "
            f"{floor_gb:.0f} GB) — render aborted to prevent a reboot. "
            f"Reduce --duration or free memory and retry.")
    if res.returncode != 0:
        raise RuntimeError(f"ACE-Step failed:\n{res.stderr[-2000:]}")
    return Path(res.stdout.strip().splitlines()[-1])
