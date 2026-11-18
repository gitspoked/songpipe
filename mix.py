import subprocess
from pathlib import Path
import numpy as np, soundfile as sf
import re as _re

FFMPEG_TIMEOUT_S = 300.0  # hang protection; ffmpeg's memory footprint is small

def resample(x, src_sr, dst_sr):
    # Linear-interp resample of a 2D (frames, channels) array. Adequate for a
    # spoken overlay; sources can disagree on rate (e.g. a 48k song vs a 44.1k overlay).
    if src_sr == dst_sr:
        return x
    n_dst = int(round(x.shape[0] * dst_sr / src_sr))
    t_src = np.arange(x.shape[0])
    t_dst = np.linspace(0, x.shape[0] - 1, n_dst)
    return np.stack([np.interp(t_dst, t_src, x[:, c]) for c in range(x.shape[1])], axis=1).astype("float32")

def duck_envelope(n, start, end, sr, duck_db=-6.0, ramp_s=0.05):
    env = np.ones(int(n), dtype="float32")
    start = max(0, int(start)); end = min(int(n), int(end))
    if end <= start or duck_db == 0.0:
        return env
    g = 10.0 ** (duck_db / 20.0)
    r = max(1, int(ramp_s * sr))
    env[start:end] = g
    a0 = max(0, start - r)
    if start > a0:
        env[a0:start] = np.linspace(1.0, g, start - a0, endpoint=False)
    b1 = min(int(n), end + r)
    if b1 > end:
        env[end:b1] = np.linspace(g, 1.0, b1 - end, endpoint=False)
    return env

def overlay(song, spoken, sr, offset_s, gain_db, duck_db=0.0, ramp_s=0.05):
    out = song.copy()
    gain = 10.0 ** (gain_db / 20.0)
    start = int(offset_s * sr)
    seg = spoken * gain
    end = min(start + len(seg), len(out))
    env = duck_envelope(len(out), start, end, sr, duck_db, ramp_s)
    env = env.reshape(-1, *([1] * (out.ndim - 1)))
    out = out * env
    out[start:end] += seg[: end - start]
    return out

def apply_fades(x, sr, fade_s=0.3):
    x = np.array(x, dtype="float32", copy=True)
    n = len(x); f = min(int(fade_s * sr), n // 2)
    if f <= 0:
        return x
    shape = (-1,) + (1,) * (x.ndim - 1)
    x[:f] *= np.linspace(0.0, 1.0, f).reshape(shape)
    x[-f:] *= np.linspace(1.0, 0.0, f).reshape(shape)
    return x

def loudness_ok(measured_i, target=-14.0, tol=1.5):
    return abs(measured_i - target) <= tol

def measure_loudness(path):
    try:
        out = subprocess.run(
            ["ffmpeg", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S).stderr
        hits = _re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", out)
        return float(hits[-1]) if hits else None
    except Exception:
        return None

def master(song_wav: Path, spoken_wav, out_mp3: Path, *, offset_s=0.0, gain_db=-4.0, duck_db=-6.0, fade_s=0.3) -> Path:
    song, sr = sf.read(str(song_wav), always_2d=True)
    if spoken_wav:
        spoken, ssr = sf.read(str(spoken_wav), always_2d=True)
        spoken = resample(spoken, ssr, sr)
        if spoken.shape[1] == 1 and song.shape[1] == 2:
            spoken = np.repeat(spoken, 2, axis=1)
        song = overlay(song, spoken, sr, offset_s, gain_db, duck_db=duck_db)
    song = apply_fades(song, sr, fade_s)
    out_mp3 = Path(out_mp3); out_mp3.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp3.with_suffix(".mixed.wav")
    sf.write(str(tmp), song, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-af", "loudnorm=I=-14:TP=-1.0:LRA=11",
         "-b:a", "320k", str(out_mp3)],
        check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    measured = measure_loudness(out_mp3)
    if measured is not None and not loudness_ok(measured):
        print(f"[mix] WARNING: integrated loudness {measured:.1f} LUFS off -14 target")
    return out_mp3
