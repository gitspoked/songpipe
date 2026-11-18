import subprocess as sp
import numpy as np, soundfile as sf
import pytest
from pathlib import Path
from songpipe import mix
from songpipe.mix import overlay, master, resample, duck_envelope, apply_fades, loudness_ok

def _sine(sr, secs, f=220.0):
    t = np.linspace(0, secs, int(sr*secs), endpoint=False)
    return np.column_stack([0.5*np.sin(2*np.pi*f*t)]*2).astype("float32")

def test_overlay_at_offset():
    sr = 16000; song = _sine(sr, 2.0); spk = _sine(sr, 0.5, 440.0)
    out = overlay(song, spk, sr, 1.0, 0.0)
    assert out.shape == song.shape
    assert np.allclose(out[:sr], song[:sr])
    assert not np.allclose(out[sr:sr+8000], song[sr:sr+8000])

def test_master_writes_mp3(tmp_path):
    sr = 16000
    sw = tmp_path/"song.wav"; sf.write(sw, _sine(sr,1.0), sr)
    pw = tmp_path/"spk.wav";  sf.write(pw, _sine(sr,0.3,440.0), sr)
    out = master(sw, pw, tmp_path/"final.mp3")
    assert out.exists() and out.suffix == ".mp3" and out.stat().st_size > 0

def test_master_no_spoken(tmp_path):
    sr = 16000; sw = tmp_path/"song.wav"; sf.write(sw, _sine(sr,1.0), sr)
    assert master(sw, None, tmp_path/"final.mp3").exists()

def test_resample_changes_length():
    x = _sine(48000, 0.5)
    y = resample(x, 48000, 44100)
    assert abs(y.shape[0] - int(round(x.shape[0]*44100/48000))) <= 1
    assert y.shape[1] == 2

def test_master_handles_sr_mismatch(tmp_path):
    # ACE outputs 48k, Dia 44.1k — master must resample, not crash.
    sw = tmp_path/"song.wav"; sf.write(sw, _sine(48000, 1.0), 48000)
    pw = tmp_path/"spk.wav";  sf.write(pw, _sine(44100, 0.3, 440.0), 44100)
    out = master(sw, pw, tmp_path/"final.mp3")
    assert out.exists() and out.stat().st_size > 0

def test_duck_envelope_attenuates_window_only():
    sr = 16000; n = sr  # 1s
    env = duck_envelope(n, start=8000, end=12000, sr=sr, duck_db=-6.0, ramp_s=0.05)
    g = 10.0 ** (-6.0 / 20.0)
    assert env.shape == (n,)
    assert np.isclose(env[10000], g)          # deep inside the duck
    assert np.isclose(env[0], 1.0)            # well before
    assert np.isclose(env[-1], 1.0)           # well after
    assert env.min() >= g - 1e-6 and env.max() <= 1.0 + 1e-6

def test_duck_envelope_zero_db_is_noop():
    env = duck_envelope(1000, 100, 200, 16000, duck_db=0.0)
    assert np.allclose(env, 1.0)

def test_apply_fades_zeros_edges_keeps_middle():
    sr = 16000; x = np.ones(sr, dtype="float32")
    y = apply_fades(x, sr, fade_s=0.1)
    assert np.isclose(y[0], 0.0) and np.isclose(y[-1], 0.0)
    assert np.isclose(y[sr // 2], 1.0)

def test_apply_fades_stereo():
    sr = 16000; x = np.ones((sr, 2), dtype="float32")
    y = apply_fades(x, sr, fade_s=0.1)
    assert np.allclose(y[0], 0.0) and np.allclose(y[-1], 0.0)

def test_loudness_ok_tolerance():
    assert loudness_ok(-14.0) is True
    assert loudness_ok(-15.2, tol=1.5) is True
    assert loudness_ok(-10.0, tol=1.5) is False

def _sine_wav(path, seconds=0.2, sr=8000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path

def test_master_passes_timeout_to_ffmpeg(monkeypatch, tmp_path):
    # Record EVERY ffmpeg call's timeout — master() runs two (loudnorm encode +
    # ebur128 probe); asserting only the last would let the encode lose its timeout.
    timeouts = []
    def fake_run(cmd, **k):
        timeouts.append(k.get("timeout"))
        return sp.CompletedProcess(cmd, 0, "", "")   # master never reads the mp3 back
    monkeypatch.setattr(mix.subprocess, "run", fake_run)
    song = _sine_wav(tmp_path / "song.wav")
    mix.master(song, None, tmp_path / "out.mp3")
    assert len(timeouts) == 2                        # loudnorm encode + ebur128 probe
    assert all(t == mix.FFMPEG_TIMEOUT_S for t in timeouts)

def test_master_raises_on_ffmpeg_timeout(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise sp.TimeoutExpired(cmd="ffmpeg", timeout=mix.FFMPEG_TIMEOUT_S)
    monkeypatch.setattr(mix.subprocess, "run", boom)
    song = _sine_wav(tmp_path / "song.wav")
    with pytest.raises(sp.TimeoutExpired):
        mix.master(song, None, tmp_path / "out.mp3")

def test_measure_loudness_timeout_returns_none(monkeypatch):
    def boom(*a, **k):
        raise sp.TimeoutExpired(cmd="ffmpeg", timeout=mix.FFMPEG_TIMEOUT_S)
    monkeypatch.setattr(mix.subprocess, "run", boom)
    assert mix.measure_loudness("whatever.mp3") is None
