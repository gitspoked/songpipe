import json
import numpy as np, soundfile as sf
import pytest
from pathlib import Path
import songpipe.__main__ as m
from songpipe.compose import Chosen

def _write_sine(p):
    sr = 16000; t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
    sf.write(str(p), (0.5*np.sin(2*np.pi*220*t)).astype("float32"), sr)
    return p

def test_run_wires_all_stages(monkeypatch, tmp_path):
    calls = {}
    # Hermetic: stub the safety RAM reads so the test never depends on live memory.
    monkeypatch.setattr(m.safety, "available_ram_gb", lambda: 999.0)
    monkeypatch.setattr(m.safety, "big_hogs", lambda *a, **k: [])
    monkeypatch.setattr(m.paths, "SONGS_DIR", tmp_path)
    monkeypatch.setattr(m, "compose",
        lambda r, **k: Chosen("My Song", "[Verse]\nx", "grok", "best", {"claude": "a", "grok": "b"}))
    def fake_gen(r, c, d, duration=180, seed=None, floor_gb=None):
        calls.setdefault("seeds", []).append(seed)
        calls["gen_floor"] = floor_gb
        return _write_sine(Path(d) / f"take_{seed}.wav")
    def fake_master(song, spoken, out_mp3, **k):
        calls["master"] = (song, spoken); Path(out_mp3).write_bytes(b"z"); return Path(out_mp3)
    # Take seed=7 "sings" gibberish, seed=8 sings the lyrics ("x") — veto must flip the pick.
    def fake_transcribe(wav, **k):
        calls["asr_floor"] = k.get("floor_gb")
        return "bah dah noo" if "take_7" in str(wav) else "x"
    monkeypatch.setattr(m.asr, "transcribe", fake_transcribe)
    monkeypatch.setattr(m, "generate_song", fake_gen)
    monkeypatch.setattr(m, "master", fake_master)
    # Non-default floor: 7.5 != safety.WATCHDOG_FLOOR_GB, so a hard-coded default
    # anywhere in run()'s threading fails these assertions.
    out = m.run(7, xai_model="grok-4.3", duration=10, takes=2, watchdog_floor=7.5)
    assert out.exists() and out.suffix == ".mp3" and "my-song" in str(out)
    runjson = json.loads((out.parent / "run.json").read_text())
    assert runjson["winner"] == "grok" and runjson["genre"] in m.GENRES and runjson["seed"] == 7
    assert len(runjson["takes"]) == 2 and calls["seeds"] == [7, 8]
    assert "master" in calls                                  # master stage ran
    assert runjson["safety"]["granted_s"] == 10               # safety block populated correctly
    # ASR veto: identical acoustic scores, but take 7 failed alignment → 8 must win.
    assert runjson["takes"][0]["align"] == 0.0 and runjson["takes"][1]["align"] == 1.0
    assert runjson["chosen_seed"] == 8
    assert runjson["asr_threshold"] == m.asr.DEFAULT_THRESHOLD
    assert runjson["asr_all_failed"] is False
    assert runjson["safety"]["watchdog_floor_gb"] == 7.5
    # floor threaded to ALL guarded engines: ACE render, ASR transcription
    assert calls["gen_floor"] == 7.5
    assert calls["asr_floor"] == 7.5

def test_run_no_asr_skips_transcription(monkeypatch, tmp_path):
    monkeypatch.setattr(m.safety, "available_ram_gb", lambda: 999.0)
    monkeypatch.setattr(m.safety, "big_hogs", lambda *a, **k: [])
    monkeypatch.setattr(m.paths, "SONGS_DIR", tmp_path)
    monkeypatch.setattr(m, "compose",
        lambda r, **k: Chosen("My Song", "[Verse]\nx", "grok", "best", {}))
    def fake_gen(r, c, d, duration=180, seed=None, floor_gb=None):
        p = Path(d) / f"take_{seed}.wav"
        sr = 16000; t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
        sf.write(str(p), (0.5*np.sin(2*np.pi*220*t)).astype("float32"), sr)
        return p
    def _no_transcribe(*a, **k):
        raise AssertionError("transcribe must not be called with do_asr=False")
    monkeypatch.setattr(m.asr, "transcribe", _no_transcribe)
    monkeypatch.setattr(m, "generate_song", fake_gen)
    monkeypatch.setattr(m, "master",
        lambda song, spoken, out_mp3, **k: (Path(out_mp3).write_bytes(b"z"), Path(out_mp3))[1])
    out = m.run(7, xai_model="grok-4.3", duration=10, takes=1, do_asr=False)
    runjson = json.loads((out.parent / "run.json").read_text())
    assert runjson["takes"][0]["align"] is None
    assert runjson["asr_threshold"] is None

def _hermetic_stubs(monkeypatch, tmp_path, lyrics="[Verse]\nx"):
    monkeypatch.setattr(m.safety, "available_ram_gb", lambda: 999.0)
    monkeypatch.setattr(m.safety, "big_hogs", lambda *a, **k: [])
    monkeypatch.setattr(m.paths, "SONGS_DIR", tmp_path)
    monkeypatch.setattr(m, "compose",
        lambda r, **k: Chosen("My Song", lyrics, "grok", "best", {}))
    monkeypatch.setattr(m, "master",
        lambda song, spoken, out_mp3, **k: (Path(out_mp3).write_bytes(b"z"), Path(out_mp3))[1])

def test_run_watchdog_killed_take_continues_with_survivors(monkeypatch, tmp_path, capsys):
    # Spec: "ACE take killed mid-render → that take is lost; if other takes
    # rendered, continue with them" — a mid-loop watchdog kill must not abort the run.
    _hermetic_stubs(monkeypatch, tmp_path)
    def fake_gen(r, c, d, duration=180, seed=None, floor_gb=None):
        if seed == 8:
            raise m.safety.SafetyError("ACE-Step killed by memory watchdog")
        return _write_sine(Path(d) / f"take_{seed}.wav")
    monkeypatch.setattr(m, "generate_song", fake_gen)
    monkeypatch.setattr(m.asr, "transcribe", lambda wav, **k: "x")
    out = m.run(7, xai_model="grok-4.3", duration=10, takes=3)
    assert out.exists()                                       # run completed with survivors
    runjson = json.loads((out.parent / "run.json").read_text())
    assert [t["seed"] for t in runjson["takes"]] == [7, 9]    # take 8 lost, 7 and 9 kept
    assert runjson["chosen_seed"] in (7, 9)
    assert "seed=8" in capsys.readouterr().out                # loss was announced

def test_run_all_takes_killed_raises_safety_error(monkeypatch, tmp_path):
    # Spec: "...else SafetyError" — nothing rendered means there is nothing to ship.
    _hermetic_stubs(monkeypatch, tmp_path)
    def fake_gen(r, c, d, duration=180, seed=None, floor_gb=None):
        raise m.safety.SafetyError("ACE-Step killed by memory watchdog")
    monkeypatch.setattr(m, "generate_song", fake_gen)
    monkeypatch.setattr(m.asr, "transcribe", lambda wav, **k: "x")
    with pytest.raises(m.safety.SafetyError):
        m.run(7, xai_model="grok-4.3", duration=10, takes=2)

def test_run_all_vetoed_falls_back_to_least_bad(monkeypatch, tmp_path, capsys):
    # End-to-end all-vetoed path: WARN printed, asr_all_failed recorded, the
    # max-align take is shipped, and an mp3 is still produced.
    _hermetic_stubs(monkeypatch, tmp_path, lyrics="[Verse]\nalpha beta gamma delta")
    def fake_gen(r, c, d, duration=180, seed=None, floor_gb=None):
        return _write_sine(Path(d) / f"take_{seed}.wav")
    monkeypatch.setattr(m, "generate_song", fake_gen)
    # take 7 → align 0.0, take 8 → align 0.25; both below the 0.35 threshold.
    monkeypatch.setattr(m.asr, "transcribe",
        lambda wav, **k: "zzz" if "take_7" in str(wav) else "alpha")
    out = m.run(7, xai_model="grok-4.3", duration=10, takes=2)
    assert out.exists() and out.suffix == ".mp3"              # still ships an mp3
    runjson = json.loads((out.parent / "run.json").read_text())
    assert runjson["asr_all_failed"] is True
    assert runjson["chosen_seed"] == 8                        # least-bad = max align
    assert "ALL takes below alignment" in capsys.readouterr().out

def test_run_empty_transcript_take_is_vetoed(monkeypatch, tmp_path):
    # A take whisper hears NOTHING in must score align 0.0 and lose to an aligned
    # take, even when acoustically identical (the fail-open loophole, closed).
    _hermetic_stubs(monkeypatch, tmp_path)
    def fake_gen(r, c, d, duration=180, seed=None, floor_gb=None):
        return _write_sine(Path(d) / f"take_{seed}.wav")
    monkeypatch.setattr(m, "generate_song", fake_gen)
    monkeypatch.setattr(m.asr, "transcribe",
        lambda wav, **k: "" if "take_7" in str(wav) else "x")
    out = m.run(7, xai_model="grok-4.3", duration=10, takes=2)
    runjson = json.loads((out.parent / "run.json").read_text())
    assert runjson["takes"][0]["align"] == 0.0                # empty ⇒ 0.0, not unchecked
    assert runjson["chosen_seed"] == 8

def test_run_rejects_nonpositive_takes(monkeypatch, tmp_path):
    # Must fail BEFORE any LLM spend (compose) — not traceback in select_take after.
    def no_compose(*a, **k):
        raise AssertionError("compose must not be called for takes < 1")
    monkeypatch.setattr(m, "compose", no_compose)
    monkeypatch.setattr(m.safety, "available_ram_gb", lambda: 999.0)
    monkeypatch.setattr(m.safety, "big_hogs", lambda *a, **k: [])
    monkeypatch.setattr(m.paths, "SONGS_DIR", tmp_path)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            m.run(7, xai_model="grok-4.3", duration=10, takes=bad)
