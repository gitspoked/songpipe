from pathlib import Path
from dataclasses import replace
from songpipe.roll import roll
from songpipe.compose import Chosen
from songpipe.engines.ace import build_ace_job

def test_build_ace_job_uses_preset_and_lyrics(tmp_path):
    r = roll(5)
    chosen = Chosen("T", "[Verse]\nla la\n[Chorus]\noh oh", "claude", "good", {})
    job = build_ace_job(r, chosen, tmp_path)
    assert job["lm"] == r.preset.lm == "acestep-5Hz-lm-1.7B"
    assert job["dit"] == "acestep-v15-turbo"
    assert job["params"]["caption"] == r.preset.style
    assert job["params"]["lyrics"] == chosen.lyrics
    assert job["params"]["bpm"] == r.bpm and job["params"]["seed"] == r.seed
    assert job["save_dir"] == str(tmp_path)

def test_build_ace_job_seed_override_and_caption(tmp_path):
    r = roll(5)
    r2 = replace(r, caption="desperate, racing country, urgent vocal")
    chosen = Chosen("T", "[Verse]\nla", "grok", "good", {})
    job = build_ace_job(r2, chosen, tmp_path, seed=999)
    assert job["params"]["seed"] == 999
    assert job["params"]["caption"] == "desperate, racing country, urgent vocal"
    # falls back to preset style when caption is None
    job0 = build_ace_job(r, chosen, tmp_path)
    assert job0["params"]["caption"] == r.preset.style
    assert job0["params"]["seed"] == r.seed

def test_generate_song_threads_floor_gb(monkeypatch, tmp_path):
    from songpipe.engines import ace
    from songpipe.safety import GuardedResult
    from songpipe.roll import roll
    from songpipe.compose import Chosen
    seen = {}
    wav = tmp_path / "o.wav"
    def fake_guarded(cmd, **k):
        seen.update(k)
        wav.write_bytes(b"x")
        return GuardedResult(0, str(wav) + "\n", "", False, False)
    monkeypatch.setattr(ace.safety, "run_guarded", fake_guarded)
    c = Chosen("T", "[Verse]\nx", "grok", "r", {})
    out = ace.generate_song(roll(0), c, tmp_path, duration=10, seed=1, floor_gb=9.5)
    assert seen["floor_gb"] == 9.5 and out == wav
