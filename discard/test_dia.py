import pytest
from pathlib import Path
from songpipe.roll import roll
from songpipe.compose import Chosen
from songpipe.genres import PRESETS
from songpipe.engines.dia import spoken_text, build_dia_cmd
from songpipe import paths

C = Chosen("T", "[Verse]\nfirst hook line here\n[Chorus]\nbig chorus moment", "claude", "x", {})

def _roll_for(genre):
    s = 0
    while True:
        r = roll(s)
        if r.genre == genre:
            return r
        s += 1

def test_pop_skips_dia():
    assert spoken_text(_roll_for("pop"), C) is None

def test_rock_gets_adlib_text():
    t = spoken_text(_roll_for("rock"), C)
    assert t is not None and t.startswith("[S1]")

def test_build_dia_cmd_shape(tmp_path):
    cmd = build_dia_cmd("[S1] yeah", tmp_path / "s.wav", 7)
    assert cmd[0] == str(paths.DIA_PY) and "--device" in cmd and "mps" in cmd
    assert "--local-paths" in cmd and str(tmp_path / "s.wav") in cmd and "7" in cmd

from songpipe.engines import dia
from songpipe.safety import GuardedResult

def test_speak_skips_spoken_layer_on_watchdog_kill(monkeypatch, tmp_path):
    monkeypatch.setattr(dia.safety, "run_guarded",
        lambda *a, **k: GuardedResult(-9, "", "", True, False))
    assert dia.speak(_roll_for("rock"), C, tmp_path / "s.wav") is None

def test_speak_raises_on_plain_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(dia.safety, "run_guarded",
        lambda *a, **k: GuardedResult(1, "", "err", False, False))
    with pytest.raises(RuntimeError):
        dia.speak(_roll_for("rock"), C, tmp_path / "s.wav")

def test_speak_threads_cwd_and_floor(monkeypatch, tmp_path):
    seen = {}
    def fake(cmd, **k):
        seen.update(k, cmd=cmd)
        return GuardedResult(0, "", "", False, False)
    monkeypatch.setattr(dia.safety, "run_guarded", fake)
    out = dia.speak(_roll_for("rock"), C, tmp_path / "s.wav", floor_gb=7.0)
    assert out == tmp_path / "s.wav"
    assert seen["cwd"] == str(paths.DIA_REPO) and seen["floor_gb"] == 7.0
