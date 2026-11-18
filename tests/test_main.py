import sys
import pytest
import songpipe.__main__ as m

def test_cli_rejects_takes_below_one(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(m, "run", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", ["songpipe", "5", "--takes", "0"])
    with pytest.raises(SystemExit) as ei:
        m.main()
    assert ei.value.code == 2
    assert not called                              # rejected before any work
    assert "--takes" in capsys.readouterr().err

def test_cli_flags_thread_to_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(m, "run", lambda seed, **k: seen.update(seed=seed, **k))
    monkeypatch.setattr(sys, "argv",
        ["songpipe", "5", "--no-asr", "--asr-threshold", "0.5", "--watchdog-floor", "7.5"])
    m.main()
    assert seen["seed"] == 5
    assert seen["do_asr"] is False
    assert seen["asr_threshold"] == 0.5
    assert seen["watchdog_floor"] == 7.5
