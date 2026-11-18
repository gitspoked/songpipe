import importlib

import pytest

import songpipe.env as envmod
import songpipe.paths as paths
from songpipe.env import EnvError


@pytest.fixture(autouse=True)
def _restore_paths():
    # Reload-based tests mutate module globals; put them back afterwards.
    # (monkeypatch teardown runs first, so the env is clean by then.)
    yield
    importlib.reload(paths)


def _reload_with(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return importlib.reload(paths)


def test_workspace_relocates_whole_layout(monkeypatch, tmp_path):
    p = _reload_with(monkeypatch, SONGPIPE_WORKSPACE=tmp_path)
    assert p.WORKSPACE == tmp_path
    assert p.ACE_REPO == tmp_path / "ACE-Step"
    assert p.ACE_PY == tmp_path / "ACE-Step" / ".venv" / "bin" / "python"
    assert p.ACE_CHECKPOINTS == tmp_path / "ACE-Step" / "checkpoints"
    assert p.SONGS_DIR == tmp_path / "songs"
    assert p.ENV_FILE == tmp_path / ".env"


def test_specific_overrides_beat_derived_defaults(monkeypatch, tmp_path):
    p = _reload_with(monkeypatch, SONGPIPE_WORKSPACE=tmp_path,
                     SONGPIPE_ACE_PY=tmp_path / "custom" / "python",
                     SONGPIPE_SONGS_DIR=tmp_path / "out")
    assert p.ACE_PY == tmp_path / "custom" / "python"
    assert p.SONGS_DIR == tmp_path / "out"
    assert p.ASR_PY == p.ACE_PY  # ASR rides the ACE venv unless overridden


def test_asr_py_independent_override(monkeypatch, tmp_path):
    p = _reload_with(monkeypatch, SONGPIPE_ASR_PY=tmp_path / "whisper-venv" / "bin" / "python")
    assert p.ASR_PY == tmp_path / "whisper-venv" / "bin" / "python"


def test_on_disk_lms_requires_real_shards(tmp_path):
    full = tmp_path / "acestep-5Hz-lm-1.7B"
    full.mkdir()
    (full / "model.safetensors").write_bytes(b"x")
    partial = tmp_path / "acestep-5Hz-lm-4B"  # index json only = partial download
    partial.mkdir()
    (partial / "model.safetensors.index.json").write_text("{}")
    (tmp_path / "not-an-lm").mkdir()
    assert paths.on_disk_lms(tmp_path) == {"acestep-5Hz-lm-1.7B"}


def test_on_disk_lms_missing_dir_is_empty(tmp_path):
    assert paths.on_disk_lms(tmp_path / "nope") == set()


def test_load_env_parses_file(monkeypatch, tmp_path):
    f = tmp_path / ".env"
    f.write_text('A=1\n# comment\nB="two"\nnot a pair\n')
    monkeypatch.setattr(envmod, "ENV_FILE", f)
    assert envmod.load_env() == {"A": "1", "B": "two"}


def test_getenv_prefers_process_environment(monkeypatch, tmp_path):
    f = tmp_path / ".env"
    f.write_text("SOME_KEY=from-file\n")
    monkeypatch.setattr(envmod, "ENV_FILE", f)
    monkeypatch.setenv("SOME_KEY", "from-process")
    assert envmod.getenv("SOME_KEY") == "from-process"
    monkeypatch.delenv("SOME_KEY")
    assert envmod.getenv("SOME_KEY") == "from-file"


def test_getenv_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(envmod, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("NOPE_NOT_A_KEY", raising=False)
    with pytest.raises(EnvError):
        envmod.getenv("NOPE_NOT_A_KEY")
