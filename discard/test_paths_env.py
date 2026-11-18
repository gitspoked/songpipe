import pytest
from songpipe import paths
from songpipe.env import load_env, getenv, EnvError

def test_engine_pythons_exist():
    assert paths.ACE_PY.exists()
    assert paths.DIA_PY.exists()

def test_on_disk_lms():
    # 1.7B is fully downloaded; 0.6B isn't present; 4B is a partial download
    # (index json only, no shards) so the completeness check must exclude it.
    assert "acestep-5Hz-lm-1.7B" in paths.ON_DISK_LMS
    assert "acestep-5Hz-lm-0.6B" not in paths.ON_DISK_LMS
    assert "acestep-5Hz-lm-4B" not in paths.ON_DISK_LMS

def test_env_loads_keys():
    env = load_env()
    assert "ANTHROPIC_API_KEY" in env and "XAI_API_KEY" in env and "LM_STUDIO_API_KEY" in env

def test_getenv_missing_raises():
    with pytest.raises(EnvError) as e:
        getenv("NOPE_NOT_A_KEY")
    assert str(paths.ENV_FILE) in str(e.value)
