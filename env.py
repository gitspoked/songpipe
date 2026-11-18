import os
from songpipe.paths import ENV_FILE

class EnvError(RuntimeError):
    pass

def load_env() -> dict:
    out = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def getenv(name: str) -> str:
    val = os.environ.get(name) or load_env().get(name)
    if not val:
        raise EnvError(f"{name} not set in the environment or {ENV_FILE}")
    return val
