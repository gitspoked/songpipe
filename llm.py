import httpx
from songpipe.env import getenv

ANTHROPIC, XAI, LMSTUDIO = "anthropic", "xai", "lmstudio"

def build_request(provider, system, user, model, temperature, max_tokens):
    if provider == ANTHROPIC:
        # opus-4-8 rejects temperature/top_p — omit them deliberately.
        url = "https://api.anthropic.com/v1/messages"
        key = getenv("ANTHROPIC_API_KEY")
        headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
        if key.startswith("sk-ant-oat"):
            # OAuth token (e.g. Claude Code login): Bearer auth + oauth beta header, not x-api-key.
            headers["Authorization"] = f"Bearer {key}"
            headers["anthropic-beta"] = "oauth-2025-04-20"
        else:
            headers["x-api-key"] = key
        body = {
            "model": model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        return url, headers, body
    # OpenAI-compatible (xAI, LM Studio)
    if provider == XAI:
        url = "https://api.x.ai/v1/chat/completions"
        key = getenv("XAI_API_KEY")
    elif provider == LMSTUDIO:
        url = "http://127.0.0.1:1234/v1/chat/completions"
        key = getenv("LM_STUDIO_API_KEY")
    else:
        raise ValueError(f"unknown provider {provider!r}")
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens,
    }
    return url, headers, body

def parse_response(provider, data):
    if provider == ANTHROPIC:
        return next(b["text"] for b in data["content"] if b.get("type") == "text")
    return data["choices"][0]["message"]["content"]

def chat(provider, system, user, model, temperature=0.9, max_tokens=1500):
    url, headers, body = build_request(provider, system, user, model, temperature, max_tokens)
    resp = httpx.post(url, headers=headers, json=body, timeout=120.0)
    resp.raise_for_status()
    return parse_response(provider, resp.json()).strip()
