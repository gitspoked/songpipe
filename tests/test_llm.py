import pytest
from songpipe.providers import llm

@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch):
    # Hermetic: never read real keys from the environment or an .env file.
    monkeypatch.setattr(llm, "getenv", lambda name: "FAKE-KEY")

def test_anthropic_request_has_no_temperature():
    url, headers, body = llm.build_request(llm.ANTHROPIC, "sys", "hi", "claude-opus-4-8", 0.9, 1500)
    assert url == "https://api.anthropic.com/v1/messages"
    # auth is either x-api-key (standard key) or Bearer + oauth beta (oat token)
    assert ("x-api-key" in headers) or (headers.get("anthropic-beta") == "oauth-2025-04-20")
    assert headers["anthropic-version"] == "2023-06-01"
    assert "temperature" not in body and body["system"] == "sys"
    assert body["model"] == "claude-opus-4-8"
    assert body["messages"] == [{"role": "user", "content": "hi"}]

def test_anthropic_oauth_token_uses_bearer(monkeypatch):
    monkeypatch.setattr(llm, "getenv", lambda name: "sk-ant-oat01-FAKE" if name == "ANTHROPIC_API_KEY" else "x")
    _, headers, _ = llm.build_request(llm.ANTHROPIC, "s", "u", "claude-opus-4-8", 0.9, 100)
    assert headers["Authorization"] == "Bearer sk-ant-oat01-FAKE"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert "x-api-key" not in headers

def test_anthropic_standard_key_uses_xapikey(monkeypatch):
    monkeypatch.setattr(llm, "getenv", lambda name: "sk-ant-api03-FAKE" if name == "ANTHROPIC_API_KEY" else "x")
    _, headers, _ = llm.build_request(llm.ANTHROPIC, "s", "u", "claude-opus-4-8", 0.9, 100)
    assert headers["x-api-key"] == "sk-ant-api03-FAKE"
    assert "Authorization" not in headers

def test_xai_request_openai_shape():
    url, headers, body = llm.build_request(llm.XAI, "sys", "hi", "grok-4", 0.7, 800)
    assert url == "https://api.x.ai/v1/chat/completions"
    assert headers["Authorization"].startswith("Bearer ")
    assert body["temperature"] == 0.7
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}

def test_lmstudio_endpoint_local():
    url, headers, body = llm.build_request(llm.LMSTUDIO, "s", "u", "nousresearch/hermes-4-70b", 0.5, 500)
    assert url == "http://127.0.0.1:1234/v1/chat/completions"

def test_parse_anthropic_picks_text_block():
    data = {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}]}
    assert llm.parse_response(llm.ANTHROPIC, data) == "hello"

def test_parse_openai_shape():
    data = {"choices": [{"message": {"content": "world"}}]}
    assert llm.parse_response(llm.XAI, data) == "world"
    assert llm.parse_response(llm.LMSTUDIO, data) == "world"
