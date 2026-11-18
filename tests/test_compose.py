from songpipe.roll import roll
from songpipe.compose import brief, parse_winner, compose, Chosen

def test_brief_mentions_genre_and_tags():
    r = roll(1)
    b = brief(r)
    assert r.genre in b.lower() and "[verse]" in b.lower() and "[chorus]" in b.lower()

def test_brief_includes_theme_when_given():
    r = roll(1)
    b = brief(r, theme="Miles brought Lanna back")
    assert "Miles brought Lanna back" in b
    assert "[verse]" in b.lower() and "[chorus]" in b.lower()

def test_brief_no_theme_unchanged():
    r = roll(1)
    assert brief(r) == brief(r, theme=None)

def test_parse_winner_reads_index():
    assert parse_winner("WINNER: 2") == 1     # "2" -> second draft -> index 1
    assert parse_winner("I pick draft 1 because...") == 0
    assert parse_winner("they are equal") == 0  # default

def test_compose_qc_failure_returns_draft(monkeypatch):
    """When the critic call raises (e.g. Grok 429), compose falls back to score=0.0 draft."""
    r = roll(2)
    from songpipe import compose as C

    def fake_chat(provider, system, user, model, **kw):
        if system == C.critic.CRITIC_SYS:
            raise RuntimeError("Grok 429 rate limit")
        # Writers succeed normally
        if "grok" in model:
            return "TITLE: Grok Song\n[Verse]\ngrok draft"
        return "TITLE: Claude Song\n[Verse]\nclaude draft"

    monkeypatch.setattr(C.llm, "chat", fake_chat)
    chosen = compose(r, xai_model="grok-4.3", qc_iters=1)
    assert isinstance(chosen, Chosen), "compose must return a Chosen even when QC crashes"
    assert chosen.qc["score"] == 0.0, "fallback score must be 0.0"
    assert chosen.lyrics, "chosen must have lyrics from a draft"


def test_compose_uses_qc_score_to_pick_winner(monkeypatch):
    r = roll(2)
    # Stub the LLM so no network: writers return drafts, critic returns JSON, rewrite echoes.
    from songpipe import compose as C
    def fake_chat(provider, system, user, model, **kw):
        if system == C.critic.CRITIC_SYS:
            # grok draft scores higher than claude draft
            return '{"score": 9, "lyric_fixes": [], "param_fixes": {}}' if "grok" in user.lower() \
                   else '{"score": 5, "lyric_fixes": ["x"], "param_fixes": {}}'
        if "grok" in model:
            return "TITLE: Grok Song\n[Verse]\ngrok draft"
        return "TITLE: Claude Song\n[Verse]\nclaude draft"
    monkeypatch.setattr(C.llm, "chat", fake_chat)
    chosen = compose(r, xai_model="grok-4.3", theme="t", qc_iters=1)
    assert isinstance(chosen, Chosen)
    assert chosen.qc["score"] == 9 and "Grok" in chosen.title
