import songpipe.asr as asr
from songpipe.safety import GuardedResult

LYRICS = """[Verse]
Dust on the dashboard, miles on my mind
Chasing that white line, leaving it all behind
[Chorus]
Take me home, take me home tonight"""

# ── alignment ────────────────────────────────────────────────────────────────

def test_alignment_faithful_transcript_is_one():
    t = ("dust on the dashboard miles on my mind chasing that white line "
         "leaving it all behind take me home take me home tonight")
    assert asr.alignment(LYRICS, t) == 1.0

def test_alignment_gibberish_is_low():
    assert asr.alignment(LYRICS, "bah dah noo lelele wooo mmm nanana") < 0.2

def test_alignment_partial_melisma_passes_default_threshold():
    t = "dust on the dashboard miles my mind take me home tonight"
    assert asr.alignment(LYRICS, t) >= asr.DEFAULT_THRESHOLD

def test_alignment_extra_transcript_content_does_not_penalize():
    t = ("dust on the dashboard miles on my mind chasing that white line "
         "leaving it all behind take me home take me home tonight "
         "yeah yeah woo repeated chorus again take me home")
    assert asr.alignment(LYRICS, t) == 1.0

def test_alignment_strips_section_tags():
    assert asr.alignment("[Verse]\nhello world", "hello world") == 1.0

def test_alignment_case_and_punctuation_insensitive():
    assert asr.alignment("Don't stop, believin'!", "dont stop believin") == 1.0

def test_alignment_typographic_apostrophes_match_ascii():
    # LLM lyrics use U+2019; whisper emits ASCII. Both sides must normalize the same.
    assert asr.alignment("don’t stop believin’, you’re gonna make it",
                         "don't stop believin, you're gonna make it") == 1.0

def test_alignment_typographic_apostrophes_both_directions():
    # U+2018/U+2019/U+02BC on either side must not shatter contractions.
    assert asr.alignment("can't quit now", "can’t quit now") == 1.0
    assert asr.alignment("‘tis ainʼt over", "tis aint over") == 1.0

def test_alignment_empty_transcript_is_zero():
    assert asr.alignment(LYRICS, "") == 0.0

def test_alignment_empty_lyrics_is_one():
    assert asr.alignment("[Chorus]", "whatever") == 1.0

# ── select_take ──────────────────────────────────────────────────────────────

def test_select_take_vetoes_low_alignment():
    takes = [{"seed": 1, "score": 5.0, "align": 0.10},
             {"seed": 2, "score": 3.0, "align": 0.90}]
    best, all_vetoed = asr.select_take(takes, 0.35)
    assert best["seed"] == 2 and all_vetoed is False

def test_select_take_unchecked_take_passes():
    takes = [{"seed": 1, "score": 5.0, "align": None},
             {"seed": 2, "score": 3.0, "align": 0.90}]
    best, all_vetoed = asr.select_take(takes, 0.35)
    assert best["seed"] == 1 and all_vetoed is False

def test_select_take_all_vetoed_falls_back_to_max_align():
    takes = [{"seed": 1, "score": 5.0, "align": 0.05},
             {"seed": 2, "score": 3.0, "align": 0.20}]
    best, all_vetoed = asr.select_take(takes, 0.35)
    assert best["seed"] == 2 and all_vetoed is True

def test_select_take_acoustic_still_ranks_survivors():
    takes = [{"seed": 1, "score": 3.0, "align": 0.80},
             {"seed": 2, "score": 5.0, "align": 0.40}]
    best, _ = asr.select_take(takes, 0.35)
    assert best["seed"] == 2   # both eligible; higher acoustic score wins

# ── transcribe (fail-open) ───────────────────────────────────────────────────

def test_transcribe_returns_stripped_transcript(monkeypatch):
    monkeypatch.setattr(asr.safety, "run_guarded",
        lambda *a, **k: GuardedResult(0, "  hello world \n", "", False, False))
    assert asr.transcribe("x.wav") == "hello world"

def test_transcribe_failopen_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(asr.safety, "run_guarded",
        lambda *a, **k: GuardedResult(1, "", "boom", False, False))
    assert asr.transcribe("x.wav") is None

def test_transcribe_failopen_on_watchdog_kill(monkeypatch):
    monkeypatch.setattr(asr.safety, "run_guarded",
        lambda *a, **k: GuardedResult(-9, "", "", True, False))
    assert asr.transcribe("x.wav") is None

def test_transcribe_failopen_on_timeout(monkeypatch):
    monkeypatch.setattr(asr.safety, "run_guarded",
        lambda *a, **k: GuardedResult(-9, "", "", False, True))
    assert asr.transcribe("x.wav") is None

def test_transcribe_empty_output_returns_empty_string(monkeypatch):
    # rc==0 with empty stdout means whisper RAN and heard nothing — that is the
    # gibberish signal, not a tool failure. Must return "" (alignment scores 0.0),
    # not None (which would let the most unintelligible takes bypass the veto).
    monkeypatch.setattr(asr.safety, "run_guarded",
        lambda *a, **k: GuardedResult(0, "   \n", "", False, False))
    assert asr.transcribe("x.wav") == ""

def test_transcribe_empty_output_scores_zero_alignment(monkeypatch):
    monkeypatch.setattr(asr.safety, "run_guarded",
        lambda *a, **k: GuardedResult(0, "", "", False, False))
    transcript = asr.transcribe("x.wav")
    assert transcript is not None
    assert asr.alignment(LYRICS, transcript) == 0.0

def test_transcribe_failopen_on_exception(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no venv")
    monkeypatch.setattr(asr.safety, "run_guarded", boom)
    assert asr.transcribe("x.wav") is None

def test_transcribe_threads_floor_and_timeout(monkeypatch):
    seen = {}
    def fake(cmd, **k):
        seen.update(k, cmd=cmd)
        return GuardedResult(0, "words", "", False, False)
    monkeypatch.setattr(asr.safety, "run_guarded", fake)
    asr.transcribe("x.wav", floor_gb=7.0)
    assert seen["floor_gb"] == 7.0
    assert seen["timeout_s"] == asr.TRANSCRIBE_TIMEOUT_S
    assert str(asr._TRANSCRIBE) in seen["cmd"][1]
