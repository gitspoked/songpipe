from songpipe.roll import roll
from songpipe import critic

def test_parse_critique_extracts_and_clamps():
    txt = 'noise {"score": 12, "lyric_fixes": ["add imagery"], "param_fixes": {"bpm": 96}} tail'
    c = critic.parse_critique(txt)
    assert c["score"] == 10.0  # clamped to 0..10
    assert c["lyric_fixes"] == ["add imagery"]
    assert c["param_fixes"] == {"bpm": 96}

def test_parse_critique_handles_garbage():
    c = critic.parse_critique("not json at all")
    assert c["score"] == 0.0 and c["lyric_fixes"] == [] and c["param_fixes"] == {}

def test_critique_and_rewrite_prompts_include_inputs():
    cp = critic.critique_prompt("BRIEF", "DRAFTLYRICS", "THEMEXYZ")
    assert "DRAFTLYRICS" in cp and "THEMEXYZ" in cp
    rp = critic.rewrite_prompt("BRIEF", "DRAFTLYRICS", ["tighten chorus"], "THEMEXYZ")
    assert "DRAFTLYRICS" in rp and "tighten chorus" in rp

def test_apply_param_fixes_whitelists_in_range():
    r = roll(2)  # country: bpm range 80..100, keyscales include "G major"
    out = critic.apply_param_fixes(r, {"bpm": 95, "keyscale": "G major",
                                       "caption": "urgent, racing", "evil": "ignored"})
    assert out.bpm == 95 and out.keyscale == "G major" and out.caption == "urgent, racing"
    # out-of-range / unknown values are ignored
    out2 = critic.apply_param_fixes(r, {"bpm": 5000, "keyscale": "Z minor"})
    assert out2.bpm == r.bpm and out2.keyscale == r.keyscale

def test_qc_loop_stops_at_threshold():
    # critic returns 9 immediately -> no rewrite, iters=0
    calls = {"rewrite": 0}
    def crit(d): return {"score": 9.0, "lyric_fixes": [], "param_fixes": {"bpm": 90}}
    def rew(d, fixes): calls["rewrite"] += 1; return d + " X"
    out = critic.qc_loop("DRAFT", crit, rew, max_iters=2, threshold=8.0)
    assert out["score"] == 9.0 and out["iters"] == 0 and calls["rewrite"] == 0
    assert out["param_fixes"] == {"bpm": 90}

def test_qc_loop_iterates_then_caps():
    scores = iter([4.0, 6.0, 7.0])  # never reaches 8 -> caps at max_iters
    def crit(d): return {"score": next(scores), "lyric_fixes": ["fix"], "param_fixes": {}}
    def rew(d, fixes): return d + "+"
    out = critic.qc_loop("D", crit, rew, max_iters=2, threshold=8.0)
    assert out["iters"] == 2 and out["score"] == 7.0
    assert out["lyrics"] == "D++" and len(out["history"]) == 3
