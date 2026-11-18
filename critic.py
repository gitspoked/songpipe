"""Grok QC loop: an external strict critic that scores lyrics + parameters
BEFORE the expensive render, so weak songs get fixed for ~free.

Grok is treated as a strong INPUT, not law (see spec). The loop is cross-role
(critic != the writer editing itself); output is forced JSON to avoid fragile
parsing. qc_loop takes injected critic/rewrite callables so it is unit-tested
with stubs (no live LLM).
"""
from __future__ import annotations
import json, re
from dataclasses import replace

CRITIC_SYS = ("You are a skeptical hit songwriter and producer. You CANNOT listen to audio; "
              "judge ONLY the lyrics and musical parameters. No flattery. Reply with a single "
              "JSON object and nothing else.")

def critique_prompt(brief: str, draft: str, theme: str = None) -> str:
    t = f"\nThe song must tell this story: {theme}\n" if theme else "\n"
    return (f"Brief: {brief}{t}"
            f"Lyrics draft:\n{draft}\n\n"
            'Return JSON: {"score": <0-10 overall>, '
            '"lyric_fixes": [<=3 concrete edits], '
            '"param_fixes": {"bpm": <int or omit>, "keyscale": "<or omit>", '
            '"caption": "<style words for the music model, or omit>"}}. '
            "Score 8+ only if imagery is concrete, the hook is sharp, and it sings cleanly.")

def rewrite_prompt(brief: str, draft: str, fixes, theme: str = None) -> str:
    t = f"\nStory to tell: {theme}\n" if theme else "\n"
    bullets = "\n".join(f"- {f}" for f in fixes)
    return (f"Brief: {brief}{t}"
            f"Revise these lyrics, addressing every fix. Keep the 'TITLE: ...' first line and "
            f"[Verse]/[Chorus] tags. Output only the improved lyrics.\n\n"
            f"Fixes:\n{bullets}\n\nLyrics:\n{draft}")

def parse_critique(text: str) -> dict:
    obj = {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}
    try:
        score = float(obj.get("score", 0))
    except Exception:
        score = 0.0
    score = max(0.0, min(10.0, score))
    fixes = obj.get("lyric_fixes") or []
    if not isinstance(fixes, list):
        fixes = [str(fixes)]
    fixes = [str(f) for f in fixes]
    pf = obj.get("param_fixes") or {}
    if not isinstance(pf, dict):
        pf = {}
    return {"score": score, "lyric_fixes": fixes, "param_fixes": pf}

def apply_param_fixes(roll, fixes: dict) -> Roll:
    p = roll.preset
    bpm, ks, cap = roll.bpm, roll.keyscale, roll.caption
    if "bpm" in fixes:
        try:
            b = int(fixes["bpm"])
            if p.bpm_lo <= b <= p.bpm_hi:
                bpm = b
        except Exception:
            pass
    if fixes.get("keyscale") in p.keyscales:
        ks = fixes["keyscale"]
    if fixes.get("caption"):
        cap = str(fixes["caption"])
    return replace(roll, bpm=bpm, keyscale=ks, caption=cap)

def qc_loop(draft, critic, rewrite, max_iters: int = 2, threshold: float = 8.0) -> dict:
    history = []
    current = draft
    for i in range(max_iters + 1):
        crit = critic(current)
        history.append({"score": crit["score"], "fixes": crit["lyric_fixes"]})
        if crit["score"] >= threshold or i == max_iters:
            return {"lyrics": current, "score": crit["score"],
                    "param_fixes": crit.get("param_fixes", {}),
                    "history": history, "iters": i}
        current = rewrite(current, crit["lyric_fixes"])
