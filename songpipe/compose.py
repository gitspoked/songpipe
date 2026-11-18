import re
from dataclasses import dataclass
from songpipe.roll import Roll
from songpipe.providers import llm
from songpipe import critic

@dataclass
class Chosen:
    title: str
    lyrics: str
    winner: str
    reason: str
    drafts: dict
    qc: dict = None

_SYS = "You are a hit songwriter. Write vivid, singable, original lyrics. Output the title on the first line as 'TITLE: ...', then the lyrics using [Verse]/[Chorus]/[Bridge] section tags. No commentary."

def brief(r: Roll, theme: str = None) -> str:
    subject = (f"Tell this true story, faithfully and vividly: {theme}. "
               if theme else "Surprise me. ")
    return (f"Write {r.genre} lyrics for a song. {subject}"
            f"Style: {r.preset.style}. "
            f"Tempo about {r.bpm} BPM, key {r.keyscale}. "
            f"Use [Verse] and [Chorus] section tags. Keep it under 24 lines.")

def _split_title(text: str):
    m = re.search(r"TITLE:\s*(.+)", text)
    title = (m.group(1).strip() if m else "Untitled")
    body = re.sub(r".*TITLE:.*\n?", "", text, count=1).strip() if m else text.strip()
    return title, body

def parse_winner(text: str) -> int:
    m = re.search(r"\b([12])\b", text)
    return (int(m.group(1)) - 1) if m else 0

def _writer_draft(provider, b, model):
    return llm.chat(provider, _SYS, b, model, max_tokens=1200)

def compose(r: Roll, *, xai_model: str, theme: str = None, qc_iters: int = 2,
            qc_threshold: float = 8.0, do_qc: bool = True,
            claude_model: str = "claude-opus-4-8") -> Chosen:
    b = brief(r, theme=theme)
    drafts = {}
    try:
        drafts["claude"] = _writer_draft(llm.ANTHROPIC, b, claude_model)
    except Exception:
        drafts["claude"] = None
    try:
        drafts["grok"] = _writer_draft(llm.XAI, b, xai_model)
    except Exception:
        drafts["grok"] = None
    live = [(k, v) for k, v in drafts.items() if v]
    if not live:
        raise RuntimeError("both lyric writers failed")

    def make_critic():
        return lambda d: critic.parse_critique(
            llm.chat(llm.XAI, critic.CRITIC_SYS, critic.critique_prompt(b, d, theme),
                     xai_model, max_tokens=600))

    def make_rewrite():
        return lambda d, fixes: llm.chat(
            llm.XAI, _SYS, critic.rewrite_prompt(b, d, fixes, theme), xai_model, max_tokens=1200)

    results = []
    for k, text in live:
        if do_qc:
            try:
                qc = critic.qc_loop(text, make_critic(), make_rewrite(),
                                    max_iters=qc_iters, threshold=qc_threshold)
            except Exception:
                # #2: QC LLM failure (429, timeout, …) must not discard a good draft
                qc = {"lyrics": text, "score": 0.0, "param_fixes": {}, "history": [], "iters": 0}
        else:
            qc = {"lyrics": text, "score": 0.0, "param_fixes": {}, "history": [], "iters": 0}
        results.append((k, qc))

    win_key, win_qc = max(results, key=lambda kv: kv[1]["score"])
    title, lyrics = _split_title(win_qc["lyrics"])
    reason = f"QC score {win_qc['score']:.1f} after {win_qc['iters']} rewrite(s)"
    return Chosen(title=title, lyrics=lyrics, winner=win_key, reason=reason,
                  drafts=drafts, qc=win_qc)
