"""ASR lyric-alignment veto for best-of-N takes.

Whisper transcripts of sung takes are compared to the intended lyrics by unique-word
RECALL. This is a FLOOR (veto gibberish), never a rank signal — word overlap can't rank
good singing (melisma/timing), which is why score.py stays wav-only. Recall (not WER)
because whisper hearing extra content (ad-libs, repeated choruses) must not penalize;
only MISSING lyric words indicate gibberish.

Fail-open: a broken/killed transcription returns None and the take passes unchecked —
ASR protects the output, not the machine (contrast safety.py: fail-closed). But a
SUCCESSFUL run that hears nothing returns "" (alignment 0.0): an empty transcript of
non-empty lyrics is the gibberish signal itself, not a tool failure.
"""
import re
from pathlib import Path
from songpipe import paths, safety

DEFAULT_THRESHOLD = 0.35       # loose on purpose: whisper mangles good singing routinely
TRANSCRIBE_TIMEOUT_S = 120.0   # whisper-small does a 30s clip in seconds; margin = model load
_TRANSCRIBE = Path(__file__).parent / "engines" / "_transcribe.py"
_TAG = re.compile(r"\[[^\]]*\]")
# ASCII ' plus typographic ‘ ’ (U+2018/U+2019, routine in LLM lyrics) and modifier
# letter apostrophe (U+02BC) — all must join contractions, or "don’t" shatters into
# {don, t} and can never match whisper's ASCII "dont".
_APOS = re.compile("['‘’ʼ]")


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", _APOS.sub("", _TAG.sub(" ", text.lower())))


def alignment(lyrics: str, transcript: str) -> float:
    """Fraction of unique lyric words heard in the transcript (0.0–1.0)."""
    lyric_words = set(_words(lyrics))
    if not lyric_words:
        return 1.0
    heard = set(_words(transcript))
    return len(lyric_words & heard) / len(lyric_words)


def select_take(takes, threshold):
    """Best acoustic score among ASR-eligible takes; (take, all_vetoed).

    Eligible = align is None (unchecked ⇒ pass) or align >= threshold.
    All vetoed ⇒ fall back to the highest-alignment take (score breaks ties).
    """
    eligible = [t for t in takes if t.get("align") is None or t["align"] >= threshold]
    if eligible:
        return max(eligible, key=lambda t: t["score"]), False
    return max(takes, key=lambda t: (t["align"], t["score"])), True


def transcribe(wav, *, floor_gb: float = safety.WATCHDOG_FLOOR_GB):
    """Whisper transcript of `wav` via the ASR python (paths.ASR_PY), or None
    (= take passes unchecked).

    A clean exit with empty stdout is NOT a failure — whisper ran and heard no words
    (its no-speech filters dropped everything). Return "" so alignment scores the take
    0.0: the most unintelligible takes must not slip past the veto via fail-open.
    (_transcribe.py exits nonzero on any exception, so rc==0 + empty stdout
    uniquely means "heard nothing", never "tool broke".)
    """
    try:
        res = safety.run_guarded([str(paths.ASR_PY), str(_TRANSCRIBE), str(wav)],
                                 floor_gb=floor_gb, timeout_s=TRANSCRIBE_TIMEOUT_S)
    except Exception as e:
        print(f"[songpipe] WARN asr: transcription errored ({e}) — take passes unchecked")
        return None
    if res.killed_by_watchdog or res.timed_out or res.returncode != 0:
        why = ("watchdog kill" if res.killed_by_watchdog else
               "timeout" if res.timed_out else f"exit {res.returncode}")
        print(f"[songpipe] WARN asr: {why} — take passes unchecked")
        return None
    transcript = res.stdout.strip()
    if not transcript:
        print("[songpipe] WARN asr: empty transcript — whisper heard nothing (alignment 0.0)")
    return transcript
