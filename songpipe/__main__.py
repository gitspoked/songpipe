import argparse, json, re, sys
from dataclasses import replace
from pathlib import Path
import soundfile as sf
from songpipe import paths
from songpipe.genres import GENRES
from songpipe.roll import roll
from songpipe.compose import compose, Chosen
from songpipe import safety, score, critic, asr
from songpipe.engines.ace import generate_song
from songpipe.mix import master

def _slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-") or "song"

def _unique_dir(base: Path) -> Path:
    # LLMs reuse titles ("SUDO SANDWICH" twice in one night); landing a run in
    # a non-empty dir silently destroys the previous song — never overwrite.
    if not base.exists() or not any(base.iterdir()):
        return base
    k = 2
    while True:
        d = base.with_name(f"{base.name}-{k}")
        if not d.exists() or not any(d.iterdir()):
            return d
        k += 1

def load_lyrics(path, title=None):
    """Read a lyric sheet; honor a 'TITLE: ...' first line, else fall back to
    the supplied --title, else the filename."""
    text = Path(path).read_text().strip()
    m = re.search(r"TITLE:\s*(.+)", text)
    if m:
        if not title:
            title = m.group(1).strip()
        text = re.sub(r".*TITLE:.*\n?", "", text, count=1).strip()
    if not title:
        title = Path(path).stem.replace("-", " ").replace("_", " ").title()
    return title, text

def run(seed, *, xai_model, duration=180, theme=None, force=False, max_mem_frac=None,
        takes=3, steps=None, qc_iters=2, qc_threshold=8.0, do_qc=True,
        asr_threshold=asr.DEFAULT_THRESHOLD, do_asr=True,
        watchdog_floor=safety.WATCHDOG_FLOOR_GB, lyrics_file=None, title=None):
    if takes < 1:  # fail here, before any paid LLM calls — not in select_take after
        raise ValueError(f"takes must be >= 1 (got {takes})")
    r = roll(seed)
    print(f"[songpipe] genre={r.genre} bpm={r.bpm} key={r.keyscale} seed={seed}")
    if theme:
        print(f"[songpipe] theme: {theme}")
    for name, gb in safety.big_hogs():
        print(f"[safety] {gb:.1f} GB held by {name} — quit it to free headroom")
    pf = safety.preflight(duration, force=force, safety_fraction=max_mem_frac)
    print(f"[safety] available={pf.available_gb:.1f}GB budget={pf.budget_gb:.1f}GB "
          f"est_peak({duration}s)={safety.estimate_peak_gb(duration):.1f}GB")
    if pf.forced:
        print(f"[safety] --force: running {duration}s anyway — REBOOT RISK if memory runs out")
    elif pf.clamped:
        print(f"[safety] clamped {pf.requested}s → {pf.granted}s to stay within budget")
    duration = pf.granted

    if lyrics_file:
        # Sing EXISTING lyrics: skip compose/QC entirely. The ASR veto now
        # doubles as a fidelity check against the supplied words.
        t, lyr = load_lyrics(lyrics_file, title)
        chosen = Chosen(title=t, lyrics=lyr, winner="user",
                        reason="lyrics supplied via --lyrics-file", drafts={}, qc=None)
        print(f"[songpipe] singing supplied lyrics: {chosen.title!r} "
              f"({len(lyr.splitlines())} lines)")
    else:
        chosen = compose(r, xai_model=xai_model, theme=theme, qc_iters=qc_iters,
                         qc_threshold=qc_threshold, do_qc=do_qc)
        print(f"[songpipe] lyrics by {chosen.winner}: {chosen.title!r} ({chosen.reason})")
    r = critic.apply_param_fixes(r, (chosen.qc or {}).get("param_fixes", {}))
    if steps is not None:
        r = replace(r, preset=replace(r.preset, inference_steps=steps))

    out_dir = _unique_dir(Path(paths.SONGS_DIR) / _slug(chosen.title))
    if out_dir.name != _slug(chosen.title):
        print(f"[songpipe] slug taken — writing to {out_dir.name}/ instead")
    out_dir.mkdir(parents=True, exist_ok=True)

    take_meta = []
    watchdog_err = None
    for k in range(takes):
        s = seed + k
        try:
            wav = generate_song(r, chosen, out_dir, duration=duration, seed=s,
                                floor_gb=watchdog_floor)
        except safety.SafetyError as e:
            # Spec: a watchdog-killed take is lost; continue with the takes that
            # rendered (else SafetyError below). Never discard finished GPU work.
            watchdog_err = e
            print(f"[songpipe] WARN: take seed={s} lost ({e}) — "
                  f"continuing with remaining takes")
            continue
        x, sr = sf.read(str(wav), always_2d=True)
        sc = score.combine_scores(score.clip_ratio(x), score.silence_ratio(x, sr),
                                  score.loudness_range(x))
        transcript = asr.transcribe(wav, floor_gb=watchdog_floor) if do_asr else None
        align = None if transcript is None else asr.alignment(chosen.lyrics, transcript)
        take_meta.append({"seed": s, "wav": str(wav), "score": sc,
                          "align": align, "transcript": transcript})
        a_str = "-" if align is None else f"{align:.2f}"
        print(f"[songpipe] take seed={s} score={sc:.3f} align={a_str}")
    if not take_meta:
        raise watchdog_err if watchdog_err else safety.SafetyError("no takes rendered")
    best, asr_all_failed = asr.select_take(take_meta, asr_threshold)
    if asr_all_failed:
        b_str = "-" if best["align"] is None else f"{best['align']:.2f}"
        print(f"[songpipe] WARN asr: ALL takes below alignment {asr_threshold:.2f} — "
              f"using least-bad (seed={best['seed']}, align={b_str}); "
              f"vocals may be gibberish")
    song_wav = Path(best["wav"])
    print(f"[songpipe] best take seed={best['seed']} → {song_wav.name}")

    final = out_dir / f"{_slug(chosen.title)}.mp3"
    master(song_wav, None, final)
    (out_dir / "run.json").write_text(json.dumps({
        "seed": seed, "genre": r.genre, "bpm": r.bpm, "keyscale": r.keyscale,
        "caption": r.caption, "title": chosen.title, "winner": chosen.winner,
        "reason": chosen.reason, "drafts": chosen.drafts, "xai_model": xai_model,
        "theme": theme, "qc": chosen.qc,
        "takes": take_meta, "chosen_seed": best["seed"],
        "asr_threshold": asr_threshold if do_asr else None,
        "asr_all_failed": asr_all_failed,
        "safety": {"requested_s": pf.requested, "granted_s": pf.granted,
                   "available_gb": round(pf.available_gb, 1),
                   "budget_gb": round(pf.budget_gb, 1),
                   "clamped": pf.clamped, "forced": pf.forced,
                   "watchdog_floor_gb": watchdog_floor},
    }, indent=2))
    print(f"[songpipe] DONE: {final}")
    return final

def main():
    ap = argparse.ArgumentParser(prog="songpipe")
    ap.add_argument("seed", nargs="?", type=int, default=42)
    ap.add_argument("--xai-model", default="grok-4.3")
    ap.add_argument("--duration", type=int, default=180)
    ap.add_argument("--theme", default=None, help="Story/subject the lyrics must tell")
    ap.add_argument("--lyrics-file", default=None,
                    help="sing this lyric sheet verbatim (skips compose/QC)")
    ap.add_argument("--title", default=None,
                    help="song title when using --lyrics-file (else TITLE: line or filename)")
    ap.add_argument("--force", action="store_true",
                    help="bypass the memory clamp/ceiling (REBOOT RISK; watchdog still runs)")
    ap.add_argument("--max-mem-frac", type=float, default=None,
                    help="share of available RAM the estimate may use (default 0.55)")
    ap.add_argument("--takes", type=int, default=3, help="best-of-N renders, pick the healthiest")
    ap.add_argument("--steps", type=int, default=None, help="override ACE inference_steps")
    ap.add_argument("--qc-iters", type=int, default=2, help="max Grok critic rewrite passes")
    ap.add_argument("--qc-threshold", type=float, default=8.0, help="QC score that stops the loop")
    ap.add_argument("--no-qc", action="store_true", help="disable the Grok QC loop")
    ap.add_argument("--asr-threshold", type=float, default=asr.DEFAULT_THRESHOLD,
                    help="min whisper word-recall vs lyrics; takes below are vetoed")
    ap.add_argument("--no-asr", action="store_true",
                    help="skip the ASR lyric-alignment check")
    ap.add_argument("--watchdog-floor", type=float, default=safety.WATCHDOG_FLOOR_GB,
                    help="SIGKILL guarded tools if available RAM (GB) drops below this")
    a = ap.parse_args()
    if a.takes < 1:
        ap.error("--takes must be >= 1")
    try:
        run(a.seed, xai_model=a.xai_model, duration=a.duration, theme=a.theme,
            force=a.force, max_mem_frac=a.max_mem_frac, takes=a.takes, steps=a.steps,
            qc_iters=a.qc_iters, qc_threshold=a.qc_threshold, do_qc=not a.no_qc,
            asr_threshold=a.asr_threshold, do_asr=not a.no_asr,
            watchdog_floor=a.watchdog_floor, lyrics_file=a.lyrics_file, title=a.title)
    except safety.SafetyError as e:
        print(f"[safety] REFUSED: {e}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
