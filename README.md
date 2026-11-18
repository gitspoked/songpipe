# songpipe

Seed number in, finished song out — entirely on your own machine (plus the LLM
APIs you point it at).

Give songpipe a seed and it rolls a genre, tempo, and key; has two LLM
lyricists (Claude and Grok) each write a draft; runs the winning draft through
a critic/rewrite QC loop; renders best-of-N takes with
[ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) on Apple-Silicon
MLX; vetoes takes whose vocals don't match the lyrics (whisper transcript
recall); and masters the best take to a −14 LUFS mp3 with ffmpeg.

```
seed ──▶ roll (genre/bpm/key) ──▶ compose (2 LLM drafts + QC loop)
                                        │
                                        ▼
             ACE-Step render × N takes (memory-guarded subprocess)
                                        │
                                        ▼
        acoustic scoring + ASR lyric-alignment veto picks the best take
                                        │
                                        ▼
                 ffmpeg master (−14 LUFS, fades) ──▶ songs/<slug>/<slug>.mp3
```

Or skip the lyricists entirely and sing your own words:

```sh
songpipe --lyrics-file examples/gravel-road-morning.txt --duration 90
```

## Requirements

- **macOS on Apple Silicon.** ACE-Step runs via MLX/Metal; the built-in memory
  safety layer reads `vm_stat` and is macOS-specific.
- **Python 3.12+** and **ffmpeg** (`brew install ffmpeg`).
- **An ACE-Step v1.5 checkout with its own venv and downloaded checkpoints.**
  songpipe does not bundle ACE-Step — it drives your existing install as a
  subprocess. Follow the
  [ACE-Step-1.5 setup instructions](https://github.com/ace-step/ACE-Step-1.5),
  and note that the model weights are distributed under their own license
  terms, separate from this project.
- **LLM API keys** for composing lyrics: `ANTHROPIC_API_KEY` and/or
  `XAI_API_KEY`. One missing lyricist is tolerated (the other writes solo);
  the QC critic uses xAI, so without `XAI_API_KEY` run with `--no-qc`.
  With `--lyrics-file` no keys are needed at all.

## Install

```sh
pip install songpipe        # or: pip install . from a checkout
```

## Configure

Everything machine-specific is an environment variable. Defaults derive from
`SONGPIPE_WORKSPACE`, so the minimal setup is:

```sh
export SONGPIPE_WORKSPACE=~/music        # songs land in ~/music/songs
export SONGPIPE_ACE_REPO=~/src/ACE-Step-1.5
```

| Variable | Meaning | Default |
|---|---|---|
| `SONGPIPE_WORKSPACE` | root for output and `.env` | current directory |
| `SONGPIPE_ACE_REPO` | ACE-Step checkout | `<workspace>/ACE-Step` |
| `SONGPIPE_ACE_PY` | python inside the ACE venv | `<ace_repo>/.venv/bin/python` |
| `SONGPIPE_ACE_CHECKPOINTS` | ACE model checkpoints | `<ace_repo>/checkpoints` |
| `SONGPIPE_ASR_PY` | python of a venv with `mlx-whisper` | the ACE python |
| `SONGPIPE_SONGS_DIR` | where finished songs land | `<workspace>/songs` |
| `SONGPIPE_ENV_FILE` | API-key env file | `<workspace>/.env` |

API keys are read from the process environment first, then from the env file
(`KEY=value` lines).

**ASR veto (optional but recommended):** `pip install mlx-whisper` into the
ACE venv (or any venv, pointed at by `SONGPIPE_ASR_PY`). If whisper isn't
available, transcription fails open — takes pass unchecked and you lose the
gibberish-vocal veto, nothing else.

## Use

```sh
songpipe 42 --duration 120 --theme "a mechanic who fixes hearts on the side"
songpipe 7 --takes 5 --qc-iters 3          # more takes, more QC rewrites
songpipe --lyrics-file my-song.txt --title "My Song"
songpipe 42 --no-asr --no-qc               # fastest, least filtered
```

Each run lands in its own directory — `songs/<title-slug>/` — with the
mastered mp3, every rendered take, and a `run.json` recording the roll, the
drafts, QC history, per-take scores/transcripts, and the safety decisions.
An existing non-empty directory is never overwritten; a colliding title gets
a `-2` suffix.

Lyric sheets are plain text, optionally starting with a `TITLE: ...` line;
`[Verse]`/`[Chorus]`/`[Bridge]` section tags are understood by ACE-Step.
Two originals to try live in `examples/`.

## The memory-safety layer (read before long renders)

MLX allocates *wired* (non-pageable) unified memory. On a Mac with swap
exhausted, a too-long render doesn't get OOM-killed — it can panic the kernel
and reboot the machine. songpipe therefore:

1. **Preflights** every render: estimates peak RAM from duration and clamps
   the duration to a budget (55% of available RAM by default,
   `--max-mem-frac` to change, `--force` to bypass — at your own risk).
2. **Watchdogs** the render: if available RAM drops below `--watchdog-floor`
   (default 6 GB) mid-run, the ACE subprocess is SIGKILLed and the take is
   lost rather than the machine. Surviving takes still ship.
3. **Caps MLX's Metal buffer cache** inside the render subprocess, which is
   what keeps a 30-second render from spiking past 40 GB.

Durations above 240 s additionally require `--force`.

## Development

```sh
pip install -e .[dev]
pytest
```

The test suite is hermetic — no ACE install, API keys, or network needed.

## License

Copyright (C) 2026 Matthew Klein and all other respective copyright holders. 
Apache-2.0. ACE-Step and its model weights, whisper models, and the LLM APIs
you call are licensed by their respective owners.
