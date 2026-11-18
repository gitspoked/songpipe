# songpipe — generative local music studio

Roll a genre → Claude + Grok write competing lyrics → hermes-4-70b judges →
ACE-Step sings → Dia ad-libs (genre-gated) → loudnorm master → mp3.

## Run
    .venv-songpipe/bin/python -m songpipe 42 --xai-model grok-4.3 --duration 180

Output: `mkMusic/songs/<slug>/<slug>.mp3` + stems + `run.json`.
Keys read from `mkMusic/.env` (ANTHROPIC_API_KEY, XAI_API_KEY, LM_STUDIO_API_KEY).
LM Studio must be serving on :1234. Engines: ACE `.venv-ace`, Dia `.venv`.
Spec: docs/superpowers/specs/2026-06-23-songpipe-design.md
