import random
from dataclasses import dataclass
from songpipe.genres import GENRES, PRESETS, Preset

@dataclass
class Roll:
    genre: str
    preset: Preset
    bpm: int
    keyscale: str
    seed: int
    caption: str = None

def roll(seed: int) -> Roll:
    rng = random.Random(seed)
    genre = rng.choice(GENRES)
    p = PRESETS[genre]
    bpm = rng.randint(p.bpm_lo, p.bpm_hi)
    keyscale = rng.choice(p.keyscales)
    return Roll(genre=genre, preset=p, bpm=bpm, keyscale=keyscale, seed=seed)
