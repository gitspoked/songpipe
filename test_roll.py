from songpipe.roll import roll
from songpipe.genres import GENRES, PRESETS

def test_roll_deterministic():
    a, b = roll(123), roll(123)
    assert a.genre == b.genre and a.bpm == b.bpm and a.keyscale == b.keyscale

def test_roll_in_range():
    r = roll(7)
    assert r.genre in GENRES
    assert r.preset.bpm_lo <= r.bpm <= r.preset.bpm_hi
    assert r.keyscale in r.preset.keyscales

def test_every_genre_has_preset():
    for g in GENRES:
        assert g in PRESETS and PRESETS[g].lm in ("acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B")
