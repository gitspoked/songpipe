import numpy as np
from songpipe import score

def _sine(sr=16000, secs=1.0, f=220.0, amp=0.5):
    t = np.linspace(0, secs, int(sr*secs), endpoint=False)
    return (amp*np.sin(2*np.pi*f*t)).astype("float32")

def test_clip_ratio_zero_for_clean_silence_for_clipped():
    assert score.clip_ratio(_sine(amp=0.5)) == 0.0
    clipped = np.ones(16000, dtype="float32")
    assert score.clip_ratio(clipped) == 1.0

def test_silence_ratio_one_for_zeros_zero_for_tone():
    sr = 16000
    assert score.silence_ratio(np.zeros(sr, dtype="float32"), sr) == 1.0
    assert score.silence_ratio(_sine(sr=sr), sr) < 0.1

def test_loudness_range_positive_for_tone_zero_for_silence():
    assert score.loudness_range(_sine()) > 0.0
    assert score.loudness_range(np.zeros(16000, dtype="float32")) == 0.0

def test_combine_scores_ranks_clean_above_clipped_and_silent():
    clean = score.combine_scores(0.0, 0.0, 3.0)
    clipped = score.combine_scores(1.0, 0.0, 0.0)
    silent = score.combine_scores(0.0, 1.0, 0.0)
    assert clean > clipped and clean > silent

def test_metrics_accept_2d_stereo():
    sr = 16000; st = np.column_stack([_sine(sr=sr)]*2)
    assert score.silence_ratio(st, sr) < 0.1
    assert score.loudness_range(st) > 0.0
