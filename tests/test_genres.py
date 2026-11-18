from songpipe.genres import PRESETS, GENRES

def test_presets_use_quality_render_settings():
    for g in GENRES:
        p = PRESETS[g]
        assert p.inference_steps >= 20, f"{g} steps too low for quality"
        assert p.guidance_scale >= 1.0
