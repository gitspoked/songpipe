from songpipe.__main__ import load_lyrics


def test_title_line_extracted(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("TITLE: King of the Lot\n[Verse]\nline one\n")
    title, lyrics = load_lyrics(f)
    assert title == "King of the Lot"
    assert lyrics == "[Verse]\nline one"


def test_explicit_title_beats_title_line(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("TITLE: Wrong\n[Verse]\nline\n")
    title, lyrics = load_lyrics(f, title="Right")
    assert title == "Right"
    assert "TITLE" not in lyrics


def test_filename_fallback(tmp_path):
    f = tmp_path / "flat-as-a_pancake.txt"
    f.write_text("[Verse]\nline\n")
    title, lyrics = load_lyrics(f)
    assert title == "Flat As A Pancake"
    assert lyrics == "[Verse]\nline"
