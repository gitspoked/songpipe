from songpipe.__main__ import _unique_dir


def test_fresh_slug_used_as_is(tmp_path):
    assert _unique_dir(tmp_path / "song") == tmp_path / "song"


def test_empty_existing_dir_is_reused(tmp_path):
    (tmp_path / "song").mkdir()
    assert _unique_dir(tmp_path / "song") == tmp_path / "song"


def test_occupied_dir_is_never_reused(tmp_path):
    d = tmp_path / "song"
    d.mkdir()
    (d / "song.mp3").write_bytes(b"x")
    assert _unique_dir(d) == tmp_path / "song-2"


def test_counter_skips_occupied_suffixes(tmp_path):
    for name in ("song", "song-2"):
        d = tmp_path / name
        d.mkdir()
        (d / "take.wav").write_bytes(b"x")
    assert _unique_dir(tmp_path / "song") == tmp_path / "song-3"


def test_empty_suffix_dir_is_reused(tmp_path):
    d = tmp_path / "song"
    d.mkdir()
    (d / "run.json").write_bytes(b"{}")
    (tmp_path / "song-2").mkdir()  # exists but empty -> safe to use
    assert _unique_dir(tmp_path / "song") == tmp_path / "song-2"
