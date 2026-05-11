from turkey_audio_detection.ids import make_detection_id, make_item_id


def test_detection_id_is_deterministic() -> None:
    a = make_detection_id("C:/x.wav", 1.2345, 3.3333, "wild_turkey")
    b = make_detection_id("C:/x.wav", 1.2345, 3.3333, "wild_turkey")
    assert a == b


def test_detection_id_normalizes_windows_path_separators() -> None:
    """Backslash and forward-slash forms of the same path must produce the same ID."""
    a = make_detection_id("C:\\data\\ARU_01\\file.wav", 1.0, 4.0, "Meleagris gallopavo")
    b = make_detection_id("C:/data/ARU_01/file.wav", 1.0, 4.0, "Meleagris gallopavo")
    assert a == b


def test_item_id_changes_with_clip_bounds() -> None:
    det = "det_abc"
    a = make_item_id(det, 0.0, 3.0)
    b = make_item_id(det, 1.0, 4.0)
    assert a != b
