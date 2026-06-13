from asr_pipeline.normalization import indic_normalize, light_cleanup, whisper_normalize


def test_light_cleanup_removes_annotation_noise():
    text = "  hello   [noise]   world !! "
    assert light_cleanup(text) == "hello world!"


def test_indic_normalize_keeps_devanagari_text():
    text = "\u0928\u092e\u0938\u094d\u0924\u0947   \u0926\u0941\u0928\u093f\u092f\u093e"
    assert indic_normalize(text)


def test_whisper_normalize_lowercases_latin_text():
    assert whisper_normalize("HELLO, India!") == "hello india"
