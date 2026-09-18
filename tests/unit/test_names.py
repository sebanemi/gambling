from football_predictor.data.normalizers.names import NameNormalizer


def test_collapses_whitespace_and_strips():
    normalizer = NameNormalizer()
    assert normalizer("  Man United   ") == "Man United"


def test_removes_accents():
    normalizer = NameNormalizer()
    assert normalizer("Mönchengladbach") == "Monchengladbach"
    assert normalizer("Atlético") == "Atletico"


def test_idempotent():
    normalizer = NameNormalizer()
    assert normalizer(normalizer("  Arsenal  ")) == "Arsenal"