import pytest

from football_predictor.betting.value import edge, expected_value, implied_probability, is_value_bet


def test_expected_value_formula():
    # EV = p * odds - 1
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.6, 2.0) == pytest.approx(0.2)
    assert expected_value(0.4, 2.0) == pytest.approx(-0.2)


def test_is_value_bet_threshold():
    assert is_value_bet(0.6, 2.0, threshold=0.05) is True
    assert is_value_bet(0.5, 2.0, threshold=0.05) is False
    assert is_value_bet(0.5, 2.0, threshold=-0.5) is True


def test_implied_probability():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)


def test_edge():
    assert edge(0.6, 2.0) == pytest.approx(0.1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        expected_value(0.0, 2.0)
    with pytest.raises(ValueError):
        expected_value(1.5, 2.0)
    with pytest.raises(ValueError):
        expected_value(0.5, 0.0)
    with pytest.raises(ValueError):
        implied_probability(0.0)
