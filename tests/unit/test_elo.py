import pytest

from football_predictor.features.elo import EloCalculator, result_score
from tests.helpers import hm


def test_result_score():
    assert result_score(2, 1) == 1.0
    assert result_score(1, 1) == 0.5
    assert result_score(0, 3) == 0.0


def test_initial_rating_when_no_matches():
    assert EloCalculator().ratings_after([]) == {}


def test_expected_score_properties():
    calc = EloCalculator()
    assert calc.expected_score(1500, 1500) == pytest.approx(0.5)
    assert calc.expected_score(1600, 1500) > 0.5
    assert calc.expected_score(1500, 1600) < 0.5
    assert calc.expected_score(1600, 1500) + calc.expected_score(1500, 1600) == pytest.approx(1.0)


def test_winner_gains_rating_zero_sum():
    calc = EloCalculator(k_factor=20, home_advantage=0)
    ratings = calc.ratings_after([hm(1, 1, 1, 2, 1, 0)])
    assert ratings[1] > 1500
    assert ratings[2] < 1500
    assert ratings[1] + ratings[2] == pytest.approx(3000.0)


def test_draw_moves_favored_team_down():
    calc = EloCalculator(k_factor=20, home_advantage=0)
    ratings = calc.ratings_after([hm(1, 1, 1, 2, 1, 1)])
    assert ratings[1] < 1600  # el favorito pierde rating tras empatar
    assert ratings[2] > 1400  # el débil gana rating


def test_home_advantage_shifts_expected_score():
    calc = EloCalculator(home_advantage=60)
    with_adv = calc.expected_score(1500 + 60, 1500)
    without = calc.expected_score(1500, 1500)
    assert with_adv > without


def test_k_factor_scales_update():
    small = EloCalculator(k_factor=10, home_advantage=0)
    large = EloCalculator(k_factor=40, home_advantage=0)
    match = hm(1, 1, 1, 2, 1, 0)
    rating_small = small.ratings_after([match])[1]
    rating_large = large.ratings_after([match])[1]
    assert (rating_large - 1500) == pytest.approx(4 * (rating_small - 1500))


def test_chronological_and_order_independent():
    matches = [
        hm(1, 2, 1, 2, 1, 0),   # A wins
        hm(2, 1, 3, 1, 0, 2),   # A loses to C (day 1)
    ]
    ordered = EloCalculator(home_advantage=0).ratings_after(matches)
    shuffled = EloCalculator(home_advantage=0).ratings_after(list(reversed(matches)))
    assert ordered == shuffled