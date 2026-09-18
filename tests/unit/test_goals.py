import pytest

from football_predictor.features.goals import GoalsAverageCalculator
from tests.helpers import hm


def test_averages_over_all_history():
    series = [hm(1, 1, 1, 2, 2, 1), hm(2, 2, 3, 1, 3, 0)]
    goals_for, goals_against = GoalsAverageCalculator.averages_for(series, 1, None)
    assert goals_for == pytest.approx((2 + 0) / 2)
    assert goals_against == pytest.approx((1 + 3) / 2)


def test_window_limits_to_recent_n():
    series = [hm(1, 1, 1, 2, 2, 1), hm(2, 2, 3, 1, 0, 3)]
    # Último partido: A de visitante gana 0-3 a C.
    goals_for, goals_against = GoalsAverageCalculator.averages_for(series, 1, 1)
    assert goals_for == pytest.approx(3.0)
    assert goals_against == pytest.approx(0.0)


def test_window_zero_means_all_history():
    series = [hm(1, 1, 1, 2, 3, 0)]
    assert GoalsAverageCalculator.averages_for(series, 1, 0) == (3.0, 0.0)


def test_empty_history_returns_zeros():
    assert GoalsAverageCalculator.averages_for([], 1, None) == (0.0, 0.0)
    assert GoalsAverageCalculator.averages_for([], 1, 5) == (0.0, 0.0)