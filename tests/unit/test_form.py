import pytest

from football_predictor.features.form import FormCalculator
from tests.helpers import hm


def _a_series():
    # A: W 2-1 (d1), D 1-1 (d2), L 0-1 (d3), W 3-0 (d4)
    return [
        hm(1, 1, 1, 2, 2, 1),
        hm(2, 2, 1, 2, 1, 1),
        hm(3, 3, 1, 2, 0, 1),
        hm(4, 4, 1, 2, 3, 0),
    ]


def test_window_uses_last_n():
    form = FormCalculator.team_form(_a_series(), 1, (3,), 5)
    w3 = form.overall_3
    assert w3.played == 3
    assert (w3.wins, w3.draws, w3.losses) == (1, 1, 1)  # D, L, W
    assert w3.goals_for == 1 + 0 + 3
    assert w3.goals_against == 1 + 1 + 0
    assert w3.goal_difference == w3.goals_for - w3.goals_against
    assert w3.points_per_game == pytest.approx(4 / 3)


def test_partial_window_uses_available():
    form = FormCalculator.team_form(_a_series()[:1], 1, (3, 5, 10), 5)
    assert form.overall_3.played == 1
    assert form.overall_5.played == 1
    assert form.overall_10.played == 1
    assert form.overall_3.wins == 1


def test_home_away_context_split():
    # A juega d1 y d3 de local, d2 y d4 de visitante.
    series = [
        hm(1, 1, 1, 2, 2, 1),   # A local, G
        hm(2, 2, 2, 1, 1, 1),   # A visitante, E
        hm(3, 3, 1, 2, 0, 1),   # A local, P
        hm(4, 4, 2, 1, 0, 3),   # A visitante, G
    ]
    form = FormCalculator.team_form(series, 1, (3,), 5)
    assert form.home_5.played == 2          # d1 (G), d3 (P)
    assert form.home_5.wins == 1
    assert form.home_5.losses == 1
    assert form.away_5.played == 2          # d2 (E), d4 (G)
    assert form.away_5.wins == 1
    assert form.away_5.draws == 1


def test_empty_series_zeros():
    form = FormCalculator.team_form([], 1, (3, 5, 10), 5)
    assert form.overall_3.played == 0
    assert form.overall_3.points_per_game == 0.0
    assert form.overall_3.goal_difference == 0