import pytest

from football_predictor.features.builder import FeatureBuilder
from tests.helpers import FakeHistoryProvider, hm, target


def _dataset():
    return [
        hm(1, 1, 1, 3, 2, 0),   # A 2-0 C
        hm(2, 2, 2, 1, 1, 0),   # B 1-0 A
        hm(3, 3, 1, 4, 1, 1),   # A 1-1 D
    ]


def test_features_use_only_prior_history():
    builder = FeatureBuilder(FakeHistoryProvider(_dataset()))
    fv = builder.build_for_match(target(10, 10, 1, 2))

    # Forma de A: V (2-0), D (0-1), E (1-1)
    assert fv.home_form.overall_3.played == 3
    assert (fv.home_form.overall_3.wins, fv.home_form.overall_3.draws, fv.home_form.overall_3.losses) == (1, 1, 1)
    assert fv.home_form.overall_3.points_per_game == pytest.approx(4 / 3)
    # Forma de B: un solo partido ganado
    assert fv.away_form.overall_3.played == 1
    assert fv.away_form.overall_3.wins == 1

    # Promedios de goles
    assert fv.home_goals_for_avg == pytest.approx((2 + 0 + 1) / 3)
    assert fv.home_goals_against_avg == pytest.approx((0 + 1 + 1) / 3)
    assert fv.away_goals_for_avg == pytest.approx(1.0)
    assert fv.away_goals_against_avg == pytest.approx(0.0)

    assert fv.elo_difference == pytest.approx(fv.home_elo - fv.away_elo)
    assert fv.home_advantage == 60.0
    assert fv.match_id == 10
    assert fv.date == target(10, 10, 1, 2).date


def test_future_and_same_day_matches_are_ignored():
    base = _dataset()
    polluted = base + [
        hm(50, 10, 1, 5, 9, 0),   # mismo día del target (excluido)
        hm(51, 12, 1, 5, 5, 0),   # en el futuro (excluido)
        hm(52, 11, 2, 5, 7, 1),   # en el futuro (excluido)
    ]
    clean = FeatureBuilder(FakeHistoryProvider(base)).build_for_match(target(10, 10, 1, 2))
    with_pollution = FeatureBuilder(FakeHistoryProvider(polluted)).build_for_match(target(10, 10, 1, 2))
    assert clean.model_dump() == with_pollution.model_dump()


def test_flatten_is_flat_and_numeric():
    builder = FeatureBuilder(FakeHistoryProvider(_dataset()))
    flat = builder.build_for_match(target(10, 10, 1, 2)).flatten()

    assert isinstance(flat, dict)
    assert all(isinstance(v, float) for v in flat.values())
    for key in ("home_elo", "elo_difference", "home_form_points_per_game_3"):
        assert key in flat


def test_empty_history_defaults():
    builder = FeatureBuilder(FakeHistoryProvider([]))
    fv = builder.build_for_match(target(1, 1, 1, 2))
    assert fv.home_elo == 1500.0
    assert fv.away_elo == 1500.0
    assert fv.elo_difference == 0.0
    assert fv.home_form.overall_3.played == 0
    assert fv.home_goals_for_avg == 0.0
    assert fv.away_goals_against_avg == 0.0