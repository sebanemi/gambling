import pytest

from football_predictor.models.elo_model import EloModel
from tests.helpers import fv


def test_symmetric_teams_draw_peaks():
    model = EloModel(draw_max=0.30, draw_sigma=300.0)
    result = model.predict(fv(1, 1, 2, home_elo=1500.0, away_elo=1500.0, home_advantage=0.0))
    assert result.draw == pytest.approx(0.30)
    assert result.home_win == pytest.approx(result.away_win)


def test_home_advantage_favors_local():
    model = EloModel()
    result = model.predict(fv(1, 1, 2, home_elo=1400.0, away_elo=1400.0, home_advantage=60.0))
    assert result.home_win > result.away_win
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)


def test_stronger_home_wins_more():
    model = EloModel()
    result = model.predict(fv(1, 1, 2, home_elo=1650.0, away_elo=1500.0))
    assert result.home_win > result.away_win
    assert result.home_win < 1.0 and result.away_win > 0.0


def test_stronger_away_wins_more():
    model = EloModel()
    result = model.predict(fv(1, 1, 2, home_elo=1500.0, away_elo=1700.0, home_advantage=0.0))
    assert result.away_win > result.home_win


def test_draw_probability_decays_with_rating_gap():
    model = EloModel()
    close = model.predict(fv(1, 1, 2, home_elo=1500.0, away_elo=1500.0)).draw
    far = model.predict(fv(2, 1, 2, home_elo=1800.0, away_elo=1500.0)).draw
    assert far < close


def test_deterministic():
    model = EloModel()
    assert model.predict(fv(1, 1, 2, home_elo=1550.0, away_elo=1490.0)) == model.predict(
        fv(1, 1, 2, home_elo=1550.0, away_elo=1490.0)
    )