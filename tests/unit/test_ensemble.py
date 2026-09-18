import pytest

from football_predictor.models.base import PredictionResult
from football_predictor.models.ensemble import EnsembleModel
from tests.helpers import fv


class _FakeModel:
    def __init__(self, home_win: float, draw: float, away_win: float, goals: tuple[float, float] | None = None):
        self._prob = (home_win, draw, away_win)
        self._goals = goals

    def predict(self, features):
        return PredictionResult(
            match_id=features.match_id,
            home_win=self._prob[0],
            draw=self._prob[1],
            away_win=self._prob[2],
            home_goals=self._goals[0] if self._goals else None,
            away_goals=self._goals[1] if self._goals else None,
        )


def test_weights_are_normalized_and_averaged():
    a = _FakeModel(0.7, 0.2, 0.1)
    b = _FakeModel(0.1, 0.2, 0.7)
    ensemble = EnsembleModel([a, b], [1.0, 1.0])  # 0.5 / 0.5

    result = ensemble.predict(fv(1, 1, 2))
    assert result.home_win == pytest.approx(0.4)
    assert result.draw == pytest.approx(0.2)
    assert result.away_win == pytest.approx(0.4)
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
    assert result.match_id == 1


def test_unbalanced_weights():
    a = _FakeModel(1.0, 0.0, 0.0)
    b = _FakeModel(0.0, 1.0, 0.0)
    ensemble = EnsembleModel([a, b], [3, 1])  # 0.75 / 0.25
    result = ensemble.predict(fv(1, 1, 2))
    assert result.home_win == pytest.approx(0.75)
    assert result.draw == pytest.approx(0.25)


def test_goals_averaged_only_from_models_that_provide_them():
    a = _FakeModel(0.5, 0.2, 0.3, goals=(1.8, 0.9))
    b = _FakeModel(0.4, 0.2, 0.4)
    result = EnsembleModel([a, b], [1, 1]).predict(fv(1, 1, 2))
    assert result.home_goals == pytest.approx(1.8)
    assert result.away_goals == pytest.approx(0.9)


def test_invalid_configs_raise():
    with pytest.raises(ValueError):
        EnsembleModel([], [])
    with pytest.raises(ValueError):
        EnsembleModel([_FakeModel(0.5, 0.2, 0.3)], [1.0, 2.0])
    with pytest.raises(ValueError):
        EnsembleModel([_FakeModel(0.5, 0.2, 0.3)], [0.0])