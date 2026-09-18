import pytest

from football_predictor.models.base import Outcome
from football_predictor.models.ml import MLModel
from tests.helpers import fv


def _synthetic_dataset(n_per_class: int = 40):
    features = []
    outcomes = []
    for i in range(n_per_class):
        features.append(fv(i, 1, 2, home_elo=1500.0 + 300.0, away_elo=1500.0))
        outcomes.append(Outcome.HOME)
        features.append(fv(1000 + i, 1, 2, home_elo=1500.0, away_elo=1500.0 + 300.0))
        outcomes.append(Outcome.AWAY)
        features.append(fv(2000 + i, 1, 2, home_elo=1500.0, away_elo=1500.0))
        outcomes.append(Outcome.DRAW)
    return features, outcomes


def test_learns_separable_synthetic_data():
    features, outcomes = _synthetic_dataset()
    model = MLModel()
    model.fit(features, outcomes)

    for fv_vec, expected in zip(features[:15], outcomes[:15]):
        result = model.predict(fv_vec)
        predicted = Outcome(max((Outcome.HOME, Outcome.DRAW, Outcome.AWAY), key=lambda o: getattr(result, {
            Outcome.HOME: "home_win",
            Outcome.DRAW: "draw",
            Outcome.AWAY: "away_win",
        }[o])))
        assert predicted == expected
        assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)


def test_predict_requires_fit():
    with pytest.raises(ValueError):
        MLModel().predict(fv(1, 1, 2))


def test_fit_is_deterministic():
    features, outcomes = _synthetic_dataset()
    first = MLModel(random_state=42)
    second = MLModel(random_state=42)
    first.fit(features, outcomes)
    second.fit(features, outcomes)
    assert first.predict(fv(1, 1, 2)) == second.predict(fv(1, 1, 2))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        MLModel().fit([fv(1, 1, 2)], [Outcome.HOME, Outcome.DRAW])