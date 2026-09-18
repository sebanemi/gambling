import pytest

from football_predictor.config.settings import Settings
from football_predictor.services.model_trainer import (
    ALL_MODELS,
    ModelTrainer,
    parse_ensemble_weights,
)
from tests.helpers import FakeHistoryProvider, fv, hm


def _dataset():
    # 12 partidos con victorias locales, empates y victorias visitantes.
    return [
        hm(1, 1, 1, 2, 1, 0),
        hm(2, 2, 2, 3, 2, 2),
        hm(3, 3, 3, 1, 0, 2),
        hm(4, 4, 1, 3, 2, 1),
        hm(5, 5, 2, 1, 1, 1),
        hm(6, 6, 3, 2, 2, 0),
        hm(7, 7, 1, 2, 2, 0),
        hm(8, 8, 2, 3, 1, 0),
        hm(9, 9, 3, 1, 2, 2),
        hm(10, 10, 1, 3, 0, 1),
        hm(11, 11, 2, 1, 0, 2),
        hm(12, 12, 3, 2, 1, 0),
    ]


def test_train_all_returns_all_models():
    trainer = ModelTrainer(FakeHistoryProvider(_dataset()), Settings())
    models = trainer.train_all()
    assert set(models) == set(ALL_MODELS)


def test_all_models_predict_consistent_probabilities():
    trainer = ModelTrainer(FakeHistoryProvider(_dataset()), Settings())
    models = trainer.train_all()
    features = fv(100, 3, 1, home_elo=1550.0, away_elo=1490.0)

    for name in ALL_MODELS:
        result = models[name].predict(features)
        assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
        assert all(p > 0 for p in (result.home_win, result.draw, result.away_win))
        assert result.features_snapshot is not None


def test_training_is_deterministic():
    settings = Settings()
    first = ModelTrainer(FakeHistoryProvider(_dataset()), settings).train_all()
    second = ModelTrainer(FakeHistoryProvider(_dataset()), settings).train_all()
    features = fv(100, 3, 1)

    for name in ALL_MODELS:
        assert first[name].predict(features) == second[name].predict(features)


def test_train_all_requires_history():
    with pytest.raises(ValueError):
        ModelTrainer(FakeHistoryProvider([]), Settings()).train_all()


def test_parse_ensemble_weights():
    assert parse_ensemble_weights("0.4, 0.3, 0.3") == [0.4, 0.3, 0.3]
    with pytest.raises(ValueError):
        parse_ensemble_weights("1,2")
    with pytest.raises(ValueError):
        parse_ensemble_weights("0,0,0")