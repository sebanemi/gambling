import pytest

from football_predictor.models.base import PredictionResult
from football_predictor.models.poisson import PoissonModel
from tests.helpers import fv, hm


def _history_with_strong_home():
    # A golea de local y además gana de visita; B anota poco.
    return [
        hm(1, 1, 1, 2, 3, 0),
        hm(2, 2, 2, 1, 0, 2),   # A gana de visita
        hm(3, 3, 1, 4, 2, 0),
        hm(4, 4, 4, 2, 2, 1),
        hm(5, 5, 1, 4, 3, 1),
    ]


def test_fit_is_deterministic():
    history = _history_with_strong_home()
    first = PoissonModel()
    second = PoissonModel()
    first.fit(history)
    second.fit(history)

    assert first._gamma_home == second._gamma_home
    assert first._attack == second._attack
    assert first._defense == second._defense


def test_strong_home_team_dominates():
    model = PoissonModel()
    model.fit(_history_with_strong_home())

    lam_home, lam_away = model.expected_goals(1, 2)
    assert lam_home > lam_away

    result = model.predict(fv(10, 1, 2))
    assert result.home_win > result.away_win
    assert result.home_win > result.draw > 0
    assert result.home_goals is not None and result.away_goals is not None


def test_probabilities_sum_to_one_and_are_positive():
    model = PoissonModel()
    model.fit(_history_with_strong_home())
    for pr in (model.predict(fv(1, 1, 2)), model.predict(fv(2, 2, 4)), model.predict(fv(3, 4, 1))):
        assert pr.home_win + pr.draw + pr.away_win == pytest.approx(1.0)
        assert all(p > 0 for p in (pr.home_win, pr.draw, pr.away_win))


def test_unknown_teams_fall_back_to_baseline():
    model = PoissonModel()
    model.fit(_history_with_strong_home())
    result = model.predict(fv(99, 999, 998))
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
    assert result.home_goals is not None


def test_fit_requires_history():
    with pytest.raises(ValueError):
        PoissonModel().fit([])


def test_draw_correction_raises_draw_probability_end_to_end():
    # Historial de puros empates ⇒ matchup balanceado.
    history = [hm(i, i % 7 + 1, 1, 2, 1, 1) for i in range(1, 15)]

    plain = PoissonModel(draw_correction=1.0)
    boosted = PoissonModel(draw_correction=2.0)
    plain.fit(history)
    boosted.fit(history)

    if boosted.draw_correction != 1.0:
        assert boosted.predict(fv(1, 1, 2)).draw > plain.predict(fv(1, 1, 2)).draw


def test_prediction_result_normalizes_mistuned_probs():
    result = PredictionResult(match_id=1, home_win=0.3, draw=0.2, away_win=0.1)
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
    assert (result.home_win, result.draw, result.away_win) == pytest.approx((0.5, 1 / 3, 1 / 6))


def test_prediction_result_rejects_entry_probs():
    with pytest.raises(ValueError):
        PredictionResult(match_id=1, home_win=0.0, draw=0.0, away_win=0.0)