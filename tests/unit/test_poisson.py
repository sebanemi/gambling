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


def test_negative_rho_shifts_scoreline_cells_per_dixon_coles():
    # ρ negativo: sube 1-0 y 0-1, baja 1-1; 0-0 queda intacto (a escala).
    baseline = PoissonModel(rho=0.0)
    dc = PoissonModel(rho=-0.3)

    j0 = baseline._joint_distribution(1.0, 1.0)
    j1 = dc._joint_distribution(1.0, 1.0)

    assert j1[1, 0] > j0[1, 0]
    assert j1[0, 1] > j0[0, 1]
    assert j1[1, 1] < j0[1, 1]


def test_rho_fitted_is_used_for_prediction():
    # Muchos 1-0 y 0-1 ⇒ ρ negativo óptimo (correlación en pocos goles).
    history = [hm(i, i % 5 + 1, 1, 2, 1, 0) for i in range(1, 30)]
    model = PoissonModel()
    model.fit(history)
    assert model.rho < 0.0
    assert model.predict(fv(1, 1, 2)).home_win + model.predict(fv(1, 1, 2)).draw + model.predict(fv(1, 1, 2)).away_win == pytest.approx(1.0)


def test_rho_out_of_range_rejected():
    with pytest.raises(ValueError):
        PoissonModel(rho=-0.7)
    with pytest.raises(ValueError):
        PoissonModel(rho=0.7)


def test_prediction_result_normalizes_mistuned_probs():
    result = PredictionResult(match_id=1, home_win=0.3, draw=0.2, away_win=0.1)
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
    assert (result.home_win, result.draw, result.away_win) == pytest.approx((0.5, 1 / 3, 1 / 6))


def test_prediction_result_rejects_entry_probs():
    with pytest.raises(ValueError):
        PredictionResult(match_id=1, home_win=0.0, draw=0.0, away_win=0.0)