import math

import pytest

from football_predictor.evaluation.ensemble_optimizer import mix_log_loss, optimize_weights


def _samples(probs_a, probs_b, n, outcome_goal=2):
    """n muestras walk-forward: modelo A con probs_a, B con probs_b."""
    a = [(i, tuple(probs_a), outcome_goal, 0) for i in range(n)]
    b = [(i, tuple(probs_b), outcome_goal, 0) for i in range(n)]
    return {"poisson": a, "elo": b}


def test_optimizer_prefers_the_good_model():
    good = (0.9, 0.05, 0.05)
    uniform = (1 / 3, 1 / 3, 1 / 3)
    weights, log_loss = optimize_weights(_samples(good, uniform, 200))

    assert weights is not None
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # Sobre 200 partidos todos ganados de local, el buen modelo gana casi todo el peso.
    assert weights["poisson"] > 0.9
    assert weights["elo"] < 0.1
    assert log_loss == pytest.approx(-math.log(0.9), abs=1e-6)


def test_mix_log_loss_uniform_matches_manual():
    samples = _samples((0.9, 0.05, 0.05), (1 / 3, 1 / 3, 1 / 3), 100)
    result = mix_log_loss(samples, {"poisson": 0.5, "elo": 0.5})
    expected = -math.log(0.5 * 0.9 + 0.5 * (1 / 3))
    assert result == pytest.approx(expected)


def test_optimizer_requires_two_models():
    assert optimize_weights({"poisson": [(1, (0.9, 0.05, 0.05), 2, 0)]}) is None


def test_optimizer_requires_enough_samples():
    several = _samples((0.8, 0.1, 0.1), (0.2, 0.6, 0.2), 10)
    assert optimize_weights(several) is None


def test_mix_log_loss_requires_two_models():
    assert mix_log_loss({"poisson": [(1, (0.9, 0.05, 0.05), 2, 0)]}, {"poisson": 1.0}) is None