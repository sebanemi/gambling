import pytest

from football_predictor.evaluation.metrics import evaluate_model, ranking_loss, top1_correct


def test_ranking_loss_penalizes_low_confidence_on_correct_outcome():
    assert ranking_loss((0.6, 0.2, 0.2), 2, 1) == pytest.approx(0.4)
    assert ranking_loss((0.6, 0.2, 0.2), 0, 1) == pytest.approx(0.8)      # acertó el menos probable
    assert ranking_loss((0.1, 0.1, 0.8), 1, 1) == pytest.approx(0.9)      # empate poco probable


def test_ranking_loss_perfect_prediction_is_zero():
    assert ranking_loss((1.0, 0.0, 0.0), 3, 0) == pytest.approx(0.0)


def test_top1_correct_uses_argmax():
    assert top1_correct((0.6, 0.2, 0.2), 2, 1) is True
    assert top1_correct((0.6, 0.2, 0.2), 1, 2) is False
    assert top1_correct((0.2, 0.5, 0.3), 1, 1) is True


def test_evaluate_model_aggregates():
    samples = [
        ((0.8, 0.1, 0.1), 2, 0),   # correcto, confianza alta
        ((0.3, 0.3, 0.4), 1, 1),   # empate real pero predijo away
    ]
    report = evaluate_model("test", samples)
    assert report.matches == 2
    assert report.ranking_loss == pytest.approx((0.2 + 0.7) / 2)
    assert report.top1_accuracy == pytest.approx(0.5)
    assert report.correct_top1 == 1


def test_evaluate_model_empty_report():
    report = evaluate_model("test", [])
    assert report.matches == 0
    assert report.ranking_loss == 0.0
    assert report.top1_accuracy == 0.0