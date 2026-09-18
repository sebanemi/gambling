import pytest

from football_predictor.config.settings import Settings
from football_predictor.evaluation.backtester import Backtester, BacktestReport
from football_predictor.services.model_trainer import ALL_MODELS
from tests.helpers import FakeHistoryProvider, hm


def _dataset():
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


def test_backtest_reports_all_models():
    report = Backtester(FakeHistoryProvider(_dataset()), Settings(), min_prior_matches=2).run(ALL_MODELS)
    assert isinstance(report, BacktestReport)
    assert report.evaluated == 10  # pasos con >=2 partidos previos

    for name in ALL_MODELS:
        assert name in report.reports
        evaluation = report.reports[name]
        assert 0 < evaluation.matches <= report.evaluated
        assert 0.0 <= evaluation.ranking_loss <= 1.0
        assert 0.0 <= evaluation.top1_accuracy <= 1.0
        # Muestras auditables por paso.
        assert len(report.samples(name)) == evaluation.matches


def test_backtest_is_deterministic():
    settings = Settings()
    dataset = _dataset()
    first = Backtester(FakeHistoryProvider(dataset), settings, min_prior_matches=2).run(ALL_MODELS)
    second = Backtester(FakeHistoryProvider(dataset), settings, min_prior_matches=2).run(ALL_MODELS)

    assert first.reports == second.reports
    for name in ALL_MODELS:
        assert first.samples(name) == second.samples(name)


def test_requires_history_and_valid_models():
    with pytest.raises(ValueError):
        Backtester(FakeHistoryProvider([]), Settings(), min_prior_matches=1).run()

    with pytest.raises(ValueError):
        Backtester(FakeHistoryProvider(_dataset()), Settings(), min_prior_matches=1).run(("nope",))


def test_min_prior_matches_filters_steps():
    dataset = _dataset()
    strict = Backtester(FakeHistoryProvider(dataset), Settings(), min_prior_matches=6)
    lax = Backtester(FakeHistoryProvider(dataset), Settings(), min_prior_matches=1)

    assert strict.run().evaluated == 12 - 6
    assert lax.run().evaluated == 11


def test_walk_forward_ignores_future_results():
    """Causalidad: mutar el resultado del partido k NO altera predicciones
    de pasos cuyo histórico previo no contiene a k."""
    dataset = _dataset()
    mutated = _dataset()
    k = 5  # mutamos el partido id=6 (índice 5)
    mutated[k] = mutated[k].__class__(
        match_id=mutated[k].match_id,
        date=mutated[k].date,
        home_team_id=mutated[k].home_team_id,
        away_team_id=mutated[k].away_team_id,
        home_goals=0,
        away_goals=9,
    )

    settings = Settings()
    original = Backtester(FakeHistoryProvider(dataset), settings, min_prior_matches=2).run(ALL_MODELS)
    changed = Backtester(FakeHistoryProvider(mutated), settings, min_prior_matches=2).run(ALL_MODELS)

    # Los 4 primeros pasos (predicción de los partidos 3..6) no pueden ver
    # el resultado mutado del partido 6: sus PROBABILIDADES deben ser
    # idénticas. El resultado real del partido 6 sí cambia (es el target).
    for name in ALL_MODELS:
        preds_original = [(mid, prob) for mid, prob, _, _ in original.samples(name)][:4]
        preds_changed = [(mid, prob) for mid, prob, _, _ in changed.samples(name)][:4]
        assert preds_original == preds_changed
        # El único sample con goles distintos es el propio partido mutado.
        goals_original = [(mid, hg, ag) for mid, hg, ag in [(s[0], s[2], s[3]) for s in original.samples(name)]]
        goals_changed = [(mid, hg, ag) for mid, hg, ag in [(s[0], s[2], s[3]) for s in changed.samples(name)]]
        assert goals_original != goals_changed
        assert original.samples(name)[:3] == changed.samples(name)[:3]