"""Same-league: el filtro por competición no filtra futuro ni rompe defaults."""

import pytest

from football_predictor.config.settings import Settings
from football_predictor.domain.entities import HistoricalMatch, TargetMatch
from football_predictor.evaluation.backtester import Backtester
from football_predictor.features.builder import FeatureBuilder
from football_predictor.services.model_trainer import ModelTrainer
from football_predictor.services.scoped_history import CompetitionScopedHistory
from tests.helpers import FakeHistoryProvider, day, hm


def _hm_comp(match_id: int, day_number: int, home: int, away: int, hg: int, ag: int, comp: int):
    m = hm(match_id, day_number, home, away, hg, ag)
    return HistoricalMatch(**{**m.__dict__, "competition_id": comp})


def _dataset():
    # Liga 7: 1 vence a 2. Liga 9 (otra): 2 vence a 1 (señal opuesta).
    return [
        _hm_comp(1, 1, 1, 2, 2, 0, 7),
        _hm_comp(2, 2, 2, 1, 1, 0, 9),
        _hm_comp(3, 3, 1, 2, 3, 1, 7),
        _hm_comp(4, 4, 2, 1, 0, 2, 9),
        _hm_comp(5, 5, 1, 2, 1, 0, 7),
        _hm_comp(6, 6, 1, 2, 2, 2, 7),
    ]


def test_scope_filters_by_competition_only():
    scoped = CompetitionScopedHistory(FakeHistoryProvider(_dataset()), 7)
    all_matches = scoped.get_all_matches()
    assert {m.match_id for m in all_matches} == {1, 3, 5, 6}
    before = scoped.get_matches_before(day(5))
    assert {m.match_id for m in before} == {1, 3}
    # La puerta por fecha sigue intacta (nada del futuro).
    assert all(m.date < day(5) for m in before)


def test_scope_unknown_competition_excluded():
    dataset = _dataset()
    assert all(m.competition_id is not None for m in dataset)
    legacy = FakeHistoryProvider([hm(1, 1, 1, 2, 1, 0)])
    assert CompetitionScopedHistory(legacy, 7).get_all_matches() == []


def test_trainer_scoped_uses_only_league():
    trainer = ModelTrainer(FakeHistoryProvider(_dataset()), Settings(), competition_id=7)
    models = trainer.train_all()
    assert set(models) == {"poisson", "elo", "ml", "gbm", "ensemble"}
    # Features del target con solo liga 7: el 1 viene ganando.
    target = TargetMatch(match_id=99, date=day(10), home_team_id=1, away_team_id=2)
    features = trainer.feature_builder().build_for_match(target)
    assert features.home_goals_for_avg > features.away_goals_for_avg


def test_trainer_unscoped_keeps_old_behavior():
    trainer = ModelTrainer(FakeHistoryProvider(_dataset()), Settings())
    models = trainer.train_all()
    assert set(models) == {"poisson", "elo", "ml", "gbm", "ensemble"}


def test_backtester_scoped_counts_only_league():
    report = Backtester(
        FakeHistoryProvider(_dataset()), Settings(), min_prior_matches=1, competition_id=7
    ).run(("elo",))
    # Liga 7 tiene 4 partidos: pasos evaluados = 3 (índices 1..3 del filtrado).
    assert report.evaluated == 3
    assert report.reports["elo"].matches == 3


def test_builder_scoped_provider_matches_manual_filter():
    scoped = CompetitionScopedHistory(FakeHistoryProvider(_dataset()), 9)
    builder = FeatureBuilder(scoped)
    target = TargetMatch(match_id=99, date=day(10), home_team_id=1, away_team_id=2)
    features = builder.build_for_match(target)
    # Liga 9: equipo 1 marcó 0+2 (avg 1.0), equipo 2 marcó 1+0 (avg 0.5).
    # Sin scope daría otro promedio: el filtro sí cambia las features.
    assert features.home_goals_for_avg == pytest.approx(1.0)
    assert features.away_goals_for_avg == pytest.approx(0.5)


def test_historical_match_defaults_to_no_competition():
    m = hm(1, 1, 1, 2, 1, 0)
    assert m.competition_id is None
