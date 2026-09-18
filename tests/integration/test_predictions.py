"""End-to-end contra PostgreSQL real: predecir, persistir y evaluar."""

import datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from football_predictor.data.importers.csv_importer import CsvImporter
from football_predictor.data.providers.csv_provider import CsvDataProvider
from football_predictor.database.models.match import Match
from football_predictor.database.models.season import Season
from football_predictor.database.models.team import Team
from football_predictor.database.repositories.match import MatchRepository
from football_predictor.database.repositories.prediction import PredictionRepository
from football_predictor.services.model_trainer import ALL_MODELS
from football_predictor.services.prediction_service import PredictionService

_HEADER = "date,competition,season,home_team,away_team,home_goals,away_goals\n"
_FUTURE = datetime.date(2030, 1, 1)


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    rows = [
        "2024-01-10,Liga Argentina,2024,Estudiantes,River Plate,2,1",
        "2024-01-17,Liga Argentina,2024,Boca Juniors,River Plate,0,0",
        "2024-01-24,Liga Argentina,2024,Estudiantes,Boca Juniors,1,3",
        "2024-02-02,Liga Argentina,2024,River Plate,Boca Juniors,0,1",
    ]
    path = tmp_path / "matches.csv"
    path.write_text(_HEADER + "\n".join(rows), encoding="utf-8")
    return path


def _team_ids(session) -> dict[str, int]:
    rows = session.execute(select(Team.id, Team.name)).all()
    return {name: team_id for team_id, name in rows}


def _add_future_match(session, team_ids: dict[str, int]) -> Match:
    season_id = session.scalar(select(Season.id))
    repo = MatchRepository(session)
    match = repo.add(
        competition_id=session.scalar(select(Match.competition_id).limit(1)),
        season_id=season_id,
        date=_FUTURE,
        home_team_id=team_ids["Estudiantes"],
        away_team_id=team_ids["River Plate"],
        home_goals=None,
        away_goals=None,
        status="scheduled",
    )
    session.flush()
    return match


def test_predict_persists_and_repredicts_upsert(db_session, csv_path):
    CsvImporter(db_session, CsvDataProvider(csv_path)).import_all()
    team_ids = _team_ids(db_session)
    future = _add_future_match(db_session, team_ids)

    service = PredictionService(db_session)
    results, stats = service.predict_match(match_id=future.id, model_name="all")

    assert set(results) == set(ALL_MODELS)
    repository = PredictionRepository(db_session)
    assert repository.count(match_id=future.id) == 4

    for result in results.values():
        assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
        assert result.features_snapshot is not None  # snapshot auditable

    # Re-predecir NO duplica filas (upsert por match_id + model_name).
    results2, _ = service.predict_match(match_id=future.id, model_name="all")
    assert repository.count(match_id=future.id) == 4
    assert repository.count() == 4


def test_evaluate_over_finished_match(db_session, csv_path):
    CsvImporter(db_session, CsvDataProvider(csv_path)).import_all()

    # Predice el partido finalizado más reciente (cada modelo por separado).
    latest = db_session.scalar(select(Match).order_by(Match.date.desc()).limit(1))
    results, _ = PredictionService(db_session).predict_match(match_id=latest.id, model_name="all")
    assert set(results) == set(ALL_MODELS)

    service = PredictionService(db_session)
    for model_name in ("poisson", "elo", "ml", "ensemble"):
        report = service.evaluate(model_name)
        assert report.matches >= 1
        assert 0.0 <= report.ranking_loss <= 1.0
        assert 0.0 <= report.top1_accuracy <= 1.0
        assert report.correct_top1 == report.matches if report.top1_accuracy == 1.0 else report.correct_top1 <= report.matches


def test_predict_unknown_model_raises(db_session, csv_path):
    CsvImporter(db_session, CsvDataProvider(csv_path)).import_all()
    future = _add_future_match(db_session, _team_ids(db_session))
    with pytest.raises(ValueError):
        PredictionService(db_session).predict_match(match_id=future.id, model_name="does-not-exist")