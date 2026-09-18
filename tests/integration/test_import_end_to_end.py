"""End-to-end contra PostgreSQL real: importación y constraints."""

import datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from football_predictor.data.importers.csv_importer import CsvImporter
from football_predictor.data.providers.csv_provider import CsvDataProvider
from football_predictor.database.models.competition import Competition
from football_predictor.database.models.match import Match
from football_predictor.database.models.season import Season
from football_predictor.database.models.team import Team
from football_predictor.database.repositories.match import MatchRepository

_HEADER = "date,competition,season,home_team,away_team,home_goals,away_goals\n"


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    rows = [
        "2024-01-10,Liga Argentina,2024,Estudiantes,River Plate,2,1",
        "2024-01-17,Liga Argentina,2024,Estudiantes,River Plate,0,0",
        "2024-01-24,Liga Argentina,2024,Boca Juniors,River Plate,1,3",
        "2025-02-02,Liga Argentina,2025,River Plate,Boca Juniors,4,0",
    ]
    path = tmp_path / "matches.csv"
    path.write_text(_HEADER + "\n".join(rows), encoding="utf-8")
    return path


def test_import_and_persist_on_postgres(db_session, csv_path):
    summary = CsvImporter(db_session, CsvDataProvider(csv_path)).import_all()

    assert summary.valid == 4
    assert summary.inserted_matches == 4
    assert summary.new_teams == 3
    assert summary.new_competitions == 1
    assert summary.new_seasons == 2

    assert db_session.scalar(select(func.count()).select_from(Match)) == 4
    assert db_session.scalar(select(func.count()).select_from(Team)) == 3
    assert db_session.scalar(select(func.count()).select_from(Competition)) == 1
    assert db_session.scalar(select(func.count()).select_from(Season)) == 2


def test_deduplication_on_second_import(db_session, csv_path):
    importer = CsvImporter(db_session, CsvDataProvider(csv_path))
    importer.import_all()
    second = importer.import_all()

    assert second.inserted_matches == 0
    assert second.duplicates == 4


def test_unique_constraint_rejects_duplicate_key(db_session, csv_path):
    importer = CsvImporter(db_session, CsvDataProvider(csv_path))
    importer.import_all()

    competition_id = db_session.scalar(select(Competition.id))
    season_id = db_session.scalar(select(Season.id))
    teams = db_session.execute(select(Team.id, Team.name)).all()
    by_name = {name: team_id for team_id, name in teams}

    repo = MatchRepository(db_session)
    with pytest.raises(IntegrityError):
        repo.add(
            competition_id=competition_id,
            season_id=season_id,
            date=datetime.date(2024, 1, 10),
            home_team_id=by_name["Estudiantes"],
            away_team_id=by_name["River Plate"],
            home_goals=9,
            away_goals=9,
        )
        db_session.flush()