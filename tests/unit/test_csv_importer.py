from pathlib import Path

from sqlalchemy import func, select

from football_predictor.data.importers.csv_importer import CsvImporter, ImportSummary
from football_predictor.data.providers.csv_provider import CsvDataProvider
from football_predictor.database.models.competition import Competition
from football_predictor.database.models.match import Match
from football_predictor.database.models.season import Season
from football_predictor.database.models.team import Team

_HEADER = "date,competition,season,home_team,away_team,home_goals,away_goals\n"
_VALID_ROWS = [
    "2024-08-17,Premier League,2024,Arsenal,Chelsea,2,1",
    "2024-08-24,Premier League,2024,Chelsea,Arsenal,0,3",
    "2023-08-12,Premier League,2023,Arsenal,Newcastle,0,0",
    "2024-09-01,La Liga,2024,Real Madrid,FC Barcelona,2,0",
]
_INVALID = [
    "bad-date,Premier League,2024,Arsenal,Chelsea,2,1",
    "2024-08-17,Premier League,2024,Arsenal,Arsenal,1,1",
    "2024-08-17,Premier League,2024,Arsenal,Chelsea,2.5,1",
]
_DUPLICATE = "2024-08-17,Premier League,2024,Arsenal,Chelsea,2,1"


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(_HEADER + "\n".join(rows), encoding="utf-8")
    return path


def _import(path: Path, session) -> ImportSummary:
    return CsvImporter(session, CsvDataProvider(path)).import_all()


def test_import_summary(sqlite_session, tmp_path):
    path = _write_csv(tmp_path / "sample.csv", [*_VALID_ROWS, *_INVALID, _DUPLICATE])
    summary = _import(path, sqlite_session)

    assert summary.rows_read == len(_VALID_ROWS) + len(_INVALID) + 1
    assert summary.valid == len(_VALID_ROWS) + 1  # incluye el duplicado valido
    assert summary.invalid == len(_INVALID)
    # Equipos unicos: Arsenal, Chelsea, Newcastle, Real Madrid, FC Barcelona = 5
    assert summary.new_teams == 5
    assert summary.new_competitions == 2
    assert summary.new_seasons == 3  # PL 2024, PL 2023, LaLiga 2024
    assert summary.inserted_matches == len(_VALID_ROWS)
    assert summary.duplicates == 1


def test_second_import_is_all_duplicates(sqlite_session, tmp_path):
    path = _write_csv(tmp_path / "sample.csv", _VALID_ROWS)
    first = _import(path, sqlite_session)
    second = _import(path, sqlite_session)

    assert first.inserted_matches == len(_VALID_ROWS)
    assert second.inserted_matches == 0
    assert second.duplicates == len(_VALID_ROWS)


def test_rows_persisted(sqlite_session, tmp_path):
    path = _write_csv(tmp_path / "sample.csv", _VALID_ROWS)
    _import(path, sqlite_session)

    assert sqlite_session.scalar(select(func.count()).select_from(Match)) == len(_VALID_ROWS)
    assert sqlite_session.scalar(select(func.count()).select_from(Team)) == 5
    assert sqlite_session.scalar(select(func.count()).select_from(Competition)) == 2
    assert sqlite_session.scalar(select(func.count()).select_from(Season)) == 3


def test_season_dates_derived(sqlite_session, tmp_path):
    path = _write_csv(tmp_path / "sample.csv", _VALID_ROWS)
    _import(path, sqlite_session)

    season_2024 = sqlite_session.execute(
        select(Season).join(Competition, Season.competition_id == Competition.id).where(
            Season.name == "2024", Competition.name == "Premier League"
        )
    ).scalar_one()
    assert season_2024.start_date.isoformat() == "2024-08-17"
    assert season_2024.end_date.isoformat() == "2024-08-24"


def test_normalization_applies_to_teams(sqlite_session, tmp_path):
    rows = [
        "2024-08-17,PL,2024,Atletico Madrid,FC Barcelona,1,1",
        "2024-08-24,PL,2024,Atletico  Madrid,FC Barcelona,2,2",
    ]
    path = _write_csv(tmp_path / "sample.csv", rows)
    summary = _import(path, sqlite_session)

    assert summary.new_teams == 2  # Atletico Madrid y FC Barcelona
    assert summary.inserted_matches == 2
    teams = sqlite_session.execute(select(Team.name)).scalars().all()
    assert "Atletico Madrid" in teams
    assert "FC Barcelona" in teams