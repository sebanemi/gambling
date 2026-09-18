from datetime import date

from sqlalchemy import select

from football_predictor.data.importers.match_importer import MatchImporter
from football_predictor.database.models.match import Match
from football_predictor.database.repositories.match_history import MatchHistoryRepository
from football_predictor.domain.entities import MatchRecord


class _FakeProvider:
    source = "fake"

    def __init__(self, records: list[MatchRecord]) -> None:
        self._records = records

    def get_matches(self) -> list[MatchRecord]:
        return self._records

    def get_teams(self):
        return []

    def get_competitions(self):
        return []


def _match(**kw):
    base = {
        "competition_name": "Premier League",
        "season_name": "2026",
        "date": date(2026, 9, 20),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": None,
        "away_goals": None,
        "status": "scheduled",
    }
    base.update(kw)
    return MatchRecord(**base)


def test_scheduled_fixture_is_imported(sqlite_session):
    provider = _FakeProvider([_match()])
    summary = MatchImporter(sqlite_session, provider).import_all()

    assert summary.inserted_matches == 1
    assert summary.updated_matches == 0
    match = sqlite_session.scalar(select(Match))
    assert match is not None
    assert match.status == "scheduled"
    assert match.home_goals is None
    assert match.away_goals is None


def test_scheduled_result_updated_on_reimport(sqlite_session):
    MatchImporter(sqlite_session, _FakeProvider([_match()])).import_all()

    provider = _FakeProvider([_match(home_goals=2, away_goals=1, status="finished")])
    summary = MatchImporter(sqlite_session, provider).import_all()

    assert summary.inserted_matches == 0
    assert summary.duplicates == 0
    assert summary.updated_matches == 1
    match = sqlite_session.scalar(select(Match))
    assert match is not None
    assert match.home_goals == 2
    assert match.away_goals == 1
    assert match.status == "finished"


def test_reimport_finished_is_duplicate(sqlite_session):
    provider = _FakeProvider([_match(home_goals=2, away_goals=1)])
    first = MatchImporter(sqlite_session, provider).import_all()
    second = MatchImporter(sqlite_session, provider).import_all()

    assert first.inserted_matches == 1
    assert second.inserted_matches == 0
    assert second.duplicates == 1
    assert second.updated_matches == 0


def test_scheduled_excluded_from_history(sqlite_session):
    records = [
        _match(away_team="Arsenal", home_team="Liverpool", home_goals=1, away_goals=0, status="finished"),
        _match(home_goals=0, away_goals=0, status="finished"),
        _match(away_team="Newcastle", home_team="Manchester", status="scheduled"),
    ]
    MatchImporter(sqlite_session, _FakeProvider(records)).import_all()

    history = MatchHistoryRepository(sqlite_session).get_all_matches()
    assert len(history) == 2
    assert all(m.home_goals is not None for m in history)