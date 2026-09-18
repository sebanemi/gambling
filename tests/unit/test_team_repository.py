from datetime import date

from football_predictor.database.models.competition import Competition
from football_predictor.database.models.match import Match
from football_predictor.database.models.season import Season
from football_predictor.database.models.team import Team
from football_predictor.database.repositories.team import TeamRepository


def _seed(session):
    competition = Competition(name="Premier League")
    session.add(competition)
    session.flush()
    season = Season(name="2025", competition_id=competition.id)
    session.add(season)
    session.flush()

    veteran = Team(name="Brentford FC", country="ENG", source="api")
    duel = Team(name="Brentford", country="", source="csv")
    brighton = Team(name="Brighton & Hove Albion FC", country="ENG")
    bournemouth = Team(name="AFC Bournemouth", country="ENG")
    session.add_all([veteran, duel, brighton, bournemouth])
    session.flush()

    def match(d, home, away, goals=None):
        m = Match(
            competition_id=competition.id,
            season_id=season.id,
            date=d,
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=goals if goals else 0,
            away_goals=goals if goals else 0,
        )
        session.add(m)

    match(date(2024, 8, 1), veteran, brighton)
    match(date(2024, 8, 15), veteran, bournemouth)
    session.flush()
    return {"veteran": veteran, "duel": duel}


def test_resolve_strips_suffix_and_prefers_active_team(sqlite_session):
    ids = _seed(sqlite_session)
    repo = TeamRepository(sqlite_session)

    # "Brentford" existe como "duel" (0 partidos) y "Brentford FC" (2 partidos).
    assert repo.resolve("Brentford") == ids["veteran"].id
    assert repo.resolve(" brentford fc ") == ids["veteran"].id


def test_resolve_substring_for_compounds(sqlite_session):
    _seed(sqlite_session)
    repo = TeamRepository(sqlite_session)

    assert repo.resolve("Brighton") is not None
    assert repo.resolve("Bournemouth") is not None


def test_resolve_unknown_returns_none(sqlite_session):
    _seed(sqlite_session)
    assert TeamRepository(sqlite_session).resolve("Equipo Inexistente FC") is None
    assert TeamRepository(sqlite_session).resolve("") is None