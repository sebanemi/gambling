"""MatchHistoryRepository contra PostgreSQL real: filtros y orden."""

from datetime import date

from football_predictor.database.models.competition import Competition
from football_predictor.database.models.match import Match
from football_predictor.database.models.season import Season
from football_predictor.database.models.team import Team
from football_predictor.database.repositories.match_history import MatchHistoryRepository


def _seed(session) -> None:
    competition = Competition(name="Liga", country="AR")
    season = Season(name="2026", competition=competition)
    alpha = Team(name="Alpha")
    beta = Team(name="Beta")
    gamma = Team(name="Gamma")
    session.add_all([competition, alpha, beta, gamma])
    session.flush()

    def finished_match(day: date, home, away, home_goals: int, away_goals: int) -> Match:
        return Match(
            competition=competition,
            season=season,
            date=day,
            home_team=home,
            away_team=away,
            home_goals=home_goals,
            away_goals=away_goals,
            status="finished",
        )

    session.add_all(
        [
            finished_match(date(2026, 1, 1), alpha, beta, 1, 0),
            finished_match(date(2026, 1, 5), alpha, gamma, 3, 1),
            finished_match(date(2026, 1, 9), gamma, beta, 0, 2),
            # Excluidos: >= cutoff / no finished / goles NULL.
            Match(
                competition=competition,
                season=season,
                date=date(2026, 2, 1),
                home_team=beta,
                away_team=gamma,
                home_goals=2,
                away_goals=1,
            ),
            Match(
                competition=competition,
                season=season,
                date=date(2026, 1, 3),
                home_team=beta,
                away_team=gamma,
                home_goals=1,
                away_goals=1,
                status="scheduled",
            ),
            Match(
                competition=competition,
                season=season,
                date=date(2026, 1, 4),
                home_team=beta,
                away_team=gamma,
                home_goals=None,
                away_goals=None,
                status="postponed",
            ),
        ]
    )
    session.flush()


def test_filters_by_cutoff_state_and_goals(db_session):
    _seed(db_session)
    repo = MatchHistoryRepository(db_session)

    history = repo.get_matches_before(date(2026, 1, 31))

    assert [m.date for m in history] == [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 9)]
    assert all(m.home_goals is not None and m.away_goals is not None for m in history)
    assert all(m.date < date(2026, 1, 31) for m in history)


def test_strict_date_boundary(db_session):
    _seed(db_session)
    repo = MatchHistoryRepository(db_session)

    assert [m.date for m in repo.get_matches_before(date(2026, 1, 9))] == [date(2026, 1, 1), date(2026, 1, 5)]
    assert repo.get_matches_before(date(2026, 1, 1)) == []


def test_chronological_order(db_session):
    _seed(db_session)
    repo = MatchHistoryRepository(db_session)

    history = repo.get_matches_before(date(2026, 1, 31))
    assert [m.date for m in history] == sorted(m.date for m in history)