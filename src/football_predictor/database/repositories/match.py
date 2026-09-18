from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.match import Match


class MatchRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def existing(self) -> dict[tuple[int, int, int, int, date], Match]:
        """Mapa de claves únicas → Match ya presentes en BD.

        Permite distinguir un duplicado (mismo fixture ya importado) de un
        fixture ``scheduled`` cuyo resultado aún no se registró: en ese caso
        el importador actualiza la fila en vez de duplicar.
        """
        rows = self._session.execute(select(Match)).scalars().all()
        return {
            (m.competition_id, m.season_id, m.home_team_id, m.away_team_id, m.date): m
            for m in rows
        }

    def add(
        self,
        *,
        competition_id: int,
        season_id: int,
        date: date,
        home_team_id: int,
        away_team_id: int,
        home_goals: int | None,
        away_goals: int | None,
        status: str = "finished",
        source: str = "csv",
    ) -> Match:
        match = Match(
            competition_id=competition_id,
            season_id=season_id,
            date=date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
            status=status,
            source=source,
        )
        self._session.add(match)
        return match

    def find_by_teams_and_date(self, date: date, home_team_id: int, away_team_id: int) -> Match | None:
        return self._session.scalar(
            select(Match).where(
                Match.date == date,
                Match.home_team_id == home_team_id,
                Match.away_team_id == away_team_id,
            )
        )

    def find_next_scheduled(
        self, home_team_id: int, away_team_id: int, start: date
    ) -> Match | None:
        """Primer partido programado entre dos equipos desde `start` (inclusive)."""
        return self._session.scalar(
            select(Match)
            .where(
                Match.home_team_id == home_team_id,
                Match.away_team_id == away_team_id,
                Match.date >= start,
            )
            .order_by(Match.date)
            .limit(1)
        )

    def upcoming(self, team_id: int, start: date, limit: int = 4) -> list[Match]:
        """Próximos partidos de un equipo (juega de local o visitante)."""
        rows = self._session.execute(
            select(Match)
            .where(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
                Match.date >= start,
            )
            .order_by(Match.date, Match.id)
            .limit(limit)
        ).scalars()
        return list(rows)