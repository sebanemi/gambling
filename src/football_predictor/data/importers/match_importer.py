"""Importer genérico: materializa partidos de cualquier ``FootballDataProvider``.

Orquesta repositories dentro de una única transacción y produce un
:class:`ImportSummary` auditable. Los duplicados se detectan contra la
clave única de ``matches`` (ya presente en BD o repetida en la misma
fuente). El provider CSV conserva su resumen filas/errores vía ``_records``.
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from football_predictor.database.repositories.competition import CompetitionRepository
from football_predictor.database.repositories.match import MatchRepository
from football_predictor.database.repositories.match_statistics import MatchStatisticsRepository
from football_predictor.database.repositories.season import SeasonRepository
from football_predictor.database.repositories.team import TeamRepository
from football_predictor.domain.entities import MatchRecord
from football_predictor.utils.logging import get_logger

log = get_logger("data.importers.match")


@dataclass(frozen=True)
class ImportSummary:
    rows_read: int
    valid: int
    invalid: int
    new_teams: int
    new_competitions: int
    new_seasons: int
    inserted_matches: int
    duplicates: int
    updated_matches: int = 0
    updated_statistics: int = 0


class MatchImporter:
    """Importa partidos de un provider en la sesión (commit único al final)."""

    def __init__(
        self,
        session: Session,
        provider: object,
        *,
        source: str | None = None,
    ) -> None:
        self._session = session
        self._provider = provider
        self._source = source or getattr(provider, "source", "api")
        self._teams_repo = TeamRepository(session)
        self._competitions_repo = CompetitionRepository(session)
        self._seasons_repo = SeasonRepository(session)
        self._matches_repo = MatchRepository(session)
        self._stats_repo = MatchStatisticsRepository(session)

    def _records(self) -> tuple[list[MatchRecord], int, int]:
        """records, rows_read, invalid. Cada provider puede enriquecerlo."""
        records = list(self._provider.get_matches())
        return records, len(records), 0

    def import_all(self) -> ImportSummary:
        records, rows_read, invalid = self._records()

        teams = self._teams_repo.all_ids()
        competitions = self._competitions_repo.all_ids()
        seasons = self._seasons_repo.all_ids()
        existing = self._matches_repo.existing()

        new_teams = 0
        new_competitions = 0
        new_seasons = 0
        inserted = 0
        duplicates = 0
        updated = 0
        stats_updated = 0
        seen: set[tuple[int, int, int, int, date]] = set()
        season_date_ranges: dict[int, list[date]] = {}

        for record in records:
            competition_id, comp_created = self._competition_id(competitions, record)
            season_id, season_created = self._season_id(
                seasons, competition_id, record
            )
            home_id, home_created = self._team_id(teams, record.home_team)
            away_id, away_created = self._team_id(teams, record.away_team)

            new_teams += home_created + away_created
            new_competitions += comp_created
            new_seasons += season_created

            key = (competition_id, season_id, home_id, away_id, record.date)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)

            current = existing.get(key)
            has_score = record.home_goals is not None and record.away_goals is not None
            if current is not None:
                # Fixture ya presente: si era scheduled y ahora hay marcador,
                # se actualiza en lugar de duplicar. Si ya tenía resultado (o
                # sigue sin marcador), se actualizan stats si vienen.
                was_scheduled = current.home_goals is None or current.away_goals is None
                if has_score and was_scheduled:
                    current.home_goals = record.home_goals
                    current.away_goals = record.away_goals
                    current.status = record.status or "finished"
                    updated += 1
                    season_date_ranges.setdefault(season_id, []).append(record.date)
                # Siempre actualizar estadísticas si vienen en el registro (incluso si ya tenía resultado)
                if any(
                    getattr(record, f, None) is not None
                    for f in (
                        "home_yellow_cards", "away_yellow_cards",
                        "home_red_cards", "away_red_cards",
                        "home_corners", "away_corners",
                        "home_shots", "away_shots",
                        "home_shots_on_target", "away_shots_on_target",
                        "home_fouls", "away_fouls",
                        "home_throw_ins", "away_throw_ins",
                        "home_penalties", "away_penalties",
                        "home_xg", "away_xg",
                        "home_possession", "away_possession",
                    )
                ):
                    self._upsert_stats(current.id, record)
                    stats_updated += 1
                if not (has_score and was_scheduled):
                    duplicates += 1
                continue

            # Ningún partido sin marcador se descarta: los fixtures scheduled
            # se importan (status preservado) para poder predecirlos antes de
            # jugarse. Quedan fuera del histórico de features/modelos porque
            # MatchHistoryRepository solo filtra finished con goles.
            match = self._matches_repo.add(
                competition_id=competition_id,
                season_id=season_id,
                date=record.date,
                home_team_id=home_id,
                away_team_id=away_id,
                home_goals=record.home_goals,
                away_goals=record.away_goals,
                status=record.status or "finished",
                source=self._source,
            )
            inserted += 1
            season_date_ranges.setdefault(season_id, []).append(record.date)

            # Estadísticas del nuevo partido
            if any(
                getattr(record, f, None) is not None
                for f in (
                    "home_yellow_cards", "away_yellow_cards",
                    "home_red_cards", "away_red_cards",
                    "home_corners", "away_corners",
                )
            ):
                self._session.flush()
                self._upsert_stats(match.id, record)
                stats_updated += 1

        self._session.flush()
        self._update_season_ranges(season_date_ranges)
        self._session.commit()

        log.info(
            "Import %s: rows=%d inserted=%d updated=%d duplicates=%d stats=%d",
            self._source,
            rows_read,
            inserted,
            updated,
            duplicates,
            stats_updated,
        )
        return ImportSummary(
            rows_read=rows_read,
            valid=len(records),
            invalid=invalid,
            new_teams=new_teams,
            new_competitions=new_competitions,
            new_seasons=new_seasons,
            inserted_matches=inserted,
            duplicates=duplicates,
            updated_matches=updated,
            updated_statistics=stats_updated,
        )

    def _upsert_stats(self, match_id: int, record: MatchRecord) -> None:
        self._stats_repo.upsert(
            match_id,
            home_yellow_cards=record.home_yellow_cards,
            away_yellow_cards=record.away_yellow_cards,
            home_red_cards=record.home_red_cards,
            away_red_cards=record.away_red_cards,
            home_corners=record.home_corners,
            away_corners=record.away_corners,
        )

    def import_masters(self) -> tuple[int, int]:
        """Importa solo competiciones y equipos (fuentes sin partidos:
        standings, playerelo). Devuelve (competencias creadas, equipos creados)."""
        competitions = self._competitions_repo.all_ids()
        teams = self._teams_repo.all_ids()
        new_competitions = 0
        new_teams = 0

        for info in self._provider.get_competitions():
            key = (info.name, info.country)
            if key not in competitions:
                self._competitions_repo.create(name=info.name, country=info.country, type=info.type)
                competitions[key] = 0
                new_competitions += 1

        for info in self._provider.get_teams():
            key = (info.name, info.country)
            if key not in teams:
                self._teams_repo.create(name=info.name, country=info.country, source=info.source or self._source)
                teams[key] = 0
                new_teams += 1

        self._session.commit()
        return new_competitions, new_teams

    def _competition_id(
        self,
        cache: dict[tuple[str, str], int],
        record: MatchRecord,
    ) -> tuple[int, bool]:
        key = (record.competition_name, "")
        competition_id = cache.get(key)
        if competition_id is None:
            competition_id = self._competitions_repo.create(name=record.competition_name)
            cache[key] = competition_id
            return competition_id, True
        return competition_id, False

    def _season_id(
        self,
        cache: dict[tuple[int, str], int],
        competition_id: int,
        record: MatchRecord,
    ) -> tuple[int, bool]:
        key = (competition_id, record.season_name)
        season_id = cache.get(key)
        if season_id is None:
            season_id = self._seasons_repo.create(
                competition_id=competition_id, name=record.season_name
            )
            cache[key] = season_id
            return season_id, True
        return season_id, False

    def _team_id(
        self, cache: dict[tuple[str, str], int], name: str
    ) -> tuple[int, bool]:
        key = (name, "")
        team_id = cache.get(key)
        if team_id is None:
            team_id = self._teams_repo.create(name=name)
            cache[key] = team_id
            return team_id, True
        return team_id, False

    def _update_season_ranges(self, ranges: dict[int, list[date]]) -> None:
        for season_id, dates in ranges.items():
            self._seasons_repo.update_dates(
                season_id=season_id, start_date=min(dates), end_date=max(dates)
            )