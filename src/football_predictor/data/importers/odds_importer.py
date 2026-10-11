"""Importer de cuotas (odds) desde CSV.

Formato esperado (header exacto):

    date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds

- ``date``: YYYY-MM-DD (fecha del partido tal como está en BD).
- ``home_team`` / ``away_team``: nombres resueltos con tolerancia
  (``TeamRepository.resolve``).
- ``bookmaker``: nombre libre (p.ej. ``bet365``, ``pinnacle``).
- ``*_odds``: cuotas decimales > 1.0.

Cada fila hace upsert en ``match_odds`` (match_id + bookmaker).
Las filas cuyo partido no existe en BD se cuentan como ``unknown_matches``
(en vez de fallar toda la importación).
"""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from football_predictor.database.repositories.match import MatchRepository
from football_predictor.database.repositories.match_odds import MatchOddsRepository
from football_predictor.database.repositories.team import TeamRepository
from football_predictor.utils.logging import get_logger

log = get_logger("data.importers.odds")

_REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "bookmaker",
    "home_odds",
    "draw_odds",
    "away_odds",
)


@dataclass
class OddsImportSummary:
    rows_read: int = 0
    upserted: int = 0
    invalid: int = 0
    unknown_matches: int = 0


class OddsImporter:
    """Importa cuotas desde un CSV a la sesión provista (commit único al final)."""

    def __init__(self, session: Session, path: Path | str) -> None:
        self._session = session
        self._path = Path(path)

    def import_all(self) -> OddsImportSummary:
        if not self._path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self._path}")

        summary = OddsImportSummary()
        teams = TeamRepository(self._session)
        matches = MatchRepository(self._session)
        odds_repo = MatchOddsRepository(self._session)

        with self._path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV vacío: {self._path}")
            missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"faltan columnas requeridas en {self._path}: {sorted(missing)}")

            for raw in reader:
                if not raw or not any(v and str(v).strip() for v in raw.values()):
                    continue
                summary.rows_read += 1
                try:
                    match_date = date.fromisoformat(str(raw["date"]).strip())
                    bookmaker = str(raw["bookmaker"]).strip()
                    home_odds = float(str(raw["home_odds"]).strip())
                    draw_odds = float(str(raw["draw_odds"]).strip())
                    away_odds = float(str(raw["away_odds"]).strip())
                    if not bookmaker:
                        raise ValueError("bookmaker vacío")
                    for value in (home_odds, draw_odds, away_odds):
                        if value <= 1.0:
                            raise ValueError(f"cuota decimal debe ser > 1.0 (got {value})")
                except (ValueError, TypeError, KeyError) as exc:
                    log.warning("fila %d inválida: %s", summary.rows_read, exc)
                    summary.invalid += 1
                    continue

                home_id = teams.resolve(str(raw["home_team"]))
                away_id = teams.resolve(str(raw["away_team"]))
                if home_id is None or away_id is None:
                    summary.unknown_matches += 1
                    continue
                match = matches.find_by_teams_and_date(match_date, home_id, away_id)
                if match is None:
                    summary.unknown_matches += 1
                    continue

                odds_repo.upsert(match.id, bookmaker, home_odds, draw_odds, away_odds)
                summary.upserted += 1

        self._session.commit()
        return summary


__all__ = ["OddsImportSummary", "OddsImporter"]
