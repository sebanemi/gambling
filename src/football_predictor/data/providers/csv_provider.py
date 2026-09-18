"""Provider CSV: implementa ``FootballDataProvider`` para datasets locales.

Pipeline por fila: leer en crudo -> validar -> normalizar nombres ->
producir :class:`MatchRecord`. Las filas inválidas no se descartan en
silencio: se cuentan y reportan (resumen del importer).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from football_predictor.data.normalizers import NameNormalizer
from football_predictor.data.validators.csv import CsvMatchRow, CsvRowValidator, CsvValidationError
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo

_REQUIRED_COLUMNS = ("date", "competition", "season", "home_team", "away_team", "home_goals", "away_goals")


class CsvSchemaError(ValueError):
    """El CSV no tiene el set de columnas requerido."""


@dataclass
class CsvLoadResult:
    records: list[MatchRecord] = field(default_factory=list)
    rows_read: int = 0
    invalid: list[CsvValidationError] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.records)


class CsvDataProvider:
    """Lee un CSV de partidos respetando el pipeline validar+normalizar."""

    def __init__(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        name_normalizer: NameNormalizer | None = None,
    ) -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._normalizer = name_normalizer or NameNormalizer()
        self._validator = CsvRowValidator()
        self._result: CsvLoadResult | None = None

    def _load(self) -> CsvLoadResult:
        if self._result is not None:
            return self._result

        if not self._path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self._path}")

        with self._path.open(newline="", encoding=self._encoding) as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise CsvSchemaError(f"CSV vacío: {self._path}")

            missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise CsvSchemaError(
                    f"faltan columnas requeridas en {self._path}: {sorted(missing)}"
                )

            records: list[MatchRecord] = []
            invalid: list[CsvValidationError] = []
            rows_read = 0

            for raw in reader:
                if not any(str(v).strip() for v in raw.values()):
                    continue  # línea en blanco de cierre
                rows_read += 1
                try:
                    row = self._validator.parse(raw, rows_read + 1)  # +1 = header
                except CsvValidationError as exc:
                    invalid.append(exc)
                    continue
                records.append(self._to_record(row))

        self._result = CsvLoadResult(records=records, rows_read=rows_read, invalid=invalid)
        return self._result

    def _to_record(self, row: CsvMatchRow) -> MatchRecord:
        return MatchRecord(
            competition_name=self._normalizer(row.competition),
            season_name=self._normalizer(row.season),
            date=row.date,
            home_team=self._normalizer(row.home_team),
            away_team=self._normalizer(row.away_team),
            home_goals=row.home_goals,
            away_goals=row.away_goals,
            status="finished",
            home_yellow_cards=getattr(row, "home_yellow_cards", None),
            away_yellow_cards=getattr(row, "away_yellow_cards", None),
            home_red_cards=getattr(row, "home_red_cards", None),
            away_red_cards=getattr(row, "away_red_cards", None),
            home_corners=getattr(row, "home_corners", None),
            away_corners=getattr(row, "away_corners", None),
            home_shots=getattr(row, "home_shots", None),
            away_shots=getattr(row, "away_shots", None),
            home_shots_on_target=getattr(row, "home_shots_on_target", None),
            away_shots_on_target=getattr(row, "away_shots_on_target", None),
            home_fouls=getattr(row, "home_fouls", None),
            away_fouls=getattr(row, "away_fouls", None),
            home_throw_ins=getattr(row, "home_throw_ins", None),
            away_throw_ins=getattr(row, "away_throw_ins", None),
            home_penalties=getattr(row, "home_penalties", None),
            away_penalties=getattr(row, "away_penalties", None),
            home_xg=getattr(row, "home_xg", None),
            away_xg=getattr(row, "away_xg", None),
            home_possession=getattr(row, "home_possession", None),
            away_possession=getattr(row, "away_possession", None),
        )

    # --- FootballDataProvider ---

    def get_matches(self) -> list[MatchRecord]:
        return list(self._load().records)

    def get_teams(self) -> list[TeamInfo]:
        teams: dict[tuple[str, str], TeamInfo] = {}
        for record in self._load().records:
            for name in (record.home_team, record.away_team):
                teams.setdefault((name, ""), TeamInfo(name=name))
        return list(teams.values())

    def get_competitions(self) -> list[CompetitionInfo]:
        competitions: dict[str, CompetitionInfo] = {}
        for record in self._load().records:
            competitions.setdefault(
                record.competition_name, CompetitionInfo(name=record.competition_name)
            )
        return list(competitions.values())

    def load_result(self) -> CsvLoadResult:
        return self._load()