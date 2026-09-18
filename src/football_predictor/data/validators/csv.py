"""Validación estricta de filas del CSV de partidos.

Valida una fila en crudo (dict de strings) y produce un
:class:`CsvMatchRow` tipado. Cualquier fila que no cumpla el contrato
genera :class:`CsvValidationError` con el motivo; el importador cuenta
las inválidas y sigue.
"""

from dataclasses import dataclass
from datetime import date, datetime

from pydantic import BaseModel, Field

# Formatos soportados de fecha. Documentar en README; ISO-8601 es el default.
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")


class CsvMatchRow(BaseModel):
    """Fila ya parseada y tipada. Los valores llegan estrictos del parser."""

    date: date
    competition: str = Field(min_length=1)
    season: str = Field(min_length=1)
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    home_yellow_cards: int | None = Field(default=None, ge=0)
    away_yellow_cards: int | None = Field(default=None, ge=0)
    home_red_cards: int | None = Field(default=None, ge=0)
    away_red_cards: int | None = Field(default=None, ge=0)
    home_corners: int | None = Field(default=None, ge=0)
    away_corners: int | None = Field(default=None, ge=0)
    home_shots: int | None = Field(default=None, ge=0)
    away_shots: int | None = Field(default=None, ge=0)
    home_shots_on_target: int | None = Field(default=None, ge=0)
    away_shots_on_target: int | None = Field(default=None, ge=0)
    home_fouls: int | None = Field(default=None, ge=0)
    away_fouls: int | None = Field(default=None, ge=0)
    home_throw_ins: int | None = Field(default=None, ge=0)
    away_throw_ins: int | None = Field(default=None, ge=0)
    home_penalties: int | None = Field(default=None, ge=0)
    away_penalties: int | None = Field(default=None, ge=0)
    home_xg: float | None = Field(default=None, ge=0)
    away_xg: float | None = Field(default=None, ge=0)
    home_possession: float | None = Field(default=None, ge=0, le=100)
    away_possession: float | None = Field(default=None, ge=0, le=100)


@dataclass(frozen=True)
class CsvValidationError(ValueError):
    row_number: int
    message: str

    def __str__(self) -> str:
        return f"row {self.row_number}: {self.message}"


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"fecha inválida '{value}' (usar YYYY-MM-DD, DD/MM/YYYY o DD-MM-YYYY)")


def _parse_goals(value: str, label: str) -> int:
    value = value.strip()
    if not value.isdigit():
        raise ValueError(f"{label} debe ser un entero no negativo, got '{value}'")
    return int(value)


class CsvRowValidator:
    """Valida una fila cruda del CSV y la convierte en :class:`CsvMatchRow`."""

    def parse(self, raw: dict[str, str], row_number: int) -> CsvMatchRow:
        try:
            for required in ("date", "competition", "season", "home_team", "away_team", "home_goals", "away_goals"):
                if not raw.get(required, "").strip():
                    raise ValueError(f"columna '{required}' vacía")

            if raw["home_team"].strip().lower() == raw["away_team"].strip().lower():
                raise ValueError("home_team y away_team son el mismo equipo")

            return CsvMatchRow(
                date=_parse_date(raw["date"]),
                competition=raw["competition"],
                season=raw["season"],
                home_team=raw["home_team"],
                away_team=raw["away_team"],
                home_goals=_parse_goals(raw["home_goals"], "home_goals"),
                away_goals=_parse_goals(raw["away_goals"], "away_goals"),
                home_yellow_cards=self._parse_optional_int(raw.get("home_yellow_cards"), "home_yellow_cards"),
                away_yellow_cards=self._parse_optional_int(raw.get("away_yellow_cards"), "away_yellow_cards"),
                home_red_cards=self._parse_optional_int(raw.get("home_red_cards"), "home_red_cards"),
                away_red_cards=self._parse_optional_int(raw.get("away_red_cards"), "away_red_cards"),
                home_corners=self._parse_optional_int(raw.get("home_corners"), "home_corners"),
                away_corners=self._parse_optional_int(raw.get("away_corners"), "away_corners"),
                home_shots=self._parse_optional_int(raw.get("home_shots"), "home_shots"),
                away_shots=self._parse_optional_int(raw.get("away_shots"), "away_shots"),
                home_shots_on_target=self._parse_optional_int(raw.get("home_shots_on_target"), "home_shots_on_target"),
                away_shots_on_target=self._parse_optional_int(raw.get("away_shots_on_target"), "away_shots_on_target"),
                home_fouls=self._parse_optional_int(raw.get("home_fouls"), "home_fouls"),
                away_fouls=self._parse_optional_int(raw.get("away_fouls"), "away_fouls"),
                home_throw_ins=self._parse_optional_int(raw.get("home_throw_ins"), "home_throw_ins"),
                away_throw_ins=self._parse_optional_int(raw.get("away_throw_ins"), "away_throw_ins"),
                home_penalties=self._parse_optional_int(raw.get("home_penalties"), "home_penalties"),
                away_penalties=self._parse_optional_int(raw.get("away_penalties"), "away_penalties"),
                home_xg=self._parse_optional_float(raw.get("home_xg"), "home_xg"),
                away_xg=self._parse_optional_float(raw.get("away_xg"), "away_xg"),
                home_possession=self._parse_optional_float(raw.get("home_possession"), "home_possession"),
                away_possession=self._parse_optional_float(raw.get("away_possession"), "away_possession"),
            )
        except CsvValidationError:
            raise
        except ValueError as exc:
            raise CsvValidationError(row_number=row_number, message=str(exc)) from exc

    @staticmethod
    def _parse_optional_int(value: str | None, label: str) -> int | None:
        if value is None:
            return None
        v = value.strip()
        if not v:
            return None
        # Accept float strings like "4.0" and convert to int
        try:
            return int(float(v))
        except ValueError:
            raise ValueError(f"{label} debe ser un número, got '{value}'")

    @staticmethod
    def _parse_optional_float(value: str | None, label: str) -> float | None:
        if value is None:
            return None
        v = value.strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"{label} debe ser un número, got '{value}'")