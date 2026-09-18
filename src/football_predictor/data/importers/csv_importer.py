"""Importer CSV: especialización del importer genérico para datasets locales.

El conteo de filas/errores proviene de ``CsvDataProvider.load_result``;
el resto (competencias, temporadas, equipos, dedup) lo hereda de
:class:`MatchImporter`.
"""

from sqlalchemy.orm import Session

from football_predictor.data.importers.match_importer import ImportSummary, MatchImporter
from football_predictor.data.providers.csv_provider import CsvDataProvider
from football_predictor.domain.entities import MatchRecord
from football_predictor.utils.logging import get_logger

log = get_logger("data.importers.csv")


class CsvImporter(MatchImporter):
    """Importa un dataset CSV en la sesión provista (commit único al final)."""

    def __init__(self, session: Session, provider: CsvDataProvider) -> None:
        super().__init__(session, provider, source="csv")

    def _records(self) -> tuple[list[MatchRecord], int, int]:
        result = self._provider.load_result()
        return result.records, result.rows_read, len(result.invalid)


__all__ = ["CsvImporter", "ImportSummary"]