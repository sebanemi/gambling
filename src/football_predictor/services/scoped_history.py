"""Alcance por competición del histórico (same-league).

Un :class:`HistoryProvider` decorado que solo deja pasar partidos de una
competición. Entrenar y armar features solo con la liga del partido a
predecir evita que ligas inconexas (p.ej. Premier League vs Liga
Profesional, equipos que nunca se cruzan) contaminen Elo/forma/Promedios
y los priors del Poisson/ML.

Sin leakage nuevo: solo *reduce* el histórico visible, nunca agrega
partidos futuros (la puerta por fecha sigue intacta abajo).
"""

from datetime import date

from football_predictor.domain.entities import HistoricalMatch
from football_predictor.domain.protocols import HistoryProvider


class CompetitionScopedHistory:
    """Filtra un provider a una sola ``competition_id``."""

    def __init__(self, inner: HistoryProvider, competition_id: int) -> None:
        self._inner = inner
        self._competition_id = int(competition_id)

    @property
    def competition_id(self) -> int:
        return self._competition_id

    def _scoped(self, matches: list[HistoricalMatch]) -> list[HistoricalMatch]:
        return [m for m in matches if m.competition_id == self._competition_id]

    def get_matches_before(self, cutoff: date) -> list[HistoricalMatch]:
        return self._scoped(self._inner.get_matches_before(cutoff))

    def get_all_matches(self) -> list[HistoricalMatch]:
        return self._scoped(self._inner.get_all_matches())
