"""Ventaja de local como feature.

Constante configurable (puntos Elo), compartida con el sistema Elo.
Una versión derivada del histórico (tasa de victorias local por liga)
puede agregarse después sin cambiar la interfaz.
"""


class HomeAdvantage:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value