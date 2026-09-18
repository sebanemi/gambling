"""Modelo de ML multiclase sobre las features aplanadas (§15 de la especificación).

Pipeline estable: ``StandardScaler`` + ``LogisticRegression`` (multinomial).
Las columnas son el orden determinista de ``FeatureVector.flatten()``, de
modo que entrenamiento y predicción siempre alinean. Del mismo FeatureVector
sale, además, el snapshot auditable de la predicción.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from football_predictor.domain.features import FeatureVector
from football_predictor.models.base import Outcome, PredictionResult


class MLModel:
    def __init__(self, random_state: int = 42, max_iter: int = 2000) -> None:
        self._random_state = int(random_state)
        self._max_iter = int(max_iter)
        self._pipeline: object | None = None
        self._classes: list[int] | None = None
        self._columns = list(_template().flatten().keys())
        self._active_columns: list[int] | None = None

    def fit(self, training_features: Sequence[FeatureVector], outcomes: Sequence[Outcome]) -> None:
        if len(training_features) != len(outcomes):
            raise ValueError("training_features y outcomes deben tener el mismo largo")
        if not training_features:
            raise ValueError("MLModel.fit requiere al menos un ejemplo")

        x = self._matrix(training_features, apply_mask=False)
        y = np.array([int(o) for o in outcomes], dtype=int)
        self.fit_matrix(x, y)

    @classmethod
    def make_matrix(cls, training_features: Sequence[FeatureVector]) -> np.ndarray:
        """Matriz (N x columnas) en el orden fijo de columnas del modelo."""
        if not training_features:
            raise ValueError("make_matrix requiere al menos un ejemplo")
        columns = list(_template().flatten().keys())
        return np.array([[fv.flatten()[column] for column in columns] for fv in training_features], dtype=float)

    def fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Entrena desde una matriz ya construida (sobre las mismas columnas)."""
        if x.shape[0] != y.shape[0]:
            raise ValueError("x e y deben tener la misma cantidad de filas")
        if x.shape[0] == 0:
            raise ValueError("MLModel.fit_matrix requiere al menos un ejemplo")

        if len(np.unique(y)) < 2:
            raise ValueError("MLModel.fit_matrix requiere al menos dos resultados distintos (1X2) en el historial")

        # Columnas con varianza nula (historial pequeño, features siempre 0)
        # anularían el StandardScaler con NaN: se descartan de forma fija.
        std = x.std(axis=0)
        self._active_columns = [i for i, value in enumerate(std) if value > 0]
        if not self._active_columns:
            raise ValueError("todas las features del entrenamiento son constantes; el ML no puede aprender")

        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=self._max_iter, random_state=self._random_state),
        )
        pipeline.fit(x[:, self._active_columns], y)
        self._pipeline = pipeline
        self._classes = [int(c) for c in pipeline.classes_]

    def _matrix(self, features: Sequence[FeatureVector], apply_mask: bool = True) -> np.ndarray:
        matrix = np.array([[fv.flatten()[column] for column in self._columns] for fv in features], dtype=float)
        if apply_mask and self._active_columns is not None:
            matrix = matrix[:, self._active_columns]
        return matrix

    def predict(self, features: FeatureVector) -> PredictionResult:
        if self._pipeline is None:
            raise ValueError("MLModel.predict requiere fit() previo")

        row = self._matrix([features])
        probabilities = (self._pipeline.predict_proba(row))[0]

        order = self._classes or []
        proba = {Outcome(int(c)): float(probabilities[i]) for i, c in enumerate(order)}

        return PredictionResult(
            match_id=features.match_id,
            home_win=proba.get(Outcome.HOME, 0.0),
            draw=proba.get(Outcome.DRAW, 0.0),
            away_win=proba.get(Outcome.AWAY, 0.0),
            features_snapshot=features.flatten(),
        )


def _template() -> FeatureVector:
    """FeatureVector todo-cero usado sólo para fijar el orden de columnas."""
    from datetime import date

    from football_predictor.domain.features import TeamForm

    return FeatureVector(
        match_id=0,
        home_team_id=0,
        away_team_id=0,
        date=date(2000, 1, 1),
        home_elo=0.0,
        away_elo=0.0,
        elo_difference=0.0,
        home_form=TeamForm(),
        away_form=TeamForm(),
        home_goals_for_avg=0.0,
        home_goals_against_avg=0.0,
        away_goals_for_avg=0.0,
        away_goals_against_avg=0.0,
        home_yellow_cards_for_avg=0.0,
        home_yellow_cards_against_avg=0.0,
        away_yellow_cards_for_avg=0.0,
        away_yellow_cards_against_avg=0.0,
        home_red_cards_for_avg=0.0,
        home_red_cards_against_avg=0.0,
        away_red_cards_for_avg=0.0,
        away_red_cards_against_avg=0.0,
        home_corners_for_avg=0.0,
        home_corners_against_avg=0.0,
        away_corners_for_avg=0.0,
        away_corners_against_avg=0.0,
        home_advantage=0.0,
    )