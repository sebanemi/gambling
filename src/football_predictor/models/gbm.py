"""Gradient Boosting Model (GBM) for multiclass prediction.

Supports multiple backends: XGBoost, LightGBM, CatBoost.
Uses the same 128 features as MLModel for consistency.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from football_predictor.domain.features import FeatureVector, feature_columns, matrix_from_vectors
from football_predictor.models.base import Outcome, PredictionResult


class GBMModel:
    """Gradient Boosting Model with configurable backend."""

    def __init__(
        self,
        backend: str = "lightgbm",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ) -> None:
        """Initialize GBM model.

        Args:
            backend: One of "lightgbm", "xgboost", "catboost"
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (shrinkage)
            random_state: Random seed for reproducibility
            class_weight: "balanced" pondera cada muestra con 1/frecuencia de
                su clase (corrige el sesgo anti-empate); "none"/None lo desactiva.
        """
        self._backend = backend.lower()
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._learning_rate = float(learning_rate)
        self._random_state = int(random_state)
        self._class_weight = None if class_weight in (None, "none") else class_weight

        self._model: object | None = None
        self._scaler: StandardScaler | None = None
        self._columns = list(feature_columns())
        self._active_columns: list[int] | None = None

        if self._backend not in ("lightgbm", "xgboost", "catboost"):
            raise ValueError(f"Unsupported backend: {backend}. Use 'lightgbm', 'xgboost', or 'catboost'")

    def fit(self, training_features: Sequence[FeatureVector], outcomes: Sequence[Outcome]) -> None:
        """Train the GBM model on feature vectors and outcomes."""
        if len(training_features) != len(outcomes):
            raise ValueError("training_features y outcomes deben tener el mismo largo")
        if not training_features:
            raise ValueError("GBMModel.fit requiere al menos un ejemplo")

        x = self._matrix(training_features, apply_mask=False)
        y = np.array([int(o) for o in outcomes], dtype=int)
        self.fit_matrix(x, y)

    @classmethod
    def make_matrix(cls, training_features: Sequence[FeatureVector]) -> np.ndarray:
        """Create feature matrix in the fixed column order."""
        if not training_features:
            raise ValueError("make_matrix requiere al menos un ejemplo")
        return matrix_from_vectors(training_features)

    def fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train from a pre-built feature matrix."""
        if x.shape[0] != y.shape[0]:
            raise ValueError("x e y deben tener la misma cantidad de filas")
        if x.shape[0] == 0:
            raise ValueError("GBMModel.fit_matrix requiere al menos un ejemplo")

        if len(np.unique(y)) < 2:
            raise ValueError("GBMModel.fit_matrix requiere al menos dos resultados distintos (1X2) en el historial")

        # Handle constant columns (same as MLModel)
        std = x.std(axis=0)
        self._active_columns = [i for i, value in enumerate(std) if value > 0]
        if not self._active_columns:
            raise ValueError("todas las features del entrenamiento son constantes; el GBM no puede aprender")

        x_active = x[:, self._active_columns]

        # Scale features (important for XGBoost, optional for others but good practice)
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x_active)

        sample_weight = self._balanced_sample_weight(y)

        # Initialize the appropriate backend
        if self._backend == "lightgbm":
            self._model = self._init_lightgbm(x_scaled, y)
        elif self._backend == "xgboost":
            self._model = self._init_xgboost(x_scaled, y)
        elif self._backend == "catboost":
            self._model = self._init_catboost(x_scaled, y)
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

        # Fit the model
        if sample_weight is not None:
            try:
                self._model.fit(x_scaled, y, sample_weight=sample_weight)
            except TypeError:
                # Backend sin soporte de sample_weight: entrena sin ponderar.
                self._model.fit(x_scaled, y)
        else:
            self._model.fit(x_scaled, y)

    def _balanced_sample_weight(self, y: np.ndarray) -> np.ndarray | None:
        """Pesos 1/frecuencia por clase (None si balanceo desactivado)."""
        if self._class_weight != "balanced":
            return None
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return None
        weight_by_class = {int(c): len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
        return np.array([weight_by_class[int(v)] for v in y], dtype=float)

    def _init_lightgbm(self, x: np.ndarray, y: np.ndarray) -> object:
        """Initialize LightGBM classifier."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is not installed. Install with: pip install lightgbm")

        return lgb.LGBMClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            verbose=-1,  # suppress output
        )

    def _init_xgboost(self, x: np.ndarray, y: np.ndarray) -> object:
        """Initialize XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")

        return xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            objective="multi:softprob",
            num_class=3,
        )

    def _init_catboost(self, x: np.ndarray, y: np.ndarray) -> object:
        """Initialize CatBoost classifier."""
        try:
            import catboost as cb
        except ImportError:
            raise ImportError("catboost is not installed. Install with: pip install catboost")

        return cb.CatBoostClassifier(
            iterations=self._n_estimators,
            depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_seed=self._random_state,
            loss_function="MultiClass",
            verbose=False,  # suppress output
        )

    def _matrix(self, features: Sequence[FeatureVector], apply_mask: bool = True) -> np.ndarray:
        """Convert features to matrix."""
        matrix = matrix_from_vectors(features)
        if apply_mask and self._active_columns is not None:
            matrix = matrix[:, self._active_columns]
        return matrix

    def predict(self, features: FeatureVector, *, include_snapshot: bool = True) -> PredictionResult:
        """Predict match outcome probabilities."""
        if self._model is None or self._scaler is None:
            raise ValueError("GBMModel.predict requiere fit() previo")

        row = self._matrix([features])
        row_scaled = self._scaler.transform(row)

        probabilities = self._model.predict_proba(row_scaled)[0]

        # Mapeo por classes_ (no posicional): el orden de columnas de
        # predict_proba sigue a model.classes_, que puede no ser [0,1,2]
        # (p.ej. pasos walk-forward tempranos con solo 2 clases vistas).
        if hasattr(self._model, "classes_"):
            classes = list(self._model.classes_)
            proba_dict = {Outcome(int(cls)): float(prob) for cls, prob in zip(classes, probabilities)}
        elif len(probabilities) == 3:
            proba_dict = {
                Outcome.HOME: float(probabilities[0]),
                Outcome.DRAW: float(probabilities[1]),
                Outcome.AWAY: float(probabilities[2]),
            }
        else:
            raise ValueError("Model returned unexpected probability shape")

        return PredictionResult(
            match_id=features.match_id,
            home_win=proba_dict.get(Outcome.HOME, 0.0),
            draw=proba_dict.get(Outcome.DRAW, 0.0),
            away_win=proba_dict.get(Outcome.AWAY, 0.0),
            features_snapshot=features.flatten() if include_snapshot else None,
        )
