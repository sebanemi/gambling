"""Gradient Boosting Model (GBM) for multiclass prediction.

Supports multiple backends: XGBoost, LightGBM, CatBoost.
Uses the same 128 features as MLModel for consistency.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from football_predictor.domain.features import FeatureVector
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
    ) -> None:
        """Initialize GBM model.

        Args:
            backend: One of "lightgbm", "xgboost", "catboost"
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (shrinkage)
            random_state: Random seed for reproducibility
        """
        self._backend = backend.lower()
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._learning_rate = float(learning_rate)
        self._random_state = int(random_state)

        self._model: object | None = None
        self._scaler: StandardScaler | None = None
        self._columns = list(_template().flatten().keys())
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
        columns = list(_template().flatten().keys())
        return np.array([[fv.flatten()[column] for column in columns] for fv in training_features], dtype=float)

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
        self._model.fit(x_scaled, y)

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
        matrix = np.array([[fv.flatten()[column] for column in self._columns] for fv in features], dtype=float)
        if apply_mask and self._active_columns is not None:
            matrix = matrix[:, self._active_columns]
        return matrix

    def predict(self, features: FeatureVector) -> PredictionResult:
        """Predict match outcome probabilities."""
        if self._model is None or self._scaler is None:
            raise ValueError("GBMModel.predict requiere fit() previo")

        row = self._matrix([features])
        row_scaled = self._scaler.transform(row)

        probabilities = self._model.predict_proba(row_scaled)[0]

        # Ensure probabilities are in the correct order (HOME, DRAW, AWAY)
        # Some backends might return in different order
        if len(probabilities) == 3:
            proba_dict = {
                Outcome.HOME: float(probabilities[0]),
                Outcome.DRAW: float(probabilities[1]),
                Outcome.AWAY: float(probabilities[2]),
            }
        else:
            # Fallback: use model's classes_ attribute if available
            if hasattr(self._model, "classes_"):
                classes = self._model.classes_
                proba_dict = {Outcome(int(cls)): float(prob) for cls, prob in zip(classes, probabilities)}
            else:
                raise ValueError("Model returned unexpected probability shape")

        return PredictionResult(
            match_id=features.match_id,
            home_win=proba_dict.get(Outcome.HOME, 0.0),
            draw=proba_dict.get(Outcome.DRAW, 0.0),
            away_win=proba_dict.get(Outcome.AWAY, 0.0),
            features_snapshot=features.flatten(),
        )


def _template() -> FeatureVector:
    """FeatureVector template for column order."""
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
        home_shots_for_avg=0.0,
        home_shots_against_avg=0.0,
        away_shots_for_avg=0.0,
        away_shots_against_avg=0.0,
        home_shots_on_target_for_avg=0.0,
        home_shots_on_target_against_avg=0.0,
        away_shots_on_target_for_avg=0.0,
        away_shots_on_target_against_avg=0.0,
        home_fouls_for_avg=0.0,
        home_fouls_against_avg=0.0,
        away_fouls_for_avg=0.0,
        away_fouls_against_avg=0.0,
        home_throw_ins_for_avg=0.0,
        home_throw_ins_against_avg=0.0,
        away_throw_ins_for_avg=0.0,
        away_throw_ins_against_avg=0.0,
        home_penalties_for_avg=0.0,
        home_penalties_against_avg=0.0,
        away_penalties_for_avg=0.0,
        away_penalties_against_avg=0.0,
        home_xg_for_avg=0.0,
        home_xg_against_avg=0.0,
        away_xg_for_avg=0.0,
        away_xg_against_avg=0.0,
        home_possession_for_avg=0.0,
        home_possession_against_avg=0.0,
        away_possession_for_avg=0.0,
        away_possession_against_avg=0.0,
        home_advantage=0.0,
    )
