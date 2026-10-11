"""Configuración centralizada de la aplicación.

Los valores se leen desde variables de entorno y/o un archivo ``.env``
(soporte de pydantic-settings). Ningún módulo debe leer ``os.environ``
directamente; todos deben usar ``get_settings()``.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de configuración de la aplicación.

    Los defaults replican el entorno de docker-compose para que el
    proyecto funcione out-of-the-box sin ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- PostgreSQL ---
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "football_predictor"
    postgres_user: str = "football"
    postgres_password: str = "football"

    # --- Fuentes externas (Fase 7) ---
    football_api_key: str = ""
    football_api_base_url: str = ""

    # football-data.org (v4)
    football_data_api_key: str = ""
    football_data_base_url: str = "https://api.football-data.org/v4"
    # football-charts.com
    football_charts_api_key: str = ""
    football_charts_base_url: str = "https://footballcharts-backend.onrender.com/api/v1"
    # football98 (RapidAPI, host GiulianoCrescimbeni). Requiere X-RapidAPI-Key.
    football98_api_key: str = ""
    football98_host: str = "football98.p.rapidapi.com"
    # 5dollarfootballapi.com
    five_dollar_api_key: str = ""
    five_dollar_base_url: str = "https://api.5dollarfootballapi.com/v1"
    # SofaScore (sin clave)
    sofascore_base_url: str = "https://api.sofascore.com"
    # playerelo.football
    playerelo_api_key: str = ""
    playerelo_base_url: str = "https://data-api.playerelo.football"
    # azharimm/football-standings-api (ESPN, sin clave)
    standings_base_url: str = "https://football-standings-api.vercel.app"

    # --- Elo (Fase 3-4) ---
    elo_initial_rating: float = 1500.0
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 60.0

    # --- Features ---
    # Ventana para promedios de goles (None = todo el historial).
    features_goals_window: int | None = None
    # Ventana para forma como local / como visitante.
    features_home_away_window: int = 5
    # Ventana para promedios de estadísticas (tarjetas/córners).
    features_stats_window: int | None = None

    # --- Modelos ---
    # Factor Dixon-Coles del Poisson: None = estimar ρ por MLE junto al resto
    # (recomendado); un valor fijo (p.ej. 0.0) lo congela.
    models_poisson_rho: float | None = None
    # Regularización ridge sobre attack/defense del Poisson.
    models_poisson_regularization: float = 0.1
    # Modelo Elo: prob. de empate máxima y dispersión según diferencia.
    models_elo_draw_max: float = 0.30
    models_elo_draw_sigma: float = 300.0
    # Modelo ML: semilla fija para determinismo.
    models_ml_random_state: int = 42
    # Modelo ML: ponderación de clases. Medido en Liga Profesional (1362
    # pasos walk-forward): "balanced" deforma las probs y empeora log-loss
    # (1.11) y top-1; "none" calibra mejor (1.07) y acierta más (0.43).
    # Valores: "balanced" o "none".
    models_ml_class_weight: str = "none"
    # Modelo ML: C inverso de regularización (más chico = más regularizado).
    # Medido: C=1.0 -> log-loss 1.36; C=0.001 -> 1.07 (128 features,
    # historiales cortos al inicio del walk-forward).
    models_ml_c: float = 0.001
    # Modelo GBM: backend y hiperparámetros. Medido en Liga Profesional:
    # 100 árboles prof.6 sobreajustan (log-loss 1.31); pocos árboles
    # playos calibran mejor (20x3 -> 1.08). Sin ponderar clases, igual
    # que el ML (balanced deforma las probs).
    models_gbm_backend: str = "lightgbm"  # lightgbm, xgboost, catboost
    models_gbm_n_estimators: int = 20
    models_gbm_max_depth: int = 3
    models_gbm_learning_rate: float = 0.05
    models_gbm_random_state: int = 42
    # Modelo GBM: balanceo de clases ("balanced" = pesos inversos a la
    # frecuencia; "none" = sin ponderar, recomendado: ver arriba).
    models_gbm_class_weight: str = "none"
    # Pesos del ensemble "poisson,elo,ml,gbm" (se normalizan). Estimados por
    # log-loss walk-forward sobre el histórico disponible.
    models_ensemble_weights: str = "0.4,0.3,0.2,0.1"

    # --- Betting ---
    # Estrategia de apuestas
    betting_strategy: str = "kelly"  # fixed, kelly, proportional, confidence
    betting_initial_bankroll: float = 1000.0
    betting_kelly_fraction: float = 0.25  # fracción de Kelly para seguridad
    betting_fixed_stake: float = 10.0
    betting_proportional_percentage: float = 0.02  # 2% del bankroll
    betting_max_stake_percentage: float = 0.05  # 5% máximo por apuesta
    betting_value_threshold: float = 0.05  # 5% edge mínimo
    betting_min_odds: float = 1.1
    betting_max_odds: float = 10.0
    betting_backtest_min_matches: int = 10

    @property
    def database_url(self) -> str:
        """URL de conexión SQLAlchemy para PostgreSQL."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna la configuración, cacheadas por proceso."""
    return Settings()