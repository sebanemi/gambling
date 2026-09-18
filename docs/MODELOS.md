# Features y Modelos

Detalle técnico de cómo se construyen las features y qué calcula cada modelo.
Todo se entrena/predice **solo con histórico anterior a la fecha del partido**
(ver garantías en [ARQUITECTURA.md](ARQUITECTURA.md)).

## Entities clave (dominio puro)

- `HistoricalMatch`: partido finalizado con ids numéricos — única fuente de features.
- `TargetMatch`: el partido a predecir (`id`, `date`, `home_team_id`, `away_team_id`).
- `FeatureVector`: modelo Pydantic con todas las features de un partido.

## Features (`features/`)

### Elo (`elo.py`)

- Ratings **cero-sum** actualizados partido a partido en orden cronológico:
  `R' = R + K·(S − E)` con:
  - `E = 1 / (1 + 10^{(R_op − R_mio)/400})` (escala 400).
  - `S` = 1 local gana / 0.5 empate / 0 pierde (desde la perspectiva local).
  - El local juega con `home_advantage` pips extra.
- Equipos sin partidos previos usan `initial_rating`.
- Defaults (`.env`/settings): `ELO_INITIAL_RATING=1500`, `ELO_K_FACTOR=20`,
  `ELO_HOME_ADVANTAGE=60`.

### Forma (`form.py`)

Para cada equipo, con la ventana configurable `home_away_window=5` y las ventanas
generales `form_windows=(3, 5, 10)`:

- `overall_{3,5,10}`: últimas N apariciones; campo `home_5` / `away_5`: últimas 5
  como local / como visitante.
- Por ventana: `played, wins, draws, losses, points_per_game, goals_for,
  goals_against, goal_difference`.

### Goles (`goals.py`)

- Promedio `goals_for` y `goals_against` desde la perspectiva del equipo.
- `goals_window` (settings `FEATURES_GOALS_WINDOW`, default `None` = todo el historial).

### `FeatureVector.flatten()` → 88 columnas

8 base (`home_elo`, `away_elo`, `elo_difference`, `home_goals_for_avg`,
`home_goals_against_avg`, `away_goals_for_avg`, `away_goals_against_avg`,
`home_advantage`) + 2 lados × (3 ventanas overall + 2 contextos) × 8 campos.

Este dict plano es el `features_snapshot` persistido en `predictions`.

## Modelos (`models/`)

Contrato común: del `FeatureVector` devuelven un `PredictionResult` con
probabilidades **1X2 normalizadas a suma 1**, goles esperados (si aplica) y snapshot.

### Poisson (`poisson.py`)

Doble Poisson independiente con **ridge** y **corrección de empate**:

- `λ_home = exp(γ_home + attack[local] + defense[visita] + home_adv)`
- `λ_away = exp(γ_away + attack[visita] + defense[local])`
- Fit por máxima verosimilitud (`scipy.optimize.minimize`, método `L-BFGS-B`,
  máximo 2000 iteraciones) minimizando NLL + `λ·(Σ attack² + Σ defense²)`.
- Distribución conjunta en grilla 15×15; si `draw_correction ≠ 1.0` multiplica la
  diagonal (corrección tipo Dixon-Coles) y renormaliza.
- Devuelve goles esperados `(λ_home, λ_away)`.
- Parámetros (settings): `MODELS_POISSON_REGULARIZATION=0.1`,
  `MODELS_POISSON_DRAW_CORRECTION=1.0` (1.0 = sin corrección).

### Elo (`elo_model.py`)

Convierte el rating **pre-partido** (ya viene en las features) en probabilidades:

- `p_home = E(home_rating + home_adv, away_rating)`
- `P_draw = draw_max · exp(−diff² / (2σ²))` (gaussiana centrada en 0)
- `P_home = p_home·(1 − P_draw)`, `P_away = (1 − p_home)·(1 − P_draw)`
- Parámetros: `MODELS_ELO_DRAW_MAX=0.30`, `MODELS_ELO_DRAW_SIGMA=300`.
- No estima goles.

### ML (`ml.py`)

- **Regresión logística multinomial** (`StandardScaler → LogisticRegression`,
  `max_iter=2000`, `random_state=42`) sobre las 88 columnas.
- Descarta columnas de varianza nula (evita NaN en el scaler).
- Se entrena con features **anti-leakage** por partido (puerta temporal) y
  etiquetas `Outcome(1X2)` de los goles reales.
- No estima goles.

### Ensemble (`ensemble.py`)

- Media ponderada de probabilidades 1X2 de `[poisson, elo, ml]`.
- Pesos por defecto `0.4, 0.3, 0.3` (settings `MODELS_ENSEMBLE_WEIGHTS`, se normalizan).
- Goles esperados = promedio solo de los modelos que los producen (poisson).

### Claves y entrenador

- Claves: `poisson`, `elo`, `ml`, `ensemble` (+ `all` en `predict-match`/`backtest`).
- `ModelTrainer.train_all()` entrena los 4 y devuelve el dict.
- `parse_ensemble_weights()` exige exactamente 3 floats con suma > 0.

## Evaluación (`evaluation/`)

### Métricas (`metrics.py`)

Filosofía: medir **calidad del ranking 1X2**, no la magnitud de las probabilidades.

- `ranking_loss = 1 − P(resultado real)` — ideal 0; estrictamente decreciente con
  la confianza correcta.
- `top1_accuracy`: fracción donde `argmax(probas)` coincide con el resultado real.
- Reporte: `matches, ranking_loss, top1_accuracy, correct_top1`.

### Backtest (`backtester.py`)

Walk-forward cronológico estricto:

- `matches` ordenadas por `(date, id)`; en cada paso usa `matches[:index]`.
- Salta el paso si `len(prior) < min_prior_matches` (default 5).
- Entrena **por paso**: Poisson y ML que no convergen se omiten en ese paso; el
  ensemble usa los miembros sobrevivientes con pesos truncados.
- Acumula muestras auditables `(match_id, probas, hg, ag)` y agrega métricas por
  modelo. **No persiste nada.**

## Parámetros efectivos

| Setting | Default | Usa |
|---------|---------|-----|
| `elo_initial_rating` | 1500.0 | EloCalculator |
| `elo_k_factor` | 20.0 | EloCalculator |
| `elo_home_advantage` | 60.0 | EloCalculator, FeatureVector |
| `features_goals_window` | `None` | GoalsAverageCalculator (todo el historial) |
| `features_home_away_window` | 5 | forma local/visitante |
| `models_poisson_draw_correction` | 1.0 | diagonal Dixon-Coles |
| `models_poisson_regularization` | 0.1 | ridge attack/defense |
| `models_elo_draw_max` | 0.30 | gaussiana de empate |
| `models_elo_draw_sigma` | 300.0 | gaussiana de empate |
| `models_ml_random_state` | 42 | LogisticRegression |
| `models_ensemble_weights` | `"0.4,0.3,0.3"` | EnsembleModel (poisson,elo,ml) |