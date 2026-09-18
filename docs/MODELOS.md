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

### `FeatureVector.flatten()` → 128 columnas

8 base (`home_elo`, `away_elo`, `elo_difference`, `home_goals_for_avg`,
`home_goals_against_avg`, `away_goals_for_avg`, `away_goals_against_avg`,
`home_advantage`) + 10 métricas de stats × 4 campos (`home_{m}_for_avg`,
`home_{m}_against_avg`, `away_{m}_for_avg`, `away_{m}_against_avg`) + 2 lados ×
(3 ventanas overall + 2 contextos) × 8 campos.

Las 10 métricas de stats (`STATISTIC_METRICS` en `domain/features.py`):
`yellow_cards`, `red_cards`, `corners`, `shots`, `shots_on_target`, `fouls`,
`throw_ins`, `penalties`, `xg`, `possession`.

Este dict plano es el `features_snapshot` persistido en `predictions`.

## Modelos (`models/`)

Contrato común: del `FeatureVector` devuelven un `PredictionResult` con
probabilidades **1X2 normalizadas a suma 1**, goles esperados (si aplica) y snapshot.

### Poisson (`poisson.py`)

Doble Poisson con **ridge** y **corrección de Dixon-Coles (1997)**:

- `λ_home = exp(γ_home + attack[local] + defense[visita] + home_adv)`
- `λ_away = exp(γ_away + attack[visita] + defense[local])`
- Fit por máxima verosimilitud (`scipy.optimize.minimize`, `L-BFGS-B`) sobre
  NLL + `λ·(Σ attack² + Σ defense²)`.
- **Dixon-Coles**: el factor `τ(x,y,ρ)` corrige la independencia en las celdas
  de pocos goles (`τ(0,0)=1`, `τ(0,1)=τ(1,0)=1−ρ`, `τ(1,1)=1+ρ`). El parámetro
  `ρ` (típicamente negativo: subestima 1-0/0-1 y sobreestima 1-1) se **estima
  por MLE junto con el resto** (con cota en [−0.5, 0.5]), o se congela pasando
  un valor. Corrige más la distribución del scoreline que el 1X2.
- Devuelve goles esperados `(λ_home, λ_away)`.
- Parámetros (settings): `MODELS_POISSON_REGULARIZATION=0.1`,
  `MODELS_POISSON_RHO=<vacío>` (`None` ⇒ estima; un número lo fija).

### Elo (`elo_model.py`)

Convierte el rating **pre-partido** (ya viene en las features) en probabilidades:

- `p_home = E(home_rating + home_adv, away_rating)`
- `P_draw = draw_max · exp(−diff² / (2σ²))` (gaussiana centrada en 0)
- `P_home = p_home·(1 − P_draw)`, `P_away = (1 − p_home)·(1 − P_draw)`
- Parámetros: `MODELS_ELO_DRAW_MAX=0.30`, `MODELS_ELO_DRAW_SIGMA=300`.
- No estima goles.

### ML (`ml.py`)

- **Regresión logística multinomial** (`StandardScaler → LogisticRegression`,
  `max_iter=2000`, `random_state=42`) sobre las 128 columnas.
- Descarta columnas de varianza nula (evita NaN en el scaler).
- Se entrena con features **anti-leakage** por partido (puerta temporal) y
  etiquetas `Outcome(1X2)` de los goles reales.
- No estima goles.

### Ensemble (`ensemble.py`)

- Media ponderada de probabilidades 1X2 de `[poisson, elo, ml]`.
- Pesos por defecto (settings `MODELS_ENSEMBLE_WEIGHTS`, se normalizan) estimados
  por **log-loss walk-forward** sobre el histórico: `0.486, 0.465, 0.049`.
  Recalculá los tuyos con `predictor backtest` (tabla "Pesos óptimos").
- Goles esperados = promedio solo de los modelos que los producen (poisson).
- `ensemble_optimizer.py`: los pesos óptimos son la mezcla convexa que minimiza
  el log-loss; el problema es **estrictamente convexo** (`logsumexp`), SLSQP
  encuentra el óptimo global. Es un stacking de nivel 1 directo, y como las
  muestras son walk-forward, los pesos son out-of-sample (sin leakage).

### Claves y entrenador

- Claves: `poisson`, `elo`, `ml`, `ensemble` (+ `all` en `predict-match`/`backtest`).
- `ModelTrainer.train_all()` entrena los 4 y devuelve el dict.
- `parse_ensemble_weights()` exige exactamente 3 floats con suma > 0.

## Evaluación (`evaluation/`)

### Métricas (`metrics.py`)

Filosofía: medir ranking y **calibración** de las probabilidades 1X2.

- `ranking_loss = 1 − P(resultado real)` — ideal 0; ranking.
- `log_loss = −ln P(resultado real)` — propia y estrictamente calibrada.
- `brier` — error cuadrático multiclase (0 a 2; ideal 0), mide calibración+resolución.
- `top1_accuracy`: fracción donde `argmax(probas)` coincide con el resultado real.
- Reporte: `matches, ranking_loss, log_loss, brier, top1_accuracy, correct_top1`.

### Backtest (`backtester.py`)

Walk-forward cronológico estricto:

- `matches` ordenadas por `(date, id)`; en cada paso usa `matches[:index]`.
- Salta el paso si `len(prior) < min_prior_matches` (default 5).
- **Optimizado**: las features se construyen una sola vez con `FeatureStream`
  (O(n), con la puerta por fecha anti-leakage del `FeatureBuilder`); la matriz del
  ML se precomputa y se reutiliza por slices; el Poisson se recalienta entre pasos
  con `warm_start` de su último punto óptimo (mismo objetivo ⇒ mismo óptimo).
- Entrena **por paso**: Poisson y ML que no convergen se omiten en ese paso; el
  ensemble usa los miembros sobrevivientes con pesos truncados.
- Acumula muestras auditables `(match_id, probas, hg, ag)` y agrega métricas por
  modelo. **No persiste nada.**
- Si se pide el ensemble, con las muestras de los miembros estima **los pesos
  óptimos del ensemble** (ver `ensemble_optimizer.py`) y los reporta.

## Parámetros efectivos

| Setting | Default | Usa |
|---------|---------|-----|
| `elo_initial_rating` | 1500.0 | EloCalculator |
| `elo_k_factor` | 20.0 | EloCalculator |
| `elo_home_advantage` | 60.0 | EloCalculator, FeatureVector |
| `features_goals_window` | `None` | GoalsAverageCalculator (todo el historial) |
| `features_home_away_window` | 5 | forma local/visitante |
| `features_stats_window` | `None` | averages de stats (todo el historial) |
| `models_poisson_rho` | `None` | Dixon-Coles (None ⇒ estima ρ por MLE) |
| `models_poisson_regularization` | 0.1 | ridge attack/defense |
| `models_elo_draw_max` | 0.30 | gaussiana de empate |
| `models_elo_draw_sigma` | 300.0 | gaussiana de empate |
| `models_ml_random_state` | 42 | LogisticRegression |
| `models_ensemble_weights` | `"0.4,0.3,0.3"` | EnsembleModel (poisson,elo,ml) |