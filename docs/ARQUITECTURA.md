# Arquitectura

Aplicación **100% CLI** para análisis estadístico y predicción de partidos de fútbol.
Sin servidor web: el flujo es `CLI → dominio → modelos → PostgreSQL`.

## Stack

Python ≥ 3.12 · PostgreSQL 16 · SQLAlchemy 2 · Alembic · Pydantic 2 · numpy/scipy ·
scikit-learn · Typer · Rich · httpx · pytest

## Capas

```
┌──────────────────────────────────────────────────────────────┐
│  CLI (Typer + Rich)          cli/commands/*.py               │
├──────────────────────────────────────────────────────────────┤
│  Servicios                  services/prediction_service.py   │
│                             services/model_trainer.py        │
├──────────────────────────────────────────────────────────────┤
│  Modelos                    models/{poisson,elo_model,       │
│                              ml,ensemble,base}.py            │
├──────────────────────────────────────────────────────────────┤
│  Features                   features/{elo,form,goals,        │
│                              home_advantage,builder}.py      │
├──────────────────────────────────────────────────────────────┤
│  Dominio (puro)             domain/{entities,protocols,      │
│                                          features}.py        │
├──────────────────────────────────────────────────────────────┤
│  Datos                      data/providers/*  ·  importers/  │
│                             data/validators  ·  normalizers/ │
├──────────────────────────────────────────────────────────────┤
│  Persistencia               database/{models,repositories,   │
│                              session,migrations}.py          │
└──────────────────────────────────────────────────────────────┘
```

Regla de dependencia: las capas superiores dependen de las inferiores vía
**protocolos** (`domain/protocols.py`), no de implementaciones concretas.

## Flujo de datos

### 1. Importación

```
CSV o API externa
   → FootballDataProvider (Protocol)   → MatchRecord/TeamInfo/CompetitionInfo
   → MatchImporter / CsvImporter
       · valida (CsvRowValidator) y normaliza nombres (NameNormalizer)
       · crea maestros: competition → season → team
       · inserta matches (deduplica por clave única, commit único)
   → PostgreSQL (tablas teams, competitions, seasons, matches)
```

- Los partidos **sin marcador** (`scheduled`) también se importan, con `status`
  preservado: así es posible predecir fixtures futuros reales. Quedan **fuera del
  histórico** de features/modelos porque `MatchHistoryRepository` solo filtra
  `finished` con goles no nulos.
- Filas inválidas de un CSV **no abortan la importación**: se cuentan en `invalid` y se reportan.

### 2. Entrenamiento (por demanda, no por defecto)

Cada comando que predice (`predict-match`, `backtest`) entrena en memoria con el
histórico disponible (todo el histórico para `predict-match`; solo el pasado del
partido en el backtest).

```
get_all_matches() → FeatureStream.build_all() (una pasada O(n))
   → PoissonModel.fit(matches[, warm_start]) (MLE, L-BFGS-B)
   → EloModel                                (sin entrenamiento, rating es feature)
   → MLModel.fit_matrix(matriz, outcomes)    (Scaler + LogisticRegression)
   → EnsembleModel([poisson, elo, ml], pesos) (media ponderada)
```

### 3. Predicción

```
target (match existente en BD)
   → FeatureBuilder.build_for_match(target)       → FeatureVector (128 features planas)
   → models[name].predict(features)               → PredictionResult (1X2 + goles + snapshot)
   → PredictionRepository.upsert(match_id, model, result)  → tabla predictions
```

### 4. Evaluación

- `evaluate` lee predicciones persistidas sobre partidos `finished` y agrega
  `ranking_loss` y `top1_accuracy`.
- `backtest` repite el paso 2+3 con ventana walk-forward cronológica estricta y
  **no persiste** nada: acumula muestras en memoria.

## Garantías anti-leakage (Fase 3)

Tres capas defensivas para que ninguna predicción vea el futuro:

1. **`MatchHistoryRepository.get_matches_before(cutoff)`**: SQL con `date < cutoff`
   (estricta) y solo `status == 'finished'` con goles no nulos.
2. **`FeatureBuilder._snapshot(cutoff)`**: re-filtra `m.date < cutoff` sobre lo
   devuelto por el provider y cachea por fecha — un provider "leaky" no contamina.
3. **`Backtester.run()`**: itera la lista ordenada por índice; solo usa
   `matches[:index]` (estrictamente anteriores), y salta el paso si hay menos de
   `min_prior_matches`.

Verificado en `tests/backtesting/test_feature_leakage.py`.

## Persistencia

### Sesión

`database/session.py` — `session_factory()` crea el engine desde
`Settings.database_url` (`pool_pre_ping=True`) y `expire_on_commit=False`.
El commit lo hace explícitamente cada servicio o comando.

### Migraciones

`database/migrations.py` — integra Alembic programáticamente:
- `upgrade_to_head()`: aplica todo (usa `predictor init-db`).
- `current_revision()` / `head_revision()`: leen el estado (usa `predictor status`).
- `env.py` resuelve `sqlalchemy.url` desde settings en runtime y usa
  `Base.metadata` como `target_metadata` (soporta `alembic revision --autogenerate`).

### Esquema (head = 0003)

| Tabla | Clave | Único / Checks |
|-------|-------|----------------|
| `teams` | `id` | `UNIQUE (name, country)` |
| `competitions` | `id` | `UNIQUE (name, country)` |
| `seasons` | `id` | `FK competitions.id`, `UNIQUE (competition_id, name)` |
| `matches` | `id` | `FK` a competition/season/home/away; `UNIQUE (competition_id, season_id, home_team_id, away_team_id, date)`; checks: equipos distintos, goles ≥ 0, `status IN ('scheduled','finished','postponed','cancelled')` |
| `predictions` | `id` | `FK matches.id`, `UNIQUE (match_id, model_name)`; `features_snapshot` JSON |

La columna `features_snapshot` de `predictions` guarda el `FeatureVector.flatten()`
exacto usado para predecir (auditoría).

## Configuración

`config/settings.py` (pydantic-settings, lee `.env` case-insensitive) centraliza
todo: PostgreSQL, credenciales de fuentes, parámetros Elo, features y modelos.
Ningún módulo lee `os.environ` directamente; todos usan `get_settings()`.

Ver valores y defaults completos en [DATOS.md](DATOS.md) y [MODELOS.md](MODELOS.md).