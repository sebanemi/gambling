# Guía de uso de la CLI

Todos los comandos de `predictor` y sus opciones, con ejemplos. El flag global
`--verbose` / `-v` habilita logs DEBUG (van a stderr; la salida legible es de Rich).

```
predictor COMMAND [ARGS]...
```

## Vista general

| Comando | Qué hace |
|---------|----------|
| `init-db` | Crea/actualiza el esquema (alembic upgrade head) |
| `import-data` | Importa partidos o catálogo (CSV o API) |
| `status` | Estado de la BD: revisión aplicada y head |
| `predict-match` | Predice un partido y guarda el resultado con snapshot |
| `evaluate` | Evalúa predicciones guardadas sobre partidos jugados |
| `backtest` | Backtest walk-forward (solo pasado, sin leakage) |

## `init-db`

Aplica todas las migraciones pendientes.

```bash
predictor init-db
```

## `status`

```bash
predictor status
```

Muestra: host y nombre de BD, `Database state` (`OK` / `UNREACHABLE (...)`),
`Migration applied` y `Migration head`. Si la BD no responde sale con código 1.

## `import-data`

```bash
predictor import-data [--path PATH] [--provider NAME]
                      [--competition ID] [--season SEASON] [--list]
```

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--path` | `data/matches.csv` | Ruta al CSV (solo `provider=csv`) |
| `--provider` | `csv` | `csv`, `football-data`, `football-charts`, `football98`, `five-dollar`, `standings`, `playerelo` |
| `--competition` | — | Código/id/slug de competición o liga según la fuente |
| `--season` | — | Temporada (p.ej. `2024`, `2024-25`) |
| `--list` | `false` | Lista las competiciones de la fuente y sale |

Ejemplos:

```bash
# CSV local
predictor import-data --path data/sample_matches.csv

# API con partidos
predictor import-data --provider football-data --competition PL --season 2024

# Catálogo primero, luego import con id real
predictor import-data --provider five-dollar --list
predictor import-data --provider five-dollar --competition 494

# Fuente solo catálogo/equipos
predictor import-data --provider standings --competition ENG.1
```

## `predict-match`

```bash
predictor predict-match [--home EQUIPO] [--away EQUIPO] [--date YYYY-MM-DD] [--model M]
```

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--home` | pregunta | Equipo local. Acepta **parciales y variantes**: `Brentford`, `brentford fc`, `Brighton`… |
| `--away` | pregunta | Equipo visitante (igual de tolerante) |
| `--date` | próx. programado | Fecha del partido (`YYYY-MM-DD`). Si se omite, usa el siguiente fixture programado entre ambos |
| `--model` | `ensemble` | `poisson`, `elo`, `ml`, `ensemble` o `all` |

Si omitís algún dato, la CLI te lo pregunta. La búsqueda de equipos es difusa:
tolera mayúsculas, acentos, nombres parciales y sufijos decorativos
(`Brentford` se resuelve a `Brentford FC`); si hay duplicados elige el equipo con más
partidos. Si no encuentra un equipo, sugiere los nombres almacenados parecidos.

Requisito: el partido debe existir en BD (importar antes con `import-data`) — puede
ser un fixture **pasado o futuro** (`scheduled`). Debe haber histórico para entrenar.

```bash
# Con fecha (fixture ya cargado; los nombres se resuelven solos):
predictor predict-match --home Brentford --away Chelsea --date 2026-09-18

# Sin fecha: usa el próximo Brentford–Chelsea programado
predictor predict-match --home Brentford --away Chelsea

# Render de validación sobre un resultado ya jugado:
predictor predict-match --home Arsenal --away Chelsea --date 2024-09-14 --model all
```

Guarda en `predictions` (upsert por `match_id + model`): probabilidades 1X2, goles
esperados y el `features_snapshot`.

## `evaluate`

```bash
predictor evaluate [--model M]
```

Evalúa predicciones persistidas sobre partidos `finished` con marcador. Reporta
`ranking_loss` (ideal 0), `log_loss` (−ln P(real): castiga sobreconfianza),
`brier` (error cuadrático multiclase) y `top1_accuracy`.

```bash
predictor evaluate                     # ensemble (default)
predictor evaluate --model poisson
```

## `backtest`

```bash
predictor backtest [--model M] [--min-matches N]
```

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--model` | `all` | `poisson`, `elo`, `ml`, `ensemble` o `all` |
| `--min-matches` | `5` | Mínimo de partidos previos por paso antes de predecir |

Walk-forward: cada paso entrena solo con el pasado del partido y evalúa
`ranking_loss`/`log_loss`/`brier`/`top1`. No persiste nada. Además, con las
muestras walk-forward estima **los pesos óptimos del ensemble** (mezcla convexa
que minimiza el log-loss; problema convexo, óptimo global) y los sugiere. Si
querés usarlos: `MODELS_ENSEMBLE_WEIGHTS=X,Y,Z predictor predict-match ...`.

```bash
predictor backtest
predictor backtest --model ensemble --min-matches 10
```

## Flujo típico de punta a punta

```bash
# 1. Levantar BD (Docker) y aplicar esquema
docker compose up -d postgres
predictor init-db

# 2. Cargar histórico (+ fixtures futuros de la temporada actual)
predictor import-data --path data/sample_matches.csv
predictor import-data --provider football-data --competition PL --season 2026

# 3. Ver estado real
predictor status

# 4. Predecir un partido (pasado o futuro; el fixture debe existir en BD)
predictor predict-match --home Brentford --away Chelsea --date 2026-09-18

# 5. Medir el modelo localmente sobre todo el histórico
predictor backtest
```

> Dev local: si usás el Postgres de Docker localmente, el `.env` apunta a
> `POSTGRES_HOST=postgres` (resolución interna). Fuera del contenedor Docker,
> sobrescribilo: `POSTGRES_HOST=localhost predictor backtest`.

> Una predicción probabilística **no garantiza** un resultado.