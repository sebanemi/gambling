# Football Predictor

Análisis estadístico y predicción de resultados de fútbol, **100% CLI** y
**sin data leakage**: cada predicción se basa únicamente en el pasado del partido.

> Una predicción probabilística **no garantiza** un resultado.

## Qué hace

- Importa partidos **(pasados y próximos fixtures)** desde **CSV local** o 6 APIs externas.
- Construye features anti-leakage: Elo, forma, promedios de goles, ventaja local.
- Entrena 4 modelos: **Poisson** (doble Poisson + ridge), **Elo**, **ML**
  (regresión logística multinomial) y un **ensemble** ponderado.
- Predice probabilidades **1X2** (+ goles esperados) con snapshot auditable.
- Evalúa con **ranking loss** y **top-1 accuracy**, incluyendo un **backtest
  walk-forward** estricto.

## Stack

Python 3.12 · PostgreSQL 16 · SQLAlchemy 2 · Alembic · Pydantic 2 · numpy/scipy ·
scikit-learn · Typer · Rich · pytest

## Estado

Proyecto completo — Fase 9 (documentación) terminada. Ver [Roadmap](docs/ROADMAP.md).

## Instalación rápida (Docker)

```bash
cp .env.example .env

# Inicializar la base de datos
docker compose run --rm predictor init-db

# Estado
docker compose run --rm predictor status

# Ayuda
docker compose run --rm predictor --help
```

## Desarrollo local

```bash
docker compose up -d postgres     # solo la BD
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
predictor --help
```

> Fuera del contenedor, el `.env` apunta a `POSTGRES_HOST=postgres` (DNS interno).
> Para correr localmente contra el Postgres de Docker usá
> `POSTGRES_HOST=localhost predictor <comando>`.

## Uso rápido (wrapper `predict.ps1`)

En PowerShell, `.\predict.ps1` levanta Postgres (si hace falta), setea la
conexión y ejecuta cualquier comando del CLI sin configuración previa:

```powershell
.\predict.ps1 predict-match --home "Malaga CF" --away "Villarreal CF" --date 2026-09-17
.\predict.ps1 status
.\predict.ps1 backtest
.\predict.ps1 import-data --path data/sample_matches.csv
```

Sin argumentos muestra la ayuda del CLI. Equivale a los pasos manuales de
abajo (`docker compose up -d postgres` + `POSTGRES_HOST=localhost`).

## Uso en 5 comandos

```bash
# 1. Esquema de la BD
predictor init-db

# 2. Cargar histórico: CSV de ejemplo y/o una temporada real por API
predictor import-data --path data/sample_matches.csv
predictor import-data --provider football-data --competition PL --season 2026

# 3. Ver estado
predictor status

# 4. Predecir un partido (debe existir en BD: jugado o fixture futuro "scheduled")
predictor predict-match --home "Brentford FC" --away "Chelsea FC" --date 2026-09-18

# 5. Medir los modelos sin mirar el futuro
predictor backtest
```

## Notas de uso real

- **Equipos**: lookup exacto por nombre normalizado. Las APIs devuelven
  `Manchester City FC`, `Arsenal FC`, etc. Ver los catálogos disponibles con
  `predictor import-data --provider <fuente> --list`.
- **Última temporada**: al importar la temporada en curso, los fixtures
  **futuros** entran como `scheduled` y se pueden predecir antes de jugarse.
  Re-importar la misma temporada actualiza su marcador (`updated_matches`).
- **Rendimiento**: una predicción con los 4 modelos tarda ~10 s (re-entrena todo
  el histórico). El `backtest` walk-forward es **O(n²)**: usalo sobre datasets
  chicos o con `--min-matches` alto (1,100 partidos lo llevan a >10 min).

## Comandos

| Comando | Opciones | Qué hace |
|---------|----------|----------|
| `init-db` | — | Aplica migraciones (alembic upgrade head) |
| `status` | — | Revisión aplicada y head de la BD |
| `import-data` | `--path`, `--provider`, `--competition`, `--season`, `--list` | Importa CSV o API (partidos o catálogo) |
| `predict-match` | `--home`, `--away`, `--date`, `--model` | Predice 1X2, guarda con snapshot |
| `evaluate` | `--model` | Evalúa predicciones guardadas |
| `backtest` | `--model`, `--min-matches` | Backtest walk-forward sin leakage |

Opciones default relevantes: `predict-match --model ensemble`;
`backtest --model all --min-matches 5`; `import-data --provider csv`.

## Fuentes de datos

Seis providers externos (ver [DATOS.md](docs/DATOS.md)):

- **football-data.org** (v4) — partidos por competición/temporada, `X-Auth-Token`.
- **football-charts.com** — resultados/tablas, token gratuito opcional.
- **football98 (RapidAPI)** — resultados, `X-RapidAPI-Key`.
- **5dollarfootballapi.com** — fixtures con marcador (requiere id de liga).
- **football-standings-api (ESPN)** — catálogo/equipos, sin clave.
- **playerelo.football** — clubes con Team Elo externo, Bearer.

Configurá las claves en `.env` (ver `.env.example`). CSV local: columnas
`date,competition,season,home_team,away_team,home_goals,away_goals`.

## Modelos

Resumen en [MODELOS.md](docs/MODELOS.md): fórmulas de Poisson/Elo, pipeline ML,
ensemble (`0.4 poisson · 0.3 elo · 0.3 ml`) y métricas de evaluación.

## Arquitectura

Capas, flujo de datos y garantías anti-leakage en [ARQUITECTURA.md](docs/ARQUITECTURA.md).
Guía de uso detallada en [USO.md](docs/USO.md).

## Tests

```bash
# Unit (SQLite, sin BD)
.venv\Scripts\python.exe -m pytest tests/unit

# Unit + integración (requiere Postgres en POSTGRES_HOST)
.venv\Scripts\python.exe -m pytest
```

Incluye tests obligatorios de ausencia de data leakage
(`tests/backtesting/test_feature_leakage.py`).