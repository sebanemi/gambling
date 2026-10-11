# Guía de uso de la CLI

Todos los comandos de `predictor` y sus opciones, con ejemplos. El flag global
`--verbose` / `-v` habilita logs DEBUG (van a stderr; la salida legible es de Rich).

```
predictor COMMAND [ARGS]...
```

Ayuda rápida (las 3 muestran los comandos disponibles):

```bash
predictor --help              # lista de comandos (Typer, siempre disponible)
predictor commands            # lista corta en español (igual que --help)
predictor help                # guía completa en español
predictor help predict-match  # guía de un comando
predictor predict-match --help  # opciones de un comando
```

## Vista general

| Comando | Qué hace |
|---------|----------|
| `init-db` | Crea/actualiza el esquema (alembic upgrade head) |
| `import-data` | Importa partidos o catálogo (CSV o API) |
| `import-odds` | Importa cuotas históricas desde CSV |
| `status` | Estado de la BD: revisión aplicada y head |
| `predict-match` | Predice un partido y guarda el resultado con snapshot |
| `evaluate` | Evalúa predicciones guardadas sobre partidos jugados |
| `backtest` | Backtest walk-forward (solo pasado, sin leakage) |
| `simulate-betting` | Simula apuestas con cuotas reales (walk-forward) |
| `value-bets` | Lista apuestas de valor en fixtures programados |
| `collect-sofascore` | Recolecta stats de SofaScore (requiere IP residencial) |
| `help` | Guía completa o de un comando (`help predict-match`) |
| `commands` | Lista corta de comandos (alias legible de `--help`) |

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
| `--model` | `ensemble` | `poisson`, `elo`, `ml`, `gbm`, `ensemble` o `all` |
| `--all-leagues` | `false` | Entrena con todas las ligas (default: solo la liga del partido) |

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
esperados y el `features_snapshot`. Además muestra el **nivel de confianza**:
`COIN-FLIP` (margen <0.05 o top <0.40: no hay edge, mejor no apostar),
`LEAN` (señal débil) o `CONFIDENT`. Un acierto con probs 0.36/0.33/0.31 es
suerte, no skill: mirá el margen y la entropía antes de apostar.

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
| `--model` | `all` | `poisson`, `elo`, `ml`, `gbm`, `ensemble` o `all` |
| `--min-matches` | `5` | Mínimo de partidos previos por paso antes de predecir |
| `--competition` | todas | Solo esta liga (p.ej. `--competition 'Liga Profesional'`) |

Walk-forward: cada paso entrena solo con el pasado del partido y evalúa
`ranking_loss`/`log_loss`/`brier`/`top1`. No persiste nada. Además, con las
muestras walk-forward estima **los pesos óptimos del ensemble** (mezcla convexa
que minimiza el log-loss; problema convexo, óptimo global) y los sugiere. Si
querés usarlos: `MODELS_ENSEMBLE_WEIGHTS=W,X,Y,Z predictor predict-match ...`
(4 pesos: poisson,elo,ml,gbm).

```bash
predictor backtest
predictor backtest --model ensemble --min-matches 10
```

## `import-odds`

```bash
predictor import-odds --path data/sample_odds.csv
```

Importa cuotas decimales (`date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds`;
upsert por partido + bookmaker). Las filas sin partido en BD se cuentan como
`unknown_matches`.

## `simulate-betting`

```bash
predictor simulate-betting [--model M] [--strategy S] [--initial-bankroll X]
                           [--value-threshold T] [--bookmaker B] [--no-save]
```

Walk-forward con **cuotas reales**: solo apuesta si hay cuota importada y
`EV > umbral`. Persiste el resumen en `betting_simulations` y cada apuesta en
`betting_history` (salvo `--no-save`).

```bash
predictor import-odds --path data/sample_odds.csv
predictor simulate-betting --model ensemble --strategy kelly
```

## `value-bets`

```bash
predictor value-bets [--model M] [--threshold T] [--bookmaker B] [--limit N]
```

Entrena con todo el histórico y lista los outcomes de fixtures `scheduled`
con `EV > umbral`, ordenados por EV. Detalle en [BETTING.md](BETTING.md).

## `collect-sofascore`

```bash
predictor collect-sofascore --help
```

Recolecta stats completas de SofaScore y las importa. Requiere IP residencial.

## `help`

```bash
predictor help [COMANDO]
```

Muestra esta guía completa o la sección de un comando (`predictor help backtest`).

## `commands`

```bash
predictor commands
```

Lista corta de todos los comandos. Equivale a `predictor --help` pero en español
y sin necesitar `docs/USO.md`.

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