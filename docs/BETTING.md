# Guía de apuestas (betting)

Simulación de estrategias de apuestas con cuotas reales, **sin data leakage**:
cada apuesta usa solo el pasado del partido y exige cuotas históricas
importadas (sin cuotas no hay apuesta — el simulador anterior usaba cuotas
ficticias derivadas del propio modelo y se eliminó).

> Una predicción probabilística **no garantiza** un resultado. Apostar
> conlleva riesgo de pérdida. Esto es una herramienta de análisis, no
> asesoramiento financiero.

## Formato de cuotas (CSV)

```csv
date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds
2024-08-17,River Plate,Boca Juniors,bet365,2.10,3.40,3.20
```

- `date`: `YYYY-MM-DD` del partido tal como está en BD.
- Equipos: se resuelven con tolerancia (`TeamRepository.resolve`).
- Cuotas decimales `> 1.0`. Upsert por `(match_id, bookmaker)`.
- Las filas sin partido en BD se cuentan como `unknown_matches` (no fallan).

Ver ejemplo: `data/sample_odds.csv`.

## Flujo típico

```bash
predictor init-db
predictor import-data --path data/sample_matches.csv
predictor import-odds --path data/sample_odds.csv

# Simulación walk-forward con cuotas reales (guarda en BD)
predictor simulate-betting --model ensemble --strategy kelly

# Apuestas de valor en fixtures programados
predictor value-bets --model ensemble --threshold 0.05
```

## Comandos

### `import-odds`

```bash
predictor import-odds --path data/sample_odds.csv
```

### `simulate-betting`

```bash
predictor simulate-betting [--model M] [--strategy S] [--initial-bankroll X]
                           [--value-threshold T] [--bookmaker B] [--no-save]
```

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--model` | `ensemble` | `poisson`, `elo`, `ml`, `gbm`, `ensemble` |
| `--strategy` | `BETTING_STRATEGY` (`kelly`) | `fixed`, `kelly`, `proportional`, `confidence` |
| `--initial-bankroll` | `BETTING_INITIAL_BANKROLL` (`1000`) | Bankroll inicial |
| `--value-threshold` | `BETTING_VALUE_THRESHOLD` (`0.05`) | EV mínimo para apostar |
| `--bookmaker` | mejor cuota | Bookmaker concreto o mejor entre todos |
| `--no-save` | `false` | No persistir en `betting_simulations` / `betting_history` |

Solo apuesta cuando hay cuota real y `EV > umbral`. El stake se capa al
`BETTING_MAX_STAKE_PERCENTAGE` del bankroll (además del bankroll total).
Reporta: nº apuestas, win rate, ROI, drawdown máximo, profit/loss y conteo
de omitidos (`sin cuota` / `sin valor`).

### `value-bets`

```bash
predictor value-bets [--model M] [--threshold T] [--bookmaker B] [--limit N]
```

Entrena con todo el histórico, predice cada fixture `scheduled` y muestra
los outcomes con `EV > umbral`, ordenados por EV descendente.

## Estrategias (staking)

| Estrategia | Lógica |
|------------|--------|
| `fixed` | `BETTING_FIXED_STAKE` siempre |
| `kelly` | Kelly fraccional (`BETTING_KELLY_FRACTION`, default 0.25); 0 si no hay edge |
| `proportional` | `BETTING_PROPORTIONAL_PERCENTAGE` del bankroll |
| `confidence` | Proporcional al EV (cap `BETTING_MAX_STAKE_PERCENTAGE`) |

EV por unidad apostada: `EV = P(modelo) × cuota − 1`.

## Configuración (`.env`)

| Variable | Default |
|----------|---------|
| `BETTING_STRATEGY` | `kelly` |
| `BETTING_INITIAL_BANKROLL` | `1000.0` |
| `BETTING_KELLY_FRACTION` | `0.25` |
| `BETTING_FIXED_STAKE` | `10.0` |
| `BETTING_PROPORTIONAL_PERCENTAGE` | `0.02` |
| `BETTING_MAX_STAKE_PERCENTAGE` | `0.05` |
| `BETTING_VALUE_THRESHOLD` | `0.05` |
| `BETTING_MIN_ODDS` / `BETTING_MAX_ODDS` | `1.1` / `10.0` |
| `BETTING_BACKTEST_MIN_MATCHES` | `10` |

## Tablas

- `match_odds`: cuotas por `(match_id, bookmaker)`.
- `betting_simulations`: resumen de cada `simulate-betting`.
- `betting_history`: cada apuesta individual (FK a simulación y partido).
