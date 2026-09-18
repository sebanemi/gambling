# Fuentes de datos (providers) y formato CSV

Todas las credenciales se configuran en `.env` (ver `.env.example`). Se leen vía
`config/settings.py`; ningún módulo lee `os.environ` directamente.

## Cliente HTTP compartido

`data/providers/base.py` define `ApiClient` (síncrono sobre `httpx.Client`):

- Auth por header: `Authorization: Bearer <key>`, o clave "cruda" con `auth_scheme=""`
  (p.ej. `X-Auth-Token: <key>`), o headers fijos extra (`X-RapidAPI-Key`).
- Errores mapeados a `ProviderError`: 401/403 → credencial inválida; 404 → no
  encontrado; 429 → rate limit; respuesta no-JSON → error claro.
- `ProviderConfigurationError` cuando falta una clave requerida.
- `transport` inyectable (`httpx.MockTransport`) para tests.

## Providers soportados

| Provider | Partidos | Catálogo | Auth | Clave (.env) |
|----------|----------|----------|------|--------------|
| `football-data` | ✅ | `GET /competitions` | `X-Auth-Token` (cruda) | `FOOTBALL_DATA_API_KEY` |
| `football-charts` | ✅ | `GET /leagues` | `Bearer` (opcional, gratis 5k/día) | `FOOTBALL_CHARTS_API_KEY` |
| `football98` | ✅ | `GET /competitions` | `X-RapidAPI-Key` + `X-RapidAPI-Host` | `FOOTBALL98_API_KEY`, `FOOTBALL98_HOST` |
| `five-dollar` | ✅ | `GET /leagues` (paginado) | `Bearer` | `FIVE_DOLLAR_API_KEY` |
| `sofascore` | ✅ | `GET /unique-tournament/{id}/season/{id}/events` | sin clave | — |
| `standings` | ❌ (solo catálogo/equipos) | `GET /leagues` | sin clave | — |
| `playerelo` | ❌ (solo clubes/ligas con Elo externo) | `GET /v1/leagues` | `Bearer` | `PLAYERELO_API_KEY` |

Detalle endpoint por fuente:

- **football-data.org (v4)** — `GET /competitions/{code}/matches?season=...`.
  `--competition` = código (default `PL`); `--season` opcional. Mapeo de status
  y de `score.fullTime` (partidos sin marcador llegan con goles `None`).
- **football-charts.com** — `GET /leagues/{league_key}/results/?season=...`.
  `--competition` = slug de liga (default `premier`).
- **football98 (RapidAPI)** — `GET /{championship}/results/`.
  `--competition` = slug de campeonato (default `premierleague`).
  Parser tolerante a múltiples variantes de campo (date/home/away/score).
- **5dollarfootballapi.com** — `GET /leagues/{league_id}/fixtures` paginado
  (`status=finished`, `per_page=100`). **`--competition` obligatorio** (id de
  liga; verlo con `--list`). Profundidad histórica según plan (Free 3 meses).
- **SofaScore** — `GET /unique-tournament/{tournament_id}/season/{season_id}/events`
  + `GET /event/{id}/statistics` + `GET /event/{id}/incidents`.
  `--competition` = slug de torneo (`premier-league`, `la-liga`, `bundesliga`,
  `serie-a`, `ligue-1`, `champions-league`, `copa-libertadores`, `liga-argentina`).
  `--season` = año inicio (ej. `2025`). Stats: tiros, SOT, faltas, corners, tarjetas,
  penales, xG, alineaciones con ratings, eventos minuto a minuto. Rate limit ~60 req/min.
- **football-standings-api (ESPN)** — `GET /leagues/{league_id}/standings`.
  `--competition` = id de la fuente (p.ej. `ENG.1`). Sin clave; solo poblado de
  competiciones/equipos.
- **playerelo.football** — `GET /v1/clubs?...` y `GET /v1/leagues`.
  `--competition` = slug/nombre de liga. Aporta clubes con `team_elo` externo
  (poblado de equipos/ligas, no partidos).

## Listar catálogos

Antes de importar conviene listar qué competiciones ofrece cada fuente:

```bash
predictor import-data --provider football-data --list
predictor import-data --provider football-charts --list
predictor import-data --provider five-dollar --list
predictor import-data --provider sofascore --list
predictor import-data --provider standings --list
```

Cada `--list` imprime el identificador a usar con `--competition`.

## Formato CSV local (provider `csv`, default)

Columnas requeridas, en cualquier orden:

```
date,competition,season,home_team,away_team,home_goals,away_goals
```

- **date**: `YYYY-MM-DD`, `DD/MM/YYYY` o `DD-MM-YYYY`.
- **goles**: entero ≥ 0.
- **home_team ≠ away_team** (case-insensitive).
- Columnas obligatorias no vacías; filas inválidas se cuentan en `invalid` (no
  abortan la importación).

Ejemplo (`data/sample_matches.csv`):

```csv
date,competition,season,home_team,away_team,home_goals,away_goals
2024-08-17,Primera Division,2024,River Plate,Boca Juniors,2,1
2024-08-17,Primera Division,2024,Estudiantes,Racing,0,0
```

Importación:

```bash
predictor import-data --path data/sample_matches.csv
```

**Normalización de nombres** (`data/normalizers/names.py`): colapsa whitespace,
remueve diacríticos (`Mönchengladbach → Monchengladbach`) y es idempotente.
Preserva mayúsculas.

## Importación desde API

```bash
# Partidos
predictor import-data --provider football-data --competition PL --season 2024
predictor import-data --provider five-dollar --competition <id-liga>
predictor import-data --provider football-charts --competition premier --season 2024
predictor import-data --provider sofascore --competition la-liga --season 2025

# Solo catálogo/equipos (standings, playerelo)
predictor import-data --provider standings --competition ENG.1 --season 2024
predictor import-data --provider playerelo --competition premier
```

### Semántica del resumen de importación

| Campo | Significado |
|-------|-------------|
| `rows_read` | Filas/registros leídos (CSV: incluye inválidas) |
| `valid` | Registros válidos (incluye duplicados válidos) |
| `invalid` | Registros rechazados por validación |
| `new_teams` / `new_competitions` / `new_seasons` | Maestros creados nuevos |
| `inserted_matches` | Partidos insertados (excluye duplicados) |
| `updated_matches` | `scheduled` que se re-importaron con marcador (se actualizaron) |
| `duplicates` | Partidos ya existentes (BD o repetidos en la misma fuente) |

Solo se **insertan** partidos con marcador o `scheduled`. Los fixtures futuros
(sin marcador) se importan con `status='scheduled'` para poder predecirlos antes
de jugarse; quedan excluidos del histórico de modelos (que solo usa `finished`
con goles). Al **re-importar** una temporada, los `scheduled` que ya tienen
resultado se **actualizan** (contados en `updated_matches`, no como duplicados).

La deduplicación usa la clave única
`(competition_id, season_id, home_team_id, away_team_id, date)`; re-importar el
mismo archivo de resultados devuelve `inserted_matches=0`.

### Estadísticas por partido

Si el registro trae stats (CSV o API), se persisten en `match_statistics`:
tarjetas amarillas/rojas, corners, tiros, tiros al arco, faltas, saques de banda,
penales, xG y posesión (10 métricas × local/visitante). Las stats se
**actualizan siempre** al re-importar un partido (aunque ya tengan resultado), y
una fila de stats se crea con cualquiera de las métricas presentes. Estas stats
alimentan los promedios de `FeatureStream`/`FeatureBuilder` (features ML) y
el `StatisticsModel` de `predict-match`.