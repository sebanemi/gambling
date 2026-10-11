# Roadmap — Football Predictor

Implementación por fases, cada una con salida verificable (tests). Estado actual: **Fase 9 terminada (documentación)**.

| Fase | Nombre | Entregable | Estado |
|------|--------|------------|--------|
| 1 | Infraestructura | Repo, `pyproject.toml`, Docker, settings, CI base | ✅ |
| 2 | Esquema y datos | Modelos ORM, migraciones Alembic, importación CSV | ✅ |
| 3 | Features anti-leakage | `HistoryProvider` (fecha estricta), Elo, forma, goles | ✅ |
| 4 | Modelo Elo | Ratings cero-sum + probabilidades 1X2 | ✅ |
| 5 | Modelo Poisson | Doble Poisson con ridge y corrección de empate | ✅ |
| 6 | Modelo ML + ensemble | Regresión logística multinomial + media ponderada | ✅ |
| 7 | Providers externos | Clients HTTP para 6 fuentes + fábrica | ✅ |
| 8 | Evaluación y backtest | `evaluate` y `backtest` walk-forward | ✅ |
| 9 | Documentación | `README.md` completo + `docs/` | ✅ |
| 10 | Rendimiento | `FeatureStream` O(n), backtest con matriz precomputada + warm-start Poisson | ✅ |
| 11 | Stats ricas | Persistencia xG/posesión en BD + predicción y features para 10 métricas | ✅ |
| 12 | Betting simulation | Cuotas reales (`match_odds` + `import-odds`), `simulate-betting` walk-forward O(n) con persistencia, `value-bets`, staking (fixed/kelly/proportional/confidence) | ✅ |
| 13 | Same-league + comandos | Scope por competición (default en `predict-match`, `--all-leagues`, `backtest --competition`), comando `commands`, nivel de confianza COIN-FLIP/LEAN/CONFIDENT en predicciones | ✅ |
| 14 | Datos argentinos | Backfill ESPN sin clave (`scripts/backfill_espn_arg.py`, ~1.300 partidos 2024-2026), fix de duplicados UTC-vs-local | ✅ |
| 15 | Calibración ML/GBM | Grid walk-forward: ML `C=0.001` + sin ponderar (LL 1.36→1.07), GBM 20×3 + sin ponderar (LL 1.31→1.08); pesos ensemble medidos por liga | ✅ |

## Notas de diseño transversal

- **Determinismo**: misma configuración + mismo histórico ⇒ mismos modelos (semilla ML fija, `MODELS_ML_RANDOM_STATE=42`).
- **Anti-leakage en tres capas**: `MatchHistoryRepository` filtra `date < cutoff`; `FeatureBuilder` re-filtra y cachea por fecha; `Backtester` itera por índice cronológico estricto.
- **Auditabilidad**: cada predicción guarda un `features_snapshot` (JSON con las 128 features) junto a las probabilidades.
- **Sin servicios web**: todo es CLI; la persistencia es PostgreSQL vía SQLAlchemy/Alembic.