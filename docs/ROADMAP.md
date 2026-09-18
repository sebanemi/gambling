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

## Notas de diseño transversal

- **Determinismo**: misma configuración + mismo histórico ⇒ mismos modelos (semilla ML fija, `MODELS_ML_RANDOM_STATE=42`).
- **Anti-leakage en tres capas**: `MatchHistoryRepository` filtra `date < cutoff`; `FeatureBuilder` re-filtra y cachea por fecha; `Backtester` itera por índice cronológico estricto.
- **Auditabilidad**: cada predicción guarda un `features_snapshot` (JSON con las 88 features) junto a las probabilidades.
- **Sin servicios web**: todo es CLI; la persistencia es PostgreSQL vía SQLAlchemy/Alembic.