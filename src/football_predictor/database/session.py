"""Conexión a PostgreSQL.

Se expone una ``session_factory`` por proceso. Las capas superiores
(features, modelos, CLI) nunca crean engines directamente.
"""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from football_predictor.config.settings import Settings, get_settings


def create_engine_from_settings(settings: Settings | None = None) -> Engine:
    """Crea el engine SQLAlchemy a partir de la configuración.
    
    Optimizado: configuración de pool para mejor rendimiento en entornos
    de producción y desarrollo.
    """
    settings = settings or get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,  # Conexiones permanentes en el pool
        max_overflow=10,  # Conexiones adicionales bajo carga
        pool_recycle=3600,  # Reciclar conexiones cada hora (evitar timeouts)
        echo=False,  # Desactivar SQL logging en producción
    )


def session_factory() -> sessionmaker[Session]:
    """Factory de sesiones para el engine por defecto."""
    engine = create_engine_from_settings()
    return sessionmaker(bind=engine, expire_on_commit=False)


def make_session() -> Generator[Session, None, None]:
    """Dependency: sesión con commit/rollback/close automático."""
    factory = session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()