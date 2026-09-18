"""Fixtures compartidos.

- ``sqlite_session``: BD en memoria para tests unitarios (Schema puro,
  sin Alembic).
- ``pg_url`` / ``db_session``: PostgreSQL real; crea la BD de test y la
  deja en el head de Alembic (integration).
"""

import pytest
from alembic import command
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import sessionmaker

from football_predictor.config.settings import get_settings
from football_predictor.database.migrations import configure_alembic
from football_predictor.database.models import Competition, Match, Prediction, Season, Team

TEST_DATABASE = "football_predictor_test"


def _build_url(database: str) -> str:
    s = get_settings()
    return (
        f"postgresql+psycopg://{s.postgres_user}:{s.postgres_password}"
        f"@{s.postgres_host}:{s.postgres_port}/{database}"
    )


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Crea la BD de test, aplica migraciones Alembic y la borra al final."""
    url = _build_url(TEST_DATABASE)
    admin_url = _build_url("postgres")

    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
        conn.execute(text(f'CREATE DATABASE {TEST_DATABASE}'))
    admin.dispose()

    config = configure_alembic()
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    yield url

    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture()
def db_session(pg_url: str):
    """Sesión contra PostgreSQL con tablas limpias por test."""
    engine = create_engine(pg_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        for model in (Prediction, Match, Season, Competition, Team):
            session.execute(delete(model))
        session.commit()
        yield session
    engine.dispose()


@pytest.fixture()
def sqlite_session():
    """Sesión SQLite en memoria (BAse.create_all, sin Alembic)."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    from football_predictor.database.models.base import Base

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session
    engine.dispose()