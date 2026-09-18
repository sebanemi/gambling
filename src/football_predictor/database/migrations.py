"""Helpers para ejecutar Alembic desde la aplicación (``predictor init-db``)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine

from football_predictor.config.settings import get_settings

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
_ALEMBIC_INI = _MIGRATIONS_DIR / "alembic.ini"


def configure_alembic() -> Config:
    """Construye la Config de Alembic apuntando a ``migrations/``."""
    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    return cfg


def upgrade_to_head() -> None:
    """Aplica todas las migraciones pendientes."""
    command.upgrade(configure_alembic(), "head")


def head_revision() -> str:
    """Revisión head definida en el árbol de migraciones."""
    scripts = ScriptDirectory.from_config(configure_alembic())
    return scripts.get_current_head()


def current_revision() -> str | None:
    """Revisión aplicada actualmente en la BD (o None)."""
    engine: Engine = create_engine(get_settings().database_url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()
    finally:
        engine.dispose()