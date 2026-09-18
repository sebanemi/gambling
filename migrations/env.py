"""Alembic environment.

Resuelve la URL de conexión desde la configuración de la aplicación
(NUNCA hardcodeada) y usa la metadata declarativa como target para
autogenerate.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Asegura que `src/` esté en sys.path para importar la app (fuera de Docker).
_DIR = Path(__file__).resolve().parents[1]  # raíz del proyecto
if str(_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_DIR / "src"))

from football_predictor.config.settings import get_settings  # noqa: E402
from football_predictor.database.models import Base  # noqa: E402

config = context.config
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Migraciones offline: genera SQL sin conexión a la BD."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Migraciones online: se conecta a la BD."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()