import typer

from football_predictor.database.migrations import upgrade_to_head
from football_predictor.utils.logging import get_logger

log = get_logger("cli.init_db")


def init_database() -> None:
    """Inicializa la base de datos: aplica todas las migraciones pendientes."""
    log.info("Applying database migrations...")
    upgrade_to_head()
    typer.echo("Database schema is up to date.")