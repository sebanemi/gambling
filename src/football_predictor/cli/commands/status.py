from rich.console import Console
from rich.table import Table
from sqlalchemy.exc import OperationalError
import typer

from football_predictor.config.settings import get_settings
from football_predictor.database.migrations import current_revision, head_revision

console = Console()


def status() -> None:
    """Muestra el estado de la configuración y de la base de datos."""
    settings = get_settings()

    try:
        current = current_revision()
        head = head_revision()
        db_state = "OK"
    except OperationalError as exc:
        current = ""
        head = ""
        db_state = f"UNREACHABLE ({exc})"

    table = Table(title="Predictor status")
    table.add_column("Item", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Database host", settings.postgres_host)
    table.add_row("Database name", settings.postgres_db)
    table.add_row("Database state", db_state)
    table.add_row("Migration applied", current or "(none)")
    table.add_row("Migration head", head or "(none)")

    console.print(table)

    if "UNREACHABLE" in db_state:
        raise typer.Exit(code=1)