from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.data.importers.odds_importer import OddsImporter
from football_predictor.database.session import session_factory
from football_predictor.utils.logging import get_logger

log = get_logger("cli.import_odds")
console = Console()


def import_odds(
    path: Annotated[
        Path,
        typer.Option(help="Ruta al CSV de cuotas (date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds)."),
    ] = Path("data/odds.csv"),
) -> None:
    """Importa cuotas históricas (upsert por partido + bookmaker)."""
    factory = session_factory()
    try:
        with factory() as session:
            summary = OddsImporter(session, path).import_all()
    except (FileNotFoundError, ValueError) as exc:
        log.error("Import odds failed: %s", exc)
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Odds import completed")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("Rows read", str(summary.rows_read))
    table.add_row("Upserted", str(summary.upserted))
    table.add_row("Invalid", str(summary.invalid))
    table.add_row("Unknown matches", str(summary.unknown_matches))
    console.print(table)
    if summary.unknown_matches:
        console.print(
            "[yellow]Nota:[/yellow] las filas 'unknown' no matchearon ningún partido en BD "
            "(revisá nombres/fecha o importá primero los partidos)."
        )
