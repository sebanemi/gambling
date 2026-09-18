"""Comando para recolectar stats de SofaScore vía ScraperFC e importarlas automáticamente."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from football_predictor.cli.commands.import_data import import_data
from football_predictor.collectors.sofascore import collect
from football_predictor.config.settings import get_settings
from football_predictor.utils.logging import get_logger

log = get_logger("cli.collect_sofascore")
console = Console()


def collect_sofascore(
    tournament: Annotated[str, typer.Option(help="Torneo: premier-league, la-liga, bundesliga, serie-a, ligue-1, champions-league, copa-libertadores, liga-argentina, efl-championship, eredivisie, primeira-liga")],
    season: Annotated[str | None, typer.Option(help="Temporada formato YY/YY (ej: 25/26). Default: última disponible")] = None,
    output_dir: Annotated[Path, typer.Option(help="Directorio donde guardar CSV")] = Path("data/sofascore"),
    max_matches: Annotated[int | None, typer.Option(help="Límite de partidos (para testing)")] = None,
    import_after: Annotated[bool, typer.Option("--import/--no-import", help="Importar automáticamente tras recolectar")] = True,
) -> None:
    """Recolecta estadísticas completas de SofaScore vía ScraperFC y las importa a la BD.

    Usa headless browser (bypasa WAF). Stats: shots, SOT, faltas, corners, tarjetas, penales, xG, posesión.
    """
    from football_predictor.collectors.sofascore import TOURNAMENT_MAP

    if tournament not in TOURNAMENT_MAP:
        console.print(f"[red]Torneo desconocido:[/red] {tournament}. Disponibles: {list(TOURNAMENT_MAP.keys())}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Recolectando {tournament} temporada {season or 'última'}...[/cyan]")

    try:
        csv_path = collect(tournament, season, output_dir, max_matches)
    except Exception as e:
        console.print(f"[red]Error en collector:[/red] {e}")
        raise typer.Exit(code=1) from e

    if import_after:
        console.print(f"[cyan]Importando {csv_path}...[/cyan]")
        try:
            from football_predictor.data.importers.csv_importer import CsvImporter
            from football_predictor.data.providers.csv_provider import CsvDataProvider
            from football_predictor.database.session import session_factory

            factory = session_factory()
            provider = CsvDataProvider(csv_path)
            with factory() as session:
                summary = CsvImporter(session, provider).import_all()

            from football_predictor.cli.commands.import_data import _render_summary
            _render_summary(summary)
        except Exception as e:
            console.print(f"[red]Error en importación:[/red] {e}")
            raise typer.Exit(code=1) from e

    console.print("[green]¡Completado![/green]")