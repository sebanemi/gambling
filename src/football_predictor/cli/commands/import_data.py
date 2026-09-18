from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.config.settings import get_settings
from football_predictor.data.importers.csv_importer import CsvImporter, ImportSummary
from football_predictor.data.importers.match_importer import MatchImporter
from football_predictor.data.providers.base import (
    ProviderConfigurationError,
    ProviderError,
)
from football_predictor.data.providers.csv_provider import CsvDataProvider, CsvSchemaError
from football_predictor.data.providers.factory import (
    PROVIDER_NAMES,
    build_provider,
    list_competitions,
    provider_has_matches,
)
from football_predictor.database.session import session_factory
from football_predictor.utils.logging import get_logger

log = get_logger("cli.import_data")
console = Console()

_PROVIDER_HELP = "Fuente de datos: " + ", ".join(
    f"{name} ({'partidos' if provider_has_matches(name) else 'tabla/equipos'})"
    for name in PROVIDER_NAMES
)


def import_data(
    path: Annotated[
        Path,
        typer.Option(help="Ruta al CSV de partidos (solo provider=csv)."),
    ] = Path("data/matches.csv"),
    provider: Annotated[
        str,
        typer.Option(help=_PROVIDER_HELP),
    ] = "csv",
    competition: Annotated[
        str | None,
        typer.Option(help="Código/id/slug de competición o liga según la fuente."),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option(help="Temporada (p.ej. 2024, 2024-25, 2025-2026)."),
    ] = None,
    list_catalog: Annotated[
        bool,
        typer.Option("--list", help="Lista competiciones de la fuente y sale."),
    ] = False,
) -> None:
    """Importa partidos (o catalogo) desde CSV o una fuente externa."""
    settings = get_settings()

    if provider != "csv":
        if list_catalog:
            try:
                rows = list_competitions(settings, provider)
            except (ProviderError, ProviderConfigurationError) as exc:
                _fail(exc)
            _render_catalog(provider, rows)
            return
        _import_api(provider, competition, season)
        return

    provider_obj = CsvDataProvider(path)
    log.info("Importing CSV from %s", path)
    try:
        factory = session_factory()
        with factory() as session:
            summary = CsvImporter(session, provider_obj).import_all()
    except (FileNotFoundError, CsvSchemaError, ProviderError) as exc:
        _fail(exc)

    _render_summary(summary)


def _import_api(provider: str, competition: str | None, season: str | None) -> None:
    settings = get_settings()
    try:
        external = build_provider(
            settings, provider, competition=competition, season=season
        )
    except (ProviderError, ProviderConfigurationError) as exc:
        _fail(exc)

    log.info("Importing from %s (competition=%s season=%s)", provider, competition, season)
    try:
        factory = session_factory()
        with factory() as session:
            importer = MatchImporter(session, external)
            if provider_has_matches(provider):
                if provider == "five-dollar" and not competition:
                    _fail(
                        ProviderConfigurationError(
                            "5dollarfootballapi: se requiere --competition con el id de liga. "
                            "Usar --list para ver los ids."
                        )
                    )
                summary = importer.import_all()
                _render_summary(summary)
            else:
                new_competitions, new_teams = importer.import_masters()
                _render_masters(provider, new_competitions, new_teams)
    except (ProviderError, ProviderConfigurationError) as exc:
        _fail(exc)


def _render_summary(summary: ImportSummary) -> None:
    table = Table(title="Import completed")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Rows read", str(summary.rows_read))
    table.add_row("Valid", str(summary.valid))
    table.add_row("Invalid", str(summary.invalid))
    table.add_row("New teams", str(summary.new_teams))
    table.add_row("New competitions", str(summary.new_competitions))
    table.add_row("New seasons", str(summary.new_seasons))
    table.add_row("Inserted matches", str(summary.inserted_matches))
    table.add_row("Updated matches", str(summary.updated_matches))
    table.add_row("Updated statistics", str(summary.updated_statistics))
    table.add_row("Duplicates", str(summary.duplicates))

    console.print(table)


def _render_masters(provider: str, new_competitions: int, new_teams: int) -> None:
    table = Table(title=f"Catalog import ({provider})")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("New competitions", str(new_competitions))
    table.add_row("New teams", str(new_teams))
    console.print(table)
    console.print(
        "[yellow]Nota:[/yellow] esta fuente no tiene partidos historicos; "
        "solo se poblaron competiciones/equipos."
    )


def _render_catalog(provider: str, rows: list[str]) -> None:
    table = Table(title=f"Competitions from {provider}")
    table.add_column("Entry", style="cyan")
    for row in rows:
        table.add_row(row)
    console.print(table)


def _fail(exc: Exception) -> None:
    log.error("Import failed: %s", exc)
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(code=1) from exc