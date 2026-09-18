from datetime import date as _Date
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.database.repositories.match import MatchRepository
from football_predictor.database.repositories.team import TeamRepository
from football_predictor.database.session import session_factory
from football_predictor.models.base import PredictionResult, StatisticsPredictionResult
from football_predictor.services.prediction_service import PredictionService
from football_predictor.utils.logging import get_logger

log = get_logger("cli.predict_match")
console = Console()


def predict_match(
    home: Annotated[str, typer.Option(help="Equipo local (nombre normalizado).")],
    away: Annotated[str, typer.Option(help="Equipo visitante (nombre normalizado).")],
    date: Annotated[str, typer.Option(help="Fecha del partido (YYYY-MM-DD).")],
    model: Annotated[
        str,
        typer.Option(help="Modelo: poisson, elo, ml, ensemble o all."),
    ] = "ensemble",
) -> None:
    """Predice un partido y guarda la predicción con su snapshot."""
    try:
        match_date = _Date.fromisoformat(date)
    except ValueError:
        console.print(f"[red]Error:[/red] fecha inválida '{date}' (usa YYYY-MM-DD).")
        raise typer.Exit(code=1) from None

    factory = session_factory()
    with factory() as session:
        teams = TeamRepository(session)
        home_id = teams.find_id(home)
        away_id = teams.find_id(away)
        if home_id is None or away_id is None:
            console.print("[red]Error:[/red] no se encuentra alguno de los equipos en la BD.")
            raise typer.Exit(code=1)
        if home_id == away_id:
            console.print("[red]Error:[/red] el equipo local y visitante deben ser distintos.")
            raise typer.Exit(code=1)

        match = MatchRepository(session).find_by_teams_and_date(match_date, home_id, away_id)
        if match is None:
            console.print(f"[red]Error:[/red] no existe partido {home} vs {away} el {match_date}.")
            raise typer.Exit(code=1)

        results, stats = PredictionService(session).predict_match(match_id=match.id, model_name=model)

    _render(match.id, match_date, home, away, results)
    _render_stats(stats)
    console.print(f"[dim]Predicción guardada en BD (match_id={match.id}). Modelos usados: {', '.join(results)}.[/dim]")


def _render(match_id: int, match_date: _Date, home: str, away: str, results: dict[str, PredictionResult]) -> None:
    table = Table(title=f"Predicted {home} vs {away} ({match_date}), match #{match_id}")
    table.add_column("Model", style="cyan")
    table.add_column("P(home)", justify="right")
    table.add_column("P(draw)", justify="right")
    table.add_column("P(away)", justify="right")
    table.add_column("Goals exp.", justify="right")

    for name, result in results.items():
        goals = (
            f"{result.home_goals:.2f} - {result.away_goals:.2f}"
            if result.home_goals is not None
            else "—"
        )
        table.add_row(
            name,
            f"{result.home_win:.3f}",
            f"{result.draw:.3f}",
            f"{result.away_win:.3f}",
            goals,
        )

    console.print(table)


def _render_stats(stats: StatisticsPredictionResult) -> None:
    table = Table(title="Estadísticas esperadas (Poisson por recuentos)")
    table.add_column("Métrica", style="cyan")
    table.add_column("Local", justify="right", style="green")
    table.add_column("Visitante", justify="right", style="red")

    table.add_row("Tarjetas amarillas", f"{stats.home_yellow_cards:.2f}", f"{stats.away_yellow_cards:.2f}")
    table.add_row("Tarjetas rojas", f"{stats.home_red_cards:.2f}", f"{stats.away_red_cards:.2f}")
    table.add_row("Córners", f"{stats.home_corners:.2f}", f"{stats.away_corners:.2f}")

    console.print(table)