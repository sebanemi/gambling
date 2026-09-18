from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.database.session import session_factory
from football_predictor.services.prediction_service import PredictionService
from football_predictor.utils.logging import get_logger

log = get_logger("cli.evaluate")
console = Console()


def evaluate(
    model: Annotated[
        str,
        typer.Option(help="Modelo a evaluar: poisson, elo, ml o ensemble."),
    ] = "ensemble",
) -> None:
    """Evalúa predicciones guardadas sobre partidos ya jugados."""
    factory = session_factory()
    with factory() as session:
        report = PredictionService(session).evaluate(model)

    table = Table(title=f"Evaluation — {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Matches evaluated", str(report.matches))
    table.add_row("ranking_loss", f"{report.ranking_loss:.4f}")
    table.add_row("log_loss", f"{report.log_loss:.4f}")
    table.add_row("brier (multi)", f"{report.brier:.4f}")
    table.add_row("top1_accuracy (pos1)", f"{report.top1_accuracy:.4f}")
    table.add_row("Correct top-1", str(report.correct_top1))

    console.print(table)
    if report.matches == 0:
        console.print("[yellow]Sin partidos jugados con predicción guardada. Ejecuta 'predict-match' sobre partidos pasados.[/yellow]")