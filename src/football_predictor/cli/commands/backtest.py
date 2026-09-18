from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.database.repositories.match_history import MatchHistoryRepository
from football_predictor.database.session import session_factory
from football_predictor.evaluation.backtester import Backtester
from football_predictor.services.model_trainer import (
    ALL_MODELS,
    MODEL_ELO,
    MODEL_ENSEMBLE,
    MODEL_ML,
    MODEL_POISSON,
)
from football_predictor.utils.logging import get_logger

log = get_logger("cli.backtest")
console = Console()


def backtest(
    model: Annotated[
        str,
        typer.Option(help="Modelo a evaluar: poisson, elo, ml, ensemble o all."),
    ] = "all",
    min_matches: Annotated[
        int,
        typer.Option(help="Mínimo de partidos previos por paso antes de predecir."),
    ] = 5,
) -> None:
    """Backtest walk-forward: entrena por partido con solo el pasado."""
    names = tuple(ALL_MODELS) if model == "all" else (model,)
    if any(name not in ALL_MODELS for name in names):
        console.print(
            f"[red]Error:[/red] modelo inválido '{model}'. Válidos: all, {', '.join(ALL_MODELS)}."
        )
        raise typer.Exit(code=1)

    factory = session_factory()
    with factory() as session:
        provider = MatchHistoryRepository(session)
        report = Backtester(provider, min_prior_matches=min_matches).run(names)

    _render(report.evaluated, report.min_prior_matches, report.reports)


def _render(evaluated: int, min_prior_matches: int, reports) -> None:
    table = Table(title=f"Backtest walk-forward (pasos evaluados: {evaluated})")
    table.add_column("Model", style="cyan")
    table.add_column("Matches", justify="right")
    table.add_column("ranking_loss", justify="right")
    table.add_column("top1 (pos1)", justify="right")

    for name in (
        MODEL_POISSON,
        MODEL_ELO,
        MODEL_ML,
        MODEL_ENSEMBLE,
    ):
        if name not in reports:
            continue
        report = reports[name]
        table.add_row(
            name,
            str(report.matches),
            f"{report.ranking_loss:.4f}",
            f"{report.top1_accuracy:.4f}",
        )

    console.print(table)
    console.print(
        f"[dim]Cada paso entrena con {min_prior_matches}+ partidos anteriores y predice "
        f"con features de fecha estricta (sin leakage).[/dim]"
    )