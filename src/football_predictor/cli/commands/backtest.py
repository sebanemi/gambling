from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.database.repositories.match_history import MatchHistoryRepository
from football_predictor.database.session import session_factory
from football_predictor.evaluation.backtester import Backtester
from football_predictor.evaluation.ensemble_optimizer import mix_log_loss
from football_predictor.services.model_trainer import (
    ALL_MODELS,
    MODEL_ELO,
    MODEL_ENSEMBLE,
    MODEL_GBM,
    MODEL_ML,
    MODEL_POISSON,
    parse_ensemble_weights,
)
from football_predictor.utils.logging import get_logger

log = get_logger("cli.backtest")
console = Console()


def backtest(
    model: Annotated[
        str,
        typer.Option(help="Modelo a evaluar: poisson, elo, ml, gbm, ensemble o all."),
    ] = "all",
    min_matches: Annotated[
        int,
        typer.Option(help="Mínimo de partidos previos por paso antes de predecir."),
    ] = 5,
    competition: Annotated[
        str | None,
        typer.Option(
            help="Solo esta liga (p.ej. 'Liga Profesional'). Default: todas."
        ),
    ] = None,
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
        from football_predictor.database.repositories.competition import CompetitionRepository

        competition_id: int | None = None
        if competition is not None:
            competition_id = CompetitionRepository(session).resolve(competition)
            if competition_id is None:
                available = CompetitionRepository(session).names()
                console.print(
                    f"[red]Error:[/red] competición desconocida '{competition}'. "
                    f"Disponibles: {', '.join(available)}."
                )
                raise typer.Exit(code=1)
        provider = MatchHistoryRepository(session)
        report = Backtester(provider, min_prior_matches=min_matches, competition_id=competition_id).run(names)

    _render(report.evaluated, report.min_prior_matches, report.reports)
    _render_optimal_weights(report)


def _render_optimal_weights(report) -> None:
    weights = report.recommended_weights
    if not weights:
        return
    from football_predictor.config.settings import get_settings

    default_weights = parse_ensemble_weights(get_settings().models_ensemble_weights)
    ensemble_members = (MODEL_POISSON, MODEL_ELO, MODEL_ML, MODEL_GBM)
    default_weights = {
        name: default_weights[i] for i, name in enumerate(ensemble_members) if i < len(default_weights)
    }
    default_log_loss = mix_log_loss(
        {
            name: report.samples(name)
            for name in ensemble_members
            if report.samples(name)
        },
        {name: default_weights.get(name, 0.0) for name in ensemble_members if name in default_weights},
    )

    table = Table(title="Pesos óptimos del ensemble (por log-loss walk-forward)")
    table.add_column("Miembro", style="cyan")
    table.add_column("Peso default", justify="right")
    table.add_column("Peso óptimo", justify="right", style="green")
    for name in ensemble_members:
        table.add_row(
            name,
            f"{default_weights.get(name, 0.0):.3f}",
            f"{weights.get(name, 0.0):.3f}",
        )
    table.add_row("log_loss", f"{default_log_loss:.4f}" if default_log_loss else "—", f"{report.recommended_log_loss:.4f}")

    console.print(table)
    console.print(
        "[dim]Para usar estos pesos: predictor predict-match ... --model ensemble con "
        f"MODELS_ENSEMBLE_WEIGHTS={','.join(f'{weights.get(n,0):.3f}' for n in ensemble_members)}[/dim]"
    )


def _render(evaluated: int, min_prior_matches: int, reports) -> None:
    table = Table(title=f"Backtest walk-forward (pasos evaluados: {evaluated})")
    table.add_column("Model", style="cyan")
    table.add_column("Matches", justify="right")
    table.add_column("ranking_loss", justify="right")
    table.add_column("log_loss", justify="right")
    table.add_column("brier", justify="right")
    table.add_column("top1 (pos1)", justify="right")

    for name in (
        MODEL_POISSON,
        MODEL_ELO,
        MODEL_ML,
        MODEL_GBM,
        MODEL_ENSEMBLE,
    ):
        if name not in reports:
            continue
        report = reports[name]
        table.add_row(
            name,
            str(report.matches),
            f"{report.ranking_loss:.4f}",
            f"{report.log_loss:.4f}",
            f"{report.brier:.4f}",
            f"{report.top1_accuracy:.4f}",
        )

    console.print(table)
    console.print(
        f"[dim]Cada paso entrena con {min_prior_matches}+ partidos anteriores y predice "
        f"con features de fecha estricta (sin leakage).[/dim]"
    )