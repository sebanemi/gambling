from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football_predictor.betting.backtester import BettingBacktester
from football_predictor.config.settings import get_settings
from football_predictor.database.repositories.betting import BettingRepository
from football_predictor.database.repositories.match_history import MatchHistoryRepository
from football_predictor.database.repositories.match_odds import MatchOddsRepository
from football_predictor.database.session import session_factory
from football_predictor.services.model_trainer import ALL_MODELS
from football_predictor.utils.logging import get_logger

log = get_logger("cli.simulate_betting")
console = Console()


def simulate_betting(
    model: Annotated[
        str,
        typer.Option(help="Modelo a usar: poisson, elo, ml, gbm, ensemble."),
    ] = "ensemble",
    strategy: Annotated[
        str | None,
        typer.Option(help="Estrategia de apuestas: fixed, kelly, proportional, confidence."),
    ] = None,
    initial_bankroll: Annotated[
        float | None,
        typer.Option(help="Bankroll inicial."),
    ] = None,
    value_threshold: Annotated[
        float | None,
        typer.Option(help="Umbral de valor mínimo (EV)."),
    ] = None,
    bookmaker: Annotated[
        str | None,
        typer.Option(help="Bookmaker a usar (por defecto: mejor cuota entre bookmakers)."),
    ] = None,
    no_save: Annotated[
        bool,
        typer.Option("--no-save", help="No persistir la simulación en BD."),
    ] = False,
) -> None:
    """Simula estrategia de apuestas sobre el histórico con cuotas reales."""
    settings = get_settings()

    if model not in ALL_MODELS:
        console.print(f"[red]Error:[/red] modelo inválido '{model}'. Válidos: {', '.join(ALL_MODELS)}.")
        raise typer.Exit(code=1)

    # Use settings defaults if not provided
    if strategy is None:
        strategy = settings.betting_strategy
    if initial_bankroll is None:
        initial_bankroll = settings.betting_initial_bankroll
    if value_threshold is None:
        value_threshold = settings.betting_value_threshold

    factory = session_factory()
    with factory() as session:
        provider = MatchHistoryRepository(session)
        odds_repo = MatchOddsRepository(session)

        def odds_lookup(match_id: int) -> dict[str, float] | None:
            if bookmaker:
                rows = odds_repo.get_for_match(match_id, bookmaker)
                if not rows:
                    return None
                best = rows[0]
                return {
                    "home": float(best.home_win_odds),
                    "draw": float(best.draw_odds),
                    "away": float(best.away_win_odds),
                }
            return odds_repo.get_best_odds(match_id)

        backtester = BettingBacktester(
            provider,
            settings,
            min_prior_matches=settings.betting_backtest_min_matches,
            odds_lookup=odds_lookup,
            bookmaker=bookmaker or "best",
        )

        console.print(f"[cyan]Simulando apuestas con modelo '{model}' y estrategia '{strategy}'...[/cyan]")
        console.print(f"Bankroll inicial: ${initial_bankroll:.2f}")
        console.print(f"Umbral de valor: {value_threshold:.1%}\n")

        result = backtester.run(
            model_name=model,
            strategy_name=strategy,
            initial_bankroll=initial_bankroll,
            value_threshold=value_threshold,
            min_odds=settings.betting_min_odds,
            max_odds=settings.betting_max_odds,
        )

        simulation_id: int | None = None
        if not no_save:
            record = BettingRepository(session).save_simulation(
                model_name=model,
                strategy=strategy,
                result=result,
                parameters={
                    "value_threshold": value_threshold,
                    "min_odds": settings.betting_min_odds,
                    "max_odds": settings.betting_max_odds,
                    "bookmaker": bookmaker or "best",
                    "min_prior_matches": settings.betting_backtest_min_matches,
                },
            )
            simulation_id = record.id

    _render_summary(result, simulation_id)
    _render_bets(result.bets[:10])  # Show first 10 bets


def _render_summary(result, simulation_id: int | None) -> None:
    table = Table(title="Resumen de Simulación de Apuestas")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green", justify="right")

    table.add_row("Total de apuestas", str(result.total_bets))
    table.add_row("Apuestas ganadas", str(result.winning_bets))
    table.add_row("Apuestas perdidas", str(result.losing_bets))
    table.add_row("Win rate", f"{result.win_rate:.2%}")
    table.add_row("Sin cuota (omitidos)", str(result.skipped_no_odds))
    table.add_row("Sin valor (omitidos)", str(result.skipped_no_value))
    table.add_row("Total apostado", f"${result.total_staked:.2f}")
    table.add_row("Total retorno", f"${result.total_return:.2f}")
    table.add_row("ROI", f"{result.roi:.2%}")
    table.add_row("Drawdown máximo", f"{result.max_drawdown:.2%}")
    table.add_row("Bankroll inicial", f"${result.initial_bankroll:.2f}")
    table.add_row("Bankroll final", f"${result.final_bankroll:.2f}")
    profit = result.final_bankroll - result.initial_bankroll
    table.add_row("Profit/Loss", f"${profit:+.2f}")

    console.print(table)
    if simulation_id is not None:
        console.print(f"[dim]Simulación guardada en BD (id={simulation_id}).[/dim]")
    if result.total_bets == 0:
        console.print(
            "[yellow]Sin apuestas:[/yellow] importá cuotas con "
            "`predictor import-odds --path data/sample_odds.csv` y/o bajá el umbral de valor."
        )


def _render_bets(bets) -> None:
    if not bets:
        return

    table = Table(title="Primeras 10 Apuestas")
    table.add_column("Match ID", justify="right")
    table.add_column("Fecha")
    table.add_column("Predicción")
    table.add_column("Prob.", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("Stake", justify="right")
    table.add_column("P/L", justify="right")
    table.add_column("Bankroll", justify="right")

    for bet in bets:
        pl_color = "green" if bet.profit_loss > 0 else "red"
        table.add_row(
            str(bet.match_id),
            str(bet.match_date),
            bet.prediction,
            f"{bet.model_probability:.3f}",
            f"{bet.odds:.2f}",
            f"${bet.stake:.2f}",
            f"[{pl_color}]{bet.profit_loss:+.2f}[/{pl_color}]",
            f"${bet.bankroll_after:.2f}",
        )

    console.print(table)
