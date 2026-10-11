"""Encuentra apuestas de valor en fixtures programados (scheduled)."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from football_predictor.betting.value import expected_value
from football_predictor.config.settings import get_settings
from football_predictor.database.models.match import Match
from football_predictor.database.repositories.match_history import MatchHistoryRepository
from football_predictor.database.repositories.match_odds import MatchOddsRepository
from football_predictor.database.repositories.team import TeamRepository
from football_predictor.database.session import session_factory
from football_predictor.domain.entities import TargetMatch
from football_predictor.services.model_trainer import ALL_MODELS, ModelTrainer
from football_predictor.utils.logging import get_logger

log = get_logger("cli.value_bets")
console = Console()


def value_bets(
    model: Annotated[
        str,
        typer.Option(help="Modelo a usar: poisson, elo, ml, gbm, ensemble."),
    ] = "ensemble",
    threshold: Annotated[
        float | None,
        typer.Option(help="EV mínimo para mostrar una apuesta (0.05 = 5%)."),
    ] = None,
    bookmaker: Annotated[
        str | None,
        typer.Option(help="Bookmaker a usar (por defecto: mejor cuota)."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Máximo de fixtures a evaluar."),
    ] = 20,
) -> None:
    """Lista apuestas de valor en partidos programados (usa solo el pasado)."""
    settings = get_settings()
    if threshold is None:
        threshold = settings.betting_value_threshold
    if model not in ALL_MODELS:
        console.print(f"[red]Error:[/red] modelo inválido '{model}'. Válidos: {', '.join(ALL_MODELS)}.")
        raise typer.Exit(code=1)

    factory = session_factory()
    with factory() as session:
        history = MatchHistoryRepository(session)
        try:
            models = ModelTrainer(history, settings).train_all()
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        predictor = models[model]
        builder = ModelTrainer(history, settings).feature_builder()

        fixtures = list(
            session.execute(
                select(Match).where(Match.status == "scheduled").order_by(Match.date, Match.id).limit(limit)
            ).scalars()
        )
        if not fixtures:
            console.print("[yellow]No hay fixtures programados (status='scheduled').[/yellow]")
            return

        teams = TeamRepository(session)
        odds_repo = MatchOddsRepository(session)
        rows: list[tuple] = []
        for fixture in fixtures:
            features = builder.build_for_match(
                TargetMatch(
                    match_id=fixture.id,
                    date=fixture.date,
                    home_team_id=fixture.home_team_id,
                    away_team_id=fixture.away_team_id,
                )
            )
            result = predictor.predict(features)
            probs = {"HOME": result.home_win, "DRAW": result.draw, "AWAY": result.away_win}

            if bookmaker:
                odds_rows = odds_repo.get_for_match(fixture.id, bookmaker)
                if not odds_rows:
                    continue
                odds = {
                    "HOME": float(odds_rows[0].home_win_odds),
                    "DRAW": float(odds_rows[0].draw_odds),
                    "AWAY": float(odds_rows[0].away_win_odds),
                }
                book = bookmaker
            else:
                best = odds_repo.get_best_odds(fixture.id)
                if best["home"] <= 1.0:
                    continue
                odds = {"HOME": best["home"], "DRAW": best["draw"], "AWAY": best["away"]}
                book = "best"

            for outcome in ("HOME", "DRAW", "AWAY"):
                ev = expected_value(probs[outcome], odds[outcome])
                if ev > threshold:
                    rows.append(
                        (
                            fixture.date.isoformat(),
                            teams.get_name(fixture.home_team_id) or str(fixture.home_team_id),
                            teams.get_name(fixture.away_team_id) or str(fixture.away_team_id),
                            outcome,
                            probs[outcome],
                            odds[outcome],
                            ev,
                            book,
                        )
                    )

    rows.sort(key=lambda r: r[6], reverse=True)
    if not rows:
        console.print("[yellow]Sin apuestas de valor sobre el umbral.[/yellow]")
        return

    table = Table(title=f"Apuestas de valor ({model}, EV > {threshold:.0%})")
    table.add_column("Fecha", style="cyan")
    table.add_column("Partido")
    table.add_column("Apuesta", justify="center")
    table.add_column("P(modelo)", justify="right")
    table.add_column("Cuota", justify="right")
    table.add_column("EV", justify="right", style="green")
    table.add_column("Bookmaker")
    for match_date, home, away, outcome, prob, odd, ev, book in rows:
        table.add_row(match_date, f"{home} vs {away}", outcome, f"{prob:.3f}", f"{odd:.2f}", f"{ev:+.1%}", book)
    console.print(table)
