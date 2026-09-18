from datetime import UTC, datetime
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
    home: Annotated[str | None, typer.Option(help='Equipo local (íntegro o parcial: "Brentford").')] = None,
    away: Annotated[str | None, typer.Option(help='Equipo visitante (íntegro o parcial: "Chelsea").')] = None,
    date: Annotated[str | None, typer.Option(help="Fecha del partido (YYYY-MM-DD); por defecto el próximo programado.")] = None,
    model: Annotated[
        str,
        typer.Option(help="Modelo: poisson, elo, ml, ensemble o all."),
    ] = "ensemble",
) -> None:
    """Predice un partido; pregunta los datos que falten y guarda el pronóstico."""
    if home is None:
        home = typer.prompt("Equipo local")
    if away is None:
        away = typer.prompt("Equipo visitante")

    match_date: _Date | None = None
    if date is not None:
        try:
            match_date = _Date.fromisoformat(date)
        except ValueError:
            console.print(f"[red]Error:[/red] fecha inválida '{date}' (usa YYYY-MM-DD).")
            raise typer.Exit(code=1) from None

    factory = session_factory()
    with factory() as session:
        teams = TeamRepository(session)
        home_id = teams.resolve(home)
        away_id = teams.resolve(away)
        if home_id is None:
            _suggest("local", home, teams)
            raise typer.Exit(code=1)
        if away_id is None:
            _suggest("visitante", away, teams)
            raise typer.Exit(code=1)
        if home_id == away_id:
            console.print("[red]Error:[/red] el equipo local y visitante deben ser distintos.")
            raise typer.Exit(code=1)

        home_name = teams.get_name(home_id) or home
        away_name = teams.get_name(away_id) or away

        matches = MatchRepository(session)
        if match_date is None:
            fixture = matches.find_next_scheduled(home_id, away_id, _today())
            if fixture is None:
                console.print(
                    f"[red]No hay partidos programados de {home_name} vs {away_name}.[/red]"
                )
                _show_upcoming(matches, teams, home_id, away_id)
                raise typer.Exit(code=1)
            match_date = fixture.date

        match = matches.find_by_teams_and_date(match_date, home_id, away_id)
        if match is None:
            console.print(
                f"[red]No existe partido {home_name} vs {away_name} el {match_date.isoformat()}.[/red]"
            )
            _show_upcoming(matches, teams, home_id, away_id)
            raise typer.Exit(code=1)

        results, stats = PredictionService(session).predict_match(
            match_id=match.id, model_name=model
        )

    _render(match.id, match_date, home_name, away_name, results)
    _render_stats(stats)
    console.print(
        f"[dim]Predicción guardada en BD (match_id={match.id}). Modelos usados: {', '.join(results)}.[/dim]"
    )


def _today() -> _Date:
    return datetime.now(UTC).date()


def _suggest(role: str, alias: str, teams: TeamRepository) -> None:
    similar = teams.search(alias, limit=6)
    console.print(f'[red]Error:[/red] no encuentro el equipo {role} "{alias}".')
    if similar:
        console.print("Quizá querías decir:")
        for name in similar:
            console.print(f"  [cyan]{name}[/cyan]")


def _show_upcoming(
    matches: MatchRepository, teams: TeamRepository, home_id: int, away_id: int
) -> None:
    for team_id, role in ((home_id, "local"), (away_id, "visitante")):
        fixtures = matches.upcoming(team_id, _today(), limit=3)
        if not fixtures:
            continue
        name = teams.get_name(team_id) or str(team_id)
        console.print(f"[cyan]Próximos del {role} ({name}):[/cyan]")
        for f in fixtures:
            console.print(
                f"  {f.date.isoformat()}  {f.home_team.name} — {f.away_team.name}"
            )


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
    table = Table(title="Estadísticas esperadas (Poisson por promedios)")
    table.add_column("Métrica", style="cyan")
    table.add_column("Local", justify="right", style="green")
    table.add_column("Visitante", justify="right", style="red")

    rows = (
        ("Tarjetas amarillas", stats.home_yellow_cards, stats.away_yellow_cards),
        ("Tarjetas rojas", stats.home_red_cards, stats.away_red_cards),
        ("Córners", stats.home_corners, stats.away_corners),
        ("Tiros", stats.home_shots, stats.away_shots),
        ("Tiros al arco", stats.home_shots_on_target, stats.away_shots_on_target),
        ("Faltas", stats.home_fouls, stats.away_fouls),
        ("Saques de banda", stats.home_throw_ins, stats.away_throw_ins),
        ("Penales", stats.home_penalties, stats.away_penalties),
        ("xG esperado", stats.home_xg, stats.away_xg),
        ("Posesión (%)", stats.home_possession, stats.away_possession),
    )
    for label, home, away in rows:
        if home == 0.0 and away == 0.0:
            table.add_row(label, "—", "—")
        else:
            table.add_row(label, f"{home:.2f}", f"{away:.2f}")

    console.print(table)