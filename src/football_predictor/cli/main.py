import typer

from football_predictor.cli.commands import (
    backtest,
    collect_sofascore,
    evaluate,
    import_data,
    init_db,
    predict_match,
    status,
)
from football_predictor.utils.logging import configure_logging

app = typer.Typer(
    name="predictor",
    help="Football match analysis and prediction (statistical, auditable).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log a nivel DEBUG."),
) -> None:
    """Configuración global de la CLI."""
    configure_logging("DEBUG" if verbose else "INFO")


app.command("init-db", help="Crea/actualiza el esquema de la BD (alembic upgrade head).")(
    init_db.init_database
)
app.command(
    "import-data",
    help="Importa partidos o catalogo (CSV o fuente externa API) a PostgreSQL.",
)(import_data.import_data)
app.command("status", help="Estado de la BD: revisión aplicada y head.")(status.status)
app.command(
    "predict-match",
    help="Predice un partido y guarda el resultado con snapshot de features.",
)(predict_match.predict_match)
app.command(
    "evaluate",
    help="Evalúa predicciones guardadas sobre partidos ya jugados.",
)(evaluate.evaluate)
app.command(
    "backtest",
    help="Backtest walk-forward: predice cada partido solo con su pasado.",
)(backtest.backtest)
app.command(
    "collect-sofascore",
    help="Recolecta stats completas de SofaScore (shots, SOT, faltas, xG, goleadores...) y las importa. Requiere IP residencial.",
)(collect_sofascore.collect_sofascore)


if __name__ == "__main__":
    app()