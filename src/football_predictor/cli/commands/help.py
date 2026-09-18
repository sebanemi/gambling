"""Comando ``help``: guía de comandos en español, sacada de docs/USO.md."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown

console = Console()

_DOCS = Path(__file__).resolve().parents[4] / "docs" / "USO.md"
_FALLBACK = Path(__file__).resolve().parents[3] / "docs" / "USO.md"


def _manual_text() -> str:
    for candidate in (_DOCS, _FALLBACK):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return ""


def _split_sections(text: str) -> dict[str, str]:
    """Separa el manual por encabezados ``##`comando` ``."""
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_title: str | None = None
    current: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current)
            current = []
            current_title = line[3:].strip("`").strip()
        elif current_title is not None:
            current.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(current)
    return sections


COMMANDS = {
    "init-db",
    "import-data",
    "status",
    "predict-match",
    "evaluate",
    "backtest",
    "collect-sofascore",
    "help",
}


def help_command(
    command: Annotated[
        str | None,
        typer.Argument(
            help="Comando del que querés la guía (predict-match, import-data, ...)."
        ),
    ] = None,
) -> None:
    """Muestra la guía completa de comandos o la de un comando en particular."""
    text = _manual_text()
    if not text:
        console.print(
            "[red]No encontré docs/USO.md dentro del proyecto.[/red]\n"
            "Sugerencia: mirá la guía en el repositorio del proyecto."
        )
        raise typer.Exit(code=1)

    sections = _split_sections(text)

    if command is not None:
        target = command.removeprefix("predictor ").strip("`")
        if target not in COMMANDS:
            console.print(
                f"[red]No existe el comando '{target}'.[/red]\n"
                f"Comandos: {', '.join(sorted(COMMANDS))}"
            )
            raise typer.Exit(code=1)
        section = sections.get(target)
        if section is None:
            console.print(f"[dim]El comando '{target}' aún no tiene guía propia.[/dim]")
            raise typer.Exit(code=0)
        console.print(Markdown(section))
        return

    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            break
        body.append(line)
    for title in sorted(sections):
        if title in COMMANDS or title == "Vista general":
            body.append(f"\n## {title}\n\n{sections[title]}")
    console.print(Markdown("\n".join(body).strip()))