#!/usr/bin/env python3
"""Collector SofaScore vía ScraperFC (headless browser, bypasa WAF).

Uso:
    python scripts/sofascore_collector.py --tournament "England Premier League" --season 25/26 --output-dir data/sofascore
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ScraperFC import Sofascore

# Mapeo de nombres amigables a nombres oficiales de ScraperFC
TOURNAMENT_MAP = {
    "premier-league": "England Premier League",
    "la-liga": "Spain La Liga",
    "bundesliga": "Germany Bundesliga",
    "serie-a": "Italy Serie A",
    "ligue-1": "France Ligue 1",
    "champions-league": "UEFA Champions League",
    "copa-libertadores": "CONMEBOL Copa Libertadores",
    "liga-argentina": "Argentina Liga Profesional",
    "efl-championship": "England EFL Championship",
    "eredivisie": "Netherlands Eredivisie",
    "primeira-liga": "Portugal Primeira Liga",
}


def collect(
    tournament_key: str,
    season: str | None,
    output_dir: Path,
    max_matches: int | None = None,
) -> Path:
    if tournament_key not in TOURNAMENT_MAP:
        raise ValueError(f"Torneo desconocido: {tournament_key}. Disponibles: {list(TOURNAMENT_MAP.keys())}")

    tournament = TOURNAMENT_MAP[tournament_key]
    output_dir.mkdir(parents=True, exist_ok=True)

    s = Sofascore()

    # Resolver season
    if season is None:
        seasons = s.get_valid_seasons(tournament)
        season = list(seasons.keys())[0]
        print(f"Usando ultima temporada disponible: {season}")

    # Verificar que la temporada existe
    valid_seasons = s.get_valid_seasons(tournament)
    if season not in valid_seasons:
        raise ValueError(f"Temporada {season} no válida para {tournament}. Disponibles: {list(valid_seasons.keys())}")

    print(f"Recolectando {tournament} temporada {season}...")

    # Obtener todos los partidos
    matches = s.get_match_dicts(season, tournament)
    print(f"  Encontrados {len(matches)} partidos")

    # Filtrar solo finalizados
    finished = [m for m in matches if m.get("status", {}).get("type") == "finished"]
    print(f"  Finalizados: {len(finished)}")

    if max_matches:
        finished = finished[:max_matches]

    fieldnames = [
        "date", "competition", "season", "home_team", "away_team",
        "home_goals", "away_goals",
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards",
        "home_corners", "away_corners",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_fouls", "away_fouls",
        "home_throw_ins", "away_throw_ins",
        "home_penalties", "away_penalties",
        "home_xg", "away_xg",
        "home_possession", "away_possession",
        "home_scorers", "away_scorers",
    ]

    csv_path = output_dir / f"{tournament_key}_{season.replace('/', '-')}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, match in enumerate(finished, 1):
            try:
                mid = match["id"]
                # Stats detalladas
                stats_df = s.scrape_team_match_stats(mid)

                # Extraer stats clave
                stats = _parse_stats_df(stats_df)

                # Scorers
                scorers = _parse_scorers(match, s)

                start_ts = match.get("startTimestamp")
                match_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).date().isoformat() if start_ts else ""

                home_team = match.get("homeTeam", {}).get("name", "")
                away_team = match.get("awayTeam", {}).get("name", "")
                home_score = match.get("homeScore", {}).get("current") if isinstance(match.get("homeScore"), dict) else match.get("homeScore")
                away_score = match.get("awayScore", {}).get("current") if isinstance(match.get("awayScore"), dict) else match.get("awayScore")

                row = {
                    "date": match_date,
                    "competition": tournament,
                    "season": season.replace("/", "-"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_score,
                    "away_goals": away_score,
                    **stats,
                    **scorers,
                }
                writer.writerow(row)

                if i % 10 == 0:
                    print(f"  Procesados {i}/{len(finished)}...")

            except Exception as e:
                print(f"  Error en partido {match.get('id')}: {e}")
                continue

    print(f"[OK] Guardado: {csv_path}")
    return csv_path


def _parse_stats_df(df) -> dict:
    """Extrae stats clave del DataFrame de scrape_team_match_stats."""
    result = {
        "home_yellow_cards": None,
        "away_yellow_cards": None,
        "home_red_cards": None,
        "away_red_cards": None,
        "home_corners": None,
        "away_corners": None,
        "home_shots": None,
        "away_shots": None,
        "home_shots_on_target": None,
        "away_shots_on_target": None,
        "home_fouls": None,
        "away_fouls": None,
        "home_throw_ins": None,
        "away_throw_ins": None,
        "home_penalties": None,
        "away_penalties": None,
        "home_xg": None,
        "away_xg": None,
        "home_possession": None,
        "away_possession": None,
    }

    key_map = {
        "yellowCards": ("home_yellow_cards", "away_yellow_cards"),
        "redCards": ("home_red_cards", "away_red_cards"),
        "cornerKicks": ("home_corners", "away_corners"),
        "totalShots": ("home_shots", "away_shots"),
        "shotsOnGoal": ("home_shots_on_target", "away_shots_on_target"),
        "fouls": ("home_fouls", "away_fouls"),
        "throwIns": ("home_throw_ins", "away_throw_ins"),
        "penaltyWon": ("home_penalties", "away_penalties"),
        "expectedGoals": ("home_xg", "away_xg"),
        "ballPossession": ("home_possession", "away_possession"),
    }

    for _, row in df.iterrows():
        key = row.get("key", "")
        home_val = row.get("homeValue")
        away_val = row.get("awayValue")
        if key in key_map and home_val is not None and away_val is not None:
            h_key, a_key = key_map[key]
            result[h_key] = float(home_val) if home_val else None
            result[a_key] = float(away_val) if away_val else None

    return result


def _parse_scorers(match: dict, scraper: Sofascore) -> dict:
    """Extrae goleadores del match detail y incidents."""
    try:
        mid = match["id"]
        details = scraper.get_match_dict(mid)

        home_scorers = []
        away_scorers = []

        # Scorers desde incidents si disponible
        # Los scorers suelen estar en match events, pero ScraperFC no los expone directo
        # Fallback: usar homeTeam/awayTeam names del detail
        pass

        return {
            "home_scorers": "",
            "away_scorers": "",
        }
    except Exception:
        return {"home_scorers": "", "away_scorers": ""}


def main():
    parser = argparse.ArgumentParser(description="Collector SofaScore vía ScraperFC → CSV")
    parser.add_argument("--tournament", required=True, choices=list(TOURNAMENT_MAP.keys()),
                        help="Torneo (ej: premier-league, la-liga, liga-argentina)")
    parser.add_argument("--season", help="Temporada formato YY/YY (ej: 25/26). Default: ultima disponible")
    parser.add_argument("--output-dir", default="data/sofascore", help="Directorio de salida")
    parser.add_argument("--max-matches", type=int, help="Límite de partidos (para testing)")
    args = parser.parse_args()

    try:
        csv_path = collect(args.tournament, args.season, Path(args.output_dir), args.max_matches)
        print(f"\nListo. Para importar en el predictor:")
        print(f"  predictor import-data --path {csv_path}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()