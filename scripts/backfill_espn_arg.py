#!/usr/bin/env python3
"""Backfill Liga Profesional (arg.1) desde el scoreboard público de ESPN.

Sin API key. Recorre día por día (el endpoint `dates=` devuelve ese día),
filtra eventos completados con marcador y escribe el CSV del proyecto:

    date,competition,season,home_team,away_team,home_goals,away_goals

Convención de fechas: día local America/Argentina/Buenos_Aires (igual que
`scripts/fetch_espn_arg.py` y `data/espn_arg_2026.csv`). La temporada sale
del año local, así que un rango multianual funciona en una sola corrida.

Uso:
    .venv/Scripts/python.exe scripts/backfill_espn_arg.py --from 2024-01-01 --to 2026-10-10 --out data/espn_arg_backfill.csv
    predictor import-data --path data/espn_arg_backfill.csv

Re-corridas son seguras: el importador hace upsert por
(competition, season, home, away, date) y cuenta duplicados.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/arg.1/scoreboard"
LOCAL_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) football-predictor/0.1"}


def fetch_day(client: httpx.Client, day: date, retries: int = 3) -> list[dict]:
    """Eventos del scoreboard para `day`, con reintentos."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.get(BASE_URL, params={"dates": day.strftime("%Y%m%d")}, timeout=30)
            response.raise_for_status()
            return response.json().get("events", [])
        except Exception as exc:  # noqa: BLE001 — reintento con backoff
            last = exc
            time.sleep(2 + attempt * 2)
    print(f"  ! sin datos {day.isoformat()}: {last}")
    return []


def parse_event(event: dict) -> tuple[str, str, str, int, int] | None:
    """(local_date, home, away, hg, ag) o None si no es un finalizado con marcador."""
    if not event.get("status", {}).get("type", {}).get("completed"):
        return None
    try:
        local_day = datetime.fromisoformat(event["date"]).astimezone(LOCAL_TZ).date()
    except (KeyError, ValueError):
        return None
    home = away = hs = aws = None
    competitors = event.get("competitors") or (event.get("competitions") or [{}])[0].get("competitors", [])
    for contender in competitors:
        try:
            score = int(contender.get("score", ""))
        except (TypeError, ValueError):
            return None
        if contender.get("homeAway") == "home":
            home, hs = contender["team"]["displayName"], score
        elif contender.get("homeAway") == "away":
            away, aws = contender["team"]["displayName"], score
    if not home or not away or home == away:
        return None
    return local_day.isoformat(), home, away, hs, aws


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill arg.1 desde ESPN (sin clave)")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD inicial")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD final")
    parser.add_argument("--out", required=True, help="CSV destino (formato del proyecto)")
    parser.add_argument("--competition", default="Liga Profesional")
    parser.add_argument("--sleep", type=float, default=0.15, help="pausa entre días (s)")
    args = parser.parse_args()

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)

    seen: dict[tuple[str, str, str], tuple[str, str, str, int, int]] = {}
    days = 0
    with httpx.Client(headers=HEADERS) as client:
        day = start
        while day <= end:
            for event in fetch_day(client, day):
                parsed = parse_event(event)
                if parsed is None:
                    continue
                local_day, home, away, hs, aws = parsed
                key = (local_day, home, away)
                if key not in seen:
                    seen[key] = (local_day, home, away, hs, aws)
            days += 1
            if days % 100 == 0:
                print(f"  ... {day.isoformat()} ({days} días, {len(seen)} partidos)", flush=True)
            day += timedelta(days=1)
            time.sleep(args.sleep)

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "competition", "season", "home_team", "away_team", "home_goals", "away_goals"])
        for (local_day, home, away) in sorted(seen):
            _d, _h, _a, hs, aws = seen[(local_day, home, away)]
            season = str(date.fromisoformat(local_day).year)
            writer.writerow([local_day, args.competition, season, home, away, hs, aws])
    print(f"días: {days}, partidos finalizados únicos: {len(seen)} -> {out}")


if __name__ == "__main__":
    main()
