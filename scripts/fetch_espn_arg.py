#!/usr/bin/env python3
"""Fetch Argentine Liga Profesional results from ESPN's free scoreboard API.

ESPN needs no API key and covers arg.1 (Apertura + Clausura). The scoreboard
endpoint returns ~1 week of events per request, so we page week by week and
dedupe by ESPN event id.

Usage:
    .venv/Scripts/python.exe scripts/fetch_espn_arg.py --from 2026-01-19 --to 2026-10-10
    .venv/Scripts/python.exe scripts/fetch_espn_arg.py --from 2026-01-19 --to 2026-10-10 --out data/espn_arg_2026.csv

Output CSV matches the predictor's import format:
date,competition,season,home_team,away_team,home_goals,away_goals
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

LEAGUE = "arg.1"
BASE_URL = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{LEAGUE}/scoreboard"
LOCAL_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) football-predictor/0.1"}

FINISHED = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_ENDED"}


def fetch_week(client: httpx.Client, day: date) -> list[dict]:
    r = client.get(BASE_URL, params={"dates": day.strftime("%Y%m%d")}, timeout=30)
    r.raise_for_status()
    return r.json().get("events", [])


def parse_event(event: dict) -> dict | None:
    status = event.get("status", {}).get("type", {}).get("name", "")
    if status not in FINISHED:
        return None
    try:
        local_date = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(LOCAL_TZ).date()
    except (KeyError, ValueError):
        return None
    home = away = hs = aws = None
    competitors = event.get("competitors") or (event.get("competitions") or [{}])[0].get("competitors", [])
    for c in competitors:
        try:
            score = int(c.get("score", ""))
        except (TypeError, ValueError):
            return None
        if c.get("homeAway") == "home":
            home, hs = c["team"]["displayName"], score
        elif c.get("homeAway") == "away":
            away, aws = c["team"]["displayName"], score
    if not home or not away or home == away:
        return None
    return {
        "id": str(event.get("id")),
        "date": local_date.isoformat(),
        "home_team": home,
        "away_team": away,
        "home_goals": hs,
        "away_goals": aws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arg.1 results from ESPN")
    parser.add_argument("--from", dest="from_date", default="2026-01-19")
    parser.add_argument("--to", dest="to_date", default="2026-10-10")
    parser.add_argument("--out", default="data/espn_arg_2026.csv")
    parser.add_argument("--competition", default="Liga Profesional")
    parser.add_argument("--season", default="2026")
    args = parser.parse_args()

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)

    seen: dict[str, dict] = {}
    weeks = 0
    with httpx.Client(headers=HEADERS) as client:
        day = start
        while day <= end:
            try:
                for event in fetch_week(client, day):
                    row = parse_event(event)
                    if row and row["id"] not in seen:
                        seen[row["id"]] = row
                weeks += 1
            except Exception as exc:  # noqa: BLE001 — skip a bad week, keep the rest
                print(f"week {day}: {exc}")
            time.sleep(0.3)
            day += timedelta(days=3)

    rows = sorted(seen.values(), key=lambda r: (r["date"], r["home_team"], r["away_team"]))
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "competition", "season", "home_team", "away_team", "home_goals", "away_goals"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date": row["date"],
                    "competition": args.competition,
                    "season": args.season,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_goals": row["home_goals"],
                    "away_goals": row["away_goals"],
                }
            )
    print(f"weeks queried: {weeks}, unique finished matches: {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
