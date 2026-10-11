from datetime import date

from football_predictor.data.importers.odds_importer import OddsImporter
from football_predictor.database.models.competition import Competition
from football_predictor.database.models.match import Match
from football_predictor.database.models.season import Season
from football_predictor.database.models.team import Team
from football_predictor.database.repositories.match_odds import MatchOddsRepository


def _seed(sqlite_session):
    comp = Competition(name="Liga")
    sqlite_session.add(comp)
    sqlite_session.flush()
    season = Season(name="2024", competition_id=comp.id)
    sqlite_session.add(season)
    sqlite_session.flush()
    home = Team(name="River Plate")
    away = Team(name="Boca Juniors")
    sqlite_session.add_all([home, away])
    sqlite_session.flush()
    match = Match(
        competition_id=comp.id,
        season_id=season.id,
        date=date(2024, 8, 17),
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=2,
        away_goals=1,
    )
    sqlite_session.add(match)
    sqlite_session.flush()
    return match


def test_import_odds_upserts(sqlite_session, tmp_path):
    match = _seed(sqlite_session)
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds\n"
        "2024-08-17,River Plate,Boca Juniors,bet365,2.10,3.40,3.20\n",
        encoding="utf-8",
    )
    summary = OddsImporter(sqlite_session, csv_path).import_all()
    assert summary.rows_read == 1
    assert summary.upserted == 1
    assert summary.invalid == 0

    rows = MatchOddsRepository(sqlite_session).get_for_match(match.id)
    assert len(rows) == 1
    assert float(rows[0].home_win_odds) == 2.10


def test_import_odds_counts_unknown_and_invalid(sqlite_session, tmp_path):
    _seed(sqlite_session)
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds\n"
        "2024-08-17,River Plate,Boca Juniors,bet365,2.10,3.40,3.20\n"
        "2024-08-17,River Plate,Equipo Fantasma,bet365,2.10,3.40,3.20\n"
        "2024-08-17,River Plate,Boca Juniors,bet365,0.50,3.40,3.20\n",
        encoding="utf-8",
    )
    summary = OddsImporter(sqlite_session, csv_path).import_all()
    assert summary.rows_read == 3
    assert summary.upserted == 1
    assert summary.unknown_matches == 1
    assert summary.invalid == 1


def test_import_odds_reimport_updates(sqlite_session, tmp_path):
    match = _seed(sqlite_session)
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds\n"
        "2024-08-17,River Plate,Boca Juniors,bet365,2.10,3.40,3.20\n",
        encoding="utf-8",
    )
    OddsImporter(sqlite_session, csv_path).import_all()
    csv_path.write_text(
        "date,home_team,away_team,bookmaker,home_odds,draw_odds,away_odds\n"
        "2024-08-17,River Plate,Boca Juniors,bet365,2.50,3.00,3.00\n",
        encoding="utf-8",
    )
    OddsImporter(sqlite_session, csv_path).import_all()
    rows = MatchOddsRepository(sqlite_session).get_for_match(match.id)
    assert len(rows) == 1
    assert float(rows[0].home_win_odds) == 2.50
