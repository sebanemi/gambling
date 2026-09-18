import pytest

from football_predictor.data.validators.csv import CsvRowValidator, CsvValidationError

validator = CsvRowValidator()


def _raw(**overrides):
    base = {
        "date": "2024-08-17",
        "competition": "Premier League",
        "season": "2024",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": "2",
        "away_goals": "1",
    }
    base.update(overrides)
    return base


def test_valid_row_parsed():
    row = validator.parse(_raw(), row_number=2)
    assert row.home_goals == 2
    assert row.away_goals == 1
    assert row.date.isoformat() == "2024-08-17"


def test_supports_alternate_date_formats():
    assert validator.parse(_raw(date="17/08/2024"), 2).date.isoformat() == "2024-08-17"
    assert validator.parse(_raw(date="17-08-2024"), 2).date.isoformat() == "2024-08-17"


@pytest.mark.parametrize("date_value", ["2024/08/17", "not-a-date", "", "32/13/2024"])
def test_invalid_dates(date_value):
    with pytest.raises(CsvValidationError):
        validator.parse(_raw(date=date_value), 7)


def test_empty_required_column_rejected():
    with pytest.raises(CsvValidationError, match="home_team"):
        validator.parse(_raw(home_team=" "), 3)


def test_same_team_rejected():
    with pytest.raises(CsvValidationError, match="mismo"):
        validator.parse(_raw(away_team="Arsenal"), 4)


def test_non_integer_goals_rejected():
    with pytest.raises(CsvValidationError, match="entero"):
        validator.parse(_raw(home_goals="2.5"), 5)


def test_negative_goals_rejected():
    with pytest.raises(CsvValidationError, match="entero"):
        validator.parse(_raw(home_goals="-1"), 6)


def test_error_carries_row_number():
    with pytest.raises(CsvValidationError) as exc:
        validator.parse(_raw(away_goals="x"), 12)
    assert exc.value.row_number == 12