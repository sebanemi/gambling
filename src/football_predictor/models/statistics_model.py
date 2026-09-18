"""Modelo simple para recuentos por equipo (tarjetas, córners).

Usa promedios por equipo + ventaja de local (Poisson simple).
No modela interacción ataque/defensa; solo promedios históricos con
regularización ligera hacia la media de liga.
"""

import math
from collections.abc import Sequence
from collections import defaultdict

from football_predictor.domain.entities import HistoricalMatch


class StatisticsModel:
    """Promedios de Poisson por estadística (amarillas, rojas, córners).

    λ_local = exp(γ_local + att_local - def_visita + adv)
    Con regularización ligera: att/def se encogen hacia 0 (media de liga).
    """

    _STATS = ("yellow_cards", "red_cards", "corners")

    def __init__(self, regularization: float = 0.1) -> None:
        self._regularization = float(regularization)
        self._params: dict[str, dict] = {}

    def fit(self, history: Sequence[HistoricalMatch]) -> None:
        if not history:
            return  # Sin histórico → no hay stats, predicciones serán 0.0

        # Filtrar solo partidos que tienen al menos una stat no-None
        history_with_stats = [m for m in history if any(
            getattr(m, f"home_{s}") is not None or getattr(m, f"away_{s}") is not None
            for s in self._STATS
        )]
        if not history_with_stats:
            return  # Sin stats → predicciones 0.0

        teams = sorted({m.home_team_id for m in history_with_stats} | {m.away_team_id for m in history_with_stats})
        index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        for stat in self._STATS:
            filtered = [m for m in history_with_stats if getattr(m, f"home_{stat}") is not None or getattr(m, f"away_{stat}") is not None]
            if not filtered:
                self._params[stat] = {
                    "league_avg_home": 0.0,
                    "league_avg_away": 0.0,
                    "home_adv": 0.0,
                    "att": {team: 0.0 for team in teams},
                    "defense": {team: 0.0 for team in teams},
                }
                continue

            # Calcular promedios simples por equipo
            home_sums = defaultdict(float)
            home_counts = defaultdict(int)
            away_sums = defaultdict(float)
            away_counts = defaultdict(int)
            total_home = total_away = 0

            for m in filtered:
                h_val = getattr(m, f"home_{stat}") or 0
                a_val = getattr(m, f"away_{stat}") or 0
                home_sums[m.home_team_id] += h_val
                home_counts[m.home_team_id] += 1
                away_sums[m.away_team_id] += a_val
                away_counts[m.away_team_id] += 1
                total_home += h_val
                total_away += a_val

            n_matches = len(filtered)
            league_avg_home = total_home / n_matches if n_matches else 0
            league_avg_away = total_away / n_matches if n_matches else 0

            # Regularización hacia la media de liga
            att = {}
            defense = {}
            for team in teams:
                # Attack (local): shrink hacia media de liga
                h_c = sum(1 for m in filtered if m.home_team_id == team)
                if h_c > 0:
                    att_val = (sum(getattr(m, f"home_{stat}") or 0 for m in filtered if m.home_team_id == team) / h_c - league_avg_home) * (h_c / (h_c + self._regularization))
                else:
                    att_val = 0.0
                att[team] = att_val

                # Defense (visitante): shrink hacia media de liga
                a_c = sum(1 for m in filtered if m.away_team_id == team)
                if a_c > 0:
                    def_val = (sum(getattr(m, f"away_{stat}") or 0 for m in filtered if m.away_team_id == team) / a_c - league_avg_away) * (a_c / (a_c + self._regularization))
                else:
                    def_val = 0.0
                defense[team] = def_val

            # Ventaja de local
            home_adv = math.log(league_avg_home / max(league_avg_away, 1e-6)) if league_avg_away > 0 else 0.0

            self._params[stat] = {
                "league_avg_home": league_avg_home,
                "league_avg_away": league_avg_away,
                "home_adv": home_adv,
                "att": att,
                "defense": defense,
            }

    def expected_counts(self, home_team_id: int, away_team_id: int) -> dict[str, tuple[float, float]]:
        result = {}
        for stat in self._STATS:
            p = self._params.get(stat, {})
            if not p or "att" not in p:
                result[stat] = (0.0, 0.0)
                continue
            league_h = p.get("league_avg_home", 0.0)
            league_a = p.get("league_avg_away", 0.0)
            home_adv = p.get("home_adv", 0.0)
            att = p.get("att", {})
            defense = p.get("defense", {})

            att_h = att.get(home_team_id, 0.0)
            def_h = defense.get(home_team_id, 0.0)
            att_a = att.get(away_team_id, 0.0)
            def_a = defense.get(away_team_id, 0.0)

            lam_h = max(league_h + att_h - def_a + home_adv, 0.0)
            lam_a = max(league_a + att_a - def_h, 0.0)
            result[stat] = (lam_h, lam_a)
        return result