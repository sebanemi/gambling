"""Tests sintéticos obligatorios de ausencia de DATA LEAKAGE (Fase 3).

Escenario causal: el resultado del partido N sólo puede depender de los
partidos 1..N-1. Por eso construimos features para N y las comparamos
contra un provedor que respeta ``date < cutoff``:

- proveedor "limpio": devuelve sólo partidos anteriores.
- proveedor "con fugas" (leaky): devuelve TODO el dataset, incluso el
  propio partido N y futuros.

Si el FeatureBuilder filtra correctamente, ambas salidas deben ser
idénticas: la información futura jamás se filtra a las features.
"""

from datetime import date

import pytest

from football_predictor.features.builder import FeatureBuilder
from tests.helpers import FakeHistoryProvider, hm, target


def _dataset_with_leakage_vector():
    # Serie con resultados que alterarían elo/forma/goles si se filtraran.
    return [
        hm(1, 1, 1, 2, 1, 0),   # A gana a B (día 1)
        hm(2, 2, 1, 3, 2, 2),   # A empata con C (día 2)
        hm(3, 3, 3, 2, 0, 4),   # C golea a B (día 3)
        # --- target: A vs B, día 5 ---
        hm(4, 5, 1, 2, 9, 9),   # resultado extremo del día 5 (¡debe excluirse!)
        hm(5, 6, 1, 3, 8, 0),   # futuro: A golea 8-0 (¡debe excluirse!)
        hm(6, 7, 3, 2, 0, 8),   # futuro: B goleado 0-8 (¡debe excluirse!)
    ]


def test_leaky_provider_produces_identical_features():
    dataset = _dataset_with_leakage_vector()
    clean = FeatureBuilder(FakeHistoryProvider(dataset, leaky=False)).build_for_match(target(4, 5, 1, 2))
    leaked = FeatureBuilder(FakeHistoryProvider(dataset, leaky=True)).build_for_match(target(4, 5, 1, 2))

    assert clean.model_dump() == leaked.model_dump()


def test_same_day_match_never_included():
    dataset = _dataset_with_leakage_vector()
    builder = FeatureBuilder(FakeHistoryProvider(dataset, leaky=True))
    fv = builder.build_for_match(target(4, 5, 1, 2))

    # El partido del día 5 (A 9-9 B) y los futuros no afectan forma/goles.
    assert fv.home_goals_for_avg == pytest.approx(1.5)   # (1 + 2) / 2, sin el 9
    assert fv.home_goals_against_avg == pytest.approx(1.0)  # (0 + 2) / 2
    # B: sólo jugó A-día1 (derrota) y C-día3 (victoria 4-0).
    assert fv.away_form.overall_3.played == 2
    assert (fv.away_form.overall_3.wins, fv.away_form.overall_3.losses) == (1, 1)


def test_features_of_partial_history_are_stable_across_leakage():
    # Cortamos el dataset: target sobre el 3er partido con "futuro" extra.
    for target_match_id, day_number in ((4, 5), (3, 3)):
        dataset = _dataset_with_leakage_vector()
        clean = FeatureBuilder(FakeHistoryProvider(dataset, leaky=False)).build_for_match(
            target(target_match_id, day_number, 1, 2)
        )
        leaked = FeatureBuilder(FakeHistoryProvider(dataset, leaky=True)).build_for_match(
            target(target_match_id, day_number, 1, 2)
        )
        assert clean.model_dump() == leaked.model_dump()


def test_elo_for_target_equals_elo_from_strict_predecessors():
    dataset = _dataset_with_leakage_vector()
    from football_predictor.features.elo import EloCalculator

    elo_full = EloCalculator().ratings_after(
        [m for m in dataset if m.date < date(2026, 1, 5)]
    )
    fv = FeatureBuilder(FakeHistoryProvider(dataset, leaky=True)).build_for_match(target(4, 5, 1, 2))
    assert fv.home_elo == pytest.approx(elo_full[1])
    assert fv.away_elo == pytest.approx(elo_full[2])