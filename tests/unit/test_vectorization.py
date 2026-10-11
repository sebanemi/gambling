"""Tests del fast path de vectorización (to_array / feature_columns / snapshot flag)."""

import numpy as np

from football_predictor.domain.features import feature_columns, matrix_from_vectors
from football_predictor.models.base import Outcome
from football_predictor.models.elo_model import EloModel
from football_predictor.models.ensemble import EnsembleModel
from football_predictor.models.ml import MLModel
from tests.helpers import fv


def test_columns_match_flatten_order():
    vec = fv(1, 1, 2, home_elo=1520.0, away_elo=1480.0)
    assert tuple(vec.flatten().keys()) == feature_columns()
    assert len(feature_columns()) == 128


def test_to_array_matches_flatten_values():
    vec = fv(
        1,
        1,
        2,
        home_elo=1520.0,
        away_elo=1480.0,
        home_goals_for_avg=1.5,
        away_goals_for_avg=2.0,
    )
    assert list(vec.flatten().values()) == vec.to_array()
    assert len(vec.to_array()) == 128


def test_make_matrix_matches_legacy_build():
    vecs = [fv(i, 1, 2, home_elo=1500.0 + i) for i in range(5)]
    fast = MLModel.make_matrix(vecs)
    columns = list(vecs[0].flatten().keys())
    legacy = np.array([[v.flatten()[c] for c in columns] for v in vecs], dtype=float)
    assert fast.shape == (5, 128)
    assert np.array_equal(fast, legacy)


def test_predict_without_snapshot_matches_probs():
    vecs = [fv(i, 1, 2, home_elo=1600.0 if i % 2 == 0 else 1400.0) for i in range(10)]
    outcomes = [Outcome.HOME if i % 2 == 0 else Outcome.AWAY for i in range(10)]
    model = MLModel()
    model.fit(vecs, outcomes)

    target = fv(99, 1, 2, home_elo=1600.0)
    full = model.predict(target)
    fast = model.predict(target, include_snapshot=False)
    assert fast.features_snapshot is None
    assert full.features_snapshot is not None
    assert (fast.home_win, fast.draw, fast.away_win) == (
        full.home_win,
        full.draw,
        full.away_win,
    )


def test_ensemble_without_snapshot_skips_member_snapshots():
    vecs = [fv(i, 1, 2, home_elo=1600.0 if i % 2 == 0 else 1400.0) for i in range(10)]
    outcomes = [Outcome.HOME if i % 2 == 0 else Outcome.AWAY for i in range(10)]
    ml = MLModel()
    ml.fit(vecs, outcomes)
    ensemble = EnsembleModel([ml, EloModel()], [0.5, 0.5])

    target = fv(99, 1, 2, home_elo=1600.0)
    full = ensemble.predict(target)
    fast = ensemble.predict(target, include_snapshot=False)
    assert fast.features_snapshot is None
    assert full.features_snapshot is not None
    assert (fast.home_win, fast.draw, fast.away_win) == (
        full.home_win,
        full.draw,
        full.away_win,
    )


def test_matrix_from_vectors_shape():
    vecs = [fv(i, 1, 2) for i in range(3)]
    assert matrix_from_vectors(vecs).shape == (3, 128)
