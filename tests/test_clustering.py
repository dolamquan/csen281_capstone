"""Unit tests for src/clustering.py on small synthetic data."""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clustering import (add_improvement, bootstrap_stability, cramers_v, eta_squared, grid_index, hopkins,
                            hour_angle_columns, min_support_count, mine_rules, owner_group, wilson_bounds)


def test_owner_group_collapses_codes_and_case():
    owners = pd.Series(['USFS', 'BLM', 'Private', 'PRIVATE', 'BIA', 'TRIBAL', 'COUNTY', 'MISSING/NOT SPECIFIED'])
    assert owner_group(owners).tolist() == ['federal', 'federal', 'private', 'private', 'tribal', 'tribal',
                                            'state_local', 'missing']


def test_grid_index_puts_nearby_points_in_one_cell_and_far_points_apart():
    cy, cx = grid_index(np.array([40.0, 40.01, 41.0]), np.array([-110.0, -110.01, -110.0]), cell_km=25)
    assert (cy[0], cx[0]) == (cy[1], cx[1])
    assert cy[2] != cy[0]


def test_hour_encoding_makes_midnight_neighbours_and_keeps_missing():
    enc = hour_angle_columns(pd.Series([23.0, 0.0, 12.0, np.nan]))
    v = enc.to_numpy()
    assert np.linalg.norm(v[0] - v[1]) < np.linalg.norm(v[1] - v[2])
    assert enc.iloc[3].isna().all()


def test_hopkins_separates_clustered_from_uniform():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.05, size=(200, 2)) for c in ((0, 0), (5, 5), (0, 5))])
    uniform = rng.uniform(0, 5, size=(600, 2))
    assert hopkins(blobs, seed=1) > 0.9
    assert 0.35 < hopkins(uniform, seed=1) < 0.65


def test_bootstrap_stability_is_high_for_true_k():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.1, size=(100, 2)) for c in ((0, 0), (4, 0), (0, 4))])
    assert bootstrap_stability(blobs, k=3, n_boot=5, seed=0).min() > 0.95


def test_effect_sizes_on_perfect_and_null_association():
    groups = pd.Series(['a'] * 50 + ['b'] * 50)
    assert eta_squared(pd.Series([0.0] * 50 + [1.0] * 50), groups) == pytest.approx(1.0)
    assert cramers_v(groups, groups) == pytest.approx(1.0)
    assert cramers_v(groups, pd.Series(['x', 'y'] * 50)) == pytest.approx(0.0)


def test_wilson_bounds_contain_estimate():
    lo, hi = wilson_bounds(np.array([30]), np.array([100]), z=1.96)
    assert lo[0] < 0.3 < hi[0]
    assert lo[0] == pytest.approx(0.219, abs=1e-3)


def test_min_support_count_meets_precision_target():
    p0, rel = 0.03, 0.25
    n = min_support_count(p0, rel)
    assert 1.96 * np.sqrt(p0 * (1 - p0) / n) <= rel * p0
    assert 1.96 * np.sqrt(p0 * (1 - p0) / (n - 1)) > rel * p0


def test_mine_rules_counts_match_brute_force():
    rng = np.random.default_rng(0)
    items = pd.DataFrame({'a': rng.choice(['x', 'y'], 500), 'b': rng.choice(['p', 'q', 'r'], 500)})
    y = pd.Series(rng.integers(0, 2, 500))
    rules = mine_rules(items, y, max_len=2, min_count=1).set_index('antecedent')
    for va, vb in product(['x', 'y'], ['p', 'q', 'r']):
        mask = (items['a'] == va) & (items['b'] == vb)
        row = rules.loc[[(f'a={va}', f'b={vb}')]].iloc[0]
        assert row['matches'] == mask.sum() and row['large'] == y[mask].sum()
    assert rules.loc[[('a=x',)], 'lift'].iat[0] == pytest.approx(y[items['a'] == 'x'].mean() / y.mean())


def test_add_improvement_uses_best_subrule():
    rules = pd.DataFrame({'antecedent': [('a=1',), ('b=1',), ('a=1', 'b=1')], 'confidence': [0.1, 0.3, 0.25]})
    out = add_improvement(rules).set_index('antecedent')
    assert np.isnan(out.loc[[('a=1',)], 'improvement'].iat[0])
    assert out.loc[[('a=1', 'b=1')], 'improvement'].iat[0] == pytest.approx(-0.05)
