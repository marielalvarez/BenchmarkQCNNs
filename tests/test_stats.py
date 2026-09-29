"""Pruebas del módulo de análisis estadístico (bootstrap BCa, permutación
pareada, Holm y tamaños de efecto)."""

import numpy as np

from qcnn_benchmark.stats import (
    bootstrap_bca_ci,
    cohens_dz,
    holm_correction,
    paired_sign_permutation_test,
    probability_of_superiority,
)


def test_bootstrap_bca_ci_contains_true_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.9, scale=0.02, size=30)
    lo, hi = bootstrap_bca_ci(data, rng=rng, n_bootstrap=2000)
    assert lo < np.mean(data) < hi
    assert lo < 0.9 < hi


def test_paired_permutation_test_detects_real_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.95, scale=0.01, size=20)
    y = rng.normal(loc=0.80, scale=0.01, size=20)
    p_value, mean_diff = paired_sign_permutation_test(x, y, rng=rng, n_permutations=2000)
    assert mean_diff > 0
    assert p_value < 0.01


def test_paired_permutation_test_no_difference_gives_large_p():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.9, scale=0.05, size=20)
    y = rng.normal(loc=0.9, scale=0.05, size=20)
    p_value, _ = paired_sign_permutation_test(x, y, rng=rng, n_permutations=2000)
    assert p_value > 0.05


def test_holm_correction_is_more_conservative_than_raw_p_values():
    p_values = [0.001, 0.01, 0.04, 0.20]
    adjusted, rejected = holm_correction(p_values, alpha=0.05)
    assert np.all(adjusted >= np.array(p_values))
    assert rejected[0]  # el p-valor más chico sigue siendo significativo
    assert not rejected[3]


def test_cohens_dz_sign_matches_direction():
    x = np.array([0.9, 0.92, 0.88, 0.91])
    y = np.array([0.7, 0.72, 0.68, 0.71])
    assert cohens_dz(x, y) > 0
    assert cohens_dz(y, x) < 0


def test_probability_of_superiority_extremes():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 0.0])
    assert probability_of_superiority(x, y) == 1.0
    assert probability_of_superiority(y, x) == 0.0
    assert probability_of_superiority(x, x) == 0.5
