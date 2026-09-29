"""Pruebas de las métricas de clasificación, convergencia y recursos."""

import numpy as np
import pandas as pd
import pytest

from qcnn_benchmark.metrics import (
    balanced_accuracy_from_counts,
    confusion_counts,
    convergence_summary,
    early_stopping_rate,
    f1_from_counts,
    macro_f1_from_counts,
    n_wires,
    precision_from_counts,
    recall_from_counts,
    resource_summary,
    shots_budget_summary,
    total_params,
    trainable_params,
)


def _fake_raw_df():
    return pd.DataFrame(
        {
            "dataset": ["d1"] * 5 + ["d2"] * 5,
            "model": ["wei"] * 10,
            "n_updates_run": [200, 200, 200, 200, 200, 40, 50, 60, 45, 55],
            "stopped_early_at": [None, None, None, None, None, 40, 50, 60, 45, 55],
            "best_val_loss": [0.1, 0.11, 0.09, 0.10, 0.12, 0.3, 0.31, 0.29, 0.30, 0.28],
        }
    )


def test_early_stopping_rate_mixed_group():
    df = _fake_raw_df()
    assert early_stopping_rate(df[df["dataset"] == "d1"]) == 0.0
    assert early_stopping_rate(df[df["dataset"] == "d2"]) == 1.0


def test_convergence_summary_degenerate_group_has_zero_width_ci():
    df = _fake_raw_df()
    summary = convergence_summary(df, rng=np.random.default_rng(0))
    d1 = summary[summary["dataset"] == "d1"].iloc[0]
    assert d1["n_updates_run_ci_lo"] == d1["n_updates_run_ci_hi"] == 200.0
    assert d1["early_stopping_rate"] == 0.0

    d2 = summary[summary["dataset"] == "d2"].iloc[0]
    assert d2["early_stopping_rate"] == 1.0
    assert d2["n_updates_run_ci_lo"] < d2["n_updates_run_mean"] < d2["n_updates_run_ci_hi"]


@pytest.mark.parametrize(
    "model_name,expected_wires,expected_total",
    [("hur", 8, 36), ("wei", 10, 46), ("gong", 8, 36), ("cong", 8, 51)],
)
def test_resource_facts_match_model_modules(model_name, expected_wires, expected_total):
    assert n_wires(model_name) == expected_wires
    assert total_params(model_name) == expected_total


def test_classical_analogs_have_no_wires():
    assert n_wires("cnn_wei_analoga") is None


def test_wei_trainable_params_differs_by_regime():
    assert trainable_params("wei", regime="analytic") == 46
    assert trainable_params("wei", regime="shots") == 37
    assert trainable_params("wei", regime="noise") == 37


def test_other_models_trainable_params_same_every_regime():
    for model_name in ("hur", "gong", "cong"):
        assert trainable_params(model_name, regime="analytic") == trainable_params(model_name, regime="shots")


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        trainable_params("wei", regime="not_a_regime")


def test_resource_summary_shape():
    summary = resource_summary(["hur", "wei", "gong", "cong"], regime="analytic")
    assert list(summary["model"]) == ["hur", "wei", "gong", "cong"]
    assert set(summary.columns) == {"model", "n_wires", "total_params", "trainable_params"}


def test_shots_budget_summary_reads_unique_sorted_values():
    df = pd.DataFrame({"n_shots": [1024, 256, 1024, 4096, 256]})
    assert shots_budget_summary(df) == [256, 1024, 4096]


def _perfect_predict_proba(params, x):
    # Devuelve la etiqueta guardada en x[0] para tener un clasificador perfecto
    return x[0]


def test_confusion_counts_perfect_classifier():
    X = np.array([[1.0], [1.0], [0.0], [0.0]])
    y = np.array([1, 1, 0, 0])
    counts = confusion_counts(_perfect_predict_proba, None, X, y)
    assert counts == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_confusion_counts_all_wrong():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([1, 1, 0, 0])
    counts = confusion_counts(_perfect_predict_proba, None, X, y)
    assert counts == {"tp": 0, "fp": 2, "fn": 2, "tn": 0}


def test_precision_recall_f1_from_counts_known_values():
    # tp=8, fp=2, fn=4, tn=6
    assert precision_from_counts(8, 2) == pytest.approx(0.8)
    assert recall_from_counts(8, 4) == pytest.approx(2 / 3)
    assert f1_from_counts(8, 2, 4) == pytest.approx(2 * 0.8 * (2 / 3) / (0.8 + 2 / 3))


def test_precision_recall_undefined_when_no_positive_predictions_or_labels():
    assert np.isnan(precision_from_counts(0, 0))
    assert np.isnan(recall_from_counts(0, 0))
    assert np.isnan(f1_from_counts(0, 0, 5))


def test_balanced_accuracy_matches_plain_accuracy_when_balanced():
    # clases balanceadas: balanced accuracy == accuracy
    tp, fp, fn, tn = 9, 1, 1, 9
    acc = (tp + tn) / (tp + fp + fn + tn)
    assert balanced_accuracy_from_counts(tp, fp, fn, tn) == pytest.approx(acc)


def test_balanced_accuracy_corrects_for_class_imbalance():
    # 90 positivos y 10 negativos, con un clasificador que ignora la clase minoritaria
    tp, fp, fn, tn = 90, 10, 0, 0
    acc = (tp + tn) / (tp + fp + fn + tn)
    balanced = balanced_accuracy_from_counts(tp, fp, fn, tn)
    assert acc == pytest.approx(0.9)
    assert balanced == pytest.approx(0.5)  # recall_pos=1.0, recall_neg=0.0


def test_macro_f1_symmetric_under_class_swap():
    tp, fp, fn, tn = 8, 2, 4, 6
    macro = macro_f1_from_counts(tp, fp, fn, tn)
    swapped = macro_f1_from_counts(tn, fn, fp, tp)
    assert macro == pytest.approx(swapped)
