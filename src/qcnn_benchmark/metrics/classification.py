"""Métricas predictivas evaluadas post-hoc (fuera del ciclo de gradiente)
sobre un `predict_proba_fn(params, x)` ya entrenado.

`confusion_counts` es la única que evalúa el modelo sobre `X`. Las
funciones `*_from_counts` son aritmética sobre tp, fp, fn y tn, y con eso
basta para accuracy, balanced accuracy, precision, recall y F1 (macro y
por clase). Así cada corrida solo necesita guardar 4 enteros en lugar de
las predicciones de todo el conjunto de prueba."""

import numpy as np


def predict_labels(predict_proba_fn, params, X, threshold=0.5):
    return np.array([1 if float(predict_proba_fn(params, x)) >= threshold else 0 for x in X])


def batch_accuracy(predict_proba_fn, params, X, y):
    y_pred = predict_labels(predict_proba_fn, params, X)
    return float(np.mean(y_pred == np.asarray(y).astype(int)))


def confusion_counts(predict_proba_fn, params, X, y, threshold=0.5):
    """tp, fp, fn y tn de un `predict_proba_fn` entrenado sobre (X, y), con
    la etiqueta 1 como clase positiva."""
    y_pred = predict_labels(predict_proba_fn, params, X, threshold)
    y_true = np.asarray(y).astype(int)
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
    }


def precision_from_counts(tp, fp):
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def recall_from_counts(tp, fn):
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def f1_from_counts(tp, fp, fn):
    p, r = precision_from_counts(tp, fp), recall_from_counts(tp, fn)
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def balanced_accuracy_from_counts(tp, fp, fn, tn):
    """Media del recall por clase (sensibilidad de la clase 1 y
    especificidad de la clase 0). Junto con macro-F1 evita sobreestimar el
    desempeño cuando un clasificador ignora la clase minoritaria."""
    recall_pos = recall_from_counts(tp, fn)
    recall_neg = recall_from_counts(tn, fp)
    return float((recall_pos + recall_neg) / 2)


def macro_f1_from_counts(tp, fp, fn, tn):
    """F1 promediado sobre ambas clases (clase 0 tratada como positiva
    intercambiando tp<->tn y fp<->fn)."""
    f1_pos = f1_from_counts(tp, fp, fn)
    f1_neg = f1_from_counts(tn, fn, fp)
    return float((f1_pos + f1_neg) / 2)
