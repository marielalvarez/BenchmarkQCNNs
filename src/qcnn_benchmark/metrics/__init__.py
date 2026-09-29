"""Métricas del benchmark, organizadas en cuatro familias: predictivas,
convergencia, recursos y robustez.

- Predictivas (`classification.py`): exactitud por lote, matriz de
  confusión y las métricas derivadas de ella (precision, recall, F1,
  balanced accuracy y macro-F1). Los resultados de E0, E0A, E0B y E1 se
  generaron antes de guardar la matriz de confusión, así que sus CSVs solo
  tienen `test_acc`. Recuperar tp/fp/fn/tn para esas corridas requiere
  reentrenarlas, lo cual es determinista dada la semilla (ver
  `scripts/backfill_predictive_metrics.py`).

- Convergencia (`convergence.py`): se calcula sin reentrenar a partir de
  tres valores guardados por corrida (`n_updates_run`, `stopped_early_at` y
  `best_val_loss`). `train_binary_classifier` calcula las curvas de pérdida
  completas, pero los CSVs solo guardan el resumen por corrida, así que no
  es posible reconstruir la forma de la curva de corridas ya terminadas.

- Recursos (`resources.py`): qubits y parámetros, tanto totales como
  efectivamente entrenados (difieren para Wei bajo shots o ruido). Se leen
  de los módulos de modelos y de `execution/` para no duplicar constantes.

- Robustez: aún no está implementada porque depende de los resultados de
  E2 (`E2_nisq_noise.ipynb`).
"""

from .classification import (
    balanced_accuracy_from_counts,
    batch_accuracy,
    confusion_counts,
    f1_from_counts,
    macro_f1_from_counts,
    precision_from_counts,
    predict_labels,
    recall_from_counts,
)
from .convergence import convergence_summary, early_stopping_rate
from .resources import n_wires, resource_summary, shots_budget_summary, total_params, trainable_params

__all__ = [
    "batch_accuracy",
    "predict_labels",
    "confusion_counts",
    "precision_from_counts",
    "recall_from_counts",
    "f1_from_counts",
    "balanced_accuracy_from_counts",
    "macro_f1_from_counts",
    "convergence_summary",
    "early_stopping_rate",
    "n_wires",
    "resource_summary",
    "shots_budget_summary",
    "total_params",
    "trainable_params",
]
