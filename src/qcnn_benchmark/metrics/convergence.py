"""Métricas de convergencia calculadas sobre los CSVs de resultados ya
guardados (`n_updates_run`, `stopped_early_at` y `best_val_loss` por
corrida), sin reentrenar.

Los CSVs no guardan las curvas de pérdida completas, así que este módulo
agrega esos tres valores entre semillas (media con IC bootstrap BCa) en
lugar de reconstruir la forma de la curva.
"""

import numpy as np
import pandas as pd

from qcnn_benchmark.stats import bootstrap_bca_ci


def _ci_or_degenerate(values, rng):
    """`bootstrap_bca_ci`, salvo cuando todas las semillas dan exactamente el
    mismo valor. Eso es común en `n_updates_run` cuando ninguna corrida hace
    early stopping y todas llegan al tope de actualizaciones. En ese caso el
    BCa de scipy devuelve NaN, así que el IC se reporta como el propio
    valor."""
    if np.std(values, ddof=1) == 0:
        return float(values[0]), float(values[0])
    return bootstrap_bca_ci(values, rng=rng)


def early_stopped(df):
    """Máscara booleana de las corridas de `df` con `stopped_early_at` no
    nulo."""
    return df["stopped_early_at"].notna()


def early_stopping_rate(df):
    """Fracción de corridas de `df` que activó early stopping."""
    return float(early_stopped(df).mean())


def convergence_summary(df, group_cols=("dataset", "model"), rng=None):
    """Resumen por grupo de un CSV de resultados cargado con pandas. Los
    grupos se forman con `group_cols` (por ejemplo dataset y model, o
    dataset, model y n_shots en E1). Para cada uno reporta la media con IC
    bootstrap BCa de `n_updates_run` y `best_val_loss` entre semillas, y la
    fracción de corridas con early stopping."""
    rng = np.random.default_rng(0) if rng is None else rng
    group_cols = list(group_cols)
    rows = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n_upd = group["n_updates_run"].to_numpy(dtype=float)
        best_vl = group["best_val_loss"].to_numpy(dtype=float)
        n_upd_lo, n_upd_hi = _ci_or_degenerate(n_upd, rng)
        vl_lo, vl_hi = _ci_or_degenerate(best_vl, rng)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_seeds": len(group),
                "n_updates_run_mean": float(np.mean(n_upd)),
                "n_updates_run_ci_lo": n_upd_lo,
                "n_updates_run_ci_hi": n_upd_hi,
                "early_stopping_rate": early_stopping_rate(group),
                "best_val_loss_mean": float(np.mean(best_vl)),
                "best_val_loss_ci_lo": vl_lo,
                "best_val_loss_ci_hi": vl_hi,
            }
        )
    return pd.DataFrame(rows)
