"""PCA densa (Hur et al.): 16 componentes, ajustada solo en train, escalada
a [0, π] con min/max de train."""

import numpy as np
from sklearn.decomposition import PCA

from qcnn_benchmark.data.sampling import (
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
    N_VAL_PER_CLASS,
    SAMPLE_SEED,
    stratified_split,
)

N_COMPONENTS = 16


def dense_pca_representation(raw_splits, n_components=N_COMPONENTS, seed=SAMPLE_SEED):
    pca = PCA(n_components=n_components, random_state=seed)
    # PCA y escalador se ajustan solo con train para evitar fuga de datos
    X_train_pca = pca.fit_transform(raw_splits["X_train"])
    X_val_pca = pca.transform(raw_splits["X_val"])
    X_test_pca = pca.transform(raw_splits["X_test"])

    lo, hi = X_train_pca.min(), X_train_pca.max()
    scale = lambda X: (X - lo) * (np.pi / (hi - lo))

    return {
        "X_train": scale(X_train_pca),
        "y_train": raw_splits["y_train"],
        "X_val": scale(X_val_pca),
        "y_val": raw_splits["y_val"],
        "X_test": scale(X_test_pca),
        "y_test": raw_splits["y_test"],
        "pca": pca,
        "scale_lo": lo,
        "scale_hi": hi,
    }


def build_pca_dataset(
    x_all,
    y_all,
    class_pos,
    class_neg,
    n_components=N_COMPONENTS,
    n_train=N_TRAIN_PER_CLASS,
    n_val=N_VAL_PER_CLASS,
    n_test=N_TEST_PER_CLASS,
    seed=SAMPLE_SEED,
    verbose=True,
):
    raw = stratified_split(x_all, y_all, class_pos, class_neg, n_train, n_val, n_test, seed)
    rep = dense_pca_representation(raw, n_components=n_components, seed=seed)
    # Solo train queda garantizado dentro de [0, pi]. Val y test pueden
    # salirse un poco del rango porque el escalador nunca ve sus
    # estadísticos, y eso es lo esperado.
    assert rep["X_train"].shape[1] == n_components
    assert rep["X_train"].min() >= -1e-9 and rep["X_train"].max() <= np.pi + 1e-9
    for split in ("train", "val", "test"):
        assert rep[f"X_{split}"].shape[1] == n_components
        if verbose:
            lo, hi = rep[f"X_{split}"].min(), rep[f"X_{split}"].max()
            note = "" if split == "train" else " (puede exceder [0, pi], ver nota arriba)"
            print(f"  rango de {split}: [{lo:.4f}, {hi:.4f}]{note}")
    return rep
