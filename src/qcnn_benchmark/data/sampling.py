"""Muestreo estratificado balanceado con semilla fija, compartido entre
todas las representaciones (PCA densa, amplitud, ...)."""

import numpy as np

SAMPLE_SEED = 20260802
N_TRAIN_PER_CLASS = 500
N_VAL_PER_CLASS = 250
N_TEST_PER_CLASS = 500


def stratified_split(
    x_all,
    y_all,
    class_pos,
    class_neg,
    n_train=N_TRAIN_PER_CLASS,
    n_val=N_VAL_PER_CLASS,
    n_test=N_TEST_PER_CLASS,
    seed=SAMPLE_SEED,
):
    """n_train/n_val/n_test por clase, balanceado, semilla fija.

    class_pos -> etiqueta 1, class_neg -> etiqueta 0.
    """
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    label_of = {}
    for cls in sorted([class_pos, class_neg]):
        idx_cls = np.where(y_all == cls)[0]
        n_needed = n_train + n_val + n_test
        assert len(idx_cls) >= n_needed, f"clase {cls} solo tiene {len(idx_cls)} muestras"
        idx_cls = rng.permutation(idx_cls)
        train_idx = idx_cls[:n_train]
        val_idx = idx_cls[n_train : n_train + n_val]
        test_idx = idx_cls[n_train + n_val : n_train + n_val + n_test]
        splits["train"].append(train_idx)
        splits["val"].append(val_idx)
        splits["test"].append(test_idx)
        label_of[cls] = 1 if cls == class_pos else 0

    out = {}
    for split_name, idx_list in splits.items():
        idx = np.concatenate(idx_list)
        x = x_all[idx].reshape(len(idx), -1)
        y = np.array([label_of[c] for c in y_all[idx]], dtype=np.float64)
        out[f"X_{split_name}"] = x
        out[f"y_{split_name}"] = y
    return out
