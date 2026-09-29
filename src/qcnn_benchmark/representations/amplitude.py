"""Codificación de amplitud (Wei et al.): cero-relleno 28x28 -> 32x32,
aplanado a 1024, normalizado en norma L2. Transformación fija, sin ajuste,
por lo que no hay riesgo de fuga de datos entre splits."""

import numpy as np

from qcnn_benchmark.data.sampling import (
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
    N_VAL_PER_CLASS,
    SAMPLE_SEED,
    stratified_split,
)

IMG_SIZE = 28
PAD_SIZE = 32  # 32*32 = 1024 = 2**10, exactamente 10 qubits de registro de trabajo


def amplitude_representation(raw_splits, img_size=IMG_SIZE, pad_size=PAD_SIZE):
    pad = (pad_size - img_size) // 2
    out = {}
    for split in ("train", "val", "test"):
        x = raw_splits[f"X_{split}"]
        n = x.shape[0]
        imgs = x.reshape(n, img_size, img_size).astype(np.float64)
        padded = np.zeros((n, pad_size, pad_size), dtype=np.float64)
        padded[:, pad : pad + img_size, pad : pad + img_size] = imgs
        flat = padded.reshape(n, pad_size * pad_size)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out[f"X_{split}"] = flat / norms
        out[f"y_{split}"] = raw_splits[f"y_{split}"]
    return out


def build_amplitude_dataset(
    x_all,
    y_all,
    class_pos,
    class_neg,
    img_size=IMG_SIZE,
    pad_size=PAD_SIZE,
    n_train=N_TRAIN_PER_CLASS,
    n_val=N_VAL_PER_CLASS,
    n_test=N_TEST_PER_CLASS,
    seed=SAMPLE_SEED,
    verbose=True,
):
    raw = stratified_split(x_all, y_all, class_pos, class_neg, n_train, n_val, n_test, seed)
    rep = amplitude_representation(raw, img_size=img_size, pad_size=pad_size)
    for split in ("train", "val", "test"):
        assert rep[f"X_{split}"].shape[1] == pad_size * pad_size
        norms = np.linalg.norm(rep[f"X_{split}"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), f"normas fuera de rango en {split}"
        if verbose:
            print(f"  {split}: {rep[f'X_{split}'].shape}, norma L2 promedio = {norms.mean():.6f}")
    return rep
