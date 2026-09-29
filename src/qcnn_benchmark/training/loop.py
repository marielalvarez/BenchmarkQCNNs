"""Ciclo de entrenamiento binario compartido por todos los modelos QCNN/CNN
del benchmark: BCE (promediada sobre el lote, no sumada, para que el
recorte de norma de gradiente tenga una escala consistente e independiente
del tamaño de lote), Adam, recorte de norma global de gradiente, early
stopping por pérdida de validación, selección de checkpoint por menor
pérdida de validación.

Este protocolo es propio del benchmark y no viene de los repositorios de
referencia (Hur et al., por ejemplo, usa NesterovMomentumOptimizer sin
validación, early stopping ni recorte). Usar un solo ciclo para todos los
modelos garantiza que las diferencias medidas vengan de la arquitectura y
no del procedimiento de entrenamiento.
"""

import time

import autograd.numpy as anp
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

BATCH_SIZE = 25
N_UPDATES = 200
LEARNING_RATE = 0.01
BETA1, BETA2 = 0.9, 0.999
CLIP_NORM = 5.0
VAL_CHECK_EVERY = 10
PATIENCE_CHECKS = 5
MIN_DELTA = 1e-4


def bce_loss(predict_proba_fn, params, X_batch, y_batch):
    total = 0.0
    for x, y in zip(X_batch, y_batch):
        p = predict_proba_fn(params, x)
        total = total + (y * anp.log(p) + (1 - y) * anp.log(1 - p))
    return -total / len(y_batch)


def clip_grad_global_norm(grad, max_norm=CLIP_NORM):
    grad = pnp.array(grad)
    norm = float(pnp.linalg.norm(grad))
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


def train_binary_classifier(
    predict_proba_fn,
    n_params,
    rep,
    init_fn,
    run_seed=0,
    batch_size=BATCH_SIZE,
    n_updates=N_UPDATES,
    learning_rate=LEARNING_RATE,
    beta1=BETA1,
    beta2=BETA2,
    clip_norm=CLIP_NORM,
    val_check_every=VAL_CHECK_EVERY,
    patience_checks=PATIENCE_CHECKS,
    min_delta=MIN_DELTA,
    verbose=True,
    tag="",
):
    """Entrena un clasificador binario contra `predict_proba_fn(params, x)`.

    `rep` es un dict con X_train, y_train, X_val y y_val en el formato que
    produce qcnn_benchmark.representations (X_test y y_test se ignoran
    aquí). `init_fn(rng, n_params)` inicializa el
    vector de parámetros (ver qcnn_benchmark.training.init).
    """
    init_rng = np.random.default_rng(run_seed)
    batch_rng = np.random.default_rng(run_seed + 1_000_000)

    params = init_fn(init_rng, n_params)
    opt = qml.AdamOptimizer(stepsize=learning_rate, beta1=beta1, beta2=beta2)

    def loss_fn(p, X_batch, y_batch):
        return bce_loss(predict_proba_fn, p, X_batch, y_batch)

    grad_fn = qml.grad(loss_fn, argnums=0)

    X_train, y_train = rep["X_train"], rep["y_train"]
    X_val, y_val = rep["X_val"], rep["y_val"]

    train_loss_history = []
    val_loss_history = []

    best_val_loss = np.inf
    best_params = params
    checks_without_improvement = 0
    stopped_early_at = None

    t0 = time.time()
    for update in range(1, n_updates + 1):
        batch_idx = batch_rng.integers(0, len(X_train), size=batch_size)
        X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

        grad = grad_fn(params, X_batch, y_batch)
        grad = clip_grad_global_norm(grad, clip_norm)
        params = opt.apply_grad(grad, params)
        # qml.AdamOptimizer.apply_grad trata `params` como una tupla de
        # argumentos posicionales y, con un arreglo plano, devuelve una
        # lista de escalares. Los momentos de Adam siguen siendo correctos
        # elemento a elemento, pero una lista rompe a los modelos que hacen
        # slicing o reshape de params (como las CNN análogas). Se vuelve a
        # convertir en arreglo sin tocar el estado del optimizador.
        params = pnp.array(params, requires_grad=True)

        train_loss = float(loss_fn(params, X_batch, y_batch))
        train_loss_history.append(train_loss)

        if update % val_check_every == 0:
            val_loss = float(loss_fn(params, X_val, y_val))
            val_loss_history.append((update, val_loss))

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_params = params.copy()
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"[{tag}] update {update:3d}/{n_updates}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  best_val={best_val_loss:.4f}  "
                    f"no_improve={checks_without_improvement}  t={elapsed:.0f}s"
                )

            if checks_without_improvement >= patience_checks:
                stopped_early_at = update
                if verbose:
                    print(
                        f"[{tag}] Early stopping en la actualización {update} "
                        f"(paciencia={patience_checks} chequeos, delta={min_delta})."
                    )
                break

    return {
        "params": best_params,
        "final_params": params,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "best_val_loss": best_val_loss,
        "stopped_early_at": stopped_early_at,
        "n_updates_run": len(train_loss_history),
    }
