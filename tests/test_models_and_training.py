"""Pruebas de funcionamiento, no de fidelidad. Verifican que el forward pass
y el ciclo de entrenamiento corren de punta a punta para cada modelo y que
el gradiente fluye. La comparación contra los resultados publicados está
en notebooks/00_reproduce_*.ipynb, porque entrenar circuitos simulados
cientos de pasos sería demasiado lento para una suite de pruebas.
"""

import numpy as np
import pennylane as qml

from qcnn_benchmark.metrics import batch_accuracy
from qcnn_benchmark.models import (
    cnn_gong_analoga,
    cnn_hur_analoga,
    cnn_wei_analoga,
    qcnn_cong,
    qcnn_gong,
    qcnn_hur,
    qcnn_wei,
)
from qcnn_benchmark.training import normal_init, train_binary_classifier, uniform_pi_init
from qcnn_benchmark.training.loop import bce_loss


def test_hur_param_count_matches_paper_table_i():
    assert qcnn_hur.TOTAL_PARAMS == 36


def test_wei_param_count_matches_paper_table_i():
    assert qcnn_wei.TOTAL_PARAMS == 46


def test_hur_forward_probs_sums_to_one():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 16)
    params = rng.uniform(-np.pi, np.pi, qcnn_hur.TOTAL_PARAMS)
    probs = qcnn_hur.forward_probs(x, params)
    assert len(probs) == 2
    np.testing.assert_allclose(float(sum(probs)), 1.0, atol=1e-6)


def test_wei_predict_proba_in_unit_interval():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    x = x / np.linalg.norm(x)
    params = rng.normal(0, 0.1, qcnn_wei.TOTAL_PARAMS)
    p = float(qcnn_wei.predict_proba(params, x))
    assert 0.0 <= p <= 1.0


def test_hur_training_loop_runs_and_loss_changes():
    rng = np.random.default_rng(0)
    n = 24
    X = rng.uniform(0, np.pi, (n, 16))
    y = rng.integers(0, 2, n).astype(np.float64)
    rep = {"X_train": X[:16], "y_train": y[:16], "X_val": X[16:], "y_val": y[16:]}

    result = train_binary_classifier(
        qcnn_hur.predict_proba, qcnn_hur.TOTAL_PARAMS, rep, uniform_pi_init,
        run_seed=0, batch_size=6, n_updates=4, val_check_every=2,
        patience_checks=10, verbose=False, tag="test-hur",
    )
    assert result["n_updates_run"] == 4
    assert result["train_loss_history"][0] != result["train_loss_history"][-1]
    acc = batch_accuracy(qcnn_hur.predict_proba, result["params"], rep["X_val"], rep["y_val"])
    assert 0.0 <= acc <= 1.0


def test_wei_training_loop_runs_and_loss_changes():
    rng = np.random.default_rng(0)
    n = 18
    X = rng.normal(size=(n, 1024))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.integers(0, 2, n).astype(np.float64)
    rep = {"X_train": X[:12], "y_train": y[:12], "X_val": X[12:], "y_val": y[12:]}

    result = train_binary_classifier(
        qcnn_wei.predict_proba, qcnn_wei.TOTAL_PARAMS, rep, normal_init,
        run_seed=0, batch_size=4, n_updates=4, val_check_every=2,
        patience_checks=10, verbose=False, tag="test-wei",
    )
    assert result["n_updates_run"] == 4
    assert result["train_loss_history"][0] != result["train_loss_history"][-1]
    acc = batch_accuracy(qcnn_wei.predict_proba, result["params"], rep["X_val"], rep["y_val"])
    assert 0.0 <= acc <= 1.0


def test_gong_param_count_matches_circuit6():
    assert qcnn_gong.TOTAL_PARAMS == 36


def test_gong_encoding_capacity_matches_block_size():
    assert qcnn_gong.ENCODING_CAPACITY == qcnn_gong.N_BLOCKS * (2**qcnn_gong.BLOCK_SIZE - 1)


def test_gong_forward_probs_sums_to_one_with_padded_pca8():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)  # PCA-8 < ENCODING_CAPACITY (12): se rellena con ceros
    params = rng.uniform(-np.pi, np.pi, qcnn_gong.TOTAL_PARAMS)
    probs = qcnn_gong.forward_probs(x, params)
    assert len(probs) == 2
    np.testing.assert_allclose(float(sum(probs)), 1.0, atol=1e-6)


def test_gong_training_loop_runs_and_loss_changes():
    rng = np.random.default_rng(0)
    n = 24
    X = rng.uniform(0, np.pi, (n, 8))
    y = rng.integers(0, 2, n).astype(np.float64)
    rep = {"X_train": X[:16], "y_train": y[:16], "X_val": X[16:], "y_val": y[16:]}

    result = train_binary_classifier(
        qcnn_gong.predict_proba, qcnn_gong.TOTAL_PARAMS, rep, uniform_pi_init,
        run_seed=0, batch_size=6, n_updates=4, val_check_every=2,
        patience_checks=10, verbose=False, tag="test-gong",
    )
    assert result["n_updates_run"] == 4
    assert result["train_loss_history"][0] != result["train_loss_history"][-1]
    acc = batch_accuracy(qcnn_gong.predict_proba, result["params"], rep["X_val"], rep["y_val"])
    assert 0.0 <= acc <= 1.0


def test_cong_param_count_matches_generic_su4_architecture():
    assert qcnn_cong.TOTAL_PARAMS == 51


def test_cong_forward_probs_sums_to_one():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)
    params = rng.uniform(-np.pi, np.pi, qcnn_cong.TOTAL_PARAMS)
    probs = qcnn_cong.forward_probs(x, params)
    assert len(probs) == 2
    np.testing.assert_allclose(float(sum(probs)), 1.0, atol=1e-6)


def test_cong_training_loop_runs_and_loss_changes():
    rng = np.random.default_rng(0)
    n = 24
    X = rng.uniform(0, np.pi, (n, 8))
    y = rng.integers(0, 2, n).astype(np.float64)
    rep = {"X_train": X[:16], "y_train": y[:16], "X_val": X[16:], "y_val": y[16:]}

    result = train_binary_classifier(
        qcnn_cong.predict_proba, qcnn_cong.TOTAL_PARAMS, rep, uniform_pi_init,
        run_seed=0, batch_size=6, n_updates=4, val_check_every=2,
        patience_checks=10, verbose=False, tag="test-cong",
    )
    assert result["n_updates_run"] == 4
    assert result["train_loss_history"][0] != result["train_loss_history"][-1]
    acc = batch_accuracy(qcnn_cong.predict_proba, result["params"], rep["X_val"], rep["y_val"])
    assert 0.0 <= acc <= 1.0


def test_cnn_analogas_param_counts_are_close_to_their_qcnn():
    assert cnn_hur_analoga.TOTAL_PARAMS == 35  # qcnn_hur tiene 36
    assert cnn_wei_analoga.TOTAL_PARAMS == 46  # qcnn_wei tiene 46
    assert cnn_gong_analoga.TOTAL_PARAMS == 35  # qcnn_gong tiene 36


def test_cnn_analogas_predict_proba_in_unit_interval():
    rng = np.random.default_rng(0)
    for module in (cnn_hur_analoga, cnn_wei_analoga, cnn_gong_analoga):
        x = rng.uniform(0, np.pi, module.INPUT_SIZE)
        params = module.init(rng, module.TOTAL_PARAMS)
        p = float(module.predict_proba(params, x))
        assert 0.0 <= p <= 1.0


def test_cnn_analogas_training_loop_runs_and_loss_changes():
    for module, tag in (
        (cnn_hur_analoga, "test-cnn-hur"),
        (cnn_wei_analoga, "test-cnn-wei"),
        (cnn_gong_analoga, "test-cnn-gong"),
    ):
        rng = np.random.default_rng(0)
        n = 24
        X = rng.uniform(0, np.pi, (n, module.INPUT_SIZE))
        y = rng.integers(0, 2, n).astype(np.float64)
        rep = {"X_train": X[:16], "y_train": y[:16], "X_val": X[16:], "y_val": y[16:]}

        result = train_binary_classifier(
            module.predict_proba, module.TOTAL_PARAMS, rep, module.init,
            run_seed=0, batch_size=6, n_updates=4, val_check_every=2,
            patience_checks=10, verbose=False, tag=tag,
        )
        assert result["n_updates_run"] == 4
        assert result["train_loss_history"][0] != result["train_loss_history"][-1]
        acc = batch_accuracy(module.predict_proba, result["params"], rep["X_val"], rep["y_val"])
        assert 0.0 <= acc <= 1.0


def test_cnn_analogas_kaiming_init_avoids_systematic_dead_relu():
    """Con `normal_init` la segunda Conv1d solo recibe activaciones no
    negativas y un sesgo de N(0, 0.1) puede apagar un canal completo en
    todas las posiciones, dejando el gradiente en cero salvo en el sesgo
    final. Esto motivó `_cnn_common.make_kaiming_init`.

    Solo se comprueba para la semilla 0, la misma que usan las demás
    pruebas. Con 2 o 3 canales algunas semillas pueden seguir muriendo por
    azar, y por eso el protocolo experimental promedia sobre 5 semillas."""
    for module in (cnn_hur_analoga, cnn_wei_analoga, cnn_gong_analoga):
        rng = np.random.default_rng(0)
        params = module.init(rng, module.TOTAL_PARAMS)
        x = rng.uniform(0, np.pi, module.INPUT_SIZE)
        y = np.array([1.0])

        def loss_fn(p, xb=x, yb=y):
            return bce_loss(module.predict_proba, p, xb.reshape(1, -1), yb)

        grad = np.array(qml.grad(loss_fn)(params))
        nonzero_frac = np.mean(np.abs(grad) > 1e-12)
        # Lo normal es cerca de 0.5, porque ReLU apaga más o menos la mitad
        # de las unidades. El colapso deja vivo solo el sesgo final, es
        # decir entre 0.02 y 0.03 de los parámetros.
        assert nonzero_frac > 0.2, f"{module.__name__}: gradiente nulo en >80% de los parámetros (ReLU muerta)"
