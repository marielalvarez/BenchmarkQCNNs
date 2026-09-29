"""Pruebas de funcionamiento, no de fidelidad, del motor de shots finitos y
de la transpilación. Verifican que cada modelo produce una probabilidad
válida bajo shots, que el gradiente fluye y que transpilar a un mapa de
acoplamiento no cambia el resultado analítico."""

import numpy as np
import pennylane as qml
import pytest

from qcnn_benchmark.execution import (
    COUPLING_MAP_16Q,
    get_circuit_spec,
    grid_coupling_map,
    make_shots_predict_proba,
    transpile_qnode,
)
from qcnn_benchmark.execution.shots import n_trainable_params
from qcnn_benchmark.models import qcnn_cong, qcnn_gong, qcnn_hur, qcnn_wei


def test_grid_coupling_map_16q_has_16_nodes_worth_of_edges():
    # rejilla 4x4: 2*4*3 = 24 aristas (horizontales + verticales)
    assert len(COUPLING_MAP_16Q) == 24
    nodes = {n for edge in COUPLING_MAP_16Q for n in edge}
    assert nodes == set(range(16))


@pytest.mark.parametrize(
    "model_name,module,x_dim",
    [("hur", qcnn_hur, 16), ("gong", qcnn_gong, 8), ("cong", qcnn_cong, 8)],
)
def test_shots_predict_proba_in_unit_interval(model_name, module, x_dim):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, x_dim)
    params = rng.uniform(-np.pi, np.pi, module.TOTAL_PARAMS)
    pp = make_shots_predict_proba(model_name, n_shots=256)
    p = float(pp(params, x))
    assert 0.0 <= p <= 1.0


def test_wei_shots_predict_proba_in_unit_interval():
    """Bajo shots Wei solo entrena `h` (37 parámetros y no 46, porque `beta`
    queda fijo). Ver execution.shots.make_wei_predict_proba."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    x = x / np.linalg.norm(x)
    n_params = n_trainable_params("wei")
    assert n_params == qcnn_wei.N_HAMILTONIAN_PARAMS == 37
    params = rng.normal(0, 0.1, n_params)
    pp = make_shots_predict_proba("wei", n_shots=256)
    p = float(pp(params, x))
    assert 0.0 <= p <= 1.0


def test_n_trainable_params_matches_module_total_params_except_wei():
    assert n_trainable_params("hur") == qcnn_hur.TOTAL_PARAMS
    assert n_trainable_params("gong") == qcnn_gong.TOTAL_PARAMS
    assert n_trainable_params("cong") == qcnn_cong.TOTAL_PARAMS
    assert n_trainable_params("wei") != qcnn_wei.TOTAL_PARAMS


def test_gong_shots_gradient_flows():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)
    params = qml.numpy.array(rng.uniform(-np.pi, np.pi, qcnn_gong.TOTAL_PARAMS), requires_grad=True)
    pp = make_shots_predict_proba("gong", n_shots=1024)

    grad = qml.grad(lambda p: pp(p, x))(params)
    assert np.linalg.norm(np.array(grad)) > 0.0


def test_wei_shots_gradient_flows():
    """Si `beta` fuera entrenable, parameter-shift perturbaría el vector de
    StatePrep y esto fallaría con `ValueError: state_vector has to be of
    norm 1.0`."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    x = x / np.linalg.norm(x)
    n_params = n_trainable_params("wei")
    params = qml.numpy.array(rng.normal(0, 0.1, n_params), requires_grad=True)
    pp = make_shots_predict_proba("wei", n_shots=256)

    grad = qml.grad(lambda p: pp(p, x))(params)
    assert np.linalg.norm(np.array(grad)) > 0.0


def test_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        get_circuit_spec("not_a_model")


def test_transpile_preserves_analytic_result_gong():
    """Los SWAPs que agrega la transpilación cambian cómo se ejecuta el
    circuito pero no lo que calcula."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)
    params = rng.uniform(-np.pi, np.pi, qcnn_gong.TOTAL_PARAMS)

    original = qcnn_gong._circuit(x, params)
    transpiled = transpile_qnode(qcnn_gong._circuit, coupling_map=grid_coupling_map(2, 4))(x, params)
    np.testing.assert_allclose(np.array(original), np.array(transpiled), atol=1e-8)


def test_transpile_preserves_analytic_result_cong():
    """Cong usa AngleEmbedding, un template de 8 wires. Sin expandirlo antes
    de qml.transforms.transpile, esto falla con NotImplementedError (ver
    execution/transpile.py). Gong no sirve para detectarlo porque solo usa
    puertas elementales."""
    spec = get_circuit_spec("cong")
    dev = qml.device("default.qubit", wires=spec.n_wires)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)
    params = rng.uniform(-np.pi, np.pi, qcnn_cong.TOTAL_PARAMS)

    original_qnode = qml.QNode(spec.get_qnode().func, dev)
    transpiled_qnode = transpile_qnode(qml.QNode(spec.get_qnode().func, dev), coupling_map=grid_coupling_map(2, 4))

    original = spec.call(original_qnode, params, x)
    transpiled = spec.call(transpiled_qnode, params, x)
    np.testing.assert_allclose(np.array(original), np.array(transpiled), atol=1e-8)


def test_transpile_rejects_wei_hamiltonian_readout():
    """qml.transforms.transpile no admite medir Hamiltonianos. Este test
    avisa si una actualización de PennyLane cambia ese comportamiento y deja
    desactualizada la explicación de execution/transpile.py."""
    transpiled = transpile_qnode(qcnn_wei._readout_circuit, coupling_map=grid_coupling_map(2, 5))
    g = np.zeros(1024)
    g[0] = 1.0
    h = np.zeros(37)
    with pytest.raises(NotImplementedError):
        transpiled(g, h)
