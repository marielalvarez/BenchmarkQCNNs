"""Canales de ruido (despolarizante, error de lectura, relajación térmica) y
las 7 condiciones de E2 (sin ruido, 3 aisladas y 3 compuestas), construidos
sobre `qml.NoiseModel` y `qml.noise.add_noise` (ver `channels.py`). Solo
aplica a las QCNN, porque las CNN análogas no tienen circuito.

`make_noisy_predict_proba` es el punto de entrada de E2. Reconstruye el
circuito del modelo (hur, cong o gong, vía `execution.registry`) sobre
`default.mixed`, ya que `default.qubit` no admite canales de ruido. Luego
aplica, en este orden, la transpilación opcional, la condición de ruido y
los shots. El orden importa porque el ruido debe afectar también a los
SWAPs que agrega la transpilación, y los shots se miden sobre el circuito
ya ruidoso.

Wei et al. se maneja en `make_wei_noisy_predict_proba`. Su lectura es un
`qml.expval(qml.Hamiltonian(...))` que `qml.transforms.transpile` no
admite, así que en E2 recibe shots y ruido pero no transpilación. Igual
que en `execution.shots.make_wei_predict_proba`, solo se entrena el
Hamiltoniano (`h`, 37 parámetros) y el filtro LCU (`beta`) queda fijo.
"""

import pennylane as qml

from qcnn_benchmark.execution.registry import get_circuit_spec
from qcnn_benchmark.execution.shots import WEI_FROZEN_BETA_SEED, n_trainable_params
from qcnn_benchmark.execution.transpile import transpile_qnode

from .channels import (
    GATE_TIME_US,
    NOISE_CONDITIONS,
    SEVERITY_PARAMS,
    composite_noise_model,
    depolarizing_noise_model,
    get_noise_model,
    readout_noise_model,
    relaxation_noise_model,
)

__all__ = [
    "NOISE_CONDITIONS",
    "SEVERITY_PARAMS",
    "GATE_TIME_US",
    "depolarizing_noise_model",
    "readout_noise_model",
    "relaxation_noise_model",
    "composite_noise_model",
    "get_noise_model",
    "make_noisy_predict_proba",
    "make_wei_noisy_predict_proba",
    "n_trainable_params",
]


def make_noisy_predict_proba(model_name, n_shots, condition, coupling_map=None):
    """`predict_proba(params, x)` de hur, cong o gong con `n_shots`, la
    condición de ruido `condition` (una de `NOISE_CONDITIONS`) y, si se da
    `coupling_map`, transpilación a ese mapa."""
    spec = get_circuit_spec(model_name)
    mixed_dev = qml.device("default.mixed", wires=spec.n_wires)
    qnode = qml.QNode(spec.get_qnode().func, mixed_dev)

    if coupling_map is not None:
        qnode = transpile_qnode(qnode, coupling_map=coupling_map)

    noise_model = get_noise_model(condition)
    if noise_model is not None:
        qnode = qml.noise.add_noise(qnode, noise_model=noise_model)

    qnode = qml.set_shots(qnode, shots=n_shots)

    def predict_proba(params, x):
        return spec.extract(spec.call(qnode, params, x))

    return predict_proba


def make_wei_noisy_predict_proba(n_shots, condition):
    """Versión de `make_noisy_predict_proba` para Wei et al. No acepta
    `coupling_map` y solo entrena `h` (37 parámetros) con `beta` fijo. Ver
    el docstring del módulo y `execution.shots.make_wei_predict_proba`."""
    import autograd.numpy as anp
    import numpy as np

    from qcnn_benchmark.models import qcnn_wei

    frozen_beta = np.random.default_rng(WEI_FROZEN_BETA_SEED).normal(0, 0.1, qcnn_wei.N_FILTER_PARAMS)
    mixed_dev = qml.device("default.mixed", wires=qcnn_wei.N_WORK_QUBITS)
    qnode = qml.QNode(qcnn_wei._readout_circuit.func, mixed_dev)

    noise_model = get_noise_model(condition)
    if noise_model is not None:
        qnode = qml.noise.add_noise(qnode, noise_model=noise_model)
    qnode = qml.set_shots(qnode, shots=n_shots)

    def predict_proba(h, x):
        g = qcnn_wei.conv_lcu_state(x, frozen_beta)
        logit = qnode(g, h)
        return 1.0 / (1.0 + anp.exp(-logit))

    return predict_proba
