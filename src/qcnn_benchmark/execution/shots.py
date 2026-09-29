"""Motor de shots finitos. Para cada modelo QCNN construye una versión de su
`predict_proba` que estima las mediciones con un número finito de shots
(por ejemplo 256, 1024 o 4096) en vez del valor esperado analítico. Los
adaptadores de `models/qcnn_*.py` no se modifican y siguen definiendo el
régimen analítico de E0.

`qml.set_shots(qnode, shots=n)` devuelve un qnode nuevo que ejecuta el
mismo circuito estimando cada medición por muestreo, sin alterar el qnode
original. Con shots fijos PennyLane no puede usar
`diff_method="backprop"` y recurre a la regla de desplazamiento de
parámetro, que es como se calcularía el gradiente en hardware real. Por
eso el ruido de muestreo afecta también al gradiente durante el
entrenamiento y no solo a la exactitud final.
`qcnn_benchmark.training.train_binary_classifier` se usa sin cambios, solo
recibe un `predict_proba_fn` distinto.

Este módulo modela únicamente el ruido de muestreo. El ruido de hardware
(relajación, despolarización, error de lectura) está en
`qcnn_benchmark.noise`, que se combina con este módulo en E2.
"""

import numpy as np
import autograd.numpy as anp
import pennylane as qml

from .registry import get_circuit_spec

# Semilla con la que se fija `beta` (filtro LCU) de Wei bajo shots o ruido,
# ver `make_wei_predict_proba`. Es una constante compartida para que
# noise.make_wei_noisy_predict_proba use exactamente el mismo beta. Con
# semillas distintas, un mismo vector `h` sería un modelo distinto en E1 y E2.
WEI_FROZEN_BETA_SEED = 0


def make_generic_predict_proba(model_name, n_shots):
    """`predict_proba(params, x)` de un modelo del registro genérico (hur,
    cong o gong, ver `execution.registry.CIRCUIT_SPECS`) con `n_shots`."""
    spec = get_circuit_spec(model_name)
    shots_qnode = qml.set_shots(spec.get_qnode(), shots=n_shots)

    def predict_proba(params, x):
        return spec.extract(spec.call(shots_qnode, params, x))

    return predict_proba


def make_wei_predict_proba(n_shots):
    """QCNN de Wei et al. con `n_shots`. Bajo shots solo se entrena el
    Hamiltoniano de lectura (`h`, 37 parámetros). El filtro LCU (`beta`, 9
    parámetros) queda fijo en un valor aleatorio generado con
    `WEI_FROZEN_BETA_SEED`.

    La razón es una limitación de PennyLane 0.45. La lectura de Wei usa
    `qml.StatePrep(g, wires)` para cargar `g = conv_lcu_state(x, beta)`, que
    depende de `beta`. Con shots hace falta parameter-shift o SPSA, y ambos
    perturban el vector de entrada de `StatePrep`. Ese vector llega sin
    normalizar a `MottonenStatePreparation`, que exige norma 1 y no permite
    relajar esa validación, así que el circuito falla con
    `ValueError: state_vector has to be of norm 1.0` (o produce NaN). Se
    probó con ambos métodos y no hay combinación de opciones que lo evite.

    Al fijar `beta` antes de llamar al qnode, `StatePrep` ya no recibe un
    argumento diferenciable y E1 y E2 funcionan para Wei, a cambio de
    entrenar 37 parámetros en vez de 46. Los notebooks E1_finite_shots y
    E2_nisq_noise también lo indican."""
    from qcnn_benchmark.models import qcnn_wei

    frozen_beta = np.random.default_rng(WEI_FROZEN_BETA_SEED).normal(0, 0.1, qcnn_wei.N_FILTER_PARAMS)
    shots_qnode = qml.set_shots(qcnn_wei._readout_circuit, shots=n_shots)

    def predict_proba(h, x):
        g = qcnn_wei.conv_lcu_state(x, frozen_beta)
        logit = shots_qnode(g, h)
        return 1.0 / (1.0 + anp.exp(-logit))

    return predict_proba


PREDICT_PROBA_FACTORIES = {
    "hur": lambda n_shots: make_generic_predict_proba("hur", n_shots),
    "cong": lambda n_shots: make_generic_predict_proba("cong", n_shots),
    "gong": lambda n_shots: make_generic_predict_proba("gong", n_shots),
    "wei": make_wei_predict_proba,
}


def n_trainable_params(model_name):
    """Número de parámetros que espera el `predict_proba` devuelto por
    `make_shots_predict_proba(model_name, ...)`. Coincide con
    `TOTAL_PARAMS` del modelo salvo en "wei", donde solo se entrenan los 37
    del Hamiltoniano (ver `make_wei_predict_proba`)."""
    if model_name == "wei":
        from qcnn_benchmark.models import qcnn_wei

        return qcnn_wei.N_HAMILTONIAN_PARAMS
    from qcnn_benchmark.models import qcnn_cong, qcnn_gong, qcnn_hur

    return {"hur": qcnn_hur.TOTAL_PARAMS, "cong": qcnn_cong.TOTAL_PARAMS, "gong": qcnn_gong.TOTAL_PARAMS}[model_name]


def make_shots_predict_proba(model_name, n_shots):
    """`predict_proba(params, x)` del modelo `model_name` con `n_shots`. Tiene
    la misma firma que `qcnn_benchmark.models.qcnn_*.predict_proba` para
    pasarse directo a `train_binary_classifier`. Como `n_params` debe usarse
    `n_trainable_params(model_name)` y no `TOTAL_PARAMS`, por el caso de
    "wei"."""
    return PREDICT_PROBA_FACTORIES[model_name](n_shots)
