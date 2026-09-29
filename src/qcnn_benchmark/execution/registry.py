"""Registro de circuitos por modelo QCNN. Para cada modelo indica cómo
obtener su qnode original (cuya función sin decorar está en `.func`) y cómo
llamarlo y leer su resultado. `execution.shots` y `noise` lo usan para
reconstruir el circuito sobre otro device (con shots, con ruido o
transpilado) sin duplicar esa lógica ni tocar `models/qcnn_*.py`.

Wei et al. queda fuera del registro porque su parte cuántica es solo la
lectura con el Hamiltoniano de 37 parámetros, mientras que el filtro LCU se
calcula de forma clásica fuera del qnode. Se maneja en
`execution.shots.make_wei_predict_proba` y
`noise.make_wei_noisy_predict_proba`.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CircuitSpec:
    """`get_qnode()` devuelve el qnode analítico original del modelo, del que
    se leen `.func` y `.device`. `call(qnode, params, x)` invoca ese qnode, o
    uno reconstruido sobre otro device con el mismo `.func`, pasando los
    kwargs propios del modelo. `extract(resultado_crudo)` obtiene la
    probabilidad de clase positiva a partir de lo que devuelve el qnode."""

    get_qnode: Callable[[], object]
    n_wires: int
    call: Callable[[object, object, object], object]
    extract: Callable[[object], float]


def _hur_spec():
    from qcnn_benchmark.models import qcnn_hur

    return CircuitSpec(
        get_qnode=lambda: qcnn_hur._hur_circuit.QCNN,
        n_wires=8,
        call=lambda qnode, params, x: qnode(
            x, params, U="U_6", U_params=qcnn_hur.U_PARAMS, embedding_type="Angle-compact", cost_fn="cross_entropy"
        ),
        extract=lambda probs: probs[1],
    )


def _cong_spec():
    from qcnn_benchmark.models import qcnn_cong

    return CircuitSpec(
        get_qnode=lambda: qcnn_cong._hur_circuit.QCNN,
        n_wires=8,
        call=lambda qnode, params, x: qnode(
            x, params, U="U_SU4", U_params=qcnn_cong.U_PARAMS, embedding_type="Angle", cost_fn="cross_entropy"
        ),
        extract=lambda probs: probs[1],
    )


def _gong_spec():
    from qcnn_benchmark.models import qcnn_gong

    return CircuitSpec(
        get_qnode=lambda: qcnn_gong._circuit,
        n_wires=qcnn_gong.N_QUBITS,
        call=lambda qnode, params, x: qnode(x, params),
        extract=lambda probs: probs[1],
    )


CIRCUIT_SPECS = {
    "hur": _hur_spec,
    "cong": _cong_spec,
    "gong": _gong_spec,
}


def get_circuit_spec(model_name):
    """`CircuitSpec` del modelo `model_name` ("hur", "cong" o "gong"). Wei
    se maneja aparte, ver el docstring del módulo."""
    if model_name not in CIRCUIT_SPECS:
        raise KeyError(
            f"'{model_name}' no está en el registro de circuitos genérico "
            f"({sorted(CIRCUIT_SPECS)}). Wei et al. se maneja por separado "
            "en execution.shots.make_wei_predict_proba y "
            "noise.make_wei_noisy_predict_proba."
        )
    return CIRCUIT_SPECS[model_name]()
