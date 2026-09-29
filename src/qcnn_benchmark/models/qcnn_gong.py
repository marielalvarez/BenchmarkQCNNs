"""Adaptador de Gong, Pei, Zhang & Zhou (2024), "Quantum convolutional
neural network based on variational quantum circuits" (Optics
Communications 550, 129993).

Circuito de 8 qubits, sin capa fully-connected (el paper la omite
explícitamente para clasificación binaria, Sec. 3.3): tres capas
convolución+pooling que reducen 8 -> 4 -> 2 -> 1 qubits, codificación
tree-structured hybrid amplitude (la aportación central del paper).

Codificación tree-structured hybrid amplitude (Sec. 3.2, Fig. 2)
------------------------------------------------------------------
Los N=8 qubits se dividen en m bloques independientes de n qubits
(`BLOCK_SIZE`). Dentro de cada bloque, el k-ésimo qubit (k=0..n-1) recibe
2**k rotaciones Ry controladas por los k qubits anteriores del bloque, una
por cada patrón de control posible. Es la generalización en árbol de una
rotación uniformemente controlada (Möttönen et al.), pero el ángulo es el
propio valor de la feature escalado a [0, π] y no se deriva de una
amplitud objetivo. Cada bloque de n qubits codifica 2**n - 1 features
(Ec. 3 del paper), así que la capacidad total es
`N_BLOCKS * (2**BLOCK_SIZE - 1)`.

El paper escala a [0, π] de forma global y no por bloque (párrafo tras la
Ec. 3). Aquí eso se obtiene con
`qcnn_benchmark.representations.pca.dense_pca_representation`, que ya
escala con el min y max de train. Cuando `n_components` es menor que la
capacidad, las features que faltan se rellenan con ceros, porque Ry(0) es
la identidad (por ejemplo PCA-8 con BLOCK_SIZE=2, que tiene capacidad 12).

En bloques de 4 o más qubits, qué bit de control corresponde a qué
feature es una convención arbitraria pero fija entre train, val y test. No
cambia la capacidad expresiva, solo en qué rama del árbol cae cada
componente de PCA.

Circuito convolucional: "Circuit 6" (Fig. 4f del paper)
--------------------------------------------------------
El paper reporta a Circuit 6 con exactamente 10 parámetros por
aplicación, gates Rx/Rz de un qubit e interacción de dos qubits
Rx-controlada. Eso corresponde al ansatz `unitary.U_6` del repositorio de
Hur et al. (Apache-2.0, https://github.com/takh04/QCNN), el mismo que
`qcnn_hur.py` usa como "Ansatz 8" (10 parámetros, Rx/Rz + CRX + Rx/Rz).
Ambos papers toman su numeración de la familia de circuitos de Sim,
Johnson y Aspuru-Guzik (2019). Se reutiliza U_6 sin cambios en lugar de
transcribirlo desde la figura del paper, para no arriesgar un error en un
circuito que ya está validado numéricamente.

Pooling (Fig. 5 del paper)
---------------------------
CNOT(descartado -> conservado) seguido de Rz y Rx sobre el qubit
conservado (2 parámetros). El qubit descartado no vuelve a usarse, lo cual
es correcto en un simulador de estado completo sin traza parcial
explícita, igual que en el pooling de `qcnn_wei.py`.

Topología de cableado (Fig. 3): pares adyacentes que comparten parámetros
dentro de cada capa (invariancia traslacional, Sec. 3.1). La capa 1 aplica
conv+pool 4 veces sobre (0,1)(2,3)(4,5)(6,7), la capa 2 lo aplica 2 veces
sobre (0,2)(4,6) y la capa 3 una vez sobre (0,4), terminando en el qubit 0.

Total de parámetros: 3*10 (conv) + 3*2 (pool) = 36. Por usar U_6 coincide
con el Ansatz 8 de Hur et al. (Tabla I: 36) y con el Circuit 6 de Gong et
al. (Sec. 4.1: "Circuit 6, featuring 10 parameters").
"""

import pathlib
import sys

import autograd.numpy as anp
import pennylane as qml

_MODULE_DIR = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _MODULE_DIR
while not (PROJECT_ROOT / "external").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
HUR_QCNN_DIR = PROJECT_ROOT / "external" / "hur_qcnn" / "QCNN"
assert HUR_QCNN_DIR.exists(), f"No se encontró {HUR_QCNN_DIR}"

if str(HUR_QCNN_DIR) not in sys.path:
    sys.path.insert(0, str(HUR_QCNN_DIR))

import unitary as _hur_unitary  # noqa: E402  (vendorizado, Apache-2.0, takh04/QCNN)

assert hasattr(_hur_unitary, "U_6")

N_QUBITS = 8
BLOCK_SIZE = 2  # "our encoding n=2" en la Tabla 1 del paper
N_BLOCKS = N_QUBITS // BLOCK_SIZE
ENCODING_CAPACITY = N_BLOCKS * (2**BLOCK_SIZE - 1)  # 12 features (m=4, 2**2-1=3)

CONV_U_PARAMS = 10  # unitary.U_6 == "Circuit 6" de Gong et al.
POOL_PARAMS = 2
TOTAL_PARAMS = 3 * CONV_U_PARAMS + 3 * POOL_PARAMS  # 36

_CONV_PAIRS = [
    [(0, 1), (2, 3), (4, 5), (6, 7)],  # capa 1: 8 qubits activos
    [(0, 2), (4, 6)],  # capa 2: 4 qubits activos (0, 2, 4, 6)
    [(0, 4)],  # capa 3: 2 qubits activos (0, 4)
]
_POOL_PAIRS = [
    [(1, 0), (3, 2), (5, 4), (7, 6)],  # (descartado, conservado)
    [(2, 0), (6, 4)],
    [(4, 0)],
]

_dev = qml.device("default.qubit", wires=N_QUBITS)


def _tree_block_encoding(x_block, wires):
    """Codificación tree-structured amplitude (Fig. 2) dentro de un único
    bloque de `len(wires)` qubits: 2**k rotaciones Ry controladas para el
    k-ésimo qubit del bloque, consumiendo 2**len(wires) - 1 features."""
    n = len(wires)
    idx = 0
    for level in range(n):
        n_branches = 2**level
        for branch in range(n_branches):
            angle = x_block[idx]
            idx += 1
            if level == 0:
                qml.RY(angle, wires=wires[0])
            else:
                control_values = [(branch >> b) & 1 for b in reversed(range(level))]
                qml.ctrl(qml.RY, control=wires[:level], control_values=control_values)(angle, wires=wires[level])


def _tree_hybrid_encoding(x_padded):
    for b in range(N_BLOCKS):
        wires = list(range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE))
        slot = 2**BLOCK_SIZE - 1
        x_block = x_padded[b * slot : (b + 1) * slot]
        _tree_block_encoding(x_block, wires)


def _pooling_gong(params, wires):
    """Fig. 5 del paper: CNOT(descartado -> conservado), luego Rz, Rx sobre
    el qubit conservado (2 parámetros)."""
    discard, keep = wires
    qml.CNOT(wires=[discard, keep])
    qml.RZ(params[0], wires=keep)
    qml.RX(params[1], wires=keep)


@qml.qnode(_dev)
def _circuit(x, params):
    n_missing = ENCODING_CAPACITY - x.shape[0]
    assert n_missing >= 0, (
        f"x tiene {x.shape[0]} features, más que la capacidad del encoding " f"({ENCODING_CAPACITY})"
    )
    x_padded = anp.concatenate([x, anp.zeros(n_missing)]) if n_missing > 0 else x
    _tree_hybrid_encoding(x_padded)

    p1, p2, p3 = params[0:10], params[10:20], params[20:30]
    p4, p5, p6 = params[30:32], params[32:34], params[34:36]

    for pair in _CONV_PAIRS[0]:
        _hur_unitary.U_6(p1, wires=list(pair))
    for pair in _POOL_PAIRS[0]:
        _pooling_gong(p4, wires=pair)

    for pair in _CONV_PAIRS[1]:
        _hur_unitary.U_6(p2, wires=list(pair))
    for pair in _POOL_PAIRS[1]:
        _pooling_gong(p5, wires=pair)

    for pair in _CONV_PAIRS[2]:
        _hur_unitary.U_6(p3, wires=list(pair))
    for pair in _POOL_PAIRS[2]:
        _pooling_gong(p6, wires=pair)

    return qml.probs(wires=0)


def forward_probs(x, params):
    """qml.probs(wires=0) tras las 3 capas de convolución+pooling."""
    return _circuit(x, params)


def predict_proba(params, x):
    """Probabilidad de clase positiva (índice 1). Gong et al. no publican
    código, así que se sigue la misma convención que los demás adaptadores."""
    return forward_probs(x, params)[1]
