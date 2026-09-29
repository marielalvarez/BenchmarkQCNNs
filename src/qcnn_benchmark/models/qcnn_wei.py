"""Adaptador del QCNN de Wei, Chen, Zhou & Long (2021), "A Quantum
Convolutional Neural Network on NISQ Devices" (arXiv:2104.06918v3).

La implementación de WeiNet en qml-benchmarks de Xanadu no es fiel a este
paper. Usa un filtro 3x3 fijo no entrenable, no modela el registro de
ancillas y su lectura no es el hamiltoniano de 37 parámetros de Wei et al.
Por eso el circuito se reconstruye directamente de las ecuaciones del paper
(Fig. 2, Ec. 10-16, Apéndice B). De qml-benchmarks solo se toma la idea de
realizar los desplazamientos E_1, E_2=I, E_3=E_1^† del Apéndice B como
desplazamientos cíclicos (numpy.roll), lo cual es consistente con la figura
del paper.

Arquitectura:
- 10 qubits de registro de trabajo con codificación de amplitud directa
  (imagen de 32x32 = 1024 amplitudes, ver
  qcnn_benchmark.representations.amplitude).
- Capa convolucional con 9 parámetros entrenables beta_1..beta_9, uno por
  cada desplazamiento relativo (delta_fila, delta_col) en {-1,0,1}^2,
  combinados como combinación lineal de unitarios (LCU). La post-selección
  de las 4 ancillas en |0000> se calcula con su equivalente matemático
  exacto en el caso sin ruido, la combinación lineal normalizada
  sum_k beta_k Q_k|f>. Esto evita simular el registro completo de 14 qubits
  con puertas controladas y post-selección real.
- Pooling que descarta los qubits de trabajo 4 y 9 (numeración 1-indexada
  del paper, que aquí son los wires 3 y 8), dejando 8 qubits activos.
- Lectura con un hamiltoniano de 37 parámetros (h_0 I + sum_i h_i Z_i +
  sum_{i<j} h_ij Z_i Z_j sobre los 8 qubits que sobreviven al pooling),
  seguida de una sigmoide.
"""

import warnings

import autograd.numpy as anp
import pennylane as qml

# default.qubit guarda el estado en complex128. El expval de operadores
# hermitianos siempre es real, pero autograd convierte de complejo a real en
# el backward a través de qml.StatePrep y emite este warning inofensivo.
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

N_WORK_QUBITS = 10
N_FILTER_PARAMS = 9
POOLED_LABELS_1INDEXED = {4, 9}  # "abandoning the 4-th and 9-th qubit of the work register" (paper)
POOLED_WIRES = {lbl - 1 for lbl in POOLED_LABELS_1INDEXED}  # 0-indexado: {3, 8}
SURVIVING_WIRES = [w for w in range(N_WORK_QUBITS) if w not in POOLED_WIRES]
assert len(SURVIVING_WIRES) == 8

# k=1..9 en orden fila-mayor sobre los desplazamientos relativos, igual que
# el orden de la matriz W en la Ec.(19) del paper (filas = desplazamiento de
# fila, columnas = desplazamiento de columna)
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
assert len(OFFSETS) == N_FILTER_PARAMS

# Debe coincidir con qcnn_benchmark.representations.amplitude.PAD_SIZE
PAD_SIZE = 32

# Hamiltoniano de 37 parametros: identidad + 8 Z + 28 ZZ = 1 + 8 + 28 = 37
HAMILTONIAN_OPS = [qml.Identity(0)]
for _w in SURVIVING_WIRES:
    HAMILTONIAN_OPS.append(qml.PauliZ(_w))
for _i in range(len(SURVIVING_WIRES)):
    for _j in range(_i + 1, len(SURVIVING_WIRES)):
        HAMILTONIAN_OPS.append(qml.PauliZ(SURVIVING_WIRES[_i]) @ qml.PauliZ(SURVIVING_WIRES[_j]))
N_HAMILTONIAN_PARAMS = len(HAMILTONIAN_OPS)
assert N_HAMILTONIAN_PARAMS == 37

TOTAL_PARAMS = N_FILTER_PARAMS + N_HAMILTONIAN_PARAMS  # 46, Tabla I del paper

_dev = qml.device("default.qubit", wires=N_WORK_QUBITS)


@qml.qnode(_dev)
def _readout_circuit(state_vector, h):
    """Prepara el estado post-convolución+post-selección (10 qubits) y mide
    el hamiltoniano de 37 parámetros h. El pooling (qubits 4, 9 descartados)
    no requiere una traza parcial explícita: los operadores de
    HAMILTONIAN_OPS ya excluyen esos dos wires, y <estado|O|estado> para un
    operador que no actúa sobre un qubit es invariante a trazarlo o no."""
    qml.StatePrep(state_vector, wires=range(N_WORK_QUBITS))
    return qml.expval(qml.Hamiltonian(h, HAMILTONIAN_OPS))


def conv_lcu_state(f_flat, beta):
    """Capa convolucional como combinación lineal normalizada de los 9 Q_k
    (desplazamientos cíclicos 2D). Sin ruido es exactamente equivalente a
    preparar las 4 ancillas, aplicar los Q_k controlados y post-seleccionar
    |0000>."""
    f_2d = anp.reshape(f_flat, (PAD_SIZE, PAD_SIZE))
    raw = 0.0
    for k, (dr, dc) in enumerate(OFFSETS):
        shifted = anp.roll(anp.roll(f_2d, dr, axis=0), dc, axis=1)
        raw = raw + beta[k] * anp.reshape(shifted, (-1,))
    norm = anp.sqrt(anp.sum(raw**2))
    return raw / norm


def forward_logit(x_flat, params):
    """x_flat: imagen aplanada (1024,), ya normalizada en L2 (representación).
    params: vector de 46 parámetros entrenables (9 filtro + 37 hamiltoniano)."""
    beta = params[:N_FILTER_PARAMS]
    h = params[N_FILTER_PARAMS:]
    g = conv_lcu_state(x_flat, beta)
    return _readout_circuit(g, h)


def predict_proba(params, x):
    logit = forward_logit(x, params)
    return 1.0 / (1.0 + anp.exp(-logit))
