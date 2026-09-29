"""Transpilación a un mapa de acoplamiento fijo para E2. Inserta SWAPs donde
hace falta para que el circuito respete la conectividad dada por
`coupling_map`, usando `qml.transforms.transpile` de PennyLane en lugar de
implementar el routing desde cero.

No se usa SABRE (Sun et al.) porque es específico de Qiskit y requeriría
`qiskit` y `pennylane-qiskit`. El router de PennyLane no es idéntico a
SABRE, pero cumple el objetivo de E2, que es someter al circuito a las
restricciones de conectividad de un dispositivo NISQ antes de medir con
ruido. Si hiciera falta reproducir un dispositivo físico concreto, el paso
natural sería migrar a Qiskit.

`qml.transforms.transpile` no admite valores esperados de Hamiltonianos ni
de productos tensoriales de observables. Por eso la lectura de Wei et al.
(`qml.expval(qml.Hamiltonian(...))`) no se transpila y en E2 ese modelo
solo recibe shots y ruido (ver `notebooks/E2_nisq_noise.ipynb`).

El transform también revisa `len(op.wires) > 2` antes de expandir
templates. Un template multi-wire como `AngleEmbedding(wires=range(8))`,
que usan Hur y Cong, cuenta ahí como una sola operación de 8 wires y
provoca `NotImplementedError` aunque se descomponga en puertas de un qubit.
Gong no tiene el problema porque su circuito solo usa puertas elementales.
Por eso `transpile_qnode` expande los templates antes de transpilar.
"""

import pennylane as qml


def grid_coupling_map(rows, cols):
    """Mapa de acoplamiento de una rejilla `rows` x `cols` (vecinos más
    cercanos, sin diagonales), como lista de aristas (i, j)."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                edges.append((node, node + 1))
            if r + 1 < rows:
                edges.append((node, node + cols))
    return edges


# Rejilla genérica de 4x4. No pretende representar un dispositivo real,
# solo imponer una conectividad limitada y realista.
COUPLING_MAP_16Q = grid_coupling_map(4, 4)


def transpile_qnode(qnode, coupling_map=COUPLING_MAP_16Q):
    """Devuelve un qnode nuevo, equivalente a `qnode` pero con SWAPs
    insertados para respetar `coupling_map`, sin modificar `qnode`. Para
    combinarlo con shots se aplica primero `transpile_qnode` y después
    `qml.set_shots(..., shots=n)` sobre el resultado.

    Antes de transpilar expande los templates multi-wire (por ejemplo
    `AngleEmbedding`) a puertas de uno o dos qubits, por la razón explicada
    en el docstring del módulo. Las puertas que ya son de uno o dos qubits
    no se tocan, así que el circuito de gong queda igual."""
    qnode = qml.transforms.decompose(qnode, stopping_condition=lambda op: len(op.wires) <= 2)
    return qml.transforms.transpile(qnode, coupling_map=coupling_map)
