"""Canales de ruido y las siete condiciones de E2 (ninguna, tres aisladas,
tres compuestas), construidos con `qml.NoiseModel` y `qml.noise.add_noise`.
Esa API nativa de PennyLane inserta el ruido después de cada compuerta o
antes de cada medición, lo que evita hacerlo a mano.

Los parámetros no corresponden a un dispositivo concreto. Son órdenes de
magnitud típicos de qubits superconductores NISQ, elegidos para que las
tres severidades se distingan con claridad:

- Despolarizante: probabilidad por compuerta (`DepolarizingChannel`).
- Error de lectura: `BitFlip` justo antes de cada medición. Es una
  simplificación simétrica (misma probabilidad de 0->1 y de 1->0), porque
  PennyLane no incluye un canal de lectura asimétrico.
- Relajación térmica: `ThermalRelaxationError(pe=0, t1, t2, tg)` después
  de cada compuerta, con un tiempo de compuerta `GATE_TIME_US` fijo entre
  severidades (solo T1/T2 varían).

Las condiciones "aisladas" usan los parámetros de severidad "medium" de un
solo canal, para quedar en una escala comparable a las condiciones
compuestas.
"""

import pennylane as qml

_ANY_OP = qml.BooleanFn(lambda op, **kwargs: True)
_ANY_MEAS = qml.BooleanFn(lambda mp, **kwargs: True)

GATE_TIME_US = 0.1  # tiempo de compuerta de referencia, fijo en las 3 severidades

SEVERITY_PARAMS = {
    "low": {"depolarizing_p": 0.001, "readout_p": 0.01, "t1_us": 100.0, "t2_us": 100.0},
    "medium": {"depolarizing_p": 0.01, "readout_p": 0.03, "t1_us": 50.0, "t2_us": 50.0},
    "high": {"depolarizing_p": 0.05, "readout_p": 0.08, "t1_us": 20.0, "t2_us": 20.0},
}


def depolarizing_noise_model(p):
    """Ruido despolarizante aislado: `DepolarizingChannel(p)` después de
    cada compuerta, en cada wire que toca."""

    def gate_noise(op, **kwargs):
        for w in op.wires:
            qml.DepolarizingChannel(p, wires=w)

    return qml.NoiseModel({_ANY_OP: gate_noise})


def readout_noise_model(p):
    """Error de lectura aislado: `BitFlip(p)` justo antes de cada
    medición."""
    return qml.NoiseModel({}, {_ANY_MEAS: qml.noise.partial_wires(qml.BitFlip, p)})


def relaxation_noise_model(t1_us, t2_us, gate_time_us=GATE_TIME_US):
    """Relajación térmica aislada: `ThermalRelaxationError` después de
    cada compuerta, en cada wire que toca."""

    def gate_noise(op, **kwargs):
        for w in op.wires:
            qml.ThermalRelaxationError(0.0, t1_us, t2_us, gate_time_us, wires=w)

    return qml.NoiseModel({_ANY_OP: gate_noise})


def composite_noise_model(severity):
    """Combina los tres canales (despolarizante + relajación después de
    cada compuerta, error de lectura antes de cada medición) a la
    severidad dada ("low", "medium" o "high")."""
    if severity not in SEVERITY_PARAMS:
        raise KeyError(f"severidad '{severity}' no reconocida, usa una de {sorted(SEVERITY_PARAMS)}")
    p = SEVERITY_PARAMS[severity]

    def gate_noise(op, **kwargs):
        for w in op.wires:
            qml.DepolarizingChannel(p["depolarizing_p"], wires=w)
            qml.ThermalRelaxationError(0.0, p["t1_us"], p["t2_us"], GATE_TIME_US, wires=w)

    return qml.NoiseModel(
        {_ANY_OP: gate_noise},
        {_ANY_MEAS: qml.noise.partial_wires(qml.BitFlip, p["readout_p"])},
    )


NOISE_MODEL_FACTORIES = {
    "none": lambda: None,
    "depolarizing_isolated": lambda: depolarizing_noise_model(SEVERITY_PARAMS["medium"]["depolarizing_p"]),
    "readout_isolated": lambda: readout_noise_model(SEVERITY_PARAMS["medium"]["readout_p"]),
    "relaxation_isolated": lambda: relaxation_noise_model(
        SEVERITY_PARAMS["medium"]["t1_us"], SEVERITY_PARAMS["medium"]["t2_us"]
    ),
    "composite_low": lambda: composite_noise_model("low"),
    "composite_medium": lambda: composite_noise_model("medium"),
    "composite_high": lambda: composite_noise_model("high"),
}

NOISE_CONDITIONS = list(NOISE_MODEL_FACTORIES)  # las 7 condiciones, en orden


def get_noise_model(condition):
    """`qml.NoiseModel` de la condición dada, o `None` para "none". En ese
    caso quien llama simplemente no aplica `qml.noise.add_noise`."""
    if condition not in NOISE_MODEL_FACTORIES:
        raise KeyError(f"condición de ruido '{condition}' no reconocida, usa una de {NOISE_CONDITIONS}")
    return NOISE_MODEL_FACTORIES[condition]()
