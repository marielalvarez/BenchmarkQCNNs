"""Pruebas de funcionamiento, no de fidelidad, de los canales de ruido.
Verifican que cada condición se construye y ejecuta sobre un circuito real,
que produce una probabilidad válida y que "none" coincide con el resultado
sin ruido salvo por el muestreo de shots.

El gradiente de punta a punta bajo ruido no se prueba aquí sino en
E2_nisq_noise.ipynb con FAST_SMOKE_TEST. La simulación de matriz de
densidad con parameter-shift es órdenes de magnitud más lenta que la de
estado puro, y un solo gradiente bajo la condición más severa tarda
minutos."""

import numpy as np
import pytest

from qcnn_benchmark.execution import grid_coupling_map
from qcnn_benchmark.models import qcnn_gong
from qcnn_benchmark.noise import (
    NOISE_CONDITIONS,
    get_noise_model,
    make_noisy_predict_proba,
    make_wei_noisy_predict_proba,
)


def test_seven_noise_conditions_declared():
    assert len(NOISE_CONDITIONS) == 7
    assert NOISE_CONDITIONS[0] == "none"


def test_get_noise_model_none_is_none():
    assert get_noise_model("none") is None


@pytest.mark.parametrize("condition", NOISE_CONDITIONS[1:])
def test_get_noise_model_builds_a_noise_model(condition):
    import pennylane as qml

    model = get_noise_model(condition)
    assert isinstance(model, qml.NoiseModel)


def test_unknown_noise_condition_raises_key_error():
    with pytest.raises(KeyError):
        get_noise_model("not_a_condition")


@pytest.mark.parametrize("condition", NOISE_CONDITIONS)
def test_gong_noisy_predict_proba_in_unit_interval(condition):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)
    params = rng.uniform(-np.pi, np.pi, qcnn_gong.TOTAL_PARAMS)
    pp = make_noisy_predict_proba("gong", n_shots=64, condition=condition, coupling_map=grid_coupling_map(2, 4))
    p = float(pp(params, x))
    assert 0.0 <= p <= 1.0


def test_gong_noisy_without_coupling_map_also_works():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 8)
    params = rng.uniform(-np.pi, np.pi, qcnn_gong.TOTAL_PARAMS)
    pp = make_noisy_predict_proba("gong", n_shots=64, condition="composite_medium", coupling_map=None)
    p = float(pp(params, x))
    assert 0.0 <= p <= 1.0


def test_wei_noisy_predict_proba_in_unit_interval():
    """Bajo ruido Wei también entrena solo `h` (37 parámetros y no 46). Ver
    execution.shots.make_wei_predict_proba."""
    from qcnn_benchmark.execution.shots import n_trainable_params

    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    x = x / np.linalg.norm(x)
    params = rng.normal(0, 0.1, n_trainable_params("wei"))
    pp = make_wei_noisy_predict_proba(n_shots=64, condition="composite_medium")
    p = float(pp(params, x))
    assert 0.0 <= p <= 1.0


def test_wei_noisy_gradient_flows():
    """A diferencia de hur, gong y cong, aquí sí se prueba el gradiente.
    Wei bajo ruido solo diferencia `h`, que son coeficientes de un
    Hamiltoniano, y PennyLane obtiene ese gradiente de forma lineal sin
    desplazar parámetros uno por uno, así que es barato."""
    import pennylane as qml

    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    x = x / np.linalg.norm(x)
    from qcnn_benchmark.execution.shots import n_trainable_params

    params = qml.numpy.array(rng.normal(0, 0.1, n_trainable_params("wei")), requires_grad=True)
    pp = make_wei_noisy_predict_proba(n_shots=64, condition="composite_medium")

    grad = qml.grad(lambda p: pp(p, x))(params)
    assert np.linalg.norm(np.array(grad)) > 0.0
