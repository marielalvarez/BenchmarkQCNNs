"""Estrategias de inicialización de parámetros entrenables."""

import numpy as np
from pennylane import numpy as pnp


def uniform_pi_init(rng, n_params):
    """Uniforme en [-π, π], como en el Ansatz 8 de Hur et al."""
    return pnp.array(rng.uniform(-np.pi, np.pi, n_params), requires_grad=True)


def normal_init(rng, n_params, std=0.1):
    """N(0, std), como en el circuito LCU con Hamiltoniano de Wei et al."""
    return pnp.array(rng.normal(0, std, n_params), requires_grad=True)
