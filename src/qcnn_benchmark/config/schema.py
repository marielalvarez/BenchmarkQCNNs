"""Esquema Pydantic para configuración declarativa de experimentos, con hash
canónico de reproducibilidad.

El esquema cubre lo que usan los experimentos actuales (datasets, modelos,
protocolo de entrenamiento, criterio estadístico, presupuestos de shots y
condiciones de ruido). Los campos se agregan conforme un experimento los
necesita, en lugar de anticipar un sistema de plugins genérico.

Todos los modelos son inmutables (`model_config = ConfigDict(frozen=True)`),
así que una `ExperimentConfig` ya construida se reemplaza en vez de
modificarse. El hash canónico (`ExperimentConfig.canonical_hash()`) se
calcula sobre una serialización JSON con claves ordenadas, por lo que es
estable entre máquinas y corridas mientras la configuración declarada sea
la misma.
"""

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetSpec(BaseModel):
    """Una tarea de clasificación binaria: dos clases de un dataset fuente.

    Convención de polaridad: `class_pos` es siempre la primera clase
    nombrada en `name` (por ejemplo "coat_vs_shirt" da class_pos=coat). Los
    papers de referencia no siguen un patrón consistente (Hur et al. en
    "0 vs 1" y Wei et al. en "1 vs 8" usan ambos class_pos=1). La exactitud
    de un clasificador binario no depende de esta elección (ver la nota de
    equivalencia 0-vs-1 y 1-vs-0 en `notebooks/00_reproduce_hur.ipynb`),
    pero fijarla evita ambigüedad al reportar resultados.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    source: Literal["mnist", "fashion_mnist"]
    class_pos: int
    class_neg: int


class TrainingProtocol(BaseModel):
    """Hiperparámetros de `qcnn_benchmark.training.train_binary_classifier`.
    Los valores por default replican las constantes de `training/loop.py`.
    Se declaran de nuevo aquí para que la config sea autónoma y su hash
    describa por completo lo que se corrió, sin tener que leer el código."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = 25
    n_updates: int = 200
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    clip_norm: float = 5.0
    val_check_every: int = 10
    patience_checks: int = 5
    min_delta: float = 1e-4


class StatisticalCriterion(BaseModel):
    """Criterio estadístico para comparar modelos: media con IC bootstrap BCa
    por dataset, en lugar de una sola cifra resumen, más un test pareado de
    permutación de signo con corrección de Holm para las comparaciones
    múltiples."""

    model_config = ConfigDict(frozen=True)

    aggregation: Literal["mean_per_dataset"] = "mean_per_dataset"
    ci_method: Literal["bootstrap_bca"] = "bootstrap_bca"
    n_bootstrap: int = 10_000
    confidence_level: float = 0.95
    comparison_test: Literal["paired_sign_permutation"] = "paired_sign_permutation"
    multiple_comparison_correction: Literal["holm"] = "holm"
    alpha: float = 0.05


class ExperimentConfig(BaseModel):
    """Configuración completa e inmutable de un experimento. `canonical_hash()`
    identifica la configuración de forma reproducible entre máquinas."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    description: str
    datasets: list[DatasetSpec] = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    protocol: TrainingProtocol = TrainingProtocol()
    n_seeds: int = 5
    seed_root: int
    sample_seed: int
    statistical_criterion: StatisticalCriterion = StatisticalCriterion()

    # Solo E1 y E2 usan estos campos. Quedan en `None` en los demás
    # experimentos y `canonical_json` excluye los `None`, así que su
    # existencia no altera el hash de las configs anteriores.
    shots_budgets: Optional[list[int]] = None  # E1: [256, 1024, 4096]
    noise_conditions: Optional[list[str]] = None  # E2: las 7 de `qcnn_benchmark.noise.NOISE_CONDITIONS`

    def run_seeds(self) -> list[int]:
        """Las `n_seeds` semillas de ejecución, derivadas de `seed_root`.
        Toda la aleatoriedad del experimento parte de esta semilla raíz para
        que ninguna parte del código dependa de estado aleatorio implícito."""
        return [self.seed_root + i for i in range(self.n_seeds)]

    def canonical_json(self) -> str:
        """Serialización JSON canónica usada para el hash, con claves
        ordenadas, separadores fijos y sin campos en `None`. Excluir los
        `None` permite agregar campos opcionales al esquema sin cambiar el
        hash de configs que no los usan (hay un test de regresión para
        `configs/e0a.yaml`)."""
        return json.dumps(self.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))

    def canonical_hash(self) -> str:
        """sha256 de `canonical_json()`, primeros 12 caracteres hex (estilo
        hash corto de git). Basta para distinguir configuraciones en nombres
        de archivo y reportes sin cargar un hash de 64 caracteres."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:12]
