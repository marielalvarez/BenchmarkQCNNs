"""Pruebas del esquema de configuración declarativa (Pydantic, YAML y hash
canónico)."""

import pytest
from pydantic import ValidationError

from qcnn_benchmark.config import load_config


def test_e0a_config_loads_and_validates():
    cfg = load_config("configs/e0a.yaml")
    assert cfg.experiment_id == "E0A"
    assert {d.name for d in cfg.datasets} == {"coat_vs_shirt", "4_vs_9", "1_vs_0"}
    assert cfg.models == ["hur", "wei", "gong"]
    assert cfg.n_seeds == 5


def test_run_seeds_derive_from_seed_root():
    cfg = load_config("configs/e0a.yaml")
    assert cfg.run_seeds() == [cfg.seed_root + i for i in range(cfg.n_seeds)]


def test_canonical_hash_is_stable_across_independent_loads():
    cfg_a = load_config("configs/e0a.yaml")
    cfg_b = load_config("configs/e0a.yaml")
    assert cfg_a.canonical_hash() == cfg_b.canonical_hash()


def test_canonical_hash_changes_if_config_changes():
    cfg = load_config("configs/e0a.yaml")
    tweaked = cfg.model_copy(update={"n_seeds": cfg.n_seeds + 1})
    assert tweaked.canonical_hash() != cfg.canonical_hash()


def test_config_is_immutable():
    cfg = load_config("configs/e0a.yaml")
    with pytest.raises(ValidationError):
        cfg.experiment_id = "other"


def test_e0a_canonical_hash_matches_already_computed_results():
    """Los resultados guardados de E0A (results/e0a_raw_5a6b6cb49509.csv y
    su provenance) están indexados por este hash. Si un cambio al esquema
    lo altera, el vínculo con esos archivos se rompería sin aviso."""
    cfg = load_config("configs/e0a.yaml")
    assert cfg.canonical_hash() == "5a6b6cb49509"


def test_optional_e1_e2_fields_default_to_none_and_dont_affect_hash():
    """shots_budgets y noise_conditions solo aplican a E1 y E2. En E0A
    deben quedar en None y no aparecer en el hash."""
    cfg = load_config("configs/e0a.yaml")
    assert cfg.shots_budgets is None
    assert cfg.noise_conditions is None
    assert '"shots_budgets"' not in cfg.canonical_json()
    assert '"noise_conditions"' not in cfg.canonical_json()
