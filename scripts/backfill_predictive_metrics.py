"""Calcula la matriz de confusión (de la que salen precision, recall, F1,
balanced accuracy y macro-F1) para las 105 corridas del régimen analítico
(E0A, Quantum Advantage y E0B), que originalmente solo guardaron test_acc.

Como no se guardaron los parámetros entrenados, cada corrida se reentrena
con la misma semilla, protocolo y configuración registrados en su
provenance. El entrenamiento es determinista, así que se obtiene el mismo
modelo. Como verificación se compara el test_acc recalculado con el del
CSV original, y cualquier diferencia indica que la reconstrucción no fue
fiel.

E1 no se procesa aquí porque `scripts/run_e1_expansion.py` ya guarda la
matriz de confusión desde el inicio.
"""

import json
import pathlib
import time

import numpy as np
import pandas as pd

from qcnn_benchmark.data import load_mnist_pool, load_fashion_mnist_pool
from qcnn_benchmark.representations import build_pca_dataset, build_amplitude_dataset
from qcnn_benchmark.models import (
    qcnn_hur, qcnn_wei, qcnn_gong, qcnn_cong,
    cnn_hur_analoga, cnn_wei_analoga, cnn_gong_analoga,
)
from qcnn_benchmark.training import train_binary_classifier, uniform_pi_init, normal_init
from qcnn_benchmark.metrics import batch_accuracy, confusion_counts

RESULTS_DIR = pathlib.Path("results")

MODEL_REGISTRY = {
    "hur": {"module": qcnn_hur, "init": uniform_pi_init, "normalize": True,
            "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=16, verbose=False)},
    "wei": {"module": qcnn_wei, "init": normal_init, "normalize": False,
            "build_rep": lambda x, y, pos, neg: build_amplitude_dataset(x, y, pos, neg, verbose=False)},
    "gong": {"module": qcnn_gong, "init": uniform_pi_init, "normalize": True,
             "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=8, verbose=False)},
    "cong": {"module": qcnn_cong, "init": uniform_pi_init, "normalize": True,
             "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=8, verbose=False)},
    "cnn_hur_analoga": {"module": cnn_hur_analoga, "init": cnn_hur_analoga.init, "normalize": True,
                         "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=cnn_hur_analoga.INPUT_SIZE, verbose=False)},
    "cnn_wei_analoga": {"module": cnn_wei_analoga, "init": cnn_wei_analoga.init, "normalize": True,
                         "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=cnn_wei_analoga.INPUT_SIZE, verbose=False)},
    "cnn_gong_analoga": {"module": cnn_gong_analoga, "init": cnn_gong_analoga.init, "normalize": True,
                          "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=cnn_gong_analoga.INPUT_SIZE, verbose=False)},
}

EXPERIMENTS = [
    ("e0a", "5a6b6cb49509"),
    ("e0_quantum_advantage", "12aaded3bf0b"),
    ("e0b", "aad70cb6df8c"),
]

_POOL_CACHE = {}


def get_pool(source, normalize):
    key = (source, normalize)
    if key not in _POOL_CACHE:
        loader = load_mnist_pool if source == "mnist" else load_fashion_mnist_pool
        _POOL_CACHE[key] = loader(normalize=normalize)
    return _POOL_CACHE[key]


def main():
    for tag, config_hash in EXPERIMENTS:
        provenance = json.loads((RESULTS_DIR / f"{tag}_provenance_{config_hash}.json").read_text())
        cfg = provenance["config"]
        original_df = pd.read_csv(RESULTS_DIR / f"{tag}_raw_{config_hash}.csv")

        out_path = RESULTS_DIR / f"{tag}_predictive_backfill_{config_hash}.csv"
        rows = []
        t_start = time.time()
        print(f"\n=== {tag} (hash {config_hash}): {len(cfg['datasets'])} datasets x "
              f"{len(cfg['models'])} modelos x {cfg['n_seeds']} semillas ===")

        for dataset in cfg["datasets"]:
            for model_name in cfg["models"]:
                entry = MODEL_REGISTRY[model_name]
                x_all, y_all = get_pool(dataset["source"], entry["normalize"])
                rep = entry["build_rep"](x_all, y_all, dataset["class_pos"], dataset["class_neg"])

                for seed in [cfg["seed_root"] + i for i in range(cfg["n_seeds"])]:
                    tag_run = f"{dataset['name']}-{model_name}-seed{seed}"
                    result = train_binary_classifier(
                        entry["module"].predict_proba, entry["module"].TOTAL_PARAMS, rep, entry["init"],
                        run_seed=seed,
                        batch_size=cfg["protocol"]["batch_size"],
                        n_updates=cfg["protocol"]["n_updates"],
                        learning_rate=cfg["protocol"]["learning_rate"],
                        beta1=cfg["protocol"]["beta1"],
                        beta2=cfg["protocol"]["beta2"],
                        clip_norm=cfg["protocol"]["clip_norm"],
                        val_check_every=cfg["protocol"]["val_check_every"],
                        patience_checks=cfg["protocol"]["patience_checks"],
                        min_delta=cfg["protocol"]["min_delta"],
                        verbose=False,
                        tag=tag_run,
                    )
                    test_acc_recomputed = batch_accuracy(
                        entry["module"].predict_proba, result["params"], rep["X_test"], rep["y_test"]
                    )
                    test_conf = confusion_counts(
                        entry["module"].predict_proba, result["params"], rep["X_test"], rep["y_test"]
                    )

                    original_row = original_df[
                        (original_df["dataset"] == dataset["name"]) & (original_df["model"] == model_name)
                        & (original_df["seed"] == seed)
                    ]
                    test_acc_original = float(original_row["test_acc"].iloc[0]) if len(original_row) else float("nan")
                    acc_from_counts = (test_conf["tp"] + test_conf["tn"]) / sum(test_conf.values())
                    matches = np.isclose(test_acc_recomputed, test_acc_original, atol=1e-9)

                    rows.append({
                        "dataset": dataset["name"], "model": model_name, "seed": seed,
                        "test_tp": test_conf["tp"], "test_fp": test_conf["fp"],
                        "test_fn": test_conf["fn"], "test_tn": test_conf["tn"],
                        "test_acc_recomputed": test_acc_recomputed,
                        "test_acc_original": test_acc_original,
                        "sanity_check_matches": bool(matches),
                    })
                    flag = "OK" if matches else "*** MISMATCH ***"
                    print(f"[{tag}/{tag_run}] acc_recomputed={test_acc_recomputed:.4f} "
                          f"acc_original={test_acc_original:.4f} acc_from_counts={acc_from_counts:.4f} "
                          f"{flag}  (t={time.time()-t_start:.0f}s acumulado)", flush=True)

                    pd.DataFrame(rows).to_csv(out_path, index=False)

        n_mismatch = sum(1 for r in rows if not r["sanity_check_matches"])
        print(f"=== {tag} listo: {len(rows)} filas, {n_mismatch} mismatches, "
              f"{time.time()-t_start:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
