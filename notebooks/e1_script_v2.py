#!/usr/bin/env python
# coding: utf-8

# Versión en script de notebooks/E1_finite_shots.ipynb, pensada para correr
# E1 como proceso en segundo plano sin mantener abierto un kernel de Jupyter
# durante días. Desde la raíz del repositorio:
#
#     nohup python3 -u notebooks/e1_script_v2.py > results/e1_output_v2.log 2>&1 &
#
# Las celdas de análisis y figuras se conservan tal como en el notebook. La
# sección 5 usa `display()`, que solo existe en IPython, así que con python
# normal el script termina ahí después de guardar el CSV de resultados. Para
# el análisis conviene abrir el notebook con ese CSV ya generado, o correr
# el script con `ipython`.

# # E1: shots finitos, sin ruido
#
# Objetivo: ver cómo cambian Hur, Wei y Gong (y Cong, si
# `INCLUDE_CONG_IN_E1_E2`) al pasar del simulador analítico de E0 a medición
# con 256, 1024 o 4096 shots, todavía sin ruido de hardware. La variable
# independiente es el presupuesto de shots, así que cada modelo se compara
# consigo mismo entre presupuestos.
#
# Solo aplica a las QCNN, porque las CNN análogas no tienen circuito que
# medir y se quedan en el régimen analítico de `E0_quantum_advantage.ipynb`.
#
# El entrenamiento usa `qcnn_benchmark.training.train_binary_classifier` y el
# mismo criterio estadístico de E0A y E0B. Lo único que cambia es
# `predict_proba_fn`, que viene de
# `qcnn_benchmark.execution.make_shots_predict_proba` y estima con N shots
# tanto la salida como el gradiente (ver
# `src/qcnn_benchmark/execution/shots.py`).


# In[ ]:


import json
import pathlib
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qcnn_benchmark.config import load_config
from qcnn_benchmark.data import load_mnist_pool, load_fashion_mnist_pool
from qcnn_benchmark.representations import build_pca_dataset, build_amplitude_dataset
from qcnn_benchmark.models import qcnn_hur, qcnn_wei, qcnn_gong, qcnn_cong
from qcnn_benchmark.training import train_binary_classifier, uniform_pi_init, normal_init
from qcnn_benchmark.execution import make_shots_predict_proba
from qcnn_benchmark.execution.shots import n_trainable_params
from qcnn_benchmark.metrics import batch_accuracy, confusion_counts
from qcnn_benchmark.stats import (
    bootstrap_bca_ci,
    paired_sign_permutation_test,
    holm_correction,
    cohens_dz,
    probability_of_superiority,
)


# ## 0. Parámetros manuales
#
# `WINNER_MODEL` viene de `E0A_seleccion_candidato.ipynb`.
# `INCLUDE_CONG_IN_E1_E2` se decide a partir de los resultados de
# `E0B_validez_aportacion.ipynb` (ver su Sec. 10).
#

# In[ ]:


WINNER_MODEL = "wei"
INCLUDE_CONG_IN_E1_E2 = False

assert WINNER_MODEL in ("hur", "wei", "gong"), "Completa WINNER_MODEL arriba"
assert isinstance(INCLUDE_CONG_IN_E1_E2, bool), "Completa INCLUDE_CONG_IN_E1_E2 arriba (True o False)"

# E1 corre los tres candidatos y no solo el ganador de E0A, así que
# WINNER_MODEL solo queda como dato informativo en el provenance. Cong se
# excluye por defecto porque bajo shots su entrenamiento tardó más de 40
# horas en una sola corrida.
MODELS_TO_RUN = ["hur", "wei", "gong"] + (["cong"] if INCLUDE_CONG_IN_E1_E2 else [])
print("MODELS_TO_RUN:", MODELS_TO_RUN)


# ## 1. Configuración declarativa

# In[9]:


FAST_SMOKE_TEST = False  # True usa 2 semillas, pocas actualizaciones y 1 presupuesto, solo para validar el pipeline

_CONFIG_CANDIDATES = [pathlib.Path("configs/e1.yaml"), pathlib.Path("../configs/e1.yaml")]
CONFIG_PATH = next(p for p in _CONFIG_CANDIDATES if p.exists())
config = load_config(CONFIG_PATH)

config = config.model_copy(update={"models": MODELS_TO_RUN})

if FAST_SMOKE_TEST:
    config = config.model_copy(update={
        "n_seeds": 2,
        "shots_budgets": [256],
        "protocol": config.protocol.model_copy(update={"n_updates": 10, "val_check_every": 5, "patience_checks": 2}),
    })

print("experiment_id:", config.experiment_id)
print("canonical_hash:", config.canonical_hash(), "(FAST_SMOKE_TEST)" if FAST_SMOKE_TEST else "")
print("datasets:", [d.name for d in config.datasets])
print("models:", config.models)
print("shots_budgets:", config.shots_budgets)
print("run_seeds:", config.run_seeds())


# ## 2. Estimación de costo de cómputo
# 
# Los tiempos por actualización se midieron con 3 actualizaciones reales
# (shots=256, batch=25). Escalar el costo de backprop de E0A lo subestima
# entre 10 y 20 veces, porque bajo shots PennyLane usa parameter-shift, que
# necesita unas 2×n_params evaluaciones del circuito por gradiente.
#
# Hur y Gong son órdenes de magnitud más caros que Wei. Wei solo diferencia
# los coeficientes del hamiltoniano de lectura, mientras que Hur y Gong
# diferencian todos los parámetros del circuito con parameter-shift.
#

# In[ ]:


# Segundos por actualización medidos. Para wei es el promedio de las tres
# primeras semillas de results/e1_raw_c2ee95d45e65.csv (coat_vs_shirt, 256
# shots). Las semillas posteriores de ese CSV no sirven para medir tiempo
# porque la máquina se suspendió durante la corrida.
REAL_SEC_PER_UPDATE = {"hur": 62.2, "gong": 21.0, "wei": 0.6}

n_shots_budgets = len(config.shots_budgets)
n_seeds = config.n_seeds
n_datasets = len(config.datasets)

total_sec_measured = sum(
    REAL_SEC_PER_UPDATE[m] * config.protocol.n_updates * n_shots_budgets * n_seeds * n_datasets
    for m in config.models
)

print("Estimación basada en costo REAL medido por actualización (cota superior, asume 0 early stopping):")
print(f"  {n_shots_budgets} presupuestos de shots x {n_seeds} semillas x {n_datasets} datasets x "
      f"{config.protocol.n_updates} actualizaciones x {len(config.models)} modelo(s)")
print(f"  ~{total_sec_measured/3600:.0f} h (~{total_sec_measured/86400:.1f} dias) combinado\n")
for m in config.models:
    per_model = REAL_SEC_PER_UPDATE[m] * config.protocol.n_updates * n_shots_budgets * n_seeds * n_datasets
    print(f"  {m}: ~{per_model/3600:.1f} h (~{per_model/86400:.2f} dias)")
print()
print("En la practica el early stopping reduce esto algo (la corrida ya completada")
print("de wei detuvo la mayoria de sus corridas antes de las 500 actualizaciones),")
print("pero no cambia el orden de magnitud: correr esta matriz completa toma dias,")
print("no horas, para hur y gong.")


# ## 3. Registro de modelos y representaciones

# In[11]:


MODEL_REGISTRY = {
    "hur": {"module": qcnn_hur, "init": uniform_pi_init, "normalize": True,
            "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=16, verbose=False)},
    "wei": {"module": qcnn_wei, "init": normal_init, "normalize": False,
            "build_rep": lambda x, y, pos, neg: build_amplitude_dataset(x, y, pos, neg, verbose=False)},
    "gong": {"module": qcnn_gong, "init": uniform_pi_init, "normalize": True,
             "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=8, verbose=False)},
    "cong": {"module": qcnn_cong, "init": uniform_pi_init, "normalize": True,
             "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=8, verbose=False)},
}

_POOL_CACHE = {}


def get_pool(source, normalize):
    key = (source, normalize)
    if key not in _POOL_CACHE:
        loader = load_mnist_pool if source == "mnist" else load_fashion_mnist_pool
        _POOL_CACHE[key] = loader(normalize=normalize)
    return _POOL_CACHE[key]


# ## 4. Ejecución (matriz: datasets × modelos × presupuestos de shots × semillas)

# In[ ]:


RESULTS_DIR = pathlib.Path("results") if pathlib.Path("results").exists() else pathlib.Path("../results")
raw_path = RESULTS_DIR / f"e1_raw_{config.canonical_hash()}.csv"
print("canonical_hash:", config.canonical_hash(), flush=True)
print("models:", config.models, flush=True)
print("raw_path:", raw_path, flush=True)

# Si existe un CSV parcial de una corrida interrumpida, se retoma sin repetir
# las combinaciones (dataset, modelo, n_shots, semilla) que ya tiene.
if raw_path.exists():
    existing_df = pd.read_csv(raw_path)
    rows = existing_df.to_dict("records")
    done_keys = {(r["dataset"], r["model"], r["n_shots"], r["seed"]) for r in rows}
    print(f"Reanudando: {len(rows)} filas ya hechas en {raw_path}, se saltan.", flush=True)
else:
    rows = []
    done_keys = set()

t_start = time.time()
for dataset in config.datasets:
    for model_name in config.models:
        entry = MODEL_REGISTRY[model_name]
        x_all, y_all = get_pool(dataset.source, entry["normalize"])
        rep = entry["build_rep"](x_all, y_all, dataset.class_pos, dataset.class_neg)
        n_params = n_trainable_params(model_name)  # difiere de TOTAL_PARAMS solo en "wei", ver execution/shots.py

        for n_shots in config.shots_budgets:
            predict_proba = make_shots_predict_proba(model_name, n_shots)

            for seed in config.run_seeds():
                if (dataset.name, model_name, n_shots, seed) in done_keys:
                    continue
                tag = f"{dataset.name}-{model_name}-shots{n_shots}-seed{seed}"
                # verbose=True para ver progreso en el log, porque una sola
                # corrida de Hur puede tardar más de 8 horas.
                result = train_binary_classifier(
                    predict_proba, n_params, rep, entry["init"],
                    run_seed=seed,
                    batch_size=config.protocol.batch_size,
                    n_updates=config.protocol.n_updates,
                    learning_rate=config.protocol.learning_rate,
                    beta1=config.protocol.beta1,
                    beta2=config.protocol.beta2,
                    clip_norm=config.protocol.clip_norm,
                    val_check_every=config.protocol.val_check_every,
                    patience_checks=config.protocol.patience_checks,
                    min_delta=config.protocol.min_delta,
                    verbose=True,
                    tag=tag,
                )
                train_acc = batch_accuracy(predict_proba, result["params"], rep["X_train"], rep["y_train"])
                test_acc = batch_accuracy(predict_proba, result["params"], rep["X_test"], rep["y_test"])
                test_conf = confusion_counts(predict_proba, result["params"], rep["X_test"], rep["y_test"])
                rows.append({
                    "dataset": dataset.name, "model": model_name, "n_shots": n_shots, "seed": seed,
                    "train_acc": train_acc, "test_acc": test_acc,
                    "test_tp": test_conf["tp"], "test_fp": test_conf["fp"],
                    "test_fn": test_conf["fn"], "test_tn": test_conf["tn"],
                    "n_updates_run": result["n_updates_run"], "stopped_early_at": result["stopped_early_at"],
                    "best_val_loss": result["best_val_loss"],
                })
                elapsed = time.time() - t_start
                print(f"[{tag}] test_acc={test_acc:.4f}  (t={elapsed:.0f}s / {elapsed/3600:.1f}h acumulado)",
                      flush=True)

                raw_df = pd.DataFrame(rows)
                raw_df.to_csv(raw_path, index=False)

print("Total:", time.time() - t_start, "s")
raw_df = pd.DataFrame(rows)
raw_df


# ## 5. Resultados crudos

# In[13]:


raw_df = pd.read_csv(raw_path)
display(raw_df)
raw_df.pivot_table(index=["dataset", "model"], columns="n_shots", values="test_acc", aggfunc=list)


# ## 6. Métricas agregadas (media ± IC bootstrap BCa por dataset, modelo y presupuesto de shots)

# In[14]:


agg_rows = []
for (dataset_name, model_name, n_shots), group in raw_df.groupby(["dataset", "model", "n_shots"]):
    accs = group["test_acc"].to_numpy()
    lo, hi = bootstrap_bca_ci(
        accs, n_bootstrap=config.statistical_criterion.n_bootstrap,
        confidence_level=config.statistical_criterion.confidence_level,
        rng=np.random.default_rng(0),
    )
    agg_rows.append({
        "dataset": dataset_name, "model": model_name, "n_shots": n_shots,
        "mean_test_acc": float(np.mean(accs)), "ci_lo": lo, "ci_hi": hi, "n_seeds": len(accs),
    })

agg_df = pd.DataFrame(agg_rows)
agg_df


# ## 7. Análisis estadístico (presupuestos de shots entre sí, por modelo y dataset)
# 
# Compara cada par de presupuestos de shots (256 vs. 1024, 1024 vs. 4096 y
# 256 vs. 4096) dentro de cada combinación de dataset y modelo. La
# comparación entre modelos ya se hizo en E0A y E0B.
#

# In[15]:


from itertools import combinations

comparison_rows = []
for (dataset_name, model_name), group in raw_df.groupby(["dataset", "model"]):
    pivot = group.pivot(index="seed", columns="n_shots", values="test_acc")
    for shots_a, shots_b in combinations(config.shots_budgets, 2):
        x, y = pivot[shots_a].to_numpy(), pivot[shots_b].to_numpy()
        p_value, mean_diff = paired_sign_permutation_test(
            x, y, n_permutations=config.statistical_criterion.n_bootstrap,
            rng=np.random.default_rng(0),
        )
        comparison_rows.append({
            "dataset": dataset_name, "model": model_name, "shots_a": shots_a, "shots_b": shots_b,
            "mean_diff_a_minus_b": mean_diff, "p_value": p_value,
            "cohens_dz": cohens_dz(x, y), "prob_superiority_a_over_b": probability_of_superiority(x, y),
        })

comparisons_df = pd.DataFrame(comparison_rows)
if len(comparisons_df):
    adjusted, rejected = holm_correction(comparisons_df["p_value"], alpha=config.statistical_criterion.alpha)
    comparisons_df["p_value_holm"] = adjusted
    comparisons_df["significant"] = rejected
else:
    print("Sin comparaciones: config.shots_budgets tiene menos de 2 valores (normal en FAST_SMOKE_TEST).")
comparisons_df


# ## 8. Figuras y tablas

# In[16]:


fig, axes = plt.subplots(1, len(config.datasets), figsize=(5 * len(config.datasets), 4), sharey=True)
if len(config.datasets) == 1:
    axes = [axes]
for ax, (dataset_name, group) in zip(axes, agg_df.groupby("dataset")):
    for model_name, model_group in group.groupby("model"):
        model_group = model_group.set_index("n_shots").loc[config.shots_budgets]
        yerr = [model_group["mean_test_acc"] - model_group["ci_lo"], model_group["ci_hi"] - model_group["mean_test_acc"]]
        ax.errorbar(model_group.index.astype(str), model_group["mean_test_acc"], yerr=yerr, marker="o", capsize=4, label=model_name)
    ax.set_title(dataset_name)
    ax.set_xlabel("Shots")
    ax.set_ylabel("Exactitud de prueba")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
plt.suptitle("E1: media ± IC bootstrap BCa por presupuesto de shots (config hash: " + config.canonical_hash() + ")")
plt.tight_layout()
plt.show()


# In[17]:


if len(comparisons_df):
    display(comparisons_df.style.format({
        "mean_diff_a_minus_b": "{:+.4f}", "p_value": "{:.4f}", "p_value_holm": "{:.4f}",
        "cohens_dz": "{:.2f}", "prob_superiority_a_over_b": "{:.2f}",
    }))


# ## 9. Conclusión: ¿converge la exactitud con más shots?

# In[18]:


alpha = config.statistical_criterion.alpha
max_shots = max(config.shots_budgets)
for (dataset_name, model_name), group in agg_df.groupby(["dataset", "model"]):
    group = group.set_index("n_shots").loc[config.shots_budgets]
    print(f"{dataset_name} / {model_name}: " + " -> ".join(f"{s} shots: {group.loc[s, 'mean_test_acc']:.4f}" for s in config.shots_budgets))
    row = comparisons_df[
        (comparisons_df["dataset"] == dataset_name) & (comparisons_df["model"] == model_name)
        & (comparisons_df["shots_a"] == config.shots_budgets[0]) & (comparisons_df["shots_b"] == max_shots)
    ] if len(comparisons_df) else comparisons_df
    if len(row):
        sig = bool(row["significant"].iloc[0])
        print(f"  {config.shots_budgets[0]} vs {max_shots} shots: {'diferencia significativa' if sig else 'sin diferencia significativa'} tras Holm")


# ## 10. Registro de reproducibilidad

# In[19]:


import pennylane, sklearn, scipy

provenance = {
    "experiment_id": config.experiment_id,
    "winner_model": WINNER_MODEL,
    "include_cong": INCLUDE_CONG_IN_E1_E2,
    "canonical_hash": config.canonical_hash(),
    "config": json.loads(config.canonical_json()),
    "fast_smoke_test": FAST_SMOKE_TEST,
    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    "raw_results_path": str(raw_path),
    "package_versions": {
        "pennylane": pennylane.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
    },
}

provenance_path = RESULTS_DIR / f"e1_provenance_{config.canonical_hash()}.json"
provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
print("Provenance guardado en:", provenance_path)
print(json.dumps(provenance, indent=2, ensure_ascii=False))

