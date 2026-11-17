"""
Práctica 3 — Comparativa de clasificadores.

Ejecutar desde la raíz del repositorio:
    python src/experimentos.py            # todo
    python src/experimentos.py --rapido   # menos repeticiones (para probar)

Genera en resultados/:
    tabla_default.csv, tabla_knn.csv, tabla_svm.csv, tabla_mlp.csv,
    tabla_escalado.csv, tabla_cv_anidada.csv, tabla_complejidad.csv,
    resumen.json y figuras/*.png
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from clasificadores import DistanciaMinima, UnoNN, KNN          # noqa: E402
from datasets import cargar, dataset_sintetico                   # noqa: E402
from validacion import holdout, kfold, resumir, particiones_kfold  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "resultados")
FIG = os.path.join(RES, "figuras")
os.makedirs(FIG, exist_ok=True)

DATASETS = ["Sintético", "Wine"]
SEMILLA = 0

# Paleta categórica (validada para daltonismo en pares adyacentes)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TXT2, "axes.labelcolor": TXT,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110,
    "savefig.dpi": 160, "savefig.bbox": "tight", "legend.frameon": False,
})


def modelos_default():
    """Parámetros por defecto (los de scikit-learn; K-NN con k=5 como sklearn)."""
    return {
        "Distancia Mínima": DistanciaMinima(),
        "1-NN": UnoNN(),
        "K-NN (k=5)": KNN(k=5),
        "Naïve Bayes": GaussianNB(),
        "SVM": SVC(random_state=SEMILLA),
        "MLP": MLPClassifier(random_state=SEMILLA),
    }


def fmt(m, s):
    return f"{100*m:.2f} ± {100*s:.2f}"


# ---------------------------------------------------------------------------
# 0) Descripción de datasets
# ---------------------------------------------------------------------------
def figura_datasets():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    info = {}
    for a, nombre in zip(ax, DATASETS):
        X, y, rasgos = cargar(nombre)
        if nombre == "Sintético":
            # Rasgos informativos x0 vs x1 (los irrelevantes x4, x5 dominarían el PCA)
            Z = X[:, :2]; ejes = ("x0", "x1")
        else:
            Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X)); ejes = ("PC1", "PC2")
        for c in np.unique(y):
            a.scatter(Z[y == c, 0], Z[y == c, 1], s=14, alpha=.75, color=COL[c],
                      edgecolor="white", linewidth=.4, label=f"Clase {c}")
        sufijo = "rasgos x0 vs x1" if nombre == "Sintético" else "PCA estandarizado"
        a.set_title(f"{nombre} ({X.shape[0]} × {X.shape[1]}): {sufijo}",
                    loc="left", fontsize=11, color=TXT)
        a.set_xlabel(ejes[0]); a.set_ylabel(ejes[1])
        a.legend(loc="best", fontsize=8)
        info[nombre] = {"n": int(X.shape[0]), "d": int(X.shape[1]),
                        "por_clase": np.bincount(y).tolist(), "rasgos": rasgos}
    fig.suptitle("Vista 2D de los datasets", x=0.01, ha="left",
                 fontsize=12, color=TXT)
    fig.savefig(os.path.join(FIG, "01_datasets_pca.png"))
    plt.close(fig)

    # Info del ruido del sintético
    _, _, _, ruido = dataset_sintetico(devolver_info=True)
    info["Sintético"]["etiquetas_invertidas"] = int(len(ruido["idx_flip"]))
    info["Sintético"]["outliers"] = int(len(ruido["idx_out"]))
    return info


# ---------------------------------------------------------------------------
# 1) Comparativa con parámetros por defecto
# ---------------------------------------------------------------------------
def comparativa_default(rep_holdout):
    filas, matrices = [], {}
    for nombre in DATASETS:
        X, y, _ = cargar(nombre)
        clases = np.unique(y)
        for mod_nombre, modelo in modelos_default().items():
            for val, fn in [("Hold-Out 70/30", lambda m: holdout(m, X, y, rep_holdout, SEMILLA)),
                            ("10-Fold CV", lambda m: kfold(m, X, y, 10, SEMILLA))]:
                r = resumir(fn(modelo), clases)
                filas.append({"Dataset": nombre, "Modelo": mod_nombre, "Validación": val,
                              "Accuracy": r["acc_media"], "Acc_std": r["acc_std"],
                              "F1_macro": r["f1_media"], "F1_std": r["f1_std"],
                              "t_fit_ms": r["t_fit_ms"], "t_pred_ms": r["t_pred_ms"]})
                if val == "10-Fold CV":
                    matrices[(nombre, mod_nombre)] = r["matriz_confusion"]
            print(f"  {nombre:10s} {mod_nombre:18s} listo")
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RES, "tabla_default.csv"), index=False)

    # Figura: barras accuracy
    modelos = list(modelos_default())
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, nombre in zip(axs, DATASETS):
        x = np.arange(len(modelos))
        for j, val in enumerate(["Hold-Out 70/30", "10-Fold CV"]):
            sub = df[(df.Dataset == nombre) & (df["Validación"] == val)].set_index("Modelo").loc[modelos]
            a.errorbar(x + (j - .5) * 0.28, 100 * sub.Accuracy, yerr=100 * sub.Acc_std, fmt="o",
                       ms=7, color=COL[j], ecolor=COL[j], elinewidth=1.6, capsize=0,
                       markeredgecolor="white", markeredgewidth=1.2, label=val)
        a.set_xticks(x, modelos, rotation=25, ha="right")
        a.set_title(nombre, loc="left", fontsize=11, color=TXT)
        a.grid(axis="x", visible=False)
    axs[0].set_ylabel("Accuracy (%)  media ± desv. est.")
    axs[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Comparativa con parámetros por defecto", x=0.01, ha="left", fontsize=12, color=TXT)
    fig.savefig(os.path.join(FIG, "02_comparativa_default.png"))
    plt.close(fig)

    # Figura: matrices de confusión (10-Fold, acumuladas)
    fig, axs = plt.subplots(2, 6, figsize=(15, 5.4))
    for i, nombre in enumerate(DATASETS):
        for j, mod in enumerate(modelos):
            M = matrices[(nombre, mod)]
            a = axs[i, j]
            a.imshow(M, cmap="Blues")
            for (r, c), v in np.ndenumerate(M):
                a.text(c, r, v, ha="center", va="center", fontsize=9,
                       color="white" if v > M.max() * .6 else TXT)
            a.set_xticks(range(3)); a.set_yticks(range(3)); a.grid(False)
            if i == 0: a.set_title(mod, fontsize=9.5, color=TXT)
            if j == 0: a.set_ylabel(f"{nombre}\nreal")
            if i == 1: a.set_xlabel("predicha")
    fig.suptitle("Matrices de confusión acumuladas en 10-Fold CV", x=0.01, ha="left", fontsize=12, color=TXT)
    fig.savefig(os.path.join(FIG, "03_matrices_confusion.png"))
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 2) Pruebas con parámetros distintos a los de default
# ---------------------------------------------------------------------------
def eval_cv(modelo, X, y, escalar=True):
    r = resumir(kfold(modelo, X, y, 10, SEMILLA, escalar), np.unique(y))
    return r["acc_media"], r["acc_std"], r["f1_media"]


def pruebas_knn():
    ks = list(range(1, 42, 2))
    variantes = [("euclidiana", "uniforme"), ("euclidiana", "distancia"), ("manhattan", "uniforme")]
    filas = []
    for nombre in DATASETS:
        X, y, _ = cargar(nombre)
        for met, pes in variantes:
            for k in ks:
                acc, sd, f1 = eval_cv(KNN(k=k, metrica=met, pesos=pes), X, y)
                filas.append({"Dataset": nombre, "k": k, "Métrica": met, "Pesos": pes,
                              "Accuracy": acc, "Acc_std": sd, "F1_macro": f1})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RES, "tabla_knn.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for a, nombre in zip(axs, DATASETS):
        for i, (met, pes) in enumerate(variantes):
            s = df[(df.Dataset == nombre) & (df["Métrica"] == met) & (df.Pesos == pes)]
            a.plot(s.k, 100 * s.Accuracy, color=COL[i], lw=2, marker="o", ms=3.5,
                   label=f"{met}, {pes}")
        a.axvline(5, color=TXT2, lw=.8, ls="--")
        a.text(5.4, a.get_ylim()[0] + .3, "default k=5", fontsize=8, color=TXT2)
        a.set_title(nombre, loc="left", fontsize=11, color=TXT)
        a.set_xlabel("k (número de vecinos)")
    axs[0].set_ylabel("Accuracy 10-Fold (%)")
    axs[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("K-NN: efecto de k, la métrica y la ponderación", x=0.01, ha="left", fontsize=12, color=TXT)
    fig.savefig(os.path.join(FIG, "04_knn_k.png"))
    plt.close(fig)
    return df


def pruebas_svm():
    Cs = [0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 10]
    filas = []
    for nombre in DATASETS:
        X, y, _ = cargar(nombre)
        for C in Cs:
            for g in gammas:
                acc, sd, f1 = eval_cv(SVC(C=C, gamma=g, kernel="rbf"), X, y)
                filas.append({"Dataset": nombre, "Kernel": "rbf", "C": C, "gamma": g, "grado": None,
                              "Accuracy": acc, "Acc_std": sd, "F1_macro": f1})
        for C in [0.1, 1, 10]:
            acc, sd, f1 = eval_cv(SVC(C=C, kernel="linear"), X, y)
            filas.append({"Dataset": nombre, "Kernel": "linear", "C": C, "gamma": None, "grado": None,
                          "Accuracy": acc, "Acc_std": sd, "F1_macro": f1})
            for deg in [2, 3]:
                acc, sd, f1 = eval_cv(SVC(C=C, kernel="poly", degree=deg, coef0=1), X, y)
                filas.append({"Dataset": nombre, "Kernel": "poly", "C": C, "gamma": "scale", "grado": deg,
                              "Accuracy": acc, "Acc_std": sd, "F1_macro": f1})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RES, "tabla_svm.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for a, nombre in zip(axs, DATASETS):
        s = df[(df.Dataset == nombre) & (df.Kernel == "rbf")]
        M = s.pivot(index="C", columns="gamma", values="Accuracy").loc[Cs, gammas] * 100
        im = a.imshow(M.values, cmap="Blues", vmin=30, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(M.values):
            a.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                   color="white" if v > 80 else TXT)
        a.set_xticks(range(len(gammas)), gammas); a.set_yticks(range(len(Cs)), Cs)
        a.set_xlabel("gamma"); a.grid(False)
        if nombre == DATASETS[0]:
            a.set_ylabel("C")
        a.set_title(f"{nombre} (kernel RBF)", loc="left", fontsize=11, color=TXT)
    fig.colorbar(im, ax=axs, shrink=.85, label="Accuracy 10-Fold (%)")
    fig.suptitle("SVM: rejilla C × gamma", x=0.01, ha="left", fontsize=12, color=TXT)
    fig.savefig(os.path.join(FIG, "05_svm_rejilla.png"))
    plt.close(fig)
    return df


CONFIGS_MLP = [
    ("Default: (100,), relu, adam, α=1e-4, 200 iter", {}),
    ("(100,), relu, adam, α=1e-4, 2000 iter", {"max_iter": 2000}),
    ("(10,), relu, 2000 iter", {"hidden_layer_sizes": (10,), "max_iter": 2000}),
    ("(50,25), relu, 2000 iter", {"hidden_layer_sizes": (50, 25), "max_iter": 2000}),
    ("(100,50), tanh, 2000 iter", {"hidden_layer_sizes": (100, 50), "activation": "tanh", "max_iter": 2000}),
    ("(100,), relu, α=1 (más regularización)", {"alpha": 1.0, "max_iter": 2000}),
    ("(100,), relu, α=0.01, early stopping", {"alpha": 0.01, "early_stopping": True, "max_iter": 2000}),
    ("(100,), sgd, lr=0.01, 2000 iter", {"solver": "sgd", "learning_rate_init": 0.01, "max_iter": 2000}),
    ("(100,), lbfgs, α=0.1", {"solver": "lbfgs", "alpha": 0.1, "max_iter": 2000}),
]


def pruebas_mlp():
    filas = []
    for nombre in DATASETS:
        X, y, _ = cargar(nombre)
        for desc, p in CONFIGS_MLP:
            m = MLPClassifier(random_state=SEMILLA, **p)
            acc, sd, f1 = eval_cv(m, X, y)
            filas.append({"Dataset": nombre, "Configuración": desc, "Accuracy": acc,
                          "Acc_std": sd, "F1_macro": f1})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RES, "tabla_mlp.csv"), index=False)
    return df


def pruebas_escalado():
    """¿Qué pasa si NO se estandariza? (afecta a los métodos basados en distancia)."""
    filas = []
    for nombre in DATASETS:
        X, y, _ = cargar(nombre)
        for mod_nombre, m in modelos_default().items():
            con = eval_cv(m, X, y, True)[0]
            sin = eval_cv(m, X, y, False)[0]
            filas.append({"Dataset": nombre, "Modelo": mod_nombre,
                          "Con estandarización": con, "Sin estandarización": sin})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RES, "tabla_escalado.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 3) Búsqueda de hiperparámetros: Grid Search con validación cruzada ANIDADA
# ---------------------------------------------------------------------------
REJILLAS = {
    "K-NN": (KNN(), {"modelo__k": [1, 3, 5, 7, 9, 11, 15, 21, 31],
                     "modelo__metrica": ["euclidiana", "manhattan"],
                     "modelo__pesos": ["uniforme", "distancia"]}),
    "SVM": (SVC(random_state=SEMILLA), [
        {"modelo__kernel": ["rbf"], "modelo__C": [0.1, 1, 10, 100],
         "modelo__gamma": ["scale", 0.01, 0.1, 1]},
        {"modelo__kernel": ["linear"], "modelo__C": [0.1, 1, 10]},
        {"modelo__kernel": ["poly"], "modelo__C": [0.1, 1, 10], "modelo__degree": [2, 3],
         "modelo__coef0": [1]}]),
    "MLP": (MLPClassifier(random_state=SEMILLA, max_iter=1000), {
        "modelo__hidden_layer_sizes": [(20,), (100,), (50, 25)],
        "modelo__activation": ["relu", "tanh"],
        "modelo__alpha": [1e-4, 1e-2, 1.0]}),
}


def cv_anidada(k_ext=5, k_int=5):
    """Lazo externo (k_ext pliegues): estima el desempeño de TODO el procedimiento.
    Lazo interno (GridSearchCV, k_int pliegues): elige los hiperparámetros usando
    solo el entrenamiento del pliegue externo. Así el número reportado no está
    sesgado por haber elegido los parámetros mirando los datos de prueba."""
    filas = []
    for nombre in DATASETS:
        X, y, _ = cargar(nombre)
        for mod_nombre, (modelo, rejilla) in REJILLAS.items():
            accs, f1s, elegidos = [], [], []
            t0 = time.perf_counter()
            for ent, pru in particiones_kfold(y, k_ext, SEMILLA):
                pipe = Pipeline([("escalado", StandardScaler()), ("modelo", modelo)])
                gs = GridSearchCV(pipe, rejilla, cv=StratifiedKFold(k_int, shuffle=True, random_state=1),
                                  scoring="accuracy", n_jobs=-1)
                gs.fit(X[ent], y[ent])
                yp = gs.predict(X[pru])
                accs.append(accuracy_score(y[pru], yp))
                f1s.append(f1_score(y[pru], yp, average="macro"))
                elegidos.append(str({k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}))
            mas_comun, veces = Counter(elegidos).most_common(1)[0]
            filas.append({"Dataset": nombre, "Modelo": mod_nombre,
                          "Accuracy anidada": np.mean(accs), "Acc_std": np.std(accs, ddof=1),
                          "F1_macro": np.mean(f1s),
                          "Parámetros más elegidos": f"{mas_comun} ({veces}/{k_ext})",
                          "Tiempo_s": time.perf_counter() - t0})
            print(f"  {nombre:10s} {mod_nombre:6s} anidada {100*np.mean(accs):.2f}%  {mas_comun}")
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RES, "tabla_cv_anidada.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 4) Complejidad temporal empírica
# ---------------------------------------------------------------------------
def complejidad(tamanos, reps=3):
    filas = []
    for npc in tamanos:
        X, y, _ = dataset_sintetico(n_por_clase=npc, semilla=123)
        X = StandardScaler().fit_transform(X)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y)); corte = int(.7 * len(y))
        ent, pru = idx[:corte], idx[corte:]
        for mod_nombre, m in modelos_default().items():
            tf, tp = [], []
            for _ in range(reps):
                t0 = time.perf_counter(); m.fit(X[ent], y[ent]); tf.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); m.predict(X[pru]); tp.append(time.perf_counter() - t0)
            filas.append({"Modelo": mod_nombre, "n_total": len(y), "n_ent": len(ent), "m_prueba": len(pru),
                          "t_fit_s": np.median(tf), "t_pred_s": np.median(tp)})
        print(f"  n={len(y)} listo")
    df = pd.DataFrame(filas)

    # Pendiente en log-log (exponente empírico) usando los 4 tamaños mayores
    pend = []
    for mod in df.Modelo.unique():
        s = df[df.Modelo == mod].sort_values("n_total").tail(4)
        bf = np.polyfit(np.log(s.n_ent), np.log(s.t_fit_s), 1)[0]
        bp = np.polyfit(np.log(s.n_ent), np.log(s.t_pred_s), 1)[0]
        pend.append({"Modelo": mod, "exponente_fit": bf, "exponente_pred": bp})
    dp = pd.DataFrame(pend)
    df.to_csv(os.path.join(RES, "tabla_complejidad.csv"), index=False)
    dp.to_csv(os.path.join(RES, "tabla_exponentes.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for a, col, tit in [(axs[0], "t_fit_s", "Entrenamiento"), (axs[1], "t_pred_s", "Predicción (30 % de prueba)")]:
        for i, mod in enumerate(modelos_default()):
            s = df[df.Modelo == mod]
            a.plot(s.n_total, s[col] * 1000, color=COL[i], lw=2, marker="o", ms=4, label=mod)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(tit, loc="left", fontsize=11, color=TXT)
        a.set_xlabel("n (tamaño total del dataset, escala log)")
    axs[0].set_ylabel("tiempo (ms, escala log)")
    axs[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Complejidad temporal empírica (dataset sintético, d = 6)", x=0.01, ha="left",
                 fontsize=12, color=TXT)
    fig.savefig(os.path.join(FIG, "06_complejidad.png"))
    plt.close(fig)
    return df, dp


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    rep = 3 if args.rapido else 20
    tamanos = [200, 400, 800] if args.rapido else [200, 400, 800, 1600, 3200, 6400]

    t0 = time.perf_counter()
    print("0) Datasets"); info = figura_datasets()
    print("1) Comparativa default"); d1 = comparativa_default(rep)
    print("2) K-NN"); pruebas_knn()
    print("2) SVM"); pruebas_svm()
    print("2) MLP"); pruebas_mlp()
    print("2) Escalado"); pruebas_escalado()
    print("3) CV anidada"); cv_anidada()
    print("4) Complejidad"); complejidad(tamanos)

    resumen = {"datasets": info, "repeticiones_holdout": rep,
               "tiempo_total_s": time.perf_counter() - t0}
    with open(os.path.join(RES, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"Listo en {resumen['tiempo_total_s']:.1f} s")


if __name__ == "__main__":
    main()
