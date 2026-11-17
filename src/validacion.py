"""
Métodos de validación implementados a mano (estratificados):
    - Hold-Out 70/30
    - K-Fold Cross-Validation (k = 10)

Cada modelo se envuelve en un Pipeline con estandarización z-score; el
escalador se ajusta SOLO con los datos de entrenamiento de cada partición para
no filtrar información del conjunto de prueba.
"""
import time
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def particion_holdout(y, prop_prueba=0.30, semilla=0):
    """Índices (entrenamiento, prueba) estratificados: 70 % / 30 % por clase."""
    rng = np.random.default_rng(semilla)
    ent, pru = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_pru = int(round(prop_prueba * len(idx)))
        pru.append(idx[:n_pru])
        ent.append(idx[n_pru:])
    return np.concatenate(ent), np.concatenate(pru)


def particiones_kfold(y, k=10, semilla=0):
    """Genera k pares (entrenamiento, prueba) estratificados."""
    rng = np.random.default_rng(semilla)
    pliegues = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, trozo in enumerate(np.array_split(idx, k)):
            pliegues[i].append(trozo)
    pliegues = [np.concatenate(p) for p in pliegues]
    todos = np.arange(len(y))
    for i in range(k):
        pru = pliegues[i]
        ent = np.setdiff1d(todos, pru)
        yield ent, pru


def con_escalado(modelo, escalar=True):
    return make_pipeline(StandardScaler(), modelo) if escalar else clone(modelo)


def evaluar_particion(modelo, X, y, ent, pru, escalar=True):
    m = con_escalado(modelo, escalar)
    t0 = time.perf_counter()
    m.fit(X[ent], y[ent])
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X[pru])
    t_pred = time.perf_counter() - t0
    return {
        "accuracy": accuracy_score(y[pru], y_pred),
        "f1_macro": f1_score(y[pru], y_pred, average="macro"),
        "t_fit": t_fit,
        "t_pred": t_pred,
        "y_true": y[pru],
        "y_pred": y_pred,
    }


def holdout(modelo, X, y, repeticiones=1, semilla=0, escalar=True):
    """Hold-Out 70/30. Con repeticiones>1 se hace Hold-Out repetido con
    distintas semillas y se reporta media y desviación estándar."""
    res = [evaluar_particion(modelo, X, y, *particion_holdout(y, 0.30, semilla + r), escalar)
           for r in range(repeticiones)]
    return res


def kfold(modelo, X, y, k=10, semilla=0, escalar=True):
    return [evaluar_particion(modelo, X, y, ent, pru, escalar)
            for ent, pru in particiones_kfold(y, k, semilla)]


def resumir(resultados, clases):
    acc = np.array([r["accuracy"] for r in resultados])
    f1 = np.array([r["f1_macro"] for r in resultados])
    yt = np.concatenate([r["y_true"] for r in resultados])
    yp = np.concatenate([r["y_pred"] for r in resultados])
    return {
        "acc_media": acc.mean(), "acc_std": acc.std(ddof=1) if len(acc) > 1 else 0.0,
        "f1_media": f1.mean(), "f1_std": f1.std(ddof=1) if len(f1) > 1 else 0.0,
        "t_fit_ms": 1000 * np.mean([r["t_fit"] for r in resultados]),
        "t_pred_ms": 1000 * np.mean([r["t_pred"] for r in resultados]),
        "matriz_confusion": confusion_matrix(yt, yp, labels=clases),
    }
