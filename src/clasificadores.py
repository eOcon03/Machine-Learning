"""
Clasificadores implementados DESDE CERO (solo NumPy):
    - Distancia Mínima (centroide más cercano)
    - 1-NN
    - K-NN

Se heredan BaseEstimator y ClassifierMixin de scikit-learn únicamente para que
las clases sean compatibles con GridSearchCV / Pipeline (get_params, set_params,
score). Toda la lógica de entrenamiento y predicción está escrita a mano.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def distancias(A, B, metrica="euclidiana"):
    """Matriz de distancias entre las filas de A (m x d) y B (n x d) -> (m x n).

    euclidiana: ||a-b||_2 usando ||a||^2 + ||b||^2 - 2 a.b  (vectorizado, O(m n d))
    manhattan : sum |a-b|                                   (O(m n d))
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if metrica == "euclidiana":
        a2 = np.sum(A * A, axis=1)[:, None]
        b2 = np.sum(B * B, axis=1)[None, :]
        d2 = a2 + b2 - 2.0 * A @ B.T
        np.maximum(d2, 0.0, out=d2)          # evita negativos por redondeo
        return np.sqrt(d2)
    if metrica == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    raise ValueError(f"Métrica desconocida: {metrica}")


# ----------------------------------------------------------------------------
# 1) Distancia Mínima
# ----------------------------------------------------------------------------
class DistanciaMinima(BaseEstimator, ClassifierMixin):
    """Clasificador de distancia mínima (nearest centroid).

    Entrenamiento: calcula el vector medio (centroide) de cada clase.
        Complejidad: O(n d)
    Predicción: asigna la clase del centroide más cercano.
        Complejidad: O(m c d)   (m muestras de prueba, c clases)
    """

    def __init__(self, metrica="euclidiana"):
        self.metrica = metrica

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroides_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        D = distancias(X, self.centroides_, self.metrica)
        return self.classes_[np.argmin(D, axis=1)]


# ----------------------------------------------------------------------------
# 2) y 3) K-NN (1-NN es el caso k = 1)
# ----------------------------------------------------------------------------
class KNN(BaseEstimator, ClassifierMixin):
    """K vecinos más cercanos (fuerza bruta).

    Entrenamiento: solo memoriza X, y ("lazy learner").  O(1) / O(n d) si se copia.
    Predicción, por cada muestra de prueba:
        - distancias a las n muestras de entrenamiento:  O(n d)
        - selección de los k menores (argpartition):     O(n)
        - votación:                                      O(k)
        Total: O(m n d)  para m muestras de prueba.

    Parámetros
    ----------
    k        : número de vecinos.
    metrica  : 'euclidiana' | 'manhattan'
    pesos    : 'uniforme'  -> cada vecino vale 1
               'distancia' -> cada vecino vale 1/d (vecinos cercanos pesan más)
    bloque   : tamaño de lote para no construir matrices de distancias enormes.
    """

    def __init__(self, k=5, metrica="euclidiana", pesos="uniforme", bloque=1024):
        self.k = k
        self.metrica = metrica
        self.pesos = pesos
        self.bloque = bloque

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, self.y_idx_ = np.unique(y, return_inverse=True)
        return self

    def _predict_bloque(self, Xb):
        n = self.X_.shape[0]
        k = min(self.k, n)
        D = distancias(Xb, self.X_, self.metrica)                  # (b x n)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]             # k vecinos (sin ordenar)
        d_k = np.take_along_axis(D, idx, axis=1)
        etiquetas = self.y_idx_[idx]                               # (b x k)

        if self.pesos == "distancia":
            w = 1.0 / np.maximum(d_k, 1e-12)
        else:
            w = np.ones_like(d_k)

        c = len(self.classes_)
        votos = np.zeros((Xb.shape[0], c))
        for j in range(c):
            votos[:, j] = np.sum(w * (etiquetas == j), axis=1)

        # Desempate: entre las clases empatadas gana la del vecino más cercano
        max_v = votos.max(axis=1, keepdims=True)
        empatadas = np.isclose(votos, max_v)
        pred = np.argmax(votos, axis=1)
        filas_empate = np.where(empatadas.sum(axis=1) > 1)[0]
        if filas_empate.size:
            orden = np.argsort(d_k[filas_empate], axis=1)
            for r, fila in enumerate(filas_empate):
                for pos in orden[r]:
                    cl = etiquetas[fila, pos]
                    if empatadas[fila, cl]:
                        pred[fila] = cl
                        break
        return pred

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        salida = np.empty(X.shape[0], dtype=int)
        for i in range(0, X.shape[0], self.bloque):
            salida[i:i + self.bloque] = self._predict_bloque(X[i:i + self.bloque])
        return self.classes_[salida]


class UnoNN(KNN):
    """1-NN: caso particular de K-NN con k = 1."""

    def __init__(self, metrica="euclidiana", bloque=1024):
        super().__init__(k=1, metrica=metrica, pesos="uniforme", bloque=bloque)

    def get_params(self, deep=True):
        return {"metrica": self.metrica, "bloque": self.bloque}
