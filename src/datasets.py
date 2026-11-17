"""
Datasets de la práctica.

1) Sintético con RUIDO ESTRUCTURADO (generado aquí, reproducible con semilla).
2) Benchmark: Wine (UCI) — 178 muestras, 13 rasgos numéricos reales, 3 clases.
"""
import numpy as np
from sklearn.datasets import load_wine


def dataset_sintetico(n_por_clase=200, semilla=42, devolver_info=False):
    """Genera un problema de 3 clases en 6 dimensiones con ruido estructurado.

    Estructura de la señal (rasgos informativos x0..x3):
      * Clase 0: gaussiana centrada en el origen con covarianza ANISOTRÓPICA
                 y correlacionada (elipse alargada, no esfera).
      * Clase 1: gaussiana con covarianza correlacionada en otra dirección.
      * Clase 2: BIMODAL — dos sub-cúmulos colocados simétricamente a ambos
                 lados de la clase 0. Su centroide cae casi encima del de la
                 clase 0, así que un clasificador por centroides (Distancia
                 Mínima) no puede separarlas: la frontera real no es lineal.

    Ruido ESTRUCTURADO (no es ruido blanco i.i.d.):
      a) Heterocedástico: la desviación del ruido crece con |x0|
         (el ruido depende de la posición en el espacio).
      b) Dos rasgos irrelevantes (x4, x5) fuertemente correlacionados entre sí
         y de gran escala que no aportan información de clase.
      c) Ruido de etiqueta CONCENTRADO EN LA FRONTERA: del 25 % de muestras
         más cercanas a una componente de otra clase, cada una cambia su
         etiqueta a esa clase vecina con probabilidad 0.35 (≈ 10 % del total).
         Es error irreducible y está localizado, no repartido al azar.
      d) Outliers direccionales: 3 % de los puntos se desplazan a lo largo de
         una misma dirección fija (simula una falla sistemática de sensor).
    """
    rng = np.random.default_rng(semilla)
    n = n_por_clase

    # --- señal --------------------------------------------------------------
    cov0 = np.array([[1.2, 0.0, 0.5, 0.0],
                     [0.0, 0.3, 0.0, 0.0],
                     [0.5, 0.0, 0.6, 0.2],
                     [0.0, 0.0, 0.2, 0.4]])
    cov1 = np.array([[0.6, -0.4, 0.0, 0.1],
                     [-0.4, 0.8, 0.0, 0.0],
                     [0.0, 0.0, 0.5, 0.0],
                     [0.1, 0.0, 0.0, 0.5]])
    cov2 = np.array([[0.5, 0.0, 0.0, 0.0],
                     [0.0, 0.5, 0.2, 0.0],
                     [0.0, 0.2, 0.4, 0.0],
                     [0.0, 0.0, 0.0, 0.4]])
    medias = np.array([[0.0, 0.0, 0.0, 0.0],      # clase 0
                       [0.5, 2.2, 1.0, 0.0],      # clase 1
                       [-3.0, 0.4, -0.5, 0.0],    # clase 2, sub-cúmulo a
                       [3.0, 0.4, 0.5, 0.0]])     # clase 2, sub-cúmulo b
    comp_clase = np.array([0, 1, 2, 2])

    mitad = n // 2
    X = np.vstack([rng.multivariate_normal(medias[0], cov0, n),
                   rng.multivariate_normal(medias[1], cov1, n),
                   rng.multivariate_normal(medias[2], cov2, mitad),
                   rng.multivariate_normal(medias[3], cov2, n - mitad)])
    y = np.array([0] * n + [1] * n + [2] * n)

    # --- a) ruido heterocedástico ------------------------------------------
    escala = 0.10 + 0.20 * np.abs(X[:, [0]])
    X = X + rng.normal(size=X.shape) * escala

    # --- c) ruido de etiqueta en la frontera (se calcula con la señal) ------
    d = np.linalg.norm(X[:, None, :] - medias[None], axis=2)          # (N x 4)
    propia = np.where(comp_clase[None, :] == y[:, None], d, np.inf).min(axis=1)
    d_otra = np.where(comp_clase[None, :] != y[:, None], d, np.inf)
    otra_comp = d_otra.argmin(axis=1)
    margen = d_otra.min(axis=1) - propia             # pequeño => cerca de frontera
    frontera = np.argsort(margen)[:int(0.25 * len(y))]
    idx_flip = frontera[rng.random(len(frontera)) < 0.35]
    y_ruido = y.copy()
    y_ruido[idx_flip] = comp_clase[otra_comp[idx_flip]]

    # --- b) rasgos irrelevantes correlacionados ----------------------------
    z = rng.normal(size=(X.shape[0], 1))
    ruido_corr = np.hstack([3.0 * z + 0.4 * rng.normal(size=z.shape),
                            -2.5 * z + 0.4 * rng.normal(size=z.shape)])
    X = np.hstack([X, ruido_corr])

    # --- d) outliers direccionales -----------------------------------------
    n_out = int(0.03 * len(y))
    idx_out = rng.choice(len(y), n_out, replace=False)
    direccion = np.array([1.0, 1.0, -1.0, 0.5, 0.0, 0.0])
    direccion /= np.linalg.norm(direccion)
    X[idx_out] += direccion * rng.uniform(4, 7, size=(n_out, 1))

    nombres = [f"x{i}" for i in range(X.shape[1])]
    if devolver_info:
        info = {"idx_flip": idx_flip, "idx_out": idx_out, "y_limpia": y}
        return X, y_ruido, nombres, info
    return X, y_ruido, nombres


def dataset_wine():
    datos = load_wine()
    return datos.data, datos.target, list(datos.feature_names)


def cargar(nombre):
    if nombre == "Sintético":
        return dataset_sintetico()
    if nombre == "Wine":
        return dataset_wine()
    raise ValueError(nombre)
