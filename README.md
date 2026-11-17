# Práctica 3 — Comparativa de clasificadores (Machine Learning)

Comparativa de seis clasificadores sobre dos datasets con dos métodos de validación.

| Clasificador | Implementación |
|---|---|
| Distancia Mínima | desde cero (`src/clasificadores.py`) |
| 1-NN | desde cero (`src/clasificadores.py`) |
| K-NN | desde cero (`src/clasificadores.py`) |
| Naïve Bayes (Gaussiano) | scikit-learn `GaussianNB` |
| SVM | scikit-learn `SVC` |
| Perceptrón Multicapa | scikit-learn `MLPClassifier` |

**Datasets**
- **Sintético con ruido estructurado** (`src/datasets.py`): 600 muestras, 6 rasgos, 3 clases. Clase 2 bimodal, covarianzas anisotrópicas, ruido heterocedástico, 2 rasgos irrelevantes correlacionados, ruido de etiqueta concentrado en la frontera (~10 %) y 3 % de outliers direccionales.
- **Wine (UCI)**: 178 muestras, 13 rasgos numéricos reales, 3 clases.

**Validación** (`src/validacion.py`, implementada a mano y estratificada)
- Hold-Out 70/30 (repetido 20 veces con semillas distintas → media ± desv. est.)
- 10-Fold Cross-Validation

Todos los modelos usan estandarización z-score ajustada solo con el conjunto de entrenamiento de cada partición.

## Ejecutar

```bash
pip install -r requirements.txt
python src/experimentos.py           # completo (~6 min en 2 núcleos)
python src/experimentos.py --rapido  # versión corta para probar
```

## Salidas (`resultados/`)

| Archivo | Contenido |
|---|---|
| `tabla_default.csv` | Accuracy, F1-macro y tiempos con parámetros por defecto (ambas validaciones) |
| `tabla_knn.csv` | K-NN con k = 1…41, métrica euclidiana/manhattan, pesos uniforme/distancia |
| `tabla_svm.csv` | SVM: rejilla C × gamma (RBF), kernel lineal y polinomial |
| `tabla_mlp.csv` | MLP con 9 configuraciones |
| `tabla_escalado.csv` | Efecto de estandarizar vs no estandarizar |
| `tabla_cv_anidada.csv` | Grid Search con validación cruzada anidada (5 × 5) |
| `tabla_complejidad.csv`, `tabla_exponentes.csv` | Tiempos vs n y exponente empírico (pendiente log-log) |
| `figuras/*.png` | Gráficas usadas en el reporte |

## Estructura

```
src/
  clasificadores.py   # Distancia Mínima, 1-NN, K-NN (NumPy)
  datasets.py         # generador sintético + Wine
  validacion.py       # Hold-Out y K-Fold estratificados
  experimentos.py     # corre todo y genera tablas y figuras
resultados/
```
