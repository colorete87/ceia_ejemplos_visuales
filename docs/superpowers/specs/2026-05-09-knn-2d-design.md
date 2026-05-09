# Diseño: Visualizador interactivo de kNN en 2D

Fecha: 2026-05-09

Tercer ejemplo del repo (junto a `proceso_gausiano/` y `proceso_gausiano_2d/`).
Demo interactivo del clasificador k-NN en `R²` con visualización en vivo de la
región de decisión, los vecinos seleccionados y un historial de scores.

## Estructura del proyecto

Nueva carpeta autocontenida con su propio `pyproject.toml` (uv):

```
knn_2d/
├── README.md
├── knn_interactive.py
├── pyproject.toml
└── uv.lock
```

Dependencias: `numpy`, `matplotlib`. Sin sklearn (se implementa kNN a mano,
matchea el patrón de los otros ejemplos del repo).

Actualizar `README.md` raíz para listar el nuevo ejemplo.

## Modelo de datos

```python
state = {
    "X_train": np.ndarray,      # (n_train, 2)
    "y_train": np.ndarray,      # (n_train,)  enteros 0..n_classes-1
    "X_test":  np.ndarray,      # (n_test, 2)
    "y_test":  np.ndarray,      # (n_test,)
    "y_test_pred": np.ndarray | None,  # set por "Test completo"
    "n_classes": int,           # slider 2-6
    "n_train": int,             # slider 10-500
    "n_test": int,              # slider 5-200
    "k": int,                   # slider 1-25
    "metric": str,              # "Euclidiana" | "Manhattan" | "Minkowski" | "Mahalanobis"
    "minkowski_p": float,       # slider 1.0-5.0, default 3.0
    "weights": str,             # "uniform" | "1/d" | "gaussiano"
    "gaussian_h": float,        # slider 0.1-3.0, default 1.0
    "show_decision": bool,      # checkbox, default True
    "show_test": bool,          # checkbox, default True
    "scores": list[dict],       # historial: {"k", "metric", ...,"type", "score"}
    "last_query": tuple | None, # (x, y, pred_class, neighbor_indices)
}
```

Datos generados al iniciar y al apretar "Regenerar datos":
`make_random_dataset(n_classes, n_train, n_test, rng)` → cada clase es un
cluster gaussiano 2D con media uniforme en `[-3.5, 3.5]²`, separación mínima
entre medias para que las clases se distingan, y covarianza con leve
correlación. Split equitativo (`n_train // n_classes` por clase, con leve
desbalance del resto).

## Componentes

### Distancias (`_dist_*`)

- `_dist_euclidean(X1, X2)` → `(n1, n2)` matriz de distancias.
- `_dist_manhattan(X1, X2)` → suma de |·|.
- `_dist_minkowski(X1, X2, p)` → `(Σ |x-y|^p)^(1/p)`.
- `_dist_mahalanobis(X1, X2, VI)` → `sqrt((x-y)^T VI (x-y))`. `VI` se
  precalcula como `inv(cov(X_train))` cada vez que se regenera el dataset.

### Pesos (`_w_*`)

- `_w_uniform(d)` → `1.0` para todos.
- `_w_inv_dist(d, eps=1e-9)` → `1 / (d + eps)`.
- `_w_gaussian(d, h)` → `exp(-d² / (2 h²))`.

### Predicción

```python
def predict_knn(X_query, X_train, y_train, n_classes, k, metric_fn, weight_fn):
    """
    Devuelve:
      pred:        (n_query,) clase predicha (entero 0..n_classes-1)
      confidence:  (n_query,) peso del voto ganador / peso total
      neighbors:   (n_query, k) índices en X_train de los k vecinos
    """
```

Implementación: calcula distancias, toma `argpartition` para los k menores,
suma pesos por clase, devuelve `argmax`. Numpy puro.

## Layout

Patrón de los otros ejemplos: panel izquierdo con grupos de widgets, plot
central, panel inferior. Suma un panel derecho angosto para la lista de
scores.

Figura `figsize=(13, 7.5)` (un poco más ancha que las anteriores para
acomodar la columna de scores).

### Panel izquierdo (recuadros, de arriba abajo)

- **Vista**: `[x] Mostrar región de decisión` `[x] Mostrar test points`
- **Evaluación**: 3 botones
  - `Eval. punto (click)`  — habilita modo: el próximo click sobre el plot
    evalúa esa coordenada y dibuja vecinos. (También se puede saltar el
    botón: cualquier click siempre evalúa.)
  - `Test completo`        — clasifica todo `X_test`, agrega a scores.
  - `K-fold CV (5)`        — corre 5-fold sobre el train, agrega a scores.
- **Control**: `Regenerar datos`, `Reset todo`

### Panel derecho (lista de scores)

Texto monospace de hasta ~12 entradas (las más recientes arriba):

```
[5]  test    k=3 eucl uni  acc=0.870
[4]  kfold   k=5 mink p=3  acc=0.831
[3]  test    k=3 eucl uni  acc=0.860
[2]  kfold   k=7 mahal     acc=0.792
[1]  test    k=7 mahal     acc=0.812
```

`type ∈ {test, kfold}`. Click no se loguea (es exploración).

### Plot central

- Train points: círculos sólidos coloreados por clase (paleta de hasta 6
  colores distintos: tab10 truncado).
- Test points: círculos huecos del color de su clase verdadera (visibles si
  el toggle está ON).
- Tras "Test completo": borde verde (acertó) o rojo (erró) en cada test.
- Región de decisión: shading tenue (alpha ≈ 0.15) sobre grilla 60×60. Se
  recomputa al cambiar k, métrica, weights o el dataset.
- Última query (post-click): X grande negra + líneas finas a los k vecinos
  + caption con la clase predicha y la confianza.

### Panel inferior (sliders + recuadros)

Sliders sueltos, en una fila:
- `# clases` (2, 6, default 3)
- `# train`  (10, 500, default 150)
- `# test`   (5, 200, default 50)
- `k`        (1, 25, default 5)

Recuadros lado a lado:
- **Métrica**: radio 4 opciones + slider `p` (Minkowski, 1.0-5.0).
- **Pesos**: radio 3 opciones + slider `h` (Gaussiano, 0.1-3.0).

Los sliders dependientes (`p`, `h`) usan `Slider.set_active(False)` cuando
su métrica/peso no está seleccionado (queda visible pero no responde a
interacción). Su valor último queda guardado y se reusa cuando vuelve a
activarse esa métrica.

## Reglas de invalidación

- **Cualquier slider o radio** que afecte la frontera de decisión (k, métrica,
  p, weights, h) → recomputa la región de decisión y la query del último
  click si existe. **No** loguea score.
- **`# clases`, `# train`, `# test`, "Regenerar datos"** → regenera dataset,
  borra `last_query`, borra `y_test_pred`, borra `scores`.
- **`Reset todo`** → mismo efecto que regenerar + resetea sliders a defaults.
- Toggles de vista no recomputan datos, sólo afectan visibilidad.

## Casos límite

- `k > n_train`: clamp a `n_train`.
- `n_test = 0`: deshabilitar "Test completo" (botón gris).
- Mahalanobis con `n_train < 3`: `cov` es singular; mostrar nota o caer a
  `VI = I` (identidad) con un warning silencioso.
- Empate de votos en kNN: criterio determinista (gana la clase con índice
  menor entre las empatadas). Documentar.

## Tests / verificación

Smoke test al final del desarrollo (script aparte o inline):
- Render con datos de defaults → exporta PNG, verificar visualmente.
- Probar los 3 botones de evaluación, los 4 radios de métrica, los 3 de
  weights → no crashea.
- Verificar que `predict_knn` con `k=1` y un train point coincidente da
  ese mismo punto.
- Verificar que clamp de `k` funciona si bajás `n_train` por debajo de `k`.

## Lo que NO incluye (out of scope)

- Mahalanobis per-clase.
- Cross-validation con número de folds configurable (queda fijo en 5).
- Exportar resultados a CSV.
- Animaciones.
- Búsqueda de hiperparámetros automática.
