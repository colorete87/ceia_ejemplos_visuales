# Diseño — `mmc_2d/` (Maximal Margin Classifier 2D)

Demo interactivo de Maximal Margin Classifier en 2D, con extensión a Support
Vector Machine de margen blando. Cuarto ejemplo de la colección, autocontenido y
con el mismo patrón estructural que `knn_2d/`.

## Objetivos didácticos

- Visualizar el hiperplano `β₀ + β₁·x₁ + β₂·x₂ = 0` y su margen geométrico.
- Permitir al alumno mover los parámetros del hiperplano a mano (sliders) y ver
  cómo cambia el margen en tiempo real.
- Mostrar el margen óptimo calculado automáticamente (Hard margin) y la
  variante con regularización (Soft margin / SVM).
- Mostrar, todo el tiempo, los support vectors y las flechas a su pie sobre el
  hiperplano, así como el ancho del corredor del margen.
- Permitir editar el dataset a mano (agregar, mover, borrar puntos) para
  explorar el efecto de los datos sobre la solución óptima.

## Estructura del proyecto

```
mmc_2d/
  mmc_interactive.py
  pyproject.toml          # numpy + matplotlib + scipy
  README.md
```

Subproyecto autocontenido siguiendo la convención del repo: nada compartido con
los otros ejemplos. Comentarios y UI en español. Mismo skeleton de
`knn_2d/knn_interactive.py` (funciones puras al tope, estado global, una sola
función `redraw()`, handlers que mutan estado y disparan `redraw()`).

## Convenciones internas

- Etiquetas: y ∈ {−1, +1} a nivel modelo. En la UI se llaman "Clase 0" (y=−1,
  color azul `#1f77b4`) y "Clase 1" (y=+1, color naranja `#ff7f0e`),
  consistente con `CLASS_COLORS[0:2]` de `knn_2d`.
- Parametrización: `β₀ + β₁·x₁ + β₂·x₂ = 0` (notación ISL).

## Modelo (funciones puras al tope del archivo)

- `decision_value(X, b0, b1, b2)` → vector `β₀ + β₁·x₁ + β₂·x₂`.
- `signed_margin(X, y, b0, b1, b2)` → `yᵢ·(β·xᵢ) / ‖(β₁, β₂)‖₂`. Margen
  geométrico firmado por punto.
- `margin_width(b1, b2)` → `2 / ‖(β₁, β₂)‖₂`. Ancho total del corredor cuando
  los SV están en `±1`.
- `fit_hard_margin(X, y)` → resuelve

  ```
  min  ½·(β₁² + β₂²)
  s.t. yᵢ·(β₀ + β₁·xᵢ₁ + β₂·xᵢ₂) ≥ 1   ∀i
  ```

  con `scipy.optimize.minimize(method="SLSQP")`. Variables = `(β₀, β₁, β₂)`.
  Devuelve `{b0, b1, b2}` o `None` si SLSQP no converge o si las constraints
  quedan violadas — proxy de infactibilidad / datos no separables.

- `fit_soft_margin(X, y, C)` → resuelve

  ```
  min  ½·(β₁² + β₂²) + C·Σᵢ ξᵢ
  s.t. yᵢ·(β₀ + β₁·xᵢ₁ + β₂·xᵢ₂) ≥ 1 − ξᵢ
       ξᵢ ≥ 0
  ```

  con SLSQP. Variables = `(β₀, β₁, β₂, ξ₁, …, ξₙ)`. Siempre devuelve solución.

- `make_random_dataset(n_per_class, rng)` → 2 clusters gaussianos. Medias
  sorteadas con separación mínima (igual idea que `_random_means` en knn_2d);
  covarianzas suaves (`σ ∈ [0.4, 1.0]`, `|ρ| ≤ 0.6`). Devuelve `(X, y)` con `y
  ∈ {−1, +1}`.

- `support_vectors_mask(X, y, b0, b1, b2, classifier, tol=1e-3)` → bool array:
  - en Hard: `|yᵢ·(β·xᵢ) − 1| < tol` (puntos sobre el margen),
  - en Soft: `yᵢ·(β·xᵢ) ≤ 1 + tol` (puntos sobre el margen o dentro).

## Estado global

```python
state = {
    "X": np.empty((0, 2)),
    "y": np.empty(0, dtype=int),     # -1 / +1
    "n_per_class": 25,
    "classifier": "Hard",            # "Hard" | "Soft"
    "C": 1.0,
    "beta0": 0.0, "beta1": 1.0, "beta2": -1.0,
    "auto_solution": None,           # dict {b0,b1,b2} | None  → recta punteada extra
    "auto_infeasible": False,        # True si último fit_hard_margin falló
    "click_mode": "Agregar A",       # "Agregar A" | "Agregar B" | "Mover" | "Borrar"
    "show_decision": False,
    "show_corridor": True,
    "show_arrows": True,
    "drag_idx": None,                # índice del punto siendo arrastrado, o None
}
```

## Layout

`fig = plt.figure(figsize=(13, 7.5))`. Distribución similar a `knn_2d`:

- **Plot principal** centro-izquierda (`xlim/ylim = [-5, 5]`, `aspect="equal"`).
- **Panel "Métricas"** a la derecha, monoespaciado (estilo del panel `Scores` de
  `knn_2d`). Contenido:

  ```
  Margen:    0.842
  ‖β‖:       2.376
  # SV:      4
  Accuracy:  0.96
  Status:    OK
  ```

  `Status` muestra `Ø no separable` cuando el último Auto en modo Hard falló.

- **Grupos de controles** (con `_add_group_box`):
  - **"Hiperplano"** (izq, arriba): sliders **β₀** (`[-5, 5]`), **β₁**
    (`[-3, 3]`), **β₂** (`[-3, 3]`).
  - **"Clasificador"** (izq, medio): RadioButton **Hard / Soft** + slider **C**
    (`[0.01, 100]`, log-spaced; activo sólo en Soft).
  - **"Datos"** (izq, abajo): slider **# por clase** (5–100), botón
    **Regenerar**.
  - **"Edición"** (abajo, centro): RadioButton **Agregar A / Agregar B / Mover
    / Borrar**.
  - **"Control"** (abajo, derecha): botones **Auto**, **Limpiar óptimo**,
    **Reset todo**.
  - **"Vista"** (esquina sup. izq.): CheckButtons **Región decisión /
    Sombreado margen / Flechas a SV**.

## `redraw()` — orden de capas

Toda la lógica visual centralizada acá; cada handler termina llamando a
`redraw()`.

1. **Región de decisión**: `imshow` sobre grilla 100×100 si `show_decision`.
   Usa el hiperplano del usuario (no el `auto_solution`).
2. **Sombreado del corredor**: polígono entre las dos paralelas `β·x = ±1` del
   usuario si `show_corridor`.
3. **Hiperplano del usuario**: línea sólida + las dos paralelas del margen.
4. **Hiperplano óptimo**: línea punteada (color tenue distinto) si
   `auto_solution is not None`. Sin paralelas del óptimo, sólo la recta central
   para no saturar.
5. **Puntos**: scatter por clase (azul / naranja).
6. **Support vectors**: círculos huecos grandes sobre los SV según
   `support_vectors_mask` con la β del usuario.
7. **Flechas perpendiculares**: desde cada SV hasta su pie en el hiperplano del
   usuario, si `show_arrows`. Implementadas con `ax.annotate(...,
   arrowprops=...)`.
8. **Panel "Métricas"**: margen geométrico, `‖β‖`, número de SV, accuracy y
   `Status`.

Artistas se reusan entre redraws (igual técnica que knn_2d con
`set_offsets`/`set_data`); las flechas y las paralelas, que cambian de
identidad en cada redraw, se almacenan en listas y se limpian con
`_safe_remove`.

## Handlers

- **Sliders β₀ / β₁ / β₂** (`on_changed`): mutan `state` y llaman a `redraw()`.
  No tocan `auto_solution` (la recta punteada queda como referencia).
- **RadioButton Hard / Soft** (`on_clicked`): setea `state["classifier"]`,
  activa/desactiva slider C. Limpia `auto_infeasible` (se recalculará en el
  próximo Auto).
- **Slider C** (`on_changed`): sólo actualiza `state["C"]`. No fuerza redraw
  (no afecta a la visualización actual).
- **Slider # por clase + botón Regenerar**: resortean datos, limpian
  `auto_solution`, redraw.
- **RadioButton modo click** (`on_clicked`): cambia `state["click_mode"]`.
- **Mouse events** (3 conexiones: `button_press_event`, `motion_notify_event`,
  `button_release_event`):
  - `on_press` cuando `event.inaxes == ax`:
    - "Agregar A": agrega punto en `(x, y)` con `y = −1`.
    - "Agregar B": agrega punto en `(x, y)` con `y = +1`.
    - "Mover": busca el punto más cercano a `(x, y)` y guarda su índice en
      `state["drag_idx"]`.
    - "Borrar": elimina el punto más cercano.
  - `on_motion`: si `drag_idx is not None`, actualiza `state["X"][drag_idx]` y
    redraw.
  - `on_release`: `state["drag_idx"] = None`.
- **Botón Auto** (`on_clicked`):
  - Soft: corre `fit_soft_margin(X, y, C)`. Setea los sliders β a la solución
    (sus `on_changed` se encargan de actualizar `state` y redraw). Guarda
    `auto_solution`, `auto_infeasible=False`.
  - Hard: corre `fit_hard_margin(X, y)`. Si devuelve solución: igual que arriba.
    Si devuelve `None`: `auto_infeasible=True`, sliders no se tocan,
    `auto_solution` se queda como estaba.
- **Botón Limpiar óptimo**: `state["auto_solution"] = None; redraw()`.
- **Botón Reset todo**: vuelve `state` a defaults, regenera datos, limpia auto,
  resetea sliders y radios.

## Detalles a preservar

- **Cascada de `set_val` en Auto / Reset**: setear los 3 sliders β dispara 3
  redraws. Se controla con un flag `_suppress_redraw=True` durante el set, y un
  único `redraw()` al final. Igual técnica para Reset.
- **Drag responsive**: durante el drag, los artists del scatter se actualizan
  in-place (mismo enfoque que el `train_scatter.set_offsets` de knn_2d) para
  evitar parpadeos.
- **Sin tests, sin lint** (consistente con el resto del repo).
- **Entry point**: `if __name__ == "__main__": plt.show()` al final del
  archivo.

## Dependencias

`pyproject.toml`:

```toml
[project]
name = "mmc-interactive"
version = "0.1.0"
description = "Demo interactivo de Maximal Margin Classifier (y SVM soft margin) en 2D."
requires-python = ">=3.13"
dependencies = [
    "matplotlib>=3.10.9",
    "numpy>=2.4.4",
    "scipy>=1.14",
]
```

## Documentación

- `mmc_2d/README.md`: descripción de funcionalidades + instrucciones `uv sync`
  / `uv run python mmc_interactive.py`, en el mismo estilo que
  `knn_2d/README.md`.
- README raíz: agregar fila a la tabla de ejemplos.

## Notas y alcance

- **No incluye kernels** (poly, RBF). Estrictamente clasificador lineal en 2D.
- **No incluye multi-clase**. Exactamente 2 clases (esencia del MMC).
- **No incluye k-fold ni accuracy en test separado**. Se reporta accuracy sobre
  el dataset visible (no hay split train/test).
- **No comparte código con los otros demos** — duplicación intencional para
  mantener cada subproyecto autocontenido.
- **TODO.md** del repo raíz: no requiere actualización (este ejemplo no estaba
  listado; sigue siendo un agregado al margen de los ítems pendientes).
