# Maximal Margin Classifier (2D) — demo interactivo

Demo didáctico de Maximal Margin Classifier y SVM con margen blando (soft
margin) sobre datos sintéticos 2D.

## Funcionalidades

- Hiperplano `β₀ + β₁·x₁ + β₂·x₂ = 0` controlable con sliders.
- Selector **Hard / Soft margin** con slider `C` (log-spaced) para el soft.
- Botón **Auto**: resuelve el QP correspondiente vía `scipy.optimize.minimize`
  (SLSQP) y mueve los sliders al óptimo. El hiperplano óptimo queda dibujado
  además como línea punteada roja, persistente aunque el usuario vuelva a
  mover los sliders.
- **Edición manual**: agregar puntos de clase A o B, moverlos por drag o
  borrarlos.
- Visualización del margen: dos paralelas, sombreado del corredor, círculos
  sobre los support vectors y flechas perpendiculares al hiperplano.
- Panel "Métricas" a la derecha: ancho del margen, `‖β‖`, número de SV,
  accuracy y `Status` (muestra `Ø no separable` cuando Auto en modo Hard se
  encuentra con datos no separables).
- Toggles de vista: región de decisión, sombreado del margen, flechas.

## Ejecutar

```bash
cd mmc_2d
uv sync
uv run python mmc_interactive.py
```

Si el backend gráfico por default no es interactivo:

```bash
MPLBACKEND=TkAgg uv run python mmc_interactive.py
```
