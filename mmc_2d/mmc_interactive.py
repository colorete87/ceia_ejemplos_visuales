"""
Demo interactivo de Maximal Margin Classifier (y SVM soft margin) en 2D.

Funcionalidades:
  - Sliders para los parámetros del hiperplano β0 + β1·x1 + β2·x2 = 0.
  - Selector Hard / Soft margin, slider C para Soft.
  - Botón Auto: resuelve el QP y dibuja el óptimo (línea punteada).
  - Edición manual de puntos: agregar A/B, mover, borrar.
  - Visualización del margen, support vectors y flechas perpendiculares.

Ejecutar:
    uv run python mmc_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy.optimize import minimize


RNG = np.random.default_rng()


if __name__ == "__main__":
    print("mmc_2d bootstrap OK")
