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


# ===========================================================
# Modelo geométrico (puro numpy, sin matplotlib)
# ===========================================================
def decision_value(X, b0, b1, b2):
    """β0 + β1·x1 + β2·x2 para cada fila de X (n,2)."""
    X = np.atleast_2d(X)
    return b0 + b1 * X[:, 0] + b2 * X[:, 1]


def signed_margin(X, y, b0, b1, b2, eps=1e-12):
    """Margen geométrico firmado por punto: y_i · (β·x_i) / ||β||_2.
    y ∈ {-1, +1}."""
    norm = np.sqrt(b1 * b1 + b2 * b2) + eps
    return y * decision_value(X, b0, b1, b2) / norm


def margin_width(b1, b2, eps=1e-12):
    """Ancho total del corredor del margen: 2 / ||β||_2."""
    return 2.0 / (np.sqrt(b1 * b1 + b2 * b2) + eps)


def support_vectors_mask(X, y, b0, b1, b2, classifier, tol=1e-3):
    """Devuelve bool array (n,) con los puntos que son support vectors.
    - Hard: |y_i·(β·x_i) - 1| < tol  (sobre el margen funcional ±1)
    - Soft: y_i·(β·x_i) <= 1 + tol    (sobre el margen o dentro)
    """
    margin_fn = y * decision_value(X, b0, b1, b2)
    if classifier == "Hard":
        return np.abs(margin_fn - 1.0) < tol
    return margin_fn <= 1.0 + tol


def accuracy(X, y, b0, b1, b2):
    """Accuracy del clasificador y_hat = sign(β·x)."""
    if len(X) == 0:
        return float("nan")
    y_hat = np.sign(decision_value(X, b0, b1, b2))
    y_hat = np.where(y_hat == 0, 1, y_hat)  # romper empate
    return float(np.mean(y_hat == y))


if __name__ == "__main__":
    # Sanity asserts mientras desarrollamos
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    y = np.array([1, -1, 1, -1])
    # Hiperplano x1 = 0 (β0=0, β1=1, β2=0): clasifica por signo de x1
    assert np.allclose(decision_value(X, 0, 1, 0), [1, -1, 1, -1])
    assert np.allclose(signed_margin(X, y, 0, 1, 0), [1, 1, 1, 1])
    assert abs(margin_width(1, 0) - 2.0) < 1e-9
    assert accuracy(X, y, 0, 1, 0) == 1.0
    print("mmc_2d funciones puras OK")
