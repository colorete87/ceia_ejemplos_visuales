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


# ===========================================================
# Optimizadores QP via scipy SLSQP
# ===========================================================
def fit_hard_margin(X, y):
    """Resuelve el QP del hard-margin:
        min  ½ (β1² + β2²)
        s.t. y_i · (β0 + β1·x_i1 + β2·x_i2) >= 1   ∀i

    Devuelve dict {b0, b1, b2} o None si SLSQP no converge o las
    constraints quedan violadas (proxy de no separabilidad).
    """
    if len(X) == 0:
        return None
    n = len(X)
    # Variables: theta = (b0, b1, b2)
    def obj(theta):
        return 0.5 * (theta[1] ** 2 + theta[2] ** 2)
    def obj_grad(theta):
        return np.array([0.0, theta[1], theta[2]])

    constraints = []
    for i in range(n):
        xi, yi = X[i], y[i]
        constraints.append({
            "type": "ineq",
            "fun": lambda th, xi=xi, yi=yi: yi * (th[0] + th[1] * xi[0] + th[2] * xi[1]) - 1.0,
            "jac": lambda th, xi=xi, yi=yi: np.array([yi, yi * xi[0], yi * xi[1]]),
        })

    # Punto inicial: vector entre medias de las clases, normalizado
    mask_pos = y > 0
    if mask_pos.any() and (~mask_pos).any():
        mu_pos = X[mask_pos].mean(axis=0)
        mu_neg = X[~mask_pos].mean(axis=0)
        w0 = mu_pos - mu_neg
        b1_0, b2_0 = w0
        b0_0 = -0.5 * (b1_0 * (mu_pos[0] + mu_neg[0]) + b2_0 * (mu_pos[1] + mu_neg[1]))
        theta0 = np.array([b0_0, b1_0, b2_0])
    else:
        theta0 = np.array([0.0, 1.0, 0.0])

    res = minimize(obj, theta0, jac=obj_grad, method="SLSQP",
                   constraints=constraints,
                   options={"maxiter": 200, "ftol": 1e-8})

    if not res.success:
        return None
    # Validar que las constraints se cumplan (residuos negativos = violación)
    b0, b1, b2 = res.x
    residuals = y * (b0 + b1 * X[:, 0] + b2 * X[:, 1]) - 1.0
    if np.any(residuals < -1e-4):
        return None
    return {"b0": float(b0), "b1": float(b1), "b2": float(b2)}


def fit_soft_margin(X, y, C):
    """Resuelve el QP del soft-margin (SVM):
        min  ½ (β1² + β2²) + C · Σ ξ_i
        s.t. y_i · (β0 + β1·x_i1 + β2·x_i2) >= 1 - ξ_i
             ξ_i >= 0

    Variables = (b0, b1, b2, ξ_1, …, ξ_n). Siempre devuelve sol.
    """
    if len(X) == 0:
        return None
    n = len(X)

    def obj(theta):
        b1, b2 = theta[1], theta[2]
        xi = theta[3:]
        return 0.5 * (b1 ** 2 + b2 ** 2) + C * np.sum(xi)
    def obj_grad(theta):
        g = np.zeros_like(theta)
        g[1] = theta[1]
        g[2] = theta[2]
        g[3:] = C
        return g

    constraints = []
    for i in range(n):
        xi_pt, yi = X[i], y[i]
        # y_i (b0 + b1 x + b2 y) - 1 + xi_i >= 0
        def fn_margin(th, i=i, xi_pt=xi_pt, yi=yi):
            return yi * (th[0] + th[1] * xi_pt[0] + th[2] * xi_pt[1]) - 1.0 + th[3 + i]
        def jac_margin(th, i=i, xi_pt=xi_pt, yi=yi):
            g = np.zeros(3 + n)
            g[0] = yi
            g[1] = yi * xi_pt[0]
            g[2] = yi * xi_pt[1]
            g[3 + i] = 1.0
            return g
        constraints.append({"type": "ineq", "fun": fn_margin, "jac": jac_margin})
        # xi_i >= 0
        def fn_xi(th, i=i):
            return th[3 + i]
        def jac_xi(th, i=i):
            g = np.zeros(3 + n)
            g[3 + i] = 1.0
            return g
        constraints.append({"type": "ineq", "fun": fn_xi, "jac": jac_xi})

    # Inicialización: hard-margin si factible, si no fallback a vector entre medias
    init_hard = fit_hard_margin(X, y)
    if init_hard is not None:
        theta0 = np.array([init_hard["b0"], init_hard["b1"], init_hard["b2"]] + [0.0] * n)
    else:
        mask_pos = y > 0
        if mask_pos.any() and (~mask_pos).any():
            mu_pos = X[mask_pos].mean(axis=0)
            mu_neg = X[~mask_pos].mean(axis=0)
            w0 = mu_pos - mu_neg
            b1_0, b2_0 = w0
            b0_0 = -0.5 * (b1_0 * (mu_pos[0] + mu_neg[0]) + b2_0 * (mu_pos[1] + mu_neg[1]))
            theta0 = np.array([b0_0, b1_0, b2_0] + [1.0] * n)
        else:
            theta0 = np.array([0.0, 1.0, 0.0] + [1.0] * n)

    res = minimize(obj, theta0, jac=obj_grad, method="SLSQP",
                   constraints=constraints,
                   options={"maxiter": 300, "ftol": 1e-7})
    if not res.success:
        return None
    b0, b1, b2 = res.x[0], res.x[1], res.x[2]
    return {"b0": float(b0), "b1": float(b1), "b2": float(b2)}


# ===========================================================
# Generación de datos: 2 clusters gaussianos
# ===========================================================
def _random_means_2(rng, x_min=-3.5, x_max=3.5, min_sep=2.5, max_tries=200):
    """Sortea dos medias separadas al menos min_sep."""
    m1 = rng.uniform(x_min, x_max, size=2)
    for _ in range(max_tries):
        m2 = rng.uniform(x_min, x_max, size=2)
        if np.linalg.norm(m1 - m2) >= min_sep:
            return m1, m2
    return m1, m1 + np.array([min_sep, 0.0])


def _random_cov(rng, sigma_lo=0.4, sigma_hi=1.0, rho_max=0.6):
    sx = rng.uniform(sigma_lo, sigma_hi)
    sy = rng.uniform(sigma_lo, sigma_hi)
    rho = rng.uniform(-rho_max, rho_max)
    return np.array([[sx ** 2,        rho * sx * sy],
                     [rho * sx * sy,  sy ** 2]])


def make_random_dataset(n_per_class, rng=None):
    """Devuelve (X, y) con n_per_class de cada clase. y ∈ {-1, +1}."""
    rng = rng or RNG
    m_neg, m_pos = _random_means_2(rng)
    L_neg = np.linalg.cholesky(_random_cov(rng))
    L_pos = np.linalg.cholesky(_random_cov(rng))
    z_neg = rng.standard_normal((n_per_class, 2))
    z_pos = rng.standard_normal((n_per_class, 2))
    X_neg = z_neg @ L_neg.T + m_neg
    X_pos = z_pos @ L_pos.T + m_pos
    X = np.concatenate([X_neg, X_pos])
    y = np.concatenate([np.full(n_per_class, -1, dtype=int),
                        np.full(n_per_class, +1, dtype=int)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


if __name__ == "__main__":
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    y = np.array([1, -1, 1, -1])
    assert np.allclose(decision_value(X, 0, 1, 0), [1, -1, 1, -1])
    assert abs(margin_width(1, 0) - 2.0) < 1e-9
    assert fit_hard_margin(X, y) is not None
    assert fit_soft_margin(X, y, C=1.0) is not None

    Xr, yr = make_random_dataset(20)
    assert Xr.shape == (40, 2)
    assert set(yr.tolist()) == {-1, 1}
    print("mmc_2d dataset OK")
