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


# ===========================================================
# Estado global y constantes de UI
# ===========================================================
X_MIN, X_MAX = -5.0, 5.0
CLASS_COLORS = ["#1f77b4", "#ff7f0e"]  # clase -1 (idx 0) / clase +1 (idx 1)
GRID = 100
gxs = np.linspace(X_MIN, X_MAX, GRID)
GX1, GX2 = np.meshgrid(gxs, gxs)
GRID_PTS = np.column_stack([GX1.ravel(), GX2.ravel()])

DEFAULTS = {
    "n_per_class": 25,
    "classifier": "Hard",
    "C": 1.0,
    "beta0": 0.0, "beta1": 1.0, "beta2": -1.0,
    "click_mode": "Agregar A",
    "show_decision": False,
    "show_corridor": True,
    "show_arrows": True,
}

state = dict(DEFAULTS)
state.update({
    "X": np.empty((0, 2)),
    "y": np.empty(0, dtype=int),
    "auto_solution": None,
    "auto_infeasible": False,
    "drag_idx": None,
})


def _regenerate_data():
    Xr, yr = make_random_dataset(state["n_per_class"], RNG)
    state["X"] = Xr
    state["y"] = yr
    state["auto_solution"] = None
    state["auto_infeasible"] = False


# ===========================================================
# Figura
# ===========================================================
fig = plt.figure(figsize=(13, 7.5))
ax = fig.add_axes([0.27, 0.36, 0.50, 0.58])
ax.set_aspect("equal")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(X_MIN, X_MAX)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.grid(alpha=0.3)
ax.set_title("Maximal Margin Classifier (2D)", loc="left", fontsize=10)

ax_metrics = fig.add_axes([0.80, 0.36, 0.18, 0.58])
ax_metrics.set_xticks([])
ax_metrics.set_yticks([])
ax_metrics.set_facecolor("#fafafa")
ax_metrics.set_title("Métricas", fontsize=10, weight="bold")
metrics_text = ax_metrics.text(
    0.04, 0.96, "", transform=ax_metrics.transAxes,
    ha="left", va="top", fontsize=10, family="monospace",
)

def _add_group_box(x, y, w, h, label):
    rect = Rectangle((x, y), w, h, transform=fig.transFigure,
                     linewidth=0.9, edgecolor="#666666",
                     facecolor="none", zorder=0)
    fig.add_artist(rect)
    return fig.text(x + w / 2, y + h + 0.006, label,
                    ha="center", va="bottom", fontsize=9, weight="bold",
                    color="#333333")


def _safe_remove(artist):
    try:
        artist.remove()
    except Exception:
        pass


# Artists reusables
data_scatter = ax.scatter([], [], s=40, edgecolors="black", linewidths=0.5)
hyperplane_line, = ax.plot([], [], color="black", lw=2.0, zorder=5)
margin_lines = [None, None]   # las dos paralelas β·x = ±1
corridor_poly = [None]          # Polygon del sombreado
decision_image = [None]         # AxesImage opcional
sv_scatter = ax.scatter([], [], s=180, facecolors="none",
                         edgecolors="black", linewidths=1.6, zorder=6)
arrow_artists = []              # lista de FancyArrowPatch / Annotation


def _hyperplane_xy(b0, b1, b2, level=0.0):
    """Devuelve (xs, ys) para dibujar la recta β0+β1·x1+β2·x2 = level
    dentro del rango [X_MIN, X_MAX]. Maneja el caso vertical (β2≈0)."""
    if abs(b2) > 1e-6:
        xs = np.array([X_MIN, X_MAX])
        ys = (level - b0 - b1 * xs) / b2
        return xs, ys
    if abs(b1) > 1e-6:
        ys = np.array([X_MIN, X_MAX])
        xs = np.full_like(ys, (level - b0) / b1)
        return xs, ys
    return np.array([X_MIN, X_MAX]), np.array([0.0, 0.0])


def redraw():
    b0, b1, b2 = state["beta0"], state["beta1"], state["beta2"]
    norm2 = b1 * b1 + b2 * b2

    # 1. Región de decisión
    _safe_remove(decision_image[0])
    decision_image[0] = None
    if state["show_decision"] and norm2 > 1e-12:
        pred_grid = np.sign(decision_value(GRID_PTS, b0, b1, b2)).reshape(GRID, GRID)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(CLASS_COLORS)
        # Map sign -1 -> 0, +1 -> 1
        decision_image[0] = ax.imshow(
            ((pred_grid + 1) // 2).astype(int),
            extent=(X_MIN, X_MAX, X_MIN, X_MAX),
            origin="lower", cmap=cmap, alpha=0.15,
            vmin=0, vmax=1, interpolation="nearest", zorder=0)

    # 2. Sombreado del corredor del margen
    _safe_remove(corridor_poly[0])
    corridor_poly[0] = None
    if state["show_corridor"] and norm2 > 1e-9:
        xs_a, ys_a = _hyperplane_xy(b0, b1, b2, level=1.0)
        xs_b, ys_b = _hyperplane_xy(b0, b1, b2, level=-1.0)
        # Polígono cerrado: (a0, a1, b1, b0)
        xs_poly = np.concatenate([xs_a, xs_b[::-1]])
        ys_poly = np.concatenate([ys_a, ys_b[::-1]])
        corridor_poly[0] = ax.fill(xs_poly, ys_poly,
                                    color="#777777", alpha=0.12, zorder=1)[0]

    # 3. Hiperplano + paralelas
    xs, ys = _hyperplane_xy(b0, b1, b2, level=0.0)
    hyperplane_line.set_data(xs, ys)
    for ln in margin_lines:
        _safe_remove(ln)
    margin_lines[0] = None
    margin_lines[1] = None
    if norm2 > 1e-9:
        for i, level in enumerate([1.0, -1.0]):
            xs_m, ys_m = _hyperplane_xy(b0, b1, b2, level=level)
            ln, = ax.plot(xs_m, ys_m, color="black", lw=0.9, ls="--",
                          alpha=0.6, zorder=4)
            margin_lines[i] = ln

    # 3b. Hiperplano óptimo (auto) como línea punteada de referencia
    if "auto_lines" not in redraw.__dict__:
        redraw.auto_lines = []
    for ln in redraw.auto_lines:
        _safe_remove(ln)
    redraw.auto_lines = []
    if state["auto_solution"] is not None:
        ab0 = state["auto_solution"]["b0"]
        ab1 = state["auto_solution"]["b1"]
        ab2 = state["auto_solution"]["b2"]
        if abs(ab1) > 1e-9 or abs(ab2) > 1e-9:
            xs_a, ys_a = _hyperplane_xy(ab0, ab1, ab2, level=0.0)
            ln, = ax.plot(xs_a, ys_a, color="#d62728", lw=1.6, ls=":",
                          alpha=0.9, zorder=4.5)
            redraw.auto_lines.append(ln)

    # 4. Scatter de datos
    if len(state["X"]) > 0:
        idx = ((state["y"] + 1) // 2).astype(int)
        colors = [CLASS_COLORS[i] for i in idx]
        data_scatter.set_offsets(state["X"])
        data_scatter.set_facecolors(colors)
    else:
        data_scatter.set_offsets(np.empty((0, 2)))

    # 5. Support vectors
    for a in arrow_artists:
        _safe_remove(a)
    arrow_artists.clear()
    if len(state["X"]) > 0 and norm2 > 1e-9:
        mask_sv = support_vectors_mask(state["X"], state["y"], b0, b1, b2,
                                        state["classifier"])
        sv_scatter.set_offsets(state["X"][mask_sv]
                                if mask_sv.any() else np.empty((0, 2)))
        # 6. Flechas perpendiculares desde cada SV hasta su pie en el hiperplano
        if state["show_arrows"] and mask_sv.any():
            X_sv = state["X"][mask_sv]
            # Distancia firmada del SV al hiperplano (no al margen): d = (β·x) / ||β||
            d = decision_value(X_sv, b0, b1, b2) / np.sqrt(norm2)
            # Pie sobre el hiperplano: x_foot = x - d · (β/||β||)
            unit = np.array([b1, b2]) / np.sqrt(norm2)
            X_foot = X_sv - d[:, None] * unit[None, :]
            for (x0, y0), (xf, yf) in zip(X_sv, X_foot):
                ann = ax.annotate(
                    "", xy=(xf, yf), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=1.0, alpha=0.8),
                    zorder=7)
                arrow_artists.append(ann)
    else:
        sv_scatter.set_offsets(np.empty((0, 2)))

    # 7. Panel Métricas
    if norm2 > 1e-9:
        m_width = margin_width(b1, b2)
    else:
        m_width = float("nan")
    n_sv = int(mask_sv.sum()) if (len(state["X"]) > 0 and norm2 > 1e-9) else 0
    if len(state["X"]) > 0:
        acc = accuracy(state["X"], state["y"], b0, b1, b2)
    else:
        acc = float("nan")
    if state["auto_infeasible"]:
        status = "Ø no separable"
    else:
        status = "OK"
    lines = [
        f"Margen:   {m_width:6.3f}",
        f"‖β‖:      {np.sqrt(norm2):6.3f}",
        f"# SV:     {n_sv:>3d}",
        f"Accuracy: {acc:6.3f}",
        f"Status:   {status}",
    ]
    metrics_text.set_text("\n".join(lines))

    fig.canvas.draw_idle()


# ===========================================================
# Sliders del hiperplano
# ===========================================================
_add_group_box(0.025, 0.78, 0.21, 0.15, "Hiperplano")
ax_b0 = plt.axes([0.07, 0.87, 0.16, 0.020])
sl_b0 = Slider(ax_b0, "β₀", -5.0, 5.0, valinit=DEFAULTS["beta0"])
ax_b1 = plt.axes([0.07, 0.83, 0.16, 0.020])
sl_b1 = Slider(ax_b1, "β₁", -3.0, 3.0, valinit=DEFAULTS["beta1"])
ax_b2 = plt.axes([0.07, 0.79, 0.16, 0.020])
sl_b2 = Slider(ax_b2, "β₂", -3.0, 3.0, valinit=DEFAULTS["beta2"])


def _on_beta(_v):
    state["beta0"] = float(sl_b0.val)
    state["beta1"] = float(sl_b1.val)
    state["beta2"] = float(sl_b2.val)
    if not _suppress_redraw[0]:
        redraw()


sl_b0.on_changed(_on_beta)
sl_b1.on_changed(_on_beta)
sl_b2.on_changed(_on_beta)

# ===========================================================
# Selector de clasificador + C
# ===========================================================
_add_group_box(0.025, 0.59, 0.21, 0.17, "Clasificador")
ax_clf = plt.axes([0.04, 0.62, 0.09, 0.12])
radio_clf = RadioButtons(ax_clf, ["Hard", "Soft"],
                          active=["Hard", "Soft"].index(DEFAULTS["classifier"]))

ax_C = plt.axes([0.16, 0.66, 0.08, 0.020])
sl_C = Slider(ax_C, "C", -2.0, 2.0,
              valinit=np.log10(DEFAULTS["C"]),
              valfmt="10^%.2f")
sl_C.set_active(DEFAULTS["classifier"] == "Soft")


def _on_clf(label):
    state["classifier"] = label
    sl_C.set_active(label == "Soft")
    state["auto_infeasible"] = False  # se recalcula al apretar Auto
    redraw()


def _on_C(v):
    state["C"] = 10.0 ** float(v)


radio_clf.on_clicked(_on_clf)
sl_C.on_changed(_on_C)

# ===========================================================
# Control (Auto, Limpiar óptimo, Reset)
# ===========================================================
_add_group_box(0.025, 0.36, 0.21, 0.17, "Control")
ax_btn_auto = plt.axes([0.04, 0.46, 0.18, 0.04])
btn_auto = Button(ax_btn_auto, "Auto",
                   color="#ffd8c2", hovercolor="#f8b690")
ax_btn_clear = plt.axes([0.04, 0.41, 0.18, 0.04])
btn_clear_opt = Button(ax_btn_clear, "Limpiar óptimo")
ax_btn_reset = plt.axes([0.04, 0.36, 0.18, 0.04])
btn_reset = Button(ax_btn_reset, "Reset todo")


_suppress_redraw = [False]


def _set_betas_silent(b0, b1, b2):
    """Setea los 3 sliders sin disparar 3 redraws individuales."""
    _suppress_redraw[0] = True
    try:
        sl_b0.set_val(b0)
        sl_b1.set_val(b1)
        sl_b2.set_val(b2)
    finally:
        _suppress_redraw[0] = False
    state["beta0"] = b0
    state["beta1"] = b1
    state["beta2"] = b2


def _on_btn_auto(_event):
    if len(state["X"]) == 0:
        return
    if state["classifier"] == "Hard":
        sol = fit_hard_margin(state["X"], state["y"])
        if sol is None:
            state["auto_infeasible"] = True
            redraw()
            return
    else:
        sol = fit_soft_margin(state["X"], state["y"], state["C"])
        if sol is None:
            return
    state["auto_infeasible"] = False
    state["auto_solution"] = sol
    # Clamp a los rangos de los sliders para que set_val no salga del rango
    b0c = float(np.clip(sol["b0"], -5.0, 5.0))
    b1c = float(np.clip(sol["b1"], -3.0, 3.0))
    b2c = float(np.clip(sol["b2"], -3.0, 3.0))
    _set_betas_silent(b0c, b1c, b2c)
    redraw()


def _on_btn_clear_opt(_event):
    state["auto_solution"] = None
    redraw()


btn_auto.on_clicked(_on_btn_auto)
btn_clear_opt.on_clicked(_on_btn_clear_opt)
# btn_reset.on_clicked se conecta en una task posterior (Task 11)

_regenerate_data()
redraw()

if __name__ == "__main__":
    plt.show()
