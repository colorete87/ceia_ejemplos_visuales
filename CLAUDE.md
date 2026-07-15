# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo shape

Didactic examples for a Machine Learning postgrad course (LSE). Each subdirectory is a **self-contained project** with its own `pyproject.toml` and `uv.lock` — there is no top-level Python project. Current examples:

- `proceso_gausiano/` — 1D Gaussian Process demo (`gp_interactive.py`)
- `proceso_gausiano_2d/` — 2D GP demo with bivariate mixture target (`gp_interactive_2d.py`)
- `knn_2d/` — interactive 2D k-NN classifier (`knn_interactive.py`)

User-facing docs (READMEs) are in Spanish; code identifiers mix Spanish and English.

## Commands

All projects are managed with [uv](https://docs.astral.sh/uv/) and require Python ≥3.13. Always operate from inside a single example directory:

```bash
cd <example-dir>
uv sync                          # install deps into .venv/
uv run python <script>.py        # run the demo
MPLBACKEND=TkAgg uv run python <script>.py   # fallback if default backend isn't interactive
```

There is no test suite, no linter config, and no build step — these are single-file matplotlib demos run directly.

## Architecture pattern (shared across all three demos)

Each `*_interactive*.py` is a ~600-line single-file matplotlib app following the same skeleton. Knowing this layout makes navigation fast:

1. **Module-level model code** (top of file): pure NumPy functions — distance metrics / kernel functions, weighting schemes, posterior computation, dataset generation. No matplotlib here.
2. **Module-level mutable state**: a global RNG (`RNG = np.random.default_rng()`) plus globals holding current dataset, observations, posterior history, and references to matplotlib artists.
3. **Figure layout**: helpers like `_add_group_box(...)` draw labeled rectangle groupings; widget axes are placed with `fig.add_axes([...])` at hard-coded coordinates. Widgets are `Slider`, `RadioButtons`, `Button`, `CheckButtons` from `matplotlib.widgets`.
4. **`redraw()`**: the single function that recomputes the model from current state and refreshes all artists. Almost every event handler ends by calling `redraw()`.
5. **Event handlers** (`_on_*`, `on_click`): one per widget; they mutate global state then trigger `redraw()`. Artist cleanup goes through `_safe_remove(artist)` (and `_remove_errorbar(...)` in the GP demos) to avoid stale references.
6. **Bottom of file**: widgets get connected (`.on_clicked`, `.on_changed`, `fig.canvas.mpl_connect('button_press_event', on_click)`), an initial `redraw()` is called, and `plt.show()` blocks.

When changing a demo, the usual flow is: (a) add/modify a pure-math helper at the top, (b) add a widget + handler, (c) extend `redraw()` to reflect the new state. Posterior "history" effects (fading old curves/contours) are implemented by keeping a list of old artists and tweaking their alpha on each redraw.

## Conventions to preserve

- Keep each example self-contained — do **not** introduce shared modules across the three demos. The duplication is intentional for didactic use.
- Comments and UI labels are in Spanish; match the surrounding style rather than translating.
- Dependencies stay minimal (`numpy`, `matplotlib`, and `scipy` only where needed for the GP demos). Add new deps via `uv add` inside the specific project, not at the repo root.

## Outstanding work

`TODO.md` at the repo root tracks pending examples (e.g. fixing KNN test points & variance, adding Bayesian parameter estimation in 1D/2D). Check it before proposing new examples.
