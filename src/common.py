"""
Shared helpers for CS F320 Assignment-2 (PCA).
Implements the Common-Setup conventions C1-C6 once, so every problem uses
exactly the same definitions.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
SUMMARY = os.path.join(ROOT, "summary_pages")

FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols",
            "proanthocyanins", "color_intensity", "hue",
            "od280/od315_of_diluted_wines", "proline"]
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
TOL = 1e-8


# ----------------------------------------------------------------- logging
class Tee:
    """Copy everything printed to the terminal into a text file as well."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *a):
        sys.stdout = self.stdout
        self.file.close()


def problem_dirs(name):
    """outputs/<name>/ and outputs/<name>/figures/ (created if missing)."""
    d = os.path.join(OUT, name)
    f = os.path.join(d, "figures")
    os.makedirs(f, exist_ok=True)
    os.makedirs(SUMMARY, exist_ok=True)
    return d, f


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------------- data
def load_data():
    tr = pd.read_csv(os.path.join(DATA, "wine_train.csv"))
    te = pd.read_csv(os.path.join(DATA, "wine_test.csv"))
    assert list(tr.columns[:13]) == FEATURES, "unexpected column order"
    return (tr[FEATURES].values.astype(float), tr["class"].values.astype(int),
            te[FEATURES].values.astype(float), te["class"].values.astype(int))


# ---------------------------------------------------- C2 standardisation
def zscore_fit(Xtr):
    """C1/C2: training mean and sample std (divisor N-1)."""
    return Xtr.mean(axis=0), Xtr.std(axis=0, ddof=1)


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


# ------------------------------------------------------ C3 sample covariance
def sample_cov(X):
    """S = Xc^T Xc / (N-1) with Xc = X - mean (1/(N-1) convention)."""
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - 1)


# ------------------------------------------------ C4 eigendecomposition
def fix_sign(U):
    """Flip each column so that its largest-|entry| is positive (C4)."""
    U = U.copy()
    for j in range(U.shape[1]):
        i = np.argmax(np.abs(U[:, j]))
        if U[i, j] < 0:
            U[:, j] = -U[:, j]
    return U


def eig_sorted(S):
    """eigh -> eigenvalues descending, unit eigenvectors as columns, sign-fixed."""
    lam, U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    return lam[order], fix_sign(U[:, order])


# --------------------------------------------------- C5 explained variance
def explained_ratio(lam):
    r = lam / lam.sum()
    return r, np.cumsum(r)


# ---------------------------------------------------------------- plotting
def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.relpath(path, ROOT)}")


def single(plot_fn, path, figsize=(7, 5)):
    """Render one plot function as its own figure file."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    save_fig(fig, path)


def onepage(plot_fns, path, title, nrows, ncols, cell=(6.2, 4.6)):
    """Render several plot functions on ONE page (for viva)."""
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, fn in zip(axes, plot_fns):
        fn(ax)
    for ax in axes[len(plot_fns):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.relpath(path, ROOT)}")
