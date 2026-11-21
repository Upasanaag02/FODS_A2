"""
P2: variance captured vs. number of components k.
 - scree + cumulative plots, k_elbow (r_k >= 0.10), k_0.80/0.90/0.95, Kaiser
 - table of r_k, R(k), J_train(k), J_test(k)
 - numerical check of  J_train(k) = sum_{j>k} lambda_j
Outputs -> outputs/P2/ and summary_pages/P2_onepage.png
"""
import os
import numpy as np
import pandas as pd
from common import *

d, figdir = problem_dirs("P2")

with Tee(os.path.join(d, "terminal_output.txt")):
    Xtr, ytr, Xte, yte = load_data()
    N, D = Xtr.shape
    Nte = Xte.shape[0]

    # C1/C2: z-score with TRAIN statistics; test transformed with the same ones
    mu, sd = zscore_fit(Xtr)
    Ztr = zscore_apply(Xtr, mu, sd)
    Zte = zscore_apply(Xte, mu, sd)

    S = sample_cov(Ztr)
    lam, U = eig_sorted(S)
    r, R = explained_ratio(lam)
    trS = np.trace(S)

    # ---- component-count rules
    k_elbow = max(k for k in range(1, D + 1) if r[k - 1] >= 0.10)
    k_tau = {t: int(np.argmax(R >= t)) + 1 for t in (0.80, 0.90, 0.95)}
    k_kaiser = int(np.sum(lam > trS / D))

    # ---- reconstruction errors, computed DIRECTLY from reconstructions
    def J(Z, k):
        Uk = U[:, :k]
        Zhat = (Z @ Uk) @ Uk.T                     # z_hat = U_k U_k^T z
        return np.sum((Z - Zhat) ** 2) / (Z.shape[0] - 1)

    Jtr = np.array([J(Ztr, k) for k in range(1, D + 1)])
    Jte = np.array([J(Zte, k) for k in range(1, D + 1)])
    tail = np.array([lam[k:].sum() for k in range(1, D + 1)])   # sum_{j>k} lambda_j
    ident = np.abs(Jtr - tail).max()

    header("P2: component-count rules")
    print(f"tr(S) = {trS:.4f}, average eigenvalue = {trS / D:.4f}")
    print(f"k_elbow (last k with r_k >= 0.10) = {k_elbow}")
    print(f"R(k_elbow) = {R[k_elbow - 1]:.4f}")
    for t, k in k_tau.items():
        print(f"k_{t:.2f} = {k}   (R({k}) = {R[k - 1]:.4f})")
    print(f"Kaiser (lambda_j > average eigenvalue) retains {k_kaiser} components")

    header("P2: table for k = 1..13")
    tab = pd.DataFrame({"k": np.arange(1, D + 1), "r_k": r, "R(k)": R,
                        "J_train(k)": Jtr, "J_test(k)": Jte})
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tab.round(4).to_csv(os.path.join(d, "p2_table.csv"), index=False)

    header("P2: reconstruction-error identity")
    print(f"max_k |J_train(k) - sum_(j>k) lambda_j| = {ident:.1e}  -> below 1e-8: {ident < TOL}")
    print(f"(also J_train(k)/tr(S) = 1 - R(k): max diff = "
          f"{np.abs(Jtr / trS - (1 - R)).max():.1e})")

    header("P2: facts for the paragraph")
    dJ = np.diff(Jte)
    print(f"J_test non-increasing in k: {bool(np.all(dJ <= 1e-12))}  "
          f"(max successive change = {dJ.max():.2e})")
    print(f"J_test(13) = {Jte[-1]:.2e}, J_train(13) = {Jtr[-1]:.2e}")
    print(f"J_test(k) - J_train(k) at k=1,3,5,8: "
          f"{[round(float(Jte[k - 1] - Jtr[k - 1]), 4) for k in (1, 3, 5, 8)]}")

    # ================================================================ plots
    ks = np.arange(1, D + 1)

    def p_scree(ax):
        ax.bar(ks, r, color="steelblue", label="$r_j$")
        ax.axvline(k_elbow + 0.5, color="red", ls="--", label=f"$k_{{elbow}}$ = {k_elbow}")
        ax.axhline(0.10, color="grey", ls=":", label="0.10 threshold")
        ax.set_xlabel("component j"); ax.set_ylabel("explained variance ratio $r_j$")
        ax.set_title("P2: scree plot"); ax.set_xticks(ks); ax.legend()

    def p_cum(ax):
        ax.plot(ks, R, "o-", color="darkgreen", label="R(k)")
        for t, c in zip((0.80, 0.90, 0.95), ("tab:orange", "tab:red", "tab:purple")):
            ax.axhline(t, ls="--", color=c, label=f"{t:.2f}  ($k_{{{t:.2f}}}$={k_tau[t]})")
        ax.set_xlabel("number of components k"); ax.set_ylabel("cumulative ratio R(k)")
        ax.set_title("P2: cumulative explained variance")
        ax.set_xticks(ks); ax.legend(loc="lower right")

    def p_J(ax):
        ax.plot(ks, Jtr, "o-", label="$J_{train}(k)$")
        ax.plot(ks, Jte, "s--", label="$J_{test}(k)$")
        ax.set_xlabel("number of components k"); ax.set_ylabel("reconstruction error J(k)")
        ax.set_title("P2: reconstruction error vs k"); ax.set_xticks(ks); ax.legend()

    def p_ident(ax):
        ax.plot(ks, Jtr, "o", ms=9, mfc="none", label="$J_{train}(k)$ from reconstructions")
        ax.plot(ks, tail, "x", ms=7, label=r"$\sum_{j>k}\lambda_j$")
        ax.set_xlabel("number of components k"); ax.set_ylabel("value")
        ax.set_title(f"P2: identity check (max diff = {ident:.1e})")
        ax.set_xticks(ks); ax.legend()

    single(p_scree, os.path.join(figdir, "p2_scree.png"))
    single(p_cum, os.path.join(figdir, "p2_cumulative.png"))
    single(p_J, os.path.join(figdir, "p2_reconstruction_error.png"))
    single(p_ident, os.path.join(figdir, "p2_identity_check.png"))
    onepage([p_scree, p_cum, p_J, p_ident], os.path.join(SUMMARY, "P2_onepage.png"),
            "P2 — Variance captured vs. number of components (one-page summary)", 2, 2)
