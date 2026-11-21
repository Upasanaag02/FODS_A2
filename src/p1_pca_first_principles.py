"""
P1: PCA from first principles.
 - build S (C3), eigendecompose (C4) with our own code
 - print eigenvalues, orthonormality / trace checks
 - verify against sklearn PCA (sign-fixed per C4)
 - loadings of PC1 and PC2
Outputs -> outputs/P1/ (terminal_output.txt, figures/, tables) and
           summary_pages/P1_onepage.png
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from common import *

d, figdir = problem_dirs("P1")

with Tee(os.path.join(d, "terminal_output.txt")):
    Xtr, ytr, Xte, yte = load_data()
    N, D = Xtr.shape

    # ---- C1/C2: standardise using training statistics
    mu, sd = zscore_fit(Xtr)
    Z = zscore_apply(Xtr, mu, sd)
    header("P1: setup")
    print(f"N = {N}, D = {D}")
    print(f"max |column mean of Z|  = {np.abs(Z.mean(0)).max():.1e}")
    print(f"max |column std(ddof=1)-1| = {np.abs(Z.std(0, ddof=1) - 1).max():.1e}")

    # ---- C3 + C4: our own PCA
    S = sample_cov(Z)
    lam, U = eig_sorted(S)

    header("P1: eigenvalues (descending)")
    for j, v in enumerate(lam, 1):
        print(f"lambda {j} = {v:.4f}")

    # ---- checks
    header("P1: orthonormality and trace checks")
    orth = np.linalg.norm(U.T @ U - np.eye(D), "fro")
    trS = np.trace(S)
    tr_diff = abs(trS - lam.sum())
    print(f"||U^T U - I||_F      = {orth:.1e}")
    print(f"tr(S)                = {trS:.4f}")
    print(f"|tr(S) - sum(lambda)| = {tr_diff:.1e}")
    print(f"(expected tr(S) = D = {D}; S_jj all 1: {np.allclose(np.diag(S), 1)})")
    print(f"S symmetric: {np.allclose(S, S.T)};  min eigenvalue = {lam.min():.4f} (>=0, PSD)")

    # ---- verification against sklearn
    header("P1: verification vs sklearn.decomposition.PCA")
    pca = PCA(n_components=13, svd_solver="full").fit(Z)
    U_sk = fix_sign(pca.components_.T)          # apply C4 sign rule to library output
    d_lam = np.abs(lam - pca.explained_variance_).max()
    d_U = np.abs(U - U_sk).max()
    print(f"max|lambda_ours - explained_variance_| = {d_lam:.1e}  -> agree: {d_lam < TOL}")
    print(f"max|U_ours - components_.T (sign-fixed)| = {d_U:.1e}  -> agree: {d_U < TOL}")

    # ---- loadings of PC1, PC2
    header("P1: loadings of u1 and u2 (by feature name)")
    load = pd.DataFrame({"PC1": U[:, 0], "PC2": U[:, 1]}, index=FEATURES)
    print(load.round(4).to_string())
    load.round(4).to_csv(os.path.join(d, "loadings_pc1_pc2.csv"))
    pd.DataFrame({"eigenvalue": lam}, index=[f"lambda_{j}" for j in range(1, 14)]) \
        .round(4).to_csv(os.path.join(d, "eigenvalues.csv"))
    for pc in ["PC1", "PC2"]:
        top = load[pc].abs().sort_values(ascending=False).head(3).index
        print(f"\nTop-3 |loading| on {pc}:")
        for f in top:
            print(f"   {f:<30s} {load.loc[f, pc]:+.4f}")

    # ================================================================ plots
    scores = Z @ U[:, :2]

    def p_eig(ax):
        ax.bar(range(1, D + 1), lam, color="steelblue", label="eigenvalue $\\lambda_j$")
        ax.axhline(1, color="red", ls="--", label="average eigenvalue = 1")
        ax.set_xlabel("component j"); ax.set_ylabel("eigenvalue")
        ax.set_title("P1: eigenvalues of S (own implementation)")
        ax.set_xticks(range(1, D + 1)); ax.legend()

    def p_load(ax):
        idx = np.arange(D); w = 0.4
        ax.bar(idx - w / 2, U[:, 0], w, label="PC1")
        ax.bar(idx + w / 2, U[:, 1], w, label="PC2")
        ax.set_xticks(idx); ax.set_xticklabels(FEATURES, rotation=75, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("loading"); ax.set_xlabel("feature")
        ax.set_ylim(-0.4, 0.75); ax.set_title("P1: loadings of u1 and u2"); ax.legend(loc="upper left", ncol=2)

    def p_scores(ax):
        for c in range(3):
            m = ytr == c
            ax.scatter(scores[m, 0], scores[m, 1], c=CLASS_COLORS[c], s=22, label=f"class {c}")
        ax.set_xlabel("PC1 score"); ax.set_ylabel("PC2 score")
        ax.set_title("P1: training scores on PC1-PC2\n(class used for colour only, not in PCA)")
        ax.legend()

    def p_corr(ax):
        im = ax.imshow(S, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(D)); ax.set_yticks(range(D))
        ax.set_xticklabels(range(D), fontsize=7); ax.set_yticklabels(range(D), fontsize=7)
        ax.set_xlabel("feature index"); ax.set_ylabel("feature index")
        ax.set_title("P1: S of z-scored data = correlation matrix")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, label="correlation")

    def p_verify(ax):
        vals = [d_lam, d_U, orth, tr_diff]
        names = ["|λ − sklearn|", "|U − sklearn|", "‖UᵀU−I‖_F", "|tr S − Σλ|"]
        ax.bar(names, [max(v, 1e-18) for v in vals], color="seagreen", label="measured")
        ax.axhline(TOL, color="red", ls="--", label="threshold 1e-8")
        ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-6)
        ax.set_ylabel("value (log scale)"); ax.set_xlabel("check")
        ax.set_title("P1: all verification residuals are below 1e-8")
        ax.tick_params(axis="x", labelsize=8); ax.legend()

    single(p_eig, os.path.join(figdir, "p1_eigenvalues.png"))
    single(p_load, os.path.join(figdir, "p1_loadings_pc1_pc2.png"), (9, 5.5))
    single(p_scores, os.path.join(figdir, "p1_pc1_pc2_scores.png"))
    single(p_corr, os.path.join(figdir, "p1_covariance_heatmap.png"))
    single(p_verify, os.path.join(figdir, "p1_verification_residuals.png"))
    onepage([p_eig, p_load, p_scores, p_corr, p_verify],
            os.path.join(SUMMARY, "P1_onepage.png"),
            "P1 — PCA from first principles (one-page summary)", 2, 3)
