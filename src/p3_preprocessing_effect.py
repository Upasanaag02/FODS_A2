"""
P3: effect of preprocessing (V1 centred, V2 min-max, V3 z-score) on PCA and on
a 5-NN classifier trained on the 2-D PC1-PC2 projection.
Outputs -> outputs/P3/ and summary_pages/P3_onepage.png
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from common import *

d, figdir = problem_dirs("P3")

with Tee(os.path.join(d, "terminal_output.txt")):
    Xtr, ytr, Xte, yte = load_data()
    N, D = Xtr.shape
    Nte = Xte.shape[0]

    # ---- three preprocessing variants, all fitted on TRAIN only (C1)
    def prep_V1(Xtr, X):
        return X - Xtr.mean(0)

    def prep_V2(Xtr, X):
        mn, mx = Xtr.min(0), Xtr.max(0)
        scaled_tr = (Xtr - mn) / (mx - mn)
        return (X - mn) / (mx - mn) - scaled_tr.mean(0)

    def prep_V3(Xtr, X):
        mu, sd = zscore_fit(Xtr)
        return zscore_apply(X, mu, sd)

    variants = {"V1 (centred only)": prep_V1,
                "V2 (min-max)": prep_V2,
                "V3 (z-score)": prep_V3}

    results = {}
    rows = []
    for name, prep in variants.items():
        Ptr, Pte = prep(Xtr, Xtr), prep(Xtr, Xte)
        S = sample_cov(Ptr)
        lam, U = eig_sorted(S)
        r, _ = explained_ratio(lam)
        C_tr, C_te = Ptr @ U[:, :2], Pte @ U[:, :2]
        knn = KNeighborsClassifier(n_neighbors=5).fit(C_tr, ytr)
        acc = knn.score(C_te, yte)
        top1 = FEATURES[int(np.argmax(np.abs(U[:, 0])))]
        results[name] = dict(Ctr=C_tr, Cte=C_te, r=r, U=U, acc=acc, S=S)
        rows.append({"variant": name, "r1": r[0], "r2": r[1], "r1+r2": r[0] + r[1],
                     "largest |loading| on PC1": top1,
                     "5-NN test accuracy": acc})

    header("P3: summary table")
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tab.round(4).to_csv(os.path.join(d, "p3_table.csv"), index=False)
    print("\n5-NN test accuracy as counts:")
    for name, res in results.items():
        print(f"  {name:<18s} {round(res['acc'] * Nte)}/{Nte} correct = {res['acc']:.4f}")

    header("P3: V1 - three features with largest raw sample variance")
    var = pd.Series(Xtr.var(axis=0, ddof=1), index=FEATURES)
    for f, v in var.sort_values(ascending=False).head(3).items():
        print(f"  {f:<20s} variance = {v:.4f}")
    print(f"  (tr(S) for V1 = {var.sum():.4f};  proline alone = "
          f"{var['proline'] / var.sum():.4f} of total)")
    print("\nV1 PC1 loadings (top 3 by |loading|):")
    u1 = pd.Series(results["V1 (centred only)"]["U"][:, 0], index=FEATURES)
    for f in u1.abs().sort_values(ascending=False).head(3).index:
        print(f"  {f:<20s} {u1[f]:+.4f}")

    header("P3: extra facts used in the paragraph")
    print("Largest |loading| on PC2 per variant:")
    for name, res in results.items():
        u2 = res["U"][:, 1]
        print(f"  {name:<18s} {FEATURES[int(np.argmax(np.abs(u2)))]}")

    # ================================================================ plots
    def make_panel(name):
        res = results[name]

        def fn(ax):
            for c in range(3):
                m = ytr == c
                ax.scatter(res["Ctr"][m, 0], res["Ctr"][m, 1], c=CLASS_COLORS[c],
                           marker="o", s=22, label=f"train class {c}")
            for c in range(3):
                m = yte == c
                ax.scatter(res["Cte"][m, 0], res["Cte"][m, 1], c=CLASS_COLORS[c],
                           marker="x", s=40, label=f"test class {c}")
            ax.set_xlabel("PC1 score"); ax.set_ylabel("PC2 score")
            ax.set_title(f"{name}: PC1-PC2 projection\n"
                         f"r1+r2 = {res['r'][0] + res['r'][1]:.3f}, 5-NN acc = {res['acc']:.3f}")
            ax.legend(fontsize=7, ncol=2)
        return fn

    panels = [make_panel(n) for n in variants]

    def p_var(ax):
        order = np.argsort(var.values)[::-1]
        ax.bar(range(D), var.values[order], color="indianred", label="raw sample variance $s_j^2$")
        ax.set_xticks(range(D))
        ax.set_xticklabels(np.array(FEATURES)[order], rotation=75, ha="right", fontsize=7)
        ax.set_yscale("log"); ax.set_ylabel("variance (log scale)"); ax.set_xlabel("feature")
        ax.set_title("P3: raw per-feature variance (why V1 is dominated by proline)")
        ax.legend()

    def p_r(ax):
        names = list(variants); x = np.arange(3); w = 0.38
        ax.bar(x - w / 2, [results[n]["r"][0] for n in names], w, label="$r_1$")
        ax.bar(x + w / 2, [results[n]["r"][1] for n in names], w, label="$r_2$")
        ax.set_xticks(x); ax.set_xticklabels([n.split(" ")[0] for n in names])
        ax.set_xlabel("preprocessing variant"); ax.set_ylabel("explained variance ratio")
        ax.set_title("P3: $r_1$ and $r_2$ per variant"); ax.legend()

    def p_acc(ax):
        names = list(variants)
        ax.bar([n.split(" ")[0] for n in names], [results[n]["acc"] for n in names],
               color=["tab:red", "tab:purple", "tab:green"], label="5-NN test accuracy")
        ax.set_ylim(0, 1.05); ax.set_xlabel("preprocessing variant"); ax.set_ylabel("accuracy")
        for i, n in enumerate(names):
            ax.text(i, results[n]["acc"] + 0.01, f"{results[n]['acc']:.3f}", ha="center")
        ax.set_title("P3: 5-NN accuracy on 2-D projection"); ax.legend()

    # required 3-panel figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, fn in zip(axes, panels):
        fn(ax)
    save_fig(fig, os.path.join(figdir, "p3_projections_3panel.png"))
    single(p_var, os.path.join(figdir, "p3_raw_variances.png"), (9, 5))
    single(p_r, os.path.join(figdir, "p3_r1_r2_by_variant.png"))
    single(p_acc, os.path.join(figdir, "p3_knn_accuracy.png"))
    onepage(panels + [p_var, p_r, p_acc], os.path.join(SUMMARY, "P3_onepage.png"),
            "P3 — Effect of preprocessing and the 2-D projection (one-page summary)", 2, 3)
