"""
P4: PCA vs filter (F-score) vs wrapper (sequential forward selection) for a
multinomial logistic-regression classifier.
Outputs -> outputs/P4/ and summary_pages/P4_onepage.png
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from common import *

warnings.filterwarnings("ignore")
d, figdir = problem_dirs("P4")
KS = [1, 2, 3, 5, 8]

with Tee(os.path.join(d, "terminal_output.txt")):
    Xtr, ytr, Xte, yte = load_data()
    N, D = Xtr.shape
    Nte = Xte.shape[0]
    C = 3

    # C1/C2: standardise with training statistics
    mu, sd = zscore_fit(Xtr)
    Ztr, Zte = zscore_apply(Xtr, mu, sd), zscore_apply(Xte, mu, sd)

    def clf():
        return LogisticRegression(C=1.0, max_iter=1000)

    # folds constructed ONCE and reused for every CV evaluation
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def test_acc(cols_tr, cols_te):
        return clf().fit(cols_tr, ytr).score(cols_te, yte)

    # ---- Baseline: all 13 standardised features
    base = test_acc(Ztr, Zte)
    header("P4: baseline (all 13 standardised features)")
    print(f"baseline test accuracy = {base:.4f}  ({round(base * Nte)}/{Nte})")
    print(f"within-one-sample threshold = baseline - 1/{Nte} = {base - 1 / Nte:.4f}")

    # ---- (A) PCA
    lam, U = eig_sorted(sample_cov(Ztr))
    acc_A = {k: test_acc(Ztr @ U[:, :k], Zte @ U[:, :k]) for k in KS}

    # ---- (B) Filter: F-score on training set
    m = Ztr.mean(0)
    num = np.zeros(D); den = np.zeros(D)
    for c in range(C):
        Zc = Ztr[ytr == c]
        mc = Zc.mean(0)
        num += len(Zc) * (mc - m) ** 2
        den += ((Zc - mc) ** 2).sum(0)
    F = (num / (C - 1)) / (den / (N - C))
    rank = np.argsort(-F, kind="stable")
    sel_B = {k: sorted(rank[:k].tolist()) for k in KS}
    acc_B = {k: test_acc(Ztr[:, sel_B[k]], Zte[:, sel_B[k]]) for k in KS}

    header("P4: F-score ranking (best first)")
    for i, j in enumerate(rank, 1):
        print(f"{i:2d}. {FEATURES[j]:<30s} F = {F[j]:.4f}")

    # ---- (C) Wrapper: our own SFS (increasing column index, keep first max)
    def cv_score(cols):
        return cross_val_score(clf(), Ztr[:, cols], ytr, cv=folds,
                               scoring="accuracy").mean()

    order, step_scores, n_eval = [], [], 0
    for step in range(max(KS)):
        best_j, best_s = None, -1.0
        for j in range(D):                      # increasing column index
            if j in order:
                continue
            s = cv_score(order + [j]); n_eval += 1
            if s > best_s:                      # strict '>' keeps the FIRST maximum
                best_j, best_s = j, s
        order.append(best_j); step_scores.append(best_s)
    sel_C = {k: sorted(order[:k]) for k in KS}
    acc_C = {k: test_acc(Ztr[:, sel_C[k]], Zte[:, sel_C[k]]) for k in KS}

    header("P4: SFS selection order and CV accuracy per step")
    for i, (j, s) in enumerate(zip(order, step_scores), 1):
        print(f"step {i}: add {FEATURES[j]:<30s} 5-fold CV acc = {s:.4f}")
    exp_evals = sum(D - k for k in range(max(KS)))
    print(f"candidate subsets evaluated: {n_eval} (derivation says 13+12+...+6 = {exp_evals}); "
          f"model fits = {n_eval * 5}")

    # ---- cross-check with sklearn SequentialFeatureSelector
    header("P4: cross-check with sklearn SequentialFeatureSelector")
    all_match = True
    for k in KS:
        sfs = SequentialFeatureSelector(clf(), n_features_to_select=k, direction="forward",
                                        scoring="accuracy", cv=folds).fit(Ztr, ytr)
        sk = np.flatnonzero(sfs.get_support()).tolist()
        ok = sk == sel_C[k]
        all_match &= ok
        print(f"k = {k}: own == sklearn : {ok}")
    print(f"all sets identical: {all_match}")

    # ---- results table
    header("P4: test accuracy table")
    tab = pd.DataFrame({"k": KS,
                        "(A) PCA": [acc_A[k] for k in KS],
                        "(B) Filter F-score": [acc_B[k] for k in KS],
                        "(C) Wrapper SFS": [acc_C[k] for k in KS]})
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tab.round(4).to_csv(os.path.join(d, "p4_table.csv"), index=False)
    print("\nSame table as number correct out of 54:")
    cnt = tab.copy()
    for c in tab.columns[1:]:
        cnt[c] = (tab[c] * Nte).round().astype(int)
    print(cnt.to_string(index=False))
    print(f"baseline: {round(base * Nte)}/{Nte}")

    header("P4: features chosen by (B) and (C)")
    for k in (3, 8):
        b = [FEATURES[j] for j in sel_B[k]]
        c = [FEATURES[j] for j in sel_C[k]]
        shared = sorted(set(b) & set(c))
        print(f"k = {k}")
        print(f"  (B) filter : {b}")
        print(f"  (C) wrapper: {c}")
        print(f"  shared ({len(shared)}): {shared}")

    header("P4: smallest k within one test sample of baseline")
    thr = base - 1 / Nte
    for nm, a in (("(A) PCA", acc_A), ("(B) Filter", acc_B), ("(C) Wrapper", acc_C)):
        ok = [k for k in KS if a[k] >= thr - 1e-12]
        print(f"  {nm:<12s}: {ok[0] if ok else 'none in table'}")
    p = base
    print(f"\nBinomial standard error of one accuracy near {p:.3f} with n={Nte}: "
          f"{np.sqrt(p * (1 - p) / Nte):.4f}  (one sample = {1 / Nte:.4f})")

    # ================================================================ plots
    def p_acc(ax):
        ax.plot(KS, [acc_A[k] for k in KS], "o-", label="(A) PCA")
        ax.plot(KS, [acc_B[k] for k in KS], "s-", label="(B) Filter (F-score)")
        ax.plot(KS, [acc_C[k] for k in KS], "^-", label="(C) Wrapper (SFS)")
        ax.axhline(base, ls="--", color="k", label=f"baseline (all 13) = {base:.3f}")
        ax.set_xlabel("k (number of features / components)"); ax.set_ylabel("test accuracy")
        ax.set_xticks(KS); ax.set_title("P4: test accuracy vs k"); ax.legend(loc="lower right")

    def p_F(ax):
        ax.bar(range(D), F[rank], color="teal", label="F-score $F_j$")
        ax.set_xticks(range(D)); ax.set_xticklabels(np.array(FEATURES)[rank], rotation=75,
                                                    ha="right", fontsize=7)
        ax.set_xlabel("feature (ranked)"); ax.set_ylabel("F-score")
        ax.set_title("P4: F-score ranking (filter)"); ax.legend()

    def p_sel(ax):
        M = np.zeros((D, 2 * len(KS)))
        for i, k in enumerate(KS):
            M[sel_B[k], i] = 1
            M[sel_C[k], len(KS) + i] = 1
        ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=1.3)
        ax.set_yticks(range(D)); ax.set_yticklabels(FEATURES, fontsize=7)
        ax.set_xticks(range(2 * len(KS)))
        ax.set_xticklabels([f"B k={k}" for k in KS] + [f"C k={k}" for k in KS],
                           rotation=60, fontsize=7)
        ax.axvline(len(KS) - 0.5, color="red")
        ax.set_xlabel("method (B = filter, C = wrapper) and k"); ax.set_ylabel("feature")
        ax.set_title("P4: which features are selected (dark = chosen)")

    def p_sfs(ax):
        ax.plot(range(1, len(step_scores) + 1), step_scores, "o-", color="tab:green",
                label="5-fold CV accuracy of SFS set")
        ax.set_xlabel("SFS step (number of features)"); ax.set_ylabel("CV accuracy (train)")
        ax.set_xticks(range(1, len(step_scores) + 1)); ax.set_title("P4: SFS greedy path")
        ax.legend()

    single(p_acc, os.path.join(figdir, "p4_accuracy_vs_k.png"))
    single(p_F, os.path.join(figdir, "p4_fscore_ranking.png"), (9, 5.5))
    single(p_sel, os.path.join(figdir, "p4_selected_features.png"), (8, 6))
    single(p_sfs, os.path.join(figdir, "p4_sfs_path.png"))
    onepage([p_acc, p_F, p_sel, p_sfs], os.path.join(SUMMARY, "P4_onepage.png"),
            "P4 — PCA vs feature selection (one-page summary)", 2, 2)
