# CS F320 – Assignment 2: Principal Component Analysis (Wine data)

All four problems (P1–P4) implemented in Python following the assignment's
conventions C1–C6 (train-only fitting, z-score with N−1, sample covariance,
`eigh` + sign rule, explained-variance ratio, 4-dp reporting).

## Folder structure
```
FODS_A2/
├── data/                       wine_train.csv, wine_test.csv
├── src/
│   ├── common.py               C1–C6 helpers (zscore, sample_cov, eig_sorted + sign fix, Tee logger, plot helpers)
│   ├── p1_pca_first_principles.py
│   ├── p2_variance_vs_components.py
│   ├── p3_preprocessing_effect.py
│   └── p4_pca_vs_feature_selection.py
├── outputs/
│   ├── P1/ … P4/               terminal_output.txt, figures/ (separate plots), CSV tables
│   └── ALL_terminal_output.txt combined terminal log of all problems
├── summary_pages/              P1_onepage.png … P4_onepage.png  (all plots of a problem on ONE page, for viva)
├── ANSWERS.md                  written paragraphs required by the assignment
├── run_all.py
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python run_all.py            # runs P1–P4, saves terminal output + figures
# or one problem:  cd src && python p1_pca_first_principles.py
```
Every script prints to the terminal **and** saves the identical text to `outputs/Pk/terminal_output.txt`.

## Headline results
| Problem | Key numbers |
|---|---|
| P1 | tr(S)=13.0000, ‖UᵀU−I‖_F≈3e-15, own PCA == sklearn (≈5e-15). PC1: flavanoids, total_phenols, OD280; PC2: color_intensity, alcohol, proline |
| P2 | k_elbow=3 (R=0.6595), k_0.80=5, k_0.90=7, k_0.95=10, Kaiser=3; identity error ≈9e-16 |
| P3 | r1: V1 0.9978 / V2 0.4271 / V3 0.3573; 5-NN acc: 0.7222 / 0.9815 / 0.9444 |
| P4 | baseline 0.9815; PCA reaches it at k=3, filter k=5, wrapper k=8 |

## Viva cheat-sheet
* **Why eigenvectors?** Maximise uᵀSu s.t. uᵀu=1 → Lagrangian → Su=λu; variance along u_j is λ_j.
* **Why tr(S)=13?** z-scored ⇒ S is the correlation matrix, diagonal = 1.
* **Reconstruction error = discarded eigenvalues:** J_train(k)=Σ_{j>k}λ_j; J_test is non-increasing but not equal to that sum.
* **Why V1 fails:** proline (variance ≈ 94,907) dominates, so PC1 ≈ proline.
* **PCA vs selection:** PCA is unsupervised and mixes all features; filter/wrapper use labels but keep original features. 54 test samples ⇒ 1 sample ≈ 1.85 pts, so small gaps are noise.
