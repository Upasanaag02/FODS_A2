"""Run P1-P4 in order; each saves its own terminal output + figures; also
writes one combined log: outputs/ALL_terminal_output.txt"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
scripts = ["p1_pca_first_principles.py", "p2_variance_vs_components.py",
           "p3_preprocessing_effect.py", "p4_pca_vs_feature_selection.py"]

for s in scripts:
    print(f"\n######## running {s} ########")
    subprocess.run([sys.executable, s], cwd=SRC, check=True)

with open(os.path.join(ROOT, "outputs", "ALL_terminal_output.txt"), "w", encoding="utf-8") as out:
    for p in ("P1", "P2", "P3", "P4"):
        out.write(f"\n\n########## {p} ##########\n")
        with open(os.path.join(ROOT, "outputs", p, "terminal_output.txt"), encoding="utf-8") as f:
            out.write(f.read())
print("\nDone. Combined log: outputs/ALL_terminal_output.txt ; one-page figures: summary_pages/")
