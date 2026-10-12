"""Write LaTeX tables of per-split results for the supplementary material
(../supp_tables.tex), read directly from the revision run outputs."""
import json
import os

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
MODELS = ["RF", "XGBoost", "LightGBM", "CNN"]
VARIANTS = [("cv_all_models", "All 36 bands"), ("cv_all_models_noSAR", "Without SAR"),
            ("cv_all_models_noOPT", "Without optical"), ("cv_all_models_noTER", "Without terrain")]
METHODS = [("XGB_TreeSHAP", "XGB TreeSHAP"), ("XGB_Perm", "XGB perm."),
           ("CNN_IG", "CNN IG"), ("CNN_GradSHAP", "CNN grad.\\ SHAP"), ("CNN_Perm", "CNN perm.")]


def load(name):
    with open(os.path.join(RUNS, name, "results.json")) as f:
        return json.load(f)


out = []
out.append(r"""\begin{table}[htbp]
  \centering
  \caption{Test AUC on each of the five repeated spatial splits. CNN values
           are means over three seeds.}
  \label{tab:s_splits}
  \small
  \begin{tabular}{@{}llccccccc@{}}
    \toprule
    Bands & Model & Split 0 & Split 1 & Split 2 & Split 3 & Split 4 & Mean & SD \\
    \midrule""")
for i, (run, label) in enumerate(VARIANTS):
    r = load(run)
    for j, m in enumerate(MODELS):
        v = r[m]["per_split"]
        first = label if j == 0 else ""
        out.append(f"    {first} & {m} & " + " & ".join(f"{x:.3f}" for x in v) +
                   f" & {r[m]['mean']:.3f} & {r[m]['std']:.3f} \\\\")
    if i < len(VARIANTS) - 1:
        out.append(r"    \midrule")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

x = load("xai_agreement")
out.append(r"""\begin{table}[htbp]
  \centering
  \caption{Spearman rank correlation between explanation methods on each
           repeated split. W: within-model pair; C: cross-model pair.}
  \label{tab:s_rho}
  \small
  \begin{tabular}{@{}lcccccccc@{}}
    \toprule
    Pair & Type & Split 0 & Split 1 & Split 2 & Split 3 & Split 4 & Mean & SD \\
    \midrule""")
name = dict(METHODS)
for key, v in x["pairs"].items():
    a, b = key.split("|")
    t = "W" if v["type"] == "within-model" else "C"
    out.append(f"    {name[a]} -- {name[b]} & {t} & " +
               " & ".join(f"{r:.2f}" for r in v["per_split"]) +
               f" & {v['mean']:.2f} & {v['std']:.2f} \\\\")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
out.append(r"""\begin{table}[htbp]
  \centering
  \caption{Stability of each explanation method across splits: Spearman
           correlation of its band ranking between pairs of splits (10 pairs).}
  \label{tab:s_stab}
  \small
  \begin{tabular}{@{}lccc@{}}
    \toprule
    Method & Mean & Min & Max \\
    \midrule""")
for m, lab in METHODS:
    s = x["cross_split_stability"][m]
    out.append(f"    {lab} & {s['mean']:.2f} & {s['min']:.2f} & {s['max']:.2f} \\\\")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
with open(os.path.join(os.path.dirname(__file__), "..", "supp_tables.tex"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote supp_tables.tex")
