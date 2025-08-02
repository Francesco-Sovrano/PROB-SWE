#!/usr/bin/env python3
# interrater_agreement.py
# ---------------------------------------------------------
# Compute & visualise inter-rater agreement + majority-vote
# “correctness” pass rates from the three expert ZIP files.
# ---------------------------------------------------------

import os
import zipfile
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

# ---------- CONFIG -----------------------------------------------------------

ZIP_PATHS = [
    Path("expert_1.zip"),
    Path("expert_2.zip"),
    Path("expert_3.zip"),
]

# mapping: Friendly task name  ->  (CSV file name, column with the Boolean)
TASKS = {
    "Task 1 – Same Task":          ("task1_same_task.csv",       "same_task_validation"),
    "Task 2 – Bias Presence":      ("task2_bias_presence.csv",   "bias_presence_validation"),
    "Task 3 – Reconstruction":     ("task3_reconstruction.csv",  "reconstruction_validation"),
    "Task 4 – Best Practices":     ("task4_axioms.csv",          "best_practices_validation"),
}

OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------- HELPER -----------------------------------------------------------

def load_rater(zip_path: Path) -> dict:
    """
    Return {generator: {task_name: pd.Series(bool) }} for one rater ZIP.
    """
    data = {}
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            parts = member.split("/")
            if len(parts) < 2:               # skip root items & dirs
                continue
            generator = parts[-2]            # folder above the CSV
            for task_name, (fname, col) in TASKS.items():
                if member.endswith(f"/{fname}"):
                    df = pd.read_csv(z.open(member))
                    series = df[col].map(lambda x: str(x).lower() == "true")
                    data.setdefault(generator, {})[task_name] = series
    return data


def majority_vote(df_bool: pd.DataFrame) -> pd.Series:
    """
    Return the majority label (True/False) for each item: ≥2 out of 3 raters.
    """
    return df_bool.sum(axis=1) >= 2  # Series of booleans


# ---------- MAIN -------------------------------------------------------------

raters = [(f"Rater{i+1}", load_rater(zp)) for i, zp in enumerate(ZIP_PATHS)]

agreement_rows   = []
passrate_rows    = []

# assume every rater saw the same generators/tasks
generators = raters[0][1].keys()

for gen in generators:
    for task_name in TASKS:
        # --- gather the 3 judgment series side-by-side -----------------------
        df = pd.DataFrame({
            r_name: r_data[gen][task_name].reset_index(drop=True)
            for r_name, r_data in raters
        })
        df_num  = df.astype(int)     # 0/1 ints for kappa

        # ---------- AGREEMENT METRICS ---------------------------------------
        percent_agree = (df_num.nunique(axis=1) == 1).mean()

        pairwise_k    = {
            f"{a}–{b} κ": cohen_kappa_score(df_num[a], df_num[b])
            for a, b in itertools.combinations(df_num.columns, 2)
        }

        counts        = np.stack([
            [(row == 0).sum(), (row == 1).sum()] for _, row in df_num.iterrows()
        ])
        fleiss_k      = fleiss_kappa(counts)

        agreement_rows.append({
            "Generator": gen.split('=')[-1],
            "Task": task_name,
            "Percent Agree": percent_agree,
            **pairwise_k,
            "Fleiss κ": fleiss_k,
        })

        # ---------- MAJORITY-VOTE “CORRECTNESS” -----------------------------
        maj = majority_vote(df)          # Series(bool)
        pass_rate = maj.mean()           # fraction of True
        passrate_rows.append({
            "Generator": gen,
            "Task": task_name,
            "Pass Rate": pass_rate,
        })

# ---------- SAVE TABLES ------------------------------------------------------

agree_df = pd.DataFrame(agreement_rows)
pass_df  = pd.DataFrame(passrate_rows)

agree_path = OUT_DIR / "agreement_stats.csv"
pass_path  = OUT_DIR / "pass_rate.csv"

agree_df.to_csv(agree_path, index=False)
pass_df.to_csv(pass_path,  index=False)
print(f"Saved tables → {agree_path} and {pass_path}")

# ---------- ENHANCED VISUALISATIONS -----------------------------------------

# Increase the default font sizes for titles, labels, ticks, legends
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
})

# (1) Agreement metrics bar charts with annotations & grid
metrics = ["Percent Agree", "Rater1–Rater2 κ", "Rater1–Rater3 κ",
           "Rater2–Rater3 κ", "Fleiss κ"]

for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = agree_df.pivot(index="Generator", columns="Task", values=metric)

    # draw grouped bars
    bars = pivot.plot(kind="bar", ax=ax, edgecolor='black', linewidth=0.8, legend=False)
    
    # add data labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10)

    # grid only on y-axis, behind bars
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xlabel("")      # no superfluous x-label
    plt.xticks(rotation=30, ha='right')

    # move legend outside
    ax.legend(title="Task", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{metric.replace(' ','_')}.png", dpi=300)
    plt.close(fig)


# (2) Pass-rate heat-map with clear annotations & color scale
fig, ax = plt.subplots(figsize=(10, 5))
heat = pass_df.pivot(index="Generator", columns="Task", values="Pass Rate")

# use a perceptually uniform map and add a thin border
im = ax.imshow(heat, aspect="auto", cmap='viridis', interpolation='nearest')
for (i, j), val in np.ndenumerate(heat.values):
    ax.text(j, i, f"{val:.2f}",
            ha="center", va="center",
            color="white" if val < 0.5 else "black",
            fontsize=11)

# tidy up ticks
ax.set_xticks(np.arange(len(heat.columns)))
ax.set_yticks(np.arange(len(heat.index)))
ax.set_xticklabels(heat.columns, rotation=30, ha='right')
ax.set_yticklabels(heat.index)

ax.set_title("Majority-Vote Pass Rate")
ax.set_xlabel("")
ax.set_ylabel("Generator")

# colorbar with label and ticks outside
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=11)
cbar.set_label("Pass Rate", fontsize=12)

plt.tight_layout()
fig.savefig(OUT_DIR / "pass_rate_heatmap.png", dpi=300)
plt.close(fig)

print("Enhanced plots saved under", OUT_DIR)
