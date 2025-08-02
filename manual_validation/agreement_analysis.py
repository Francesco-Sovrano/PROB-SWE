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

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

# ---- Force vector text embedding ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

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

OUT_DIR = Path("./analysis_results")
OUT_DIR.mkdir(exist_ok=True)

model_mapping = {
	"gpt-4.1-nano": "gpt-4.1-nano",
	"gpt-4.1-mini": "gpt-4.1-mini",
	"gpt-4o-mini": "gpt-4o-mini",
	"llama-3.1-8b-instant": "llama-3.1",
	"llama-3.3-70b-versatile": "llama-3.3",
	"deepseek-r1-distill-llama-70b": "deepseek-r1",
}

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
					print(zip_path, member)
					df = pd.read_csv(
						z.open(member),
						engine='python',         # use the more forgiving Python parser
						# on_bad_lines='skip',     # skip rows with the wrong number of fields
						dtype=str                # read everything as string so you don’t lose data
					)
					series = df[col].map(lambda x: str(x).lower() == "true")
					data.setdefault(generator, {})[task_name] = series
	return data

def unanimity_vote(df_bool):
	return df_bool.mean(axis=1) == 1.0  # Series of booleans

def majority_vote(df_bool):
	return df_bool.mean(axis=1) > 0.5  # Series of booleans


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
			"Generator": model_mapping[gen.split('=')[-1]],
			"Task": task_name,
			"Percent Agree": percent_agree,
			**pairwise_k,
			"Fleiss κ": fleiss_k,
		})

		# ---------- VOTE-BASED “CORRECTNESS” -----------------------------
		passrate_rows.append({
			"Generator": gen,
			"Task": task_name,
			"Pass Rate": majority_vote(df).mean(), # fraction of True
			"Unanimity Vote": unanimity_vote(df).mean() # fraction of True
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
metrics = ["Percent Agree"]

for metric in metrics:
	fig, ax = plt.subplots(figsize=(12, 3))
	pivot = agree_df.pivot(index="Generator", columns="Task", values=metric)

	# add a "Mean" row: the average metric across all generators for each Task
	pivot.loc["all"] = pivot.mean(axis=0)

	# draw grouped bars
	bars = pivot.plot(kind="bar", ax=ax, edgecolor='black', linewidth=1, legend=False)
	
	# add data labels on top of each bar
	for p in ax.patches:
		height = p.get_height()
		ax.annotate(f"{height:.2f}",
					(p.get_x() + p.get_width() / 2, height),
					ha='center', va='bottom', fontsize=9, bbox=dict(
						boxstyle="round,pad=0.2",
						facecolor="white",
						edgecolor="none",
						alpha=0.9
					))

	# grid only on y-axis, behind bars
	ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
	ax.set_axisbelow(True)

	ax.set_title(metric)
	ax.set_ylabel(metric)
	ax.set_xlabel("")      # no superfluous x-label
	plt.xticks(rotation=0, ha='center')

	# move legend outside
	ax.legend(title="Task", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
	plt.tight_layout()
	fig.savefig(OUT_DIR / f"{metric.replace(' ','_')}.pdf", dpi=300)
	plt.close(fig)


# (2) Pass-rate heat-map with clear annotations & color scale

# first, extract the part after "=" in your raw Generator strings
pass_df["Generator"] = (
	pass_df["Generator"]
	.astype(str)
	.str.split("=", n=1)
	.str[-1]
)

# Majority-vote pivot
heat_maj = (
    pass_df.pivot(index="Task", columns="Generator", values="Pass Rate")
          .rename(columns=model_mapping)
)
heat_maj["all"] = heat_maj.mean(axis=1)

# Unanimity-vote pivot
heat_uni = (
    pass_df.pivot(index="Task", columns="Generator", values="Unanimity Vote")
          .rename(columns=model_mapping)
)
heat_uni["all"] = heat_uni.mean(axis=1)

# shared color scale (0–1 if these are rates; change if needed)
vmin, vmax = 0.0, 1.0

# --- figure + axes ---
fig = plt.figure(figsize=(12, 4))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])  # shared colorbar axis

# Majority
h0 = sns.heatmap(
    heat_maj, ax=ax0,
    cmap="icefire", vmin=vmin, vmax=vmax,
    annot=True, fmt=".2f",
    linewidths=0.5, linecolor="white",
    cbar=False
)
ax0.set_title("Majority-Vote Pass Rate by Task", pad=12)
ax0.set_ylabel("")  # use figure-level ylabel instead
ax0.set_xlabel("Data Generator")
ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0)
ax0.set_xticklabels(ax0.get_xticklabels(), rotation=30, ha="right")

# Unanimity
h1 = sns.heatmap(
    heat_uni, ax=ax1,
    cmap="icefire", vmin=vmin, vmax=vmax,
    annot=True, fmt=".2f",
    linewidths=0.5, linecolor="white",
    cbar=True, cbar_ax=cax
)
ax1.set_title("Unanimity-Vote Pass Rate by Task", pad=12)
ax1.set_ylabel("")
ax1.set_xlabel("Data Generator")
ax1.set_yticklabels([])  # hide duplicate y ticks on the right
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

# label + style the shared colorbar
cbar = h1.collections[0].colorbar
cbar.set_label("Rate", rotation=90, labelpad=10)
cbar.ax.tick_params(labelsize=11)

# plt.tight_layout()

plt.savefig(OUT_DIR / "pass_rates_combined.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("Enhanced plots saved under", OUT_DIR)
