#!/usr/bin/env python3
"""
Compact bias‑awareness heatmap for each GPAI system, with:
 - custom model labels
 - numerical annotations in each cell
 - a color‑blind‑friendly palette
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ---- Force vector text embedding ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

model_mapping = {
	"gpt-4.1-nano": "gpt-4.1-nano",
	"gpt-4.1-mini": "gpt-4.1-mini",
	"gpt-4o-mini":   "gpt-4o-mini",
	"llama-3.1-8b-instant":   "llama-3.1",
	"llama-3.3-70b-versatile": "llama-3.3",
	"deepseek-r1-distill-llama-70b": "deepseek-r1",
}

bias_label_mapping = {
	"anchoring bias": "Anchor.",
	"availability bias": "Avail.",
	"bandwagon effect":   "Bandwagon",
	"confirmation bias":   "Confirm.",
	"framing effect": "Framing",
	"hindsight bias": "Hindsight",
	"hyperbolic discounting": "Hyperbolic",
	"overconfidence bias": "Overconf.",
}

def load_bias_data(data_dir='generated_data'):
	"""Returns a DataFrame indexed by mapped model names, columns=numeric bias types."""
	rows = []
	for fname in os.listdir(data_dir):
		if not fname.endswith('.json') or fname.startswith('.'):
			continue

		parts = fname.split('model=')
		if len(parts) != 2 or not parts[1].endswith('.json'):
			continue
		model = parts[1][:-5]  # strip ".json"

		path = os.path.join(data_dir, fname)
		try:
			with open(path, 'r') as f:
				metrics = json.load(f)
			if not isinstance(metrics, dict):
				raise ValueError("expected JSON object")
		except Exception as e:
			print(f"Warning: skipping {fname}: {e}")
			continue

		numeric = {bias_label_mapping.get(k,k): v for k, v in metrics.items() if isinstance(v, (int, float))}
		if not numeric:
			print(f"Warning: no numeric metrics in {fname}, skipping")
			continue

		numeric['model'] = model_mapping.get(model,model)#.replace('-','\n')
		rows.append(numeric)

	if not rows:
		return pd.DataFrame()

	df = pd.DataFrame(rows).set_index('model')
	return df

def plot_heatmap(df, outpath="./generated_data/6_bias_awareness_analysis_heatmap.pdf"):
	"""Render a compact heatmap with manual annotations and the icefire palette,
	centering labels and adding a mean_score column as the last column."""
	# add mean_score column
	df = df.copy()
	df["All Biases"] = df.mean(axis=1)

	# reorder rows by ascending mean_score
	order = df["All Biases"].sort_values().index
	df = df.loc[order]

	# figure sizing
	fig, ax = plt.subplots(
		figsize=(df.shape[1] * 0.3 + 1, df.shape[0] * 0.1 + 1),
		dpi=150,
	)

	# draw heatmap (no annot, colorbar off)
	sns.heatmap(
		df,
		ax=ax,
		cmap='icefire',
		cbar=False,
		linewidths=0,
		linecolor="none",
		square=False,
		vmin=0, vmax=1
	)

	# per‐cell annotation with contrast, centered in cells at j+0.5, i+0.5
	vmax = df.values.max()
	for i in range(df.shape[0]):
		for j in range(df.shape[1]):
			val = df.iat[i, j]
			txt_color = "white" if 0.15 < val < 0.85 else "black"
			ax.text(
				j + 0.5, i + 0.5,                   # center of the cell
				f"{val:.2f}".replace("0.", "."),   # compact format
				ha="center", va="center",
				fontsize=5,
				color=txt_color
			)

	# axis labels & ticks
	ax.set_xticks([x + 0.5 for x in range(df.shape[1])])
	ax.set_xticklabels(df.columns, rotation=25, ha="right", fontsize=6)
	ax.set_yticks([y + 0.5 for y in range(df.shape[0])])
	ax.set_yticklabels(df.index, fontsize=6)
	ax.set_ylabel("")

	# remove spines
	for spine in ax.spines.values():
		spine.set_visible(False)

	# recreate colorbar without border
	sm = plt.cm.ScalarMappable(cmap='icefire', norm=plt.Normalize(vmin=0, vmax=1))
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
	cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
	cbar.set_ticklabels(["0", ".25", ".5", ".75", "1"])
	cbar.ax.tick_params(labelsize=6)
	cbar.outline.set_visible(False)  # strip border

	plt.tight_layout(pad=0.2)
	fig.savefig(outpath, dpi=300, bbox_inches="tight")
	plt.close(fig)

def main():
	df = load_bias_data('generated_data')
	if df.empty:
		print("No valid numeric bias JSON files found in 'generated_data/'.")
		return
	plot_heatmap(df)

if __name__ == '__main__':
	main()
