import os
import glob
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # ← use seaborn

model_mapping = {
	"gpt-4.1-nano": "gpt-4.1-nano",
	"gpt-4.1-mini": "gpt-4.1-mini",
	"gpt-4o-mini": "gpt-4o-mini",
	"llama-3.1-8b-instant": "llama-3.1",
	"llama-3.3-70b-versatile": "llama-3.3",
	"deepseek-r1-distill-llama-70b": "deepseek-r1",
}

# Load and compute correlations as before
file_pattern = 'generated_data/2_llm_outputs_model=*.data_model_list=[[]*[]].csv'
files = glob.glob(file_pattern)

results = []
for f in files:
	model = os.path.basename(f).split('model=')[1].split('.data_model_list')[0]
	df = pd.read_csv(f)
	if 'sensitive_to_bias' not in df.columns:
		continue
	# Compute lengths
	df['biased_len'] = df['biased'].map(lambda x: len(x.split(' ')))
	df['unbiased_len'] = df['unbiased'].map(lambda x: len(x.split(' ')))
	df['delta_len'] = df['biased_len'] - df['unbiased_len']
	for metric in [
		'pair_similarity',
		'pair_levenshtein_distance',
		'agreement_rate',
		'unbiased_prompt_reconstruction_similarity',
		'unbiased_len',
		'biased_len',
		'delta_len',
	]:
		if metric in df.columns:
			x = df['sensitive_to_bias']
			y = df[metric]
			mask = x.notna() & y.notna()
			if mask.sum() > 1:
				# y_masked = y[mask]
				# if metric in ['unbiased','biased']:
				# 	y_masked = y_masked.map(len)
				corr, p = pearsonr(x[mask], y[mask])
			else:
				corr, p = np.nan, np.nan
			results.append((model, metric, corr, p))

	# Compute and print statistics about biased/unbiased dilemma lengths by bias_name
	stats = (
		df.groupby('bias_name')[['delta_len']]
		.agg(['mean', 'std', 'median', 'count'])
		.round(2)
	)

	print(f"\n=== {model_mapping.get(model, model)} ===")
	print(stats)

res_df = pd.DataFrame(results, columns=['model','metric','corr','p_value'])
res_df["model"] = res_df["model"].map(model_mapping).fillna(res_df["model"])
pivot = res_df.pivot(index='metric', columns='model', values='corr')

# build an annotation DataFrame with stars
annot = pivot.copy().astype(str)
for i in pivot.index:
	for j in pivot.columns:
		corr = pivot.loc[i,j]
		p    = res_df[
			(res_df.model==j)&(res_df.metric==i)
		]['p_value'].values[0]
		star = '**' if p<0.001 else '*' if p<0.05 else ''
		if np.isnan(corr):
			annot.loc[i,j] = ""
		else:
			annot.loc[i,j] = f"{corr:.2f}{star}"

# descriptive labels
descriptive_labels = {
	'agreement_rate': 'Intra-Model\nAgreement Rate',
	'pair_levenshtein_distance': 'Pair\nLevenshtein Dist.',
	'pair_similarity': 'Pair\nCosine Similarity',
	'unbiased_prompt_reconstruction_similarity': 'Program–Dilemma\nAlignment',
	'unbiased_len': 'Unbiased\nDilemma Length',
	'biased_len': 'Biased\nDilemma Length',
	'delta_len': 'Delta (B.-Unb.)\nDilemma Length',
}
# pivot.index = [descriptive_labels[m] for m in pivot.index]  # rename labels

# build a rank map that understands BOTH keys and pretty labels
order_map = {}
for i, (k, lbl) in enumerate(descriptive_labels.items()):
    order_map[k] = i          # original key
    order_map[lbl] = i        # pretty label

# make a view for plotting only
pivot = (pivot
           .rename(index=descriptive_labels)  # rename where possible; leaves others alone
           .sort_index(
               key=lambda idx: idx.map(lambda x: order_map.get(x, float("inf"))),
               kind="stable"                  # keep relative order of items not in your list
           ))

# keep annot aligned if it's a pandas object
annot = (annot.rename(index=descriptive_labels)
                   .reindex(pivot.index)
              if hasattr(annot, "reindex") else annot)


# Plot with seaborn
plt.figure(figsize=(8, 3.5))

sns.heatmap(
	pivot,
	cmap='vlag',
	vmin=-1, vmax=1,
	annot=annot,
	fmt="",
	linewidths=0.5,
	linecolor='gray',
	cbar_kws={'label': 'Pearson r'}
)

plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("")
plt.ylabel("")
# plt.title('Heatmap of Correlations (Bias Sensitivity vs. Proxy Metrics)', pad=15, fontsize=14)

# Save as PDF
output_path = 'generated_data/7_sensitivity_to_correctness_correlation_analysis.pdf'
plt.tight_layout()
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Saved heatmap to {output_path}")
