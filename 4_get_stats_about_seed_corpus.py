import os
import re
import random
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict, Counter
import json
import pandas as pd
import re
import argparse
import textwrap
from lib import *

# ---- Force vector text embedding ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

scenarios_dir = './seed_corpus'

# We assume that the top-level extracted folder (scenarios_dir) contains one directory per bias.
bias_dirs = [d for d in os.listdir(scenarios_dir) if os.path.isdir(os.path.join(scenarios_dir, d))]
stats = []
for bias in bias_dirs:
	bias_path = os.path.join(scenarios_dir, bias)
	# Assume two decision-making tasks per bias
	task_dirs = [d for d in os.listdir(bias_path) if os.path.isdir(os.path.join(bias_path, d))]
	
	for task in task_dirs:
		task_path = os.path.join(bias_path, task)
		# List txt files: we assume one file is the biased version and one is less/non biased.
		txt_files = [f for f in os.listdir(task_path) if f.endswith('.txt') and 'axioms' not in f]
		if len(txt_files) != 2:
			print(f"Unexpected number of txt files in {task_path}")
			continue
		
		if txt_files[0].startswith('0'):
			unbiased_file = txt_files[0]
			biased_file = txt_files[-1]
		elif txt_files[-1].startswith('0'):
			unbiased_file = txt_files[-1]
			biased_file = txt_files[0]
		
		if not (biased_file and unbiased_file):
			print(f"Could not determine biased vs non-biased in {task_path}")
			continue

		with open(os.path.join(task_path, biased_file), 'r', encoding='utf-8') as file:
			vignette_biased = file.read()
		with open(os.path.join(task_path, unbiased_file), 'r', encoding='utf-8') as file:
			vignette_unbiased = file.read()

		# print(run_prolog_script(os.path.join(task_path, biased_file.replace('.txt','.pl'))))
		biased_dict = {
			'type': 'biased',
			'n_chars': len(vignette_biased),
			'n_words': len(vignette_biased.split(' ')),
			**run_prolog_script(os.path.join(task_path, biased_file.replace('.txt','.pl')))
		}
		stats.append(biased_dict)

		unbiased_dict = {
			'type': 'unbiased',
			'n_chars': len(vignette_unbiased),
			'n_words': len(vignette_unbiased.split(' ')),
			**run_prolog_script(os.path.join(task_path, unbiased_file.replace('.txt','.pl')))
		}
		stats.append(unbiased_dict)

		assert biased_dict["choice"] == unbiased_dict["choice"], f'Choice mismatch for {biased_dict["file"]}'
		assert biased_dict["profiler"]['nodes'] == unbiased_dict["profiler"]['nodes'], f'Reasoning nodes mismatch for {biased_dict["file"]}'
		assert biased_dict["inferences"] == unbiased_dict["inferences"], f'Inference steps mismatch for {biased_dict["file"]}'

		print(json.dumps({'similarity': get_texts_similarity(vignette_biased, vignette_unbiased), 'levenshtein_distance': get_texts_similarity_with_Levenshtein(vignette_unbiased, vignette_biased), 'task': task_path}, indent=4))
		
# Compute averages for each category
def compute_averages(stat_list):
	total_chars = sum(item['n_chars'] for item in stat_list)
	total_words = sum(item['n_words'] for item in stat_list)
	total_inferences = sum(item["inferences"] for item in stat_list)
	total_nodes = sum(item["profiler"]['nodes'] for item in stat_list)
	count = len(stat_list)
	return total_chars / count if count > 0 else 0, total_words / count if count > 0 else 0, total_inferences / count if count > 0 else 0, total_nodes / count if count > 0 else 0

biased_stats = [s for s in stats if s['type'] == 'biased']
unbiased_stats = [s for s in stats if s['type'] == 'unbiased']

avg_biased_chars, avg_biased_words, avg_biased_inferences, avg_biased_nodes = compute_averages(biased_stats)
avg_unbiased_chars, avg_unbiased_words, avg_unbiased_inferences, avg_unbiased_nodes = compute_averages(unbiased_stats)
avg_all_chars, avg_all_words, avg_all_inferences, avg_all_nodes = compute_averages(stats)

# Prepare output string
output = (
	f"Average Character and Word Counts\n"
	f"----------------------------------\n"
	f"Biased:\n  Average characters: {avg_biased_chars:.2f}\n  Average words: {avg_biased_words:.2f}\n  Average inference steps: {avg_biased_inferences:.2f}\n  Average rules followed: {avg_biased_nodes:.2f}\n\n"
	f"Unbiased:\n  Average characters: {avg_unbiased_chars:.2f}\n  Average words: {avg_unbiased_words:.2f}\n  Average inference steps: {avg_unbiased_inferences:.2f}\n  Average rules followed: {avg_unbiased_nodes:.2f}\n\n"
	f"Overall:\n  Average characters: {avg_all_chars:.2f}\n  Average words: {avg_all_words:.2f}\n  Average inference steps: {avg_all_inferences:.2f}\n  Average rules followed: {avg_all_nodes:.2f}\n\n"
	'Biased-Dilemma Stats: '+json.dumps(biased_stats, indent=4)+'\n'
	'Unbiased-Dilemma Stats: '+json.dumps(unbiased_stats, indent=4)+'\n'
)

# Ensure results directory exists
results_dir = "./generated_data"
os.makedirs(results_dir, exist_ok=True)

# Save the output to a text file
output_file = os.path.join(results_dir, f"4_seed_corpus_stats.txt")
with open(output_file, "w") as f:
	f.write(output)

print("Averages computed and saved to", output_file)


# Directory containing generated data files
data_dir = './generated_data'

# Regex to capture model name and run count
FILENAME_PATTERN = re.compile(
	r"""^2_llm_outputs_model=(?P<model>.+?)(?=\.data_model_list=|\.n_independent_runs_per_task=|\.inject_axioms=|\.seed_corpus_only=|\.inject_axioms_in_prolog=|\.min_intra_model_agreement_rate_on_dilemma=|\.temperature=|\.top_p=|\.csv$)
		(?:\.data_model_list=(?P<data_model_list>.+?)(?=\.data_model_list=|\.n_independent_runs_per_task=|\.inject_axioms=|\.seed_corpus_only=|\.inject_axioms_in_prolog=|\.min_intra_model_agreement_rate_on_dilemma=|\.temperature=|\.top_p=|\.csv$))?
		(?:\.n_independent_runs_per_task=(?P<runs>\d+\.?\d*))?
		(?:\.inject_axioms=(?P<inject_axioms>True|False))?
		(?:\.seed_corpus_only=(?P<seed_corpus_only>True|False))?
		(?:\.inject_axioms_in_prolog=(?P<inject_axioms_in_prolog>True|False))?
		(?:\.min_intra_model_agreement_rate_on_dilemma=(?P<min_intra_model_agreement_rate_on_dilemma>\d+\.?\d*))?
		(?:\.temperature=(?P<temperature>\d+\.?\d*))?
		(?:\.top_p=(?P<top_p>\d+\.?\d*))?
		\.csv$
	""",
	re.VERBOSE
)

# Step 1: Find matching files and select highest runs per model
files_by_model = {}
for fname in os.listdir(data_dir):
	match = FILENAME_PATTERN.match(fname)
	if not match:
		continue
	if match.group('inject_axioms'):
		continue
	if match.group('seed_corpus_only'):
		continue
	if match.group('inject_axioms_in_prolog'):
		continue
	model = match.group('model')
	runs = int(match.group('runs')) if match.group('runs') else 5
	path = os.path.join(data_dir, fname)

	# Keep only the file with the highest run count per model
	prev = files_by_model.get(model)
	if prev is None or runs > prev['runs'] or len(prev['path']) < len(path):
		files_by_model[model] = {'runs': runs, 'path': path, 'model': model}

print('Files for 4_decision_agreement_on_seed_corpus:',json.dumps(files_by_model, indent=4))

# Step 2: Process each selected file
results = defaultdict(list)
for model, info in files_by_model.items():
	df = pd.read_csv(info['path'])
	
	# Filter rows: prompt_is_biased == False and unbiased_path not empty
	filtered = df[(df['unbiased_path'].notna()) & (df['unbiased_path'] != '')]
	
	# Group suggested_decision by unbiased_path
	grouped = filtered.groupby('unbiased_path')['suggested_decision_without_bias'].apply(list).to_dict()
	results[model] = grouped

# Step 3: Merge decisions across models for each path
merged = defaultdict(list)
for model_groups in results.values():
	for path, decisions in model_groups.items():
		merged[path].extend(decisions)

# print(json.dumps(merged, indent=4))

# Function to plot decision agreement as a stacked bar chart and save as PDF
def plot_decision_agreement(merged_decisions, output_path, wrap_width=30):
	"""
	Plots a stacked bar chart showing agreement of decisions A vs B for each unbiased_path, wrapping x-labels for readability, and saves as PDF.

	Parameters:
	merged_decisions (dict): Mapping from unbiased_path to list of suggested_decision values.
	output_path (str): Path to save the output PDF.
	wrap_width (int): Maximum characters per line for x-tick labels.
	"""
	paths = list(merged_decisions.keys())
	counts = {path: Counter(decisions) for path, decisions in merged_decisions.items()}
	a_counts = [counts[path].get('Option A', 0) for path in paths]
	b_counts = [counts[path].get('Option B', 0) for path in paths]

	x = range(len(paths))
	plt.figure(figsize=(8.5, 4))

	# Plot stacked bars with clear styling
	plt.bar(x, a_counts, label='Option A', edgecolor='white')
	plt.bar(x, b_counts, bottom=a_counts, label='Option B', edgecolor='white')

	# Annotate counts
	for idx in x:
		if a_counts[idx] > 0:
			plt.text(idx, a_counts[idx] / 2, str(a_counts[idx]), ha='center', va='center', fontsize=10, color='white')
		if b_counts[idx] > 0:
			plt.text(idx, a_counts[idx] + b_counts[idx] / 2, str(b_counts[idx]), ha='center', va='center', fontsize=10, color='white')

	# Create wrapped labels
	raw_labels = [f"{p.split('/')[-2][2:].replace('-',' ')}\n({p.split('/')[-3].split(' - ')[-1].replace('_',' ')})" for p in paths]
	# wrapped_labels = ["\n".join(textwrap.wrap(label, wrap_width)) for label in raw_labels]

	plt.xticks(x, raw_labels, rotation=45, ha='right', fontsize=11)
	plt.yticks(fontsize=11)
	# plt.xlabel('Dilemma', fontsize=12)
	plt.ylabel('Count of Decisions', fontsize=12)
	# plt.title('Decision Agreement by Dilemma', fontsize=14)

	# Add grid for readability
	plt.grid(axis='y', linestyle='--', alpha=0.6)

	# Legend inside
	plt.legend(
		loc="upper left",          # pick any corner you like
		framealpha=0.9,             # slight white box so text stays readable
		edgecolor="none",
		fontsize=11
	)
	# # Legend outside
	# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)

	plt.tight_layout()
	plt.savefig(output_path) 

plot_decision_agreement(merged, os.path.join(data_dir,'4_decision_agreement_on_seed_corpus.pdf'))

# # Step 3: Print grouped results
# for model, groups in results.items():
# 	print(model, len(groups))
# 	for path, decisions in groups.items():
# 		decisions = list(map(lambda x: x, decisions))
# 		print(len(decisions), path)