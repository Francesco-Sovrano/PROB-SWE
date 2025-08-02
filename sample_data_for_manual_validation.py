import argparse
import os
import pandas as pd
import numpy as np

def sample_bias_entries(
	input_path: str,
	output_dir: str,
	seed: int = 42,
	n: int = 5
) -> None:
	"""
	Reads a CSV file with bias data, samples N entries per bias_name, and writes out
	selected columns to separate CSV files for each validation task.

	Args:
		input_path: Path to the input CSV file.
		output_dir: Directory where output CSVs will be saved.
		seed: Random seed for reproducibility.
		n: Number of samples per bias_name.
	"""
	# Load data
	df = pd.read_csv(input_path)

	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	# Set seed
	np.random.seed(seed)

	# Define columns of interest
	cols = [
		'bias_name',
		'prompt_with_bias',
		'prompt_without_bias',
		'correct_option',
		'bias_is_towards_incorrect_only_in_biased_version',
		'reconstructed_unbiased_prompt',
		'axioms_description'
	]

	# Group and sample
	sampled = (
		df.groupby('bias_name', group_keys=False)
		  .apply(lambda g: g.sample(n=min(n, len(g)), random_state=seed))
		  .reset_index(drop=True)
	)

	# Select only needed columns
	sampled = sampled[cols]

	# Save full sampled set
	full_path = os.path.join(output_dir, 'sampled_full.csv')
	sampled.to_csv(full_path, index=False)

	# Create subtables/views for each validation task
	# Task 1: same-task check
	t1 = sampled[['prompt_without_bias','prompt_with_bias']].copy()
	t1 = t1.rename(columns={
		'prompt_without_bias': 'version_A',
		'prompt_with_bias': 'version_B'
	})
	t1.insert(0, 'same_task_validation', '')  # evaluator fills True/False
	t1.to_csv(os.path.join(output_dir, 'task1_same_task.csv'), index=False)

	# Task 2: bias presence check
	t2 = sampled[[
		'bias_name',
		'prompt_without_bias',
		'prompt_with_bias',
		'correct_option',
		'bias_is_towards_incorrect_only_in_biased_version'
	]].copy()
	# Rename columns: prompt_without_bias -> version_A, prompt_with_bias -> version_B
	t2 = t2.rename(columns={
		'prompt_without_bias': 'version_A',
		'prompt_with_bias': 'version_B'
	})
	t2.insert(0, 'bias_presence_validation', '')  # evaluator fills True/False
	t2.to_csv(os.path.join(output_dir, 'task2_bias_presence.csv'), index=False)

	# Task 3: reconstructed vs unbiased prompt
	t3 = sampled[[
		'prompt_without_bias',
		'reconstructed_unbiased_prompt',
	]].copy()
	t3 = t3.rename(columns={
		'prompt_without_bias': 'prompt',
		'reconstructed_unbiased_prompt': 'reconstructed_prompt',
	})
	t3.insert(0, 'reconstruction_validation', '')  # evaluator fills True/False
	t3.to_csv(os.path.join(output_dir, 'task3_reconstruction.csv'), index=False)

	# Task 4: axioms validation
	t4 = sampled[['axioms_description']].copy()
	t4 = t4.rename(columns={
		'axioms_description': 'best_practices',
	})
	t4.insert(0, 'best_practices_validation', '')  # evaluator fills True/False
	t4.to_csv(os.path.join(output_dir, 'task4_axioms.csv'), index=False)

	print(f"Sampled data and task-specific tables saved to: {output_dir}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Sample biased entries for validation tasks.'
	)
	parser.add_argument(
		'--input',
		help='Path to the input CSV file containing bias entries.'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42,
		help='Random seed for reproducible sampling.'
	)
	parser.add_argument(
		'--n',
		type=int,
		default=5,
		help='Number of samples per bias_name.'
	)
	args = parser.parse_args()

	sample_bias_entries(
		args.input,
		os.path.join('manual_validation', '.'.join(args.input.split('/')[-1].split('.')[:-1])),
		seed=args.seed,
		n=args.n
	)
