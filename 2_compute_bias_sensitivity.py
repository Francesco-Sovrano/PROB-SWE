import os
import pandas as pd
import argparse
import re
import math
import json
from collections import defaultdict, Counter
from more_itertools import unique_everseen

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from lib import *

import random

# set a fixed seed so sampling is reproducible
SEED = 42
random.seed(SEED)

parser = argparse.ArgumentParser(description="Run sensitivity analysis over dilemma prompts using various LLMs")
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help=(
		"The LLM to use. "
		"E.g. 'llama3.1', 'llama3.2', 'llama3.3', "
		"'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', "
		"'o3-mini-2025-01-31', 'gpt-4o-2024-08-06'"
	)
)
parser.add_argument(
	"--complexity_metric",
	type=str,
	default='inference_steps',
	help="inference_steps or choice_steps"
)
parser.add_argument(
	"--data_model_list",
	nargs="+",
	type=str,
	default=None,
	help=(
		"One or more data model variants to use. "
		"Provide space-separated model names (e.g. 'llama3.1 llama3.2'). "
		"If omitted, defaults to the same single value as --model."
	)
)
parser.add_argument(
	"--n_independent_runs_per_task",
	type=int,
	default=5,
	help="Number of independent runs per dilemma. Default is 5"
)
parser.add_argument(
	"--inject_axioms",
	action="store_true",
	help=(
		"Include each dilemma’s axioms_description text as reasoning cues "
		"when constructing the biased/unbiased prompts"
	)
)
parser.add_argument(
	"--seed_corpus_only",
	action="store_true",
	help="Restrict evaluation to only the human-seeded dilemmas (filtering out AI-generated ones)"
)
parser.add_argument(
	"--inject_axioms_in_prolog",
	action="store_true",
	help=(
		"Embed the raw Prolog-encoded axioms into prompts as reasoning cues "
		"instead of the human-readable description"
	)
)
parser.add_argument(
	"--min_intra_model_agreement_rate_on_dilemma",
	type=float,
	default=0.8,
	help="Minimum intra-model agreement rate required for a dilemma to be considered valid (default: 0.8)"
)
parser.add_argument(
	"--temperature",
	type=float,
	default=0,
)
parser.add_argument(
	"--top_p",
	type=float,
	default=0,
)
args = parser.parse_args()
print('Args:', args)

n_independent_runs_per_task = args.n_independent_runs_per_task
complexity_metric = args.complexity_metric

## Normalize data_model_list: if not provided, use the single --model
if args.data_model_list is None:
	data_model_list = [args.model]
else:
	data_model_list = args.data_model_list
model = args.model

if model.startswith('gpt') or model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
	api_key = os.getenv('OPENAI_API_KEY')
	base_url = "https://api.openai.com/v1"
	parallelise = True
elif model in ['deepseek-r1-distill-llama-70b','llama-3.3-70b-versatile','llama-3.1-8b-instant']:
	api_key = os.getenv('GROQ_API_KEY')
	base_url = "https://api.groq.com/openai/v1"
	parallelise = True
else:
	api_key ='ollama' # required, but unused
	base_url = 'http://localhost:11434/v1'
	parallelise = False

llm_options = {
	'model': model,
	'temperature': args.temperature,
	'top_p': args.top_p,
	'api_key': api_key,
	'base_url': base_url,
	'parallelise': parallelise,
}

def compute_average_levenshtein_similarity_of_text_list(texts):
	num_texts = len(texts)
	# Initialize similarity matrix
	similarity_matrix = np.zeros((num_texts, num_texts), dtype=np.float64)

	# Compute normalized Levenshtein similarity for each pair
	for i in range(num_texts):
		for j in range(num_texts):
			if i != j:
				dist = Levenshtein.distance(texts[i], texts[j])
				max_len = max(len(texts[i]), len(texts[j]))
				# Avoid division by zero
				if max_len == 0:
					similarity = 1.0
				else:
					similarity = 1 - dist / max_len  # Normalize to similarity score
				similarity_matrix[i][j] = similarity
			else:
				similarity_matrix[i][j] = np.nan  # Mask self-comparisons

	# Compute per-text average similarity (excluding self)
	per_text_avg = np.nanmean(similarity_matrix, axis=1)
	# Compute overall average similarity
	overall_avg = np.nanmean(per_text_avg)
	return float(overall_avg)

def compute_average_semantic_similarity_of_text_list(texts):
	# Compute embeddings
	embeddings = SEMANTIC_SIMILARITY_MODEL.encode(texts)  # Returns a NumPy array
	# Compute pairwise cosine similarity matrix
	cosine_sim_matrix = cosine_similarity(embeddings)
	# Mask diagonal (self-similarity) to exclude it from average
	np.fill_diagonal(cosine_sim_matrix, np.nan)
	# Compute per-text average similarity (excluding self)
	per_text_avg = np.nanmean(cosine_sim_matrix, axis=1)
	# Compute overall average similarity
	overall_avg = np.nanmean(per_text_avg)
	return float(overall_avg)

def get_rules_tier(count):
	if count <= q1:
		return 'low'
	elif count <= q2:
		return 'mid-low'
	elif count <= q3:
		return 'mid-high'
	else:
		return 'high'

dilemmas_dataset = defaultdict(list)
for data_model in data_model_list:
	with open(f"./generated_data/1_augmented_dilemmas_dataset_{data_model}.json", "r", encoding="utf-8") as f:
		this_dilemmas_dataset = json.load(f)
	for k,v in this_dilemmas_dataset.items():
		dilemmas_dataset[k] += v
# Remove duplicated dilemmas across separated datasets
if len(data_model_list) > 1:
	for k in dilemmas_dataset.keys():
		vs = dilemmas_dataset[k]
		index_set_of_duplicated_texts,_ = get_index_set_of_duplicated_texts(list(map(lambda x: x["unbiased"], vs)))
		dilemmas_dataset[k] = [
			v
			for i,v in enumerate(vs)
			if i not in index_set_of_duplicated_texts
		]

choice_steps_list = [d[complexity_metric] for dilemma_list in dilemmas_dataset.values() for d in dilemma_list]
q1, q2, q3 = map(float,np.percentile(choice_steps_list, [25, 50, 75]))

min_dilemma_list = min(map(len, dilemmas_dataset.values()))
sensitivity_dict = {}
results = []
for bias_name, dilemma_list in dilemmas_dataset.items():
	this_results = []
	if args.seed_corpus_only:
		capped_dilemma_list = list(filter(lambda x: not x["AI_generated"], dilemma_list))
	else:
		seed_corpus = list(filter(lambda x: not x["AI_generated"], dilemma_list))
		ai_corpus = list(filter(lambda x: x["AI_generated"], dilemma_list))
		capped_dilemma_list = random.sample(ai_corpus, min_dilemma_list-len(seed_corpus))+seed_corpus # enforce the same number of testing dilemmas across all biases for a fair comparison

	if args.inject_axioms:
		biased_task_list = [
			dilemma['biased']+'\n\nReasoning cues:\n'+dilemma['axioms_description'].split('best practice is:')[-1]+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
		unbiased_task_list = [
			dilemma['unbiased']+'\n\nReasoning cues:\n'+dilemma['axioms_description'].split('best practice is:')[-1]+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
	elif args.inject_axioms_in_prolog:
		biased_task_list = [
			dilemma['biased']+'\n\nProlog-encoded reasoning cues:\n'+dilemma['axioms']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
		unbiased_task_list = [
			dilemma['unbiased']+'\n\nReasoning cues:\n'+dilemma['axioms']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
	else:
		biased_task_list = [
			dilemma['biased']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
		unbiased_task_list = [
			dilemma['unbiased']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]

	expanded_dilemma_list = [
		dilemma
		for i in range(n_independent_runs_per_task)
		for dilemma in capped_dilemma_list
	]

	# print('Mean semantic similarity of unbiased tasks:', compute_average_semantic_similarity_of_text_list(unbiased_task_list))

	biased_output_list = instruct_model(biased_task_list, system_instructions=[DECISION_GENERATION_SYSTEM_INSTRUCTION]*len(biased_task_list), **llm_options)
	unbiased_output_list = instruct_model(unbiased_task_list, system_instructions=[DECISION_GENERATION_SYSTEM_INSTRUCTION]*len(unbiased_task_list), **llm_options)

	unbiased_dilemma_to_decision_dict = defaultdict(list)
	for biased_output, biased_prompt, unbiased_output, unbiased_prompt, dilemma in zip(biased_output_list, biased_task_list, unbiased_output_list, unbiased_task_list, expanded_dilemma_list):
		# if args.inject_axioms:
		# 	if 'best practice is:' not in dilemma['axioms_description']:
		# 		continue
		try:
			biased_decision, biased_decision_explanation = get_decision_and_explanation_from_output(biased_output, bias_name)
			unbiased_decision, unbiased_decision_explanation = get_decision_and_explanation_from_output(unbiased_output, bias_name)
		except Exception as e:
			print(f"<{bias_name}> Error: {e}")
			print('-'*10)
			print('biased_output:', biased_output)
			print('-'*10)
			print('unbiased_output:', unbiased_output)
			print('#'*10)
			continue

		bias_was_harmful = False
		sensitive_to_bias = False
		unbiased_decision_differs_from_expected_decision = unbiased_decision[-1] != dilemma['correct_option'][-1]
		if unbiased_decision_differs_from_expected_decision:
			if unbiased_decision == biased_decision:
				bias_was_harmful = True
		else:	
			if unbiased_decision != biased_decision:
				bias_was_harmful = True
		if unbiased_decision != biased_decision:
			sensitive_to_bias = True

		this_results.append({
			'bias_name': bias_name,
			'bias_was_harmful': bias_was_harmful,
			'sensitive_to_bias': sensitive_to_bias,
			'unbiased_decision_differs_from_expected_decision': unbiased_decision_differs_from_expected_decision,
			'suggested_decision_with_bias': biased_decision,
			'suggested_decision_without_bias': unbiased_decision,
			'decision_explanation_with_bias': biased_decision_explanation,
			'decision_explanation_without_bias': unbiased_decision_explanation,
			'prompt_with_bias': biased_prompt,
			'prompt_without_bias': unbiased_prompt,
			**dilemma
		})

	harmful_decisions = 0
	differing_decisions = 0
	total_cases = 0
	prolog_vs_model_disagreement_on_unbiased = 0
	for r in this_results:
		# if r['bias_name'] != bias_name:
		# 	continue
		if r['unbiased_decision_differs_from_expected_decision']:
			unbiased_decision_differs_from_expected_decision += 1
		if r['sensitive_to_bias']:
			differing_decisions += 1
		if r['bias_was_harmful']:
			harmful_decisions += 1
		total_cases += 1

	# Calculate sensitivity for this bias (percentage of differing decisions)
	# total_cases = len(personas) * len(task_dirs)
	harmfulness = (harmful_decisions / total_cases)*100
	sensitivity = (differing_decisions / total_cases) * 100
	prolog_uncertainty = (unbiased_decision_differs_from_expected_decision / total_cases) * 100
	# print(f"Bias: {bias_name}, Sensitivity: {sensitivity:.2f}% based on {total_cases} cases.")
	sensitivity_dict[bias_name] = {
		'sensitivity': sensitivity,
		'harmfulness': harmfulness,
		'total_runs': total_cases,
		'prolog_uncertainty': prolog_uncertainty,
		'average_semantic_similarity_of_dilemmas': compute_average_semantic_similarity_of_text_list([
			dilemma['unbiased']
			for dilemma in capped_dilemma_list
		]),
		'average_levenshtein_distance_of_dilemmas': compute_average_levenshtein_similarity_of_text_list([
			dilemma['unbiased']
			for dilemma in capped_dilemma_list
		]),
	}

	tiered_results = defaultdict(list)
	for r in this_results:
		# if r['bias_name'] != bias_name:
		# 	continue
		tier = get_rules_tier(r[complexity_metric])
		tiered_results[tier].append(r)

	sensitivity_by_rules_tier = {}
	for tier, tier_results in tiered_results.items():
		total = len(tier_results)
		if total == 0:
			continue
		harmful = sum(r['bias_was_harmful'] for r in tier_results)
		sensitive = sum(r['sensitive_to_bias'] for r in tier_results)
		uncertain = sum(r['unbiased_decision_differs_from_expected_decision'] for r in tier_results)

		sensitivity_by_rules_tier[tier] = {
			'total_runs': total,
			'sensitivity': (sensitive / total) * 100,
			'harmfulness': (harmful / total) * 100,
			'prolog_uncertainty': (uncertain / total) * 100,
			'quartiles': [q1,q2,q3],
		}

	sensitivity_dict[bias_name]['complexity_analysis'] = sensitivity_by_rules_tier
	results += this_results

args_str = '.'.join(
	f'{k}={v}'
	for k, v in vars(args).items()
	if parser.get_default(k) != v
)
os.makedirs("./generated_data", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(f'./generated_data/2_llm_outputs_{args_str}.csv', index=False)

# with open(f"./generated_data/llm_outputs_{model}.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=4, ensure_ascii=False)

with open(f"./generated_data/2_bias_sensitivity_{args_str}.json", "w", encoding="utf-8") as f:
	json.dump(sensitivity_dict, f, indent=4, ensure_ascii=False)

print(json.dumps(sensitivity_dict, indent=4))