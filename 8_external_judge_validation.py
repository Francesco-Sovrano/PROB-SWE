import os
import pandas as pd
import argparse
import json
from collections import defaultdict, Counter
import numpy as np
import math

from lib import *

parser = argparse.ArgumentParser(description="Script to augment dilemma pairs by bias type and batch size")
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
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'llama3.1', 'llama3.2', 'llama3.3', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o3-mini-2025-01-31', 'gpt-4o-2024-08-06')"
)
parser.add_argument(
	"--n_independent_runs_per_task",
	type=int,
	default=5,
	help="Number of independent runs per task to assess agreement rate (default: 5)"
)
parser.add_argument(
	"--n_prolog_construction_reruns",
	type=int,
	default=2,
	help="Number of times to rerun validation for each AI-generated dilemma (default: 10)"
)
parser.add_argument(
	"--min_intra_model_agreement_rate_on_dilemma",
	type=float,
	default=0.8,
	help="Minimum intra-model agreement rate required for a dilemma to be considered valid (default: 0.8)"
)
args = parser.parse_args()
print('Args:', args)

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

llm_options_for_diverse_output = {
	'model': model,
	# 'temperature': 0,
	# 'top_p': 0,
	'api_key': api_key,
	'base_url': base_url,
	'parallelise': parallelise,
}

llm_options = {
	'model': model,
	'temperature': 0,
	'top_p': 0,
	'api_key': api_key,
	'base_url': base_url,
	'parallelise': parallelise,
}

def filter_dilemma_list(bias_name, bias_definition, dilemma_list):
	def _filter_by_bias_towards_incorrect(ds):
		print(f'<_filter_by_bias_towards_incorrect> Running on {len(ds)} dilemmas')
		
		system_instructions = [
			BIAS_TOWARDS_INCORRECT_SYSTEM_INSTRUCTION.format(
				bias_label=bias_name, 
				wrong_option="Option A" if dilemma["correct_option"].endswith('B') else "Option B",
				# correct_option=dilemma["correct_option"].replace('_',' ').replace('option','Option'),
				bias_definition=bias_definition
			)
			for i in range(args.n_prolog_construction_reruns)
			for dilemma in ds
		]
		validation_output_list = instruct_model([
				f'## Version A:\n{dilemma["unbiased"]}\n\n## Version B:\n{dilemma["biased"]}'+' '*i
				for i in range(args.n_prolog_construction_reruns)
				for dilemma in ds
			], 
			system_instructions=system_instructions,
			**llm_options
		)

		for dilemma in ds:
			dilemma['decisions_towards_validity'] = 0
		
		for i in range(args.n_prolog_construction_reruns):
			this_validation_output_list = validation_output_list[i*len(ds):(i+1)*len(ds)]
			for (decision, explanation), dilemma, validation_output in zip(map(get_bias_validation_and_explanation_from_output, this_validation_output_list), ds, this_validation_output_list):
				if decision:
					dilemma['bias_is_towards_incorrect_only_in_biased_version'] = explanation
					dilemma['decisions_towards_validity'] += 1
					# yield dilemma
				# else:
				# 	print('Wrong option:',"Option A" if dilemma["correct_option"].endswith('B') else "Option B")
				# 	print('decisions_towards_validity', dilemma['decisions_towards_validity'])
				# 	print('Correct option:', dilemma["correct_option"])
				# 	print(f'## Version A:\n{dilemma["unbiased"]}\n\n## Version B:\n{dilemma["biased"]}')
				# 	print('@'*10)
				# 	print(validation_output)
				# 	print('-'*10)

		return filter(lambda x: x["decisions_towards_validity"] >= math.ceil(args.n_prolog_construction_reruns*(2/3)), ds)

	def _filter_by_bias_existence_in_biased_version(ds):
		print(f'<filter_by_bias_existence_in_biased_version> Running on {len(ds)} dilemmas')
		validation_output_list = instruct_model([
				dilemma["biased"]
				for dilemma in ds
			], 
			system_instructions=[BIAS_EXISTENCE_SYSTEM_INSTRUCTION.format(bias_label=bias_name, bias_definition=bias_definition)]*len(ds),
			**llm_options
		)
		for (decision, explanation), dilemma in zip(map(get_bias_validation_and_explanation_from_output, validation_output_list), ds):
			if decision:
				# dilemma['biased_has_valid_bias'] = explanation
				yield dilemma
			# elif bias_name=='hyperbolic discounting':
			# 	print(decision, explanation)
			# 	print('-'*10)
			# 	print(dilemma["biased"])
			# 	print('@'*10)

	def _filter_by_output_matching(ds, check_with_prolog=True):
		print(f'<filter_by_output_matching> Running on {len(ds)} dilemmas')
		max_attempts = args.n_independent_runs_per_task
		unbiased_llm_output_flatten_list = instruct_model([
				dilemma['unbiased']+' '*attempt
				for dilemma in ds
				for attempt in range(max_attempts)
			], 
			system_instructions=[DECISION_GENERATION_SYSTEM_INSTRUCTION]*len(ds)*max_attempts, 
			**llm_options
		)
		assert len(unbiased_llm_output_flatten_list) == len(ds) * max_attempts, f"{len(unbiased_llm_output_flatten_list)} != {len(ds) * max_attempts}"

		unbiased_llm_output_by_dilemma = [
			unbiased_llm_output_flatten_list[i * max_attempts : (i + 1) * max_attempts]
			for i in range(len(ds))
		]

		for dilemma, unbiased_llm_output_list in zip(ds, unbiased_llm_output_by_dilemma):
			
			predicted_option_dict = defaultdict(int)
			for unbiased_llm_output in unbiased_llm_output_list:
				try:
					unbiased_decision, _ = get_decision_and_explanation_from_output(unbiased_llm_output, bias_name)
				except Exception as e:
					# print(f'Err.8: {e}')
					# print('@'*10)
					continue
				predicted_option_dict[unbiased_decision.split(' ')[-1].lower()] += 1

			if check_with_prolog:
				n_predicted_option_is_correct = predicted_option_dict[dilemma['correct_option'].split('_')[-1].lower()]
			else:
				n_predicted_option_is_correct = max(predicted_option_dict.values())

			
			if n_predicted_option_is_correct/max_attempts < args.min_intra_model_agreement_rate_on_dilemma:
				# print('Err.9: Invalid prompt-to-Prolog conversion: the unbiased prompt produces a different output too often. Perhaps the reasoning is too complex or ambiguous for the model to follow, or axioms are ill-defined.')
				# print('@'*10)
				continue

			dilemma['agreement_rate'] = n_predicted_option_is_correct/max_attempts

			yield dilemma

	discarded_dilemmas_count_by_filter = defaultdict(int)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_bias_existence_in_biased_version(dilemma_list))
	discarded_dilemmas_count_by_filter['bias_towards_incorrect'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_output_matching(dilemma_list))
	discarded_dilemmas_count_by_filter['output_matching'] += old_dilemma_list_len - len(dilemma_list)
	
	# old_dilemma_list_len = len(dilemma_list)
	# dilemma_list = list(_filter_by_bias_towards_incorrect(dilemma_list))
	# discarded_dilemmas_count_by_filter['bias_towards_incorrect'] += old_dilemma_list_len - len(dilemma_list)
	
	print(f"<{bias_name}> Valid dilemmas: {len(dilemma_list)}")
	return dilemma_list, discarded_dilemmas_count_by_filter

for data_model in data_model_list:
	with open(f"./generated_data/1_augmented_dilemmas_dataset_{data_model}.json", "r", encoding="utf-8") as f:
		this_dilemmas_dataset = json.load(f)
	
	bias_name_list = list(this_dilemmas_dataset.keys())
	n_generated_pairs = {bias: 0 for bias in bias_name_list}
	discarded_dilemmas_count_by_bias = {bias: 0 for bias in bias_name_list}
	discarded_dilemmas_count_by_filter_by_bias = {bias: defaultdict(int) for bias in bias_name_list}

	for bias_name, extracted_dilemma_list in this_dilemmas_dataset.items():
		bias_definition = read_file_content(os.path.join(os.path.dirname(os.path.dirname(this_dilemmas_dataset[bias_name][0]["unbiased_path"])), 'definition.txt'))
		filtered_dilemma_list, discarded_dilemmas_count_by_filter = filter_dilemma_list(bias_name, bias_definition, extracted_dilemma_list)
		n_generated_pairs[bias_name] = len(extracted_dilemma_list)
		discarded_dilemmas_count_by_bias[bias_name] += len(extracted_dilemma_list) - len(filtered_dilemma_list)
		for k,v in discarded_dilemmas_count_by_filter.items():
			discarded_dilemmas_count_by_filter_by_bias[bias_name][k] += v
	
	with open(f"./generated_data/8_external_judge_validation_{model}_{data_model}.json", "w", encoding="utf-8") as f:
		json.dump(
			{
				'discarded_dilemmas_count': sum(discarded_dilemmas_count_by_bias.values()), 
				'discarded_dilemmas_count_ratio': f"{100*(sum(discarded_dilemmas_count_by_bias.values())/sum(n_generated_pairs.values())):.2f}%", 

				'discarded_dilemmas_count_by_bias': discarded_dilemmas_count_by_bias, 
				'discarded_dilemmas_count_by_filter_by_bias': discarded_dilemmas_count_by_filter_by_bias, 
			}, 
			f, 
			indent=4, 
			ensure_ascii=False
		)
