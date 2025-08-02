import pandas as pd
import argparse
import re
import math
import os
import json
from collections import defaultdict
from lib import *

parser = argparse.ArgumentParser(description="Script to augment dilemma pairs by bias type and batch size")
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'llama3.1', 'llama3.2', 'llama3.3', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o3-mini-2025-01-31', 'gpt-4o-2024-08-06')"
)
parser.add_argument(
	"--n_runs",
	type=int,
	default=10,
	help="Number of augmentation runs to perform (default: 10)"
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
	"--batch_size",
	type=int,
	default=5,
	help="Number of dilemmas to generate per batch (default: 5)"
)
parser.add_argument(
	"--min_intra_model_agreement_rate_on_dilemma",
	type=float,
	default=0.8,
	help="Minimum intra-model agreement rate required for a dilemma to be considered valid (default: 0.8)"
)
args = parser.parse_args()
print('Args:', args)

ROOT_DIRECTORY = './seed_corpus'

NAIVE_AUGMENTATION_PROMPT = """I have a set of dilemma pairs that I need you to augment with at least {batch_size} similarly structured but very diverse dilemmas. Each dilemma must:
- be about realistic software engineering challenges whose solution does not require complex reasoning but rather simple reasoning based on well-known best-practices or common sense logic;
- include a task description in 1st person outlining a realistic software engineering challenge whose solution does not require complex reasoning but rather simple reasoning based on well-known best-practices or common sense logic;
- end with a short question about how to address the scenario, and two options (A and B), one correct (either A or B) and one incorrect, from which the recipient must choose (see examples below).
For each new pair, first write the unbiased version, then produce a minimally edited biased version that uses natural language to induce {bias_label} to favour the incorrect answer, without altering any other detail. The bias-inducing linguistic features should influence biased readers only and they should not affect unbiased reasoners in any possible way.

Examples from the set of pairs of dilemmas you have to augment are given below.

# Pair 1

## Unbiased Dilemma
```
{pair_1_nobias}
```

## Biased Dilemma
```
{pair_1_bias}
```

# Pair 2

## Unbiased Dilemma
```
{pair_2_nobias}
```

## Biased Dilemma
```
{pair_2_bias}
```"""

BIAS_EXISTENCE_SYSTEM_INSTRUCTION = (
	"You will be given a task description as input, ending with a question. Do not answer the question! Instead, analyze the task description and determine whether it contains or may lead to a {bias_label} (see definition below) and no other types of bias.\n\n"
	"Definition of '{bias_label}': {bias_definition}\n\n"
	"Response Format (strictly follow this structure without adding extra text or formatting):\n"
	"```\n"
	"Explanation: <Provide a brief explanation for your judgment>\n"
	"Decision: <Your decision: 'Yes, it has a {bias_label} and no other bias.' or 'No, it does not have a {bias_label} or it may have other types of biases'>\n"
	"```"
)

BIAS_TOWARDS_INCORRECT_SYSTEM_INSTRUCTION = (
	"You are an AI assistant designed to provide step-by-step responses. You will be given two versions (A and B) of a task, each ending with a question. Do not answer the questions! Instead, analyze the two task versions and determine whether BOTH of these two POINTS hold true:\n"
	"1. Version A does NOT explicitly contain a {bias_label} (see definition below) that may directly favour {wrong_option} over the other options.\n"
	"2. Version B explicitly contains a {bias_label} (see definition below) that may directly favour {wrong_option} over the other options.\n"
	"Definition of '{bias_label}': {bias_definition}\n\n"
	"Response Format (strictly follow this structure without adding extra text or formatting):\n"
	"```\n"
	"Explanation: <Provide a very short explanation for your judgment>\n"
	"Decision: <Your decision: 'Yes, BOTH the two POINTS hold true.' or 'No, there is one of the POINTS which is false.' >\n"
	"```"
)

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

def get_bias_name_from_file_path(task_file_path):
	path_parts = os.path.normpath(task_file_path).split(os.sep)
	if len(path_parts) >= 3:
		bias_name = path_parts[-3]  # grandparent folder
		if '-' in bias_name:
			bias_name = bias_name.split('-')[-1]
		bias_name = bias_name.strip().replace('_',' ')
	else:
		bias_name = "unknown"
	return bias_name

def read_file_content(path):
	with open(path, "r", encoding="utf-8") as f:
		return f.read()

def get_seed_bias_data_dict(root_dir):
	# Traverse the directory to collect paths to .txt files and dilemma names
	bias_data_dict = defaultdict(lambda: defaultdict(dict))

	for root, dirs, files in os.walk(root_dir):
		for file in files:
			if not file.endswith(".txt"):
				continue
			if file.endswith("axioms.txt"):
				continue
			if file.endswith("definition.txt"):
				continue
			if not '1-biased' in file and not '0-unbiased' in file:
				continue
			full_path = os.path.join(root, file)
			bias_dir = os.path.basename(os.path.dirname(root))
			bias_name = bias_dir
			if '-' in bias_name:
				bias_name = bias_name.split('-')[-1]
			bias_name = bias_name.strip().replace('_',' ')
			dilemma_name = os.path.basename(root).split("-", 1)[-1]
			bias_data_dict[bias_name][dilemma_name]['bias' if file.startswith('1-') else 'no_bias'] = {
				"file_path": full_path,
				"file_content": read_file_content(full_path),
			}
	return bias_data_dict

def extract_dilemmas(text, run_id, bias_name):
	pattern = re.compile(
		r"^.*?Pair[^\d]*(?P<pair>\d+).*?"
		r"^.*?Unbiased[^\n]*\n+[`]+(?P<unbiased>[^`]+)[`]+.*?"
		r"^.*?Biased[^\n]*\n+[`]+(?P<biased>[^`]+)[`]+",
		re.DOTALL | re.MULTILINE
	)
	# re.MULTILINE affects where ^ and $ anchors match. Without the switch, ^ and $ match only at the start and end, respectively, of the whole text. With the switch, they also match just before or after a newline
	# re.DOTALL affects what the . pattern can match. Without the switch, . matches any character except a newline. With the switch, newlines are matched as well

	dilemma_pairs = []
	for m in re.finditer(pattern, text):
		pair = m.group('pair').strip()
		unbiased = m.group('unbiased').strip()
		biased = m.group('biased').strip()

		if not pair or not unbiased or not biased:
			continue
		
		dilemma_pairs.append({
			"run_id": run_id,
			"AI_generated": True,
			"pair": int(pair),
			"unbiased": unbiased,
			"biased": biased,
		})

	# if not dilemma_pairs:
	# 	print(f'Cannot find valid dilemmas in text:')
	# 	print(text)
	# 	print('#'*10)
	# else:
	# 	print(json.dumps(dilemma_pairs, indent=4))
	# 	print('@'*10)

	# assert dilemma_pairs, f"Cannot find dilemmas for text: {text}"
	return dilemma_pairs

def init_seed_corpus(dilemma_list):
	seed_1, seed_2 = filter(lambda x: not x["AI_generated"], dilemma_list)
	# Initialize seed dilemma 1
	seed_1.update({
		"valid": True,
		"axioms": clean_prolog_code(read_file_content(os.path.join(os.path.dirname(seed_1["biased_path"]), 'axioms.pl'))),
		"axioms_description": read_file_content(os.path.join(os.path.dirname(seed_1["biased_path"]), 'axioms.txt')),
		"unbiased_prolog": clean_prolog_code(read_file_content(seed_1["unbiased_path"].replace('.txt','.pl'))),
		"biased_prolog": clean_prolog_code(read_file_content(seed_1["biased_path"].replace('.txt','.pl'))),
	})
	seed_1_unbiased_prolog_dict = run_prolog_script_from_content(get_prolog_program_content(seed_1['axioms'], seed_1['unbiased_prolog']))
	seed_1.update({
		"correct_option": seed_1_unbiased_prolog_dict['choice'],
		"inference_steps": seed_1_unbiased_prolog_dict['inferences'],
		"choice_steps": seed_1_unbiased_prolog_dict['profiler']['nodes'],
	})
	seed_1.update({
		"reconstructed_unbiased_prompt": None,
		"unbiased_prompt_reconstruction_similarity": None,
		# "reconstructed_biased_prompt": None,
		# "biased_prompt_reconstruction_similarity": None,
	})

	# Initialize seed dilemma 2
	seed_2.update({
		"valid": True,
		"axioms": clean_prolog_code(read_file_content(os.path.join(os.path.dirname(seed_2["biased_path"]), 'axioms.pl'))),
		"axioms_description": read_file_content(os.path.join(os.path.dirname(seed_2["biased_path"]), 'axioms.txt')),
		"unbiased_prolog": clean_prolog_code(read_file_content(seed_2["unbiased_path"].replace('.txt','.pl'))),
		"biased_prolog": clean_prolog_code(read_file_content(seed_2["biased_path"].replace('.txt','.pl'))),
	})
	seed_2_unbiased_prolog_dict = run_prolog_script_from_content(get_prolog_program_content(seed_2['axioms'], seed_2['unbiased_prolog']))
	seed_2.update({
		"correct_option": seed_2_unbiased_prolog_dict['choice'],
		"inference_steps": seed_2_unbiased_prolog_dict['inferences'],
		"choice_steps": seed_2_unbiased_prolog_dict['profiler']['nodes'],
	})
	seed_2.update({
		"reconstructed_unbiased_prompt": None,
		"unbiased_prompt_reconstruction_similarity": None,
		# "reconstructed_biased_prompt": None,
		# "biased_prompt_reconstruction_similarity": None,
	})
	

def filter_dilemma_list(bias_name, bias_definition, dilemma_list, approved_dilemma_list):
	def _filter_by_intra_dilemma_similarity(ds):
		print(f'<filter_by_intra_dilemma_similarity> Running on {len(ds)} dilemmas')
		for dilemma in ds:
			unbiased = dilemma['unbiased']
			biased = dilemma['biased']

			if unbiased[-1] != '?' or biased[-1] != '?' or 'biased' in biased.lower() or 'biased' in unbiased.lower(): # correct answer might have been injected
				continue

			if bias_name != 'framing effect':
				pair_similarity = get_texts_similarity(unbiased, biased)
				if not (0.9 <= pair_similarity < 0.99): # similarity range in seed corpus
					continue

				pair_levenshtein_distance = get_texts_similarity_with_Levenshtein(unbiased, biased)
				if not (0.7 <= pair_levenshtein_distance < 0.99): # levenshtein distance range in seed corpus
					continue
			else:
				pair_similarity = get_texts_similarity(unbiased, biased)
				if not (0.85 <= pair_similarity < 0.99): # similarity range in seed corpus
					continue

				pair_levenshtein_distance = get_texts_similarity_with_Levenshtein(unbiased, biased)
				if not (0.55 <= pair_levenshtein_distance < 0.99): # levenshtein distance range in seed corpus
					continue

			dilemma.update({"pair_similarity": pair_similarity, "pair_levenshtein_distance": pair_levenshtein_distance,})
			yield dilemma

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

	def _filter_by_inter_dilemma_similarity(ds, check_against_valid_dilemmas_only=False):
		print(f'<filter_by_inter_dilemma_similarity> Running on {len(ds)} dilemmas')
		texts = list(map(lambda x: x['unbiased'], approved_dilemma_list + ds)) # approved_dilemma_list + ds gives priority to approved_dilemma_list discarding from ds
		duplicated_texts_index_set, similarity_set_list = get_index_set_of_duplicated_texts(texts)
		# print('duplicated_texts_index_set:', duplicated_texts_index_set)
		for i, dilemma in enumerate(ds):
			adjusted_i = len(approved_dilemma_list)+i
			if adjusted_i not in duplicated_texts_index_set:
				yield dilemma
			elif check_against_valid_dilemmas_only:
				similarity_sets_with_dilemma = filter(lambda x: adjusted_i in x, similarity_set_list)
				if not any(map(lambda x: sorted(x)[0] < len(approved_dilemma_list), similarity_sets_with_dilemma)):
					yield dilemma

	def _filter_by_logic(ds):
		print(f'<filter_by_logic> Running on {len(ds)} dilemmas')
		seed_1, seed_2 = filter(lambda x: not x["AI_generated"], approved_dilemma_list)
		
		this_text2prolog_system_instruction = TEXT2PROLOG_SYSTEM_INSTRUCTION.format(
			bias_label=bias_name, 
			###
			unbiased_description_example_1=read_file_content(seed_1["unbiased_path"]),
			biased_description_example_1=read_file_content(seed_1["biased_path"]),
			prolog_axioms_example_1=seed_1["axioms"],
			unbiased_prolog_example_1=seed_1["unbiased_prolog"],
			biased_prolog_example_1=seed_1["biased_prolog"],
			axioms_description_example_1=seed_1["axioms_description"],
			###
			# unbiased_description_example_2=read_file_content(seed_2["unbiased_path"]),
			# biased_description_example_2=read_file_content(seed_2["biased_path"]),
			# prolog_axioms_example_2=seed_2["axioms"],
			# unbiased_prolog_example_2=seed_2["unbiased_prolog"],
			# biased_prolog_example_2=seed_2["biased_prolog"],
			# axioms_description_example_2=seed_2["axioms_description"],
		)

		dilemmas_to_process = list(filter(lambda x: x["AI_generated"], ds))
		for i in range(args.n_prolog_construction_reruns):
			if not dilemmas_to_process:
				break
			print(f'<filter_by_logic> Run {i+1}: {len(dilemmas_to_process)}/{len(ds)} dilemmas to process')
			next_dilemmas_to_process = []

			prompt_list = [
				TEXT2PROLOG_PROMPT_TEMPLATE.format(
					biased_dilemma_description=dilemma['biased'], 
					unbiased_dilemma_description=dilemma['unbiased']
				)+' '*i
				for dilemma in dilemmas_to_process
			]
			llm_output_list = instruct_model(prompt_list, system_instructions=[this_text2prolog_system_instruction]*len(prompt_list), **llm_options)
			prolog_program_dict_list = list(map(lambda x: x[0] if x else None, map(extract_prolog_program_dict_list, llm_output_list)))
			assert len(prolog_program_dict_list) == len(dilemmas_to_process), f"{len(prolog_program_dict_list)} != {len(dilemmas_to_process)}"
			
			for dilemma, prolog_program_dict, prompt in zip(dilemmas_to_process, prolog_program_dict_list, prompt_list):
				if bias_name!='framing effect':
					if not prolog_program_dict or not prolog_program_dict['axioms'] or not prolog_program_dict['axioms_description'] or not prolog_program_dict['unbiased_prolog'] or not prolog_program_dict['biased_prolog']:
						# print("Err.0: Ill-formed LLM output")
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

					if prolog_program_dict['biased_prolog'] == prolog_program_dict['unbiased_prolog']:
						# print("Err.1: Invalid Prolog programs: the biased and unbiased Prolog programs are identical")
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

					dilemma.update(prolog_program_dict)

					dilemma_biased_prolog_dict = run_prolog_script_from_content(get_prolog_program_content(dilemma['axioms'], dilemma['biased_prolog']))
					dilemma_unbiased_prolog_dict = run_prolog_script_from_content(get_prolog_program_content(dilemma['axioms'], dilemma['unbiased_prolog']))
					if not dilemma_biased_prolog_dict or not dilemma_unbiased_prolog_dict:
						# print('Err.2: Invalid Prolog programs: cannot run biased or unbiased Prolog program')
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

					if dilemma_biased_prolog_dict['choice'] != dilemma_unbiased_prolog_dict['choice']:
						# print('Err.3: Invalid Prolog programs: the biased and unbiased Prolog programs produce different outputs')
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

					if dilemma_biased_prolog_dict['inferences'] != dilemma_unbiased_prolog_dict['inferences']:
						# print('Err.4: Invalid Prolog programs: the biased and unbiased Prolog programs follow different inference steps; bias-inducing features must not affect the overall logic')
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

					if dilemma_biased_prolog_dict['profiler']['nodes'] != dilemma_unbiased_prolog_dict['profiler']['nodes']:
						# print('Err.5: Invalid Prolog programs: the biased and unbiased Prolog programs follow different rules/nodes; bias-inducing features must not affect the overall logic')
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue
				else:
					if not prolog_program_dict or not prolog_program_dict['axioms'] or not prolog_program_dict['axioms_description'] or not prolog_program_dict['unbiased_prolog']:
						# print("Err.0: Ill-formed LLM output")
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

					dilemma.update(prolog_program_dict)

					dilemma['biased_prolog'] = dilemma['unbiased_prolog']

					dilemma_unbiased_prolog_dict = run_prolog_script_from_content(get_prolog_program_content(dilemma['axioms'], dilemma['unbiased_prolog']))
					if not dilemma_unbiased_prolog_dict:
						# print('Err.2: Invalid Prolog programs: cannot run biased or unbiased Prolog program')
						# print('@'*10)
						next_dilemmas_to_process.append(dilemma)
						continue

				if dilemma_unbiased_prolog_dict['inferences'] <= 0:
					# print('Err.6: Invalid Prolog programs: not possible to compute inference steps')
					# print('@'*10)
					next_dilemmas_to_process.append(dilemma)
					continue

				dilemma.update({
					"correct_option": dilemma_unbiased_prolog_dict['choice'],
					"inference_steps": dilemma_unbiased_prolog_dict['inferences'],
					"choice_steps": dilemma_unbiased_prolog_dict['profiler']['nodes'],
					"valid": True,
				})
			dilemmas_to_process = next_dilemmas_to_process

		return filter(lambda x: x.get("valid",False), ds)

	def _filter_by_prolog2text_alignment(ds, reconstruction_similarity_threshold=0.65):
		print(f'<filter_by_prolog2text_alignment> Running on {len(ds)} dilemmas')
		seed_1, seed_2 = filter(lambda x: not x["AI_generated"], approved_dilemma_list)
		
		dilemmas_to_process = list(filter(lambda x: x["AI_generated"], ds))
		# for i in range(args.n_prolog_construction_reruns):
		for i in range(1):
			if not dilemmas_to_process:
				break
			print(f'<filter_by_prolog2text_alignment> Run {i+1}: {len(dilemmas_to_process)}/{len(ds)} dilemmas to process')
			next_dilemmas_to_process = []

			reconstructed_unbiased_prompt_list = instruct_model([
					PROLOG2TEXT_SYSTEM_INSTRUCTION.format(
						dilemma_example=seed_2['unbiased_prolog'],
						description_example=seed_2['unbiased'],
						dilemma_to_encode=dilemma['unbiased_prolog'],
					)+' '*i
					for dilemma in dilemmas_to_process
				], 
				**llm_options
			)
			assert len(reconstructed_unbiased_prompt_list) == len(dilemmas_to_process), f"{len(reconstructed_unbiased_prompt_list)} != {len(dilemmas_to_process)}"
			for reconstructed_unbiased_prompt, dilemma in zip(reconstructed_unbiased_prompt_list, dilemmas_to_process):
				unbiased_prompt_reconstruction_similarity = get_texts_similarity(dilemma['unbiased'],reconstructed_unbiased_prompt)
				if ('best practice' in reconstructed_unbiased_prompt and 'best practice' not in dilemma['unbiased']) or unbiased_prompt_reconstruction_similarity < reconstruction_similarity_threshold:
					# print('Err.7: Invalid Prolog programs: the reconstructed prompt does not match with the original prompt; potential Prolog conversion error.')
					# print('Original unbiased prompt:', dilemma['unbiased'])
					# print('-'*10)
					# print('Reconstructed unbiased prompt:', reconstructed_unbiased_prompt)
					# print('-'*10)
					# print(f'Reconstruction similarity: {unbiased_prompt_reconstruction_similarity:.2f}/1')
					# print('@'*10)
					dilemma['valid'] = False
					next_dilemmas_to_process.append(dilemma)
					continue

				# if bias_name!='framing effect':
				# 	reconstructed_biased_prompt = instruct_model([
				# 		PROLOG2TEXT_SYSTEM_INSTRUCTION.format(
				# 			dilemma_example=seed_2['biased_prolog'],
				# 			description_example=seed_2['biased'],
				# 			dilemma_to_encode=dilemma['biased_prolog'],
				# 		)], 
				# 		**llm_options
				# 	)[0]
				# 	biased_prompt_reconstruction_similarity = get_texts_similarity(dilemma['biased'],reconstructed_biased_prompt)
				# 	if ('best practice' in reconstructed_biased_prompt and 'best practice' not in dilemma['biased']) or biased_prompt_reconstruction_similarity < reconstruction_similarity_threshold:
				# 		print('Err.7: Invalid Prolog programs: the reconstructed prompt does not match with the original prompt; potential Prolog conversion error.')
				# 		print('Original biased prompt:', dilemma['biased'])
				# 		print('-'*10)
				# 		print('Reconstructed biased prompt:', reconstructed_biased_prompt)
				# 		print('-'*10)
				# 		print(f'Reconstruction similarity: {biased_prompt_reconstruction_similarity:.2f}/1')
				# 		print('@'*10)
				# 		next_dilemmas_to_process.append(dilemma)
				# 		continue

				dilemma.update({
					"reconstructed_unbiased_prompt": reconstructed_unbiased_prompt,
					"unbiased_prompt_reconstruction_similarity": float(unbiased_prompt_reconstruction_similarity),
					# "reconstructed_biased_prompt": reconstructed_biased_prompt,
					# "biased_prompt_reconstruction_similarity": float(biased_prompt_reconstruction_similarity),
					"valid": True,
				})
			dilemmas_to_process = next_dilemmas_to_process

		return filter(lambda x: x.get("valid",False), ds)

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
	dilemma_list = list(_filter_by_intra_dilemma_similarity(dilemma_list))
	discarded_dilemmas_count_by_filter['intra_dilemma_similarity'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_inter_dilemma_similarity(dilemma_list, check_against_valid_dilemmas_only=True)) # This helps saving some money; fully filter by inter-dilemma similarity only at the end since we might discard useful dilemmas if we do it now
	discarded_dilemmas_count_by_filter['inter_dilemma_similarity'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_bias_existence_in_biased_version(dilemma_list))
	discarded_dilemmas_count_by_filter['bias_towards_incorrect'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_logic(dilemma_list))
	discarded_dilemmas_count_by_filter['logic'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_prolog2text_alignment(dilemma_list))
	discarded_dilemmas_count_by_filter['prolog2text_alignment'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_output_matching(dilemma_list))
	discarded_dilemmas_count_by_filter['output_matching'] += old_dilemma_list_len - len(dilemma_list)
	
	# consider adding an explanation-to-axiom matching filter or thoroughly explain why we don't need it
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_inter_dilemma_similarity(dilemma_list))
	discarded_dilemmas_count_by_filter['inter_dilemma_similarity'] += old_dilemma_list_len - len(dilemma_list)
	
	old_dilemma_list_len = len(dilemma_list)
	dilemma_list = list(_filter_by_bias_towards_incorrect(dilemma_list))
	discarded_dilemmas_count_by_filter['bias_towards_incorrect'] += old_dilemma_list_len - len(dilemma_list)
	
	print(f"<{bias_name}> Valid dilemmas: {len(dilemma_list)}")
	return dilemma_list, discarded_dilemmas_count_by_filter
	
bias_data_dict = get_seed_bias_data_dict(ROOT_DIRECTORY)
augmented_dilemmas_dataset = defaultdict(list)
bias_prompt_dict = {}
for bias_name, dilemma_version_dict in bias_data_dict.items():
	dilemma_version_list = list(dilemma_version_dict.values())

	pair_1_bias = dilemma_version_list[0]['bias']['file_content']
	pair_1_nobias = dilemma_version_list[0]['no_bias']['file_content']
	pair_2_bias = dilemma_version_list[1]['bias']['file_content']
	pair_2_nobias = dilemma_version_list[1]['no_bias']['file_content']
	
	pair_1_bias_path = dilemma_version_list[0]['bias']['file_path']
	pair_1_nobias_path = dilemma_version_list[0]['no_bias']['file_path']
	pair_2_bias_path = dilemma_version_list[1]['bias']['file_path']
	pair_2_nobias_path = dilemma_version_list[1]['no_bias']['file_path']

	augmented_dilemmas_dataset[bias_name] += [
		{
			"run_id": None,
			"AI_generated": False,
			"pair": 1,
			"unbiased": pair_1_nobias,
			"biased": pair_1_bias,
			"unbiased_path": pair_1_nobias_path,
			"biased_path": pair_1_bias_path,
			"pair_similarity": get_texts_similarity(pair_1_nobias, pair_1_bias),
			"pair_levenshtein_distance": get_texts_similarity_with_Levenshtein(pair_1_nobias, pair_1_bias),
		},
		{
			"run_id": None,
			"AI_generated": False,
			"pair": 2,
			"unbiased": pair_2_nobias,
			"biased": pair_2_bias,
			"unbiased_path": pair_2_nobias_path,
			"biased_path": pair_2_bias_path,
			"pair_similarity": get_texts_similarity(pair_2_nobias, pair_2_bias),
			"pair_levenshtein_distance": get_texts_similarity_with_Levenshtein(pair_2_nobias, pair_2_bias),
		}
	]
		
	prompt = NAIVE_AUGMENTATION_PROMPT.format(bias_label=bias_name, batch_size=args.batch_size, pair_1_bias=pair_1_bias, pair_1_nobias=pair_1_nobias, pair_2_bias=pair_2_bias, pair_2_nobias=pair_2_nobias)
	# print(prompt)
	# print('#'*10)
	bias_prompt_dict[bias_name] = prompt

for bias_name, dilemma_list in augmented_dilemmas_dataset.items():
	init_seed_corpus(dilemma_list)

bias_name_list = list(bias_data_dict.keys())
n_generated_pairs = {bias: 0 for bias in bias_name_list}
target_pairs_amount = args.n_runs*args.batch_size
processed_runs = {bias: 0 for bias in bias_name_list}
reruns = 0
discarded_dilemmas_count_by_bias = {bias: 0 for bias in bias_name_list}
discarded_dilemmas_count_by_filter_by_bias = {bias: defaultdict(int) for bias in bias_name_list}
while any(map(lambda x: x < target_pairs_amount, n_generated_pairs.values())):
	reruns += 1
	print(f'Rerun ID: {reruns}')

	multi_run_bias_prompt_list = []
	multi_run_bias_name_list = []
	multi_run_id_list = []
	for bias_name in bias_name_list:
		missing_dilemmas = target_pairs_amount-n_generated_pairs[bias_name]
		if n_generated_pairs[bias_name] >= target_pairs_amount:
			continue
		else:
			print(f'{bias_name} is missing {missing_dilemmas} dilemmas')
		
		runs_needed_now = math.ceil(missing_dilemmas/args.batch_size)
		for i in range(processed_runs[bias_name], processed_runs[bias_name]+runs_needed_now):
			multi_run_bias_prompt_list.append(bias_prompt_dict[bias_name]+' '*i)
			multi_run_bias_name_list.append(bias_name)
			multi_run_id_list.append(i)
		processed_runs[bias_name] += runs_needed_now

	assert len(multi_run_bias_prompt_list) == len(multi_run_bias_name_list) == len(multi_run_id_list)
	multi_run_model_output_list = instruct_model(multi_run_bias_prompt_list, **llm_options_for_diverse_output)
	tmp_augmented_dilemmas_dataset = defaultdict(list)
	for bias_name, run_id, model_output in zip(multi_run_bias_name_list, multi_run_id_list, multi_run_model_output_list):
		tmp_augmented_dilemmas_dataset[bias_name] += extract_dilemmas(model_output, run_id, bias_name)

	for bias_name, extracted_dilemma_list in tmp_augmented_dilemmas_dataset.items():
		print(f"<{bias_name}> Processing dilemmas")
		bias_definition = read_file_content(os.path.join(os.path.dirname(os.path.dirname(augmented_dilemmas_dataset[bias_name][0]["unbiased_path"])), 'definition.txt'))
		print(f"<{bias_name}> Definition: {bias_definition}")
		filtered_dilemma_list, discarded_dilemmas_count_by_filter = filter_dilemma_list(bias_name, bias_definition, extracted_dilemma_list, augmented_dilemmas_dataset[bias_name])
		augmented_dilemmas_dataset[bias_name] += filtered_dilemma_list
		n_generated_pairs[bias_name] = len(augmented_dilemmas_dataset[bias_name])

		discarded_dilemmas_count_by_bias[bias_name] += len(extracted_dilemma_list) - len(filtered_dilemma_list)
		for k,v in discarded_dilemmas_count_by_filter.items():
			discarded_dilemmas_count_by_filter_by_bias[bias_name][k] += v

# Save to JSON file
os.makedirs("./generated_data", exist_ok=True)
with open(f"./generated_data/1_augmented_dilemmas_dataset_{model}.json", "w", encoding="utf-8") as f:
	json.dump(augmented_dilemmas_dataset, f, indent=4, ensure_ascii=False)

with open(f"./generated_data/1_stats_dataset_{model}.json", "w", encoding="utf-8") as f:
	json.dump(
		{
			'discarded_dilemmas_count': sum(discarded_dilemmas_count_by_bias.values()), 
			'discarded_dilemmas_count_by_bias': discarded_dilemmas_count_by_bias, 
			'discarded_dilemmas_count_by_filter_by_bias': discarded_dilemmas_count_by_filter_by_bias, 
			'n_generated_pairs': n_generated_pairs
		}, 
		f, 
		indent=4, 
		ensure_ascii=False
	)
