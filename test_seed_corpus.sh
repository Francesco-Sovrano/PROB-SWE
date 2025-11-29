#!/bin/bash


SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

source .env/bin/activate

BATCH_SIZE=0
N_RUNS=0 # Consider assigning more runs to the less capable models, as they may fail more often
N_PROLOG_CONSTRUCTION_RERUNS=0
N_INDEPENDENT_RUNS_PER_TASK=50

mkdir -p ./logs

#################################

### gpt-4.1-nano
python3 1_generate_data.py --model gpt-4.1-nano --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.gpt-4.1-nano.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 2_compute_bias_sensitivity.py --model gpt-4.1-nano --seed_corpus_only --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-nano.log
python3 2_compute_bias_sensitivity.py --model gpt-4.1-nano --seed_corpus_only --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-nano.inject_axioms.log

### gpt-4o-mini
python3 1_generate_data.py --model gpt-4o-mini --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.gpt-4o-mini.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 2_compute_bias_sensitivity.py --model gpt-4o-mini --seed_corpus_only --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4o-mini.log
python3 2_compute_bias_sensitivity.py --model gpt-4o-mini --seed_corpus_only --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4o-mini.inject_axioms.log

### gpt-4.1-mini
python3 1_generate_data.py --model gpt-4.1-mini --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.gpt-4.1-mini.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 2_compute_bias_sensitivity.py --model gpt-4.1-mini --seed_corpus_only --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-mini.log
python3 2_compute_bias_sensitivity.py --model gpt-4.1-mini --seed_corpus_only --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-mini.inject_axioms.log

### llama-3.1-8b-instant
python3 1_generate_data.py --model llama-3.1-8b-instant --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.llama-3.1-8b-instant.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 2_compute_bias_sensitivity.py --model llama-3.1-8b-instant --seed_corpus_only --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.llama-3.1-8b-instant.log
python3 2_compute_bias_sensitivity.py --model llama-3.1-8b-instant --seed_corpus_only --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.llama-3.1-8b-instant.inject_axioms.log

### llama-3.3-70b-versatile
python3 1_generate_data.py --model llama-3.3-70b-versatile --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.llama-3.3-70b-versatile.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 2_compute_bias_sensitivity.py --model llama-3.3-70b-versatile --seed_corpus_only --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.llama-3.3-70b-versatile.log
python3 2_compute_bias_sensitivity.py --model llama-3.3-70b-versatile --seed_corpus_only --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.llama-3.3-70b-versatile.inject_axioms.log

### deepseek-r1-distill-llama-70b
python3 1_generate_data.py --model deepseek-r1-distill-llama-70b --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.deepseek-r1-distill-llama-70b.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 2_compute_bias_sensitivity.py --model deepseek-r1-distill-llama-70b --seed_corpus_only --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.deepseek-r1-distill-llama-70b.log
python3 2_compute_bias_sensitivity.py --model deepseek-r1-distill-llama-70b --seed_corpus_only --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.deepseek-r1-distill-llama-70b.inject_axioms.log

python3 4_get_stats_about_seed_corpus.py &> ./logs/4_get_stats_about_seed_corpus.log
