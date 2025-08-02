#!/bin/bash

source .env/bin/activate


BATCH_SIZE=5
N_RUNS=20 # Consider assigning more runs to the less capable models, as they may fail more often
N_PROLOG_CONSTRUCTION_RERUNS=3
N_INDEPENDENT_RUNS_PER_TASK=5

mkdir -p ./logs

#################################
#### Data generation)
#################################

# # gpt-4.1-nano is not good enough for data generation, it's uncapable of augmenting data.
# python3 1_generate_data.py --model gpt-4.1-nano --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.gpt-4.1-nano.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log
python3 1_generate_data.py --model gpt-4o-mini --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.gpt-4o-mini.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log 
python3 1_generate_data.py --model gpt-4.1-mini --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.gpt-4.1-mini.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log 
python3 1_generate_data.py --model deepseek-r1-distill-llama-70b --batch_size "$BATCH_SIZE" --n_runs "$N_RUNS" --n_prolog_construction_reruns "$N_PROLOG_CONSTRUCTION_RERUNS" --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK" &> ./logs/1_generate_data.deepseek-r1-distill-llama-70b.$BATCH_SIZE.$N_RUNS.$N_PROLOG_CONSTRUCTION_RERUNS.$N_INDEPENDENT_RUNS_PER_TASK.log

python3 sample_data_for_manual_validation.py --input generated_data/2_llm_outputs_model\=gpt-4o-mini.csv 
python3 sample_data_for_manual_validation.py --input generated_data/2_llm_outputs_model\=gpt-4.1-mini.csv 
python3 sample_data_for_manual_validation.py --input generated_data/2_llm_outputs_model\=deepseek-r1-distill-llama-70b.csv 

#################################
#### X vs X
#################################

# ## gpt-4.1-nano # This model is not good enough for data generation, it's uncapable of augmenting data.
# python3 2_compute_bias_sensitivity.py --model gpt-4.1-nano --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-nano.log
# # python3 2_compute_bias_sensitivity.py --model gpt-4.1-nano --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-nano.inject_axioms.log
# # python3 2_compute_bias_sensitivity.py --model gpt-4.1-nano --inject_axioms_in_prolog --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-nano.inject_axioms_in_prolog.log

## gpt-4o-mini
python3 2_compute_bias_sensitivity.py --model gpt-4o-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4o-mini.log
# python3 2_compute_bias_sensitivity.py --model gpt-4o-mini --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4o-mini.inject_axioms.log
# python3 2_compute_bias_sensitivity.py --model gpt-4o-mini --inject_axioms_in_prolog --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4o-mini.inject_axioms_in_prolog.log

## gpt-4.1-mini
python3 2_compute_bias_sensitivity.py --model gpt-4.1-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-mini.log
# python3 2_compute_bias_sensitivity.py --model gpt-4.1-mini --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-mini.inject_axioms.log
# python3 2_compute_bias_sensitivity.py --model gpt-4.1-mini --inject_axioms_in_prolog --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-mini.inject_axioms_in_prolog.log

## deepseek-r1-distill-llama-70b
python3 2_compute_bias_sensitivity.py --model deepseek-r1-distill-llama-70b --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.deepseek-r1-distill-llama-70b.log
# python3 2_compute_bias_sensitivity.py --model deepseek-r1-distill-llama-70b --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.deepseek-r1-distill-llama-70b.inject_axioms.log
# python3 2_compute_bias_sensitivity.py --model deepseek-r1-distill-llama-70b --inject_axioms_in_prolog --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.deepseek-r1-distill-llama-70b.inject_axioms_in_prolog.log

#################################
#### All vs X
#################################

python3 2_compute_bias_sensitivity.py --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b --model gpt-4o-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.gpt-4o-mini.log
# python3 2_compute_bias_sensitivity.py --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b --model gpt-4o-mini --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.gpt-4o-mini.inject_axioms.log

python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.gpt-4.1-mini.log
# python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-mini --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.gpt-4.1-mini.inject_axioms.log

python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model deepseek-r1-distill-llama-70b --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.deepseek-r1-distill-llama-70b.log
# python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model deepseek-r1-distill-llama-70b --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.deepseek-r1-distill-llama-70b.inject_axioms.log

python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-nano --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.gpt-4.1-nano.log
# python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-nano --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.gpt-4.1-nano.inject_axioms.log

python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model llama-3.3-70b-versatile --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.llama-3.3-70b-versatile.log
# python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model llama-3.3-70b-versatile --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.llama-3.3-70b-versatile.inject_axioms.log

python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model llama-3.1-8b-instant --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.llama-3.1-8b-instant.log
# python3 2_compute_bias_sensitivity.py --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model llama-3.1-8b-instant --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.all.llama-3.1-8b-instant.inject_axioms.log

############

# python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b --model gpt-4o-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.gpt-4o-mini.log
# python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b --model gpt-4o-mini --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.gpt-4o-mini.inject_axioms.log

# python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.gpt-4.1-mini.log
# python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-mini --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.gpt-4.1-mini.inject_axioms.log

# # python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model deepseek-r1-distill-llama-70b --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.deepseek-r1-distill-llama-70b.log
# # python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model deepseek-r1-distill-llama-70b --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.deepseek-r1-distill-llama-70b.inject_axioms.log

# python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-nano --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.gpt-4.1-nano.log
# python3 2_compute_bias_sensitivity.py --temperature 1 --top_p 1 --data_model_list gpt-4o-mini gpt-4.1-mini deepseek-r1-distill-llama-70b --model gpt-4.1-nano --inject_axioms --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/2_compute_bias_sensitivity.$N_INDEPENDENT_RUNS_PER_TASK.1.1.all.gpt-4.1-nano.inject_axioms.log


#################################
#### Visualize bias sensitivity
#################################

python3 3_visualize_bias_sensitivity.py
# python3 3_visualize_bias_sensitivity.py --inject_axioms
# python3 3_visualize_bias_sensitivity.py --temperature 1 --top_p 1

#################################
#### Get stats about seed corpus
#################################

python3 4_get_stats_about_seed_corpus.py &> ./logs/4_get_stats_about_seed_corpus.log

#################################
#### Bias-awareness analysis
#################################

python3 5_bias_awareness_analysis.py --model gpt-4.1-nano --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/5_bias_awareness_analysis.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-nano.log &
python3 5_bias_awareness_analysis.py --model gpt-4o-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/5_bias_awareness_analysis.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4o-mini.log &
python3 5_bias_awareness_analysis.py --model gpt-4.1-mini --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/5_bias_awareness_analysis.$N_INDEPENDENT_RUNS_PER_TASK.gpt-4.1-mini.log &
python3 5_bias_awareness_analysis.py --model llama-3.1-8b-instant --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/5_bias_awareness_analysis.$N_INDEPENDENT_RUNS_PER_TASK.llama-3.1-8b-instant.log &
python3 5_bias_awareness_analysis.py --model llama-3.3-70b-versatile --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/5_bias_awareness_analysis.$N_INDEPENDENT_RUNS_PER_TASK.llama-3.3-70b-versatile.log &
python3 5_bias_awareness_analysis.py --model deepseek-r1-distill-llama-70b --n_independent_runs_per_task "$N_INDEPENDENT_RUNS_PER_TASK"  &> ./logs/5_bias_awareness_analysis.$N_INDEPENDENT_RUNS_PER_TASK.deepseek-r1-distill-llama-70b.log 

python3 6_visualize_bias_awareness_analysis.py