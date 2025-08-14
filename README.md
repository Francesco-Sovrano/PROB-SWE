# PROBE‑SWE — Replication Package

> PRolog‑based dilemmas for Observing cognitive Bias Effects in Software Engineering (PROBE‑SWE)

This repository contains the replication package for the paper _“Is General-Purpose AI Reasoning Sensitive to Data-Induced Cognitive Biases? Dynamic Benchmarking on Typical Software Engineering Dilemmas”_.  
It introduces **PROBE‑SWE**, a dynamic benchmarking framework for generating, validating, and analysing **data‑induced cognitive biases** in **general‑purpose AI (GPAI)** systems applied to **software‑engineering tasks** (e.g., GPT‑4o).

**What you get here**
- A curated **seed corpus** of human‑written dilemma pairs covering multiple bias families.
- Scripts to **augment** the corpus with AI‑generated variants and to **measure bias sensitivity**.
- Prolog tooling to **profile reasoning complexity** for each dilemma.
- Reproducible **analyses and figures** saved under `generated_data/`.
- Utilities for **bias‑awareness** analyses and **manual validation** workflows.

---

## Abstract

Human cognitive biases in software engineering can lead to costly errors. While general-purpose AI (GPAI) systems may help mitigate these biases due to their non-human nature, their training on human-generated data raises a critical question: **Do GPAI systems themselves exhibit cognitive biases?**

To investigate this, we present the first dynamic benchmarking framework to evaluate data-induced cognitive biases in GPAI within software engineering workflows. Starting with a seed set of 16 hand-crafted realistic tasks, each featuring one of 8 cognitive biases (e.g., anchoring, framing) and corresponding unbiased variants, we test whether bias-inducing linguistic cues unrelated to task logic can lead GPAI systems from correct to incorrect conclusions.

To scale the benchmark and ensure realism, we develop an on-demand augmentation pipeline relying on GPAI systems to generate task variants that preserve bias-inducing cues while varying surface details. This pipeline ensures correctness (88–99% on average, according to human evaluation), promotes diversity, and controls reasoning complexity by leveraging Prolog-based reasoning and LLM-as-a-judge validation. It also verifies that the embedded biases are both harmful and undetectable by logic-based, unbiased reasoners.

We evaluate leading GPAI systems (GPT, LLaMA, DeepSeek) and find a consistent tendency to rely on shallow linguistic heuristics over deep reasoning. All systems exhibit cognitive biases (ranging from 5.9% to 35% across types), with bias sensitivity increasing sharply with task complexity (up to 49%), highlighting critical risks in real-world software engineering deployments.

---

## Table of Contents
1. [Repository Layout](#repository-layout)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Reproducing the Paper’s Results](#reproducing-the-papers-results)
6. [Scripts & Key Options](#scripts--key-options)
7. [Seed Corpus](#seed-corpus)
7. [Manual Validation Workflow](#manual-validation-workflow)
8. [Outputs & File Naming](#outputs--file-naming)
9. [Troubleshooting](#troubleshooting)
10. [Citing](#citing)
11. [License](#license)
12. [Contact](#contact)

---

## Repository Layout

The following items are present in the archive:

- 1_generate_data.py
- 2_compute_bias_sensitivity.py
- 3_visualize_bias_sensitivity.py
- 4_get_stats_about_seed_corpus.py
- 5_bias_awareness_analysis.py
- 6_visualize_bias_awareness_analysis.py
- 7_sensitivity_to_correctness_correlation_analysis.py
- lib.py
- prolog_profiler_wrapper.pl
- requirements.txt
- run_all_experiments.sh
- analysis.ipynb
- seed_corpus/
- generated_data/
- manual_validation/
- qualitative_bias_sensitivity_analysis/
- LICENSE

Bias families in the seed corpus:
- overconfidence bias, hyperbolic discounting, confirmation bias, hindsight bias, availability bias, framing effect, bandwagon effect, anchoring bias

> Tip: all generated outputs are written to `./generated_data/` and are safe to delete/regenerate.

---

## Requirements

- **Python**: 3.11+
- **Pip packages**: see `requirements.txt` (installed below)
- **SWI‑Prolog**: required for profiling (`prolog_profiler_wrapper.pl` uses `swipl` on PATH)
  - macOS: `brew install swi-prolog`
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y swi-prolog`
  - Windows: Download installer from [https://www.swi-prolog.org](https://www.swi-prolog.org) and follow setup instructions.
- **API access**:
  - OpenAI API: Set `OPENAI_API_KEY` in your environment.
  - Groq API: Set `GROQ_API_KEY` in your environment.
- **Fonts** (for vector‑friendly PDFs): matplotlib is configured to embed text as vectors.

---

## Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/Francesco-Sovrano/PROB-SWE.git
    cd PROB-SWE
    ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Ensure SWI-Prolog is installed and on your `PATH` (used by `prolog_profiler_wrapper.pl`).
   ```bash
   swipl --version
   ```

5. Set your OpenAI API key (bash/zsh):

   Sign in at [https://platform.openai.com](https://platform.openai.com) and generate a new secret key under your account settings.

   Set your environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

   Alternatively, you can create a `.env` file in the project root with:

   ```dotenv
   OPENAI_API_KEY=your_api_key_here
   ```

   and install `python-dotenv` to load it automatically.

   Verify access:

   ```bash
   python -c "import openai; print(openai.Model.list())"
   ```

   You should see a list of available models if your key is valid.

6. Set your Groq API key (bash/zsh):

   Visit GroqCloud and generate a key. ([console.groq.com][1])

   Either export it in your shell:

   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

   or add it to a `.env` file:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key_here
   ```


---

## Quick Start

The commands below run a **small, low‑cost sanity check** end‑to‑end.  
They will read the existing seed corpus, generate a tiny batch, compute sensitivity, and render figures.

```bash
# 1) Generate a tiny augmented dataset (adjust model to what you have access to)
python 1_generate_data.py \
  --model gpt-4o-mini \
  --batch_size 2 \
  --n_runs 2 \
  --n_prolog_construction_reruns 1

# 2) Compute bias sensitivity for the generated outputs
python 2_compute_bias_sensitivity.py \
  --model gpt-4o-mini \
  --n_independent_runs_per_task 2

# 3) Plot sensitivity figures
python 3_visualize_bias_sensitivity.py --show_figures

# 4) Seed corpus stats
python 4_get_stats_about_seed_corpus.py

# 5) Bias‑awareness analysis (model assesses whether a decision is bias‑affected)
python 5_bias_awareness_analysis.py --model gpt-4o-mini

# 6) Heatmap for bias‑awareness
python 6_visualize_bias_awareness_analysis.py

# 7) Sensitivity vs correctness correlation
python 7_sensitivity_to_correctness_correlation_analysis.py
```

All outputs will appear under `generated_data/`.

> The repository already contains example outputs under `generated_data/` to help you verify your setup.

---

## Reproducing the Paper’s Results

The repo includes a `run_all_experiments.sh` pipeline. 

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## Scripts & Key Options

Below is a brief overview of the main scripts and the most relevant options.  
All scripts support `-h/--help` for full usage.

### 1. `1_generate_data.py`
Augments the human‑seeded dilemmas with AI‑generated variants, validating them with multiple runs.

Key options:
- `--model <str>`: model name (e.g., `gpt-4o-mini`, `gpt-4.1-mini`)
- `--batch_size <int>`: number of dilemmas to generate per batch (default: 5)
- `--n_runs <int>`: validation runs per AI‑generated dilemma (default: 2 in the file; raise for stronger validation)
- `--n_prolog_construction_reruns <int>`: retry building valid Prolog programs (default: 2)
- `--min_intra_model_agreement_rate_on_dilemma <float>`: keep only dilemmas that pass agreement (default: 0.8)

Outputs (examples): `generated_data/1_augmented_dilemmas_dataset_<model>.json`, `generated_data/1_stats_dataset_<model>.json`

### 2. `2_compute_bias_sensitivity.py`
Runs each (biased, unbiased) dilemma pair multiple times to quantify **sensitivity** and **harmfulness**.

Key options:
- `--model <str>`: evaluation model
- `--n_independent_runs_per_task <int>`: runs per dilemma (default: 5)
- `--complexity_metric <str>`: `inference_steps` (default) or `choice_steps`
- `--seed_corpus_only`: restrict to human‑seeded dilemmas
- `--inject_axioms`: include human‑readable axioms as reasoning cues
- `--inject_axioms_in_prolog`: include Prolog‑encoded axioms as cues

Outputs (examples):  
- CSV of raw runs: `generated_data/2_llm_outputs_<ARGS>.csv`  
- Aggregates: `generated_data/2_bias_sensitivity_<ARGS>.json`

### 3. `3_visualize_bias_sensitivity.py`
Produces line/violin/heatmap plots summarizing sensitivity across biases and complexity tiers.

Useful flags:
- `--show_figures`: open windows in addition to writing PDFs
- `--seed_corpus_only`, `--inject_axioms`, `--inject_axioms_in_prolog` as above

Outputs: `generated_data/3_*.pdf`, and `generated_data/3_bias_sensitivity_stats_by_complexity_stats.csv`

### 4. `4_get_stats_about_seed_corpus.py`
Descriptive stats for the human‑seeded corpus; also produces a figure.

Outputs: `generated_data/4_seed_corpus_stats.txt`, `generated_data/4_decision_agreement_on_seed_corpus.pdf`

### 5. `5_bias_awareness_analysis.py`
Estimates whether a model can **recognize** its own bias‑affected decisions (bias awareness score per bias type).

Key options:
- `--model <str>`: assessing model
- `--data_model <str>`: (optional) analyze runs produced by a different model

Outputs: `generated_data/5_bias_awareness_analysis_model=<model>.json`

### 6. `6_visualize_bias_awareness_analysis.py`
Renders a compact heatmap of bias‑awareness scores for each model.  
Output: `generated_data/6_bias_awareness_analysis_heatmap.pdf`

### 7. `7_sensitivity_to_correctness_correlation_analysis.py`
Analyzes the correlation between sensitivity and correctness.  
Output: `generated_data/7_sensitivity_to_correctness_correlation_analysis.pdf`

---

## Seed Corpus

Under `seed_corpus/` you’ll find subdirectories for each bias (e.g. `confirmation_bias`, `hindsight_bias`), containing:

* `0-unbiased_task.txt` / `.pl`
* `1-biased_task.txt` / `.pl`
* `axioms.txt` / `.pl`

These define the natural-language prompts and Prolog axioms to inject.

---

## Manual Validation Workflow

This package includes a helper to sample AI outputs for expert validation and task‑specific CSVs.  
- Script: `sample_data_for_manual_validation.py`
- Instructions: `manual_validation/instructions.md`

**Example**:
```bash
python sample_data_for_manual_validation.py \
  --input ./generated_data/2_llm_outputs_model=gpt-4o-mini.csv \
  --output_dir ./manual_validation/analysis_results \
  --seed 42 \
  --n 5
```

This produces `task1_same_task.csv`, `task2_bias_presence.csv`, etc., with blank judgment columns for annotators.

> Completed CSVs can be compared or aggregated using helper scripts under `manual_validation/` and `qualitative_bias_sensitivity_analysis/`.

---

## Outputs & File Naming

All analysis artefacts are written to `./generated_data/`. Representative examples in the provided archive include:
- `1_augmented_dilemmas_dataset_<model>.json` — augmented dilemmas
- `2_llm_outputs_<ARGS>.csv` — raw per‑run decisions
- `2_bias_sensitivity_<ARGS>.json` — aggregated sensitivity & harmfulness (incl. complexity tiers)
- `3_*` PDFs — figures (lines, violins, heatmaps, stats tables)
- `4_*` — corpus stats
- `5_bias_awareness_analysis_model=<model>.json`
- `6_bias_awareness_analysis_heatmap.pdf`
- `7_sensitivity_to_correctness_correlation_analysis.pdf`

> `<ARGS>` encodes key CLI flags (e.g., `model=...`, `data_model_list=[...]`, etc.) to ensure traceability.

---

## Troubleshooting

- **`swipl: command not found`** → Install SWI‑Prolog and ensure it’s on PATH (`swipl --version` should work).
- **OpenAI quota / model not found** → Check your `OPENAI_API_KEY` and that the `--model` name is available to your account.
- **Matplotlib font warnings** → The scripts already set vector‑friendly font types; the warnings are harmless.
- **Large runs are slow/costly** → Start with the **Quick Start** settings and scale up once everything works.

---

## Citing

If you use PROBE‑SWE in academic work, please cite the accompanying paper:

```bibtex
@inproceedings{{probe-swe,
  title     = {{Is General-Purpose AI Reasoning Sensitive to Data-Induced Cognitive Biases? Dynamic Benchmarking on Typical Software Engineering Dilemmas}},
  author    = {{Francesco Sovrano, Gabriele Dominici, Rita Sevastjanova, Alessandra Stramiglio, Alberto Bacchelli}},
  year      = {{2025}},
}}
```

---

## License

This project is released under the **MIT License** (see `LICENSE`).

---

## Contact

Questions or suggestions? Open an issue or reach out to Francesco Sovrano.
