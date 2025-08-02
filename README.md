# Is General-Purpose AI Reasoning Sensitive to Data-Induced Cognitive Biases? Dynamic Benchmarking on Typical Software Engineering Dilemmas

A framework for generating, validating, and analyzing data-induced cognitive biases in software-engineering tasks. We create a seed corpus of bias-focused tasks, generate model outputs, measure model sensitivity to bias, and compute corpus statistics.

## Table of Contents

- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [1. Generate Data](#1-generate-data)  
  - [2. Validate Data](#2-validate-data)  
  - [3. Compute Bias Sensitivity](#3-compute-bias-sensitivity)  
  - [4. Corpus Statistics](#4-corpus-statistics)  
  - [All at Once](#all-at-once)  
- [Seed Corpus](#seed-corpus)  
- [Generated Data & Logs](#generated-data--logs)  
- [License](#license)  

---

## Features

- **Seed corpus** of paired “biased” vs. “unbiased” tasks covering multiple cognitive biases (e.g., confirmation bias, hindsight bias).  
- **Automated data generation** via LLMs (GPT-4.1-nano, GPT-4O-mini) with Prolog-based axioms injection.  
- **Validation** of generated outputs for consistency and coverage.  
- **Bias sensitivity analysis** to quantify model performance differences on biased vs. unbiased prompts.  
- **Corpus statistics** for deeper insights into task distribution, length, and complexity.

---

## Prerequisites

* Python ≥ 3.11
* **SWI‑Prolog (swipl)** installed and on your `PATH`:

  * **macOS** (Homebrew): `brew install swi-prolog`
  * **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install swi-prolog`
  * **Windows**: Download installer from [https://www.swi-prolog.org](https://www.swi-prolog.org) and follow setup instructions.
* An OpenAI API key (see [OpenAI API Setup](#openai-api-setup))

---

## Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/your-org/data-induced-cognitive-bias-detection-for-software-engineering.git
    cd data-induced-cognitive-bias-detection-for-software-engineering
    ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Ensure Prolog is installed and on your `PATH` (used by `prolog_profiler_wrapper.pl`).

---

## OpenAI API Setup

1. **Obtain an API key**:

   * Sign in at [https://platform.openai.com](https://platform.openai.com) and generate a new secret key under your account settings.

2. **Set your environment variable**:

   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

   Alternatively, you can create a `.env` file in the project root with:

   ```dotenv
   OPENAI_API_KEY=your_api_key_here
   ```

   and install `python-dotenv` to load it automatically.

3. **Verify access**:

   ```bash
   python -c "import openai; print(openai.Model.list())"
   ```

   You should see a list of available models if your key is valid.

---

## Usage

Run the end-to-end pipeline:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## Seed Corpus

Under `seed_corpus/` you’ll find subdirectories for each bias (e.g. `confirmation_bias`, `hindsight_bias`), containing:

* `0-unbiased_task.txt` / `.pl`
* `1-biased_task.txt` / `.pl`
* `axioms.txt` / `.pl`

These define the natural-language prompts and Prolog axioms to inject.

---

## Generated Data & Logs

* **`generated_data/`**: contains the output of the `run_all_experiments.sh` script raw LLM responses, structured CSV outputs, and intermediate files.
* **`logs/`**: detailed run logs for each step, useful for debugging and audit trails.

---

## License

This project is licensed under the [MIT License](LICENSE).
