# Bias Validation: Instructions for Validators

This document guides validators through reviewing sampled prompt pairs and related artefacts for their evaluation. 

You will find four task-specific CSV files in the output directory. Each file is structured for straight‑forward validation, containing blank columns for **your judgments**.

Judgments formatting instructions:
* **Use** exactly `True` or `False` (capitalized) in each validation column. You can add any relevant comments afterwards.
* **Save** your completed files with the same names and send back the full directory for aggregation.

Below you'll find more details about each task.

---

## Task 1: Same‑Task Check

* **File**: `task1_same_task.csv`
* **Columns**:

  * `version_A`: a neutral prompt
  * `version_B`: a biased prompt
  * `same_task_validation`: **YOUR JUDGMENT** (`True` or `False`)
* **Question**: Do both versions ask the same underlying question or are about the same task?

---

## Task 2: Bias Presence

* **File**: `task2_bias_presence.csv`
* **Columns**:

  * `bias_name`: the type of bias
  * `version_A`: prompt without bias
  * `version_B`: prompt with bias
  * `correct_option`: the intended correct answer choice
  * `bias_is_towards_incorrect_only_in_biased_version`: explanation flag
  * `bias_presence_validation`: **YOUR JUDGMENT** (`True` or `False`)
* **Question**: Does `version_B` introduce the specified bias (against `correct_option` or towards the other option) and is that bias absent in `version_A`?

---

## Task 3: Reconstruction Check

* **File**: `task3_reconstruction.csv`
* **Columns**:

  * `prompt`
  * `reconstructed_prompt`
  * `reconstruction_validation`: **YOUR JUDGMENT** (`True` or `False`)
* **Question**: Is the reconstructed prompt equivalent in meaning to the original unbiased prompt? (it's ok if one version mentions days where the other mention months)

---

## Task 4: Axioms Description

* **File**: `task4_axioms.csv`
* **Columns**:

  * `best_practices`
  * `best_practices_validation`: **YOUR JUDGMENT** (`True` or `False`)
* **Question**: Does the best practices clearly convey valid software engineering best practices?

---

Thank you for your thorough reviews! 