# Case Generation and Evaluation Process

This repository automates the generation of cases, aspect filtering, merging data, adding LLM context, and performing evaluation.

## Steps Overview

1. **Manual Setup** : You need to define custom parameters manually in the relevant source files before starting the process.
2. **Run Scripts** : The following scripts are executed in order:

* `run.py` (case generation initialization)
* `aspects_filter.py` (aspect filtering)
* `test_case_builder.py` (case generation)
* `Merge_json.py` (merge data)
* `add_context_LLM.py` (add context via LLM)

1. **Evaluation** : Use the current evaluation method or prepare for future integration into a unified pipeline.

## Manual Parameter Definition

Before running the scripts, set the necessary parameters in your source files (e.g., configuration files). You need to manually open and update these files with your specific values. The automation process assumes the parameters are properly set before execution.

## Run a demo of generation:

python main.py
