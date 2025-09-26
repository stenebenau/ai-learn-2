# CRM Deduplication LLM

This project fine-tunes and benchmarks small, instruction-tuned Large Language Models (LLMs) for a CRM data deduplication task. The goal is to create a model that acts like a human data steward: it analyzes two CRM records and outputs a strict JSON object with a decision (`DUPLICATE`, `POSSIBLE`, `NO_MATCH`) and a human-readable reasoning.

## Features

- **Supervised Fine-Tuning (SFT)** using memory-efficient QLoRA on a custom dataset.
- **Constrained JSON Output** using a JSON Schema to guarantee valid, structured responses from the model.
- **Comprehensive Evaluation** of standard classification KPIs (Accuracy, F1-Score, Confusion Matrix).
- **Latency & Throughput Benchmarking** against multiple backends (single-process Transformers vs. high-throughput vLLM).
- **Orchestration via `Makefile`** for a simple, repeatable workflow.

## Models Supported

The project is configured to compare the following models:

- `microsoft/Phi-4-mini-instruct`
- `Qwen/Qwen3-4B-Instruct-2507`
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Prerequisites

- **Hardware**: An NVIDIA GPU with CUDA support. A GPU with >= 24GB VRAM is recommended for training all models.
- **Software**:
    - `conda` or `mamba` for environment management.
    - `git` for cloning the repository.
- **Hugging Face Account**:
    - You must have access to gated models like Llama 3.2 on the Hugging Face Hub.
    - Authenticate with the Hugging Face CLI in your terminal before you begin:
      ```bash
      huggingface-cli login
      ```

## Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create the Conda environment:**
    This command uses `env.yml` to install all required dependencies in a new conda environment named `crm-dedup-llm`.
    ```bash
    make setup
    ```

3.  **Activate the environment:**
    You must activate this environment in every new terminal session to use the project.
    ```bash
    conda activate crm-dedup-llm
    ```

## Workflow and Usage

The project workflow is managed through `make` commands.

### Step 1: Prepare Data

Place your complete dataset in `data/all.jsonl`. The file must be in the [JSON Lines](https://jsonlines.org/) format, where each line is a JSON object containing `"input"` and `"output"` keys, matching the structure in `goal.md`.

Then, run the preparation script:

```bash
make prep
```

This command performs two actions:
1.  It runs `src/prepare_splits.py` to create stratified `train`, `val`, and `test` splits from `data/all.jsonl`.
2.  It runs `src/build_prompts.py` to convert the `train` and `val` splits into the conversational format required for training, saving them in `data/processed/`.

> **Note:** If `data/all.jsonl` is not found, the script will generate a small set of dummy data so you can test the rest of the pipeline.

### Step 2: Train a Model

Train a model using the `train-<model_name>` target, where `<model_name>` corresponds to a configuration file in the `configs/` directory (e.g., `phi4mini`, `mistral7b`).

```bash
# Example: Train the Phi-4-mini model
make train-phi4mini
```

This process performs QLoRA fine-tuning and saves the final, merged model ready for inference into the `outputs/<model_name>-merged/` directory.

### Step 3: Evaluate Model KPIs

After training, evaluate the model's performance on the test set to measure classification accuracy, F1-scores, and other metrics.

```bash
# Example: Evaluate the trained Phi-4-mini model
make kpi-phi4mini
```

This command runs `src/evaluate_kpis.py` and saves a detailed JSON report to `reports/phi4mini_metrics.json`.

### Step 4: Benchmark Latency & Throughput

This step measures the model's inference speed using `vLLM`. It requires two terminals.

1.  **In Terminal 1, start the vLLM server:**
    Make sure your conda environment is active. Point the server to the merged model you trained in Step 2.

    ```bash
    # Example: Start a server for the fine-tuned Phi-4-mini model
    python -m vllm.entrypoints.openai.api_server \
      --model outputs/phi4mini-merged \
      --trust-remote-code \
      --dtype bfloat16
    ```
    Leave this server running.

2.  **In Terminal 2, run the benchmark client:**
    The client will send requests to the server at various concurrency levels and measure performance.

    ```bash
    # Example: Benchmark the Phi-4-mini model
    make bench-phi4mini
    ```
    This command runs `src/eval_latency_throughput.py` and saves the results to `reports/phi4mini_latency.csv`.

### Step 5: Compare Models

After training and evaluating all models, open the generated JSON and CSV files in the `reports/` directory. Consolidate the key metrics into the summary table in `reports/model_comparison.md` to compare performance and select the best model for your needs.

## Repository Structure

```
configs/            # Model/training configs and the JSON schema for decoding.
data/               # Raw, split, and processed datasets.
src/                # All Python source code.
reports/            # Output directory for metrics, benchmarks, and comparison.
env.yml             # Conda environment definition.
Makefile            # Orchestrates all project tasks.
README.md           # This file.
```

## Makefile Targets

- `setup`: Create or update the conda environment.
- `prep`: Prepare data splits and build prompts for training.
- `train-<model>`: Train a model (e.g., `make train-phi4mini`).
- `kpi-<model>`: Evaluate a model's KPIs (e.g., `make kpi-phi4mini`).
- `bench-<model>`: Benchmark a model's latency/throughput (e.g., `make bench-phi4mini`).
- `clean`: Remove all generated files (processed data, models, reports).
