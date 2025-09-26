# Makefile

.PHONY: all setup prep train-% kpi-% bench-% clean

# Variables
CONFIG_DIR := configs
DATA_DIR := data
PROCESSED_DIR := $(DATA_DIR)/processed
OUTPUT_DIR := outputs
REPORTS_DIR := reports
SRC_DIR := src

# Default target
all:
	@echo "Available targets:"
	@echo "  setup          - Create/update conda environment"
	@echo "  prep           - Prepare data splits and build prompts for training"
	@echo "  train-<model>  - Train a model (e.g., make train-phi4mini)"
	@echo "  kpi-<model>    - Evaluate a model's KPIs (e.g., make kpi-phi4mini)"
	@echo "  bench-<model>  - Benchmark a model's latency/throughput (e.g., make bench-phi4mini)"
	@echo "  clean          - Remove generated files"

# Environment setup
setup:
	conda env create -f env.yml || conda env update -f env.yml --prune

# Data preparation
prep:
	@echo "Running data preparation..."
	python $(SRC_DIR)/prepare_splits.py --in $(DATA_DIR)/all.jsonl --out $(DATA_DIR)
	@echo "Building prompts for training and validation..."
	mkdir -p $(PROCESSED_DIR)
	python $(SRC_DIR)/build_prompts.py --input-file $(DATA_DIR)/train.jsonl --output-file $(PROCESSED_DIR)/train.jsonl
	python $(SRC_DIR)/build_prompts.py --input-file $(DATA_DIR)/val.jsonl --output-file $(PROCESSED_DIR)/val.jsonl
	@echo "Data preparation complete. Prompted data is in $(PROCESSED_DIR)"

# Pattern rule for training
# Example: make train-phi4mini
train-%:
	@echo "Starting training for $*..."
	accelerate launch $(SRC_DIR)/sft_train.py --config $(CONFIG_DIR)/$*.yaml

# Pattern rule for KPI evaluation
# Example: make kpi-phi4mini
kpi-%:
	@echo "Evaluating KPIs for $*..."
	python $(SRC_DIR)/evaluate_kpis.py \
	  --model_path $(OUTPUT_DIR)/$*-merged \
	  --backend sglang \
	  --schema $(CONFIG_DIR)/decoding.jsonschema \
	  --data $(DATA_DIR)/test.jsonl \
	  --out $(REPORTS_DIR)/$*_metrics.json

# Pattern rule for latency/throughput benchmarking
# Example: make bench-phi4mini
bench-%:
	@echo "Benchmarking latency/throughput for $*..."
	python $(SRC_DIR)/eval_latency_throughput.py \
	  --model_path $(OUTPUT_DIR)/$*-merged \
	  --backend vllm \
	  --data $(DATA_DIR)/test.jsonl \
	  --schema $(CONFIG_DIR)/decoding.jsonschema \
	  --concurrency 1 4 8 16 \
	  --out $(REPORTS_DIR)/$*_latency.csv

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf $(PROCESSED_DIR)
	rm -rf $(OUTPUT_DIR)
	rm -f $(REPORTS_DIR)/*.json $(REPORTS_DIR)/*.csv
	@echo "Cleanup complete."
