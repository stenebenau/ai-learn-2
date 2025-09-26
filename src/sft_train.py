# src/sft_train.py
import sys
from importlib.metadata import PackageNotFoundError, version
from packaging.version import parse

# --- Environment Check for transformers version ---
# This script requires a recent version of the 'transformers' library.
# This check ensures the environment is set up correctly.
try:
    required_version = "4.44.0"
    installed_version = version("transformers")
    if parse(installed_version) < parse(required_version):
        sys.stderr.write(
            f"ERROR: Your 'transformers' version is {installed_version}, but version >= {required_version} is required.\n"
            "This can lead to import errors like 'cannot import name LossKwargs'.\n\n"
            "Please update your environment by activating it and running 'make setup':\n"
            "  conda activate crm-dedup-llm\n"
            "  make setup\n"
        )
        sys.exit(1)
except PackageNotFoundError:
    sys.stderr.write(
        "ERROR: The 'transformers' library is not installed.\n\n"
        "Please set up your environment by activating it and running 'make setup':\n"
        "  conda activate crm-dedup-llm\n"
        "  make setup\n"
    )
    sys.exit(1)


# --- Troubleshooting Note for Stale Caches ---
# If you have updated 'transformers' and still see an ImportError related to
# model code (e.g., "cannot import name 'LossKwargs' from .../.cache/huggingface/..."),
# your Hugging Face cache for the model is likely stale.
#
# To fix this, run the following command from your project root:
#   make clean-cache
#
# This will remove the cached model-specific Python files and force a fresh download.


import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(config_path: str):
    """Main function to run the SFT training process."""
    logging.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    peft_config = config["peft"]
    train_config = config["train"]

    # --- 1. Load Model and Tokenizer ---
    logging.info(f"Loading base model: {config['model_id']}")

    # QLoRA configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if train_config["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True, # Required for some models like Qwen
    )
    model.config.use_cache = False # Recommended for training

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer `pad_token` set to `eos_token`")

    # --- 2. Prepare Datasets ---
    data_dir = Path("data/processed")
    train_dataset = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    val_dataset = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="train")
    logging.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")

    # --- 3. Configure PEFT (LoRA) ---
    lora_config = LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["alpha"],
        lora_dropout=peft_config["dropout"],
        target_modules=peft_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_config["gradient_ckpt"])

    # --- 4. Set up Training Arguments ---
    output_dir_base = Path("outputs")
    model_name = Path(config_path).stem
    output_dir = output_dir_base / f"{model_name}-lora-adapters"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["grad_accum"],
        learning_rate=train_config["lr"],
        num_train_epochs=train_config["epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=train_config["bf16"],
        fp16=not train_config["bf16"],
        gradient_checkpointing=train_config["gradient_ckpt"],
        report_to="none", # can be changed to "tensorboard" or "wandb"
        load_best_model_at_end=True,
    )

    # --- 5. Initialize SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="messages", # SFTTrainer will format our 'messages' column
        max_seq_length=train_config["max_seq_len"],
        args=training_args,
        packing=True, # Pack multiple short examples into one sequence for efficiency
    )

    # --- 6. Train ---
    logging.info("Starting SFT training...")
    trainer.train()
    logging.info("Training complete.")

    # --- 7. Merge and Save Final Model ---
    logging.info("Merging LoRA adapters and saving the final model...")
    
    # It's recommended to unload the adapter before merging
    del model
    torch.cuda.empty_cache()

    # Reload base model in fp16/bf16 and merge
    merged_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16 if train_config["bf16"] else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Load the LoRA adapter
    merged_model.load_adapter(str(output_dir))
    # Merge the adapter into the base model
    merged_model = merged_model.merge_and_unload()

    merged_model_path = output_dir_base / f"{model_name}-merged"
    merged_model.save_pretrained(str(merged_model_path))
    tokenizer.save_pretrained(str(merged_model_path))

    logging.info(f"Merged model saved to {merged_model_path}")
    logging.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using SFT and LoRA.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
