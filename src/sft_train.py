'''
This script performs Supervised Fine-Tuning (SFT) on a causal language model
using the TRL library, QLoRA for parameter-efficient fine-tuning, and BitsAndBytes
for 4-bit quantization.

It is designed to be run from the command line and configured via a YAML file.

Usage:
    accelerate launch src/sft_train.py --config configs/your_config.yaml
'''

import argparse
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from packaging.version import parse
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Environment Sanity Check ---
def check_environment():
    """
    Checks if the installed 'transformers' library meets the minimum version requirement.
    This helps prevent common errors related to outdated library APIs.
    """
    try:
        required_version = "4.44.0"
        installed_version_str = version("transformers")
        installed_version = parse(installed_version_str)

        if installed_version < parse(required_version):
            logging.error(
                f"Your 'transformers' version is {installed_version_str}, but this script "
                f"requires version >= {required_version}. Please upgrade your library."
            )
            sys.exit(1)
        logging.info(f"Transformers version {installed_version_str} meets requirements.")

    except PackageNotFoundError:
        logging.error(
            "The 'transformers' library is not installed. Please set up your environment."
        )
        sys.exit(1)


# --- Core Functions ---
def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path: str):
    """
    Main function to orchestrate the SFT training process.
    """
    check_environment()
    config = load_config(config_path)
    model_config = config.get("model", {})
    peft_config = config.get("peft", {})
    train_config = config.get("train", {})

    # --- 1. Load Model and Tokenizer ---
    model_id = model_config.get("id")
    logging.info(f"Loading base model: {model_id}")

    # Configure 4-bit quantization (QLoRA)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if train_config.get("bf16") else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load the model with quantization and map it to the current CUDA device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically map to available GPUs
        trust_remote_code=True, # Essential for models like Phi or Qwen
    )
    # Disable cache for training, as it's only useful for inference
    model.config.use_cache = False
    model.config.pretraining_tp = 1 # Fix for some models that have a parallel pre-training setup

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer `pad_token` was not set. Setting it to `eos_token`.")

    # --- 2. Prepare Datasets ---
    data_dir = Path("data/processed")
    train_dataset = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    val_dataset = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="train")
    logging.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples.")

    # --- 3. Configure PEFT (LoRA) ---
    lora_config = LoraConfig(
        r=peft_config.get("r", 16),
        lora_alpha=peft_config.get("alpha", 32),
        lora_dropout=peft_config.get("dropout", 0.05),
        target_modules=peft_config.get("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare the model for k-bit training, which freezes base layers and adds LoRA adapters
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=train_config.get("gradient_ckpt", True)
    )

    # --- 4. Set up Training Arguments ---
    output_dir_base = Path("outputs")
    run_name = Path(config_path).stem
    output_dir = output_dir_base / f"{run_name}-lora-adapters"

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config.get("batch_size", 4),
        gradient_accumulation_steps=train_config.get("grad_accum", 2),
        learning_rate=train_config.get("lr", 2e-4),
        num_train_epochs=train_config.get("epochs", 3),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch", # Evaluate at the end of each epoch
        bf16=train_config.get("bf16", False),
        fp16=not train_config.get("bf16", False),
        gradient_checkpointing=train_config.get("gradient_ckpt", True),
        report_to="none", # Can be "tensorboard" or "wandb"
        load_best_model_at_end=True, # Requires evaluation_strategy and save_strategy to be the same
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # --- 5. Initialize SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="messages", # The column in your dataset that SFTTrainer should format
        max_seq_length=train_config.get("max_seq_len", 512),
        args=training_arguments,
        packing=True, # Pack multiple short examples into one sequence for efficiency
    )

    # --- 6. Train ---
    logging.info("Starting Supervised Fine-Tuning...")
    trainer.train()
    logging.info("Training complete.")

    # --- 7. Merge and Save Final Model ---
    # This step creates a new, standalone model by merging the LoRA adapters
    # with the original base model.
    logging.info("Merging LoRA adapters and saving the final model...")
    
    # Free up memory by deleting the trainer and model
    del trainer
    del model
    torch.cuda.empty_cache()

    # Reload the base model in full precision (or bf16) to merge
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if train_config.get("bf16") else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load the LoRA adapter and merge
    from peft import PeftModel
    merged_model = PeftModel.from_pretrained(base_model, str(output_dir))
    merged_model = merged_model.merge_and_unload()

    # Save the merged model and tokenizer
    merged_model_path = output_dir_base / f"{run_name}-merged"
    merged_model.save_pretrained(str(merged_model_path))
    tokenizer.save_pretrained(str(merged_model_path))

    logging.info(f"Merged model saved successfully to: {merged_model_path}")
    logging.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model using SFT and LoRA.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)