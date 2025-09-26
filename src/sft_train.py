'''
This script performs Supervised Fine-Tuning (SFT) on a causal language model
using the TRL library, QLoRA for parameter-efficient fine-tuning, and BitsAndBytes
for 4-bit quantization.

This version is updated to use the modern APIs for both transformers and TRL.
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
# IMPORTANT: Import the new SFTConfig object
from trl import SFTConfig, SFTTrainer

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    config = load_config(config_path)
    model_config = config.get("model", {})
    peft_config = config.get("peft", {})
    train_config = config.get("train", {})

    # --- 1. Load Model and Tokenizer ---
    model_id = model_config.get("id")
    logging.info(f"Loading base model: {model_id}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if train_config.get("bf16") else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

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
        # Use the new, correct argument name for the evaluation strategy
        eval_strategy="epoch",
        bf16=train_config.get("bf16", False),
        fp16=not train_config.get("bf16", False),
        gradient_checkpointing=train_config.get("gradient_ckpt", True),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # --- 5. Initialize SFTTrainer (with the new SFTConfig) ---
    
    # Create the new SFTConfig object to hold SFT-specific parameters
    sft_config = SFTConfig(
        dataset_text_field="messages",
        max_seq_length=train_config.get("max_seq_len", 512),
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_arguments,
        # Pass the new config object here
        sft_config=sft_config,
    )

    # --- 6. Train ---
    logging.info("Starting Supervised Fine-Tuning...")
    trainer.train()
    logging.info("Training complete.")

    # --- 7. Merge and Save Final Model ---
    logging.info("Merging LoRA adapters and saving the final model...")
    
    del trainer
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if train_config.get("bf16") else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    from peft import PeftModel
    merged_model = PeftModel.from_pretrained(base_model, str(output_dir))
    merged_model = merged_model.merge_and_unload()

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