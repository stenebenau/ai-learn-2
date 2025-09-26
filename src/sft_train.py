'''
This script performs Supervised Fine-Tuning (SFT) on a causal language model
using the modern APIs for the TRL and Transformers libraries.
'''

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# Use the new, unified SFTConfig from TRL
from trl import SFTConfig, SFTTrainer

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- NEW: Data Formatting Function ---
def formatting_func(example):
    """
    Takes a raw data example from the JSONL file and transforms it into the
    chat format that the SFTTrainer expects.
    """
    # Serialize the input records into a formatted string for the 'user' turn
    input_records = example['input']
    user_prompt = f"""### Input
        **Record 1:**
        ```json
        {json.dumps(input_records['record1'], indent=2)}
        ```
        **Record 2:**
        ```json
        {json.dumps(input_records['record2'], indent=2)}
        ```
        Given these two records, determine if they are duplicates and provide your reasoning in JSON format."""
    
    # Serialize the output into a string for the 'assistant' turn
    assistant_response = f"""### Output
                {json.dumps(example['output'], indent=2)}
        """
    # Create the 'messages' list in the required conversational format
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
    return {"messages": messages}

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


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

    # --- 2. Load Datasets (NO MANUAL MAPPING) ---
    data_dir = Path("data/processed")
    train_dataset = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    val_dataset = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="validation")
    logging.info(val_dataset)
    logging.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples.")
    
    logging.info(f"Loaded and formatted {len(train_dataset)} training and {len(val_dataset)} validation examples.")

    # --- 3. Configure PEFT (LoRA) ---
    lora_config = LoraConfig(
        r=peft_config.get("r", 16),
        lora_alpha=peft_config.get("alpha", 32),
        lora_dropout=peft_config.get("dropout", 0.05),
        target_modules=peft_config.get("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # --- 4. Set up Unified SFT Configuration ---
    output_dir_base = Path("outputs")
    run_name = Path(config_path).stem
    output_dir = output_dir_base / f"{run_name}-lora-adapters"

    # Use SFTConfig to hold ALL arguments. This replaces TrainingArguments.
    sft_config = SFTConfig(
        # SFT-specific arguments
        dataset_text_field="messages",
        max_seq_length=train_config.get("max_seq_len", 512),
        packing=True,
        
        # Training arguments
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config.get("batch_size", 4),
        gradient_accumulation_steps=train_config.get("grad_accum", 2),
        learning_rate=train_config.get("lr", 2e-4),
        num_train_epochs=train_config.get("epochs", 3),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=train_config.get("bf16", False),
        fp16=not train_config.get("bf16", False),
        gradient_checkpointing=train_config.get("gradient_ckpt", True),
        report_to="none",
        load_best_model_at_end=True,
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
        # Pass the single, unified config object to the 'args' parameter
        args=sft_config
    )

    # --- 6. Train ---
    logging.info("Starting Supervised Fine-Tuning with modern TRL API...")
    trainer.train()
    logging.info("Training complete.")

    # --- 7. Merge and Save Final Model ---
    logging.info("Saving the final merged model...")
    merged_model_path = output_dir_base / f"{run_name}-merged"
    trainer.save_model(str(merged_model_path))
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