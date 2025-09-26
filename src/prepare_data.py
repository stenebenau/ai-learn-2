import json
import argparse
from pathlib import Path
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import sys

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def format_and_save_dataset(input_file: Path, output_file: Path, tokenizer_id: str):
    """
    Loads a JSONL dataset, applies the chat template to the 'messages' column,
    and saves the result as a new JSONL file with a single 'text' field.
    """
    logging.info(f"Loading tokenizer: '{tokenizer_id}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Loading raw dataset from: '{input_file}'")
    dataset = load_dataset("json", data_files=str(input_file), split="train")

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    logging.info(f"Applying chat template and writing to '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for example in dataset:
            # The apply_chat_template function converts the list of dicts into a single formatted string
            formatted_text = tokenizer.apply_chat_template(example['messages'], tokenize=False)
            
            # Write the result into a new JSON object with a 'text' key
            f_out.write(json.dumps({"text": formatted_text}) + '\n')
            processed_count += 1

    logging.info(f"Successfully processed and saved {processed_count} examples.")


def main():
    """
    Main function to run the data preprocessing.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess chat-formatted JSONL data into a plain text JSONL file."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output-file", type=Path, required=True, help="Path for the formatted output JSONL file.")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model ID to load the correct tokenizer.")
    args = parser.parse_args()

    try:
        format_and_save_dataset(args.input_file, args.output_file, args.model_id)
    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()