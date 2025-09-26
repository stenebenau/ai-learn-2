# src/build_prompts.py
import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# System prompt as defined in goal.md
SYSTEM_PROMPT = (
    "You are a CRM data steward. Return **strict JSON** with keys: `result` "
    "(one of `DUPLICATE`, `POSSIBLE`, `NO_MATCH`) and `reasoning` "
    "(one short sentence). No extra text."
)


def format_example(example: dict) -> dict | None:
    """
    Formats a single example from the raw JSONL into the conversational format.

    Args:
        example: A dictionary loaded from a line in the raw JSONL file.

    Returns:
        A dictionary in the required conversational format, or None if input is malformed.
    """
    try:
        # User content is a pretty-printed JSON of the input records
        user_content = json.dumps(example["input"], indent=2)

        # Assistant content is the gold JSON output, as a compact string
        assistant_content = json.dumps(example["output"])

        # The final structure for the dataset, ready for the trainer
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
    except (KeyError, TypeError) as e:
        logging.error(f"Skipping malformed record: {example}. Error: {e}")
        return None


def build_prompts(input_file: Path, output_file: Path):
    """
    Reads a JSONL file, formats each line into a conversational prompt,
    and writes the result to a new JSONL file.
    """
    logging.info(f"Reading from {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Processing {len(lines)} records and writing to {output_file}...")
    processed_count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in lines:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                formatted_record = format_example(record)
                if formatted_record:
                    f_out.write(json.dumps(formatted_record) + "\n")
                    processed_count += 1
            except json.JSONDecodeError:
                logging.warning(f"Skipping line due to JSON decode error: {line.strip()}")

    logging.info(f"Successfully processed and wrote {processed_count} records.")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Build conversational prompts from raw JSONL data."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the input JSONL file (e.g., data/train.jsonl).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to the output JSONL file for the formatted prompts.",
    )
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    build_prompts(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
