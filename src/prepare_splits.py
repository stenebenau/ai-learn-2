import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_dummy_splits(output_dir: Path, input_file_name: str):
    """Creates dummy data files if the source file is missing."""
    print(f"Warning: Input file not found at '{input_file_name}'")
    print("Creating dummy data files for demonstration purposes.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_records = [
        {"input": {"record1": {"id": "1", "name": "John Smith"}, "record2": {"id": "2", "name": "Jon Smith"}}, "output": {"result": "POSSIBLE", "reasoning": "Names are similar."}},
        {"input": {"record1": {"id": "3", "email": "test@test.com"}, "record2": {"id": "4", "email": "test@test.com"}}, "output": {"result": "DUPLICATE", "reasoning": "Emails are identical."}},
        {"input": {"record1": {"id": "5", "company": "Acme Inc"}, "record2": {"id": "6", "company": "Globex Corp"}}, "output": {"result": "NO_MATCH", "reasoning": "Companies are different."}},
    ]
    
    def write_jsonl(path: Path, records: list):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    # Create small but representative splits (80/10/10 ratio is not critical for dummy data)
    write_jsonl(output_dir / "train.jsonl", dummy_records * 8)
    write_jsonl(output_dir / "val.jsonl", dummy_records)
    write_jsonl(output_dir / "test.jsonl", dummy_records)

    print(f"Dummy splits created in {output_dir}. Totals: 24 train, 3 val, 3 test.")
    print(f"To use real data, place your dataset at '{input_file_name}' and re-run 'make prep'.")

def stringify_record_values(d: dict) -> dict:
    """Recursively converts all values in a dictionary to strings."""
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = stringify_record_values(v)
        elif v is not None:
            d[k] = str(v)
    return d

def prepare_splits(input_file: Path, output_dir: Path):
    """
    Loads a JSONL dataset, performs a stratified 80/10/10 split,
    and saves the splits to train.jsonl, val.jsonl, and test.jsonl.
    """
    if not input_file.exists():
        create_dummy_splits(output_dir, str(input_file))
        return

    print(f"Loading dataset from {input_file}...")
    
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print("Normalizing data types by converting all input record values to strings...")
    for record in records:
        try:
            if "input" in record and isinstance(record["input"], dict):
                if "record1" in record["input"]:
                    record["input"]["record1"] = stringify_record_values(record["input"]["record1"])
                if "record2" in record["input"]:
                    record["input"]["record2"] = stringify_record_values(record["input"]["record2"])
        except (KeyError, TypeError):
            pass

    dataset = Dataset.from_list(records)

    def get_label(example):
        return {"label": example["output"]["result"]}

    dataset = dataset.map(get_label, num_proc=4)

    print("Casting label column for stratification...")
    unique_labels = sorted(dataset.unique("label"))
    dataset = dataset.cast_column("label", ClassLabel(names=unique_labels))

    print("Performing data splitting (80/10/10)...")
    min_members_for_split = 2
    stratify_by = "label"

    # --- First split: train vs temp (val+test) ---
    label_counts = pd.Series(dataset["label"]).value_counts()
    if (label_counts < min_members_for_split).any():
        problematic_classes = label_counts[label_counts < min_members_for_split]
        logging.warning(
            f"Cannot perform first stratified split because some classes have fewer than {min_members_for_split} members."
        )
        logging.warning(f"Problematic classes and their counts:\n{problematic_classes}")
        logging.warning("Switching to a non-stratified split for this level. Data distribution will not be preserved.")
        stratify_by = None

    train_test_split = dataset.train_test_split(
        test_size=0.2, stratify_by_column=stratify_by, seed=42
    )
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]

    # --- Second split: val vs test ---
    if stratify_by is not None:
        label_counts_temp = pd.Series(temp_dataset["label"]).value_counts()
        if (label_counts_temp < min_members_for_split).any():
            problematic_classes = label_counts_temp[label_counts_temp < min_members_for_split]
            logging.warning(
                "Cannot perform second stratified split on the validation/test set because some classes have fewer than "
                f"{min_members_for_split} members in the temporary split."
            )
            logging.warning(f"Problematic classes and their counts:\n{problematic_classes}")
            logging.warning("Switching to a non-stratified split for this level. Data distribution will not be preserved.")
            stratify_by = None
    
    val_test_split = temp_dataset.train_test_split(
        test_size=0.5, stratify_by_column=stratify_by, seed=42
    )
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    # Remove the temporary 'label' column
    train_dataset = train_dataset.remove_columns("label")
    val_dataset = val_dataset.remove_columns("label")
    test_dataset = test_dataset.remove_columns("label")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.to_json(output_dir / "train.jsonl", force_ascii=False)
    val_dataset.to_json(output_dir / "val.jsonl", force_ascii=False)
    test_dataset.to_json(output_dir / "test.jsonl", force_ascii=False)

    print(f"Splits saved to {output_dir}:")
    print(f"  - train.jsonl: {len(train_dataset)} records")
    print(f"  - val.jsonl:   {len(val_dataset)} records")
    print(f"  - test.jsonl:  {len(test_dataset)} records")

def main():
    parser = argparse.ArgumentParser(description="Prepare stratified data splits from a single JSONL file.")
    parser.add_argument("--in", dest="input_file", type=Path, required=True, help="Input JSONL file (e.g., data/all.jsonl).")
    parser.add_argument("--out", dest="output_dir", type=Path, required=True, help="Output directory to save splits (e.g., data/).")
    args = parser.parse_args()
    
    prepare_splits(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
