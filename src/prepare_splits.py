import argparse
import json
from pathlib import Path
from datasets import load_dataset

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

def prepare_splits(input_file: Path, output_dir: Path):
    """
    Loads a JSONL dataset, performs a stratified 80/10/10 split,
    and saves the splits to train.jsonl, val.jsonl, and test.jsonl.
    """
    if not input_file.exists():
        create_dummy_splits(output_dir, str(input_file))
        return

    print(f"Loading dataset from {input_file}...")
    dataset = load_dataset("json", data_files=str(input_file), split="train")

    # Extract the label for stratification
    def get_label(example):
        return {"label": example["output"]["result"]}

    dataset = dataset.map(get_label, num_proc=4)

    print("Performing stratified split (80/10/10)...")
    # Split into 80% train and 20% temp
    train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]

    # Split temp into 50% validation and 50% test (10% and 10% of original)
    val_test_split = temp_dataset.train_test_split(test_size=0.5, stratify_by_column="label", seed=42)
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
