# src/evaluate_kpis.py
import sys
from importlib.metadata import PackageNotFoundError, version
from packaging.version import parse

# --- Environment Check for transformers version ---
# This script requires a recent version of the 'transformers' library for the 'transformers' backend.
# This check ensures the environment is set up correctly.
try:
    required_version = "4.44.0"
    installed_version = version("transformers")
    if parse(installed_version) < parse(required_version):
        sys.stderr.write(
            f"ERROR: Your 'transformers' version is {installed_version}, but version >= {required_version} is required.\n"
            "This can lead to import errors when using the 'transformers' backend.\n\n"
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


import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# System prompt (must be identical to the one used in training)
SYSTEM_PROMPT = (
    "You are a CRM data steward. Return **strict JSON** with keys: `result` "
    "(one of `DUPLICATE`, `POSSIBLE`, `NO_MATCH`) and `reasoning` "
    "(one short sentence). No extra text."
)

# Define labels for classification reports
LABELS = ["DUPLICATE", "POSSIBLE", "NO_MATCH"]


# --- Backend Abstraction ---
class GenerationBackend(ABC):
    """Abstract base class for generation backends."""

    def __init__(self, model_path: str, schema: dict, gen_kwargs: dict):
        self.model_path = model_path
        self.schema = schema
        self.gen_kwargs = gen_kwargs
        logging.info(f"Initialized {self.__class__.__name__} for model: {model_path}")

    @abstractmethod
    def generate(self, prompts: list[list[dict]]) -> list[str]:
        """Generates text for a batch of prompts."""
        pass


# --- SGLang Backend ---
class SGLangBackend(GenerationBackend):
    def __init__(self, model_path: str, schema: dict, gen_kwargs: dict):
        super().__init__(model_path, schema, gen_kwargs)
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("SGLang is not installed. Please run 'pip install \"sglang>=0.3.0\"'.")

        self.backend = sgl.Runtime(model_path=model_path, trust_remote_code=True)
        sgl.set_default_backend(self.backend)
        self.schema_str = json.dumps(schema)

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        import sglang as sgl

        @sgl.function
        def constrained_json_qa(s, conversation):
            s += sgl.user(sgl.ChatCompletion(conversation))
            s += sgl.assistant(
                sgl.gen(
                    "json_output",
                    max_tokens=self.gen_kwargs["max_new_tokens"],
                    temperature=self.gen_kwargs["temperature"],
                    top_p=self.gen_kwargs["top_p"],
                    json_schema=self.schema_str,
                )
            )

        states = constrained_json_qa.run_batch(
            [{"conversation": p} for p in prompts], progress_bar=True
        )
        return [s["json_output"] for s in states]


# --- Transformers+Outlines Backend ---
class TransformersBackend(GenerationBackend):
    def __init__(self, model_path: str, schema: dict, gen_kwargs: dict):
        super().__init__(model_path, schema, gen_kwargs)
        try:
            from outlines.integrations import JSONLogitsProcessor
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Outlines is not installed. Please run 'pip install outlines'.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.logits_processor = JSONLogitsProcessor(schema, self.tokenizer)

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        outputs = []
        for conv in tqdm(prompts, desc="Generating with Transformers+Outlines"):
            prompt_str = self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                logits_processor=[self.logits_processor],
                max_new_tokens=self.gen_kwargs["max_new_tokens"],
                temperature=self.gen_kwargs["temperature"],
                top_p=self.gen_kwargs["top_p"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode only the newly generated tokens
            output_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs.append(generated_text)
        return outputs


def get_backend(
    backend_name: str, model_path: str, schema: dict, gen_kwargs: dict
) -> GenerationBackend:
    if backend_name == "sglang":
        return SGLangBackend(model_path, schema, gen_kwargs)
    elif backend_name == "transformers":
        return TransformersBackend(model_path, schema, gen_kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


# --- Helper Functions ---
def create_prompt(record: dict) -> list[dict]:
    """Creates a conversational prompt for a single record."""
    user_content = json.dumps(record["input"], indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def get_differing_fields(record1: dict, record2: dict) -> set[str]:
    """Finds keys where values differ between two records."""
    all_keys = set(record1.keys()) | set(record2.keys())
    differing = set()
    for key in all_keys:
        val1 = str(record1.get(key, "")).strip().lower()
        val2 = str(record2.get(key, "")).strip().lower()
        if val1 != val2:
            differing.add(key)
    return differing


def check_faithfulness(reasoning: str, differing_fields: set[str]) -> bool:
    """Heuristic: True if reasoning mentions a field that differs between records."""
    if not reasoning or not differing_fields:
        return False
    reasoning_lower = reasoning.lower()
    return any(field.lower() in reasoning_lower for field in differing_fields)


# --- Main Evaluation Logic ---
def main(args):
    logging.info(f"Loading JSON schema from {args.schema}")
    with open(args.schema, "r") as f:
        schema = json.load(f)

    logging.info(f"Loading test data from {args.data}")
    test_data = load_dataset("json", data_files=str(args.data), split="train")

    prompts = [create_prompt(record) for record in test_data]
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    backend = get_backend(args.backend, str(args.model_path), schema, gen_kwargs)

    logging.info(f"Generating predictions for {len(prompts)} test examples...")
    start_time = time.time()
    raw_outputs = backend.generate(prompts)
    total_time = time.time() - start_time
    logging.info(f"Generation finished in {total_time:.2f} seconds.")

    predictions, ground_truths, json_valid_count, faithful_count = [], [], 0, 0
    for i, raw_output in enumerate(raw_outputs):
        true_label = test_data[i]["output"]["result"]
        ground_truths.append(true_label)

        try:
            parsed_output = json.loads(raw_output)
            pred_label = parsed_output.get("result")
            reasoning = parsed_output.get("reasoning", "")

            if pred_label in LABELS:
                predictions.append(pred_label)
                json_valid_count += 1
            else:
                predictions.append(None)  # Invalid label

            differing_fields = get_differing_fields(
                test_data[i]["input"]["record1"], test_data[i]["input"]["record2"]
            )
            if check_faithfulness(reasoning, differing_fields):
                faithful_count += 1
        except (json.JSONDecodeError, AttributeError):
            predictions.append(None)
            logging.warning(f"Could not parse JSON output: {raw_output}")

    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    if not valid_indices:
        logging.error("No valid predictions generated. Cannot compute metrics.")
        return

    y_true = [ground_truths[i] for i in valid_indices]
    y_pred = [predictions[i] for i in valid_indices]

    logging.info("Calculating metrics...")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    results = {
        "model_path": str(args.model_path),
        "num_test_examples": len(test_data),
        "num_valid_predictions": len(y_true),
        "generation_time_seconds": round(total_time, 2),
        "kpis": {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1_score": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
            "per_class_f1_score": {label: score for label, score in zip(LABELS, f1)},
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(y_true, y_pred, labels=LABELS), index=LABELS, columns=LABELS
            ).to_dict(),
        },
        "structured_output_kpis": {
            "json_validity_percent": (json_valid_count / len(test_data)) * 100,
            "reasoning_faithfulness_percent": (faithful_count / len(test_data)) * 100,
        },
    }

    logging.info(f"--- Evaluation Results for {Path(args.model_path).name} ---")
    logging.info(f"Accuracy: {results['kpis']['accuracy']:.4f}")
    logging.info(f"Macro F1-Score: {results['kpis']['macro_f1_score']:.4f}")
    logging.info(f"JSON Validity: {results['structured_output_kpis']['json_validity_percent']:.2f}%")
    logging.info(f"Reasoning Faithfulness: {results['structured_output_kpis']['reasoning_faithfulness_percent']:.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on KPIs.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the merged model directory.")
    parser.add_argument("--backend", type=str, required=True, choices=["sglang", "transformers"], help="Generation backend.")
    parser.add_argument("--schema", type=Path, required=True, help="Path to the JSON schema for constrained decoding.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the test JSONL file.")
    parser.add_argument("--out", type=Path, required=True, help="Path to save the output metrics JSON file.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    main(parser.parse_args())
