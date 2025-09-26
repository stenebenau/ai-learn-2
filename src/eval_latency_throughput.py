# src/eval_latency_throughput.py

import argparse
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Coroutine, Dict, List

import aiohttp
import numpy as np
import pandas as pd
from datasets import load_dataset
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


# --- Helper Functions ---
def create_prompt(record: dict) -> list[dict]:
    """Creates a conversational prompt for a single record."""
    user_content = json.dumps(record["input"], indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --- Backend Abstraction ---
class BenchmarkBackend(ABC):
    """Abstract base class for benchmarking backends."""

    def __init__(self, model_id: str, schema: dict, gen_kwargs: dict):
        self.model_id = model_id
        self.schema = schema
        self.gen_kwargs = gen_kwargs
        logging.info(
            f"Initialized {self.__class__.__name__} for model/endpoint: {model_id}"
        )

    @abstractmethod
    async def benchmark(self, prompts: list[list[dict]], concurrency: int) -> dict:
        """Runs a benchmark for a given concurrency level and returns metrics."""
        pass


# --- vLLM OpenAI-Compatible Server Backend ---
class VLLMBenchmarkBackend(BenchmarkBackend):
    """Client for a vLLM OpenAI-compatible API server."""

    def __init__(self, endpoint: str, schema: dict, gen_kwargs: dict):
        super().__init__(model_id=endpoint, schema=schema, gen_kwargs=gen_kwargs)
        self.endpoint = f"{endpoint}/chat/completions"

    async def _make_request(
        self, session: aiohttp.ClientSession, prompt: list[dict]
    ) -> tuple[float, int]:
        """Makes a single async request and returns (latency, num_output_tokens)."""
        payload = {
            "model": "vllm-model",  # model name is ignored by vLLM server but required by API
            "messages": prompt,
            "temperature": self.gen_kwargs["temperature"],
            "top_p": self.gen_kwargs["top_p"],
            "max_tokens": self.gen_kwargs["max_new_tokens"],
            "extra_body": {"json_schema": json.dumps(self.schema)},
        }
        start_time = time.monotonic()
        async with session.post(self.endpoint, json=payload) as response:
            if response.status != 200:
                logging.error(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
                return -1.0, 0

            resp_json = await response.json()
            latency = time.monotonic() - start_time
            num_output_tokens = resp_json["usage"]["completion_tokens"]
            return latency, num_output_tokens

    async def benchmark(self, prompts: list[list[dict]], concurrency: int) -> dict:
        logging.info(
            f"Starting benchmark for concurrency={concurrency} with {len(prompts)} prompts..."
        )
        async with aiohttp.ClientSession() as session:
            tasks: list[Coroutine[Any, Any, tuple[float, int]]] = [
                self._make_request(session, p) for p in prompts
            ]

            latencies = []
            total_output_tokens = 0
            benchmark_start_time = time.monotonic()

            for i in tqdm(
                range(0, len(tasks), concurrency), desc=f"Concurrency {concurrency}"
            ):
                batch_tasks = tasks[i : i + concurrency]
                results = await asyncio.gather(*batch_tasks)

                for lat, tokens in results:
                    if lat > 0:
                        latencies.append(lat)
                        total_output_tokens += tokens

            benchmark_total_time = time.monotonic() - benchmark_start_time

            if not latencies:
                return {"error": "All requests failed."}

            return {
                "concurrency": concurrency,
                "p50_latency_ms": np.percentile(latencies, 50) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000,
                "avg_latency_ms": np.mean(latencies) * 1000,
                "requests_per_sec": len(latencies) / benchmark_total_time,
                "output_tokens_per_sec": total_output_tokens / benchmark_total_time,
                "num_successful_requests": len(latencies),
            }


# --- Transformers+Outlines Backend (Single-process baseline) ---
class TransformersBenchmarkBackend(BenchmarkBackend):
    """Single-process benchmark using Transformers and Outlines for baseline latency."""

    def __init__(self, model_path: str, schema: dict, gen_kwargs: dict):
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model_id=model_path, schema=schema, gen_kwargs=gen_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.generator = outlines.generate.json(
            model,
            schema,
            max_tokens=gen_kwargs["max_new_tokens"],
            tokenizer=self.tokenizer,
        )

    async def benchmark(self, prompts: list[list[dict]], concurrency: int) -> dict:
        if concurrency != 1:
            logging.warning(
                f"Transformers backend runs in single-process mode. Concurrency is forced to 1 (was {concurrency})."
            )

        logging.info(
            f"Starting benchmark for concurrency=1 with {len(prompts)} prompts..."
        )
        latencies = []
        total_output_tokens = 0
        benchmark_start_time = time.monotonic()

        for prompt in tqdm(prompts, desc="Concurrency 1"):
            prompt_str = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

            start_time = time.monotonic()
            generated_text = self.generator(prompt_str)
            latency = time.monotonic() - start_time

            output_ids = self.tokenizer(generated_text, return_tensors="pt").input_ids
            num_output_tokens = output_ids.shape[1]

            latencies.append(latency)
            total_output_tokens += num_output_tokens

        benchmark_total_time = time.monotonic() - benchmark_start_time

        return {
            "concurrency": 1,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "avg_latency_ms": np.mean(latencies) * 1000,
            "requests_per_sec": len(latencies) / benchmark_total_time,
            "output_tokens_per_sec": total_output_tokens / benchmark_total_time,
            "num_successful_requests": len(latencies),
        }


def get_backend(
    backend_name: str, model_path: str, endpoint: str, schema: dict, gen_kwargs: dict
) -> BenchmarkBackend:
    if backend_name == "vllm":
        return VLLMBenchmarkBackend(endpoint, schema, gen_kwargs)
    elif backend_name == "transformers":
        return TransformersBenchmarkBackend(model_path, schema, gen_kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


async def main(args):
    logging.info(f"Loading JSON schema from {args.schema}")
    with open(args.schema, "r") as f:
        schema = json.load(f)

    logging.info(f"Loading test data from {args.data}")
    test_data = load_dataset(
        "json", data_files=str(args.data), split=f"train[:{args.num_prompts}]"
    )
    prompts = [create_prompt(record) for record in test_data]

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    backend = get_backend(
        args.backend, str(args.model_path), args.endpoint, schema, gen_kwargs
    )

    all_results = []
    concurrency_levels = sorted(list(set(args.concurrency)))
    for c in concurrency_levels:
        if args.backend == "transformers" and c > 1:
            if c == concurrency_levels[-1] and 1 not in concurrency_levels:
                 logging.info("Forcing concurrency to 1 for 'transformers' backend.")
                 c = 1
            else:
                logging.info(f"Skipping concurrency > 1 for 'transformers' backend.")
                continue

        results = await backend.benchmark(prompts, c)
        if "error" in results:
            logging.error(f"Benchmark failed for concurrency {c}: {results['error']}")
        else:
            logging.info(f"Results for concurrency={c}: {results}")
            all_results.append(results)
        
        if args.backend == "transformers":
            break


    if not all_results:
        logging.error("No benchmark results to save.")
        return

    df = pd.DataFrame(all_results)
    df["model_id"] = Path(args.model_path).name if args.model_path else args.endpoint
    df["backend"] = args.backend

    cols = [
        "model_id", "backend", "concurrency", "requests_per_sec",
        "output_tokens_per_sec", "p50_latency_ms", "p95_latency_ms",
        "avg_latency_ms", "num_successful_requests",
    ]
    df = df[[col for col in cols if col in df.columns]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logging.info(f"Benchmark results saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark model latency and throughput."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to the merged model directory (for 'transformers' backend).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="Endpoint for the API server (for 'vllm' backend).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["vllm", "transformers"],
        help="Benchmarking backend.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        required=True,
        help="Path to the JSON schema for constrained decoding.",
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to the test JSONL file."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=200,
        help="Number of prompts to use from the test set.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="List of concurrency levels to test.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()
    asyncio.run(main(args))
