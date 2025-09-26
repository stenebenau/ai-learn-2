# CRM Deduplication LLM — Training, Validation & Benchmark Plan

> **Purpose:**
> Build and compare small instruction-tuned LLMs that act like a **human data steward**: read **two CRM JSON records** and return **strict JSON** with a **Duplicate / Possible / No Match** decision and a short **human-readable reasoning**. The system must support CUDA GPUs for training and report **standard classification KPIs** plus **latency & throughput** under constrained JSON decoding.

---

## 0) Scope & Success Criteria

- **Task:** Supervised fine-tuning (SFT) of small LLMs on your JSONL dataset:

  ```json
  {"input":{"record1":{...},"record2":{...}},
   "output":{"result":"DUPLICATE|POSSIBLE|NO_MATCH","reasoning":"short human sentence"}}
  ```

- **Models to compare** (base checkpoints):

  - `microsoft/Phi-4-mini-instruct` (≈3.8B)
  - `Qwen/Qwen3-4B-Instruct-2507` (≈4B)
  - `meta-llama/Llama-3.2-3B-Instruct` (≈3B)
  - `mistralai/Mistral-7B-Instruct-v0.3` (≈7B)

- **Must-haves:**

  - SFT with **LoRA/QLoRA** (memory-efficient)
  - **Constrained JSON decoding** at validation/benchmark time to guarantee strict JSON output
  - KPIs: **Accuracy, Precision/Recall, Macro-F1, per-class F1, Confusion Matrix**
  - Structured-output KPIs: **JSON validity %**, **Reasoning faithfulness**
  - **Latency** (p50/p95) and **Throughput** (req/s, tok/s) across backends (HF vs vLLM)

---

## 1) Repo Layout

```

  configs/
    phi4mini.yaml
    qwen3_4b.yaml
    llama3_2_3b.yaml
    mistral7b.yaml
    decoding.jsonschema
  data/
    train.jsonl
    val.jsonl
    test.jsonl
  src/
    prepare_splits.py
    build_prompts.py
    dataset.py
    sft_train.py
    evaluate_kpis.py
    eval_latency_throughput.py
    constrained_decode/
      tgi_guidance.py        # OR sglang_backend.py OR llama_cpp_grammar.py
  reports/
    metrics.csv
    latency.csv
    throughput.csv
    model_comparison.md
  env.yml
  Makefile
  README.md
```

---

## 2) Environment (CUDA GPU)

- Python 3.12
- Install (example):

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install "transformers>=4.43" datasets accelerate peft bitsandbytes
  pip install evaluate scikit-learn
  pip install vllm
  # choose ONE structured-output backend, or wire more than one:
  pip install outlines  # for JSON schema constraints via TGI/Transformers
  # OR
  pip install "sglang>=0.3.0"
  # (optional) text-generation-inference if you prefer TGI serving
  ```

- Notes:

  - For Llama 3.2, ensure model access per license requirements.
  - Use BF16 where possible; fall back to FP16 if needed.

---

## 3) Data & Prompt/Target Packing

**Input JSONL (unchanged):**

```json
{"input":{"record1":{...},"record2":{...}},
 "output":{"result":"DUPLICATE","reasoning":"..."}}
```

**Split:**

- `src/prepare_splits.py` → stratified 80/10/10 (by `output.result`) into `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

**SFT conversation format:**

- **System:**
  “You are a CRM data steward. Return **strict JSON** with keys: `result` (one of `DUPLICATE`, `POSSIBLE`, `NO_MATCH`) and `reasoning` (one short sentence). No extra text.”
- **User:**
  A JSON block with `record1` and `record2` exactly as in the dataset.
- **Assistant (target):**
  Exactly the gold JSON object:

  ```json
  { "result": "DUPLICATE", "reasoning": "..." }
  ```

**Max sequence length:** start at 4k tokens (raise if needed for very large inputs).

---

## 4) Constrained JSON Decoding

Guarantee strict JSON output at eval/serve time using **one** of:

- **Outlines (Transformers/TGI)** with a JSON Schema
- **SGLang** with built-in JSON schema constraints
- **llama.cpp** with a GBNF grammar (if you later export to GGUF)

**Schema (`configs/decoding.jsonschema`):**

```json
{
  "type": "object",
  "properties": {
    "result": { "enum": ["DUPLICATE", "POSSIBLE", "NO_MATCH"] },
    "reasoning": { "type": "string", "minLength": 1, "maxLength": 320 }
  },
  "required": ["result", "reasoning"],
  "additionalProperties": false
}
```

**Generation defaults (uniform across models):**

- `max_new_tokens=64`, `temperature=0.1`, `top_p=0.9`

---

## 5) Training (SFT with LoRA/QLoRA)

**PEFT/QLoRA defaults (tune per model via YAML):**

- `lora_r=64`, `lora_alpha=16`, `lora_dropout=0.05`
- **Target modules:**
  Llama/Qwen/Mistral → `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
  Phi → include linear layers covering attention/MLP (or fallback to all Linears)
- Optim: `lr=2e-5`, `weight_decay=0.0`, `warmup_ratio=0.03`
- Schedule: 2–3 epochs
- Batch: `per_device_train_batch_size=4-8`, `gradient_accumulation_steps` to reach effective 64
- Precision: `bf16=True` (if supported), gradient checkpointing
- Save LoRA adapters + **merged FP16** checkpoint for inference/serving

**Example config (`configs/phi4mini.yaml`):**

```yaml
model_id: microsoft/Phi-4-mini-instruct
peft:
  method: qlora
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules:
    [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
train:
  lr: 2e-5
  epochs: 3
  batch_size: 4
  grad_accum: 16
  max_seq_len: 4096
  bf16: true
  gradient_ckpt: true
gen:
  max_new_tokens: 64
  temperature: 0.1
  top_p: 0.9
```

**Run (per model):**

```bash
accelerate launch src/sft_train.py --config configs/phi4mini.yaml
accelerate launch src/sft_train.py --config configs/qwen3_4b.yaml
accelerate launch src/sft_train.py --config configs/llama3_2_3b.yaml
accelerate launch src/sft_train.py --config configs/mistral7b.yaml
```

---

## 6) Evaluation — Classification & Structured Output

**Standard KPIs:**

- Accuracy
- Precision / Recall / **Macro-F1**
- Per-class F1 for `DUPLICATE`, `POSSIBLE`, `NO_MATCH`
- Confusion matrix

**Structured-output KPIs:**

- **JSON validity %** (should be ~100% with constraints)
- **Reasoning faithfulness** (heuristics):

  - Extract field names actually present/different in input records
  - Score whether `reasoning` mentions appropriate fields (precision/recall)
  - Penalize hallucinated fields

**Command (example):**

```bash
python src/evaluate_kpis.py \
  --model_path outputs/phi4mini-lora-merged \
  --backend sglang \
  --schema configs/decoding.jsonschema \
  --data data/test.jsonl \
  --out reports/phi4mini_metrics.json
```

---

## 7) Latency & Throughput Benchmarks

Benchmark **two backends** for each model:

1. **Transformers (single-process)** → baseline latency
2. **vLLM** → high-throughput serving (continuous batching, PagedAttention)

**Uniform decoding params:** use the same as above and enforce the **same JSON constraints**.

**Measure:**

- **Latency:** p50/p95 end-to-end per request (tokenization + constraint engine included)
- **Throughput:** requests/sec at concurrency `{1, 4, 8, 16}`, plus tokens/sec

**Commands (illustrative):**

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model outputs/qwen3_4b-lora-merged --dtype bfloat16 --max-num-seqs 1024

# Run client
python src/eval_latency_throughput.py \
  --endpoint http://localhost:8000/v1 \
  --data data/test.jsonl \
  --schema configs/decoding.jsonschema \
  --concurrency 1 4 8 16 \
  --out reports/qwen3_4b_vllm_latency.csv
```

---

## 8) Comparison Report

Create `reports/model_comparison.md` with a single table:

| Model                    | Checkpoint                         | Acc | Macro-F1 | F1 Dup | F1 Possible | F1 NoMatch | JSON Valid % | Faithful % | p50 (ms) | p95 (ms) | Req/s @8 | Tok/s |
| ------------------------ | ---------------------------------- | --: | -------: | -----: | ----------: | ---------: | -----------: | ---------: | -------: | -------: | -------: | ----- |
| Phi-4-Mini-Instruct      | microsoft/Phi-4-mini-instruct      |   … |        … |      … |           … |          … |            … |          … |        … |        … |        … |
| Qwen3-4B-Instruct        | Qwen/Qwen3-4B-Instruct-2507        |   … |        … |      … |           … |          … |            … |          … |        … |        … |        … | …     |
| Llama-3.2-3B-Instruct    | meta-llama/Llama-3.2-3B-Instruct   |   … |        … |      … |           … |          … |            … |          … |        … |        … |        … |
| Mistral-7B-Instruct-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 |   … |        … |      … |           … |          … |            … |          … |        … |        … |        … |

Include notes on:

- VRAM used during training/inference
- Any quantization used for inference (e.g., bitsandbytes INT8) and its observed impact

---

## 9) Serving Demo (JSON-only REST)

Expose a `/match` endpoint that accepts:

```json
{"record1": {...}, "record2": {...}}
```

and returns strictly:

```json
{ "result": "DUPLICATE|POSSIBLE|NO_MATCH", "reasoning": "..." }
```

Backends:

- **SGLang server** with JSON Schema constraints **or**
- **TGI** with Outlines/Guidance JSON constraints **or**
- **Transformers** + Outlines locally (smaller scale)

---

## 10) Makefile Targets (quality-of-life)

```makefile
setup:
	conda env create -f env.yml || true

prep:
	python src/prepare_splits.py --in data/all.jsonl --out data

train-%:
	accelerate launch src/sft_train.py --config configs/$*.yaml

kpi-%:
	python src/evaluate_kpis.py \
	  --model_path outputs/$*-merged \
	  --backend sglang \
	  --schema configs/decoding.jsonschema \
	  --data data/test.jsonl \
	  --out reports/$*_metrics.json

bench-%:
	python src/eval_latency_throughput.py \
	  --model_path outputs/$*-merged \
	  --backend vllm \
	  --data data/test.jsonl \
	  --schema configs/decoding.jsonschema \
	  --concurrency 1 4 8 16 \
	  --out reports/$*_latency.csv
```

---

## 11) Model-Specific Notes

- **Phi-4-Mini-Instruct** — strong at instruction following; good explanation quality for its size.
- **Qwen3-4B-Instruct-2507** — competitive reasoning at 4B; very long context; active tooling.
- **Llama-3.2-3B-Instruct** — excellent ecosystem/tooling; ensure license access is configured.
- **Mistral-7B-Instruct-v0.3** — best quality in this set but larger; still feasible on a single modern CUDA GPU for SFT; include for quality ceiling.

---

## 12) Stretch (Optional)

- **DPO / Preference tuning:** small curated set of “preferred explanations” to sharpen style post-SFT.
- **Error taxonomy loop:** auto-tag common failure modes; add counter-examples; re-SFT.

---

### Notes for the Coding Agent

- Keep **prompts minimal and consistent**; all outputs must pass **JSON schema validation**.
- Use the **same generation params** across models when comparing.
- Ensure **tokenization and constraint overhead** are included in latency measurements.
- For fairness, evaluate both plain Transformers and a **batched server** (vLLM) under the **same constraints** and **same test set**.
- Limit `max_new_tokens` to **≤64** and keep `temperature` low to reduce latency and verbosity.
