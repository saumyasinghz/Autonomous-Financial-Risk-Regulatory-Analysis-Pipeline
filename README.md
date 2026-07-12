# Autonomous Financial Risk & Regulatory Analysis Pipeline

A fully local, single-GPU internal engineering utility that ingests financial
news, retrieves regulatory context, and uses a QLoRA-tuned Small Language
Model (SLM) — served locally via vLLM — to produce structured risk
assessments, all coordinated by a LangGraph state machine and scheduled by
Apache Airflow.

## Architecture

```
Airflow (@daily)
  └─ ingest_market_stream        -> pulls a fresh slice of zeroshot/twitter-financial-news
  └─ execute_agentic_workflow    -> runs agent_graph.py over the ingested headlines

agent_graph.py (LangGraph StateGraph)
  └─ Context Enrichment Node     -> vector_store.py -> Pinecone (regulatory rules)
  └─ vLLM Extractor Node         -> HTTP -> serve_vllm.py (localhost:8000)
  └─ Verification Guardrail Node -> validates JSON schema, retries on failure

train_lora.py   -> offline QLoRA fine-tuning of the base SLM, VRAM is fully
                    released before serve_vllm.py is ever started
serve_vllm.py   -> loads the base SLM + LoRA adapter into vLLM, exposed via
                    a local FastAPI layer at http://localhost:8000
```

## Local hardware & stack matrix

| Layer            | Choice                                                        |
|------------------|----------------------------------------------------------------|
| Orchestration    | Apache Airflow (standalone, local)                             |
| Vector store     | Pinecone (`pinecone-client`)                                    |
| Dataset          | Hugging Face `datasets` — `zeroshot/twitter-financial-news`     |
| Base SLM         | `Qwen/Qwen2.5-1.5B-Instruct` (or `microsoft/Phi-3-mini-4k-instruct`) |
| Fine-tuning      | PEFT LoRA + `bitsandbytes` 4-bit quantization (QLoRA)            |
| Inference server | vLLM (`max-model-len=2048`, `gpu-memory-utilization=0.7`)        |
| Coordination     | LangGraph `StateGraph`                                           |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in PINECONE_API_KEY, HF_TOKEN, etc.
```

## Run order (local)

1. **Populate the vector store** (one-time / periodic):
   ```bash
   python vector_store.py
   ```
2. **Fine-tune the LoRA adapter** (fully releases VRAM on exit):
   ```bash
   python train_lora.py
   ```
3. **Start the local vLLM + FastAPI inference server**:
   ```bash
   python serve_vllm.py
   ```
4. **Run the agentic workflow directly** (ad-hoc, without Airflow):
   ```bash
   python agent_graph.py
   ```
5. **Or schedule everything through Airflow**:
   ```bash
   export AIRFLOW_HOME="$(pwd)/airflow_home"
   airflow standalone
   # symlink/copy airflow_dag.py into $AIRFLOW_HOME/dags, then trigger
   # `local_financial_audit_pipeline` from the Airflow UI or CLI.
   ```

## Files

- `config.py` — centralized, typed configuration shared by every module.
- `vector_store.py` — Pinecone wrapper with local CPU embeddings.
- `train_lora.py` — 4-bit QLoRA fine-tuning script with guaranteed VRAM cleanup.
- `serve_vllm.py` — vLLM + LoRA inference engine exposed via FastAPI.
- `agent_graph.py` — LangGraph `StateGraph` coordinating retrieval, extraction, and validation.
- `airflow_dag.py` — Airflow DAG (`local_financial_audit_pipeline`) tying ingestion and the agent graph together.

## Safety notes

- `serve_vllm.py` should only be started **after** `train_lora.py` has fully
  exited, since the training script explicitly destroys its model objects
  and empties the CUDA cache before returning, preventing VRAM contention
  between training and inference on a single consumer GPU.
- `max_model_len=2048` and `gpu_memory_utilization=0.7` are enforced in
  `config.py` to leave headroom for the OS display/audio stack.
