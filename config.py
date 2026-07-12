"""
config.py
---------
Centralized, typed configuration for the Autonomous Financial Risk &
Regulatory Analysis Pipeline. Every module (`airflow_dag.py`,
`vector_store.py`, `train_lora.py`, `serve_vllm.py`, `agent_graph.py`)
imports its constants from here so the local hardware / stack matrix
stays consistent across the whole workspace.

All values can be overridden via environment variables (see `.env.example`)
without touching code, which keeps the pipeline portable across local
machines with different GPU budgets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load `.env` once, at import time, for every downstream module.
load_dotenv()


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    environment: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1"))
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "financial-compliance-rules"))
    embedding_dim: int = 384  # sentence-transformers/all-MiniLM-L6-v2 output size
    metric: str = "cosine"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # kept off the GPU on purpose to preserve VRAM for the SLM


@dataclass(frozen=True)
class TrainingConfig:
    base_model_name: str = os.getenv("VLLM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir: str = "./artifacts/lora_adapter"
    dataset_name: str = "zeroshot/twitter-financial-news"
    dataset_split: str = "train"
    dataset_slice: str = "train[:512]"
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42


@dataclass(frozen=True)
class VLLMConfig:
    host: str = os.getenv("VLLM_HOST", "localhost")
    port: int = int(os.getenv("VLLM_PORT", "8000"))
    base_model: str = os.getenv("VLLM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    lora_adapter_path: str = os.getenv("VLLM_LORA_ADAPTER_PATH", "./artifacts/lora_adapter")
    lora_adapter_name: str = os.getenv("VLLM_LORA_ADAPTER_NAME", "financial-risk-lora")
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.7
    dtype: str = "bfloat16"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def generate_url(self) -> str:
        """Endpoint exposed by the custom FastAPI layer in `serve_vllm.py`."""
        return f"{self.base_url}/generate"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


@dataclass(frozen=True)
class AgentConfig:
    max_correction_retries: int = 2
    required_json_keys: tuple = ("company", "risk_level", "implication")
    valid_risk_levels: tuple = ("Low", "Medium", "High")
    request_timeout_seconds: float = 30.0


PINECONE = PineconeConfig()
EMBEDDING = EmbeddingConfig()
TRAINING = TrainingConfig()
VLLM = VLLMConfig()
AGENT = AgentConfig()
