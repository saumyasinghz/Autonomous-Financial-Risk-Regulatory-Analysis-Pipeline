"""
serve_vllm.py
--------------
Local vLLM inference server exposed via a lightweight FastAPI runtime layer.

Loads the base SLM (`config.VLLM.base_model`) together with the LoRA adapter
produced by `train_lora.py`, using vLLM's native local adapter loading
support (the programmatic equivalent of the CLI flags `--enable-lora` and
`--lora-modules`). Safety flags are hard-restricted for local consumer GPU
hardware: `max_model_len=2048` and `gpu_memory_utilization=0.7`.

Run directly:
    python serve_vllm.py

Equivalent pure-CLI launch (kept here for reference/parity with the vLLM
OpenAI-compatible server, in case a raw CLI deployment is preferred over the
custom FastAPI layer below):

    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --enable-lora \\
        --lora-modules financial-risk-lora=./artifacts/lora_adapter \\
        --max-model-len 2048 \\
        --gpu-memory-utilization 0.7 \\
        --host localhost --port 8000

`agent_graph.py` talks to this server over plain HTTP at
`http://localhost:8000/generate`.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import VLLM

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest

    _VLLM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # noqa: BLE001 - vLLM may be unavailable on non-CUDA local hosts
    SamplingParams = None  # type: ignore[assignment]
    AsyncEngineArgs = None  # type: ignore[assignment]
    AsyncLLMEngine = None  # type: ignore[assignment]
    LoRARequest = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Fully-formatted prompt text to complete.")
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    use_lora_adapter: bool = Field(
        default=True, description="Whether to route generation through the trained LoRA adapter."
    )


class GenerationResponse(BaseModel):
    request_id: str
    text: str
    finished: bool


class EngineState:
    """Holds the single lazily-initialized AsyncLLMEngine instance for this process."""

    engine: Optional["AsyncLLMEngine"] = None
    lora_request: Optional["LoRARequest"] = None
    init_error: Optional[str] = None


engine_state = EngineState()


def _build_lora_request() -> Optional["LoRARequest"]:
    """Builds a vLLM LoRARequest pointing at the local adapter saved by train_lora.py."""
    try:
        if not os.path.isdir(VLLM.lora_adapter_path):
            print(
                f"[serve_vllm] WARNING: LoRA adapter path '{VLLM.lora_adapter_path}' does not "
                f"exist yet. Serving with the base model only until training completes."
            )
            return None
        return LoRARequest(
            lora_name=VLLM.lora_adapter_name,
            lora_int_id=1,
            lora_local_path=VLLM.lora_adapter_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[serve_vllm] WARNING: failed to build LoRARequest: {exc}")
        return None


def _initialize_engine() -> None:
    """Initializes the local vLLM async engine with hardened local-GPU safety flags."""
    if _VLLM_IMPORT_ERROR is not None:
        engine_state.init_error = f"vLLM is not importable in this environment: {_VLLM_IMPORT_ERROR}"
        print(f"[serve_vllm] ERROR: {engine_state.init_error}")
        return

    try:
        engine_args = AsyncEngineArgs(
            model=VLLM.base_model,
            enable_lora=True,
            max_lora_rank=64,
            max_model_len=VLLM.max_model_len,
            gpu_memory_utilization=VLLM.gpu_memory_utilization,
            dtype=VLLM.dtype,
        )
        engine_state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        engine_state.lora_request = _build_lora_request()
        print(
            f"[serve_vllm] vLLM engine ready. model={VLLM.base_model} "
            f"max_model_len={VLLM.max_model_len} gpu_memory_utilization={VLLM.gpu_memory_utilization}"
        )
    except Exception as exc:  # noqa: BLE001
        engine_state.init_error = f"Failed to initialize vLLM engine: {exc}"
        print(f"[serve_vllm] ERROR: {engine_state.init_error}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _initialize_engine()
    yield
    # No explicit teardown required: process exit releases vLLM's CUDA context.


app = FastAPI(
    title="Local Financial Risk SLM Inference Server",
    description="FastAPI wrapper around a local vLLM engine serving a LoRA-tuned SLM.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Simple liveness/readiness probe for the local server."""
    try:
        return {
            "status": "ok" if engine_state.engine is not None else "degraded",
            "engine_ready": engine_state.engine is not None,
            "lora_loaded": engine_state.lora_request is not None,
            "init_error": engine_state.init_error,
            "base_model": VLLM.base_model,
            "max_model_len": VLLM.max_model_len,
            "gpu_memory_utilization": VLLM.gpu_memory_utilization,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Health check failed: {exc}") from exc


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest) -> GenerationResponse:
    """
    Runs a single generation request against the local vLLM engine, routing
    through the LoRA adapter unless explicitly disabled or unavailable.
    """
    if engine_state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM engine is not available: {engine_state.init_error or 'unknown initialization error'}",
        )

    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        request_id = str(uuid.uuid4())
        lora_request = engine_state.lora_request if request.use_lora_adapter else None

        final_output_text = ""
        finished = False

        results_generator = engine_state.engine.generate(
            request.prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        async for request_output in results_generator:
            if request_output.outputs:
                final_output_text = request_output.outputs[0].text
            finished = request_output.finished

        return GenerationResponse(request_id=request_id, text=final_output_text, finished=finished)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


def main() -> None:
    """Launches the FastAPI + vLLM server on localhost for local LangGraph consumption."""
    try:
        uvicorn.run(app, host=VLLM.host, port=VLLM.port, log_level="info")
    except Exception as exc:  # noqa: BLE001
        print(f"[serve_vllm] FATAL: server failed to start: {exc}")


if __name__ == "__main__":
    main()
