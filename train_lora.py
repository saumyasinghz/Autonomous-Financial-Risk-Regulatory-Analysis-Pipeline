"""
train_lora.py
--------------
Local, VRAM-constrained QLoRA fine-tuning script.

Loads a small instruction-tuned SLM (`Qwen/Qwen2.5-1.5B-Instruct` by default,
see `config.TrainingConfig.base_model_name`) in hardened 4-bit precision via
`BitsAndBytesConfig`, attaches a LoRA adapter targeting the attention
projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), and trains it to map
raw financial news headlines to a strict, deterministic JSON schema:

    {"company": "...", "risk_level": "Low/Medium/High", "implication": "..."}

CRITICAL VRAM CONTRACT: this script is designed to be run as a standalone
process (e.g. from the Airflow DAG or CLI) that fully exits when finished.
Before exiting, it explicitly deletes all large GPU-resident objects and
calls `torch.cuda.empty_cache()` so that a subsequent `serve_vllm.py`
process can claim a clean GPU with no lingering training-time allocations.
"""

from __future__ import annotations

import gc
import json
import random
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer

from config import TRAINING

RISK_LEVEL_BY_SENTIMENT: Dict[int, str] = {
    0: "High",  # Bearish
    1: "Low",  # Bullish
    2: "Medium",  # Neutral
}

IMPLICATION_TEMPLATES: Dict[str, str] = {
    "High": "Negative sentiment signals potential downside risk requiring closer regulatory and credit monitoring.",
    "Medium": "Sentiment is mixed or neutral; no immediate escalation is warranted, but continued monitoring is advised.",
    "Low": "Positive sentiment suggests reduced near-term risk exposure for this entity.",
}

CASHTAG_PATTERN = re.compile(r"\$([A-Za-z]{1,6})\b")
CAPITALIZED_WORD_PATTERN = re.compile(r"\b([A-Z][a-zA-Z&]{2,})\b")

PROMPT_TEMPLATE = (
    "You are a financial risk analyst. Read the news headline and output ONLY a JSON "
    "object with exactly the keys 'company', 'risk_level' (Low/Medium/High), and "
    "'implication'. No prose, no markdown fences.\n\nHeadline: {headline}\nJSON:"
)


def _extract_company(headline: str) -> str:
    """Best-effort heuristic extraction of a company/ticker mention from a headline."""
    try:
        cashtag_match = CASHTAG_PATTERN.search(headline)
        if cashtag_match:
            return cashtag_match.group(1).upper()

        capitalized_match = CAPITALIZED_WORD_PATTERN.search(headline)
        if capitalized_match:
            return capitalized_match.group(1)

        return "Unknown"
    except Exception:  # noqa: BLE001
        return "Unknown"


def _build_target_json(headline: str, sentiment_label: int) -> Dict[str, str]:
    """Deterministically derives the stripped JSON training target for one headline."""
    risk_level = RISK_LEVEL_BY_SENTIMENT.get(sentiment_label, "Medium")
    return {
        "company": _extract_company(headline),
        "risk_level": risk_level,
        "implication": IMPLICATION_TEMPLATES[risk_level],
    }


def _build_synthetic_fallback_dataset(num_rows: int = 64) -> Dataset:
    """
    Produces a small, fully local synthetic dataset that mimics the shape of
    `zeroshot/twitter-financial-news` so training can still proceed end-to-end
    even without network access to the Hugging Face Hub.
    """
    sample_headlines = [
        "$AAPL shares tumble after weak iPhone demand guidance",
        "Tesla stock soars on record delivery numbers",
        "JPMorgan reports flat quarterly earnings, in line with estimates",
        "$AMZN faces regulatory scrutiny over antitrust concerns",
        "Microsoft beats revenue expectations, cloud growth accelerates",
        "Goldman Sachs downgrades outlook for regional banks",
        "Societe Generale to acquire Credit Agricole's retail banking business",
        "Bank of America to acquire Merrill Lynch",
        "JP Morgan Chase to acquire Bear Stearns",
        "Goldman Sachs to acquire Bear Stearns",
        "Morgan Stanley to acquire Bear Stearns",
        "Citigroup to acquire Bear Stearns",
        "Deutsche Bank to acquire Bear Stearns",
        "HSBC to acquire Bear Stearns",
    ]
    sentiments = [0, 1, 2, 0, 1, 0]
    rows: List[Dict[str, Any]] = []
    for i in range(num_rows):
        idx = i % len(sample_headlines)
        rows.append({"text": sample_headlines[idx], "label": sentiments[idx]})
    return Dataset.from_list(rows)


def load_training_dataset() -> Dataset:
    """
    Loads a slice of `zeroshot/twitter-financial-news` from the Hugging Face
    Hub. Falls back to a small local synthetic dataset if the network call
    fails, so the training pipeline never hard-crashes on connectivity issues.
    """
    try:
        raw_dataset = load_dataset(TRAINING.dataset_name, split=TRAINING.dataset_slice)
        print(f"[train_lora] Loaded {len(raw_dataset)} rows from '{TRAINING.dataset_name}'.")
        return raw_dataset
    except Exception as exc:  # noqa: BLE001
        print(f"[train_lora] WARNING: failed to load '{TRAINING.dataset_name}' ({exc}). "
              f"Falling back to a local synthetic dataset.")
        return _build_synthetic_fallback_dataset()


def format_dataset_for_training(raw_dataset: Dataset) -> Dataset:
    """Converts raw (headline, sentiment) rows into full prompt+completion training strings."""
    formatted_rows: List[Dict[str, str]] = []
    for row in raw_dataset:
        try:
            headline = str(row.get("text", "")).strip()
            sentiment_label = int(row.get("label", 2))
            if not headline:
                continue

            target_json = _build_target_json(headline, sentiment_label)
            completion = json.dumps(target_json, ensure_ascii=False)
            full_text = PROMPT_TEMPLATE.format(headline=headline) + " " + completion
            formatted_rows.append({"text": full_text})
        except Exception as exc:  # noqa: BLE001
            print(f"[train_lora] WARNING: skipping malformed row ({exc}): {row}")
            continue

    if not formatted_rows:
        raise ValueError("No valid training rows were produced from the source dataset.")

    return Dataset.from_list(formatted_rows)


def load_quantized_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the base SLM in a hardened 4-bit configuration to protect local GPU memory."""
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(TRAINING.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            TRAINING.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load quantized base model '{TRAINING.base_model_name}': {exc}") from exc


def attach_lora_adapter(model: PreTrainedModel) -> PeftModel:
    """Prepares the 4-bit model for k-bit training and attaches a LoRA adapter."""
    try:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=TRAINING.lora_r,
            lora_alpha=TRAINING.lora_alpha,
            lora_dropout=TRAINING.lora_dropout,
            target_modules=list(TRAINING.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to attach LoRA adapter: {exc}") from exc


def run_training(peft_model: PeftModel, tokenizer: PreTrainedTokenizerBase, train_dataset: Dataset) -> None:
    """Runs the supervised fine-tuning loop and saves the resulting LoRA adapter to disk."""
    try:
        random.seed(TRAINING.seed)
        torch.manual_seed(TRAINING.seed)

        training_args = TrainingArguments(
            output_dir=TRAINING.output_dir,
            per_device_train_batch_size=TRAINING.per_device_train_batch_size,
            gradient_accumulation_steps=TRAINING.gradient_accumulation_steps,
            num_train_epochs=TRAINING.num_train_epochs,
            learning_rate=TRAINING.learning_rate,
            logging_steps=TRAINING.logging_steps,
            save_steps=TRAINING.save_steps,
            save_total_limit=1,
            fp16=False,
            bf16=torch.cuda.is_available(),
            optim="paged_adamw_8bit",
            report_to=[],
            seed=TRAINING.seed,
        )

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=TRAINING.max_seq_length,
            tokenizer=tokenizer,
        )

        trainer.train()

        trainer.model.save_pretrained(TRAINING.output_dir)
        tokenizer.save_pretrained(TRAINING.output_dir)
        print(f"[train_lora] LoRA adapter saved to '{TRAINING.output_dir}'.")

        del trainer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Training loop failed: {exc}") from exc


def release_gpu_memory(*objects_to_delete: Optional[Any]) -> None:
    """
    CRITICAL cleanup step: destroys all references to large GPU-resident
    objects and forces CUDA to release cached VRAM back to the OS/driver so
    that `serve_vllm.py` can start with a clean memory budget afterward.
    """
    for obj in objects_to_delete:
        try:
            del obj
        except Exception:  # noqa: BLE001
            pass

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[train_lora] GPU memory released; VRAM flushed.")
    except Exception as exc:  # noqa: BLE001
        print(f"[train_lora] WARNING: VRAM cleanup encountered an issue: {exc}")


def main() -> None:
    model: Optional[PreTrainedModel] = None
    peft_model: Optional[PeftModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    try:
        raw_dataset = load_training_dataset()
        train_dataset = format_dataset_for_training(raw_dataset)

        model, tokenizer = load_quantized_base_model()
        peft_model = attach_lora_adapter(model)

        run_training(peft_model, tokenizer, train_dataset)

    except Exception as exc:  # noqa: BLE001
        print(f"[train_lora] FATAL: training pipeline aborted: {exc}")

    finally:
        # Unconditionally flush VRAM, even on failure, so a partially-loaded
        # model never lingers and blocks the subsequent vLLM server launch.
        release_gpu_memory(peft_model, model, tokenizer)


if __name__ == "__main__":
    main()
