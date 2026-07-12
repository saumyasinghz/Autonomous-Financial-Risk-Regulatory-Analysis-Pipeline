"""
airflow_dag.py
--------------
Apache Airflow DAG: `local_financial_audit_pipeline`.

Runs `@daily` on a local Airflow standalone instance and performs two tasks:

1. `ingest_market_stream` — pulls a fresh slice of `zeroshot/twitter-financial-news`
   from the Hugging Face Hub to simulate an incoming streaming data delta,
   pushing the resulting headlines to XCom.
2. `execute_agentic_workflow` — pulls the ingested headlines from XCom and
   invokes the compiled LangGraph state machine (`agent_graph.py`) over them,
   logging a structured summary of the resulting risk assessments.

This file is meant to be placed in (or symlinked into) your local
`$AIRFLOW_HOME/dags` directory.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_ID = "local_financial_audit_pipeline"
DEFAULT_INGEST_SLICE_SIZE = 16

default_args: Dict[str, Any] = {
    "owner": "risk-engineering",
    "retries": 2,
    "retry_delay": dt.timedelta(minutes=5),
    "depends_on_past": False,
}


def _build_synthetic_stream_delta(slice_size: int) -> List[str]:
    """
    Local, network-free fallback used when the Hugging Face Hub is
    unreachable, so the DAG still produces a usable ingestion delta instead
    of failing the whole pipeline run.
    """
    sample_headlines = [
        "$AAPL shares tumble after weak iPhone demand guidance",
        "Tesla stock soars on record delivery numbers",
        "JPMorgan reports flat quarterly earnings, in line with estimates",
        "$AMZN faces regulatory scrutiny over antitrust concerns",
        "Microsoft beats revenue expectations, cloud growth accelerates",
        "Goldman Sachs downgrades outlook for regional banks",
        "Regional bank stress test results raise capital adequacy concerns",
        "$NFLX subscriber growth slows amid password-sharing crackdown",
    ]
    return random.choices(sample_headlines, k=slice_size)


def ingest_market_stream(**context: Any) -> List[str]:
    """
    Loads a slice of `zeroshot/twitter-financial-news` via Hugging Face
    `datasets` to simulate a fresh incoming market-news delta, then pushes
    the resulting headline list to XCom for the downstream task.
    """
    headlines: List[str] = []
    try:
        from datasets import load_dataset

        try:
            execution_date = context.get("logical_date") or context.get("execution_date")
            day_seed = execution_date.toordinal() if execution_date else 0
            random.seed(day_seed)

            slice_start = (day_seed % 500) if day_seed else 0
            slice_end = slice_start + DEFAULT_INGEST_SLICE_SIZE
            dataset_slice = f"train[{slice_start}:{slice_end}]"

            raw_dataset = load_dataset("zeroshot/twitter-financial-news", split=dataset_slice)
            headlines = [str(row.get("text", "")).strip() for row in raw_dataset if row.get("text")]
            print(
                f"[ingest_market_stream] Pulled {len(headlines)} headlines "
                f"(slice='{dataset_slice}') from zeroshot/twitter-financial-news."
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ingest_market_stream] WARNING: Hugging Face Hub load failed ({exc}); "
                f"falling back to a local synthetic delta."
            )
            headlines = _build_synthetic_stream_delta(DEFAULT_INGEST_SLICE_SIZE)

    except Exception as exc:  # noqa: BLE001
        print(f"[ingest_market_stream] ERROR: unexpected ingestion failure: {exc}")
        headlines = _build_synthetic_stream_delta(DEFAULT_INGEST_SLICE_SIZE)

    if not headlines:
        headlines = _build_synthetic_stream_delta(DEFAULT_INGEST_SLICE_SIZE)

    task_instance = context.get("ti") or context.get("task_instance")
    if task_instance is not None:
        try:
            task_instance.xcom_push(key="ingested_headlines", value=headlines)
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest_market_stream] WARNING: failed to push XCom: {exc}")

    return headlines


def execute_agentic_workflow(**context: Any) -> List[Dict[str, Any]]:
    """
    Pulls the ingested headline delta from XCom and runs the compiled
    LangGraph agentic workflow (context retrieval -> vLLM extraction ->
    guardrail validation) over each record.
    """
    summaries: List[Dict[str, Any]] = []
    try:
        task_instance = context.get("ti") or context.get("task_instance")
        headlines: List[str] = []
        if task_instance is not None:
            try:
                headlines = task_instance.xcom_pull(
                    task_ids="ingest_market_stream", key="ingested_headlines"
                ) or []
            except Exception as exc:  # noqa: BLE001
                print(f"[execute_agentic_workflow] WARNING: failed to pull XCom: {exc}")

        if not headlines:
            print("[execute_agentic_workflow] No ingested headlines found; skipping this run.")
            return summaries

        try:
            from agent_graph import run_agentic_workflow

            final_states = run_agentic_workflow(headlines)
            for state in final_states:
                summaries.append(
                    {
                        "input_text": state.get("input_text"),
                        "is_valid": state.get("is_valid"),
                        "parsed_json": state.get("parsed_json"),
                    }
                )
                print(f"[execute_agentic_workflow] Result: {summaries[-1]}")

        except Exception as exc:  # noqa: BLE001
            print(f"[execute_agentic_workflow] ERROR: agentic workflow execution failed: {exc}")

    except Exception as exc:  # noqa: BLE001
        print(f"[execute_agentic_workflow] FATAL: unexpected task failure: {exc}")

    return summaries


with DAG(
    dag_id=DAG_ID,
    description="Autonomous local financial risk & regulatory analysis pipeline.",
    default_args=default_args,
    schedule="@daily",
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    tags=["financial-risk", "local", "slm", "langgraph"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_market_stream",
        python_callable=ingest_market_stream,
    )

    agentic_workflow_task = PythonOperator(
        task_id="execute_agentic_workflow",
        python_callable=execute_agentic_workflow,
    )

    ingest_task >> agentic_workflow_task
