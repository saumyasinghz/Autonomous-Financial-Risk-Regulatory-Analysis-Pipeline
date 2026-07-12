"""
agent_graph.py
--------------
LangGraph `StateGraph` coordination layer for the Autonomous Financial Risk
& Regulatory Analysis Pipeline.

Graph shape:

    Context Enrichment Node  -->  vLLM Extractor Node  -->  Verification Guardrail Node
                                         ^                              |
                                         |____ (retry on invalid JSON) __|
                                                                         |
                                                                        END (once valid or retries exhausted)

- Context Enrichment Node: queries `vector_store.py` with the raw news text
  to gather regulatory context from Pinecone.
- vLLM Extractor Node: fires a structured completion HTTP call to the local
  vLLM server (`serve_vllm.py`, running at `localhost:8000`) to produce a
  risk-assessment JSON string.
- Verification Guardrail Node: validates the JSON keys/values. If invalid,
  routes back to the extractor node for a corrective retry (bounded by
  `config.AGENT.max_correction_retries`), otherwise terminates the graph.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from langgraph.graph import END, StateGraph

from config import AGENT, VLLM
from vector_store import ComplianceVectorStore, VectorStoreError

PROMPT_TEMPLATE = (
    "You are a financial risk analyst. Read the news headline and the regulatory "
    "context below, then output ONLY a JSON object with exactly the keys "
    "'company', 'risk_level' (Low/Medium/High), and 'implication'. No prose, no "
    "markdown fences.\n\n"
    "Regulatory context:\n{context}\n\n"
    "Headline: {headline}\nJSON:"
)

CORRECTION_SUFFIX = (
    "\n\nYour previous output was invalid or malformed: {reason}\n"
    "Re-emit ONLY a corrected, valid JSON object with exactly the keys "
    "'company', 'risk_level', and 'implication'."
)


class AgentState(TypedDict, total=False):
    """
    Execution tracking schema threaded through every node in the graph.

    Core fields (per spec): input_text, retrieved_rules, model_raw_output,
    parsed_json, is_valid.

    `correction_attempts` and `validation_error` are additional internal
    control fields used to bound and explain the guardrail's retry loop.
    """

    input_text: str
    retrieved_rules: List[Dict[str, Any]]
    model_raw_output: str
    parsed_json: Optional[Dict[str, Any]]
    is_valid: bool
    correction_attempts: int
    validation_error: Optional[str]


_vector_store_singleton: Optional[ComplianceVectorStore] = None


def _get_vector_store() -> Optional[ComplianceVectorStore]:
    """Lazily builds a single shared ComplianceVectorStore instance for the process."""
    global _vector_store_singleton
    if _vector_store_singleton is not None:
        return _vector_store_singleton

    try:
        _vector_store_singleton = ComplianceVectorStore()
        return _vector_store_singleton
    except VectorStoreError as exc:
        print(f"[agent_graph] WARNING: vector store unavailable, continuing without context: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[agent_graph] WARNING: unexpected vector store init failure: {exc}")
        return None


def context_enrichment_node(state: AgentState) -> Dict[str, Any]:
    """Queries the local vector store for regulatory context relevant to the input text."""
    input_text = state.get("input_text", "")
    try:
        store = _get_vector_store()
        if store is None:
            return {"retrieved_rules": []}

        matches = store.retrieve_contextual_rules(query=input_text, top_k=2)
        return {"retrieved_rules": matches}
    except Exception as exc:  # noqa: BLE001
        print(f"[agent_graph] context_enrichment_node failed: {exc}")
        return {"retrieved_rules": []}


def _format_context(retrieved_rules: List[Dict[str, Any]]) -> str:
    try:
        if not retrieved_rules:
            return "No additional regulatory context was retrieved."
        snippets = []
        for match in retrieved_rules:
            metadata = match.get("metadata") or {}
            text = metadata.get("text", "")
            if text:
                snippets.append(f"- {text}")
        return "\n".join(snippets) if snippets else "No additional regulatory context was retrieved."
    except Exception:  # noqa: BLE001
        return "No additional regulatory context was retrieved."


def vllm_extractor_node(state: AgentState) -> Dict[str, Any]:
    """Calls the local vLLM server to extract a structured risk-assessment JSON."""
    input_text = state.get("input_text", "")
    retrieved_rules = state.get("retrieved_rules", [])
    is_retry = state.get("model_raw_output") is not None and not state.get("is_valid", True)

    prompt = PROMPT_TEMPLATE.format(context=_format_context(retrieved_rules), headline=input_text)
    if is_retry:
        prompt += CORRECTION_SUFFIX.format(reason=state.get("validation_error", "unknown validation error"))

    try:
        with httpx.Client(timeout=AGENT.request_timeout_seconds) as client:
            response = client.post(
                VLLM.generate_url,
                json={"prompt": prompt, "max_tokens": 256, "temperature": 0.0},
            )
            response.raise_for_status()
            payload = response.json()
            raw_output = payload.get("text", "")
            return {"model_raw_output": raw_output}

    except httpx.HTTPError as exc:
        print(f"[agent_graph] vllm_extractor_node HTTP error: {exc}")
        return {"model_raw_output": ""}
    except Exception as exc:  # noqa: BLE001
        print(f"[agent_graph] vllm_extractor_node failed unexpectedly: {exc}")
        return {"model_raw_output": ""}


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses the first JSON object found in a raw model output string."""
    start_index = raw_text.find("{")
    end_index = raw_text.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("No JSON object braces found in model output.")
    candidate = raw_text[start_index : end_index + 1]
    return json.loads(candidate)


def verification_guardrail_node(state: AgentState) -> Dict[str, Any]:
    """Validates the extractor's JSON output against the required schema and value constraints."""
    raw_output = state.get("model_raw_output", "") or ""
    attempts = state.get("correction_attempts", 0)

    try:
        parsed = _extract_json_object(raw_output)

        missing_keys = [key for key in AGENT.required_json_keys if key not in parsed]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if parsed.get("risk_level") not in AGENT.valid_risk_levels:
            raise ValueError(
                f"Invalid risk_level '{parsed.get('risk_level')}'; "
                f"expected one of {AGENT.valid_risk_levels}"
            )

        if not isinstance(parsed.get("company"), str) or not parsed.get("company", "").strip():
            raise ValueError("Field 'company' must be a non-empty string.")

        if not isinstance(parsed.get("implication"), str) or not parsed.get("implication", "").strip():
            raise ValueError("Field 'implication' must be a non-empty string.")

        return {
            "parsed_json": parsed,
            "is_valid": True,
            "validation_error": None,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[agent_graph] verification_guardrail_node: invalid output on attempt {attempts + 1}: {exc}")
        return {
            "parsed_json": None,
            "is_valid": False,
            "validation_error": str(exc),
            "correction_attempts": attempts + 1,
        }


def _route_after_guardrail(state: AgentState) -> str:
    """Conditional edge: retry extraction on invalid JSON, up to the configured retry budget."""
    if state.get("is_valid", False):
        return END
    if state.get("correction_attempts", 0) >= AGENT.max_correction_retries:
        print("[agent_graph] Max correction retries exhausted; terminating with invalid result.")
        return END
    return "vllm_extractor"


def build_agent_graph() -> Any:
    """Compiles the LangGraph StateGraph wiring the three functional nodes together."""
    try:
        graph = StateGraph(AgentState)

        graph.add_node("context_enrichment", context_enrichment_node)
        graph.add_node("vllm_extractor", vllm_extractor_node)
        graph.add_node("verification_guardrail", verification_guardrail_node)

        graph.set_entry_point("context_enrichment")
        graph.add_edge("context_enrichment", "vllm_extractor")
        graph.add_edge("vllm_extractor", "verification_guardrail")
        graph.add_conditional_edges(
            "verification_guardrail",
            _route_after_guardrail,
            {"vllm_extractor": "vllm_extractor", END: END},
        )

        return graph.compile()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to compile LangGraph state graph: {exc}") from exc


def run_agentic_workflow(news_records: List[str]) -> List[AgentState]:
    """
    Entry point used by `airflow_dag.py`: runs the compiled graph once per
    ingested news record and returns the final state for each.
    """
    results: List[AgentState] = []
    try:
        compiled_graph = build_agent_graph()
    except RuntimeError as exc:
        print(f"[agent_graph] Cannot run workflow, graph failed to compile: {exc}")
        return results

    for record in news_records:
        try:
            initial_state: AgentState = {
                "input_text": record,
                "retrieved_rules": [],
                "model_raw_output": "",
                "parsed_json": None,
                "is_valid": False,
                "correction_attempts": 0,
                "validation_error": None,
            }
            final_state = compiled_graph.invoke(initial_state)
            results.append(final_state)
        except Exception as exc:  # noqa: BLE001
            print(f"[agent_graph] Workflow failed for record '{record[:60]}...': {exc}")
            continue

    return results


if __name__ == "__main__":
    sample_records = [
        "$AAPL shares tumble after weak iPhone demand guidance",
        "Regional bank stress test results raise capital adequacy concerns",
    ]
    outcomes = run_agentic_workflow(sample_records)
    for outcome in outcomes:
        print(json.dumps(outcome, default=str, indent=2))
