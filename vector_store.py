"""
vector_store.py
----------------
A clean wrapper client around Pinecone for storing and retrieving
regulatory / compliance rule text segments used as retrieval context by
the LangGraph agent.

Design notes:
- Text embedding is done with a lightweight *local* CPU model
  (`sentence-transformers/all-MiniLM-L6-v2`) so that the GPU stays free
  and fully available for the SLM used in `train_lora.py` / `serve_vllm.py`.
- All Pinecone network calls and local embedding calls are wrapped in
  explicit try/except containment loops so a transient network or index
  error never crashes the calling agent graph.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config import EMBEDDING, PINECONE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when a vector store operation fails after containment handling."""


class ComplianceVectorStore:
    """
    Thin wrapper around a single Pinecone index used to store compliance /
    regulatory rule snippets and retrieve them by semantic similarity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ) -> None:
        self.api_key: str = api_key or PINECONE.api_key
        self.index_name: str = index_name or PINECONE.index_name
        self.embedding_model_name: str = embedding_model_name or EMBEDDING.model_name

        if not self.api_key:
            raise VectorStoreError(
                "PINECONE_API_KEY is not set. Export it in your environment or "
                "populate it in a local `.env` file before instantiating "
                "ComplianceVectorStore."
            )

        try:
            self._client: Pinecone = Pinecone(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001 - deliberate broad containment
            raise VectorStoreError(f"Failed to initialize Pinecone client: {exc}") from exc

        try:
            self._embedder: SentenceTransformer = SentenceTransformer(
                self.embedding_model_name, device=EMBEDDING.device
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Failed to load local embedding model '{self.embedding_model_name}': {exc}"
            ) from exc

        self._ensure_index_exists()
        self._index = self._client.Index(self.index_name)

    def _ensure_index_exists(self) -> None:
        """Creates the target Pinecone index if it does not already exist."""
        try:
            existing_indexes = {idx["name"] for idx in self._client.list_indexes()}
            if self.index_name not in existing_indexes:
                logger.info("Pinecone index '%s' not found. Creating it now.", self.index_name)
                self._client.create_index(
                    name=self.index_name,
                    dimension=PINECONE.embedding_dim,
                    metric=PINECONE.metric,
                    spec=ServerlessSpec(cloud="aws", region=PINECONE.environment),
                )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to verify/create Pinecone index: {exc}") from exc

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embeds a batch of texts locally on CPU, returning plain Python lists."""
        try:
            vectors = self._embedder.encode(
                list(texts), convert_to_numpy=True, show_progress_bar=False
            )
            return [vector.tolist() for vector in vectors]
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Local embedding step failed: {exc}") from exc

    def upsert_compliance_rules(
        self,
        text_segments: List[str],
        source: str = "manual_upload",
        namespace: str = "compliance-rules",
    ) -> Dict[str, Any]:
        """
        Embeds a list of compliance/regulatory text segments and upserts them
        into the Pinecone index with clean structural metadata.

        Args:
            text_segments: Raw regulatory text chunks to embed and store.
            source: Free-form label describing where the rule text came from.
            namespace: Pinecone namespace to isolate this batch of vectors.

        Returns:
            A summary dict describing how many vectors were upserted, or an
            explicit error payload if the operation failed.
        """
        if not text_segments:
            logger.warning("upsert_compliance_rules called with an empty list; nothing to do.")
            return {"upserted_count": 0, "status": "skipped_empty_input"}

        try:
            embeddings = self._embed_texts(text_segments)

            vectors_to_upsert: List[Dict[str, Any]] = []
            for segment, embedding in zip(text_segments, embeddings):
                vectors_to_upsert.append(
                    {
                        "id": str(uuid.uuid4()),
                        "values": embedding,
                        "metadata": {
                            "text": segment,
                            "source": source,
                            "char_length": len(segment),
                        },
                    }
                )

            response = self._index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            upserted_count = getattr(response, "upserted_count", len(vectors_to_upsert))
            logger.info("Upserted %s compliance rule vectors into '%s'.", upserted_count, self.index_name)
            return {"upserted_count": upserted_count, "status": "success"}

        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("upsert_compliance_rules failed: %s", exc)
            return {"upserted_count": 0, "status": "error", "error": str(exc)}

    def retrieve_contextual_rules(
        self,
        query: str,
        top_k: int = 2,
        namespace: str = "compliance-rules",
    ) -> List[Dict[str, Any]]:
        """
        Queries the active Pinecone index using cosine similarity and returns
        raw context matches for the given free-text query.

        Args:
            query: Natural language text (e.g. a news headline) used to
                retrieve semantically related compliance rules.
            top_k: Number of nearest-neighbor matches to return.
            namespace: Pinecone namespace to search within.

        Returns:
            A list of match dicts (each with `id`, `score`, and `metadata`
            keys). Returns an empty list if the query fails or no rules are
            found, so callers never need to null-check.
        """
        if not query or not query.strip():
            logger.warning("retrieve_contextual_rules called with an empty query.")
            return []

        try:
            query_embedding = self._embed_texts([query])[0]

            results = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )

            matches: List[Dict[str, Any]] = []
            for match in getattr(results, "matches", []) or []:
                matches.append(
                    {
                        "id": match.get("id") if isinstance(match, dict) else match.id,
                        "score": match.get("score") if isinstance(match, dict) else match.score,
                        "metadata": match.get("metadata") if isinstance(match, dict) else match.metadata,
                    }
                )
            return matches

        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieve_contextual_rules failed for query '%s': %s", query, exc)
            return []


if __name__ == "__main__":
    try:
        store = ComplianceVectorStore()
        sample_rules = [
            "SEC Rule 10b-5 prohibits fraud in connection with the purchase or sale of securities.",
            "Basel III requires banks to maintain a minimum common equity Tier 1 capital ratio of 4.5%.",
        ]
        store.upsert_compliance_rules(sample_rules, source="smoke_test")
        results = store.retrieve_contextual_rules("insider trading fraud disclosure", top_k=2)
        for r in results:
            logger.info("Match: %s", r)
    except VectorStoreError as e:
        logger.error("Vector store smoke test failed: %s", e)
