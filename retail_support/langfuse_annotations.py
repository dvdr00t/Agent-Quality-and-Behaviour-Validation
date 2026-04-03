"""Fetch completed human annotations from a Langfuse annotation queue.

This module is the bridge between Langfuse (human review) and the DSPy judge
alignment pipeline.  It pulls traces that humans have reviewed and scored,
returning structured records suitable for conversion to DSPy training examples.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# The three score dimensions defined in setup_langfuse_scoring.py.
ANNOTATION_DIMENSIONS = {"accuracy", "policy_compliance", "tone"}


@dataclass
class AnnotatedTrace:
    """A single trace with its human annotation scores."""

    trace_id: str
    query: str
    response: str
    context: str
    scores: dict[str, float | str] = field(default_factory=dict)


def fetch_completed_annotations(
    queue_id: str | None = None,
    *,
    langfuse_client: Any | None = None,
    trace_data: dict[str, dict[str, str]] | None = None,
    limit: int = 100,
) -> list[AnnotatedTrace]:
    """Pull completed annotation queue items and their human scores.

    Parameters
    ----------
    queue_id:
        Langfuse annotation queue ID.  Falls back to
        ``LANGFUSE_ANNOTATION_QUEUE_ID`` env var.
    langfuse_client:
        An existing ``Langfuse`` instance.  One is created from env vars when
        omitted.
    trace_data:
        Optional mapping of ``trace_id`` to ``{"query": ..., "response": ...,
        "context": ...}``.  When provided, these values are used instead of
        trying to extract them from the Langfuse trace object.  This is needed
        when traces are captured via OTel (Langfuse v4) where input/output are
        stored in span attributes rather than the trace model.
    limit:
        Maximum number of queue items to fetch.

    Returns
    -------
    list[AnnotatedTrace]
        Traces with human annotation scores (``accuracy``,
        ``policy_compliance``, ``tone``).
    """
    from langfuse import Langfuse

    queue_id = queue_id or os.getenv("LANGFUSE_ANNOTATION_QUEUE_ID")
    if not queue_id:
        raise ValueError(
            "queue_id is required — pass it directly or set LANGFUSE_ANNOTATION_QUEUE_ID"
        )

    client = langfuse_client or Langfuse()
    trace_data = trace_data or {}

    # Fetch queue items — filter to completed items
    items_response = client.api.annotation_queues.list_queue_items(
        queue_id=queue_id,
        limit=limit,
        status="COMPLETED",
    )
    items = items_response.data if hasattr(items_response, "data") else items_response

    results: list[AnnotatedTrace] = []
    for item in items:
        trace_id = item.object_id

        # Try supplementary data first, fall back to trace extraction
        if trace_id in trace_data:
            td = trace_data[trace_id]
            query = td.get("query", "")
            response = td.get("response", "")
            context = td.get("context", "")
        else:
            try:
                trace = client.api.trace.get(trace_id)
            except Exception:
                logger.warning("Could not fetch trace %s — skipping", trace_id)
                continue
            query = _extract_query(trace)
            response = _extract_response(trace)
            context = _extract_context(trace)

        if not query or not response:
            logger.warning("Trace %s has no query/response — skipping", trace_id)
            continue

        # Fetch human scores for this trace
        scores = _fetch_human_scores(client, trace_id)
        if not scores:
            logger.warning("Trace %s has no human scores — skipping", trace_id)
            continue

        results.append(
            AnnotatedTrace(
                trace_id=trace_id,
                query=query,
                response=response,
                context=context,
                scores=scores,
            )
        )

    logger.info("Fetched %d annotated traces from queue %s", len(results), queue_id)
    return results


def _fetch_human_scores(client: Any, trace_id: str) -> dict[str, float | str]:
    """Retrieve human annotation scores for a trace.

    Filters to the three annotation dimensions (accuracy, policy_compliance,
    tone) regardless of the Langfuse ``source`` field, since programmatic
    annotations via ``create_score`` are tagged as ``API`` not ``ANNOTATION``.
    """
    scores: dict[str, float | str] = {}
    try:
        score_response = client.api.scores.get_many(trace_id=trace_id)
        score_list = score_response.data if hasattr(score_response, "data") else score_response
        for score in score_list:
            if score.name in ANNOTATION_DIMENSIONS:
                scores[score.name] = score.value
    except Exception:
        logger.exception("Failed to fetch scores for trace %s", trace_id)
    return scores


def _extract_query(trace: Any) -> str:
    """Best-effort extraction of the user query from a trace."""
    if hasattr(trace, "input") and trace.input:
        inp = trace.input
        if isinstance(inp, str):
            return inp
        if isinstance(inp, dict):
            return inp.get("query", "") or inp.get("message", "") or inp.get("input", "")
    return ""


def _extract_response(trace: Any) -> str:
    """Best-effort extraction of the agent response from a trace."""
    if hasattr(trace, "output") and trace.output:
        out = trace.output
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            return out.get("text", "") or out.get("response", "") or out.get("output", "")
    return ""


def _extract_context(trace: Any) -> str:
    """Best-effort extraction of retrieved context from trace metadata."""
    if hasattr(trace, "metadata") and isinstance(trace.metadata, dict):
        ctx = trace.metadata.get("context", "")
        if isinstance(ctx, str):
            return ctx
        if isinstance(ctx, list):
            return "\n\n---\n\n".join(str(c) for c in ctx)
    return ""
