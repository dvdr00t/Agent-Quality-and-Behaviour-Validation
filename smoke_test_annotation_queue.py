"""
Smoke test for Langfuse annotation queue routing.

Verifies that _maybe_flag_for_review correctly routes low-scoring traces
to the annotation queue and skips high-scoring ones.  No Langfuse
credentials required — the API client is mocked.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

passed = 0
failed = 0


def check(label: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")


def make_settings(**overrides):
    """Create a SupportSettings with annotation queue fields."""
    from retail_support.config import SupportSettings

    defaults = dict(
        provider="openai",
        model_name="gpt-4.1-mini",
        openai_api_key="sk-test",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_base_url="https://cloud.langfuse.com",
        langfuse_annotation_queue_id="queue-123",
        langfuse_annotation_enabled=True,
        langfuse_annotation_threshold=0.7,
    )
    defaults.update(overrides)
    return SupportSettings(**defaults)


def test_flags_low_scores():
    """Traces below threshold should be added to the annotation queue."""
    print("\n--- test_flags_low_scores ---")
    from retail_support.runtime import RetailSupportOrchestrator

    settings = make_settings()

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._langfuse_client = MagicMock()
        orch._langfuse_trace_id = "trace-abc"

        orch._maybe_flag_for_review(cr_score=0.5, gr_score=0.8, ar_score=0.9)

        check(
            "create_queue_item called",
            orch._langfuse_client.api.annotation_queues.create_queue_item.called,
        )
        call_kwargs = orch._langfuse_client.api.annotation_queues.create_queue_item.call_args
        check(
            "queue_id is correct",
            call_kwargs.kwargs.get("queue_id") == "queue-123"
            or (call_kwargs.args[0] if call_kwargs.args else None) == "queue-123"
            or call_kwargs[1].get("queue_id", "") == "queue-123",
        )


def test_skips_high_scores():
    """Traces at or above threshold should NOT be flagged."""
    print("\n--- test_skips_high_scores ---")
    from retail_support.runtime import RetailSupportOrchestrator

    settings = make_settings()

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._langfuse_client = MagicMock()

        orch._maybe_flag_for_review(cr_score=0.8, gr_score=0.9, ar_score=0.7)

        check(
            "create_queue_item NOT called",
            not orch._langfuse_client.api.annotation_queues.create_queue_item.called,
        )


def test_disabled_by_default():
    """When annotation is disabled, no API calls should be made."""
    print("\n--- test_disabled_by_default ---")
    from retail_support.runtime import RetailSupportOrchestrator

    settings = make_settings(langfuse_annotation_enabled=False)

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._langfuse_client = MagicMock()

        orch._maybe_flag_for_review(cr_score=0.1, gr_score=0.1, ar_score=0.1)

        check(
            "create_queue_item NOT called when disabled",
            not orch._langfuse_client.api.annotation_queues.create_queue_item.called,
        )


def test_resilient_to_api_errors():
    """API errors should be caught, not raised."""
    print("\n--- test_resilient_to_api_errors ---")
    from retail_support.runtime import RetailSupportOrchestrator

    settings = make_settings()

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._langfuse_client = MagicMock()
        orch._langfuse_trace_id = "trace-err"
        orch._langfuse_client.api.annotation_queues.create_queue_item.side_effect = RuntimeError("API down")

        try:
            orch._maybe_flag_for_review(cr_score=0.3, gr_score=0.3, ar_score=0.3)
            check("no exception raised", True)
        except Exception as e:
            check(f"no exception raised (got {e!r})", False)


def test_no_client():
    """When Langfuse client is None, should exit gracefully."""
    print("\n--- test_no_client ---")
    from retail_support.runtime import RetailSupportOrchestrator

    settings = make_settings()

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._langfuse_client = None

        try:
            orch._maybe_flag_for_review(cr_score=0.1, gr_score=0.1, ar_score=0.1)
            check("no exception with None client", True)
        except Exception as e:
            check(f"no exception with None client (got {e!r})", False)


if __name__ == "__main__":
    print("=== Annotation Queue Smoke Tests ===")
    test_flags_low_scores()
    test_skips_high_scores()
    test_disabled_by_default()
    test_resilient_to_api_errors()
    test_no_client()

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
