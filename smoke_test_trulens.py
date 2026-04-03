"""Smoke test for TruLens integration — no real LLM calls made."""

from unittest.mock import MagicMock, patch

from retail_support.config import SupportSettings
from retail_support.runtime import RetailSupportOrchestrator


def make_settings(**overrides):
    base = dict(provider="openai", model_name="gpt-4.1-mini", openai_api_key="test", trulens_enabled=True)
    base.update(overrides)
    return SupportSettings(**base)


def test_trulens_enabled_flag():
    settings = make_settings()
    with patch("retail_support.runtime.TruOpenAI"), patch("retail_support.runtime.TruSession"):
        o = RetailSupportOrchestrator(settings=settings)
    assert o._trulens_enabled, "TruLens should be enabled"
    print("PASS — _trulens_enabled is True")


def test_trulens_disabled_by_default():
    settings = SupportSettings(provider="openai", model_name="gpt-4.1-mini", openai_api_key="test")
    o = RetailSupportOrchestrator(settings=settings)
    assert not o._trulens_enabled, "TruLens should be disabled by default"
    print("PASS — _trulens_enabled is False by default")


def test_run_trulens_eval_calls_provider():
    settings = make_settings()
    mock_provider = MagicMock()
    mock_provider.context_relevance_with_cot_reasons.return_value = (0.9, {"reason": "relevant"})
    mock_provider.groundedness_measure_with_cot_reasons.return_value = (0.8, {"reason": "grounded"})
    mock_provider.relevance_with_cot_reasons.return_value = (0.85, {"reason": "on-topic"})

    with patch("retail_support.runtime.TruOpenAI", return_value=mock_provider), \
         patch("retail_support.runtime.TruSession"):
        o = RetailSupportOrchestrator(settings=settings)

    o._run_trulens_eval(
        query="What is the refund policy?",
        context_chunks=['[{"title": "Refund policy", "body": "30 days..."}]'],
        response="You can return items within 30 days of delivery.",
    )

    mock_provider.context_relevance_with_cot_reasons.assert_called_once()
    mock_provider.groundedness_measure_with_cot_reasons.assert_called_once()
    mock_provider.relevance_with_cot_reasons.assert_called_once()
    print("PASS — all three RAG Triad methods called")


def test_run_trulens_eval_skipped_when_no_context():
    settings = make_settings()
    mock_provider = MagicMock()

    with patch("retail_support.runtime.TruOpenAI", return_value=mock_provider), \
         patch("retail_support.runtime.TruSession"):
        o = RetailSupportOrchestrator(settings=settings)

    o._run_trulens_eval(query="What is order ord_1001?", context_chunks=[], response="Your order is shipped.")

    mock_provider.context_relevance_with_cot_reasons.assert_not_called()
    print("PASS — eval skipped when no context chunks (order-only turn)")


def test_run_trulens_eval_resilient_to_provider_error():
    settings = make_settings()
    mock_provider = MagicMock()
    mock_provider.context_relevance_with_cot_reasons.side_effect = RuntimeError("API error")

    with patch("retail_support.runtime.TruOpenAI", return_value=mock_provider), \
         patch("retail_support.runtime.TruSession"):
        o = RetailSupportOrchestrator(settings=settings)

    # Should not raise
    o._run_trulens_eval(
        query="What is the warranty?",
        context_chunks=['[{"title": "Warranty", "body": "90 days..."}]'],
        response="Refurbished devices have a 90-day warranty.",
    )
    print("PASS — eval failure does not crash the orchestrator")


if __name__ == "__main__":
    test_trulens_disabled_by_default()
    test_trulens_enabled_flag()
    test_run_trulens_eval_calls_provider()
    test_run_trulens_eval_skipped_when_no_context()
    test_run_trulens_eval_resilient_to_provider_error()
    print("\nAll smoke tests passed.")
