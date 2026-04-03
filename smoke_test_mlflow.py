"""Smoke test for MLflow integration — no real LLM calls or MLflow server needed."""

from unittest.mock import MagicMock, call, patch

from retail_support.config import SupportSettings
from retail_support.runtime import RetailSupportOrchestrator


def make_orchestrator(**config_overrides):
    base = dict(provider="openai", model_name="gpt-4.1-mini", openai_api_key="test")
    base.update(config_overrides)
    settings = SupportSettings(**base)
    with patch("retail_support.runtime.TruOpenAI"), patch("retail_support.runtime.TruSession"):
        return RetailSupportOrchestrator(settings=settings)


def test_mlflow_disabled_by_default():
    o = make_orchestrator()
    assert not o._mlflow_enabled
    print("PASS — _mlflow_enabled is False by default")


def test_mlflow_enabled_flag():
    with patch("retail_support.runtime.mlflow") as mock_mlflow:
        o = make_orchestrator(mlflow_enabled=True)
    assert o._mlflow_enabled
    mock_mlflow.set_experiment.assert_called_once_with("retail-support-rag-eval")
    print("PASS — _mlflow_enabled is True, experiment set")


def test_mlflow_custom_experiment_and_tracking_uri():
    with patch("retail_support.runtime.mlflow") as mock_mlflow:
        o = make_orchestrator(
            mlflow_enabled=True,
            mlflow_experiment_name="my-experiment",
            mlflow_tracking_uri="http://localhost:5000",
        )
    mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    mock_mlflow.set_experiment.assert_called_once_with("my-experiment")
    print("PASS — custom tracking URI and experiment name applied")


def test_mlflow_run_logged_with_rag_triad_scores():
    mock_tru_provider = MagicMock()
    mock_tru_provider.context_relevance_with_cot_reasons.return_value = (0.9, {})
    mock_tru_provider.groundedness_measure_with_cot_reasons.return_value = (0.8, {})
    mock_tru_provider.relevance_with_cot_reasons.return_value = (0.85, {})

    with patch("retail_support.runtime.mlflow") as mock_mlflow, \
         patch("retail_support.runtime.TruOpenAI", return_value=mock_tru_provider), \
         patch("retail_support.runtime.TruSession"):
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        o = make_orchestrator.__wrapped__ if hasattr(make_orchestrator, "__wrapped__") else None
        settings = SupportSettings(
            provider="openai", model_name="gpt-4.1-mini", openai_api_key="test",
            trulens_enabled=True, mlflow_enabled=True,
        )
        orchestrator = RetailSupportOrchestrator(settings=settings)

    orchestrator._tru_provider = mock_tru_provider

    with patch("retail_support.runtime.mlflow") as mock_mlflow:
        mock_run_ctx = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        orchestrator._run_trulens_eval(
            query="What is the refund policy?",
            context_chunks=['[{"title": "Refund policy", "body": "30 days"}]'],
            response="You can return items within 30 days.",
            target="knowledge",
        )

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_metric.assert_any_call("context_relevance", 0.9)
        mock_mlflow.log_metric.assert_any_call("groundedness", 0.8)
        mock_mlflow.log_metric.assert_any_call("answer_relevance", 0.85)
        mock_mlflow.log_param.assert_any_call("target_agent", "knowledge")

    print("PASS — MLflow run logged with all three RAG Triad metrics")


def test_mlflow_not_logged_when_trulens_disabled():
    with patch("retail_support.runtime.mlflow") as mock_mlflow:
        settings = SupportSettings(
            provider="openai", model_name="gpt-4.1-mini", openai_api_key="test",
            trulens_enabled=False, mlflow_enabled=True,
        )
        o = RetailSupportOrchestrator(settings=settings)
        o._run_trulens_eval(
            query="What is the refund policy?",
            context_chunks=["some context"],
            response="30 days.",
        )
        mock_mlflow.start_run.assert_not_called()
    print("PASS — MLflow run not logged when TruLens is disabled")


if __name__ == "__main__":
    test_mlflow_disabled_by_default()
    test_mlflow_enabled_flag()
    test_mlflow_custom_experiment_and_tracking_uri()
    test_mlflow_run_logged_with_rag_triad_scores()
    test_mlflow_not_logged_when_trulens_disabled()
    print("\nAll smoke tests passed.")
