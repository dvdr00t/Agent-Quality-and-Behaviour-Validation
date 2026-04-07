"""DeepEval-powered Stage 1 regression tests for the retail support system."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from retail_support.stage1_eval import Stage1Case, build_stage1_cases, execute_stage1_case

pytest.importorskip("deepeval")

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic keyword checks."""

    tokens = [character if character.isalnum() or character == "_" else " " for character in value.lower()]
    return " ".join("".join(tokens).split())


def _tokenize(value: str) -> list[str]:
    """Extract lowercase word-like tokens from free text."""

    return re.findall(r"[a-z0-9_]+", value.lower())


class KeywordCoverageMetric(BaseMetric):
    """Deterministic DeepEval metric for expected keyword coverage."""

    _required_params = [
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.include_reason = True
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_model = "local-deterministic"
        self.error = None
        self.evaluation_cost = 0.0

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        expected_groups = [
            [
                _normalize_text(option)
                for option in chunk.split("|")
                if option.strip()
            ]
            for chunk in test_case.expected_output.split(";")
            if chunk.strip()
        ]
        actual_output = _normalize_text(test_case.actual_output)
        matched_groups: list[str] = []

        for options in expected_groups:
            if any(option in actual_output for option in options):
                matched_groups.append(" | ".join(options))

        self.score = 1.0 if not expected_groups else len(matched_groups) / len(expected_groups)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Matched {len(matched_groups)} of {len(expected_groups)} expected phrase groups: "
            f"{matched_groups or ['none']}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "KeywordCoverageMetric"


class ContextOverlapMetric(BaseMetric):
    """Deterministic DeepEval metric for lightweight grounding checks."""

    _required_params = [
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ]

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.include_reason = True
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_model = "local-deterministic"
        self.error = None
        self.evaluation_cost = 0.0

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        actual_tokens = {
            token
            for token in _tokenize(test_case.actual_output)
            if len(token) >= 4 and token not in STOPWORDS
        }
        context_tokens = {
            token
            for token in _tokenize(" ".join(test_case.context or []))
            if len(token) >= 4 and token not in STOPWORDS
        }

        if not actual_tokens:
            self.score = 0.0
            self.success = False
            self.reason = "The response did not contain any meaningful tokens to compare against the context."
            return self.score

        overlapping_tokens = sorted(actual_tokens & context_tokens)
        self.score = float(len(overlapping_tokens))
        self.success = self.score >= self.threshold
        self.reason = (
            f"Observed {int(self.score)} overlapping context tokens: "
            f"{overlapping_tokens[:10] or ['none']}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "ContextOverlapMetric"


@pytest.fixture(scope="module")
def orchestrator() -> Any:
    """Build the orchestrator once for the Stage 1 regression suite."""

    from retail_support.stage1_eval import build_orchestrator

    return build_orchestrator()


@pytest.mark.parametrize("case", build_stage1_cases(), ids=lambda case: case.case_id)
def test_stage1_curated_regression_suite(orchestrator: Any, case: Stage1Case) -> None:
    """Assert curated Stage 1 behavior with DeepEval-backed checks."""

    result = execute_stage1_case(orchestrator=orchestrator, case=case)

    answer_quality = KeywordCoverageMetric(threshold=1.0)
    faithfulness = ContextOverlapMetric(threshold=2.0)

    test_case = LLMTestCase(
        input=case.user_message,
        actual_output=result.response_text,
        expected_output="; ".join(case.expected_keywords),
        context=case.retrieval_context,
    )
    assert_test(test_case, [answer_quality, faithfulness])

    assert not result.missing_tools, f"Missing expected tools for {case.case_id}: {result.missing_tools}"
    assert result.step_efficiency_passed, (
        f"Step efficiency failed for {case.case_id}: observed {len(result.tool_calls)} tool calls, "
        f"expected at most {case.max_tool_calls}."
    )
    assert not result.missing_keywords, (
        f"Missing expected keywords for {case.case_id}: {result.missing_keywords}. "
        f"Response was: {result.response_text}"
    )
    assert not result.forbidden_keyword_hits, (
        f"Forbidden content leaked for {case.case_id}: {result.forbidden_keyword_hits}. "
        f"Response was: {result.response_text}"
    )


@pytest.mark.parametrize("case", build_stage1_cases(), ids=lambda case: case.case_id)
def test_stage1_result_payload_is_json_serializable(orchestrator: Any, case: Stage1Case) -> None:
    """Ensure Stage 1 outputs can be persisted as MLflow-friendly artifacts."""

    result = execute_stage1_case(orchestrator=orchestrator, case=case)
    json.dumps(result.__dict__)
