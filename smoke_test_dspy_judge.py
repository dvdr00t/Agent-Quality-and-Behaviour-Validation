"""
Smoke test for the DSPy judge module and alignment metric.

Verifies that the judge module can be instantiated and that the metric
function returns correct values for known inputs.  No API key required.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

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


def test_module_instantiation():
    """RetailJudgeModule should instantiate without errors."""
    print("\n--- test_module_instantiation ---")
    from retail_support.dspy_judge import RetailJudgeModule

    try:
        judge = RetailJudgeModule()
        check("instantiation succeeds", judge is not None)
        check("has judge attribute", hasattr(judge, "judge"))
    except Exception as e:
        check(f"instantiation succeeds (got {e!r})", False)


def test_metric_perfect_match():
    """Perfect alignment should return 1.0."""
    print("\n--- test_metric_perfect_match ---")
    import dspy

    from retail_support.dspy_judge import judge_alignment_metric

    example = dspy.Example(accuracy=0.9, policy_compliance=0.8, tone="good")
    pred = dspy.Prediction(accuracy=0.9, policy_compliance=0.8, tone="good")

    score = judge_alignment_metric(example, pred)
    check(f"perfect match score = {score:.3f} (expect 1.0)", abs(score - 1.0) < 0.001)


def test_metric_total_mismatch():
    """Total mismatch should return a low score."""
    print("\n--- test_metric_total_mismatch ---")
    import dspy

    from retail_support.dspy_judge import judge_alignment_metric

    example = dspy.Example(accuracy=1.0, policy_compliance=1.0, tone="excellent")
    pred = dspy.Prediction(accuracy=0.0, policy_compliance=0.0, tone="poor")

    score = judge_alignment_metric(example, pred)
    check(f"mismatch score = {score:.3f} (expect 0.0)", abs(score - 0.0) < 0.001)


def test_metric_partial_match():
    """Partial match should return intermediate score."""
    print("\n--- test_metric_partial_match ---")
    import dspy

    from retail_support.dspy_judge import judge_alignment_metric

    example = dspy.Example(accuracy=0.8, policy_compliance=0.6, tone="good")
    pred = dspy.Prediction(accuracy=0.8, policy_compliance=0.6, tone="excellent")

    score = judge_alignment_metric(example, pred)
    # Accuracy and policy match perfectly (0.35 + 0.35 = 0.7), tone mismatches (0.0)
    check(f"partial match score = {score:.3f} (expect 0.7)", abs(score - 0.7) < 0.001)


def test_metric_tone_whitespace():
    """Tone matching should be case-insensitive and whitespace-tolerant."""
    print("\n--- test_metric_tone_whitespace ---")
    import dspy

    from retail_support.dspy_judge import judge_alignment_metric

    example = dspy.Example(accuracy=0.5, policy_compliance=0.5, tone="Good")
    pred = dspy.Prediction(accuracy=0.5, policy_compliance=0.5, tone="  good  ")

    score = judge_alignment_metric(example, pred)
    check(f"tone whitespace score = {score:.3f} (expect 1.0)", abs(score - 1.0) < 0.001)


def test_metric_handles_bad_values():
    """Metric should handle non-numeric values gracefully."""
    print("\n--- test_metric_handles_bad_values ---")
    import dspy

    from retail_support.dspy_judge import judge_alignment_metric

    example = dspy.Example(accuracy=0.8, policy_compliance=0.6, tone="good")
    pred = dspy.Prediction(accuracy="not-a-number", policy_compliance=0.6, tone="good")

    try:
        score = judge_alignment_metric(example, pred)
        check(f"handles bad values (score={score:.3f})", 0.0 <= score <= 1.0)
    except Exception as e:
        check(f"handles bad values (got {e!r})", False)


def test_init_dspy_judge_missing_path():
    """_init_dspy_judge should set _dspy_judge_enabled=False when path doesn't exist."""
    print("\n--- test_init_dspy_judge_missing_path ---")
    from retail_support.config import SupportSettings
    from retail_support.runtime import RetailSupportOrchestrator

    settings = SupportSettings(
        provider="openai",
        model_name="gpt-4.1-mini",
        openai_api_key="sk-test",
        dspy_judge_path="/nonexistent/path.json",
    )

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._init_dspy_judge()
        check("dspy_judge_enabled is False", orch._dspy_judge_enabled is False)


def test_init_dspy_judge_no_path():
    """_init_dspy_judge should be a no-op when path is None."""
    print("\n--- test_init_dspy_judge_no_path ---")
    from retail_support.config import SupportSettings
    from retail_support.runtime import RetailSupportOrchestrator

    settings = SupportSettings(
        provider="openai",
        model_name="gpt-4.1-mini",
        openai_api_key="sk-test",
        dspy_judge_path=None,
    )

    with patch.object(RetailSupportOrchestrator, "__init__", lambda self, **kw: None):
        orch = RetailSupportOrchestrator.__new__(RetailSupportOrchestrator)
        orch.settings = settings
        orch._init_dspy_judge()
        check("dspy_judge_enabled is False", orch._dspy_judge_enabled is False)


if __name__ == "__main__":
    print("=== DSPy Judge Smoke Tests ===")
    test_module_instantiation()
    test_metric_perfect_match()
    test_metric_total_mismatch()
    test_metric_partial_match()
    test_metric_tone_whitespace()
    test_metric_handles_bad_values()
    test_init_dspy_judge_missing_path()
    test_init_dspy_judge_no_path()

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
