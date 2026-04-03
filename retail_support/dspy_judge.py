"""DSPy judge module for retail support response evaluation.

Defines a judge signature aligned with the human annotation dimensions
(accuracy, policy_compliance, tone) and a metric function for DSPy
optimization that compares judge output to human labels.
"""

from __future__ import annotations

import dspy


class RetailSupportJudge(dspy.Signature):
    """Evaluate a retail support agent response on accuracy, policy compliance, and tone.

    Given a customer query, the retrieved knowledge base context, and the
    agent's response, produce numeric scores for accuracy and policy
    compliance plus a categorical tone rating.
    """

    query: str = dspy.InputField(desc="The customer's question or request")
    context: str = dspy.InputField(desc="Retrieved knowledge base content used by the agent")
    response: str = dspy.InputField(desc="The agent's response to the customer")
    accuracy: float = dspy.OutputField(desc="Factual accuracy score from 0.0 (wrong) to 1.0 (perfect)")
    policy_compliance: float = dspy.OutputField(desc="Policy compliance score from 0.0 (violates policy) to 1.0 (fully compliant)")
    tone: str = dspy.OutputField(desc="Tone rating: excellent, good, needs_improvement, or poor")


class RetailJudgeModule(dspy.Module):
    """Chain-of-thought judge that produces accuracy, policy compliance, and tone ratings."""

    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(RetailSupportJudge)

    def forward(self, query: str, context: str, response: str) -> dspy.Prediction:
        return self.judge(query=query, context=context, response=response)


TONE_VALUES = {"excellent", "good", "needs_improvement", "poor"}


def judge_alignment_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: object | None = None,
) -> float:
    """Compare DSPy judge output to human-labeled ground truth.

    Weighted combination:
      - 35% accuracy closeness (1 - |pred - human|)
      - 35% policy compliance closeness
      - 30% tone exact match

    Returns a float in [0, 1] where 1.0 means perfect alignment.
    """
    try:
        acc_diff = abs(float(pred.accuracy) - float(example.accuracy))
    except (ValueError, TypeError, AttributeError):
        acc_diff = 1.0

    try:
        pol_diff = abs(float(pred.policy_compliance) - float(example.policy_compliance))
    except (ValueError, TypeError, AttributeError):
        pol_diff = 1.0

    try:
        tone_match = 1.0 if pred.tone.strip().lower() == str(example.tone).strip().lower() else 0.0
    except (AttributeError, TypeError):
        tone_match = 0.0

    return (1.0 - acc_diff) * 0.35 + (1.0 - pol_diff) * 0.35 + tone_match * 0.3
