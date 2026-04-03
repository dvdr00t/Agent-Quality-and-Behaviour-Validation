"""
DSPy judge alignment pipeline — optimize LLM judge prompts from human feedback.

Pulls human annotations from Langfuse, converts them to DSPy training
examples, and optimizes the judge module using MIPROv2 (or
BootstrapFewShotWithRandomSearch for smaller datasets).  Results are
tracked in MLflow.

Usage:
    python optimize_judge.py [--trace-data artifacts/trace_data.json]

Required env vars:
    OPENAI_API_KEY                   — LLM for the judge
    LANGFUSE_ANNOTATION_QUEUE_ID     — queue with completed human reviews
    LANGFUSE_PUBLIC_KEY / SECRET_KEY — Langfuse auth

Optional:
    --trace-data PATH                — JSON mapping trace_id → {query, response, context}
                                       Needed when Langfuse traces don't store input/output
                                       (e.g. OTel-based Langfuse v4 traces)
    OPENAI_MODEL                     — judge model (default: gpt-4.1-mini)
    DSPY_JUDGE_OUTPUT                — output path (default: artifacts/optimized_judge.json)
    MLFLOW_ENABLED                   — log alignment run to MLflow (default: true)
    MLFLOW_EXPERIMENT_NAME           — MLflow experiment (default: judge-alignment)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("MLFLOW_ENABLED", "true")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSPy judge alignment pipeline")
    parser.add_argument(
        "--trace-data",
        type=str,
        default=None,
        help="JSON file mapping trace_id → {query, response, context}",
    )
    args = parser.parse_args()

    # --- Validate environment ---------------------------------------------------
    queue_id = os.getenv("LANGFUSE_ANNOTATION_QUEUE_ID")
    if not queue_id:
        print("ERROR: LANGFUSE_ANNOTATION_QUEUE_ID is required.")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required.")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    output_path = os.getenv("DSPY_JUDGE_OUTPUT", "artifacts/optimized_judge.json")
    mlflow_enabled = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

    # --- Load supplementary trace data ------------------------------------------
    trace_data: dict[str, dict[str, str]] | None = None
    if args.trace_data:
        with open(args.trace_data) as f:
            trace_data = json.load(f)
        print(f"Loaded trace data for {len(trace_data)} traces from {args.trace_data}")

    # --- Pull human annotations -------------------------------------------------
    from retail_support.langfuse_annotations import fetch_completed_annotations

    print(f"\nFetching human annotations from queue {queue_id}...")
    annotations = fetch_completed_annotations(queue_id=queue_id, trace_data=trace_data)

    if len(annotations) < 5:
        print(f"ERROR: Only {len(annotations)} annotated traces found. Need at least 5.")
        print("Complete more annotations in the Langfuse UI, then re-run.")
        sys.exit(1)

    print(f"Found {len(annotations)} annotated traces.")

    # --- Convert to DSPy examples -----------------------------------------------
    import dspy

    # Langfuse returns categorical scores as numeric values, not labels.
    # Map them back to the labels defined in setup_langfuse_scoring.py.
    _tone_value_to_label = {1.0: "excellent", 0.75: "good", 0.5: "needs_improvement", 0.0: "poor"}

    examples: list[dspy.Example] = []
    for ann in annotations:
        accuracy = ann.scores.get("accuracy")
        policy_compliance = ann.scores.get("policy_compliance")
        tone_raw = ann.scores.get("tone")

        if accuracy is None or policy_compliance is None or tone_raw is None:
            logger.warning(
                "Trace %s missing required scores (accuracy=%s, policy_compliance=%s, tone=%s) — skipping",
                ann.trace_id, accuracy, policy_compliance, tone_raw,
            )
            continue

        # Resolve tone: could be a numeric value (from categorical config) or a string label
        if isinstance(tone_raw, (int, float)):
            tone = _tone_value_to_label.get(float(tone_raw), "good")
        else:
            tone = str(tone_raw).strip().lower()

        examples.append(
            dspy.Example(
                query=ann.query,
                context=ann.context,
                response=ann.response,
                accuracy=float(accuracy),
                policy_compliance=float(policy_compliance),
                tone=tone,
            ).with_inputs("query", "context", "response")
        )

    if len(examples) < 5:
        print(f"ERROR: Only {len(examples)} complete examples after filtering. Need at least 5.")
        sys.exit(1)

    print(f"Prepared {len(examples)} training examples.")

    # --- Train/dev split --------------------------------------------------------
    split_idx = max(1, int(len(examples) * 0.8))
    trainset = examples[:split_idx]
    devset = examples[split_idx:] or examples[:1]  # at least 1 dev example

    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    # --- Configure DSPy LM ------------------------------------------------------
    lm = dspy.LM(model=f"openai/{model}", api_key=api_key, temperature=0)
    dspy.configure(lm=lm)

    # --- Set up MLflow tracking -------------------------------------------------
    if mlflow_enabled:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "judge-alignment"))
        mlflow.dspy.autolog()

    # --- Instantiate judge and choose optimizer ---------------------------------
    from retail_support.dspy_judge import RetailJudgeModule, judge_alignment_metric

    judge = RetailJudgeModule()

    if len(trainset) >= 50:
        print("\nUsing MIPROv2 (50+ examples)...")
        optimizer = dspy.MIPROv2(
            metric=judge_alignment_metric,
            num_candidates=7,
            num_trials=min(len(trainset), 30),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            auto="light",
        )
    else:
        print(f"\nUsing BootstrapFewShotWithRandomSearch ({len(trainset)} examples)...")
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=judge_alignment_metric,
            num_candidate_programs=min(len(trainset), 10),
            max_bootstrapped_demos=min(3, len(trainset)),
            max_labeled_demos=min(4, len(trainset)),
        )

    # --- Run optimization -------------------------------------------------------
    print("Starting optimization...\n")
    optimized_judge = optimizer.compile(
        student=judge,
        trainset=trainset,
        **({"valset": devset} if len(trainset) >= 50 else {}),
    )

    # --- Evaluate on dev set ----------------------------------------------------
    print("\nEvaluating on dev set...")
    dev_scores = []
    for ex in devset:
        pred = optimized_judge(query=ex.query, context=ex.context, response=ex.response)
        score = judge_alignment_metric(ex, pred)
        dev_scores.append(score)

    avg_alignment = sum(dev_scores) / len(dev_scores) if dev_scores else 0.0
    print(f"Average alignment on dev set: {avg_alignment:.3f}")

    # --- Save optimized program -------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    optimized_judge.save(output_path)
    print(f"Optimized judge saved to: {output_path}")

    # --- Log final metrics to MLflow --------------------------------------------
    if mlflow_enabled:
        import mlflow

        with mlflow.start_run(run_name="judge-alignment-final"):
            mlflow.log_metric("dev_alignment_score", avg_alignment)
            mlflow.log_metric("num_training_examples", len(trainset))
            mlflow.log_metric("num_dev_examples", len(devset))
            mlflow.log_param("optimizer", type(optimizer).__name__)
            mlflow.log_param("judge_model", model)
            mlflow.log_artifact(output_path)

    print(f"\nTo use the optimized judge, set:")
    print(f"  DSPY_JUDGE_PATH={output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
