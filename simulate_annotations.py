"""
Simulate human annotations on flagged traces in the Langfuse annotation queue.

This script scores each queued trace with realistic human judgments on the
three annotation dimensions (accuracy, policy_compliance, tone), marks them
as DONE, and verifies the results.  It exists to demonstrate the full
feedback loop end-to-end without requiring manual annotation in the UI.

Usage:
    python simulate_annotations.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Simulated human scores keyed by trace order in the queue.
# These represent what a domain expert would assign after reviewing
# the agent's response against the knowledge base and company policy.
#
# The queue contains 8 traces from these experiment queries (in order):
#   1. refund policy        — perfect KB match
#   2. final sale           — perfect KB match
#   3. refurbished warranty — perfect KB match
#   4. accidental damage    — perfect KB match
#   5. delayed order        — partial KB coverage
#   6. free shipping        — not in KB, agent should hedge
#   7. accepted payments    — not in KB, agent should hedge
#   8. exchange option      — not in KB, agent should hedge
SIMULATED_SCORES = [
    {"accuracy": 0.95, "policy_compliance": 1.0, "tone": "excellent"},
    {"accuracy": 0.90, "policy_compliance": 1.0, "tone": "good"},
    {"accuracy": 0.95, "policy_compliance": 0.95, "tone": "excellent"},
    {"accuracy": 0.90, "policy_compliance": 1.0, "tone": "good"},
    {"accuracy": 0.70, "policy_compliance": 0.80, "tone": "good"},
    {"accuracy": 0.40, "policy_compliance": 0.70, "tone": "good"},
    {"accuracy": 0.30, "policy_compliance": 0.60, "tone": "needs_improvement"},
    {"accuracy": 0.50, "policy_compliance": 0.70, "tone": "good"},
]


def main() -> None:
    try:
        from langfuse import Langfuse
    except ImportError:
        print("ERROR: langfuse not installed.")
        sys.exit(1)

    queue_id = os.getenv("LANGFUSE_ANNOTATION_QUEUE_ID")
    if not queue_id:
        print("ERROR: LANGFUSE_ANNOTATION_QUEUE_ID is required.")
        sys.exit(1)

    client = Langfuse()

    # Fetch pending items
    items_resp = client.api.annotation_queues.list_queue_items(queue_id)
    items = items_resp.data if hasattr(items_resp, "data") else items_resp
    pending = [i for i in items if i.status == "PENDING"]

    if not pending:
        print("No pending items in queue. Run experiment first with LANGFUSE_ANNOTATION_ENABLED=true.")
        sys.exit(1)

    print(f"Found {len(pending)} pending traces in annotation queue.\n")

    if len(pending) > len(SIMULATED_SCORES):
        print(f"WARNING: {len(pending)} items but only {len(SIMULATED_SCORES)} simulated scores. Extra items skipped.")

    # Fetch score config IDs for the three dimensions
    configs = {}
    all_configs = client.api.score_configs.get(limit=100)
    config_list = all_configs.data if hasattr(all_configs, "data") else all_configs
    for cfg in config_list:
        if cfg.name in ("accuracy", "policy_compliance", "tone"):
            configs[cfg.name] = cfg.id

    missing = {"accuracy", "policy_compliance", "tone"} - set(configs.keys())
    if missing:
        print(f"ERROR: Missing score configs: {missing}. Run setup_langfuse_scoring.py first.")
        sys.exit(1)

    print(f"Score configs: {configs}\n")

    # Annotate each trace
    for i, item in enumerate(pending):
        if i >= len(SIMULATED_SCORES):
            break

        scores = SIMULATED_SCORES[i]
        trace_id = item.object_id
        print(f"Annotating trace {i + 1}/{len(pending)}: {trace_id}")

        for dimension, value in scores.items():
            data_type = "CATEGORICAL" if dimension == "tone" else "NUMERIC"
            client.create_score(
                trace_id=trace_id,
                name=dimension,
                value=value,
                data_type=data_type,
                config_id=configs[dimension],
                comment=f"Simulated human annotation for demo (score {i + 1})",
            )
            print(f"  {dimension}={value}")

        # Mark queue item as done
        client.api.annotation_queues.update_queue_item(
            queue_id=queue_id,
            item_id=item.id,
            status="COMPLETED",
        )
        print(f"  status → COMPLETED")
        print()

    # Verify
    items_resp = client.api.annotation_queues.list_queue_items(queue_id, status="COMPLETED")
    completed = items_resp.data if hasattr(items_resp, "data") else items_resp
    print(f"Done. {len(completed)} traces marked as COMPLETED in the annotation queue.")
    print("You can now run: python optimize_judge.py")


if __name__ == "__main__":
    main()
