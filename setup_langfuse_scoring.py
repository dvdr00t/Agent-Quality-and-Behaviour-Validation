"""
One-time setup: create Langfuse score configs and annotation queue.

Creates three score configurations (accuracy, policy_compliance, tone)
and an annotation queue linked to them.  Prints the IDs for use as
environment variables.

Usage:
    python setup_langfuse_scoring.py

Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL
to be set (or present in .env).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    try:
        from langfuse import Langfuse
    except ImportError:
        print("ERROR: langfuse is not installed. Run: pip install langfuse")
        sys.exit(1)

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL")

    if not public_key or not secret_key:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required.")
        sys.exit(1)

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
    )

    print("Creating score configs...")

    accuracy_config = client.api.score_configs.create(
        name="accuracy",
        data_type="NUMERIC",
        min_value=0.0,
        max_value=1.0,
        description="How factually correct is the response given the knowledge base?",
    )
    print(f"  accuracy      config_id={accuracy_config.id}")

    policy_config = client.api.score_configs.create(
        name="policy_compliance",
        data_type="NUMERIC",
        min_value=0.0,
        max_value=1.0,
        description="Does the response follow company policy?",
    )
    print(f"  policy_compliance config_id={policy_config.id}")

    tone_config = client.api.score_configs.create(
        name="tone",
        data_type="CATEGORICAL",
        categories=[
            {"label": "excellent", "value": 1.0},
            {"label": "good", "value": 0.75},
            {"label": "needs_improvement", "value": 0.5},
            {"label": "poor", "value": 0.0},
        ],
        description="Is the tone appropriate for customer support?",
    )
    print(f"  tone          config_id={tone_config.id}")

    print("\nCreating annotation queue...")

    queue = client.api.annotation_queues.create_queue(
        name="retail-support-review",
        description="Traces flagged by automated evaluators for human review",
        score_config_ids=[accuracy_config.id, policy_config.id, tone_config.id],
    )
    print(f"  queue_id={queue.id}")

    print("\n--- Add to .env ---")
    print(f"LANGFUSE_ANNOTATION_QUEUE_ID={queue.id}")
    print("LANGFUSE_ANNOTATION_ENABLED=true")
    print()
    print("Setup complete.")


if __name__ == "__main__":
    main()
