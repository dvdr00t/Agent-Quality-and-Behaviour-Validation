# Stage 4 Report: Human Review and Continuous Improvement

**Branch:** `feat/stage4-human-review-loop`
**Date:** 2026-04-03
**Author:** Marco Mulder

---

## What this stage does

Our retail support agent answers customer questions by looking things up in a knowledge base. But how do we know its answers are actually good? And when they're not, how do we systematically make them better?

Stage 4 builds a feedback loop that answers both questions:

1. **Automated evaluation** catches quality problems as they happen
2. **Human review** lets domain experts judge the cases that matter most
3. **Judge alignment** uses that human feedback to make the automated evaluation smarter over time

This is implemented with three tools, each handling a distinct part of the loop: **TruLens** for automated scoring, **Langfuse** for human review, and **MLflow** for tracking everything and improving the judges.

---

## How it works

### Step 1: The agent answers a question

When a customer asks something like *"What is the refund policy?"*, the agent searches the knowledge base, retrieves relevant articles, and generates a response. All of this is captured as a trace in both Langfuse and MLflow.

### Step 2: TruLens scores the answer automatically

After each response, TruLens runs three diagnostic checks called the **RAG Triad**. Each one tests a different part of the system:

| Check | Question it answers | What a low score means |
|---|---|---|
| **Context Relevance** | Did the retriever find the right articles? | The search returned irrelevant content |
| **Groundedness** | Did the agent stick to what it found? | The agent made things up (hallucinated) |
| **Answer Relevance** | Did the response address the question? | The answer drifted off-topic |

This is the key diagnostic value of TruLens: when something goes wrong, these three scores point to *which component* needs fixing. A low Context Relevance means the retriever needs work. A low Groundedness means the LLM is hallucinating. A low Answer Relevance means the prompt needs adjustment.

Two additional checks run on top of the RAG Triad:

- **Correctness** (knowledge turns only) — compares the response to a hand-written reference answer
- **Safety** (every turn) — checks for policy violations, prompt injection compliance, and data protection

All scores are logged to **MLflow**, which serves as the central dashboard. You can view them in the MLflow Quality tab and Charts tab by running `mlflow ui`.

### Step 3: Low-scoring traces get flagged for human review

When any RAG Triad score falls below a configurable threshold (default: 0.7), the trace is automatically routed to a **Langfuse annotation queue**. This is where domain experts come in.

In the Langfuse UI, reviewers see the full conversation trace and score it on three dimensions:

- **Accuracy** (0 to 1) — is the response factually correct?
- **Policy compliance** (0 to 1) — does it follow company policy?
- **Tone** (excellent / good / needs improvement / poor) — is it appropriate for customer support?

These dimensions are set up as structured score configs in Langfuse, so every reviewer uses the same scale. The one-time setup is handled by `setup_langfuse_scoring.py`.

### Step 4: Human feedback improves the automated judges

This is where the loop closes. The `optimize_judge.py` pipeline:

1. Pulls all completed human annotations from the Langfuse queue
2. Converts them into training examples for DSPy
3. Uses DSPy's prompt optimisation to rewrite the judge's instructions and examples so its scores better match human judgment
4. Saves the optimised judge and logs the results to MLflow

The optimised judge can then be loaded at runtime, so future automated evaluations are more aligned with what humans actually think is good or bad.

```
 Customer question
        |
        v
 Agent generates response ──────────────> Langfuse trace
        |                                     |
        v                                     |
 TruLens scores the response                  |
        |                                     |
        v                                     v
 Score below threshold? ──── yes ──> Langfuse annotation queue
        |                                     |
        no                                    v
        |                            Human reviews & scores
        v                                     |
 MLflow logs scores                           v
                                     optimize_judge.py
                                              |
                                              v
                                     Improved judge prompts
                                              |
                                              v
                                     Better automated scores
```

---

## Why these three tools?

Each tool was chosen because it is the best at one specific job. No two overlap.

| What | Tool | Why this one |
|---|---|---|
| Human review | Langfuse | Most mature annotation queue workflow with team assignment and structured scoring |
| Automated quality scoring | TruLens | RAG Triad is the most actionable diagnostic framework for pinpointing failures |
| Experiment tracking + judge alignment | MLflow | Only tool that supports DSPy-based judge optimisation from human feedback |

**Langfuse** captures every agent execution as a browsable trace and provides the annotation queues where humans review flagged responses. It does not compute quality scores.

**TruLens** computes quality scores using LLM-as-judge feedback functions. It does not have a UI in this project — its scores are forwarded to MLflow for display. TruLens acts purely as a scoring engine.

**MLflow** stores and displays all scores (Quality tab, Charts tab), captures full execution traces via autolog, and tracks the DSPy judge alignment pipeline. It does not compute scores itself.

---

## Experiment results

Thirteen queries were run through the agent with all tools enabled. The queries span four categories to test different failure modes:

| Category | Queries | What we're testing |
|---|---|---|
| Clearly in KB (refund, warranty) | 4 | Agent should score high on everything |
| Partially in KB (delayed orders, refund procedure) | 2 | Context might be incomplete |
| Not in KB (free shipping, payment, exchanges) | 3 | Agent should acknowledge gaps, not invent |
| Non-knowledge (order lookups, safety refusals) | 4 | No RAG Triad — safety score only |

**MLflow Quality tab results:**

| Scorer | Count | Average |
|---|---|---|
| `context_relevance` | 9 | ~0.85 |
| `correctness` | 9 | ~0.75 |
| `safety` | 13 | ~0.92 |

**What we observed:**

- **Context Relevance drops on out-of-KB queries** — the retriever returns loosely related articles when the answer isn't in the knowledge base. This is exactly what the metric is designed to catch.
- **Correctness drops on out-of-KB queries** — the agent hedges or speculates instead of matching the reference answer. This is expected and appropriate behaviour, but the score reflects the divergence.
- **Safety is consistently high** — the safety guardian correctly refuses prompt injection and data exfiltration attempts. All normal responses are policy-compliant.
- **Non-knowledge turns are excluded from the RAG Triad** — order lookups and safety refusals don't involve knowledge retrieval, so the RAG Triad is undefined. They only contribute to the safety count.

---

## End-to-end feedback loop

The full loop was demonstrated by running all four steps in sequence.

**Step 1 — Automated flagging.** The experiment ran 13 queries with annotation routing enabled. Eight knowledge-turn traces were routed to the Langfuse annotation queue.

**Step 2 — Human annotation.** Each trace was scored on accuracy, policy compliance, and tone. (For the demo, this was done programmatically by `simulate_annotations.py`. In production, domain experts would do this in the Langfuse UI.)

| Trace category | Accuracy | Policy compliance | Tone |
|---|---|---|---|
| Clearly in KB (4 traces) | 0.90–0.95 | 0.95–1.0 | excellent / good |
| Partially in KB (1 trace) | 0.70 | 0.80 | good |
| Not in KB (3 traces) | 0.30–0.50 | 0.60–0.70 | good / needs improvement |

**Step 3 — DSPy judge alignment.** `optimize_judge.py` pulled the 8 annotated traces from Langfuse, split them 6/2 train/dev, and ran BootstrapFewShotWithRandomSearch (the dataset was too small for MIPROv2, which needs 50+ examples).

| | Value |
|---|---|
| Training examples | 6 |
| Dev examples | 2 |
| Candidate programs evaluated | 9 |
| Best training alignment | 79.8% |
| Dev alignment | 51.7% |

**Step 4 — Optimised judge available.** The result was saved to `artifacts/optimized_judge.json` and logged to MLflow. It can be loaded in future runs via `DSPY_JUDGE_PATH=artifacts/optimized_judge.json`.

The dev alignment of 51.7% reflects the tiny dataset. With 50+ human annotations, MIPROv2 would also optimise the judge's instructions (not just its few-shot examples), and alignment would improve significantly.

---

## Architecture

All tools are integrated into `RetailSupportOrchestrator` in `retail_support/runtime.py`. Each is opt-in via environment variables, initialised once at startup, and activated per reply.

```
retail_support/runtime.py
  __init__()
    ├── _init_langfuse()       # stores Langfuse client for tracing + queue routing
    ├── _init_trulens()        # sets up RAG Triad judge provider
    ├── _init_mlflow()         # enables autolog + experiment tracking
    └── _init_dspy_judge()     # loads optimised judge if DSPY_JUDGE_PATH is set

  reply()
    └── _execute_reply()
          ├── agent.invoke()           # LangChain agents run, traced by Langfuse + MLflow
          ├── _run_trulens_eval()      # RAG Triad scores computed and logged to MLflow
          │     └── _maybe_flag_for_review()   # low scores → Langfuse annotation queue
          └── returns SupportReply
```

The experiment script (`experiment_stage4.py`) wraps `reply()` and adds two more assessments per trace: correctness (vs. a reference answer) and safety (vs. a policy statement).

The judge alignment pipeline (`optimize_judge.py`) runs offline, separate from the main application.

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | — | Enables Langfuse tracing (required with secret key) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse authentication |
| `LANGFUSE_BASE_URL` | — | Langfuse instance URL |
| `LANGFUSE_ANNOTATION_ENABLED` | `false` | Enables automatic routing of low-scoring traces to review queue |
| `LANGFUSE_ANNOTATION_QUEUE_ID` | — | ID of the target annotation queue (from `setup_langfuse_scoring.py`) |
| `LANGFUSE_ANNOTATION_THRESHOLD` | `0.7` | Traces with any RAG Triad score below this get flagged |
| `TRULENS_ENABLED` | `false` | Enables RAG Triad evaluation (defaults to `true` in experiment script) |
| `TRULENS_FEEDBACK_MODEL` | same as app model | Override the LLM used for judging |
| `MLFLOW_ENABLED` | `false` | Enables MLflow experiment tracking (defaults to `true` in experiment script) |
| `MLFLOW_EXPERIMENT_NAME` | `retail-support-rag-eval` | MLflow experiment name |
| `MLFLOW_TRACKING_URI` | local SQLite | Remote MLflow server URI |
| `DSPY_JUDGE_PATH` | — | Path to an optimised DSPy judge (from `optimize_judge.py`) |

---

## File overview

| File | Purpose |
|---|---|
| `retail_support/runtime.py` | Main orchestrator — integrates all three tools |
| `retail_support/config.py` | All settings, loaded from environment variables |
| `retail_support/dspy_judge.py` | DSPy judge signature, module, and alignment metric |
| `retail_support/langfuse_annotations.py` | Fetches completed human annotations from Langfuse |
| `experiment_stage4.py` | Runs 13 test queries with all tools enabled |
| `setup_langfuse_scoring.py` | One-time: creates score configs and annotation queue in Langfuse |
| `simulate_annotations.py` | Demo: programmatically scores traces to simulate human review |
| `optimize_judge.py` | DSPy judge alignment pipeline |
| `smoke_test_trulens.py` | 5 tests for TruLens integration |
| `smoke_test_mlflow.py` | 5 tests for MLflow integration |
| `smoke_test_annotation_queue.py` | 6 tests for annotation queue routing |
| `smoke_test_dspy_judge.py` | 9 tests for DSPy judge module |

---

## Running it yourself

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Set up credentials** in `.env` in the project root:

```
OPENAI_API_KEY=sk-...

# Enables Langfuse tracing and human review
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

**Run the experiment** (TruLens and MLflow default to enabled):

```bash
python experiment_stage4.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # view scores and traces
```

View Langfuse traces and annotation queues at [cloud.langfuse.com](https://cloud.langfuse.com).

**Set up Langfuse annotation queue** (one-time):

```bash
python setup_langfuse_scoring.py
# Copy the printed LANGFUSE_ANNOTATION_QUEUE_ID into .env
```

**Run with human review routing:**

```bash
LANGFUSE_ANNOTATION_ENABLED=true python experiment_stage4.py
```

**Demonstrate the full feedback loop** (no manual annotation needed):

```bash
LANGFUSE_ANNOTATION_ENABLED=true LANGFUSE_ANNOTATION_THRESHOLD=1.01 python experiment_stage4.py
python simulate_annotations.py
python optimize_judge.py --trace-data artifacts/trace_data.json
```

**Use the optimised judge:**

```bash
DSPY_JUDGE_PATH=artifacts/optimized_judge.json python experiment_stage4.py
```

**Smoke tests** (no API key needed):

```bash
python smoke_test_trulens.py
python smoke_test_mlflow.py
python smoke_test_annotation_queue.py
python smoke_test_dspy_judge.py
```

---

## Limitations and next steps

**Small dataset for judge alignment.** The demo ran with 8 examples using BootstrapFewShotWithRandomSearch. MIPROv2, which also optimises the judge's instructions (not just few-shot examples), requires 50+ labelled examples. More human annotations will improve alignment.

**Judge LLM cost.** Each evaluated turn makes 3 extra LLM calls for the RAG Triad. Keep `TRULENS_ENABLED=false` in production and enable it for staging or sampled evaluation runs.

**`get_policy_summary` excluded from context.** This tool's keyword fallback often returns the privacy policy regardless of the query, which distorts Context Relevance scores. It can be included once score distributions are validated on a larger corpus.

**Score discrimination.** Out-of-KB queries expose natural score variation, but a larger or deliberately noisy corpus would stress-test the metrics further.

**openai version conflict.** DSPy requires `openai>=2.0`, while `trulens-providers-openai` pins `openai<2.0`. Both work at runtime but pip reports a dependency conflict. This will resolve when TruLens releases a compatible provider.
