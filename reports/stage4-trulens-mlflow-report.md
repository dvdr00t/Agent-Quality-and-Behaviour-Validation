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

## The application being evaluated

The system under evaluation is a retail customer support agent built with LangChain. It has four specialist agents coordinated by a supervisor:

- **Knowledge specialist** — answers policy and FAQ questions by searching a knowledge base
- **Order specialist** — looks up orders, checks refund eligibility, creates escalation tickets
- **Safety specialist** — detects prompt injection, data exfiltration, and policy bypass attempts
- **Supervisor** — routes incoming questions to the right specialist

The knowledge base is a **small, static dataset of 4 articles** (refund policy, shipping delays, warranty terms, privacy rules) stored in memory. Retrieval is done by **keyword matching** — the query and each article are tokenized, overlapping words are counted, and the top 2 articles are returned. There are no embeddings, no vector database, and no semantic search.

This means the application follows the same **retrieve → generate** pattern as a production RAG system, but with a deliberately simple implementation. The evaluation tools (TruLens, Langfuse, MLflow) treat it the same way they would treat a real RAG pipeline — the diagnostics, scoring, and feedback loop all work identically. The small corpus does mean that out-of-scope questions (like "Do you offer free shipping?") will retrieve loosely related articles rather than finding nothing, because there are only 4 articles to choose from.

---

## How it works

### Step 1: The agent answers a question

When a customer asks a question, the supervisor routes it to the appropriate specialist. For knowledge questions like *"What is the refund policy?"*, the knowledge specialist searches the knowledge base, retrieves the most relevant articles, and generates a response grounded in what it found. All of this — the routing, retrieval, and generation — is captured as a trace in both Langfuse and MLflow.

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

### What human review can and cannot fix

Human review produces two kinds of insight, and they lead to very different actions.

**Outcome 1: The judge was wrong — the agent was fine.**

A customer asks *"What is the warranty for refurbished devices?"* The agent responds: *"Refurbished devices have a 90-day warranty that covers hardware defects but not accidental damage."* This is correct — it closely matches the knowledge base article.

But TruLens scores groundedness at 0.5. Its judge prompt is overly strict about paraphrasing: the knowledge base says "90-day limited warranty" and the agent said "90-day warranty", so the judge penalises it.

A human reviewer scores accuracy at 0.95 — the answer is essentially right. That gap between 0.5 and 0.95 becomes training data for DSPy. The optimisation pipeline rewrites the judge prompt to be more tolerant of semantically equivalent paraphrasing. Next time, the automated judge scores it correctly without human help.

Nothing in the application changes. The agent was already doing a good job — the judge just couldn't tell. **This is the outcome that the feedback loop automates.**

**Outcome 2: The agent was wrong — the judge was right.**

A customer asks *"Do you offer free shipping?"* The knowledge base has no article about free shipping, but the keyword retriever returns the "Shipping delays and escalation" article because the word "shipping" matches. The agent reads that article and responds: *"We offer shipping via DHL, UPS, and FedEx"* — which doesn't answer the question.

TruLens scores Context Relevance low (the retrieved article is about delays, not shipping costs) and Answer Relevance low (the response doesn't address free shipping). A human reviewer confirms: accuracy 0.3, the agent should have said *"I don't have information about free shipping."*

The judge was right. The problem is in the application itself. Possible fixes:
- **Add a shipping costs article** to the knowledge base — a content change
- **Improve the retriever** to recognise when nothing relevant was found — a retrieval logic change
- **Adjust the agent prompt** to say "if the retrieved content doesn't answer the question, say so" — a prompt change

None of these can be done by DSPy or Langfuse. They require an engineer to decide what to fix and make the change. **The feedback loop identifies and confirms the problem, but fixing the application is manual work.**

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

All configuration is done through environment variables. There are three places where they can be set:

1. **`.env` file** — for credentials and secrets. Loaded automatically by the experiment script. This is where you put values that don't change between runs.
2. **`experiment_stage4.py` defaults** — the experiment script sets `TRULENS_ENABLED=true` and `MLFLOW_ENABLED=true` automatically so you don't have to. These can be overridden on the command line.
3. **Command line** — for per-run overrides, e.g. `LANGFUSE_ANNOTATION_ENABLED=true python experiment_stage4.py`.

**Credentials (set these in `.env`):**

| Variable | What it does |
|---|---|
| `OPENAI_API_KEY` | Required. API key for the LLM |
| `LANGFUSE_PUBLIC_KEY` | Enables Langfuse tracing. Required together with secret key |
| `LANGFUSE_SECRET_KEY` | Langfuse authentication |
| `LANGFUSE_BASE_URL` | Langfuse instance URL (e.g. `https://cloud.langfuse.com`) |

**Evaluation settings (handled automatically, override if needed):**

| Variable | Default | Where it's set | What it does |
|---|---|---|---|
| `TRULENS_ENABLED` | `true` | `experiment_stage4.py` | Enables RAG Triad scoring |
| `MLFLOW_ENABLED` | `true` | `experiment_stage4.py` | Enables MLflow experiment tracking |
| `MLFLOW_EXPERIMENT_NAME` | `retail-support-rag-eval` | code default | MLflow experiment name |
| `MLFLOW_TRACKING_URI` | local SQLite | code default | Override to point to a remote MLflow server |
| `TRULENS_FEEDBACK_MODEL` | same as app model | code default | Use a different LLM for judging |

**Annotation queue (set on command line or in `.env` after running `setup_langfuse_scoring.py`):**

| Variable | Default | What it does |
|---|---|---|
| `LANGFUSE_ANNOTATION_QUEUE_ID` | — | ID of the annotation queue. Printed by `setup_langfuse_scoring.py` |
| `LANGFUSE_ANNOTATION_ENABLED` | `false` | Set to `true` to route low-scoring traces to the queue |
| `LANGFUSE_ANNOTATION_THRESHOLD` | `0.7` | Flag traces with any RAG Triad score below this |

**DSPy judge (set after running `optimize_judge.py`):**

| Variable | Default | What it does |
|---|---|---|
| `DSPY_JUDGE_PATH` | — | Path to the optimised judge artefact (e.g. `artifacts/optimized_judge.json`) |

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
