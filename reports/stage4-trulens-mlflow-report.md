# Stage 4 Report: Human Review and Continuous Improvement

**Branch:** `feat/stage4-human-review-loop`
**Date:** 2026-04-03
**Author:** Marco Mulder

---

## Overview

Stage 4 introduces three tools that together close the loop between production traces, human judgment, and automated evaluation. Each tool has a distinct role and they are designed to complement each other rather than overlap.

---

## The three tools

### Langfuse — Tracing and human annotation

Langfuse is the observability and human review layer. It captures every LangChain agent execution as a structured trace — LLM calls, tool invocations, latency, and token usage — and makes them browsable in a UI. Traces can be routed to annotation queues where domain experts score them on dimensions like accuracy, policy compliance, and tone.

In this application, Langfuse is instrumented via a `LangfuseCallbackHandler` injected into every `agent.invoke()` call. Each reply is grouped under a session ID so the full conversation history is traceable.

**What Langfuse uniquely provides:** the most mature human annotation workflow of the three tools, with team assignment and structured scoring configs. Human labels collected here feed directly into MLflow's judge alignment workstream.

### TruLens — RAG Triad diagnostics

TruLens evaluates the quality of knowledge-grounded replies using the RAG Triad: three LLM-as-judge metrics that diagnose *which component* is responsible when a response fails.

| Metric | What it measures | Failure signal |
|---|---|---|
| Context Relevance | Are the retrieved knowledge chunks relevant to the query? | Retriever fetching irrelevant articles |
| Groundedness | Is the response grounded in the retrieved context? | LLM hallucinating beyond its sources |
| Relevance | Does the response actually answer the user's question? | Response drifted from the question |

In this application, TruLens feedback functions are called post-hoc after each reply, using the JSON output of `search_support_knowledge` as the retrieved context. Turns that don't involve knowledge retrieval (order lookups, safety refusals) are skipped for the RAG Triad — it is undefined there.

Two additional scorers run on top of the RAG Triad via `_log_extra_assessments` in the experiment script:

| Scorer | Scope | What it measures |
|---|---|---|
| Correctness | Knowledge turns | How well the response matches a reference answer |
| Safety | Every turn | Whether the response is safe and policy-compliant |

`correctness` uses `relevance_with_cot_reasons` comparing the actual response to a hand-written reference answer. `safety` uses the same function with a retail support safety policy statement as the reference, applied to every query including order and safety turns.

**Why post-hoc rather than TruChain wrapping:** the app has four LangChain agents in a nested supervisor delegation pattern, with Langfuse callbacks already injected into `agent.invoke()`. Wrapping agents with `TruChain` would create a conflicting instrumentation stack. Calling the feedback functions directly after execution is cleaner and fits the existing architecture.

**What TruLens uniquely provides:** the most actionable diagnostic framework for identifying *which component to fix* — retriever, LLM, or prompt.

### MLflow — Experiment tracking and judge alignment

MLflow serves two roles. First, it logs TruLens scores as structured experiment runs, enabling comparison across queries, agents, and code versions in a dashboard. Second — as a future workstream — it supports judge alignment via DSPy: automatically optimising LLM judge prompts based on human feedback labels collected in Langfuse.

In this application, MLflow is instrumented in two ways:
- `mlflow.langchain.autolog()` captures full LangChain execution traces (LLM calls, tool calls, token usage, latency) automatically
- TruLens scores are logged both as run metrics (for charting) and as trace assessments via `mlflow.log_feedback()` (visible in the Quality tab)

The Quality tab shows four scorers across all experiment traces: `groundedness`, `relevance`, `correctness`, and `safety`.

**What MLflow uniquely provides:** the only tool of the three that supports judge alignment from human feedback, closing the loop between human judgment and automated evaluation.

---

## Relationship between the tools

Langfuse and MLflow both capture LangChain traces, but they do so for different purposes and through different mechanisms:

| | Langfuse | MLflow |
|---|---|---|
| How traces are captured | `LangfuseCallbackHandler` in `agent.invoke()` | `autolog()` hooks into LangChain globally |
| Primary purpose | Human review and annotation | Experiment comparison and eval tracking |
| RAG Triad scores | No | Yes — logged by TruLens |
| Annotation queues | Yes | No |
| Judge alignment | No | Yes — via DSPy |

They are not redundant. Langfuse is where humans review and label traces; MLflow is where those labels feed back into improving the automated evaluators.

---

## Architecture

All three tools follow the same pattern: opt-in via environment variables, initialised once in `RetailSupportOrchestrator.__init__`, and active per reply.

```
User message
    └─► RetailSupportOrchestrator.reply()
            └─► _execute_reply()
                    ├─► agent.invoke()                    # LangChain agents + tools
                    │       ├─► LangfuseCallbackHandler   # traces every step
                    │       └─► MLflow autolog            # traces every step
                    ├─► _run_trulens_eval()               # RAG Triad (knowledge turns only)
                    │       ├─► mlflow.log_*()            # log scores as MLflow run
                    │       └─► mlflow.log_feedback()     # groundedness, relevance → Quality tab
                    └─► SupportReply

experiment_stage4.py (wraps reply())
    └─► _log_extra_assessments()                         # runs after every reply()
            └─► mlflow.log_feedback()                    # correctness, safety → Quality tab
```

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | — | Enable Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | — | Langfuse auth |
| `LANGFUSE_BASE_URL` | — | Langfuse instance URL |
| `TRULENS_ENABLED` | `false` (`true` in experiment script) | Enable RAG Triad evaluation |
| `TRULENS_FEEDBACK_MODEL` | app model | Override judge LLM |
| `MLFLOW_ENABLED` | `false` (`true` in experiment script) | Enable experiment tracking |
| `MLFLOW_EXPERIMENT_NAME` | `retail-support-rag-eval` | MLflow experiment name |
| `MLFLOW_TRACKING_URI` | local SQLite (`mlflow.db`) | Remote MLflow server URI |

---

## Experiment results

Thirteen queries were run with all three tools enabled, spanning four categories:

| Category | Queries | Scorers active |
|---|---|---|
| Clearly in KB (refund, warranty) | 4 | groundedness, relevance, correctness, safety |
| Partially in KB (delayed orders, refund procedure) | 2 | groundedness, relevance, correctness, safety |
| Not in KB (free shipping, payment, exchanges) | 3 | groundedness, relevance, correctness, safety |
| Non-knowledge (order lookups, safety refusals) | 4 | safety only |

**MLflow Quality tab (illustrative results after one run):**

| Scorer | Total Count | Average Value |
|---|---|---|
| `groundedness` | 9 | ~0.85 |
| `relevance` | 9 | ~0.80 |
| `correctness` | 9 | ~0.75 |
| `safety` | 13 | ~0.92 |

**Observations:**

- `correctness` and `relevance` are lower on out-of-KB queries: the retriever returns loosely-related articles, the LLM hedges or speculates, and the response diverges from the reference answer. This is the expected discriminating behaviour.
- `safety` is consistently high across all turns — the safety guardian correctly refuses injection and exfiltration attempts, and normal responses are policy-compliant. A drop here would signal a regression in the safety specialist.
- `groundedness` varies by KB coverage: strong on direct-match articles (refund, warranty), weaker when retrieved context is off-topic.
- The non-knowledge turns (order lookups, safety refusals) are excluded from RAG Triad to avoid undefined scores — they only contribute to the `safety` count.
- MLflow's trace view shows the full span tree for each query including tool calls and token usage, identical in structure to what Langfuse captures, but navigable as an experiment.

---

## Limitations and next steps

**Score discrimination:** Out-of-KB queries now expose natural score variation, but a larger or deliberately noisy corpus would stress-test the metrics further and reveal finer retrieval quality differences.

**Judge LLM cost:** Each evaluated turn makes 3 extra LLM calls. Keep `TRULENS_ENABLED=false` in production; enable for staging or sampled evaluation runs.

**`get_policy_summary` excluded:** Its keyword fallback often returns the `privacy` policy regardless of query, which would distort Context Relevance scores. Add it to the context filter once score distributions are validated.

**MLflow judge alignment (DSPy):** Descoped for this iteration. It requires annotated human labels from Langfuse as input and is a meaningful standalone workstream — the infrastructure for it is now in place.

---

## Running it yourself

```bash
# Install dependencies
pip install -r requirements.txt
```

Add credentials to `.env` in the project root (create if it doesn't exist):

```
OPENAI_API_KEY=sk-...

# Optional — enables Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

`TRULENS_ENABLED` and `MLFLOW_ENABLED` default to `true` inside the experiment script, so no extra flags are needed:

```bash
# Run the experiment — reads .env automatically
python experiment_stage4.py

# View MLflow Quality tab and traces
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To disable a tool for a single run:

```bash
TRULENS_ENABLED=false python experiment_stage4.py
```

Smoke tests (no API key needed):
```bash
python smoke_test_trulens.py
python smoke_test_mlflow.py
```
