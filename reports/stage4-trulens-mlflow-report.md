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

**No TruLens UI in this project:** TruLens has its own Streamlit dashboard (`tru_session.run_dashboard()`) that displays per-record RAG Triad scores and a leaderboard. That dashboard is fed by `TruChain`-recorded runs — which this project does not use. Here, TruLens acts purely as a computation engine: scores are computed by its feedback functions and immediately forwarded to MLflow via `mlflow.log_feedback()`. The TruLens database stays empty and there is nothing to see in its dashboard. **MLflow is the UI for evaluation scores in this project.**

**What TruLens uniquely provides:** the most actionable diagnostic framework for identifying *which component to fix* — retriever, LLM, or prompt.

### MLflow — Experiment tracking and judge alignment

MLflow serves two roles. First, it logs TruLens scores as structured experiment runs, enabling comparison across queries, agents, and code versions in a dashboard. Second — as a future workstream — it supports judge alignment via DSPy: automatically optimising LLM judge prompts based on human feedback labels collected in Langfuse.

In this application, MLflow is instrumented in two ways:
- `mlflow.langchain.autolog()` captures full LangChain execution traces (LLM calls, tool calls, token usage, latency) automatically
- TruLens scores are logged both as run metrics (for charting) and as trace assessments via `mlflow.log_feedback()` (visible in the Quality tab)

The Quality tab shows three scorers across all experiment traces: `context_relevance`, `correctness`, and `safety`. `groundedness` and `relevance` are computed by TruLens and logged as run metrics (visible in the Charts tab), but do not appear as trace assessments in the Quality tab.

**What MLflow uniquely provides:** the only tool of the three that supports judge alignment from human feedback. The intended loop is: a human reviews traces in Langfuse and corrects a score (e.g. marks a `groundedness` of 1.0 as wrong because the response hallucinated); those labels are exported and used by DSPy to rewrite the LLM judge prompt so it makes fewer mistakes; future automated scores then better reflect human judgment. **This loop is not implemented in the current experiment** — the TruLens judge prompts are fixed. The infrastructure for it (MLflow experiment tracking, Langfuse human annotations) is in place; DSPy optimisation is descoped as a future workstream.

---

## Relationship between the tools

| | Langfuse | TruLens | MLflow |
|---|---|---|---|
| Role in this project | Human review and annotation | LLM-as-judge scoring engine | Experiment tracking and score display |
| How it integrates | `LangfuseCallbackHandler` in `agent.invoke()` | Post-hoc feedback functions called after each reply | `autolog()` globally + `mlflow.log_feedback()` per reply |
| Has a UI | Yes — [cloud.langfuse.com](https://cloud.langfuse.com) | Yes — Streamlit dashboard (not used here) | Yes — `mlflow ui` (local) |
| Captures agent traces | Yes | No | Yes |
| Computes quality scores | No | Yes — RAG Triad + correctness + safety | No |
| Displays quality scores | No | Yes — but not used in this experiment | Yes — Quality tab and Charts tab |
| Human annotation queues | Yes | No | No |
| Judge alignment (DSPy) | No | No | Future workstream |

None of the three is redundant. Langfuse is where humans review traces and add labels; TruLens is the computation engine that produces automated scores; MLflow is where those scores are stored, compared, and displayed.

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
                    │       └─► mlflow.log_feedback()     # context_relevance → Quality tab
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
| Clearly in KB (refund, warranty) | 4 | context_relevance, correctness, safety |
| Partially in KB (delayed orders, refund procedure) | 2 | context_relevance, correctness, safety |
| Not in KB (free shipping, payment, exchanges) | 3 | context_relevance, correctness, safety |
| Non-knowledge (order lookups, safety refusals) | 4 | safety only |

**MLflow Quality tab (illustrative results after one run):**

| Scorer | Total Count | Average Value |
|---|---|---|
| `context_relevance` | 9 | ~0.85 |
| `correctness` | 9 | ~0.75 |
| `safety` | 13 | ~0.92 |

**Observations:**

- `context_relevance` is lower on out-of-KB queries: the retriever returns loosely-related articles, so the retrieved context is a poor match for the question. This is the expected discriminating behaviour.
- `correctness` is lower on out-of-KB queries: the response diverges from the reference answer, either because the LLM hedges or because it speculates beyond the KB.
- `safety` is consistently high across all turns — the safety guardian correctly refuses injection and exfiltration attempts, and normal responses are policy-compliant. A drop here would signal a regression in the safety specialist.
- `groundedness` and `relevance` are computed by TruLens and visible as run metrics in the Charts tab, but are not surfaced as trace assessments in the Quality tab.
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

# View Langfuse traces and annotation queues
# Open https://cloud.langfuse.com in your browser
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
