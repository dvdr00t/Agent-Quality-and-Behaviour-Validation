# Stage 1 Evaluation Report

## Scope

This report documents the Stage 1 development-testing setup for the retail support agent using the three requested tools:

- DeepEval for regression-style quality gates
- Ragas for synthetic RAG-oriented dataset generation
- MLflow for experiment tracking and artifact logging

## Implemented files

| File | Purpose |
| --- | --- |
| `retail_support/stage1_eval.py` | Shared Stage 1 harness for curated cases, Ragas dataset generation, MLflow logging, and markdown report generation |
| `tests/test_stage1_deepeval.py` | DeepEval-backed pytest suite for curated development-time regression checks |
| `reports/stage1-evaluation-report.md` | Initial Stage 1 report describing the delivered setup and execution flow |

## Tool mapping

### DeepEval

The Stage 1 regression suite in `tests/test_stage1_deepeval.py` validates:

- answer quality with LLM-judge scoring
- faithfulness against curated context
- expected tool usage
- step-efficiency constraints
- forbidden-content leakage for safety cases

The suite consumes curated cases from `retail_support/stage1_eval.py`, so the same scenario catalog is shared across test execution and reporting.

### Ragas

The synthetic dataset workflow in `retail_support/stage1_eval.py`:

- converts the in-repo knowledge base and policies into LangChain documents
- attempts synthetic dataset generation with `ragas.testset.TestsetGenerator`
- persists the dataset under `artifacts/stage1/ragas_synthetic_dataset.json`
- falls back to curated seed records when Ragas is unavailable or not configured

This keeps Stage 1 operational even before all external dependencies are installed.

### MLflow

The MLflow integration in `retail_support/stage1_eval.py`:

- creates or reuses a local file-based tracking store under `mlruns/`
- logs run metadata, tags, params, and metrics
- logs the curated result artifact, synthetic dataset artifact, and markdown report
- stores a prompt version tag for comparison across runs

## Curated Stage 1 cases

The delivered suite covers six development-time cases:

1. refund policy retrieval
2. order status lookup
3. eligible refund decision
4. delayed order escalation
5. final-sale refund denial
6. prompt-injection refusal

These cases are designed to exercise the Stage 1 goals described in your architecture:

- RAG-style retrieval grounding
- tool-correctness for order operations
- step-efficiency controls
- safety handling for prompt injection and unauthorized data requests

## Expected artifacts after execution

Running the Stage 1 workflow will produce these artifacts:

- `artifacts/stage1/stage1_curated_results.json`
- `artifacts/stage1/ragas_synthetic_dataset.json`
- `reports/stage1-evaluation-report.md`
- `mlruns/` entries for MLflow experiments

## Execution commands

```bash
pytest tests/test_stage1_deepeval.py -q
python -m retail_support.stage1_eval
```

## Notes

This report is the initial generated markdown artifact for Stage 1. When `python -m retail_support.stage1_eval` is executed, the same file path is overwritten with a run-specific report containing observed pass/fail status, Ragas dataset details, and MLflow run metadata.
