# Stage 1 Empirical Report: DeepEval, Ragas, and MLflow

## Purpose

This report documents how the Stage 1 tool stack — DeepEval, Ragas, and MLflow — can be used together for development-time evaluation of a production-style support agent, and grounds that recommendation in empirical evidence gathered from this repository.

It is written to support the broader deliverable on the agent evaluation landscape: no single framework covers the full evaluation lifecycle, so the practical question is not which tool is best in absolute terms, but which combination is best suited for a specific stage of the lifecycle. For Stage 1, the development-testing stage, the empirical evidence from this project supports the use of DeepEval + Ragas + MLflow as a coherent stack.

## Repository implementation used as evidence

The Stage 1 stack is implemented in the following project assets:

- [`retail_support/stage1_eval.py`](retail_support/stage1_eval.py) — orchestration of curated cases, synthetic dataset generation, artifact persistence, MLflow logging, and markdown report generation
- [`run_stage1_workflow()`](retail_support/stage1_eval.py:311) — end-to-end Stage 1 execution entrypoint
- [`build_stage1_report()`](retail_support/stage1_eval.py:392) — markdown report builder for Stage 1 outputs
- [`tests/test_stage1_deepeval.py`](tests/test_stage1_deepeval.py) — DeepEval-backed regression suite for the curated scenarios
- [`retail_support/config.py`](retail_support/config.py) — environment loading and provider selection, including Azure-compatible execution paths
- [`reports/stage1-evaluation-report.md`](reports/stage1-evaluation-report.md) — generated markdown report from the full Stage 1 workflow
- [`artifacts/stage1/stage1_curated_results.json`](artifacts/stage1/stage1_curated_results.json) — machine-readable curated evaluation output
- [`artifacts/stage1/ragas_synthetic_dataset.json`](artifacts/stage1/ragas_synthetic_dataset.json) — synthetic or fallback Stage 1 dataset artifact

## Why these three tools belong together in Stage 1

### DeepEval: enforcement layer for development testing

DeepEval is the enforcement layer of the Stage 1 stack. In this repository, it is used inside [`tests/test_stage1_deepeval.py`](tests/test_stage1_deepeval.py) to run curated agent scenarios as pytest tests. This means evaluation becomes part of normal engineering workflow rather than a separate analysis activity.

What this gives the team:

- pass/fail regression gates in test execution
- tool-usage checks over the support agent
- response-quality and grounding checks through evaluation metrics
- direct compatibility with CI-oriented execution patterns

Empirical evidence from this repository:

- The DeepEval suite was executed successfully with [`pytest`](tests/test_stage1_deepeval.py:1) against [`tests/test_stage1_deepeval.py`](tests/test_stage1_deepeval.py).
- Final observed result: 12 passed, 1 warning.
- Before stabilizing the setup, the same suite exposed concrete failures:
  - missing local package resolution before [`tests/conftest.py`](tests/conftest.py) was added
  - incorrect OpenAI-key assumptions before Azure-aware config loading was added in [`SupportSettings.from_env()`](retail_support/config.py:31)
  - safety-case instability caused by Azure content filtering on overly jailbreak-like prompts

Interpretation:

DeepEval did not merely provide a theoretical testing interface; it actively enforced a quality gate and surfaced integration issues that had to be fixed before the suite became green. This is strong empirical support for positioning DeepEval as the Stage 1 development-testing backbone.

### Ragas: dataset expansion layer for RAG-oriented validation

Ragas is the dataset expansion layer of the Stage 1 stack. In this repository, the Ragas-related path is implemented in [`retail_support/stage1_eval.py`](retail_support/stage1_eval.py), where the in-memory knowledge base and policy assets are converted into documents and then passed into synthetic dataset generation.

What this gives the team:

- synthetic test case generation from support knowledge and policies
- broader coverage than purely hand-authored regression scenarios
- a scalable path for expanding retrieval-oriented evaluation data

Empirical evidence from this repository:

- Running the full workflow through [`run_stage1_workflow()`](retail_support/stage1_eval.py:311) generated the dataset artifact at [`artifacts/stage1/ragas_synthetic_dataset.json`](artifacts/stage1/ragas_synthetic_dataset.json).
- The latest run produced 6 records.
- The generation mode recorded in [`reports/stage1-evaluation-report.md`](reports/stage1-evaluation-report.md) was fallback rather than native synthetic generation.
- The recorded reason was an API-surface mismatch: the installed Ragas version rejected the generation call with the message that [`TestsetGenerator.generate()`](retail_support/stage1_eval.py:1) received an unexpected `documents` argument.

Interpretation:

This is useful empirical evidence in two directions. First, it confirms the strategic role of Ragas: the repository has a dedicated synthetic-dataset stage and persists its outputs as artifacts. Second, it exposes a realistic operational limitation: Ragas integration is version-sensitive, and dataset generation may require adaptation across releases. That supports a balanced conclusion in the comparative analysis — Ragas is the right Stage 1 choice for RAG-heavy systems, but it is not frictionless in practice.

### MLflow: experiment and artifact backbone

MLflow is the tracking and aggregation layer of the Stage 1 stack. In this repository, the full Stage 1 workflow logs results, metadata, and artifacts to an MLflow experiment after the evaluation run completes.

What this gives the team:

- durable run history
- run-level comparison across prompt or model changes
- artifact storage for curated results, datasets, and reports
- a foundation for comparing repeated evaluation runs over time

Empirical evidence from this repository:

- Executing the Stage 1 workflow created or reused the experiment named retail-support-stage1.
- The latest recorded MLflow run id was 392e80a6a5a3436a9b569b17af7f9f86.
- The workflow logged the curated results artifact from [`artifacts/stage1/stage1_curated_results.json`](artifacts/stage1/stage1_curated_results.json).
- The workflow logged the dataset artifact from [`artifacts/stage1/ragas_synthetic_dataset.json`](artifacts/stage1/ragas_synthetic_dataset.json).
- The workflow logged the markdown report from [`reports/stage1-evaluation-report.md`](reports/stage1-evaluation-report.md).

Interpretation:

MLflow proved valuable not because it judged the agent directly, but because it made the outputs of the other two tools traceable and comparable. This validates the architectural claim that MLflow is the right unifying layer for Stage 1 rather than the primary evaluator.

## Empirical findings from the latest Stage 1 workflow run

The latest full Stage 1 workflow run generated the report at [`reports/stage1-evaluation-report.md`](reports/stage1-evaluation-report.md) and the detailed results artifact at [`artifacts/stage1/stage1_curated_results.json`](artifacts/stage1/stage1_curated_results.json).

### Aggregate outcome

- Curated workflow pass rate: 4 out of 6 cases, equal to 67%
- DeepEval regression suite result: 12 passed
- Ragas dataset output: 6 records in fallback mode
- MLflow logging: successful

### Failure modes surfaced by the workflow

The workflow-level report identified two concrete issues in the latest run:

1. The eligible-refund path failed step-efficiency expectations because the agent used more tools than required.
2. The prompt-injection refusal path failed the expected-tool check because the agent refused correctly but used [`get_policy_summary`](retail_support/runtime.py:197) rather than the expected risk-assessment path.

These findings are visible in [`reports/stage1-evaluation-report.md`](reports/stage1-evaluation-report.md) and in [`artifacts/stage1/stage1_curated_results.json`](artifacts/stage1/stage1_curated_results.json).

Interpretation:

This is the most important empirical argument for the three-tool combination. The stack does not just say whether the agent works; it reveals different kinds of evidence:

- DeepEval shows whether the development regression gate is passing
- Ragas expands the evaluation surface beyond a tiny hand-authored set
- MLflow records the exact run so discrepancies and regressions are not anecdotal

Notably, the repository now shows both a green pytest gate and a workflow report with partial failures. That is not a contradiction; it is evidence of the variability and complexity of LLM-agent behavior across runs. This strengthens, rather than weakens, the case for MLflow-backed experiment tracking and repeated evaluation.

## Comparative conclusion for the broader landscape report

Within the wider evaluation landscape, the empirical results from this repository support the following positioning for Stage 1:

| Tool | Stage 1 role | Empirical support in this repository | Practical conclusion |
| --- | --- | --- | --- |
| DeepEval | CI-style regression enforcement | The repository test suite in [`tests/test_stage1_deepeval.py`](tests/test_stage1_deepeval.py) reached 12 passing tests after surfacing real integration failures | Best fit for development-time quality gates |
| Ragas | Synthetic dataset generation for retrieval-heavy agents | The repository generated and stored a dataset artifact, while also surfacing real version-friction in the API | Best fit for expanding RAG-oriented evaluation coverage, with integration caveats |
| MLflow | Experiment tracking and artifact aggregation | The repository created an experiment and logged artifacts and run metadata | Best fit as the Stage 1 backbone for traceability and comparison |

## Gap assessment

The empirical evidence also supports a boundary around what this Stage 1 stack does not cover.

This combination is strong for development testing, but it does not yet provide:

- adversarial red-teaming at the breadth associated with promptfoo
- sandboxed benchmark execution comparable to Inspect AI
- production tracing and annotation workflows comparable to Langfuse
- embedding drift analysis comparable to Phoenix
- statistical production monitoring comparable to Evidently

So the correct conclusion for the broader deliverable is not that DeepEval, Ragas, and MLflow are a complete evaluation architecture. The conclusion is that they are a strong, evidence-backed Stage 1 stack inside a larger multi-stage evaluation architecture.

## Recommended wording for the final comparative analysis

A defensible recommendation, grounded in the repository evidence, is the following:

DeepEval, Ragas, and MLflow form a credible Stage 1 development-testing stack for production-grade agents. DeepEval operationalizes evaluation as an enforceable regression gate inside the engineering workflow. Ragas expands coverage by generating or structuring retrieval-oriented evaluation datasets from the system knowledge base. MLflow provides the experimental memory needed to compare runs, track artifacts, and understand whether behavior is improving or drifting across prompt and model changes. Empirically, this repository shows all three roles in practice: a passing DeepEval regression suite, persisted evaluation artifacts, a successful MLflow run, and a workflow report that surfaced non-trivial agent weaknesses in tool efficiency and safety-tool routing.

## Bottom line

For the objective of the broader deliverable — building a practical foundation for evaluating production-grade AI agents — the empirical evidence from this repository supports the use of DeepEval + Ragas + MLflow as the recommended Stage 1 stack, with the explicit understanding that later lifecycle stages still require specialized tools for red-teaming, observability, human review, and safety assurance.
