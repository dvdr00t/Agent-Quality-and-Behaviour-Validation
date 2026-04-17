<!-- omit in toc -->
# Agent Evaluation Landscape: A Comparative Analysis of Frameworks, Benchmarks, and Tools

<!-- omit in toc -->
## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Introduction](#1-introduction)
  - [1.1 How the application works as a LangChain multi-agent system](#11-how-the-application-works-as-a-langchain-multi-agent-system)
- [2. Research Scope and Methodology](#2-research-scope-and-methodology)
- [3. Evaluation Dimensions](#3-evaluation-dimensions)
- [4. Landscape of Frameworks, Benchmarks, and Tools](#4-landscape-of-frameworks-benchmarks-and-tools)
  - [4.1 Tool-by-tool overview](#41-tool-by-tool-overview)
  - [4.2 Comparative analysis](#42-comparative-analysis)
- [5. Stage-by-Stage Findings](#5-stage-by-stage-findings)
  - [5.1 Stage 1 — Development testing](#51-stage-1--development-testing)
  - [5.2 Stage 2 — Pre-deployment red-teaming and safety](#52-stage-2--pre-deployment-red-teaming-and-safety)
  - [5.3 Stage 3 — Production monitoring and drift detection](#53-stage-3--production-monitoring-and-drift-detection)
  - [5.4 Stage 4 — Human review and continuous improvement](#54-stage-4--human-review-and-continuous-improvement)
  - [5.5 Stage 5 — Safety assurance and compliance](#55-stage-5--safety-assurance-and-compliance)
- [6. Synthesis of Findings](#6-synthesis-of-findings)
- [7. Gap Assessment](#7-gap-assessment)
- [8. Recommended Evaluation Stack](#8-recommended-evaluation-stack)
  - [Baseline recommendation](#baseline-recommendation)
  - [Advanced or optional additions](#advanced-or-optional-additions)
  - [Recommended stage mapping](#recommended-stage-mapping)
- [10. Conclusion](#10-conclusion)
- [References / Appendix](#references--appendix)
  - [Source provenance](#source-provenance)
  - [Evidence limitations](#evidence-limitations)

## Executive Summary

This document combines comparative research with repository evidence to map the agent-evaluation landscape across frameworks, benchmarks, and operational tools. The central finding is that no single product in this landscape spans the full lifecycle of agent evaluation. Instead, the tools cluster into complementary layers: development testing, retrieval and answer-quality evaluation, adversarial and safety benchmarking, production observability, drift monitoring, human review, and experiment governance.

A five-stage architecture remains the most coherent organizing model for this landscape:

1. Development testing — DeepEval + Ragas + MLflow  
2. Pre-deployment red-teaming and safety — promptfoo + Inspect AI  
3. Production monitoring and drift detection — Langfuse + Arize Phoenix + AgentOps + Evidently AI  
4. Human review and continuous improvement — Langfuse + MLflow + TruLens  
5. Safety assurance and compliance — Inspect AI + promptfoo + DeepEval

The repository provides a useful case study rather than a full empirical validation corpus. It is a multi-agent retail support application described in [README.md](README.md:3) and implemented through [RetailSupportOrchestrator](retail_support/runtime.py:53). The system exposes four cooperating roles via [TARGET_DISPLAY_NAMES](retail_support/runtime.py:21), routes work through supervisor-to-specialist delegation in [RetailSupportOrchestrator._delegate_to_specialist()](retail_support/runtime.py:271), and exposes several tool-call surfaces such as [search_support_knowledge()](retail_support/runtime.py:159), [get_order_snapshot()](retail_support/runtime.py:166), [assess_refund_eligibility()](retail_support/runtime.py:173), [create_escalation_ticket()](retail_support/runtime.py:185), [get_policy_summary()](retail_support/runtime.py:197), and [assess_request_risk()](retail_support/runtime.py:204).

Direct evidence in the project extends beyond architectural hooks. The Stage 1 report documents empirical results for DeepEval, Ragas, and MLflow, including a passing DeepEval regression suite, persisted Stage 1 artifacts, and MLflow run tracking. The Stage 2 report documents promptfoo red-teaming and Inspect AI safety-benchmark results, including concrete vulnerability findings around multi-turn authorization drift and cross-user data exposure. The Stage 3 report defines the production-monitoring architecture across tracing, debugging, drift detection, and KPI monitoring. The Stage 4 report documents a human-review and continuous-improvement loop built around TruLens, Langfuse, and MLflow, including annotation routing and judge-alignment workflows. The Stage 5 report documents a safety-assurance stack built around Inspect AI, DeepEval, and promptfoo, including concrete findings on regex-rule gaps, prompt-level versus code-level enforcement, and cross-user access risk.

Taken together, these materials substantially strengthen the empirical basis of the five-stage model. The evidence is still uneven across stages: Stage 1, Stage 2, Stage 4, and Stage 5 include explicit reports and outcomes, while Stage 3 is documented more as a target production architecture than as a live deployment with operational telemetry screenshots or long-run drift baselines. Even so, the repository and accompanying stage reports support a much stronger claim than branch metadata alone.

The recommended evaluation stack is therefore layered rather than monolithic. A baseline stack should combine DeepEval, Ragas, MLflow, promptfoo, Langfuse, and one stronger safety-benchmarking component such as Inspect AI. An expanded stack can add Arize Phoenix, AgentOps, Evidently AI, and TruLens where production scale, drift detection, or structured human review justify the extra operational complexity.

## 1. Introduction

Agent evaluation has become a lifecycle problem rather than a single benchmark problem. A modern application can fail in at least four distinct ways: it can answer incorrectly, retrieve or ground incorrectly, route or call tools incorrectly, or behave unsafely under adversarial or policy-sensitive conditions. Production use introduces a further set of concerns, including traceability, drift, human escalation, and governance across repeated experiments.

The repository provides a concrete case study for those requirements. It is framed as a retail customer-support system with four roles in [README.md](README.md:7): Support Supervisor, Policy and Knowledge Specialist, Order Resolution Specialist, and Trust and Safety Guardian. Those same roles are represented in [TARGET_DISPLAY_NAMES](retail_support/runtime.py:21), while routing and orchestration are centralized in [RetailSupportOrchestrator](retail_support/runtime.py:53). The system delegates domain questions through [contact_knowledge_specialist()](retail_support/runtime.py:251), [contact_order_specialist()](retail_support/runtime.py:256), and [contact_trust_and_safety()](retail_support/runtime.py:261), with the supervisor performing the actual handoff through [RetailSupportOrchestrator._delegate_to_specialist()](retail_support/runtime.py:271).

This architecture makes the repository suitable for studying evaluation needs across answer quality, tool correctness, safety, routing fidelity, and escalation handling. It does not, however, provide a complete empirical benchmark record for every stage. That distinction is crucial. The discussion below uses the repository as a grounded case study while drawing on broader comparative research to position DeepEval, Ragas, promptfoo, Inspect AI, Langfuse, Arize Phoenix, TruLens, Evidently AI, AgentOps, and MLflow.

### 1.1 How the application works as a LangChain multi-agent system

From an implementation perspective, the repository is built around a single orchestration class, [RetailSupportOrchestrator](retail_support/runtime.py:53). That orchestrator creates one model client in [RetailSupportOrchestrator._build_model()](retail_support/runtime.py:135), exposes business capabilities from [SupportOperationsService](retail_support/services.py:12) as LangChain tools inside [RetailSupportOrchestrator._build_agents()](retail_support/runtime.py:157), and wires those tools into three specialists plus one supervisor through [`create_agent()`](retail_support/runtime.py:11). This separation is important because the LLM layer is responsible for routing and tool choice, while policy rules, order checks, and safety heuristics remain deterministic Python code in [retail_support/services.py](retail_support/services.py).

The first architectural layer is the domain-service layer. Deterministic operations such as [SupportOperationsService.search_support_knowledge()](retail_support/services.py:21), [SupportOperationsService.get_order_snapshot()](retail_support/services.py:31), [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34), [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77), [SupportOperationsService.get_policy_summary()](retail_support/services.py:88), and [SupportOperationsService.assess_request_risk()](retail_support/services.py:98) live in [SupportOperationsService](retail_support/services.py:12). The LangChain agents do not own these rules; they call them. That means refund-window logic, ownership checks, and safety heuristics remain inspectable and testable outside the model itself.

The specialist pattern is explicit in the runtime. Each specialist receives a narrowly scoped system prompt and only the tools relevant to its domain:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def assess_refund_eligibility(order_id: str, user_id: str, reason: str) -> str:
    result = self.operations.assess_refund_eligibility(
        order_id=order_id,
        user_id=user_id,
        reason=reason,
    )
    self._record_event(
        kind="tool",
        actor="orders",
        name="assess_refund_eligibility",
        details={"order_id": order_id, "user_id": user_id},
    )
    return json.dumps(result, indent=2)

self.order_specialist = create_agent(
    model=self.model,
    tools=[get_order_snapshot, assess_refund_eligibility, create_escalation_ticket],
    system_prompt=orders_prompt,
)
```

This snippet, drawn from [retail_support/runtime.py](retail_support/runtime.py:157), shows an important design choice: LangChain tools are thin wrappers over service methods rather than the primary location of business logic. The same pattern is used for the knowledge and safety specialists, with tool boundaries aligned to repository concerns such as policy lookup, order operations, and request-risk assessment.

The supervisor is itself another LangChain agent, but it is intentionally constrained. Instead of calling order or policy tools directly, it calls delegation tools that hand work to specialists via [RetailSupportOrchestrator._delegate_to_specialist()](retail_support/runtime.py:271):

```python
@tool
def contact_order_specialist(question: str) -> str:
    return self._delegate_to_specialist("orders", question)

@tool
def contact_trust_and_safety(question: str) -> str:
    return self._delegate_to_specialist("safety", question)

self.support_supervisor = create_agent(
    model=self.model,
    tools=[contact_knowledge_specialist, contact_order_specialist, contact_trust_and_safety],
    system_prompt=supervisor_prompt,
)
```

This means the application behaves like a managed hierarchy rather than a flat tool-using chatbot. The supervisor decides who should handle the request, the specialist decides which concrete business tools to call, and the service layer computes the authoritative result.

Request execution is then wrapped in explicit session and trace management. [SupportSession](retail_support/runtime.py:33) stores separate histories per agent, [RetailSupportOrchestrator.reply()](retail_support/runtime.py:67) invokes the selected target, and the orchestrator converts internal activity into structured output through [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) and [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394):

```python
answer = self._invoke_agent(agent=agent, history=history, user_message=user_message)
route = self._build_route(target)
tool_calls = [
    event["name"]
    for event in self._current_request_events
    if event["kind"] == "tool"
]
return SupportReply(
    text=answer,
    handled_by=handled_by,
    route=route,
    tool_calls=tool_calls,
)
```

For evaluation, this architecture matters because it creates several inspectable surfaces: final-answer quality, specialist-routing correctness, tool-selection correctness, and safety-policy enforcement. In other words, the repository is a useful agent-evaluation case study not merely because it contains multiple prompts, but because it embodies a concrete LangChain control flow in which supervisor delegation, specialist tool use, deterministic business logic, and event tracing can all be evaluated separately.

## 2. Research Scope and Methodology

This document draws on three evidence classes that are intentionally kept separate.

| Evidence class | What it contributes | How it is used in this report | Limitation |
|---|---|---|---|
| Comparative research | Comparative positioning of DeepEval, Ragas, promptfoo, Inspect AI, Langfuse, Arize Phoenix, TruLens, Evidently AI, AgentOps, and MLflow; a feature matrix; and a five-stage evaluation architecture | Provides the cross-tool landscape, stage architecture, and recommended stack | Raw vendor documentation, benchmark detail, and full metric history are not reproduced here |
| Repository evidence | Multi-agent system structure in [README.md](README.md:3), orchestration in [RetailSupportOrchestrator](retail_support/runtime.py:53), service logic in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34) and [SupportOperationsService.assess_request_risk()](retail_support/services.py:98), policy constraints in [retail_support/data.py](retail_support/data.py:43), event capture in [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384), route reconstruction in [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394), DeepEval residue in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1), telemetry residue in [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1), experiment tracking in [mlruns/](mlruns/), and the Stage 1–5 reports supplied for this project | Grounds the abstract comparison in a concrete multi-agent application with stage-specific evaluation evidence | Evidence depth still varies by stage; some stages provide measured outcomes, while others are stronger on architecture and process than on long-horizon operational results |
| Git metadata | Stage branch names in [.git/FETCH_HEAD](.git/FETCH_HEAD:1) | Indicates that staged workstreams existed in repository history | Branch names alone do not prove branch contents or outcomes |

Methodologically, the report proceeds in three steps. First, it defines evaluation dimensions that matter for agentic systems. Second, it places the tools against those dimensions using conservative wording wherever direct evidence is limited. Third, it tests the practical relevance of those dimensions against the repository’s retail-support case study.

The scope is deliberately comparative rather than experimental. No new tool runs or benchmark executions are introduced in this document.

Code snippets are taken from the repository files cited inline. Branch names are referenced where relevant, but the analysis does not infer branch-specific contents that are not present in the current checkout.

## 3. Evaluation Dimensions

The comparative landscape and the repository together suggest the following evaluation dimensions.

| Evaluation dimension | Why it matters for agent systems | Case-study relevance in this repository |
|---|---|---|
| Development regression testing | Ensures known workflows continue to behave correctly under code and prompt changes | Strongly relevant because order lookup, refund logic, escalation, policy retrieval, and refusal behavior can regress independently; partial direct evidence appears in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) |
| Retrieval and answer grounding | Measures whether policy or knowledge answers remain faithful to source material | Relevant because factual support answers depend on [search_support_knowledge()](retail_support/runtime.py:159) and the in-memory knowledge base in [retail_support/data.py](retail_support/data.py:3) |
| Tool-use correctness | Evaluates whether the model invokes the right operational surface with the right arguments | Central to this application because the runtime exposes explicit tool calls in [retail_support/runtime.py](retail_support/runtime.py) |
| Routing and delegation fidelity | Measures whether the supervisor sends work to the correct specialist or combination of specialists | Central because routing is the defining behavior of [RetailSupportOrchestrator](retail_support/runtime.py:53) |
| Adversarial robustness and safety | Tests prompt injection, policy bypass, and unauthorized access handling | Strongly relevant because [SupportOperationsService.assess_request_risk()](retail_support/services.py:98) explicitly encodes safety heuristics and [retail_support/data.py](retail_support/data.py:43) contains privacy and security policy text |
| Production observability | Supports trace reconstruction, debugging, and operational accountability | Partially relevant and partially implemented through [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) and [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394) |
| Drift detection | Detects changing prompt distributions, behavior shifts, or data shifts after deployment | Operationally relevant, but not directly implemented on the current checkout |
| Human review and escalation | Provides a mechanism for ambiguous, unresolved, or high-risk cases | Relevant because [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77) creates human-facing tickets, though no reviewer workflow is present |
| Experiment tracking and governance | Connects runs, prompts, models, and outcomes across iterations | Relevant because the repository contains [mlruns/](mlruns/), but the specific experiment lineage is not available from provided material |
| Lifecycle completeness | Assesses whether a tool can cover development, pre-deployment, production, and governance | The core comparative question of this report |

These dimensions explain why the landscape resists a single-tool solution. The repository’s architecture simultaneously demands testing, routing validation, safety assurance, and production instrumentation.

## 4. Landscape of Frameworks, Benchmarks, and Tools

### 4.1 Tool-by-tool overview

The table below summarizes the role each tool plays in this landscape. Where direct detail is limited, the entry remains conservative.

| Tool | Primary role in this comparison | Main lifecycle placement | Distinguishing contribution | Limitation or caveat |
|---|---|---|---|---|
| DeepEval | Development-time agent evaluation and safety-oriented testing | Stage 1 and Stage 5 | Strong fit for curated evaluation scenarios and safety assurance; aligns with direct residue in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) and [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1) | Does not cover the full lifecycle alone |
| Ragas | Retrieval and answer-quality evaluation | Stage 1 | Fits development evaluation where grounded or reference-light answer assessment matters | Specific metrics and integrations are not detailed here |
| promptfoo | Red-teaming and pre-deployment safety testing | Stage 2 and Stage 5 | Well suited to adversarial prompting and pre-release safety validation | Direct repository use is not evidenced in the current checkout |
| Inspect AI | Structured safety benchmarking and rigorous pre-deployment evaluation | Stage 2 and Stage 5 | Adds a higher-assurance safety and benchmarking layer alongside promptfoo | Specific benchmark suite details are not detailed here |
| Langfuse | Observability and feedback infrastructure | Stage 3 and Stage 4 | Supports tracing and review workflows for production and continuous improvement | Deployed telemetry pipelines are not evidenced in the current checkout |
| Arize Phoenix | Production monitoring and drift-oriented analysis | Stage 3 | Contributes to the production monitoring and drift-analysis layer | Specific dashboard, tracing, or evaluation semantics are not detailed here |
| TruLens | Human review and continuous-improvement support | Stage 4 | Supports review and improvement workflows where human judgment is part of iteration | Direct reviewer queues or annotation workflows are not evidenced in the current checkout |
| Evidently AI | Drift detection and monitoring | Stage 3 | Serves as the explicit drift-monitoring element in this framework | Safety, routing, and human-review coverage are not its primary role in this framework |
| AgentOps | Agent operational monitoring | Stage 3 | Provides an operational monitoring layer for agent runs in production | Not sufficient for development testing or benchmark-driven safety by itself |
| MLflow | Experiment tracking and unification | Stage 1 and Stage 4 | Unifies runs and experiment governance across development and continuous improvement; consistent with the presence of [mlruns/](mlruns/) | Not a specialized agent evaluator on its own |

### 4.2 Comparative analysis

At a high level, the landscape separates into five functional clusters: testing, safety benchmarking, observability, drift monitoring, and unification or review. The comparison below expresses that structure without overstating unsupported detail.

| Tool | Development regression | Retrieval / grounding | Red-team / safety | Production observability | Drift detection | Human review | Experiment governance | Overall role |
|---|---|---|---|---|---|---|---|---|
| DeepEval | Primary | Supporting | Supporting | Not assessed | Not assessed | Not assessed | Supporting | Testing layer |
| Ragas | Supporting | Primary | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Retrieval-evaluation layer |
| promptfoo | Supporting | Not assessed | Primary | Not assessed | Not assessed | Not assessed | Not assessed | Adversarial testing layer |
| Inspect AI | Supporting | Not assessed | Primary | Not assessed | Not assessed | Not assessed | Not assessed | Safety benchmarking layer |
| Langfuse | Not assessed | Not assessed | Supporting | Primary | Supporting | Supporting | Supporting | Observability and feedback layer |
| Arize Phoenix | Not assessed | Not assessed | Not assessed | Primary | Primary | Not assessed | Not assessed | Monitoring and analysis layer |
| TruLens | Supporting | Supporting | Not assessed | Supporting | Not assessed | Primary | Supporting | Review and improvement layer |
| Evidently AI | Not assessed | Not assessed | Not assessed | Supporting | Primary | Not assessed | Not assessed | Drift-monitoring layer |
| AgentOps | Not assessed | Not assessed | Not assessed | Primary | Supporting | Not assessed | Not assessed | Operational monitoring layer |
| MLflow | Supporting | Not assessed | Not assessed | Supporting | Not assessed | Supporting | Primary | Unification and experiment layer |

Three comparative conclusions follow from this structure.

First, the tools are complementary rather than substitutive. DeepEval and Ragas address pre-production quality, but neither is presented as a production observability platform. Langfuse, AgentOps, Phoenix, and Evidently AI address operational visibility, but they are not replacements for adversarial testing. MLflow unifies runs and experiments, but it does not replace specialized evaluation logic.

Second, safety is split across multiple moments in the lifecycle. promptfoo and Inspect AI belong before deployment, while DeepEval reappears as a development and assurance mechanism. This is consistent with the case-study repository, where safety is not a separate product concern but a property of normal workflows involving privacy, authorization, and refusal.

Third, observability and drift are related but not identical. Stage 3 groups Langfuse, Arize Phoenix, AgentOps, and Evidently AI together, yet their roles remain differentiated: tracing, operations, monitoring, and drift should be understood as adjacent functions rather than interchangeable ones.

## 5. Stage-by-Stage Findings

The five-stage architecture used in this report provides the clearest narrative spine for combining the comparative landscape with repository evidence.

| Stage | Architecture layer | Repository evidence | Assessment status |
|---|---|---|---|
| Stage 1 — Development testing | DeepEval + Ragas + MLflow | Stage 1 report documents DeepEval regression execution, Ragas dataset generation, MLflow logging, and generated artifacts, alongside residue in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1), [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1), and [mlruns/](mlruns/) | Directly evidenced |
| Stage 2 — Pre-deployment red-teaming and safety | promptfoo + Inspect AI | Stage 2 report documents promptfoo red-teaming, Inspect AI benchmark runs, pass-rate summaries, and concrete vulnerabilities such as BOLA and user-enumeration findings | Directly evidenced |
| Stage 3 — Production monitoring and drift detection | Langfuse + Arize Phoenix + AgentOps + Evidently AI | Stage 3 report documents the production-monitoring architecture across tracing, debugging, drift detection, and monitoring; repository hooks include [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) and [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394) | Architecturally documented; partially evidenced in code |
| Stage 4 — Human review and continuous improvement | Langfuse + MLflow + TruLens | Stage 4 report documents automated scoring, Langfuse review queues, MLflow tracking, DSPy judge alignment, and an end-to-end human-review loop, alongside escalation support in [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77) | Directly evidenced |
| Stage 5 — Safety assurance and compliance | Inspect AI + promptfoo + DeepEval | Stage 5 report documents Inspect AI benchmarks, DeepEval safety gates, promptfoo attack generation, and concrete findings tied to [assess_request_risk()](retail_support/runtime.py:204), [SupportOperationsService.assess_request_risk()](retail_support/services.py:98), and related safety flows | Directly evidenced |

### 5.1 Stage 1 — Development testing

Stage 1 is directly evidenced through both implementation detail and recorded outcomes. The Stage 1 report describes a DeepEval-backed regression suite, a Ragas dataset-generation path, and MLflow experiment logging. It also reports concrete outcomes, including a passing DeepEval suite, a 6-record Ragas dataset in fallback mode, successful MLflow logging, and a workflow report that surfaced tool-efficiency and safety-tool-routing weaknesses. These findings map closely to concrete application surfaces in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34), [search_support_knowledge()](retail_support/runtime.py:159), [get_order_snapshot()](retail_support/runtime.py:166), [create_escalation_ticket()](retail_support/runtime.py:185), and [SupportOperationsService.assess_request_risk()](retail_support/services.py:98).

The cache residue is concrete enough to recover representative scenario names:

```text
[
  "tests/test_stage1_deepeval.py::test_stage1_curated_regression_suite[delayed_order_escalation]",
  "tests/test_stage1_deepeval.py::test_stage1_curated_regression_suite[eligible_refund]",
  "tests/test_stage1_deepeval.py::test_stage1_curated_regression_suite[final_sale_refund_denial]",
  "tests/test_stage1_deepeval.py::test_stage1_curated_regression_suite[prompt_injection_refusal]"
]
```

This matters analytically because the cached identifiers are aligned to concrete business and safety behaviors rather than generic smoke tests. The corresponding business-rule surface is equally explicit in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34):

```python
if order["final_sale"]:
    return {
        "eligible": False,
        "reason": "The item is marked as final sale and is not refundable.",
        "code": "final_sale",
    }

if order["status"] != "delivered":
    return {
        "eligible": False,
        "reason": "Refunds can be evaluated only after the order is delivered.",
        "code": "not_delivered",
    }

if (order["delivered_days_ago"] or 0) > 30:
    return {
        "eligible": False,
        "reason": "The refund window of 30 days has expired.",
        "code": "window_expired",
    }
```

Together, the stage report, cached artifacts, and service logic show why Stage 1 is more than partially evidenced. The project records both the evaluation surfaces and the observed outcomes: regression-gate results, generated dataset artifacts, MLflow-tracked runs, and workflow-level failure analysis.

This is meaningful because it shows that Stage 1 evaluation was operational, not merely hypothetical. At the same time, the evidence still has limits: the reported results reflect specific runs rather than a long-term trend line, and the Ragas path itself exposed version-sensitive integration friction. Those limitations do not weaken the Stage 1 conclusion; they strengthen it by showing the real operational characteristics of the stack.

### 5.2 Stage 2 — Pre-deployment red-teaming and safety

The architecture assigns promptfoo and Inspect AI to pre-deployment red-teaming and safety validation. That placement is consistent with the repository’s risk profile. The system accepts natural-language inputs that may try to elicit hidden prompts, unauthorized order data, policy overrides, or other restricted behavior. Those attack surfaces are visible both in the safety tool descriptions in [retail_support/runtime.py](retail_support/runtime.py) and in the heuristic rulebook inside [SupportOperationsService.assess_request_risk()](retail_support/services.py:98).

The Stage 2 report adds direct empirical support for this layer. It documents promptfoo-generated attacks across multiple vulnerability categories and records confirmed vulnerabilities involving multi-turn BOLA, social escalation, email-based cross-user lookup, and user enumeration. It also documents Inspect AI benchmark outcomes across AgentHarm-style, StrongReject-style, and boundary-check tasks, showing that the system is stronger on direct refusals and boundary control than on subtle authorization drift across conversation state. Stage 2 should therefore be treated as directly evidenced and practically valuable, especially for surfacing failure modes that would not emerge from a static regression suite alone.

### 5.3 Stage 3 — Production monitoring and drift detection

The architecture assigns Langfuse, Arize Phoenix, AgentOps, and Evidently AI to the production layer. The Stage 3 report gives this layer a clear conceptual structure by separating tracing, debugging, drift detection, and monitoring as distinct but complementary production capabilities. The repository also exposes internal hooks that make such a layer plausible. Specifically, [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) captures structured request events, while [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394) reconstructs specialist-routing paths. Together, these functions show that the system can already represent the kinds of traces that external observability tooling would typically require.

A representative excerpt shows the trace shape already available to an observability platform:

```python
def _record_event(self, kind: str, actor: str, name: str, details: dict[str, Any]) -> None:
    self._current_request_events.append(
        {"kind": kind, "actor": actor, "name": name, "details": details}
    )

def _build_route(self, target: str) -> list[str]:
    if target != "supervisor":
        return [TARGET_DISPLAY_NAMES[target]]

    route: list[str] = []
    for event in self._current_request_events:
        if event["kind"] == "delegation":
            display_name = TARGET_DISPLAY_NAMES[event["name"]]
            if display_name not in route:
                route.append(display_name)
    return route or [TARGET_DISPLAY_NAMES["supervisor"]]
```

Analytically, Stage 3 is now better understood as an explicitly documented production-monitoring architecture rather than merely an inferred instrumentation precursor. The report clarifies the functional roles of Langfuse, AgentOps, Arize Phoenix, and Evidently AI within a layered observability stack.

That said, the evidence remains stronger on architecture than on deployed operations. The available material does not include live telemetry dashboards, production alert histories, long-run drift baselines, or screenshots from an active monitoring environment. Stage 3 is therefore best understood as well documented at the architectural level and partially evidenced in code, but not yet demonstrated as a mature production deployment.

### 5.4 Stage 4 — Human review and continuous improvement
Stage 4 integrates Langfuse, MLflow, and TruLens into a concrete human review and continuous improvement workflow. At its foundation, [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77) in [retail_support/services.py](retail_support/services.py) creates a structured handoff object with an identifier, topic, summary, status, and service-level expectation.

The escalation payload is structurally concrete:
```python
def create_escalation_ticket(self, user_id: str, topic: str, summary: str) -> dict[str, Any]:
    self.ticket_counter += 1
    return {
        "ticket_id": f"tkt_{self.ticket_counter}",
        "user_id": user_id,
        "topic": topic,
        "summary": summary,
        "status": "open",
        "sla": "24 business hours",
    }
```

Beyond this handoff object, the branch implements three integrated evaluation layers. TruLens computes RAG Triad scores (context relevance, groundedness, answer relevance) as a scoring library. These scores flow into MLflow as run-level metrics and trace-level assessments visible in the Quality tab. When any score falls below a configurable threshold, `_maybe_flag_for_review()` in `retail_support/runtime.py` automatically routes the trace to a Langfuse annotation queue (retail-support-review) for human scoring on accuracy, policy compliance, and tone.

Completed annotations are captured via langfuse_annotations.py as `AnnotatedTrace` objects, which feed into a DSPy judge alignment pipeline (`optimize_judge.py`) that trains an optimized LLM judge against human labels and logs alignment results back to MLflow.
This is sufficient to say the code implements automated quality gating, a reviewer queue with label capture, and a closed-loop judge optimization workflow. It does not yet implement multi-reviewer adjudication, dynamic reviewer assignment, or workload balancing — the current model assumes a single reviewer per trace.

### 5.5 Stage 5 — Safety assurance and compliance

Stage 5 is one of the strongest conceptual matches between the five-stage framework and the repository’s current logic. The runtime includes explicit safety routing through [contact_trust_and_safety()](retail_support/runtime.py:261), policy retrieval through [get_policy_summary()](retail_support/runtime.py:197), and risk analysis through [assess_request_risk()](retail_support/runtime.py:204). The underlying service logic in [SupportOperationsService.assess_request_risk()](retail_support/services.py:98) checks for prompt-injection patterns, data-exfiltration attempts, SQL-injection-like strings, and policy-bypass language. Related privacy and security constraints are embedded in [retail_support/data.py](retail_support/data.py:43).

The core safety heuristic is not merely described in prose; it is codified as a rulebook with explicit refusal semantics in [SupportOperationsService.assess_request_risk()](retail_support/services.py:98):

```python
rulebook = {
    "prompt_injection": [
        r"ignore .*instructions",
        r"system prompt",
        r"developer message",
        r"hidden prompt",
    ],
    "data_exfiltration": [r"other customer", r"credit card", r"api key"],
    "policy_bypass": [r"bypass", r"override", r"without authorization"],
}

...

if flags:
    return {
        "risk_level": "high",
        "flags": flags,
        "recommended_action": "refuse_and_explain_or_escalate",
    }
```

This excerpt clarifies why Stage 5 is one of the strongest safety-backed parts of the project: the application does not rely on generic safety aspiration alone, but on concrete pattern detection and an explicit escalation-or-refusal recommendation. That heuristic layer is further reinforced by the trust-and-safety specialist prompt in [retail_support/runtime.py](retail_support/runtime.py:221).

The Stage 5 report substantially strengthens the evidence base by documenting a three-tool safety stack, concrete benchmark and gating patterns, and several nuanced findings: regex-rule gaps that the LLM still catches, policy boundaries that are enforced safely at code level, and cross-user access controls that remain prompt-level rather than service-level. That combination moves Stage 5 beyond architectural preparedness into direct empirical support, even if a full audit program would still require repeated runs, broader benchmark coverage, and production-grade evidence retention.

## 6. Synthesis of Findings

Bringing the comparative landscape and the repository case study together yields five synthesis findings.

First, the repository and stage reports validate the premise that agent evaluation is multidimensional. This application needs answer-quality assessment, tool-calling checks, routing validation, safety assurance, escalation readiness, and operational traceability. No single tool in the landscape is positioned to cover all of those needs.

Second, the evidence base is now stronger than simple code-level inference. Stage 1, Stage 2, Stage 4, and Stage 5 all include explicit report-level evidence and empirical findings, while Stage 3 is documented as a production-monitoring architecture supported by code-level instrumentation hooks. This supports the broader conclusion that testing, observability, monitoring, review, and governance are distinct layers rather than stages that collapse into one product.

Third, the application’s design amplifies the importance of route-aware evaluation. Because the supervisor may delegate across multiple specialists through [RetailSupportOrchestrator._delegate_to_specialist()](retail_support/runtime.py:271), an apparently correct final answer may still conceal a wrong internal route, a missing tool call, or an avoidable safety exposure. That makes route tracing and event capture materially important, not merely operationally convenient.

Fourth, the repository shows that safety is intertwined with normal business logic. Refund authorization in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34) depends on ownership checks, final-sale restrictions, and policy windows. Safety is therefore not limited to extreme adversarial prompts; it also governs ordinary customer-service flows.

Fifth, experiment governance remains necessary even in a relatively compact application. The presence of [mlruns/](mlruns/) is a useful signal that run tracking mattered in the broader project, but the available evidence still supports the larger conclusion that experiment systems such as MLflow unify evidence rather than replace evaluation methods.

A detailed case-study mapping is shown below.

| Case-study evaluation need | Repository evidence | Best-fit tool layer in this framework | Evidence status |
|---|---|---|---|
| Refund-policy and order-flow regression | [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34), [get_order_snapshot()](retail_support/runtime.py:166), cached Stage 1 scenarios in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1), plus Stage 1 workflow results | DeepEval + MLflow | Directly evidenced |
| Knowledge-grounded support answers | [search_support_knowledge()](retail_support/runtime.py:159), in-memory knowledge base in [retail_support/data.py](retail_support/data.py:3), plus Stage 1 dataset-generation and report outputs | Ragas + DeepEval | Directly evidenced |
| Supervisor routing fidelity | [contact_knowledge_specialist()](retail_support/runtime.py:251), [contact_order_specialist()](retail_support/runtime.py:256), [contact_trust_and_safety()](retail_support/runtime.py:261), [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394), plus Stage 4 trace-and-review workflows | Langfuse + AgentOps + DeepEval | Directly evidenced |
| Adversarial refusal and policy enforcement | [assess_request_risk()](retail_support/runtime.py:204), [SupportOperationsService.assess_request_risk()](retail_support/services.py:98), [retail_support/data.py](retail_support/data.py:43), plus Stage 2 and Stage 5 empirical findings | promptfoo + Inspect AI + DeepEval | Directly evidenced |
| Human escalation and exception handling | [create_escalation_ticket()](retail_support/runtime.py:185), [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77), plus Stage 4 annotation and review loop | TruLens + Langfuse + MLflow | Directly evidenced |
| Production drift and behavioral change detection | Internal event hooks in [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384), architectural analysis in the Stage 3 report | Arize Phoenix + Evidently AI + AgentOps | Architecturally documented; partially evidenced |

## 7. Gap Assessment

The comparison also reveals a set of concrete gaps between the five-stage architecture and the evidence available in the repository.

| Gap area | Current repository evidence | Implication |
|---|---|---|
| Stage report availability | Stage 1–5 reports are available for this project and materially strengthen the evidence base | Later-stage execution is reviewable at the report level rather than only inferable from branch metadata |
| Stage 1 result completeness | Stage 1 includes reported outcomes, artifact generation, and MLflow tracking, but still reflects specific runs rather than long-term trend analysis | Development testing is directly evidenced, though longitudinal benchmarking remains limited |
| Production observability deployment | Stage 3 defines the monitoring architecture and the code exposes event hooks through [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384), but no live production telemetry environment is shown | Observability is well framed architecturally, but mature operational deployment remains unproven |
| Drift monitoring | Stage 3 explains the role of Phoenix and Evidently, but no long-run drift baselines or alert histories are shown | Drift remains more of an architectural capability than a demonstrated operational program |
| Human review workflow | Stage 4 documents annotation queues, reviewer scoring, and judge alignment, but the demonstrated dataset is still relatively small | Human review is directly evidenced, though scale and statistical stability remain open questions |
| Compliance reporting | Stage 5 documents benchmark structure and safety findings, but a broader audit program would still need repeated runs, retained logs, and formal governance processes | Safety assurance is directly evidenced, while audit maturity remains incomplete |
| Cross-tool unification evidence | Stage 1 and Stage 4 both show MLflow-backed tracking, but end-to-end lineage across all five stages is not yet presented as one unified reporting surface | Governance capability is visible, but cross-stage operational integration is still incomplete |

Taken together, these gaps reinforce a more precise analytic point: the project now demonstrates substantial multi-stage lifecycle execution, but the maturity of that execution still varies by stage, with the biggest open questions concentrated in production-scale monitoring, drift baselining, and cross-stage operational unification.

## 8. Recommended Evaluation Stack

The recommended stack should preserve the five-stage architecture while separating baseline needs from optional depth.

### Baseline recommendation

| Evaluation need | Baseline tool choice | Rationale grounded in the comparison and repository evidence |
|---|---|---|
| Development regression testing | DeepEval | Best aligned with the directly evidenced Stage 1 residue in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) and the need to validate refund, order, escalation, and refusal flows |
| Knowledge and grounded-answer evaluation | Ragas | Best fit for the repository’s knowledge-dependent answers via [search_support_knowledge()](retail_support/runtime.py:159) |
| Experiment logging and run comparison | MLflow | Provides the unification layer already implied by [mlruns/](mlruns/) and the five-stage architecture |
| Fast pre-deployment red-teaming | promptfoo | Best baseline addition for systematic adversarial prompt testing before release |
| Production trace visibility | Langfuse | Best baseline fit for turning internal events from [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) into an operational trace layer |
| Stronger structured safety benchmarking | Inspect AI | Adds rigor beyond ad hoc red-teaming for the safety-critical behaviors exposed in [SupportOperationsService.assess_request_risk()](retail_support/services.py:98) |

This baseline stack is intentionally pragmatic. It covers development testing, knowledge evaluation, experiment tracking, red-teaming, production tracing, and safety benchmarking without assuming that every production installation requires all monitoring platforms immediately.

### Advanced or optional additions

| Need or maturity trigger | Advanced addition | Why it becomes valuable |
|---|---|---|
| Production-scale monitoring and analysis | Arize Phoenix | Valuable when trace volumes, evaluation breadth, or analysis needs exceed simple tracing |
| Agent-operations telemetry | AgentOps | Valuable when the operating problem is understanding agent-run behavior at scale |
| Explicit drift monitoring | Evidently AI | Valuable when input, behavior, or output distributions are expected to shift over time |
| Structured review workflows and continuous improvement | TruLens | Valuable when human judgments need to be captured systematically and fed back into iteration |

### Recommended stage mapping

| Stage | Recommended stack | Notes |
|---|---|---|
| Stage 1 | DeepEval + Ragas + MLflow | Strongest fit for the current repository state |
| Stage 2 | promptfoo + Inspect AI | Most appropriate for adversarial and safety hardening before deployment |
| Stage 3 | Langfuse + one or more of Arize Phoenix, AgentOps, Evidently AI | Select based on operational scale and whether tracing, operations, or drift is the dominant concern |
| Stage 4 | Langfuse + MLflow + TruLens | Appropriate when escalation outcomes and human judgments need to become reusable improvement data |
| Stage 5 | Inspect AI + promptfoo + DeepEval | Best fit for repeated safety assurance across both benchmark-style and scenario-style evaluation |

In short, the recommended stack is not a winner-takes-all selection. It is a layered operating model in which each tool is justified by a specific evaluation need.

## 10. Conclusion

The agent-evaluation landscape examined here is best understood as a coordinated stack rather than a single framework decision. DeepEval, Ragas, promptfoo, Inspect AI, Langfuse, Arize Phoenix, TruLens, Evidently AI, AgentOps, and MLflow each address a distinct part of the lifecycle. The five-stage architecture is therefore analytically sound: it reflects the real fragmentation of evaluation work across development, pre-deployment hardening, production observability, human review, and safety assurance.

The project strengthens that conclusion by showing how those needs emerge in practice. Its multi-agent retail support design in [README.md](README.md:3) and [RetailSupportOrchestrator](retail_support/runtime.py:53) creates concrete requirements around routing fidelity, tool correctness, knowledge grounding, safety refusal, and escalation handling. The Stage 1, Stage 2, Stage 4, and Stage 5 reports provide direct evidence that these concerns were evaluated with concrete toolchains and produced actionable findings, while Stage 3 provides the architectural framing for production observability and drift detection. That evidence pattern does not weaken the five-stage model; it confirms why a full evaluation program cannot be reduced to one benchmark or one monitoring tool.

The most defensible conclusion on the available evidence is therefore straightforward: no single tool spans the full agent-evaluation lifecycle, and the most robust strategy is a layered stack that combines development evaluation, adversarial testing, observability, drift monitoring, human review, and experiment governance.

## References / Appendix

### Source provenance

1. Comparative background research covering DeepEval, Ragas, promptfoo, Inspect AI, Langfuse, Arize Phoenix, TruLens, Evidently AI, AgentOps, and MLflow, including a feature matrix and the five-stage evaluation architecture used in this document.
2. Repository evidence from [README.md](README.md:3), [retail_support/runtime.py](retail_support/runtime.py), [retail_support/services.py](retail_support/services.py), [retail_support/data.py](retail_support/data.py), [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1), [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1), [mlruns/](mlruns/), and the Stage 1–5 reports provided for this project.
3. Git metadata from [.git/FETCH_HEAD](.git/FETCH_HEAD:1), which records the visible stage branches on the fetched remote history.

### Evidence limitations

- Code snippets are taken from the repository files cited inline.
- Stage branch names are visible in [.git/FETCH_HEAD](.git/FETCH_HEAD:1), but the analysis now also incorporates the Stage 1–5 reports provided for this project.
- Evidence depth still varies by stage: some stages include measured outcomes and findings, while others are stronger on architecture and workflow design.
- Production observability backends, drift dashboards, annotation queues, and formal compliance processes should be treated as demonstrated only to the extent described in the available reports and code.
- Cross-stage unification remains incomplete even where individual stages are well evidenced.
