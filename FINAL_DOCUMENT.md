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

Direct evidence in the repository is strongest for Stage 1 residue and for the application logic that supports Stage 5. Cached DeepEval-related node IDs in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) show curated scenarios for delayed order escalation, refund eligibility, final-sale denial, refund-policy knowledge retrieval, order lookup, and prompt-injection refusal. A DeepEval telemetry residue is also present in [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1). However, the repository does not expose the underlying test implementation or any pass-fail metrics, so Stage 1 can only be described as partially evidenced. For Stages 2 through 5, branch names are visible in [.git/FETCH_HEAD](.git/FETCH_HEAD:1), but standalone stage reports are not present in the repository. Accordingly, those stages are treated as architecturally motivated, partially evidenced through code hooks, or not directly documented in the current checkout.

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
| Repository evidence | Multi-agent system structure in [README.md](README.md:3), orchestration in [RetailSupportOrchestrator](retail_support/runtime.py:53), service logic in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34) and [SupportOperationsService.assess_request_risk()](retail_support/services.py:98), policy constraints in [retail_support/data.py](retail_support/data.py:43), event capture in [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384), route reconstruction in [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394), DeepEval residue in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) and [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1), and experiment-tracking residue in [mlruns/](mlruns/) | Grounds the abstract comparison in a concrete multi-agent application | The repository does not include standalone Stage 2–5 reports, and Stage 1 evidence is indirect because cached test references exist without the underlying test implementation |
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
| Stage 1 — Development testing | DeepEval + Ragas + MLflow | DeepEval residue in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1), telemetry residue in [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1), experiment-tracking directory [mlruns/](mlruns/) | Partially assessed |
| Stage 2 — Pre-deployment red-teaming and safety | promptfoo + Inspect AI | Stage branch name visible in [.git/FETCH_HEAD](.git/FETCH_HEAD:2); no standalone report or execution artifact appears in the repository | Not directly documented in the repository |
| Stage 3 — Production monitoring and drift detection | Langfuse + Arize Phoenix + AgentOps + Evidently AI | Event capture in [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) and route reconstruction in [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394) | Partially assessed as an instrumentation precursor |
| Stage 4 — Human review and continuous improvement | Langfuse + MLflow + TruLens | Human escalation precursor in [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77); no reviewer workflow appears in the repository | Partially assessed as a handoff precursor |
| Stage 5 — Safety assurance and compliance | Inspect AI + promptfoo + DeepEval | Safety-specialist routing in [contact_trust_and_safety()](retail_support/runtime.py:261), policy lookup in [get_policy_summary()](retail_support/runtime.py:197), risk heuristics in [assess_request_risk()](retail_support/runtime.py:204) and [SupportOperationsService.assess_request_risk()](retail_support/services.py:98), plus privacy and security constraints in [retail_support/data.py](retail_support/data.py:43) | Partially assessed, with strong code-level evidence |

### 5.1 Stage 1 — Development testing

Stage 1 is the most directly evidenced evaluation layer in the current repository snapshot. The cached node IDs in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) indicate a curated regression suite covering delayed-order escalation, eligible refund approval, final-sale refund denial, refund-policy knowledge retrieval, order-status lookup, and prompt-injection refusal. Those scenarios map closely to concrete application surfaces in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34), [search_support_knowledge()](retail_support/runtime.py:159), [get_order_snapshot()](retail_support/runtime.py:166), [create_escalation_ticket()](retail_support/runtime.py:185), and [SupportOperationsService.assess_request_risk()](retail_support/services.py:98).

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

Together, the cache artifact and the service logic show why Stage 1 is partially evidenced: the repository exposes named regression targets and the deterministic policy rules they likely exercised, even though it does not expose the full test implementation or result thresholds.

This is meaningful because it shows that evaluation, at least at one stage, was not purely theoretical. However, the available evidence remains incomplete. The cached node IDs do not expose thresholds, scoring logic, model configuration, or pass-fail outcomes. The underlying test implementation is not included in the current checkout, so any stronger claim would go beyond the available evidence. The DeepEval telemetry residue in [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1) and the presence of [mlruns/](mlruns/) further suggest prior evaluation activity, but they do not recover the missing result details.

### 5.2 Stage 2 — Pre-deployment red-teaming and safety

The architecture assigns promptfoo and Inspect AI to pre-deployment red-teaming and safety validation. That placement is consistent with the repository’s risk profile. The system accepts natural-language inputs that may try to elicit hidden prompts, unauthorized order data, policy overrides, or other restricted behavior. Those attack surfaces are visible both in the safety tool descriptions in [retail_support/runtime.py](retail_support/runtime.py) and in the heuristic rulebook inside [SupportOperationsService.assess_request_risk()](retail_support/services.py:98).

However, the repository does not contain a standalone Stage 2 report, and no promptfoo or Inspect AI artifacts are directly evidenced in the current checkout. The only direct repository signal is the existence of the branch name for Stage 2 in [.git/FETCH_HEAD](.git/FETCH_HEAD:2). Accordingly, Stage 2 should be treated as architecturally proposed and operationally justified, but not directly documented in the repository.

### 5.3 Stage 3 — Production monitoring and drift detection

The architecture assigns Langfuse, Arize Phoenix, AgentOps, and Evidently AI to the production layer. The repository does expose internal hooks that make such a layer plausible. Specifically, [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) captures structured request events, while [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394) reconstructs specialist-routing paths. Together, these functions show that the system can already represent the kinds of traces that external observability tooling would typically require.

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

Analytically, this is not yet Langfuse, Phoenix, AgentOps, or Evidently AI; it is the internal event schema and route-reconstruction logic those tools would consume. That distinction explains why Stage 3 is described as an instrumentation precursor rather than a demonstrated production-monitoring deployment.

That said, these hooks should not be mistaken for a full production telemetry stack. No deployed tracing backend, no drift dashboard, no route-quality analytics, and no online alerting pipeline are directly evidenced in the repository. The Stage 3 branch name exists in [.git/FETCH_HEAD](.git/FETCH_HEAD:4), but branch metadata alone is not enough to claim production monitoring results. Stage 3 is therefore best understood as partially assessed through instrumentation precursors rather than a demonstrated observability deployment.

### 5.4 Stage 4 — Human review and continuous improvement

Stage 4 in the five-stage architecture combines Langfuse, MLflow, and TruLens for human review and continuous improvement. The repository contains a concrete precursor to that workflow: [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77) creates a structured human escalation ticket with an identifier, topic, summary, status, and service-level expectation. The knowledge base also encodes escalation logic in [retail_support/data.py](retail_support/data.py:15) and [retail_support/data.py](retail_support/data.py:55).

The escalation payload is also structurally concrete:

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

This supports the report’s narrower claim: the code implements a handoff object for human intervention, with stable identifiers and service expectations, but it does not yet encode reviewer assignment, adjudication outcomes, or label capture.

This is sufficient to say that the application already recognizes the need for human intervention in unresolved or delayed cases. It is not sufficient to claim the presence of reviewer queues, annotation interfaces, adjudication workflows, or closed-loop label capture. No such workflow is evident in the current checkout. The Stage 4 branch name appears in [.git/FETCH_HEAD](.git/FETCH_HEAD:5), but that is metadata evidence only.

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

This excerpt clarifies why Stage 5 is one of the strongest code-backed parts of the repository: the application does not rely on generic safety aspiration alone, but on concrete pattern detection and an explicit escalation-or-refusal recommendation. That heuristic layer is further reinforced by the trust-and-safety specialist prompt in [retail_support/runtime.py](retail_support/runtime.py:221). It remains, however, a local safety mechanism rather than a full compliance evidence package.

The cached Stage 1 scenario for prompt-injection refusal in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) further indicates that at least one safety-relevant behavior was intentionally regression-tested. Even here, however, the evidence remains partial. The repository does not directly provide compliance reports, benchmark scores, attack-coverage summaries, or documented refusal thresholds. The case for Stage 5 is therefore strong in architectural need and code-level preparedness, but incomplete in measurable outcome reporting.

## 6. Synthesis of Findings

Bringing the comparative landscape and the repository case study together yields five synthesis findings.

First, the repository validates the premise that agent evaluation is multidimensional. This application needs answer-quality assessment, tool-calling checks, routing validation, safety assurance, escalation readiness, and operational traceability. No single tool in the supplied landscape is positioned to cover all of those needs.

Second, the strongest direct repository evidence concerns regression-oriented testing and embedded safety logic rather than fully operationalized monitoring or review workflows. This supports the broader conclusion that testing, observability, monitoring, and unification are distinct layers rather than stages that collapse into one product.

Third, the application’s design amplifies the importance of route-aware evaluation. Because the supervisor may delegate across multiple specialists through [RetailSupportOrchestrator._delegate_to_specialist()](retail_support/runtime.py:271), an apparently correct final answer may still conceal a wrong internal route, a missing tool call, or an avoidable safety exposure. That makes route tracing and event capture materially important, not merely operationally convenient.

Fourth, the repository shows that safety is intertwined with normal business logic. Refund authorization in [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34) depends on ownership checks, final-sale restrictions, and policy windows. Safety is therefore not limited to extreme adversarial prompts; it also governs ordinary customer-service flows.

Fifth, experiment governance remains necessary even in a relatively compact application. The presence of [mlruns/](mlruns/) is a useful signal that run tracking mattered in the broader project, but the available evidence still supports the larger conclusion that experiment systems such as MLflow unify evidence rather than replace evaluation methods.

A detailed case-study mapping is shown below.

| Case-study evaluation need | Repository evidence | Best-fit tool layer in this framework | Evidence status |
|---|---|---|---|
| Refund-policy and order-flow regression | [SupportOperationsService.assess_refund_eligibility()](retail_support/services.py:34), [get_order_snapshot()](retail_support/runtime.py:166), cached Stage 1 scenarios in [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1) | DeepEval + MLflow | Partially assessed |
| Knowledge-grounded support answers | [search_support_knowledge()](retail_support/runtime.py:159), in-memory knowledge base in [retail_support/data.py](retail_support/data.py:3) | Ragas + DeepEval | Partially assessed |
| Supervisor routing fidelity | [contact_knowledge_specialist()](retail_support/runtime.py:251), [contact_order_specialist()](retail_support/runtime.py:256), [contact_trust_and_safety()](retail_support/runtime.py:261), [RetailSupportOrchestrator._build_route()](retail_support/runtime.py:394) | Langfuse + AgentOps + DeepEval | Partially assessed |
| Adversarial refusal and policy enforcement | [assess_request_risk()](retail_support/runtime.py:204), [SupportOperationsService.assess_request_risk()](retail_support/services.py:98), [retail_support/data.py](retail_support/data.py:43) | promptfoo + Inspect AI + DeepEval | Partially assessed |
| Human escalation and exception handling | [create_escalation_ticket()](retail_support/runtime.py:185), [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77) | TruLens + Langfuse + MLflow | Partially assessed |
| Production drift and behavioral change detection | Internal event hooks only in [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384) | Arize Phoenix + Evidently AI + AgentOps | Not directly evidenced |

## 7. Gap Assessment

The comparison also reveals a set of concrete gaps between the five-stage architecture and the evidence available in the repository.

| Gap area | Current repository evidence | Implication |
|---|---|---|
| Stage report availability | Only branch-name evidence for Stages 2–5 appears in [.git/FETCH_HEAD](.git/FETCH_HEAD:1); standalone reports are not present in the repository | The lifecycle architecture is visible, but its later-stage execution record is not directly reviewable |
| Stage 1 result completeness | Cached scenario names and telemetry residue exist, but exact metrics, thresholds, and outcomes are not available in the repository | Development testing is evidenced only indirectly |
| Production observability deployment | Internal event capture exists in [RetailSupportOrchestrator._record_event()](retail_support/runtime.py:384), but no external telemetry sink is shown | Observability is an architectural affordance rather than a demonstrated operating layer |
| Drift monitoring | No direct dashboards, baselines, alert rules, or distribution comparisons are evidenced | Drift remains a proposed operational concern rather than a measured one |
| Human review workflow | Escalation tickets exist through [SupportOperationsService.create_escalation_ticket()](retail_support/services.py:77), but annotation or adjudication queues are not evidenced | Human escalation is present without a closed improvement loop |
| Compliance reporting | Safety heuristics and policies exist, but formal assurance artifacts are not available in the cited sources | Safety preparedness is visible, while audit-grade assurance remains unproven |
| Cross-tool unification evidence | [mlruns/](mlruns/) suggests experiment tracking, but cross-stage lineage is not directly reviewable | Governance intent is present, but end-to-end provenance is incomplete |

Taken together, these gaps reinforce the central analytic point: the repository demonstrates evaluation need and partial implementation readiness more clearly than it demonstrates full lifecycle execution.

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

The repository strengthens that conclusion by showing how those needs emerge in practice. Its multi-agent retail support design in [README.md](README.md:3) and [RetailSupportOrchestrator](retail_support/runtime.py:53) creates concrete requirements around routing fidelity, tool correctness, knowledge grounding, safety refusal, and escalation handling. Direct evidence in the repository is strongest for Stage 1 residue and Stage 5-oriented code paths, while Stages 2 through 4 remain either branch-level evidence, architectural precursors, or partially implemented hooks. That evidence pattern does not weaken the five-stage model; instead, it confirms why a full evaluation program cannot be reduced to one benchmark or one monitoring tool.

The most defensible conclusion on the available evidence is therefore straightforward: no single tool spans the full agent-evaluation lifecycle, and the most robust strategy is a layered stack that combines development evaluation, adversarial testing, observability, drift monitoring, human review, and experiment governance.

## References / Appendix

### Source provenance

1. Comparative background research covering DeepEval, Ragas, promptfoo, Inspect AI, Langfuse, Arize Phoenix, TruLens, Evidently AI, AgentOps, and MLflow, including a feature matrix and the five-stage evaluation architecture used in this document.
2. Repository evidence from [README.md](README.md:3), [retail_support/runtime.py](retail_support/runtime.py), [retail_support/services.py](retail_support/services.py), [retail_support/data.py](retail_support/data.py), [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids:1), [.deepeval/.deepeval_telemetry.txt](.deepeval/.deepeval_telemetry.txt:1), and [mlruns/](mlruns/).
3. Git metadata from [.git/FETCH_HEAD](.git/FETCH_HEAD:1), which records the visible stage branches on the fetched remote history.

### Evidence limitations

- Code snippets are taken from the repository files cited inline.
- Stage branch names are visible in [.git/FETCH_HEAD](.git/FETCH_HEAD:1), but this analysis is limited to content available in the current checkout.
- Standalone Stage 2–5 reports are not present in the repository.
- Stage 1 evidence comes from cached node IDs and telemetry residue rather than full test implementations or recorded outcome metrics.
- Production observability backends, drift dashboards, annotation queues, and formal compliance reports are discussed only where they are directly evidenced in the repository.
