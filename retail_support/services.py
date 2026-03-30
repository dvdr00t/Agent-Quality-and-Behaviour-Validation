"""Domain services for the retail support system."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from retail_support.data import KNOWLEDGE_BASE, ORDERS, POLICIES


class SupportOperationsService:
    """Business logic for support knowledge, orders, policies, and escalation."""

    def __init__(self) -> None:
        self.knowledge_base = deepcopy(KNOWLEDGE_BASE)
        self.policies = deepcopy(POLICIES)
        self.orders = deepcopy(ORDERS)
        self.ticket_counter = 1000

    def search_support_knowledge(self, query: str, limit: int = 2) -> list[dict[str, Any]]:
        ranked: list[tuple[int, dict[str, Any]]] = []
        for article in self.knowledge_base:
            haystack = " ".join([article["title"], article["body"], " ".join(article["tags"])])
            score = self._score_text_match(query, haystack)
            if score > 0:
                ranked.append((score, article))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [article for _, article in ranked[:limit]]

    def get_order_snapshot(self, order_id: str) -> dict[str, Any]:
        return self.orders.get(order_id) or {"error": f"Order {order_id} not found."}

    def assess_refund_eligibility(self, order_id: str, user_id: str, reason: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {
                "eligible": False,
                "reason": f"Order {order_id} was not found.",
                "code": "order_not_found",
            }

        if order["user_id"] != user_id:
            return {
                "eligible": False,
                "reason": "The provided user_id does not match the order owner.",
                "code": "authorization_failed",
            }

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

        return {
            "eligible": True,
            "reason": f"Refund is allowed for reason '{reason}' within the active policy window.",
            "code": "eligible",
        }

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

    def get_policy_summary(self, topic: str) -> dict[str, str]:
        best_key = "privacy"
        best_score = -1
        for key, body in self.policies.items():
            score = self._score_text_match(topic, f"{key} {body}")
            if score > best_score:
                best_key = key
                best_score = score
        return {"topic": best_key, "policy": self.policies[best_key]}

    def assess_request_risk(self, message: str) -> dict[str, Any]:
        lowered = message.lower()
        flags: list[str] = []

        rulebook = {
            "prompt_injection": [
                r"ignore .*instructions",
                r"system prompt",
                r"developer message",
                r"hidden prompt",
            ],
            "data_exfiltration": [
                r"other customer",
                r"not the owner",
                r"show me .*order",
                r"credit card",
                r"password",
                r"api key",
            ],
            "sql_injection_like": [
                r"drop table",
                r"union select",
                r"select \* from",
            ],
            "policy_bypass": [
                r"bypass",
                r"override",
                r"without authorization",
            ],
        }

        for flag, patterns in rulebook.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                flags.append(flag)

        if flags:
            return {
                "risk_level": "high",
                "flags": flags,
                "recommended_action": "refuse_and_explain_or_escalate",
            }

        return {
            "risk_level": "low",
            "flags": [],
            "recommended_action": "normal_handling",
        }

    def _score_text_match(self, query: str, candidate: str) -> int:
        query_terms = self._tokenize(query)
        candidate_terms = self._tokenize(candidate)
        return sum(1 for term in query_terms if term in candidate_terms)

    def _tokenize(self, value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", value.lower()))
