"""
quarantine_engine.py — Layer 3: Epistemic Quarantine Primitive

Mathematical Foundation:
    This module implements the core architectural primitive of CIS: inference-time
    epistemic quarantine. It is the mechanism that does not exist in any of the
    200+ papers surveyed in arXiv:2510.06265.

    Definition (Epistemic Quarantine):
        Let C = {c₁, c₂, ..., cₙ} be the set of claims extracted from an LLM
        reasoning step. Let φ: C → [0, 1] be the contamination scoring function.
        Let τ ∈ (0, 1) be the contamination threshold.

        We define two disjoint partitions:
            S = {c ∈ C | φ(c) < τ}    ← Safe set (enters main reasoning graph)
            Q = {c ∈ C | φ(c) ≥ τ}    ← Quarantine set (blocked from next step)

        Such that:
            S ∪ Q = C  and  S ∩ Q = ∅

        The CRITICAL INVARIANT is:
            context(step_{k+1}) ⊆ S
            ∀q ∈ Q: q ∉ context(step_{k+1})

        This means quarantined claims are NEVER passed as context to the next
        reasoning step, breaking the contamination propagation chain.

    Containment Depth Bound (Theorem 1):
        Let ρ = P(φ(c) ≥ τ | c is truly contaminated) be the true positive rate.
        The probability that a contaminated claim survives k consecutive EQ
        checkpoints without being quarantined is:

            P(escape after k checkpoints) = (1 - ρ)^k

        Therefore, for a target escape probability ε:
            k* = ⌈log(ε) / log(1 - ρ)⌉

        For ρ = 0.7 and ε = 0.01:
            k* = ⌈log(0.01) / log(0.3)⌉ = ⌈-4.605 / -1.204⌉ = ⌈3.82⌉ = 4

        Four EQ checkpoints suffice to reduce contamination escape to < 1%.

    Implementation:
        Two NetworkX DiGraphs represent the partition:
        - main_graph: Contains safe claims → forms the context for next steps
        - quarantine_graph: Contains contaminated claims → isolated from reasoning

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cis.quarantine_engine")

CONTAMINATION_THRESHOLD: float = float(os.getenv("CONTAMINATION_THRESHOLD", "0.55"))


class QuarantineEngine:
    """Dual-graph epistemic quarantine engine — the core CIS primitive."""

    def __init__(self) -> None:
        """Initialize the two disjoint graph partitions."""
        self.main_graph: nx.DiGraph = nx.DiGraph()
        self.quarantine_graph: nx.DiGraph = nx.DiGraph()
        self._history: list[dict[str, Any]] = []
        logger.info(
            "QuarantineEngine initialized with threshold τ=%.2f",
            CONTAMINATION_THRESHOLD,
        )

    def process_claim(
        self,
        claim: dict[str, Any],
        score_result: dict[str, Any],
        session_id: str = "",
    ) -> str:
        """Route a scored claim to main_graph (SAFE) or quarantine_graph (QUARANTINED)."""
        claim_id: str = claim.get("id", "unknown")
        claim_text: str = claim.get("claim", "")
        phi: float = score_result.get("score", 0.0)
        timestamp: str = datetime.now(timezone.utc).isoformat()

        node_attrs: dict[str, Any] = {
            "id": claim_id,
            "text": claim_text,
            "score": phi,
            "wiki_match": score_result.get("wiki_match", False),
            "confidence": score_result.get("confidence", 0.0),
            "causal_flag": score_result.get("causal_flag", False),
            "reason": score_result.get("reason", ""),
            "timestamp": timestamp,
            "session_id": session_id,
        }

        if phi >= CONTAMINATION_THRESHOLD:
            # QUARANTINE: Claim is contaminated — do NOT pass to next reasoning step
            self.quarantine_graph.add_node(claim_id, **node_attrs)
            status = "QUARANTINED"
            logger.warning(
                "⚠ QUARANTINED claim %s (φ=%.4f ≥ τ=%.2f): %.60s...",
                claim_id, phi, CONTAMINATION_THRESHOLD, claim_text,
            )
        else:
            # SAFE: Claim passes verification — allowed into reasoning context
            self.main_graph.add_node(claim_id, **node_attrs)
            status = "SAFE"
            logger.info(
                "✓ SAFE claim %s (φ=%.4f < τ=%.2f): %.60s...",
                claim_id, phi, CONTAMINATION_THRESHOLD, claim_text,
            )

        # Record in history for traceability
        self._history.append({
            "claim_id": claim_id,
            "status": status,
            "score": phi,
            "timestamp": timestamp,
        })

        return status

    def get_safe_context(self) -> list[dict[str, Any]]:
        """Return ONLY safe claims — this is what gets passed to the next reasoning step.

        CRITICAL INVARIANT:
            context(step_{k+1}) = get_safe_context()
            No quarantined claim ever appears in this output.
        """
        safe_claims: list[dict[str, Any]] = []
        for node_id in self.main_graph.nodes():
            attrs = self.main_graph.nodes[node_id]
            safe_claims.append({
                "id": attrs.get("id", node_id),
                "text": attrs.get("text", ""),
                "score": attrs.get("score", 0.0),
            })
        return safe_claims

    def get_quarantined(self) -> list[dict[str, Any]]:
        """Return all quarantined claims with their contamination details."""
        quarantined: list[dict[str, Any]] = []
        for node_id in self.quarantine_graph.nodes():
            attrs = self.quarantine_graph.nodes[node_id]
            quarantined.append({
                "id": attrs.get("id", node_id),
                "text": attrs.get("text", ""),
                "score": attrs.get("score", 0.0),
                "reason": attrs.get("reason", ""),
                "wiki_match": attrs.get("wiki_match", False),
                "confidence": attrs.get("confidence", 0.0),
                "causal_flag": attrs.get("causal_flag", False),
                "timestamp": attrs.get("timestamp", ""),
            })
        return quarantined

    def get_graph_data(self) -> dict[str, Any]:
        """Export both graphs as JSON-serializable structures for the API and frontend."""
        return {
            "main_graph": {
                "nodes": [
                    {"id": n, **self.main_graph.nodes[n]}
                    for n in self.main_graph.nodes()
                ],
                "edges": [
                    {"from": u, "to": v, **d}
                    for u, v, d in self.main_graph.edges(data=True)
                ],
            },
            "quarantine_graph": {
                "nodes": [
                    {"id": n, **self.quarantine_graph.nodes[n]}
                    for n in self.quarantine_graph.nodes()
                ],
                "edges": [
                    {"from": u, "to": v, **d}
                    for u, v, d in self.quarantine_graph.edges(data=True)
                ],
            },
            "summary": {
                "total_safe": self.main_graph.number_of_nodes(),
                "total_quarantined": self.quarantine_graph.number_of_nodes(),
                "quarantine_rate": self._quarantine_rate(),
            },
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Return the full processing history for audit trail."""
        return list(self._history)

    def clear(self) -> None:
        """Reset both graphs — used between experiment runs."""
        self.main_graph.clear()
        self.quarantine_graph.clear()
        self._history.clear()
        logger.info("QuarantineEngine cleared.")

    def _quarantine_rate(self) -> float:
        """Compute the current quarantine rate Q/(S+Q)."""
        total = self.main_graph.number_of_nodes() + self.quarantine_graph.number_of_nodes()
        if total == 0:
            return 0.0
        return round(self.quarantine_graph.number_of_nodes() / total, 4)

    @staticmethod
    def containment_depth_bound(rho: float, epsilon: float) -> int:
        """Compute k* = ⌈log(ε) / log(1-ρ)⌉ — the minimum number of EQ checkpoints.

        Theorem (Containment Depth Bound):
            For a true positive detection rate ρ and target escape probability ε,
            the minimum number of consecutive EQ checkpoints k* required to ensure
            that contamination escape probability falls below ε is:

                k* = ⌈log(ε) / log(1 - ρ)⌉

        Proof:
            P(contamination escapes k checkpoints) = (1-ρ)^k
            We require (1-ρ)^k ≤ ε
            k · log(1-ρ) ≤ log(ε)        [since log(1-ρ) < 0 for ρ ∈ (0,1)]
            k ≥ log(ε) / log(1-ρ)
            k* = ⌈log(ε) / log(1-ρ)⌉     ∎

        Args:
            rho: True positive detection rate ρ ∈ (0, 1)
            epsilon: Target escape probability ε ∈ (0, 1)

        Returns:
            k*: Minimum number of EQ checkpoints (integer)

        Raises:
            ValueError: If ρ or ε are out of valid range.
        """
        if not (0.0 < rho < 1.0):
            raise ValueError(f"ρ must be in (0, 1), got {rho}")
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"ε must be in (0, 1), got {epsilon}")

        k_star: int = math.ceil(math.log(epsilon) / math.log(1.0 - rho))
        logger.info(
            "Containment Depth Bound: ρ=%.4f, ε=%.4f → k*=%d",
            rho, epsilon, k_star,
        )
        return k_star
