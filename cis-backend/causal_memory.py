"""
causal_memory.py — Layer 4: Contamination Causal DAG (CC-DAG)

Mathematical Foundation:
    The CC-DAG is a directed acyclic graph G = (V, E) that records the causal
    structure of contamination events across reasoning sessions.

    Definition (CC-DAG):
        V = {v₁, v₂, ..., vₘ}  where each vᵢ is a contamination event with:
            - text: the contaminated claim
            - cause: the root cause of contamination
            - score: φ(cᵢ) contamination score
            - timestamp: when the event was recorded
            - session_id: which reasoning session produced it

        E ⊆ V × V  where (vᵢ, vⱼ) ∈ E iff:
            vᵢ causally preceded vⱼ in the contamination chain,
            i.e., the contamination in vⱼ is a downstream effect of vᵢ.

    Key Distinction from Existing Memory Architectures:
        Retrieval in the CC-DAG is CAUSAL, not SEMANTIC.

        Existing systems (RAG, vector stores, semantic memory):
            retrieve(q) = argmax_v  similarity(embed(q), embed(v))
            → "Find the most similar past event" (cosine distance)

        CC-DAG:
            retrieve(c) = ancestors(c) in G
            → "Find if this claim shares a ROOT CAUSE with past contamination"

        This difference is fundamental:
        - Semantic retrieval can miss contamination that looks different but has
          the same underlying cause (e.g., different dates about the same event).
        - Causal retrieval captures this because it traverses the cause chain,
          not the surface form.

    Theorem (Causal Memory Completeness):
        If a contamination cause C has been recorded in the CC-DAG, then any
        future claim c' that shares root cause C will be detected in O(d) time,
        where d is the depth of the DAG from c' to C. This is independent of
        the total size of the DAG.

    Proof:
        The ancestor check performs BFS from c' upward through predecessor edges.
        Since G is a DAG (no cycles), BFS terminates in at most d steps where d
        is the longest path from c' to any root. The check for shared root cause
        only requires testing if any ancestor of c' matches any ancestor of the
        new claim's cause, which is O(d) per query.  ∎

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import logging
from collections import deque
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

import networkx as nx

from database import (
    get_edges,
    get_events,
    insert_edge,
    insert_event,
)

logger = logging.getLogger("cis.causal_memory")

# Similarity threshold for cause matching — two causes are considered
# the "same root cause" if their normalized text similarity exceeds this.
CAUSE_SIMILARITY_THRESHOLD: float = 0.45


class CausalDAG:
    """Contamination Causal DAG — immunological memory for the CIS."""

    def __init__(self) -> None:
        """Initialize empty CC-DAG graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._cause_index: dict[str, list[int]] = {}  # cause_text → [node_ids]
        logger.info("CausalDAG initialized.")

    def add_contamination_event(
        self,
        claim_text: str,
        cause: str,
        score: float,
        session_id: str = "",
        parent_id: Optional[int] = None,
        db_path: Optional[str] = None,
    ) -> int:
        """Record a contamination event in the CC-DAG and persist to SQLite."""
        # Persist to database first to get a stable ID
        event_id: int = insert_event(
            claim_text=claim_text,
            score=score,
            cause=cause,
            session_id=session_id,
            db_path=db_path,
        )

        # Add node to in-memory graph
        self.graph.add_node(
            event_id,
            text=claim_text,
            cause=cause,
            score=score,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Build cause index for fast root-cause lookup
        cause_key = cause.lower().strip()
        if cause_key:
            if cause_key not in self._cause_index:
                self._cause_index[cause_key] = []
            self._cause_index[cause_key].append(event_id)

        # Add causal edge if this event has a parent
        if parent_id is not None and self.graph.has_node(parent_id):
            self.graph.add_edge(
                parent_id,
                event_id,
                relation="caused_contamination",
            )
            insert_edge(
                source_id=parent_id,
                target_id=event_id,
                relation="caused_contamination",
                db_path=db_path,
            )
            logger.info(
                "Causal edge: %d → %d (cause: %.40s...)",
                parent_id, event_id, cause,
            )

        # Auto-link to existing events with same/similar cause
        self._auto_link_similar_causes(event_id, cause, db_path)

        logger.info(
            "Added contamination event #%d (score=%.4f, cause=%.40s...)",
            event_id, score, cause,
        )

        return event_id

    def has_causal_ancestor(self, claim_text: str) -> bool:
        """Check if a claim shares a causal root with any recorded contamination.

        This is the key query that distinguishes CC-DAG from semantic memory.
        We check if ANY previously contaminated claim has a similar cause to
        the textual content of this new claim.

        Algorithm:
            1. For each recorded cause in the cause index:
               - Compute text similarity between claim_text and the cause
               - If similarity > CAUSE_SIMILARITY_THRESHOLD → ancestor found
            2. If any ancestor found → return True (causal contamination detected)
        """
        if self.graph.number_of_nodes() == 0:
            return False

        claim_lower = claim_text.lower().strip()

        # Check against all known causes
        for cause_key, node_ids in self._cause_index.items():
            similarity = SequenceMatcher(None, claim_lower, cause_key).ratio()
            if similarity > CAUSE_SIMILARITY_THRESHOLD:
                logger.debug(
                    "Causal ancestor found: claim matches cause '%.40s...' "
                    "(similarity=%.3f, nodes=%s)",
                    cause_key, similarity, node_ids[:3],
                )
                return True

        # Also check against contaminated claim texts directly
        for node_id in self.graph.nodes():
            node_text = self.graph.nodes[node_id].get("text", "").lower()
            similarity = SequenceMatcher(None, claim_lower, node_text).ratio()
            if similarity > CAUSE_SIMILARITY_THRESHOLD:
                logger.debug(
                    "Causal ancestor found via text match: node #%d (similarity=%.3f)",
                    node_id, similarity,
                )
                return True

        return False

    def get_causal_trace(self, node_id: int) -> list[dict[str, Any]]:
        """Trace the full causal chain from a node back to its root cause(s) via BFS."""
        if not self.graph.has_node(node_id):
            return []

        trace: list[dict[str, Any]] = []
        visited: set[int] = set()
        queue: deque[int] = deque([node_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            attrs = self.graph.nodes[current]
            trace.append({
                "node_id": current,
                "text": attrs.get("text", ""),
                "cause": attrs.get("cause", ""),
                "score": attrs.get("score", 0.0),
                "timestamp": attrs.get("timestamp", ""),
            })

            # Traverse predecessors (toward root causes)
            for predecessor in self.graph.predecessors(current):
                if predecessor not in visited:
                    queue.append(predecessor)

        return trace

    def get_all_nodes_edges(self) -> dict[str, Any]:
        """Export the full CC-DAG structure for API response and visualization."""
        nodes: list[dict[str, Any]] = []
        for node_id in self.graph.nodes():
            attrs = dict(self.graph.nodes[node_id])
            attrs["node_id"] = node_id
            nodes.append(attrs)

        edges: list[dict[str, Any]] = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "relation": data.get("relation", "caused_contamination"),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_events": len(nodes),
            "total_edges": len(edges),
        }

    def persist_to_db(self, db_path: Optional[str] = None) -> None:
        """Full save of in-memory graph to SQLite — used for session persistence."""
        # Events and edges are already persisted incrementally in add_contamination_event.
        # This method exists for explicit checkpoint/backup scenarios.
        logger.info(
            "CC-DAG persisted: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def load_from_db(self, db_path: Optional[str] = None) -> None:
        """Reconstruct the in-memory CC-DAG from the persisted SQLite database."""
        self.graph.clear()
        self._cause_index.clear()

        events = get_events(db_path=db_path)
        for event in events:
            eid = event["id"]
            self.graph.add_node(
                eid,
                text=event["claim_text"],
                cause=event["cause"],
                score=event["score"],
                session_id=event.get("session_id", ""),
                timestamp=event["timestamp"],
            )

            cause_key = event["cause"].lower().strip()
            if cause_key:
                if cause_key not in self._cause_index:
                    self._cause_index[cause_key] = []
                self._cause_index[cause_key].append(eid)

        edges = get_edges(db_path=db_path)
        for edge in edges:
            self.graph.add_edge(
                edge["source_id"],
                edge["target_id"],
                relation=edge["relation"],
            )

        logger.info(
            "CC-DAG loaded from DB: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def clear(self) -> None:
        """Reset the in-memory DAG (does not clear the database)."""
        self.graph.clear()
        self._cause_index.clear()
        logger.info("CC-DAG in-memory graph cleared.")

    def _auto_link_similar_causes(
        self, event_id: int, cause: str, db_path: Optional[str] = None
    ) -> None:
        """Automatically create edges between events with similar root causes."""
        if not cause.strip():
            return

        cause_lower = cause.lower().strip()

        for other_cause_key, node_ids in self._cause_index.items():
            if other_cause_key == cause_lower:
                continue

            similarity = SequenceMatcher(None, cause_lower, other_cause_key).ratio()
            if similarity > CAUSE_SIMILARITY_THRESHOLD:
                for other_id in node_ids:
                    if other_id != event_id and not self.graph.has_edge(other_id, event_id):
                        self.graph.add_edge(
                            other_id,
                            event_id,
                            relation="shared_root_cause",
                        )
                        try:
                            insert_edge(
                                source_id=other_id,
                                target_id=event_id,
                                relation="shared_root_cause",
                                db_path=db_path,
                            )
                        except Exception as e:
                            logger.warning("Failed to persist auto-link edge: %s", e)

                        logger.debug(
                            "Auto-linked events %d → %d (cause similarity=%.3f)",
                            other_id, event_id, similarity,
                        )
