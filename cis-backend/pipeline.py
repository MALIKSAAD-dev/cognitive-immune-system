"""
pipeline.py — CIS Pipeline Orchestrator

Mathematical Foundation:
    The CIS pipeline implements the full inference-time epistemic quarantine
    workflow as a sequential composition of 5 layers:

        Pi(R) = L5 . L4 . L3 . L2 . L1 (R)

    where R is the raw LLM response and:
        L1: R -> C = {c_1, ..., c_n}           (Claim Extraction)
        L2: c_i -> phi(c_i) in [0, 1]          (Contamination Scoring)
        L3: (c_i, phi(c_i)) -> S ∪ Q           (Quarantine Partitioning)
        L4: Q -> G = (V, E)                    (Causal Memory Recording)
        L5: (c_i, S_wiki) -> T                 (Tolerance Calibration)

    Context-Grounded Prompting (Critical for Experiment Validity):
        When context is provided (e.g., from a retrieval system or user documents),
        the LLM is instructed to answer BASED ON that context. This simulates
        real-world scenarios where an LLM must trust provided information
        (RAG pipelines, tool outputs, database results).

        If the context contains contaminated information, the LLM may propagate
        it into its response. CIS then detects and quarantines those propagated
        false claims — this is the core value proposition.

    Pipeline Execution Order:
        1. Tolerance check (L5) runs BEFORE quarantine (L3)
           -> claims in T bypass quarantine entirely
        2. Causal memory (L4) feeds back into L2's scoring
           -> adaptive immunity across sessions
        3. Safe context from L3 is the ONLY input to filtered answer
           -> enforcing the EQ invariant

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional

from groq import AsyncGroq
from dotenv import load_dotenv

from claim_extractor import extract_grounded_claims, extract_claims
from contamination_scorer import score_claim
from quarantine_engine import QuarantineEngine
from causal_memory import CausalDAG
from tolerance_calibrator import ToleranceCalibrator
from database import init_db

load_dotenv()

logger = logging.getLogger("cis.pipeline")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
MODEL_ID: str = "llama-3.3-70b-versatile"


class CISPipeline:
    """Orchestrates the 5-layer Cognitive Immune System pipeline."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize all 5 layers and prepare the pipeline."""
        self._db_path = db_path
        init_db(db_path)

        self.quarantine_engine = QuarantineEngine()
        self.causal_memory = CausalDAG()
        self.tolerance_calibrator = ToleranceCalibrator(db_path=db_path)

        # Load persisted causal memory from previous sessions
        self.causal_memory.load_from_db(db_path=db_path)

        logger.info("CIS Pipeline initialized — all 5 layers ready.")

    async def analyze(
        self,
        query: str,
        context: str = "",
        session_id: Optional[str] = None,
        ablation_mode: str = "full",
    ) -> dict[str, Any]:
        """Run the full CIS pipeline: extract -> score -> quarantine -> record -> calibrate."""
        start_time = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())[:8]

        logger.info("=" * 60)
        logger.info("CIS ANALYSIS START — session=%s", session_id)
        logger.info("Query: %.100s...", query)
        logger.info("=" * 60)

        # Step 0: Get raw LLM response using context-grounded prompting
        raw_answer = await self._get_llm_response(query, context)

        if not raw_answer:
            return self._error_result("Failed to get LLM response", start_time)

        # Step 1: Extract entity-grounded claims (Layer 1 v2)
        if context:
            claims = await extract_grounded_claims(raw_answer, context)
        else:
            claims = await extract_claims(raw_answer)

        if not claims:
            return self._build_result(
                answer=raw_answer, claims=[], quarantined=[], safe_claims=[],
                causal_traces=[], start_time=start_time, session_id=session_id,
            )

        # Step 2-5: Score, quarantine, record, calibrate each claim
        self.quarantine_engine.clear()
        quarantined_details: list[dict[str, Any]] = []
        safe_details: list[dict[str, Any]] = []
        causal_traces: list[dict[str, Any]] = []

        for claim in claims:
            if not claim.get("verifiable", True):
                self.quarantine_engine.process_claim(
                    claim,
                    {"score": 0.0, "contaminated": False, "reason": "Non-verifiable claim"},
                    session_id=session_id,
                )
                safe_details.append({
                    "id": claim["id"], "text": claim["claim"], "score": 0.0,
                })
                continue

            # Layer 5 Check FIRST: Is this claim in the tolerance set T?
            if self.tolerance_calibrator.is_safe(claim["claim"]):
                self.quarantine_engine.process_claim(
                    claim,
                    {"score": 0.0, "contaminated": False, "reason": "In tolerance set T"},
                    session_id=session_id,
                )
                safe_details.append({
                    "id": claim["id"], "text": claim["claim"], "score": 0.0,
                })
                logger.info("Claim %s bypassed via tolerance set T.", claim["id"])
                continue

            # Layer 2: Score the claim (Wikipedia + Semantic Entropy + Causal)
            score_result = await score_claim(
                claim_text=claim["claim"],
                causal_memory=self.causal_memory,
                user_context=context,
                entity_wikipedia_title=claim.get("entity_wikipedia_title"),
                source_entity=claim.get("source_entity"),
                ablation_mode=ablation_mode,
            )

            # Layer 3: Quarantine or pass
            status = self.quarantine_engine.process_claim(
                claim, score_result, session_id=session_id
            )

            if status == "QUARANTINED":
                quarantined_details.append({
                    "id": claim["id"],
                    "text": claim["claim"],
                    "score": score_result["score"],
                    "reason": score_result["reason"],
                    "wiki_match": score_result.get("wiki_match", False),
                    "wiki_score": score_result.get("wiki_score", 0.5),
                    "cons_score": score_result.get("cons_score", 0.5),
                    "confidence": score_result.get("confidence", 0.0),
                    "causal_flag": score_result.get("causal_flag", False),
                })

                # Layer 4: Record in CC-DAG
                cause = self._extract_cause(claim["claim"], score_result)
                event_id = self.causal_memory.add_contamination_event(
                    claim_text=claim["claim"],
                    cause=cause,
                    score=score_result["score"],
                    session_id=session_id,
                    db_path=self._db_path,
                )
                trace = self.causal_memory.get_causal_trace(event_id)
                for t in trace:
                    causal_traces.append({
                        "from": t.get("cause", ""),
                        "to": t.get("text", ""),
                        "relation": "caused_contamination",
                    })
            else:
                safe_details.append({
                    "id": claim["id"],
                    "text": claim["claim"],
                    "score": score_result["score"],
                    "wiki_match": score_result.get("wiki_match", False),
                    "wiki_score": score_result.get("wiki_score", 0.5),
                    "cons_score": score_result.get("cons_score", 0.5),
                    "confidence": score_result.get("confidence", 0.0),
                    "causal_flag": score_result.get("causal_flag", False),
                })

                # Layer 5: Calibrate — add to tolerance set if Wikipedia is confident
                wiki_confidence = 1.0 - score_result.get("wiki_score", 1.0)
                await self.tolerance_calibrator.calibrate(
                    claim["claim"], wiki_confidence
                )

            # Rate limit: pause between claims to respect Groq free tier (100K tokens/day)
            await asyncio.sleep(1.5)

        # Generate filtered answer using only safe claims
        filtered_answer = await self._generate_filtered_answer(
            query, safe_details, raw_answer
        )

        return self._build_result(
            answer=filtered_answer,
            claims=claims,
            quarantined=quarantined_details,
            safe_claims=safe_details,
            causal_traces=causal_traces,
            start_time=start_time,
            session_id=session_id,
            raw_answer=raw_answer,
        )

    async def analyze_baseline(
        self, query: str, context: str = ""
    ) -> dict[str, Any]:
        """Get a raw LLM response WITHOUT CIS — used as experiment baseline."""
        start_time = time.perf_counter()

        answer = await self._get_llm_response(query, context)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "answer": answer or "",
            "latency_ms": latency_ms,
            "system": "baseline",
        }

    async def _get_llm_response(self, query: str, context: str = "") -> str:
        """Get a response from the Groq LLM using context-grounded prompting.
        
        Context-Grounded Prompting Theory:
            When context is provided, the LLM is instructed to treat it as the
            primary source of truth. This simulates real-world scenarios where
            the LLM receives information from external systems (RAG, tools, 
            documents) that it must incorporate into its reasoning.
            
            If the context contains contaminated information, the LLM will
            likely propagate it — which is exactly the scenario CIS is
            designed to detect and quarantine.
        """
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not set.")
            return ""

        client = AsyncGroq(api_key=GROQ_API_KEY)

        messages: list[dict[str, str]] = []

        if context:
            # Context-grounded prompt: makes LLM rely on provided context
            # This is critical for experiment validity — we need the LLM to
            # actually USE the (potentially contaminated) context
            messages.append({
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question below using "
                    "the provided context as your primary source of information. "
                    "Base your answer on the facts stated in the context. "
                    "Provide a detailed, factual answer."
                ),
            })
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            })
        else:
            messages.append({"role": "user", "content": query})

        try:
            response = await client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Groq LLM response failed: %s", e)
            return ""

    async def _generate_filtered_answer(
        self,
        query: str,
        safe_claims: list[dict[str, Any]],
        raw_answer: str,
    ) -> str:
        """Generate a final answer using ONLY safe (non-quarantined) claims as context."""
        if not safe_claims:
            return "All claims were quarantined. No reliable information available."

        if not GROQ_API_KEY:
            return raw_answer

        safe_context = "\n".join(
            f"- {c['text']}" for c in safe_claims if c.get("text")
        )

        client = AsyncGroq(api_key=GROQ_API_KEY)

        try:
            response = await client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise answering system. Answer the question "
                            "using ONLY the verified facts provided below. Do not add "
                            "any information beyond what is in the verified facts.\n\n"
                            f"VERIFIED FACTS:\n{safe_context}"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            return response.choices[0].message.content or raw_answer
        except Exception as e:
            logger.error("Filtered answer generation failed: %s", e)
            return raw_answer

    def _extract_cause(
        self, claim_text: str, score_result: dict[str, Any]
    ) -> str:
        """Extract the primary contamination cause from score components."""
        causes: list[str] = []
        if score_result.get("wiki_score", 0) >= 0.8:
            causes.append("wikipedia_contradiction")
        if score_result.get("confidence_score", 0) > 0.5:
            causes.append("low_model_confidence")
        if score_result.get("causal_flag", False):
            causes.append("causal_ancestor_contaminated")
        return " + ".join(causes) if causes else "multi_signal_threshold"

    def _build_result(
        self,
        answer: str,
        claims: list[dict[str, Any]],
        quarantined: list[dict[str, Any]],
        safe_claims: list[dict[str, Any]],
        causal_traces: list[dict[str, Any]],
        start_time: float,
        session_id: str,
        raw_answer: str = "",
    ) -> dict[str, Any]:
        """Construct the standardized pipeline output."""
        total = len(claims)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        contamination_rate = len(quarantined) / total if total > 0 else 0.0

        result: dict[str, Any] = {
            "answer": answer,
            "raw_answer": raw_answer,
            "claims_total": total,
            "claims_verifiable": sum(1 for c in claims if c.get("verifiable", True)),
            "contamination_rate": round(contamination_rate, 4),
            "quarantined": quarantined,
            "safe": safe_claims,
            "causal_trace": causal_traces,
            "latency_ms": latency_ms,
            "session_id": session_id,
            "graph_data": self.quarantine_engine.get_graph_data(),
            "tolerance_set_size": self.tolerance_calibrator.get_registry_size(),
            "containment_depth_bound": QuarantineEngine.containment_depth_bound(
                rho=min(max(contamination_rate, 0.01), 0.999), epsilon=0.01
            ) if contamination_rate > 0 else None,
        }

        logger.info(
            "CIS ANALYSIS COMPLETE — claims=%d, quarantined=%d (%.1f%%), latency=%dms",
            total, len(quarantined), contamination_rate * 100, latency_ms,
        )

        return result

    def _error_result(self, error_msg: str, start_time: float) -> dict[str, Any]:
        """Return an error result when the pipeline fails."""
        return {
            "answer": f"Error: {error_msg}", "raw_answer": "",
            "claims_total": 0, "claims_verifiable": 0,
            "contamination_rate": 0.0, "quarantined": [],
            "safe": [], "causal_trace": [],
            "latency_ms": int((time.perf_counter() - start_time) * 1000),
            "session_id": "", "graph_data": {},
            "tolerance_set_size": 0, "containment_depth_bound": None,
            "error": error_msg,
        }
