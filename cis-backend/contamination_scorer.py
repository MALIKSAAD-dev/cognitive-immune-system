"""
contamination_scorer.py — Layer 2: Multi-Signal Contamination Scoring (CIS v3)

Scientific goal:
    Replace the unreliable "Wikipedia contradiction via general LLM" probe with a
    dedicated NLI model that correctly classifies semantically equivalent
    contradictions (MNLI-trained DeBERTa-v3 cross-encoder).

Core scoring function (unchanged from v2):
    phi(c) = 0.60*S_wiki + 0.30*S_cons + 0.10*S_causal
    quarantine if phi(c) >= 0.55

v3 changes (in strict order requested):
    1) S_wiki: entity-grounded Wikipedia check using NLI contradiction detection
    2) S_cons: semantic entropy (3 paraphrases, NLI consistency)
    3) Ablation support: "full", "wiki_only", "cons_only", "none"
"""

import asyncio
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from groq import AsyncGroq
import wikipediaapi

from transformers import pipeline as hf_pipeline

load_dotenv()

logger = logging.getLogger("cis.contamination_scorer")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
SCORING_MODEL_ID: str = "llama-3.1-8b-instant"

W_WIKI: float = 0.60
W_CONS: float = 0.30
W_CAUSAL: float = 0.10

CONTAMINATION_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# NLI model (module-level, single shared instance)
# ---------------------------------------------------------------------------
nli_model = hf_pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-base",
    device=-1,
    top_k=None,
)


def get_nli_score(premise: str, hypothesis: str) -> float:
    """
    Run NLI between a Wikipedia sentence and a claim.
    Returns contamination score:
      1.0 = CONTRADICTION (claim contradicts Wikipedia = contaminated)
      0.0 = ENTAILMENT (claim confirmed by Wikipedia = safe)
      0.5 = NEUTRAL (Wikipedia doesn't address claim = uncertain)

    DeBERTa-v3 trained on MNLI correctly handles:
      premise: "Eric Ambler was a British novelist"
      hypothesis: "Eric Ambler is a British artist"
      → CONTRADICTION → returns 1.0
    """
    if not premise or not hypothesis:
        return 0.5

    try:
        results = nli_model(
            f"{premise} [SEP] {hypothesis}",
            truncation=True,
            max_length=512,
        )

        # HF returns a list; with top_k=None, it returns a list-of-list
        if isinstance(results, list) and results and isinstance(results[0], list):
            rows = results[0]
        else:
            rows = results  # type: ignore[assignment]

        scores = {r["label"].upper(): float(r["score"]) for r in rows}
        contradiction_score = scores.get("CONTRADICTION", 0.0)
        entailment_score = scores.get("ENTAILMENT", 0.0)
        neutral_score = scores.get("NEUTRAL", 0.0)

        _ = neutral_score  # neutral_score kept for explicitness, not used directly

        if contradiction_score > 0.5:
            return 1.0
        elif entailment_score > 0.5:
            return 0.0
        else:
            return 0.5
    except Exception as e:
        logger.warning("NLI scoring failed: %s", e)
        return 0.5


# ---------------------------------------------------------------------------
# Wikipedia client (summary-only; entity grounded)
# ---------------------------------------------------------------------------
_wiki_client = wikipediaapi.Wikipedia("CIS-Research/3.0", "en")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def check_claim_in_context(claim: str, context: str) -> bool:
    """
    If the claim text is substantially present in the user-provided
    context, never quarantine it. The user explicitly provided
    this information.
    """
    claim_words = set(claim.lower().split())
    context_words = set(context.lower().split())
    overlap = len(claim_words & context_words) / max(len(claim_words), 1)
    return overlap > 0.6


_groq_client: Optional[AsyncGroq] = None


def _get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    return _groq_client


# ---------------------------------------------------------------------------
# Signal 1: Wikipedia NLI contradiction detection (S_wiki)
# ---------------------------------------------------------------------------
async def compute_wiki_score_nli(claim: str, entity_wikipedia_title: str) -> float:
    """
    Entity-grounded Wikipedia check with NLI contradiction detection.
    Uses the resolved Wikipedia title from v2 entity grounding.
    Finds the most relevant sentence in the Wikipedia summary.
    Runs NLI between that sentence and the claim.
    """
    if not entity_wikipedia_title:
        return 0.5

    try:
        loop = asyncio.get_event_loop()
        page = await loop.run_in_executor(None, _wiki_client.page, entity_wikipedia_title)

        if not page.exists():
            return 0.5

        summary = (page.summary or "")[:2000]
        sentences = [s.strip() for s in summary.split(".") if len(s.strip()) > 20]
        if not sentences:
            return 0.5

        claim_words = set(claim.lower().split())
        best_sentence = max(
            sentences,
            key=lambda s: len(set(s.lower().split()) & claim_words),
        )

        return get_nli_score(best_sentence, claim)
    except Exception as e:
        logger.warning("Wikipedia NLI failed: %s", e)
        return 0.5


# ---------------------------------------------------------------------------
# Signal 2: Semantic entropy via NLI consistency (S_cons)
# ---------------------------------------------------------------------------
async def compute_semantic_entropy(claim: str) -> float:
    """
    Semantic entropy: if rephrasing a claim produces inconsistent versions,
    the claim is likely false or uncertain. High entropy = high contamination score.

    Scientific basis: Farquhar et al. 2024 (Nature) shows semantic entropy detects
    hallucinations without model internals.
    """
    if not GROQ_API_KEY:
        return 0.5

    try:
        client = _get_groq_client()
        phrasings: list[str] = []
        for _ in range(3):
            r = await client.chat.completions.create(
                model=SCORING_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rephrase this factual claim in different words. Keep the same "
                            "meaning if true. One sentence only: "
                            f"{claim}"
                        ),
                    }
                ],
                temperature=0.8,
                max_tokens=80,
            )
            phr = (r.choices[0].message.content or "").strip()
            if phr:
                phrasings.append(phr)
            await asyncio.sleep(0.2)

        if len(phrasings) < 3:
            return 0.5

        contradictions = 0
        pairs = [
            (phrasings[0], phrasings[1]),
            (phrasings[0], phrasings[2]),
            (phrasings[1], phrasings[2]),
        ]

        loop = asyncio.get_event_loop()
        for p1, p2 in pairs:
            score = await loop.run_in_executor(None, get_nli_score, p1, p2)
            if score > 0.7:
                contradictions += 1

        return contradictions / 3.0
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main scoring API used by pipeline.py
# ---------------------------------------------------------------------------
async def score_claim(
    claim_text: str,
    causal_memory: Optional[Any] = None,
    user_context: str = "",
    entity_wikipedia_title: Optional[str] = None,
    source_entity: Optional[str] = None,
    ablation_mode: str = "full",
) -> dict[str, Any]:
    """
    Compute phi(c) using three independent signals.

    Ablation modes:
      - "full": S_wiki + S_cons + S_causal (CIS v3)
      - "wiki_only": S_wiki only (S_cons=0.5 constant)
      - "cons_only": S_cons only (S_wiki=0.5 constant)
      - "none": no CIS scoring (used only if caller explicitly wants it)
    """
    claim_text = claim_text or ""
    if not claim_text.strip():
        return _empty_score("Empty claim text")

    if ablation_mode == "none":
        return _safe_result("Baseline mode (no CIS)")

    # False positive protection: if claim is already in user context, avoid quarantining.
    if user_context and _normalize(claim_text) in _normalize(user_context):
        return _safe_result("SAFE | Claim is substring of user context (FP protection)")

    # False positive protection: if claim words substantially overlap with user context,
    # the user explicitly provided this information — never quarantine.
    if user_context and check_claim_in_context(claim_text, user_context):
        return _safe_result("claim_present_in_context")

    # S_wiki
    if ablation_mode in ("full", "wiki_only"):
        title = entity_wikipedia_title or ""
        wiki_score = await compute_wiki_score_nli(claim_text, title)
    else:
        wiki_score = 0.5

    # S_cons
    if ablation_mode in ("full", "cons_only"):
        cons_score = await compute_semantic_entropy(claim_text)
    else:
        cons_score = 0.5

    # S_causal
    causal_flag = False
    if causal_memory is not None:
        try:
            causal_flag = bool(causal_memory.has_causal_ancestor(claim_text))
        except Exception as e:
            logger.warning("Causal check failed: %s", e)
            causal_flag = False
    causal_score = 1.0 if causal_flag else 0.0

    phi = round(W_WIKI * wiki_score + W_CONS * cons_score + W_CAUSAL * causal_score, 4)
    contaminated = phi >= CONTAMINATION_THRESHOLD

    result: dict[str, Any] = {
        "score": phi,
        "wiki_match": wiki_score >= 0.8,
        "wiki_score": round(wiki_score, 4),
        "cons_score": round(cons_score, 4),
        "confidence": round(10.0 * (1.0 - cons_score), 2),
        "confidence_score": round(cons_score, 4),
        "causal_flag": causal_flag,
        "causal_score": causal_score,
        "contaminated": contaminated,
        "reason": _build_reason(wiki_score, cons_score, causal_flag, contaminated, ablation_mode),
        "threshold": CONTAMINATION_THRESHOLD,
    }

    logger.info(
        "phi=%.4f wiki=%.2f cons=%.2f causal=%d -> %s (%s) claim='%.80s'",
        phi,
        wiki_score,
        cons_score,
        int(causal_flag),
        "CONTAMINATED" if contaminated else "SAFE",
        ablation_mode,
        claim_text,
    )
    return result


def _build_reason(
    wiki_score: float,
    cons_score: float,
    causal_flag: bool,
    contaminated: bool,
    ablation_mode: str,
) -> str:
    parts: list[str] = []
    parts.append("CONTAMINATED" if contaminated else "SAFE")
    parts.append(f"mode={ablation_mode}")
    parts.append(f"S_wiki={wiki_score:.2f}")
    parts.append(f"S_cons={cons_score:.2f}")
    if causal_flag:
        parts.append("S_causal=1.0")
    else:
        parts.append("S_causal=0.0")
    return " | ".join(parts)


def _empty_score(reason: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "wiki_match": False,
        "wiki_score": 0.0,
        "cons_score": 0.0,
        "confidence": 0.0,
        "confidence_score": 0.0,
        "causal_flag": False,
        "causal_score": 0.0,
        "contaminated": False,
        "reason": reason,
        "threshold": CONTAMINATION_THRESHOLD,
    }


def _safe_result(reason: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "wiki_match": False,
        "wiki_score": 0.0,
        "cons_score": 0.0,
        "confidence": 10.0,
        "confidence_score": 0.0,
        "causal_flag": False,
        "causal_score": 0.0,
        "contaminated": False,
        "reason": reason,
        "threshold": CONTAMINATION_THRESHOLD,
    }
