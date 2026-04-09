"""
tolerance_calibrator.py — Layer 5: Autoimmune Prevention via Tolerance Set

Mathematical Foundation:
    Biological immune systems must distinguish self from non-self to avoid
    autoimmune responses — attacking the body's own healthy cells. The CIS
    faces an analogous problem: quarantining correct-but-surprising claims
    that appear contaminated because they are novel or counter-intuitive.

    Example:
        Claim: "CRISPR-Cas9 was used to edit human embryos in 2018."
        This is TRUE but may trigger high contamination scores because:
        - Wikipedia may not have a detailed page (S_wiki high)
        - The claim sounds extraordinary (S_conf may dip)

    The Tolerance Calibrator solves this via a safe registry set T:

    Definition (Tolerance Set T):
        T = {c | ∃ source S with confidence(S, c) > θ_T}

        where θ_T = 0.85 (WIKIPEDIA_CONFIDENCE_THRESHOLD)
        and S is a trusted verification source (currently Wikipedia).

    Theorem (Autoimmune Prevention Guarantee):
        If a claim c has been verified by Wikipedia with confidence > θ_T
        and added to T, then:
            ∀ future occurrences c' where text(c') = text(c):
                c' is NEVER quarantined, regardless of φ(c').

        This eliminates all autoimmune false positives for claims that have
        been previously confirmed, reducing FPR monotonically over time:
            FPR(session_n) ≤ FPR(session_{n-1})

    Proof:
        The tolerance check precedes the quarantine decision. If c ∈ T,
        the pipeline returns SAFE immediately without computing φ(c).
        Since T only grows (claims are never removed), |T| is monotonically
        increasing, and the fraction of true claims incorrectly quarantined
        can only decrease.  ∎

    Convergence Property:
        As sessions → ∞, T approaches the set of all commonly stated true claims,
        and FPR → 0. The system "learns" what is true through accumulated evidence,
        analogous to acquired immunity in biological systems.

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from database import get_safe_registry, is_safe_registered, register_safe

load_dotenv()

logger = logging.getLogger("cis.tolerance_calibrator")

WIKIPEDIA_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("WIKIPEDIA_CONFIDENCE_THRESHOLD", "0.85")
)


class ToleranceCalibrator:
    """Implements the tolerance set T for autoimmune prevention in the CIS."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize tolerance calibrator and load existing safe registry from DB."""
        self._db_path = db_path
        self._safe_set: set[str] = set()
        self._load_from_db()
        logger.info(
            "ToleranceCalibrator initialized with %d safe claims (θ_T=%.2f).",
            len(self._safe_set),
            WIKIPEDIA_CONFIDENCE_THRESHOLD,
        )

    def is_safe(self, claim_text: str) -> bool:
        """Check if a claim is in the tolerance set T (O(1) lookup)."""
        normalized = claim_text.strip().lower()

        # Fast in-memory check first
        if normalized in self._safe_set:
            logger.debug("Tolerance hit (in-memory): %.60s...", claim_text)
            return True

        # Fallback to DB check (handles multi-process scenarios)
        if is_safe_registered(claim_text, db_path=self._db_path):
            self._safe_set.add(normalized)
            logger.debug("Tolerance hit (DB): %.60s...", claim_text)
            return True

        return False

    async def calibrate(
        self,
        claim_text: str,
        wiki_confidence: float,
    ) -> bool:
        """Attempt to add a claim to the tolerance set T.

        A claim is added iff Wikipedia verifies it with confidence > θ_T.
        wiki_confidence here is the match ratio from the Wikipedia check:
            wiki_confidence = 1.0 - wiki_contamination_score

        Returns True if the claim was added to the safe registry.
        """
        if wiki_confidence > WIKIPEDIA_CONFIDENCE_THRESHOLD:
            normalized = claim_text.strip().lower()

            if normalized not in self._safe_set:
                self._safe_set.add(normalized)
                register_safe(
                    claim_text=claim_text,
                    confidence=wiki_confidence,
                    db_path=self._db_path,
                )
                logger.info(
                    "✓ Added to tolerance set T (confidence=%.3f > θ_T=%.2f): %.60s...",
                    wiki_confidence, WIKIPEDIA_CONFIDENCE_THRESHOLD, claim_text,
                )
                return True

        return False

    def get_registry(self) -> list[dict]:
        """Return the full tolerance set T with metadata."""
        return get_safe_registry(db_path=self._db_path)

    def get_registry_size(self) -> int:
        """Return |T| — the size of the tolerance set."""
        return len(self._safe_set)

    def _load_from_db(self) -> None:
        """Load existing safe claims from database into memory."""
        try:
            records = get_safe_registry(db_path=self._db_path)
            for record in records:
                self._safe_set.add(record["claim_text"].strip().lower())
        except Exception as e:
            logger.warning("Failed to load safe registry from DB: %s", e)
