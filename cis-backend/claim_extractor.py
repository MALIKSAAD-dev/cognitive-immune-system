"""
claim_extractor.py - Layer 1: Entity-Grounded Claim Extraction (v2)

CIS v2 Key Innovation:
    Instead of extracting free-form atomic claims from the LLM response
    (which suffers from semantic drift via paraphrasing), we anchor every
    claim to a named entity from the original source context.

    This preserves the entity link through paraphrasing:
        Before (v1): LLM says "the wall was built in the 1800s"
            -> claim = "wall built in 1800s"
            -> Wikipedia search for "wall" FAILS
            -> S_wiki = 0.5 (uncertain), phi below threshold

        After (v2): source_context contains "Great Wall of China"
            -> ask LLM: "what does text say about Great Wall of China?"
            -> claim = "Great Wall of China was built in the 1800s"
            -> entity_wikipedia_title = "Great Wall of China" (pre-resolved)
            -> Wikipedia lookup SUCCEEDS, summary contradicts
            -> S_wiki = 1.0, phi >= 0.55 -> QUARANTINED

    Entity extraction uses LLM-based NER (via Groq) instead of spaCy,
    avoiding heavy ML dependencies while achieving comparable quality
    on proper nouns, which are the entities that matter for Wikipedia lookup.

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import wikipediaapi
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cis.claim_extractor")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
EXTRACTION_MODEL: str = "llama-3.1-8b-instant"

_wiki_client = wikipediaapi.Wikipedia(
    user_agent="CIS-Research/2.0 (Entity-Grounded Extraction; saad@research.org)",
    language="en",
)


async def extract_grounded_claims(
    response: str,
    source_context: str,
) -> list[dict[str, Any]]:
    """Extract entity-grounded claims from LLM response.

    This is the v2 extractor that fixes the semantic drift bottleneck.
    Every claim is anchored to a named entity from source_context,
    and the entity's Wikipedia title is pre-resolved.

    Args:
        response: The raw LLM response text.
        source_context: The original context that was provided to the LLM.

    Returns:
        List of grounded claim dicts with keys:
            id, claim, verifiable, source_entity, entity_type,
            entity_wikipedia_title
    """
    if not response or not response.strip():
        logger.warning("Empty response for claim extraction.")
        return []

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set.")
        return []

    # Step 1: Extract named entities from source_context, or from response if context is empty
    text_for_ner = source_context if source_context and source_context.strip() else response
    entities = await _extract_entities_llm(text_for_ner)
    if not entities:
        logger.warning("No entities found, falling back to v1 extraction.")
        return await _fallback_extract(response)

    logger.info("Found %d entities in source context: %s",
                len(entities), [e["name"] for e in entities])

    # Step 2: For each entity, ask what the response says about it
    client = AsyncGroq(api_key=GROQ_API_KEY)
    claims: list[dict[str, Any]] = []
    claim_idx = 0

    for entity in entities:
        ent_name = entity["name"]
        ent_type = entity["type"]

        # Check if entity is even mentioned in the response
        if ent_name.lower() not in response.lower():
            # Try partial match (first word or last word)
            name_parts = ent_name.split()
            found = False
            for part in name_parts:
                if len(part) > 3 and part.lower() in response.lower():
                    found = True
                    break
            if not found:
                continue

        try:
            r = await client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        f"The following text mentions {ent_name}. "
                        f"What does the text say about {ent_name}? "
                        f"State it as one factual claim in a single sentence. "
                        f"If the text does not mention {ent_name}, reply exactly: NOT_MENTIONED\n\n"
                        f"Text: {response}"
                    ),
                }],
                temperature=0.0,
                max_tokens=150,
            )
            claim_text = r.choices[0].message.content or ""
            claim_text = claim_text.strip()

            if "NOT_MENTIONED" in claim_text.upper():
                continue
            if len(claim_text) < 10:
                continue

            # Step 3 & 4: Resolve Wikipedia title
            wiki_title = await _resolve_wikipedia_title(ent_name)

            claim_idx += 1
            claims.append({
                "id": f"c{claim_idx}",
                "claim": claim_text,
                "verifiable": True,
                "source_entity": ent_name,
                "entity_type": ent_type,
                "entity_wikipedia_title": wiki_title,
            })

            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error("Failed to extract claim for entity '%s': %s", ent_name, e)
            continue

    if not claims:
        logger.warning("No grounded claims extracted, falling back to v1.")
        return await _fallback_extract(response)

    logger.info("Extracted %d entity-grounded claims.", len(claims))
    return claims


async def _extract_entities_llm(text: str) -> list[dict[str, str]]:
    """Use LLM to extract named entities from text.

    Returns list of {"name": ..., "type": ...} dicts.
    Types: PERSON, ORG, GPE, DATE, EVENT, WORK_OF_ART, LOC
    """
    if not text.strip():
        return []

    client = AsyncGroq(api_key=GROQ_API_KEY)

    try:
        r = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Extract all named entities from this text. "
                    "Include: people, organizations, locations, dates, events, "
                    "works of art, films, books, songs, and other proper nouns. "
                    "Return ONLY a JSON array like: "
                    '[{"name":"Albert Einstein","type":"PERSON"},'
                    '{"name":"MIT","type":"ORG"}]\n\n'
                    f"Text: {text[:2000]}"
                ),
            }],
            temperature=0.0,
            max_tokens=500,
        )

        raw = r.choices[0].message.content or ""
        return _parse_entity_json(raw)

    except Exception as e:
        logger.error("LLM entity extraction failed: %s", e)
        return _fallback_regex_entities(text)


def _parse_entity_json(raw: str) -> list[dict[str, str]]:
    """Parse LLM entity output into list of dicts."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [e for e in parsed if isinstance(e, dict) and "name" in e]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [e for e in parsed if isinstance(e, dict) and "name" in e]
        except json.JSONDecodeError:
            pass

    return _fallback_regex_entities(cleaned)


def _fallback_regex_entities(text: str) -> list[dict[str, str]]:
    """Regex-based entity extraction as a fallback."""
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    # Multi-word capitalized sequences (likely proper nouns)
    for match in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        name = match.group().strip()
        if name.lower() not in seen and len(name) > 3:
            seen.add(name.lower())
            entities.append({"name": name, "type": "ENTITY"})

    # Single capitalized words (skip common words)
    skip = {"the", "this", "that", "these", "those", "which", "what", "when",
            "where", "who", "how", "was", "were", "has", "have", "had",
            "not", "and", "but", "for", "are", "his", "her", "its",
            "also", "been", "being", "did", "does", "with", "from", "into",
            "they", "them", "their", "some", "many", "most", "such", "than",
            "however", "despite", "although", "because", "since", "while",
            "after", "before", "during", "through", "between", "against",
            "about", "both", "each", "other", "more", "only", "over"}
    for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        name = match.group(1)
        if name.lower() not in seen and name.lower() not in skip:
            seen.add(name.lower())
            entities.append({"name": name, "type": "ENTITY"})

    return entities[:15]  # Cap at 15 to control API calls


async def _resolve_wikipedia_title(entity_name: str) -> Optional[str]:
    """Try to find a Wikipedia page for the entity and return its title."""
    loop = asyncio.get_event_loop()

    # Try exact name first
    try:
        page = await loop.run_in_executor(None, _wiki_client.page, entity_name)
        if page.exists():
            return page.title
    except Exception:
        pass

    # Try without parenthetical disambiguation
    clean = re.sub(r"\s*\(.*?\)\s*", "", entity_name).strip()
    if clean != entity_name:
        try:
            page = await loop.run_in_executor(None, _wiki_client.page, clean)
            if page.exists():
                return page.title
        except Exception:
            pass

    return None


async def _fallback_extract(text: str) -> list[dict[str, Any]]:
    """v1-style extraction as fallback when no entities found."""
    client = AsyncGroq(api_key=GROQ_API_KEY)
    prompt = (
        "Decompose the text into atomic, independently verifiable claims. "
        "Each claim = one fact. Preserve names, dates, numbers exactly. "
        'Return ONLY JSON array: [{"id":"c1","claim":"...","verifiable":true}]'
    )
    try:
        r = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract atomic claims:\n\n{text}"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = r.choices[0].message.content or ""
        parsed = []
        try:
            # Simple JSON array extraction
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
        claims = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict) or "claim" not in item:
                continue
            claim_text = str(item.get("claim", ""))
            
            # Ground the V1 fallback claim via Regex so S_WIKI isn't bypassed!
            ents = _fallback_regex_entities(claim_text)
            source_ent = None
            wiki_title = None
            if ents:
                source_ent = ents[0]["name"]
                wiki_title = await _resolve_wikipedia_title(source_ent)

            claims.append({
                "id": item.get("id", f"c{i+1}"),
                "claim": claim_text,
                "verifiable": bool(item.get("verifiable", True)),
                "source_entity": source_ent,
                "entity_type": "UNKNOWN",
                "entity_wikipedia_title": wiki_title,
            })
        return claims
    except Exception as e:
        logger.error("Fallback extraction failed: %s", e)
        return []


# Keep backward compatibility
async def extract_claims(text: str) -> list[dict[str, Any]]:
    """Legacy v1 interface. Calls fallback extraction."""
    return await _fallback_extract(text)
