"""
src/kg/extractor.py
────────────────────
Orchestrates the full extraction pipeline for a domain:

    1. Load entity list from sample_entities/<domain>.json
    2. Scrape raw text per entity (Wikipedia / Gutenberg)
    3. Call GPT-4o with the universal extraction prompt
    4. Parse and validate the JSON response
    5. Save one JSON file per entity to data/output/<domain>/

Run directly via scripts/build_kg.py
"""

import os
import re
import json
import time
import logging
from pathlib import Path

from openai import OpenAI
from src.kg.scraper import scrape
from src.data.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "output"


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str = "gpt-4o", max_tokens: int = 2048) -> str:
    """
    Send extraction prompt to GPT-4o and return raw response string.

    Args:
        prompt     : fully formatted extraction prompt
        model      : OpenAI model name
        max_tokens : max tokens for the response

    Returns:
        Raw string response from the model.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ── JSON cleaning and parsing ─────────────────────────────────────────────────

def parse_response(raw: str, entity_name: str) -> dict | None:
    """
    Clean and parse the LLM JSON response.

    Handles common issues:
        - Markdown fences (```json ... ```)
        - Leading/trailing whitespace
        - Truncated responses

    Args:
        raw         : raw LLM response string
        entity_name : used for logging only

    Returns:
        Parsed dict, or None if parsing fails after cleaning.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse failed for '{entity_name}': {e}")

        # Attempt to extract the JSON object manually
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"  Could not recover JSON for '{entity_name}' — skipping.")
        return None


# ── Save output ───────────────────────────────────────────────────────────────

def save_entity_json(data: dict, entity_name: str, domain: str) -> Path:
    """
    Save extracted entity JSON to data/output/<domain>/<entity_name>.json

    Args:
        data        : parsed entity dict
        entity_name : used for filename
        domain      : used for subdirectory

    Returns:
        Path to the saved file.
    """
    output_dir = OUTPUT_ROOT / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    filename = entity_name.lower().replace(" ", "_") + ".json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.info(f"  Saved: {filepath}")
    return filepath


# ── Per-entity pipeline ───────────────────────────────────────────────────────

def extract_entity(entity: dict, domain: str, model: str = "gpt-4o") -> dict | None:
    """
    Run the full extraction pipeline for a single entity.

    Args:
        entity : dict with "name" and "wiki_search" keys
        domain : domain string
        model  : OpenAI model to use

    Returns:
        Parsed entity dict, or None if extraction fails.
    """
    name        = entity["name"]
    wiki_search = entity.get("wiki_search", name)

    logger.info(f"Processing: '{name}'")

    # Step 1 — Scrape
    source_text = scrape(name, wiki_search, domain)
    if not source_text:
        logger.warning(f"  No source text for '{name}' — skipping.")
        return None

    # Step 2 — Build prompt
    prompt = build_extraction_prompt(name, domain, source_text)

    # Step 3 — Call LLM
    try:
        raw = call_llm(prompt, model=model)
    except Exception as e:
        logger.error(f"  LLM call failed for '{name}': {e}")
        return None

    # Step 4 — Parse
    data = parse_response(raw, name)
    if data is None:
        return None

    # Step 5 — Save
    save_entity_json(data, name, domain)

    return data


# ── Domain-level pipeline ─────────────────────────────────────────────────────

def extract_domain(
    domain: str,
    entity_list_path: Path,
    model: str = "gpt-4o",
    sleep_between: float = 1.0,
) -> list[dict]:
    """
    Run extraction for all entities in a domain entity list.

    Args:
        domain           : domain string (e.g. "indian_mythology")
        entity_list_path : path to the sample_entities/<domain>.json file
        model            : OpenAI model to use
        sleep_between    : seconds to wait between API calls (rate limiting)

    Returns:
        List of successfully extracted entity dicts.
    """
    # Load entity list
    with open(entity_list_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    logger.info(f"Starting extraction: domain='{domain}', {len(entities)} entities")

    results = []
    failed  = []

    for i, entity in enumerate(entities):
        result = extract_entity(entity, domain, model=model)

        if result:
            results.append(result)
        else:
            failed.append(entity["name"])

        # Rate limiting — be polite to APIs
        if i < len(entities) - 1:
            time.sleep(sleep_between)

    # Summary
    logger.info(f"\nExtraction complete for '{domain}':")
    logger.info(f"  Succeeded : {len(results)}/{len(entities)}")
    if failed:
        logger.warning(f"  Failed    : {failed}")

    return results
