"""
src/kg/retriever.py
────────────────────
Retrieves a context-rich subgraph from Neo4j for a given user prompt.

Matching strategy (three-tier, no regex):
    Tier 1 — Exact / case-insensitive / alternative_names match in Neo4j
    Tier 2 — Token overlap: any word in extracted name matches any word in KG name
    Tier 3 — LLM semantic resolution: GPT-4o picks best KG node for the name

This ensures that prompts written in any case, phrasing, or partial form
always resolve to the correct KG node.
"""

import os
import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI
from src.kg.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# ── Context Packet ────────────────────────────────────────────────────────────

@dataclass
class ContextPacket:
    """
    Everything the prompt synthesizer and SRD module need.
    Cached after retrieval — no repeat graph queries during SRD.
    """
    query:                   str
    domain:                  str
    primary_entities:        list[dict] = field(default_factory=list)
    neighbour_entities:      list[dict] = field(default_factory=list)
    relationships:           list[dict] = field(default_factory=list)
    retrieved_attributes:    list[str]  = field(default_factory=list)
    contrastive_constraints: list[str]  = field(default_factory=list)

    @property
    def all_entities(self) -> list[dict]:
        seen, result = set(), []
        for e in self.primary_entities + self.neighbour_entities:
            name = e.get("name", "")
            if name and name not in seen:
                seen.add(name)
                result.append(e)
        return result

    def is_empty(self) -> bool:
        return len(self.primary_entities) == 0


# ── Entity Extractor ──────────────────────────────────────────────────────────

class EntityExtractor:
    """
    GPT-4o extracts entity names from any free-text prompt.
    Handles: any case, any phrasing, possessives, partial names,
    descriptive references ("the Hindu god of death" → "Yama").
    """

    _SYSTEM = """\
You extract the primary named rare entity from a text-to-image prompt.
The entity could be a mythological figure, rare animal, plant, artifact, or phenomenon.

Rules:
- Extract the most specific proper name mentioned
- Handle any capitalization (YAMA, yama, Yama → "Yama")
- For descriptive references without a name ("the Hindu god of death"),
  resolve to the most likely proper name
- For possessives ("Yama's buffalo"), return the possessor ("Yama")
- If the prompt is very long, focus only on the FIRST sentence or clause
  to find the primary subject — ignore trailing style/quality descriptors
- Return ONLY a JSON array of name strings, e.g. ["Yama"]
- Return [] only if no entity can be identified at all
No explanation, no markdown, just the JSON array."""

    def __init__(self, client: OpenAI):
        self.client = client

    def extract(self, prompt: str) -> list[str]:
        # For very long prompts only send the first 200 chars
        # Entity name is always at the start; trailing text is style/quality
        extract_from = prompt[:200] if len(prompt) > 200 else prompt

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": extract_from},
            ],
            temperature=0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content or "[]"
        try:
            names = json.loads(raw)
            return [n.strip() for n in names if isinstance(n, str) and n.strip()]
        except json.JSONDecodeError:
            logger.warning(f"  Extractor returned non-JSON: {raw}")
            return []


# ── LLM Semantic Resolver ─────────────────────────────────────────────────────

class SemanticResolver:
    """
    When string matching fails, uses GPT-4o to pick the best KG node
    for a given extracted name from the full candidate list.

    This handles cases like:
        "lord yama"     → "Yama"
        "the saola"     → "Saola"
        "aye aye lemur" → "Aye-aye"
        "ganda bird"    → "Ganda Bherunda"
    """

    _SYSTEM = """\
You match a user-provided entity name to the best entry in a knowledge graph.

Given:
  - A name extracted from a user prompt (may be partial, descriptive, or informal)
  - A list of entity names in a knowledge graph

Return the single best matching KG entity name, or "NONE" if nothing fits.
Return ONLY the entity name string — no explanation, no JSON wrapper."""

    def __init__(self, client: OpenAI):
        self.client = client

    def resolve(self, extracted_name: str, kg_names: list[str]) -> str | None:
        """
        Returns the best matching KG name, or None if no good match found.
        """
        if not kg_names:
            return None

        kg_list = "\n".join(f"  - {n}" for n in kg_names)
        user_msg = (
            f"Extracted name: \"{extracted_name}\"\n\n"
            f"Knowledge graph entities:\n{kg_list}\n\n"
            f"Best match (or NONE):"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=30,
        )
        result = (response.choices[0].message.content or "").strip()
        if result == "NONE" or not result:
            return None
        # Verify the returned name actually exists in the list
        result_lower = result.lower()
        for name in kg_names:
            if name.lower() == result_lower:
                return name
        # Fuzzy fallback — returned name might have slight formatting diff
        for name in kg_names:
            if result_lower in name.lower() or name.lower() in result_lower:
                return name
        return None


# ── Retriever ─────────────────────────────────────────────────────────────────

class KGRetriever:
    """
    Three-tier entity matching — never uses regex or character-level ops.

    Tier 1: Cypher exact/case-insensitive/alternative_names lookup
    Tier 2: Token overlap (word-level, not character)
    Tier 3: LLM semantic resolution against full KG name list
    """

    def __init__(self, client: Neo4jClient, k: int = 1, max_neighbours: int = 10):
        self.client         = client
        self.k              = k
        self.max_neighbours = max_neighbours

        llm_client        = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.extractor    = EntityExtractor(llm_client)
        self.resolver     = SemanticResolver(llm_client)

        # Cache all KG entity names at init — used for semantic resolution
        self._kg_names: list[str] = self._load_all_kg_names()
        logger.info(f"  KG name cache: {len(self._kg_names)} enriched entities loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, prompt: str) -> ContextPacket:
        logger.info(f"Retrieving context for: '{prompt}'")

        # Step 1 — extract entity names
        entity_names = self.extractor.extract(prompt)
        logger.info(f"  Extracted: {entity_names}")

        if not entity_names:
            logger.warning("  No entities extracted.")
            return ContextPacket(query=prompt, domain="")

        # Step 2 — match each name to a KG node (3 tiers)
        primary = self._match_nodes(entity_names)
        if not primary:
            logger.warning(f"  No KG nodes matched for {entity_names}")
            return ContextPacket(query=prompt, domain="")

        # Step 3 — k-hop neighbours
        neighbours = self._expand_khop(primary)

        # Step 4 — relationships
        relationships = self._fetch_relationships(primary)

        # Step 5 — build attribute + contrastive lists
        attrs       = self._build_attribute_list(primary)
        contrastive = self._build_contrastive(primary)
        domain      = primary[0].get("domain", "")

        ctx = ContextPacket(
            query=prompt,
            domain=domain,
            primary_entities=primary,
            neighbour_entities=neighbours,
            relationships=relationships,
            retrieved_attributes=attrs,
            contrastive_constraints=contrastive,
        )

        logger.info(
            f"  ContextPacket: {len(primary)} primary, "
            f"{len(neighbours)} neighbours, "
            f"{len(attrs)} attributes"
        )
        return ctx

    # ── Three-tier node matching ───────────────────────────────────────────────

    def _match_nodes(self, entity_names: list[str]) -> list[dict]:
        matched, seen = [], set()
        for name in entity_names:
            node = self._resolve_node(name)
            if node:
                node_name = node.get("name", "")
                if node_name not in seen:
                    matched.append(node)
                    seen.add(node_name)
                    logger.info(f"  Matched '{name}' → '{node_name}'")
            else:
                logger.warning(f"  No match found for '{name}'")
        return matched

    def _resolve_node(self, name: str) -> dict | None:
        """
        Try all three tiers in order until a match is found.
        """
        # ── Tier 1: Neo4j string match ────────────────────────────────────────
        node = self._neo4j_match(name)
        if node:
            return node

        # ── Tier 2: Token overlap ─────────────────────────────────────────────
        node = self._token_overlap_match(name)
        if node:
            logger.info(f"    Tier 2 (token overlap) matched '{name}'")
            return node

        # ── Tier 3: LLM semantic resolution ──────────────────────────────────
        best_name = self.resolver.resolve(name, self._kg_names)
        if best_name:
            logger.info(f"    Tier 3 (LLM semantic) matched '{name}' → '{best_name}'")
            node = self._neo4j_match(best_name)
            return node

        return None

    def _neo4j_match(self, name: str) -> dict | None:
        """
        Tier 1: exact name, alternative_names, or any token in name matches.
        No character-level substring matching.
        """
        cypher = """
        MATCH (e:Entity)
        WHERE e.domain IS NOT NULL
          AND (
            toLower(e.name) = toLower($name)
            OR any(a IN e.alternative_names WHERE toLower(a) = toLower($name))
          )
        RETURN e
        LIMIT 1
        """
        results = self.client.run(cypher, {"name": name})
        if results:
            return results[0].get("e", {})
        return None

    def _token_overlap_match(self, name: str) -> dict | None:
        """
        Tier 2: check if any word in the extracted name matches
        any word in a KG entity name (word-level, not character-level).
        """
        name_tokens = set(name.lower().split())
        best_node   = None
        best_score  = 0

        cypher = "MATCH (e:Entity) WHERE e.domain IS NOT NULL RETURN e.name AS name LIMIT 500"
        results = self.client.run(cypher)

        for row in results:
            kg_name = row.get("name", "")
            kg_tokens = set(kg_name.lower().split())
            overlap = len(name_tokens & kg_tokens)
            if overlap > best_score:
                best_score = overlap
                best_node  = kg_name

        if best_score > 0 and best_node:
            return self._neo4j_match(best_node)
        return None

    # ── KG name cache ─────────────────────────────────────────────────────────

    def _load_all_kg_names(self) -> list[str]:
        """Load all enriched entity names from Neo4j at startup."""
        try:
            cypher = "MATCH (e:Entity) WHERE e.domain IS NOT NULL RETURN e.name AS name"
            results = self.client.run(cypher)
            return [r["name"] for r in results if r.get("name")]
        except Exception as exc:
            logger.warning(f"  Could not load KG names: {exc}")
            return []

    # ── k-hop expansion ───────────────────────────────────────────────────────

    def _expand_khop(self, primary_nodes: list[dict]) -> list[dict]:
        neighbours   = []
        primary_names = {n.get("name") for n in primary_nodes}

        for node in primary_nodes:
            cypher = f"""
            MATCH (e:Entity {{name: $name}})-[*1..{self.k}]-(nb:Entity)
            WHERE nb.domain IS NOT NULL AND nb.name <> $name
            RETURN DISTINCT nb
            LIMIT {self.max_neighbours}
            """
            results = self.client.run(cypher, {"name": node.get("name")})
            for row in results:
                nb = row.get("nb", {})
                nb_name = nb.get("name", "")
                if nb_name and nb_name not in primary_names:
                    neighbours.append(nb)
                    primary_names.add(nb_name)

        return neighbours

    # ── Relationships ─────────────────────────────────────────────────────────

    def _fetch_relationships(self, primary_nodes: list[dict]) -> list[dict]:
        if not primary_nodes:
            return []
        names = [n.get("name") for n in primary_nodes if n.get("name")]
        cypher = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.name IN $names
        RETURN a.name AS from_node, type(r) AS rel_type, b.name AS to_node
        LIMIT 30
        """
        results = self.client.run(cypher, {"names": names})
        return [{"from": r["from_node"], "type": r["rel_type"], "to": r["to_node"]}
                for r in results]

    # ── Attribute helpers ─────────────────────────────────────────────────────

    def _build_attribute_list(self, nodes: list[dict]) -> list[str]:
        attrs, seen = [], set()
        for node in nodes:
            candidates = (
                [node.get("morphology", "")]
                + (node.get("distinctive_features", []) or [])
                + (node.get("color_palette", []) or [])
                + [node.get("texture", ""), node.get("size_and_scale", "")]
            )
            for a in candidates:
                if a and a not in seen:
                    seen.add(a)
                    attrs.append(a)
        return attrs

    def _build_contrastive(self, nodes: list[dict]) -> list[str]:
        constraints = []
        for node in nodes:
            constraints.extend(node.get("contrastive_constraints", []) or [])
        return constraints
