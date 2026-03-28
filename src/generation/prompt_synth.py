"""
src/generation/prompt_synth.py
───────────────────────────────
Contrastive Chain-of-Thought prompt synthesis (Section 3.2).

Takes a ContextPacket from the retriever and produces an enriched
prompt P' that:
    - Encodes all visual, functional, relational attributes
    - Uses explicit NOT-X contrastive constraints to prevent
      semantic collapse into generic category priors
    - Weaves in relational context (POLLINATED_BY, RIDES, etc.)

Also exposes a refine() method used by the SRD module to produce
P_{t+1} from P_t by emphasising missing attributes.

Paper reference: Section 3.2, Figure 12 (Red Ginger example)
"""

import os
import logging
from dataclasses import dataclass, field

from openai import OpenAI
from src.kg.retriever import ContextPacket

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class EnrichedPrompt:
    """
    Output of the prompt synthesizer.

    original  : user's raw prompt
    enriched  : final context-dense prompt for the T2I model
    contrastive_cues : NOT-X constraints included in the prompt
    """
    original:         str
    enriched:         str
    contrastive_cues: list[str] = field(default_factory=list)


# ── Synthesizer ───────────────────────────────────────────────────────────────

class PromptSynthesizer:
    """
    Converts a ContextPacket into an enriched contrastive prompt.

    The LLM is instructed to:
        1. Describe each entity's unique structural features in detail
        2. Explicitly contrast against generic category priors using NOT-X
        3. Weave in relational context (habitat, symbolic items, associations)
        4. Produce ONE coherent T2I prompt ~150-250 words
    """

    # ── System prompt (paper Figure 12 style) ────────────────────────────────
    _SYNTHESIS_SYSTEM = """\
You are an expert at generating precise, detailed text-to-image prompts
for rare, culturally nuanced, and visually distinctive concepts.

Your goal is to prevent "semantic collapse" — where a diffusion model
defaults to a generic visual prior instead of the specific rare concept.

Given structured knowledge graph attributes for one or more entities:

1. Write detailed visual descriptors for each entity's unique morphological
   features — shape, structure, color, texture, size, spatial arrangement.

2. Add explicit CONTRASTIVE constraints using "NOT [generic alternative]"
   phrasing to steer the model away from common priors.
   Example: "NOT edible ginger", "NOT a Western grim reaper",
            "NOT a generic deer", "NOT a plain wooden bowl"

3. Include relevant relational and contextual elements that add narrative
   grounding — habitat, symbolic items, associated entities, cultural origin.

4. Fuse everything into ONE coherent, vivid prompt suitable for a
   text-to-image diffusion model. Around 150-250 words.

Return ONLY the final prompt — no preamble, no explanation, no JSON."""

    # ── User message template ─────────────────────────────────────────────────
    _SYNTHESIS_USER = """\
ORIGINAL PROMPT:
"{original}"

ENTITY ATTRIBUTES:
{entity_block}

RELATIONAL CONTEXT:
{relational_block}

CONTRASTIVE CONSTRAINTS (must include these):
{constraints_block}

Write the enriched prompt now."""

    # ── Refinement system prompt (SRD module) ────────────────────────────────
    _REFINEMENT_SYSTEM = """\
You refine a text-to-image prompt to fix missing visual attributes.

Given the current prompt and a list of attributes NOT visible in the
last generated image, revise the prompt to strongly emphasise those
missing attributes WITHOUT removing correctly rendered ones.

Apply corrections proportionally to the decay weight:
    1.0 = strong emphasis, 0.5 = moderate, lower = subtle

Return ONLY the revised prompt — no preamble, no explanation."""

    _REFINEMENT_USER = """\
CURRENT PROMPT:
{current_prompt}

MISSING ATTRIBUTES (emphasise these strongly):
{missing_attrs}

DECAY WEIGHT: {decay:.2f}

Write the refined prompt now."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Public: synthesize ────────────────────────────────────────────────────

    def synthesize(self, ctx: ContextPacket) -> EnrichedPrompt:
        """
        Build enriched prompt from a ContextPacket.

        Args:
            ctx : ContextPacket from KGRetriever

        Returns:
            EnrichedPrompt with original + enriched prompts
        """
        if ctx.is_empty():
            logger.warning("Empty ContextPacket — returning original prompt unchanged.")
            return EnrichedPrompt(original=ctx.query, enriched=ctx.query)

        entity_block      = self._build_entity_block(ctx)
        relational_block  = self._build_relational_block(ctx)
        constraints_block = self._build_constraints_block(ctx)

        user_msg = self._SYNTHESIS_USER.format(
            original=ctx.query,
            entity_block=entity_block,
            relational_block=relational_block,
            constraints_block=constraints_block,
        )

        enriched = self._call_llm(
            system=self._SYNTHESIS_SYSTEM,
            user=user_msg,
            max_tokens=512,
        )

        logger.info(f"  Enriched prompt synthesised ({len(enriched)} chars)")
        logger.debug(f"  Enriched prompt:\n{enriched}")

        return EnrichedPrompt(
            original=ctx.query,
            enriched=enriched,
            contrastive_cues=ctx.contrastive_constraints,
        )

    # ── Public: refine (called by SRD) ───────────────────────────────────────

    def refine(
        self,
        current_prompt: str,
        missing_attributes: list[str],
        decay: float,
        round_idx: int,
    ) -> str:
        """
        Produce P_{t+1} by emphasising missing attributes.
        Called by the SRD refiner each iteration.

        Args:
            current_prompt     : prompt used in the last generation
            missing_attributes : attributes not found in last image
            decay              : weight controlling emphasis strength
            round_idx          : current SRD round (for logging)

        Returns:
            Refined prompt string.
        """
        if not missing_attributes:
            return current_prompt

        missing_str = "\n".join(f"- {a}" for a in missing_attributes)

        user_msg = self._REFINEMENT_USER.format(
            current_prompt=current_prompt,
            missing_attrs=missing_str,
            decay=decay,
        )

        refined = self._call_llm(
            system=self._REFINEMENT_SYSTEM,
            user=user_msg,
            max_tokens=512,
        )

        logger.info(
            f"  SRD Round {round_idx}: refined prompt "
            f"({len(missing_attributes)} missing attrs, decay={decay:.2f})"
        )
        return refined

    # ── Block builders ────────────────────────────────────────────────────────

    def _build_entity_block(self, ctx: ContextPacket) -> str:
        """Format entity attributes into a readable block for the LLM."""
        blocks = []

        for entity in ctx.primary_entities:
            name = entity.get("name", "")
            etype = entity.get("entity_type", "")
            domain = entity.get("domain", "")

            lines = [f"Entity: {name} ({etype}, {domain})"]

            morphology = entity.get("morphology", "")
            if morphology:
                lines.append(f"  Morphology       : {morphology}")

            features = entity.get("distinctive_features", []) or []
            if features:
                lines.append(f"  Distinctive      : {'; '.join(features)}")

            colors = entity.get("color_palette", []) or []
            if colors:
                lines.append(f"  Colors           : {', '.join(colors)}")

            texture = entity.get("texture", "")
            if texture:
                lines.append(f"  Texture          : {texture}")

            size = entity.get("size_and_scale", "")
            if size:
                lines.append(f"  Size             : {size}")

            structure = entity.get("structural_arrangement", "")
            if structure:
                lines.append(f"  Structure        : {structure}")

            primary_fn = entity.get("primary_function", "")
            if primary_fn:
                lines.append(f"  Function         : {primary_fn}")

            origin = entity.get("origin", "")
            if origin:
                lines.append(f"  Origin           : {origin}")

            period = entity.get("historical_period", "")
            if period:
                lines.append(f"  Period           : {period}")

            significance = entity.get("cultural_significance", "")
            if significance:
                lines.append(f"  Significance     : {significance}")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks) if blocks else "No entity attributes available."

    def _build_relational_block(self, ctx: ContextPacket) -> str:
        """Format relationships and neighbour context for the LLM."""
        lines = []

        # Typed edges
        for rel in ctx.relationships[:10]:
            lines.append(f"  ({rel['from']}) --[{rel['type']}]--> ({rel['to']})")

        # Neighbour entity names
        if ctx.neighbour_entities:
            nb_names = [e.get("name", "") for e in ctx.neighbour_entities[:5]]
            lines.append(f"  Related entities : {', '.join(nb_names)}")

        return "\n".join(lines) if lines else "No relational context available."

    def _build_constraints_block(self, ctx: ContextPacket) -> str:
        """Format contrastive constraints."""
        if not ctx.contrastive_constraints:
            return "None specified."
        return "\n".join(f"  {c}" for c in ctx.contrastive_constraints)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
