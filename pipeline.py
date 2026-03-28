"""
pipeline.py
────────────
Top-level RAVEL pipeline.

For each prompt, creates a structured output folder:

    output/yama_dalle3/
        00_base.png              ← vanilla backbone, raw prompt, no KG
        01_ravel.png             ← RAVEL enriched prompt, no SRD (I_0)
        02_srd_r1_gsi0.55.png   ← after SRD round 1
        03_srd_r2_gsi1.00.png   ← after SRD round 2
        final.png                ← best image (last SRD or ravel if SRD off)
        run_info.json            ← full metadata

Usage:
    from pipeline import RAVELPipeline

    with RAVELPipeline(backbone="dalle3", srd=True) as pipeline:
        result = pipeline.run("Hindu god Yama seated on a water buffalo")
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from src.kg.neo4j_client import Neo4jClient
from src.kg.retriever import KGRetriever, ContextPacket
from src.generation.prompt_synth import PromptSynthesizer
from src.generation.backbone import load_backbone, BaseBackbone
from src.srd.verifier import AttributeVerifier
from src.srd.refiner import SRDRefiner, SRDResult

logger = logging.getLogger(__name__)


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Full output of one RAVEL pipeline run.

    base_image    : vanilla backbone output — raw prompt, no KG enrichment
    ravel_image   : KG-enriched prompt, no SRD (I_0 in paper)
    final_image   : best image — last SRD round, or ravel_image if SRD off
    srd_result    : full SRD trace with per-round images and GSI
    run_dir       : folder containing all saved images
    """
    prompt:          str
    enriched_prompt: str
    backbone:        str
    base_image:      Image.Image
    ravel_image:     Image.Image
    final_image:     Image.Image
    ctx:             ContextPacket
    srd_result:      Optional[SRDResult] = None
    run_dir:         Optional[Path]      = None

    def summary(self) -> str:
        lines = [
            f"RAVEL Pipeline Result",
            f"  Prompt    : {self.prompt}",
            f"  Backbone  : {self.backbone}",
            f"  Domain    : {self.ctx.domain}",
            f"  Entities  : {[e.get('name') for e in self.ctx.primary_entities]}",
            f"  Attrs     : {len(self.ctx.retrieved_attributes)}",
            f"  SRD       : {self.srd_result is not None}",
        ]
        if self.srd_result:
            lines.append(f"  {self.srd_result.summary()}")
        if self.run_dir:
            lines.append(f"  Saved to  : {self.run_dir}/")
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAVELPipeline:
    """
    Main RAVEL generation pipeline.

    Args:
        backbone_name   : "sdxl" | "flux" | "dalle3" | "janus_pro" | "glm_image"
        srd             : enable iterative self-correction
        tau             : GSI convergence threshold (paper: 0.85)
        max_k           : max SRD iterations (paper: 3)
        d0              : initial decay (paper: 0.9)
        k_hops          : KG retrieval hops (paper: 1)
        output_dir      : root directory for per-entity output folders
        backbone_kwargs : passed to backbone constructor
    """

    def __init__(
        self,
        backbone_name:   str   = "dalle3",
        srd:             bool  = True,
        tau:             float = 0.85,
        max_k:           int   = 3,
        d0:              float = 0.9,
        k_hops:          int   = 1,
        output_dir:      Optional[str] = "output/",
        **backbone_kwargs,
    ):
        self.backbone_name = backbone_name
        self.srd_enabled   = srd
        self.tau           = tau
        self.max_k         = max_k
        self.d0            = d0
        self.output_dir    = Path(output_dir) if output_dir else None

        logger.info("Initialising RAVEL pipeline...")

        self.neo4j_client = Neo4jClient()
        self.retriever    = KGRetriever(client=self.neo4j_client, k=k_hops)
        self.synthesizer  = PromptSynthesizer()
        self.backbone: BaseBackbone = load_backbone(backbone_name, **backbone_kwargs)

        if srd:
            self.verifier   = AttributeVerifier()
            self.srd_module = SRDRefiner(
                backbone=self.backbone,
                synthesizer=self.synthesizer,
                verifier=self.verifier,
                tau=tau,
                max_k=max_k,
                d0=d0,
            )
        else:
            self.verifier   = None
            self.srd_module = None

        logger.info(
            f"Pipeline ready — backbone={backbone_name}, "
            f"srd={srd}, tau={tau}, max_k={max_k}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        prompt:    str,
        seed:      Optional[int] = None,
        save_name: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full RAVEL pipeline for a single prompt.

        Generates and saves:
            00_base.png          — raw prompt, no KG, no SRD
            01_ravel.png         — KG-enriched prompt, no SRD (I_0)
            02_srd_r1_gsiX.png   — after SRD round 1  (if SRD enabled)
            03_srd_r2_gsiX.png   — after SRD round 2  (if SRD enabled)
            ...
            final.png            — best image
            run_info.json        — full metadata

        Args:
            prompt    : free-text user prompt
            seed      : optional seed for reproducibility
            save_name : folder name override

        Returns:
            PipelineResult
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RAVEL: '{prompt}'")
        logger.info(f"{'='*60}")

        # ── Step 1: KG retrieval ───────────────────────────────────────────────
        ctx = self.retriever.retrieve(prompt)
        if ctx.is_empty():
            logger.warning("No KG context found — using original prompt.")

        # ── Step 2: Enriched prompt ────────────────────────────────────────────
        enriched = self.synthesizer.synthesize(ctx)
        p0 = enriched.enriched

        # ── Step 3: Create per-entity output folder ────────────────────────────
        run_dir = self._make_run_dir(prompt, ctx, save_name)

        # ── Step 4: Generate base image (raw prompt, no KG) ───────────────────
        logger.info(f"Generating 00_base (raw prompt, no enrichment)...")
        base_image = self.backbone.generate(prompt, seed=seed)
        if run_dir:
            base_image.save(run_dir / "00_base.png")
            logger.info("  Saved: 00_base.png")

        # ── Step 5: Generate RAVEL image (enriched prompt, no SRD) ───────────
        logger.info(f"Generating 01_ravel (KG-enriched prompt, no SRD)...")
        ravel_image = self.backbone.generate(p0, seed=seed)
        if run_dir:
            ravel_image.save(run_dir / "01_ravel.png")
            logger.info("  Saved: 01_ravel.png")

        # ── Step 6: SRD ───────────────────────────────────────────────────────
        srd_result:  Optional[SRDResult] = None
        final_image  = ravel_image
        final_prompt = p0

        if self.srd_enabled and self.srd_module and ctx.retrieved_attributes:
            logger.info("Running SRD...")

            srd_result = self.srd_module.run(
                initial_prompt=p0,
                initial_image=ravel_image,
                ctx=ctx,
                seed=seed,
                output_dir=None,
            )
            final_image  = srd_result.final_image
            final_prompt = srd_result.final_prompt

            # Save each SRD round — numbered continuing from 02
            if run_dir:
                for srd_round in srd_result.rounds:
                    idx   = srd_round.round_idx + 1   # 02, 03, ...
                    fname = (
                        f"{idx:02d}_srd_r{srd_round.round_idx}"
                        f"_gsi{srd_round.gsi:.2f}.png"
                    )
                    srd_round.image.save(run_dir / fname)
                    logger.info(f"  Saved: {fname}")

        elif self.srd_enabled and not ctx.retrieved_attributes:
            logger.warning("SRD skipped — no attributes retrieved.")

        # ── Step 7: Save final + metadata ─────────────────────────────────────
        if run_dir:
            final_image.save(run_dir / "final.png")
            logger.info("  Saved: final.png")
            self._save_run_info(
                run_dir=run_dir,
                prompt=prompt,
                enriched_prompt=p0,
                final_prompt=final_prompt,
                ctx=ctx,
                srd_result=srd_result,
                seed=seed,
            )

        result = PipelineResult(
            prompt=prompt,
            enriched_prompt=p0,
            backbone=self.backbone_name,
            base_image=base_image,
            ravel_image=ravel_image,
            final_image=final_image,
            ctx=ctx,
            srd_result=srd_result,
            run_dir=run_dir,
        )

        logger.info(result.summary())
        return result

    def run_batch(
        self,
        prompts: list[str],
        seed:    Optional[int] = None,
    ) -> list[PipelineResult]:
        """Run the pipeline for a list of prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Batch [{i+1}/{len(prompts)}]")
            results.append(self.run(prompt, seed=seed))
        return results

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.neo4j_client.close()

    def __enter__(self) -> "RAVELPipeline":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_run_dir(
        self,
        prompt:    str,
        ctx:       ContextPacket,
        save_name: Optional[str],
    ) -> Optional[Path]:
        """Create and return per-entity output folder."""
        if not self.output_dir:
            return None

        if save_name:
            stem = save_name
        elif ctx.primary_entities:
            stem = ctx.primary_entities[0].get("name", "entity")
            stem = stem.lower().replace(" ", "_")
        else:
            stem = prompt[:30].lower()
            stem = "".join(c if c.isalnum() else "_" for c in stem).strip("_")

        run_dir = self.output_dir / f"{stem}_{self.backbone_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder: {run_dir}/")
        return run_dir

    def _save_run_info(
        self,
        run_dir:         Path,
        prompt:          str,
        enriched_prompt: str,
        final_prompt:    str,
        ctx:             ContextPacket,
        srd_result:      Optional[SRDResult],
        seed:            Optional[int],
    ) -> None:
        """Save full run metadata as JSON."""
        info = {
            "prompt":           prompt,
            "enriched_prompt":  enriched_prompt,
            "final_prompt":     final_prompt,
            "backbone":         self.backbone_name,
            "seed":             seed,
            "domain":           ctx.domain,
            "entities":         [e.get("name") for e in ctx.primary_entities],
            "attributes":       ctx.retrieved_attributes,
            "contrastive":      ctx.contrastive_constraints,
            "srd_enabled":      srd_result is not None,
            "images": {
                "00_base":  "raw prompt, no KG enrichment",
                "01_ravel": "KG-enriched prompt, no SRD",
                "final":    "best image after SRD (or ravel if SRD disabled)",
            },
        }

        if srd_result:
            info["srd"] = {
                "converged":         srd_result.converged,
                "convergence_round": srd_result.convergence_round,
                "final_gsi":         round(srd_result.final_gsi, 4),
                "gsi_trajectory":    [round(g, 4) for g in srd_result.gsi_trajectory()],
                "rounds": [
                    {
                        "round":          r.round_idx,
                        "image":          f"{r.round_idx+1:02d}_srd_r{r.round_idx}_gsi{r.gsi:.2f}.png",
                        "gsi":            round(r.gsi, 4),
                        "decay":          round(r.decay, 4),
                        "converged":      r.converged,
                        "escape":         r.escape_triggered,
                        "present_attrs":  r.verification.present,
                        "missing_attrs":  r.verification.missing,
                    }
                    for r in srd_result.rounds
                ],
            }

        with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info("  Saved: run_info.json")
