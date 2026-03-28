"""
src/srd/refiner.py
───────────────────
Self-Correcting RAG-Guided Diffusion (SRD) module — Section 3.3.

Exact implementation of Algorithm 1 and Algorithm 2 from the paper.

Algorithm 1 — RAVEL SRD loop:
    Input: P_user, G, K
    1. K      ← RetrieveKnowledge(P_user, G)        [done by retriever]
    2. P_0    ← ConstructEnhancedPrompt(P_user, K)   [done by prompt_synth]
    3. I_0    ← GenerateImage(P_0)                   [done by backbone]
    4. S_tracker ← {0}^|K|; GSI ← 0
    5. for k = 1 to K:
    6.   F_curr ← AnalyzeImage(I_{k-1}, K)           [verifier]
    7.   GSI   ← ComputeGSI(F_curr, K)               [Eq. 5]
    8.   if GSI ≥ GSI_ε: break
    9.   d     ← ComputeDecay(k)                      [d = min(d0, 1/k)]
    10.  for all a ∈ K:
    11.    S_tracker[a] ← UpdateStability(F_curr[a])
    12.    if S_tracker[a] < N_lock:
    13.      P_k ← RefinePromptWeights(P_{k-1}, a, d)
    14.  I_k ← GenerateImage(P_k)
    15.  if DetectPlateau(GSI): ApplyEscapeStrategy(P_k)
    16. return I_k, P_k

Algorithm 2 — ApplyEscapeStrategy:
    1. attr ← GetStagnantAttributes(GSI_scores)
    2. if is_syntactic_failure(attr):
    3.   P_k ← RestructurePromptSyntax(P_k)
    4. else:
    5.   K_fine ← RetrieveFineGrainedKnowledge(G, attr)
    6.   P_k ← InjectSecondaryAttributes(P_k, K_fine)

Exact hyperparameters from paper:
    τ   = 0.85   (GSI convergence threshold)
    K   = 3      (max iterations)
    d_0 = 0.9    (initial decay)
    N_lock = 2   (rounds missing before escape triggers)
    Decay: d_k = min(d_0, 1.0/k)
        Round 1: min(0.9, 1.0) = 0.9
        Round 2: min(0.9, 0.5) = 0.5
        Round 3: min(0.9, 0.33) = 0.33
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from src.kg.retriever import ContextPacket
from src.generation.backbone import BaseBackbone
from src.generation.prompt_synth import PromptSynthesizer
from src.srd.verifier import AttributeVerifier, VerificationResult

logger = logging.getLogger(__name__)


# ── Per-round record ──────────────────────────────────────────────────────────

@dataclass
class SRDRound:
    """Record of a single SRD iteration."""
    round_idx:        int
    prompt:           str
    image:            Image.Image
    verification:     VerificationResult
    gsi:              float                  # GSI_t
    decay:            float                  # d_k
    converged:        bool                   # GSI_t >= τ
    escape_triggered: bool = False


# ── Full SRD result ───────────────────────────────────────────────────────────

@dataclass
class SRDResult:
    """
    Complete output of one SRD run.

    rounds             : all SRDRound records (one per iteration)
    final_image        : last generated image
    final_prompt       : prompt used for final image
    final_gsi          : GSI at termination
    converged          : True if GSI ≥ τ before hitting K
    convergence_round  : which round achieved convergence (None if not)

    The rounds list enables convergence statistics:
        % converging at R1, R2, hitting K-cap
    """
    rounds:            list[SRDRound]
    final_image:       Image.Image
    final_prompt:      str
    final_gsi:         float
    converged:         bool
    convergence_round: Optional[int]

    def gsi_trajectory(self) -> list[float]:
        return [r.gsi for r in self.rounds]

    def summary(self) -> str:
        traj = [f"{g:.3f}" for g in self.gsi_trajectory()]
        lines = [
            f"SRD: {len(self.rounds)} round(s)",
            f"  GSI trajectory : {traj}",
            f"  Converged      : {self.converged}",
        ]
        if self.converged:
            lines.append(f"  Convergence at : Round {self.convergence_round}")
        return "\n".join(lines)


# ── SRD Refiner ───────────────────────────────────────────────────────────────

class SRDRefiner:
    """
    Implements Algorithm 1 and Algorithm 2 exactly as described in the paper.

    Args:
        backbone    : T2I backbone (SDXL / Flux / DALL-E 3 / Infinity / Janus-Pro)
        synthesizer : PromptSynthesizer for prompt refinement
        verifier    : AttributeVerifier (GPT-4o vision)
        tau         : GSI convergence threshold (paper: 0.85)
        max_k       : max iterations (paper: 3)
        d0          : initial decay (paper: 0.9)
        n_lock      : rounds missing before escape triggers (paper: ~2)
        plateau_window : rounds with no improvement to declare plateau
        min_gsi_delta  : minimum GSI improvement to not be a plateau
    """

    def __init__(
        self,
        backbone:       BaseBackbone,
        synthesizer:    PromptSynthesizer,
        verifier:       AttributeVerifier,
        tau:            float = 0.85,
        max_k:          int   = 3,
        d0:             float = 0.9,
        n_lock:         int   = 2,
        plateau_window: int   = 2,
        min_gsi_delta:  float = 0.05,
    ):
        self.backbone       = backbone
        self.synthesizer    = synthesizer
        self.verifier       = verifier
        self.tau            = tau
        self.max_k          = max_k
        self.d0             = d0
        self.n_lock         = n_lock
        self.plateau_window = plateau_window
        self.min_gsi_delta  = min_gsi_delta

    # ── Algorithm 1 ───────────────────────────────────────────────────────────

    def run(
        self,
        initial_prompt: str,
        initial_image:  Image.Image,
        ctx:            ContextPacket,
        seed:           Optional[int] = None,
        output_dir:     Optional[Path] = None,
    ) -> SRDResult:
        """
        Execute Algorithm 1 — the full SRD loop.

        Args:
            initial_prompt : enriched prompt P_0 (post KG synthesis)
            initial_image  : I_0 (generated before SRD starts)
            ctx            : ContextPacket — provides cached A_ret
                             (no re-query to graph during refinement)
            seed           : optional seed for reproducibility
            output_dir     : if set, saves intermediate images here

        Returns:
            SRDResult with all per-round data
        """
        # A_ret — attribute checklist (cached, no re-query)
        attributes = ctx.retrieved_attributes

        # Line 4: S_tracker ← {0}^|K|
        # Counts how many consecutive rounds each attribute has been missing
        s_tracker: dict[str, int] = {a: 0 for a in attributes}

        rounds:         list[SRDRound] = []
        current_prompt: str            = initial_prompt
        current_image:  Image.Image    = initial_image

        logger.info(
            f"SRD starting: {len(attributes)} attributes, "
            f"τ={self.tau}, K={self.max_k}, d0={self.d0}"
        )

        # Lines 5-15: main loop
        for k in range(1, self.max_k + 1):

            # Line 6: F_curr ← AnalyzeImage(I_{k-1}, K)
            verification = self.verifier.verify(current_image, attributes)

            # Line 7: GSI ← ComputeGSI(F_curr, K)   [Eq. 5]
            gsi = verification.gsi

            # Line 8: if GSI ≥ GSI_ε: break
            converged = gsi >= self.tau

            # Line 9: d ← ComputeDecay(k)
            decay = self._compute_decay(k)

            # Record this round
            srd_round = SRDRound(
                round_idx=k,
                prompt=current_prompt,
                image=current_image,
                verification=verification,
                gsi=gsi,
                decay=decay,
                converged=converged,
            )
            rounds.append(srd_round)

            logger.info(f"  Round {k}: GSI={gsi:.3f} | converged={converged} | decay={decay:.2f}")

            if output_dir:
                self._save_intermediate(current_image, k, gsi, output_dir)

            if converged:
                logger.info(f"  SRD converged at round {k} ✓")
                break

            if k == self.max_k:
                logger.info(f"  SRD hit max iterations K={self.max_k}")
                break

            # Lines 10-13: update S_tracker and refine prompt
            # Line 11: S_tracker[a] ← UpdateStability(F_curr[a])
            for attr in attributes:
                if attr in verification.missing:
                    s_tracker[attr] = s_tracker.get(attr, 0) + 1
                else:
                    # Reset counter if attribute is now present
                    s_tracker[attr] = 0

            # Line 12-13: only refine for attrs below N_lock threshold
            # (attrs missing for N_lock+ rounds go to escape strategy instead)
            attrs_to_refine = [
                a for a in verification.missing
                if s_tracker.get(a, 0) < self.n_lock
            ]

            # Line 15: DetectPlateau → ApplyEscapeStrategy
            if self._detect_plateau(rounds):
                logger.info(f"  Plateau detected at round {k} — applying escape strategy")
                current_prompt = self._escape_strategy(
                    current_prompt, s_tracker, ctx
                )
                srd_round.escape_triggered = True
            else:
                # Standard refinement on non-locked missing attributes
                if attrs_to_refine:
                    current_prompt = self.synthesizer.refine(
                        current_prompt=current_prompt,
                        missing_attributes=attrs_to_refine,
                        decay=decay,
                        round_idx=k,
                    )

            # Line 14: I_k ← GenerateImage(P_k)
            current_image = self.backbone.generate(current_prompt, seed=seed)

        # Build result
        convergence_round = next(
            (r.round_idx for r in rounds if r.converged), None
        )

        result = SRDResult(
            rounds=rounds,
            final_image=current_image,
            final_prompt=current_prompt,
            final_gsi=rounds[-1].gsi if rounds else 0.0,
            converged=convergence_round is not None,
            convergence_round=convergence_round,
        )

        logger.info(result.summary())
        return result

    # ── Algorithm 2 — ApplyEscapeStrategy ────────────────────────────────────

    def _escape_strategy(
        self,
        current_prompt: str,
        s_tracker:      dict[str, int],
        ctx:            ContextPacket,
    ) -> str:
        """
        Algorithm 2: APPLYESCAPESTRATEGY.

        Triggered when plateau detected. Two branches:
            - Syntactic failure: reorder prompt sentences
            - Semantic failure:  inject secondary fine-grained attributes
                                 from cached neighbour entities (no re-query)
        """
        # Line 1: GetStagnantAttributes — missing for ≥ N_lock rounds
        stagnant = [
            a for a, count in s_tracker.items()
            if count >= self.n_lock
        ]
        logger.debug(f"  Escape: stagnant attrs = {stagnant}")

        if not stagnant:
            # Nothing clearly stagnant — restructure syntax
            return self._restructure_syntax(current_prompt)

        # Line 2: is_syntactic_failure
        # Heuristic: if all stagnant attrs are short phrases (≤3 words)
        # it's likely a syntactic issue — the model sees the words but
        # doesn't render them because of prompt ordering
        is_syntactic = all(len(a.split()) <= 3 for a in stagnant)

        if is_syntactic:
            # Line 3: RestructurePromptSyntax
            return self._restructure_syntax(current_prompt)
        else:
            # Lines 5-6: RetrieveFineGrainedKnowledge from cached context
            # (no graph re-query — uses ctx.neighbour_entities)
            secondary = self._get_secondary_attributes(stagnant, ctx)
            return self._inject_secondary(current_prompt, secondary)

    def _restructure_syntax(self, prompt: str) -> str:
        """
        RestructurePromptSyntax — reorder sentences to change model attention.
        Moves the last sentence to the front, changing emphasis ordering.
        """
        sentences = [s.strip() for s in prompt.split(".") if s.strip()]
        if len(sentences) > 2:
            # Rotate: last sentence becomes first
            sentences = [sentences[-1]] + sentences[:-1]
        restructured = ". ".join(sentences)
        logger.debug("  Escape: restructured prompt syntax")
        return restructured

    def _get_secondary_attributes(
        self,
        stagnant: list[str],
        ctx: ContextPacket,
    ) -> list[str]:
        """
        RetrieveFineGrainedKnowledge — pull secondary attributes from
        cached neighbour entities in the ContextPacket.
        No additional graph queries — uses C_P already retrieved.
        """
        secondary: list[str] = []
        stagnant_set = set(stagnant)

        for neighbour in ctx.neighbour_entities:
            features = neighbour.get("distinctive_features", []) or []
            for attr in features:
                if attr not in stagnant_set and attr not in secondary:
                    secondary.append(attr)
            if len(secondary) >= 5:
                break

        return secondary

    def _inject_secondary(self, prompt: str, secondary: list[str]) -> str:
        """
        InjectSecondaryAttributes — append fine-grained secondary
        attributes to the prompt with explicit emphasis.
        """
        if not secondary:
            return prompt
        additions = "; ".join(secondary[:5])
        injected = f"{prompt} Ensure precise rendering of: {additions}."
        logger.debug(f"  Escape: injected {len(secondary)} secondary attrs")
        return injected

    # ── Decay function ─────────────────────────────────────────────────────────

    def _compute_decay(self, k: int) -> float:
        """
        ComputeDecay(k) — exact formula from paper (Table 12 trace).
            d_k = min(d_0, 1.0 / k)
        Round 1: min(0.9, 1.0) = 0.9
        Round 2: min(0.9, 0.5) = 0.5
        Round 3: min(0.9, 0.33) = 0.33
        """
        return min(self.d0, 1.0 / k)

    # ── Plateau detection ──────────────────────────────────────────────────────

    def _detect_plateau(self, rounds: list[SRDRound]) -> bool:
        """
        DetectPlateau — returns True if GSI has not improved by
        min_gsi_delta over the last plateau_window rounds.
        """
        if len(rounds) < self.plateau_window:
            return False
        recent = [r.gsi for r in rounds[-self.plateau_window:]]
        return (max(recent) - min(recent)) < self.min_gsi_delta

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_intermediate(
        image:     Image.Image,
        round_idx: int,
        gsi:       float,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"srd_round_{round_idx}_gsi{gsi:.2f}.png"
        image.save(path)
        logger.debug(f"  Saved: {path}")
