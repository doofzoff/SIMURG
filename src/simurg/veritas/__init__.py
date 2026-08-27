# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Veritas — faithful-generation stack
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# Veritas turns SIMURG from a decoding-corruption guard into a full
# faithful-generation stack. Five layers, all online and white-box (they read the
# self-hosted decoder's own logprobs — a signal black-box guards cannot have):
#
#   L0  SIMURG          decoding-corruption sentinel (loops / drift / garbage)
#   L1  context_reliance  answer grounded in the provided evidence (CAD-style)
#   L2  fact_entropy    token-time epistemic uncertainty on fact-bearing tokens
#   L3  semantic_entropy  Nature-2024 self-consistency, fired surgically
#   L4  verify          targeted Chain-of-Verification on flagged claims
#   L5  abstain         conformal-calibrated gate: confident | hedge | abstain
#
# The guarantee is not "never wrong" (no method is) — it is: where the model
# cannot be trusted, Veritas ABSTAINS instead of asserting.
# ═══════════════════════════════════════════════════════════════════════════════
from .abstain import AbstentionGate
from .context_reliance import context_grounding
from .fact_entropy import (FactUncertaintyDetector, TokenSignal, is_fact_token,
                           token_entropy, token_margin)
from .guard import VeritasGuard
from .semantic_entropy import semantic_entropy
from .verify import verify_claim

__all__ = [
    "VeritasGuard", "FactUncertaintyDetector", "TokenSignal", "AbstentionGate",
    "semantic_entropy", "verify_claim", "context_grounding",
    "token_entropy", "token_margin", "is_fact_token",
]
