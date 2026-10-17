"""Self-consistency selection: test-time compute that needs no verifier.

Sample k answers, then return the ONE sample whose stated values agree with the most other
samples (Wang et al., 2023, adapted to free-form numerical answers: "agree" = some value in
one answer's scored span is within `tol` of some value in the other's). The selected sample is
scored exactly like a single sample, so vote@k is directly comparable with acc@1 -- unlike
pass@k, which picks the right sample with knowledge of the target and is only an upper bound.

Measured on the 153 held-out numerical items (batched decoder, 8 seeds x 4 samples pooled into
disjoint 8-subsets; paired bootstrap 95% CI; python -m src.analysis.tier_vote):

    model        acc@1   vote@8   vote@8 - acc@1          answer in prompt (19)   must-compute (134)
    SFT          4.9%    5.7%     +0.9 pp [-0.5, +2.3]    18.6% -> 25.0%          2.9% -> 3.0%
    GRPO v4      6.4%    9.2%     +2.7 pp [+0.9, +4.7]    29.3% -> 43.4%          3.2% -> 4.3%
    GRPO v5      6.4%    8.5%     +2.1 pp [+0.4, +4.0]    28.1% -> 47.4%          3.3% -> 3.0%

Positive for every model, but the gain is on the 19 items whose answer is already in the
prompt (the given number is the mode). On must-compute items it is unstable across seed sets
(+0.1 pp for SFT on seeds 21-24+31-34, +2.7 pp on seeds 31-36) around a ~3% base:
agreement cannot select a right answer the model almost never produces. Dropping values copied
from the prompt before voting made it worse, not better: correct answers often restate a given
quantity. Report vote@k by tier, never only in aggregate.
"""

from __future__ import annotations

from typing import List, Sequence


def _close(a: float, b: float, tol: float) -> bool:
    d = abs(b) if abs(b) > 1e-7 else 1.0
    return abs(a - b) / d <= tol


def agree(x: Sequence[float], y: Sequence[float], tol: float = 0.05) -> bool:
    return any(_close(a, b, tol) or _close(b, a, tol) for a in x for b in y)


def select_consensus(tails: List[Sequence[float]], tol: float = 0.05) -> int:
    """Index of the sample agreeing with the most others; ties go to the earliest sample
    (no access to the target)."""
    def support(i: int) -> int:
        return sum(agree(tails[i], tails[j], tol) for j in range(len(tails)) if j != i)
    return max(range(len(tails)), key=lambda i: (support(i), -i))
