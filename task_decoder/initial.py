"""Seed program: decode surface-code syndromes.

You are optimising ONE function, ``decode_batch``. Given a batch of detection
events from a rotated surface-code memory experiment, predict whether the
logical observable was flipped.

The evaluator owns the circuits, the sampling and the truth. You never see the
observables -- only the syndromes -- so the only way to score well is to decode
well.

CONTRACT
    decode_batch(detectors, ctx) -> np.ndarray

    detectors  (shots, num_detectors) array of bool/uint8 detection events
    ctx        DecodeContext, described below

    Returns a (shots, num_observables) array of 0/1 predictions.

CONTEXT
    ctx.dem              the stim DetectorErrorModel (errors already decomposed)
    ctx.matching         a pymatching.Matching built from the DEM, ready to use
    ctx.new_matching(w)  a fresh Matching with edge weights replaced by w
    ctx.edges            list of (detector_indices, probability, observable_mask)
                         for every error mechanism in the DEM
    ctx.num_detectors, ctx.num_observables

SCORING
    For each panel instance the evaluator measures your logical error rate and
    divides the plain-MWPM logical error rate by yours. 1.0 means you tied plain
    MWPM; higher is better. The score is the mean over instances.

    Plain MWPM is the baseline. Belief-matching (belief propagation used to
    reweight the matching graph per shot) reaches about 1.55x on this panel, so
    that much headroom is known to exist. Reaching it requires using
    the syndrome, not just the static DEM weights: the DEM weights are already
    the correct independent-error log-likelihoods, so any purely static
    reweighting will not help.

HARD RULES (violating any of these scores the candidate as incorrect)
    - Return the right shape and dtype, values in {0, 1}.
    - Be deterministic. The evaluator calls you twice on identical inputs and
      rejects the candidate if the answers differ. Seed any randomness.
    - Do not import the evaluator or the benchmark module, touch the filesystem
      or the network, or attempt to recover the observables.
    - Stay inside the time limit. The whole panel must decode in a few minutes (the limit is 600s, and full belief-matching uses ~320s of it);
      a per-shot Python loop over 20000 shots is usually too slow unless it is
      doing very little.
"""

from __future__ import annotations

import numpy as np


# EVOLVE-BLOCK-START
def decode_batch(detectors: np.ndarray, ctx) -> np.ndarray:
    """Predict observable flips from detection events.

    Baseline strategy: plain minimum-weight perfect matching, then a crude
    "low confidence" second pass -- shots with an unusually heavy syndrome get
    re-decoded against a uniformly softened graph, on the theory that heavy
    syndromes are where the static weights are least trustworthy.

    The theory is wrong in this form. The DEM weights are already the correct
    independent-error log-likelihoods, so scaling them all by one constant
    changes nothing about which matching is minimal -- it only wastes time, and
    the syndrome-weight threshold ends up mislabelling ordinary shots. This
    seed therefore scores slightly BELOW plain MWPM.

    What actually pays is using each individual syndrome to decide which edges
    are more likely than the static model says.
    """
    predictions = ctx.matching.decode_batch(detectors)

    syndrome_weight = detectors.sum(axis=1)
    threshold = np.mean(syndrome_weight) + 2.0 * np.std(syndrome_weight)
    heavy = np.flatnonzero(syndrome_weight > threshold)

    if heavy.size:
        softened = ctx.new_matching(
            [0.8 * w for w in ctx.default_weights]
        )
        predictions[heavy] = softened.decode_batch(detectors[heavy])

    return predictions.astype(np.uint8)
# EVOLVE-BLOCK-END
