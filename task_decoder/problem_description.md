# Decoding surface-code syndromes better than plain MWPM

You are improving a decoder for a rotated surface-code memory experiment under
circuit-level depolarising noise.

Physical qubits fail; the code detects those failures as a **syndrome** (a set of
detection events). A decoder turns that syndrome into a guess about whether the
*logical* observable was flipped. Guess well and the logical error rate drops,
which is the entire point of error correction.

You are editing one function:

    decode_batch(detectors, ctx) -> (shots, num_observables) array of 0/1

## How you are scored

Three instances — surface code d=5 at p=0.005 and p=0.008, and d=7 at p=0.005 —
20000 shots each, identical shots for every candidate. For each, the evaluator
measures your logical error rate and divides plain MWPM's by yours, so **1.0
means you tied plain minimum-weight perfect matching** and higher is better. The
score is the mean.

## Where the headroom is, and where it is not

Minimum-weight perfect matching on the decomposed detector error model is
already strong: the edge weights are the correct log-likelihood ratios *under
the assumption that errors are independent*. That assumption is what fails.

Circuit-level noise produces correlated errors — a Y error flips both an X and a
Z detector, hook errors from the syndrome circuit correlate neighbouring
detectors — and plain MWPM decodes as if they were independent. Decoders that
exploit those correlations do measurably better. Belief-matching (belief
propagation over the error model, used to reweight the matching graph for each
individual shot, then matching) reaches **1.55x** on this panel (1.52x, 1.48x and 1.65x on the three instances).

The practical consequence: **a purely static reweighting cannot help.** Scaling
or reshaping `ctx.default_weights` once, the same way for every shot, only
changes the log-likelihoods by a constant and leaves the minimal matching where
it was. The gain has to come from using *this* syndrome to decide which edges
are more likely than the static model says.

## What you have

- `ctx.matching` — a `pymatching.Matching` built from the DEM
- `ctx.new_matching(weights)` — a fresh Matching, same topology, your weights
- `ctx.edges` — `(u, v, probability, fault_ids)` per error mechanism
- `ctx.default_weights` — the DEM's own weights, in `ctx.edges` order
- `ctx.dem` — the `stim.DetectorErrorModel` itself

You never see the observables. You cannot compute your own score.

## Hard requirements

- Right shape, right dtype, values in {0, 1}.
- Deterministic: the evaluator decodes twice on identical inputs and rejects the
  candidate if the answers differ.
- No filesystem, no network, no importing the evaluator or the benchmarks.
- The whole panel must decode within 600 seconds. For scale, full
  belief-matching needs about 320s of that, so a correlated decoder is
  affordable but a careless one is not. A per-shot Python loop over
  20000 shots is usually too slow unless it does very little — vectorise, or
  reserve the expensive path for the minority of shots that need it.
