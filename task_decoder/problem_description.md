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
individual shot, then matching) reaches about **1.5x** on this panel — measured
1.52x on `surface-d5-p005` and 1.48x on `surface-d5-p008`. It is also far too
slow to use as a submission: it takes minutes per instance, well past the time
limit. Matching its accuracy cheaply is the problem.

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
- Deterministic: the evaluator decodes twice, in two separate processes, and
  rejects the candidate if the answers differ. Memoising the first answer does
  not help — the second process starts fresh.
- No filesystem, no network, no importing the evaluator or the benchmarks. You
  run in a subprocess that cannot import them, and a score implausibly far above
  plain MWPM (>3x) is rejected rather than accepted.
- The whole panel must decode within 240 seconds. Plain MWPM does all three
  instances in about 0.2s, so there is a lot of room — but a per-shot Python
  loop over 20000 shots will still blow through it. Vectorise, or reserve the
  expensive path for the minority of shots that need it.
