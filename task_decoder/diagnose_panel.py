"""Check the decoder panel is worth searching, before spending a run on it.

Two questions, both of which have sunk evolutionary searches before:

  1. Is there headroom? If no known-better decoder beats plain MWPM on an
     instance, the search has no gradient there and will burn generations.
  2. Is the headroom reachable inside the evaluator's time limit? Belief-matching
     is the reference: it is the standard correlated decoder and the thing the
     search is being asked to rediscover or beat.

    python task_decoder/diagnose_panel.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import benchmarks  # noqa: E402
from evaluate import TIME_LIMIT_SECONDS, load_baselines, logical_error_rate  # noqa: E402


def main() -> int:
    from beliefmatching import BeliefMatching

    baselines = load_baselines()
    panel = benchmarks.load_panel()

    print(f"{'instance':<18} {'MWPM LER':>10} {'BM LER':>10} {'headroom':>9} {'BM time':>9}")
    print("-" * 60)

    ratios, total = [], 0.0
    failures = []
    for instance in panel:
        mwpm = baselines[instance.name]["mwpm_ler"]
        bm = BeliefMatching.from_detector_error_model(instance.dem, max_bp_iters=20)
        t0 = time.time()
        predictions = bm.decode_batch(instance.detectors)
        elapsed = time.time() - t0
        total += elapsed
        ler = logical_error_rate(predictions, instance.observables)
        ratio = mwpm / max(ler, 1.0 / instance.shots)
        ratios.append(ratio)
        print(f"{instance.name:<18} {mwpm:>10.5f} {ler:>10.5f} {ratio:>8.3f}x {elapsed:>8.1f}s")
        if ratio <= 1.02:
            failures.append(f"{instance.name}: only {ratio:.3f}x headroom -- little to find here")

    print("-" * 60)
    print(f"mean headroom {np.mean(ratios):.3f}x over the panel, "
          f"belief-matching total {total:.0f}s of the {TIME_LIMIT_SECONDS:.0f}s limit")

    if total > TIME_LIMIT_SECONDS:
        failures.append(
            f"the reference decoder itself needs {total:.0f}s, over the "
            f"{TIME_LIMIT_SECONDS:.0f}s limit -- raise the limit or cut shots"
        )

    if failures:
        print("\nPROBLEMS:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPanel is searchable: every instance has headroom and the reference fits the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
