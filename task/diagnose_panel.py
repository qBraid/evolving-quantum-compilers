"""Panel health check. Run this before trusting a benchmark instance.

    python diagnose_panel.py [--samples 300]

An instance is only worth evolving against if better layouts actually exist and
are findable. This script measures, per instance:

    trivial      cost of the identity layout (the score denominator)
    sabre        cost of Qiskit's own SabreLayout (the bar worth beating)
    rand_best    best layout found by random sampling
    rand_median  median random layout

and then flags two failure modes:

    NARROW    random sampling cannot beat the trivial layout. Either the
              instance is already solved or its landscape is a needle in a
              haystack. Either way there is no gradient to climb.

    ARTEFACT  the trivial layout is drastically better than a typical random
              one AND better than the best sampled layout. That usually means
              the score is tracking a quirk of the routing pass rather than the
              quality of the embedding -- exactly what you do not want a search
              to learn.

Neither flag is automatically fatal, but an instance that trips one should be
justified in writing or replaced.
"""

from __future__ import annotations

import argparse
import random
import statistics

from benchmarks import PANEL
from evaluate import routed_two_qubit_count


def diagnose(samples: int, seed: int) -> int:
    rng = random.Random(seed)
    flagged = 0

    header = (
        f"{'instance':<26} {'trivial':>8} {'sabre':>8} "
        f"{'rand_best':>10} {'rand_med':>9}  notes"
    )
    print(header)
    print("-" * len(header))

    for inst in PANEL:
        circuit = inst.circuit()
        coupling = inst.coupling_map()
        num_logical = circuit.num_qubits
        num_physical = coupling.size()

        trivial = routed_two_qubit_count(circuit, coupling, list(range(num_logical)))
        sabre = routed_two_qubit_count(circuit, coupling, None)

        counts = []
        for _ in range(samples):
            layout = list(range(num_physical))
            rng.shuffle(layout)
            counts.append(routed_two_qubit_count(circuit, coupling, layout[:num_logical]))
        rand_best = min(counts)
        rand_med = statistics.median(counts)

        notes = []
        if rand_best >= trivial:
            notes.append("NARROW")
        if rand_med > 1.3 * trivial and rand_best >= trivial:
            notes.append("ARTEFACT")
        if notes:
            flagged += 1

        print(
            f"{inst.instance_id:<26} {trivial:>8} {sabre:>8} "
            f"{rand_best:>10} {rand_med:>9.0f}  {' '.join(notes)}"
        )

    print()
    if flagged:
        print(f"{flagged} instance(s) flagged. Read the docstring before shipping them.")
    else:
        print("All instances have searchable headroom.")
    return flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    raise SystemExit(0 if diagnose(args.samples, args.seed) == 0 else 1)
