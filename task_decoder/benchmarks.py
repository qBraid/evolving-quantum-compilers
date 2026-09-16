"""The benchmark panel for the decoder task. Candidates never import this.

Each instance is a rotated surface-code memory experiment under circuit-level
depolarising noise, sampled with Stim. The evaluator owns the circuits, the
sampling and the scoring; a candidate only ever sees detection events.

Kept deliberately small and seeded so every candidate is scored on identical
shots -- a decoder that looks better only because it drew easier samples is a
measurement error, not a discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import stim

# (name, distance, rounds, physical error rate, shots)
#
# p is chosen per distance so the logical error rate lands in a band that is
# statistically resolvable at this shot count but not saturated. d=7 at p=0.005
# is the hardest and the one where a correlated decoder has the most to gain.
PANEL: Tuple[Tuple[str, int, int, float, int], ...] = (
    ("surface-d5-p005", 5, 5, 0.005, 20000),
    ("surface-d5-p008", 5, 5, 0.008, 20000),
    ("surface-d7-p005", 7, 7, 0.005, 20000),
)

SEED = 20260916


@dataclass(frozen=True)
class Instance:
    name: str
    distance: int
    rounds: int
    noise: float
    shots: int
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    detectors: np.ndarray      # (shots, num_detectors) bool
    observables: np.ndarray    # (shots, num_observables) bool


def build_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
        before_round_data_depolarization=noise,
    )


_CACHE: Dict[str, Instance] = {}


def load_panel() -> List[Instance]:
    """Build (once per process) and return the benchmark instances."""
    if _CACHE:
        return [_CACHE[name] for name, *_ in PANEL]

    for index, (name, distance, rounds, noise, shots) in enumerate(PANEL):
        circuit = build_circuit(distance, rounds, noise)
        # decompose_errors is what makes the DEM matchable: it splits
        # hyperedges (Y errors, hook errors) into graphlike pieces.
        dem = circuit.detector_error_model(decompose_errors=True)
        sampler = circuit.compile_detector_sampler(seed=SEED + index)
        detectors, observables = sampler.sample(shots, separate_observables=True)
        _CACHE[name] = Instance(
            name=name,
            distance=distance,
            rounds=rounds,
            noise=noise,
            shots=shots,
            circuit=circuit,
            dem=dem,
            detectors=np.ascontiguousarray(detectors),
            observables=np.ascontiguousarray(observables),
        )
    return [_CACHE[name] for name, *_ in PANEL]
