"""Trusted benchmark panel for the qubit-layout task.

This module is owned by the evaluator. Evolved candidates never import it and
never construct their own circuits -- they only ever receive a circuit and a
coupling map as arguments and return a layout. That separation is what makes
the score hard to game: the evaluator holds the reference circuit and measures
the routed gate count itself.

Every instance is deterministic. Nothing here depends on wall-clock time, the
process RNG, or the environment.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.transpiler import CouplingMap

# Gate set every circuit is compiled into before counting. Fixing this makes
# "number of two-qubit gates" a well-defined number rather than an artifact of
# whatever basis the input happened to use.
BASIS_GATES: List[str] = ["rz", "sx", "x", "cx"]

# Pinned so that routing is reproducible. Routing (SabreSwap) is stochastic;
# pinning the seed means a score difference between two candidates is caused by
# their layouts and nothing else.
TRANSPILE_SEED: int = 1234


# --------------------------------------------------------------------------
# circuit families
# --------------------------------------------------------------------------


def _qaoa(num_qubits: int, edges: List[Tuple[int, int]], reps: int = 2) -> QuantumCircuit:
    """QAOA-style circuit: the interaction graph *is* ``edges``.

    Layout matters enormously here -- every edge that is not mapped onto a
    physical coupling has to be bridged with SWAPs.
    """
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for _ in range(reps):
        for u, v in edges:
            qc.cx(u, v)
            qc.rz(0.3, v)
            qc.cx(u, v)
        for q in range(num_qubits):
            qc.rx(0.2, q)
    return qc


def _random_regular_edges(num_qubits: int, degree: int, seed: int) -> List[Tuple[int, int]]:
    """Deterministic near-``degree``-regular graph, built without networkx."""
    rng = random.Random(seed)
    pairs = list(itertools.combinations(range(num_qubits), 2))
    rng.shuffle(pairs)
    deg = {q: 0 for q in range(num_qubits)}
    edges: List[Tuple[int, int]] = []
    for u, v in pairs:
        if deg[u] < degree and deg[v] < degree:
            edges.append((u, v))
            deg[u] += 1
            deg[v] += 1
    return edges


def _random_sparse_edges(num_qubits: int, num_edges: int, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs = list(itertools.combinations(range(num_qubits), 2))
    rng.shuffle(pairs)
    return pairs[:num_edges]


def _ring_edges(num_qubits: int) -> List[Tuple[int, int]]:
    """Cycle interaction graph: 0-1-2-...-(n-1)-0."""
    return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]


def _two_local(num_qubits: int, reps: int = 3, stride: int = 3) -> QuantumCircuit:
    """Hardware-efficient ansatz with a deliberately long-range entangler.

    ``stride``-separated pairs are cheap on an all-to-all machine and expensive
    on a line, so the layout has real work to do.
    """
    qc = QuantumCircuit(num_qubits)
    for _ in range(reps):
        for q in range(num_qubits):
            qc.ry(0.1 * (q + 1), q)
        for q in range(num_qubits):
            partner = (q + stride) % num_qubits
            if partner != q:
                qc.cx(q, partner)
    return qc


def _qft(num_qubits: int) -> QuantumCircuit:
    """QFT -- an all-to-all interaction graph.

    Included on purpose as a *control*. Almost no layout can help a fully
    connected interaction graph, so a candidate that claims a large win here is
    doing something suspicious, and a candidate that regresses here has
    overfitted to the sparse instances.
    """
    qc = QuantumCircuit(num_qubits)
    qc.append(QFTGate(num_qubits), range(num_qubits))
    return qc.decompose(reps=4)


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One (circuit, target connectivity) pair."""

    instance_id: str
    description: str
    build_circuit: Callable[[], QuantumCircuit]
    build_coupling: Callable[[], CouplingMap]

    def circuit(self) -> QuantumCircuit:
        return self.build_circuit()

    def coupling_map(self) -> CouplingMap:
        return self.build_coupling()


PANEL: List[Instance] = [
    Instance(
        instance_id="qaoa3reg-12q-grid3x4",
        description="12-qubit QAOA on a random 3-regular graph, mapped to a 3x4 grid.",
        build_circuit=lambda: _qaoa(12, _random_regular_edges(12, 3, seed=11)),
        build_coupling=lambda: CouplingMap.from_grid(3, 4),
    ),
    Instance(
        instance_id="qaoa-sparse-10q-line",
        description="10-qubit QAOA on a random sparse graph (15 edges), mapped to a line.",
        build_circuit=lambda: _qaoa(10, _random_sparse_edges(10, 15, seed=23)),
        build_coupling=lambda: CouplingMap.from_line(10),
    ),
    Instance(
        instance_id="ring-12q-grid3x4",
        description=(
            "12-qubit ring (cycle) interaction graph on a 3x4 grid. A Hamiltonian "
            "cycle exists in the grid, so a perfect embedding is possible -- this "
            "is the instance with the most headroom in the panel."
        ),
        build_circuit=lambda: _qaoa(12, _ring_edges(12)),
        build_coupling=lambda: CouplingMap.from_grid(3, 4),
    ),
    Instance(
        instance_id="twolocal-10q-line",
        description="10-qubit hardware-efficient ansatz, stride-3 entangler, on a line.",
        build_circuit=lambda: _two_local(10, reps=3, stride=3),
        build_coupling=lambda: CouplingMap.from_line(10),
    ),
    Instance(
        instance_id="qft-8q-line",
        description="8-qubit QFT on a line. All-to-all control instance: little headroom by design.",
        build_circuit=lambda: _qft(8),
        build_coupling=lambda: CouplingMap.from_line(8),
    ),
]

PANEL_BY_ID = {inst.instance_id: inst for inst in PANEL}
