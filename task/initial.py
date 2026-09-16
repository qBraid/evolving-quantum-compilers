"""Seed program: choose an initial qubit layout.

You are optimising ONE function, ``choose_layout``. Given a logical circuit and
the physical connectivity of a device, decide which physical qubit each logical
qubit should start on.

The evaluator then routes the circuit with a fixed, seeded SabreSwap pass and
counts the two-qubit gates in the result. Fewer is better. Your layout is the
only thing that changes between candidates -- routing, basis and seed are all
pinned -- so every gate you save is attributable to the layout you picked.

CONTRACT
    choose_layout(circuit, coupling_map) -> list[int]

    circuit       qiskit.QuantumCircuit, the logical circuit
    coupling_map  qiskit.transpiler.CouplingMap, the device connectivity

    Returns a list ``layout`` of length ``circuit.num_qubits`` where
    ``layout[i]`` is the PHYSICAL qubit that logical qubit ``i`` is placed on.
    Entries must be distinct and in ``range(coupling_map.size())``.

HARD RULES (violating any of these scores the candidate as incorrect)
    - Return a valid layout as described above.
    - Be deterministic. The evaluator calls you twice and compares. If you use
      randomness, seed it yourself with a fixed constant.
    - Do not import the evaluator, read its files, or touch the filesystem or
      network. You get the circuit and the coupling map; that is all you need.
    - Stay inside the time limit. A few seconds per instance is plenty.
"""

from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


# --------------------------------------------------------------------------
# Helpers available to you. You may edit, replace, or ignore these -- but note
# that only code inside the EVOLVE-BLOCK is mutable, so if you want to change a
# helper, move it inside the block.
# --------------------------------------------------------------------------


def interaction_graph(circuit: QuantumCircuit) -> dict:
    """Map ``(logical_u, logical_v) -> number of two-qubit gates between them``.

    This is the thing you are trying to embed into the device's connectivity.
    """
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    weights: dict = {}
    for instruction in circuit.data:
        qubits = instruction.qubits
        if len(qubits) != 2:
            continue
        u, v = index[qubits[0]], index[qubits[1]]
        key = (u, v) if u < v else (v, u)
        weights[key] = weights.get(key, 0) + 1
    return weights


def physical_adjacency(coupling_map: CouplingMap) -> dict:
    """Map ``physical_qubit -> set(neighbouring physical qubits)``."""
    adjacency: dict = {q: set() for q in range(coupling_map.size())}
    for u, v in coupling_map.get_edges():
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def physical_distances(coupling_map: CouplingMap) -> List[List[int]]:
    """All-pairs shortest-path distances between physical qubits."""
    return coupling_map.distance_matrix.astype(int).tolist()


# EVOLVE-BLOCK-START
def choose_layout(circuit: QuantumCircuit, coupling_map: CouplingMap) -> List[int]:
    """Place logical qubits onto physical qubits.

    Baseline strategy: put the busiest logical qubit on the best-connected
    physical qubit, and keep going down both rankings in step.

    This ignores the *structure* of the interaction graph entirely -- it only
    looks at degree. Two logical qubits that talk to each other constantly can
    still end up on opposite sides of the device. There is a lot of room here.
    """
    num_logical = circuit.num_qubits
    weights = interaction_graph(circuit)
    adjacency = physical_adjacency(coupling_map)

    # How much two-qubit work does each logical qubit participate in?
    logical_load = {q: 0 for q in range(num_logical)}
    for (u, v), count in weights.items():
        logical_load[u] += count
        logical_load[v] += count

    # Busiest logical qubits first; ties broken by index so this is deterministic.
    logical_order = sorted(
        range(num_logical), key=lambda q: (-logical_load[q], q)
    )

    # Best-connected physical qubits first, same tie-breaking rule.
    physical_order = sorted(
        range(coupling_map.size()), key=lambda q: (-len(adjacency[q]), q)
    )

    layout = [0] * num_logical
    for logical, physical in zip(logical_order, physical_order):
        layout[logical] = physical
    return layout
# EVOLVE-BLOCK-END


def run_layout(instance_id: str, circuit: QuantumCircuit, coupling_map: CouplingMap) -> dict:
    """Entry point the evaluator calls. Do not change this function.

    It exists only to hand your layout back to the evaluator alongside the id of
    the instance it was computed for. The evaluator does the routing and the
    counting itself against its own copy of the circuit.
    """
    layout = choose_layout(circuit, coupling_map)
    return {"instance_id": instance_id, "layout": [int(q) for q in layout]}
