# Qubit layout selection

A quantum circuit is written as if any qubit can interact with any other. Real
hardware is not like that: each device has a fixed coupling graph, and a
two-qubit gate can only be applied to a pair of qubits joined by an edge. The
compiler bridges the gap by inserting SWAP gates to shuttle states into place.

SWAPs are expensive. Each one costs three CNOTs, and on current hardware CNOTs
dominate both the runtime and the error budget of a circuit. Reducing them is
one of the highest-leverage things a quantum compiler does.

How many SWAPs are needed depends heavily on where each logical qubit *starts*.
That starting assignment is the initial layout, and choosing it well is your
job here.

## What you are optimising

```python
choose_layout(circuit, coupling_map) -> list[int]
```

- `circuit` is a `qiskit.QuantumCircuit`.
- `coupling_map` is a `qiskit.transpiler.CouplingMap` describing which physical
  qubit pairs can host a two-qubit gate.
- Return a list `layout` where `layout[i]` is the physical qubit that logical
  qubit `i` starts on. Entries must be distinct and within the device.

The evaluator then routes the circuit with a fixed, seeded SabreSwap pass and
counts the two-qubit gates in the compiled result.

## Score

For each benchmark instance:

```
speedup = (gates using the identity layout) / (gates using your layout)
```

The reported score is the mean of `speedup` across all five instances. Higher
is better. `1.0` means you matched the identity layout; above `1.0` means you
beat it.

Routing method, basis gates, optimisation level and random seed are all pinned.
The layout is the only thing that varies, so any change in the score is caused
by your layout and nothing else.

## Benchmark panel

Five instances, spanning different interaction structures and device topologies:

| instance | interaction graph | device |
|---|---|---|
| `qaoa3reg-12q-grid3x4` | random 3-regular | 3x4 grid |
| `qaoa-sparse-10q-line` | random sparse | 10-qubit line |
| `ring-12q-grid3x4` | 12-cycle | 3x4 grid |
| `twolocal-10q-line` | stride-3 entangler | 10-qubit line |
| `qft-8q-line` | all-to-all (QFT) | 8-qubit line |

`qft-8q-line` is a deliberate control. Its interaction graph is complete, so no
layout can help much. Do not contort the heuristic to chase it; just do not
regress on it.

`ring-12q-grid3x4` has the most room: a 3x4 grid contains a Hamiltonian cycle,
so a perfect embedding of the ring exists and needs no SWAPs at all.

## Reference points

Measured on this panel, as two-qubit gate counts:

| instance | identity layout | Qiskit `SabreLayout` |
|---|---|---|
| `qaoa3reg-12q-grid3x4` | 156 | 113 |
| `qaoa-sparse-10q-line` | 195 | 172 |
| `ring-12q-grid3x4` | 102 | 48 |
| `twolocal-10q-line` | 154 | 111 |
| `qft-8q-line` | 188 | 167 |

`SabreLayout` is Qiskit's production layout pass. Matching it is a strong
result; beating it on any instance is a genuinely good one.

## Rules

- Be deterministic. The evaluator calls your function twice on identical inputs
  and rejects the candidate if the two layouts differ. Seed any randomness with
  a fixed constant.
- Return a valid layout: correct length, distinct entries, all within the
  device.
- No filesystem access, no network, no importing the evaluator or its
  benchmarks. Everything you need arrives as an argument.
- Stay within the time limit. A few seconds per instance is ample; a search
  that runs for minutes will be killed.
