"""Trusted evaluator for the qubit-layout task.

    python evaluate.py --program_path initial.py --results_dir results/

Writes ``metrics.json`` and ``correct.json`` into ``results_dir``.

Division of trust
-----------------
The candidate returns a layout and nothing else. This file builds the circuit,
performs the routing, and counts the gates, all against its own copy of the
benchmark. A candidate cannot report its own score, cannot choose its own
benchmark, and cannot alter the routing pass -- so the only way to move the
number is to actually pick a better layout.

Baselines are cached in ``baselines.json`` next to this file. Delete that file
to force a recompute (it takes a few seconds).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from qiskit import transpile

HERE = os.path.dirname(os.path.abspath(__file__))

# Shinka invokes this file as `python /abs/path/to/evaluate.py` from whatever
# directory the run was launched in, so `benchmarks` is not importable by
# default. Put our own directory on the path first.
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from benchmarks import BASIS_GATES, PANEL, PANEL_BY_ID, TRANSPILE_SEED  # noqa: E402

BASELINE_PATH = os.path.join(HERE, "baselines.json")

# Score reported when the candidate is unusable. Zero rather than a negative
# number because the score is a ratio against the trivial layout, so zero is
# already "infinitely worse than doing nothing".
FAILURE_SCORE = 0.0


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def routed_two_qubit_count(circuit, coupling_map, layout: Optional[List[int]]) -> int:
    """Route ``circuit`` onto ``coupling_map`` from ``layout`` and count 2q gates.

    Everything except the layout is pinned: same basis, same routing pass, same
    seed, same optimisation level. ``layout=None`` lets Qiskit choose its own
    (SabreLayout), which is how the strong baseline is measured.
    """
    kwargs: Dict[str, Any] = dict(
        coupling_map=coupling_map,
        basis_gates=BASIS_GATES,
        routing_method="sabre",
        optimization_level=1,
        seed_transpiler=TRANSPILE_SEED,
    )
    if layout is None:
        kwargs["layout_method"] = "sabre"
    else:
        kwargs["initial_layout"] = layout

    out = transpile(circuit, **kwargs)
    ops = out.count_ops()
    # After compiling to BASIS_GATES the only two-qubit gate is cx.
    return int(ops.get("cx", 0))


def compute_baselines() -> Dict[str, Dict[str, int]]:
    """Trivial and SabreLayout gate counts for every panel instance."""
    baselines: Dict[str, Dict[str, int]] = {}
    for inst in PANEL:
        circuit = inst.circuit()
        coupling = inst.coupling_map()
        trivial = list(range(circuit.num_qubits))
        baselines[inst.instance_id] = {
            "trivial": routed_two_qubit_count(circuit, coupling, trivial),
            "sabre": routed_two_qubit_count(circuit, coupling, None),
        }
    return baselines


def load_baselines() -> Dict[str, Dict[str, int]]:
    """Baselines for the current panel, recomputed whenever they could be stale.

    The cache is keyed on the Qiskit version. Qiskit's routing changes between
    releases, so baselines computed under one version are not comparable with
    candidate gate counts measured under another -- reusing them would silently
    shift every score. Cheaper to recompute (a few seconds) than to be subtly
    wrong.
    """
    import qiskit

    fingerprint = {
        "qiskit_version": qiskit.__version__,
        "transpile_seed": TRANSPILE_SEED,
        "basis_gates": list(BASIS_GATES),
        "instances": sorted(inst.instance_id for inst in PANEL),
    }

    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH) as fh:
                cached = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            return cached["baselines"]

    baselines = compute_baselines()
    with open(BASELINE_PATH, "w") as fh:
        json.dump(
            {"fingerprint": fingerprint, "baselines": baselines},
            fh,
            indent=2,
            sort_keys=True,
        )
    return baselines


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


# Layout seen the first time each instance was issued. The harness issues every
# instance twice, so the second sighting is a free determinism check. Populated
# per evaluation; each evaluation runs in a fresh process.
#
# NOTE: this assumes the harness runs sequentially in one process, which is the
# default (``run_workers=1``). If you ever raise ``run_workers``, move this check
# into ``aggregate`` instead -- it is duplicated there as a safety net.
_FIRST_LAYOUT: Dict[str, List[int]] = {}


def validate_result(result: Any) -> Tuple[bool, Optional[str]]:
    """Structural check on one candidate return value."""
    if not isinstance(result, dict):
        return False, f"run_layout must return a dict, got {type(result).__name__}."

    instance_id = result.get("instance_id")
    if instance_id not in PANEL_BY_ID:
        return False, f"Unknown instance_id {instance_id!r}."

    layout = result.get("layout")
    if not isinstance(layout, (list, tuple)):
        return False, f"layout must be a list, got {type(layout).__name__}."

    inst = PANEL_BY_ID[instance_id]
    num_logical = inst.circuit().num_qubits
    num_physical = inst.coupling_map().size()

    if len(layout) != num_logical:
        return False, (
            f"{instance_id}: layout has {len(layout)} entries, "
            f"expected {num_logical}."
        )
    if not all(isinstance(q, int) and not isinstance(q, bool) for q in layout):
        return False, f"{instance_id}: layout entries must be ints."
    if len(set(layout)) != len(layout):
        return False, f"{instance_id}: layout entries must be distinct."
    if not all(0 <= q < num_physical for q in layout):
        return False, (
            f"{instance_id}: layout entries must be in [0, {num_physical})."
        )

    layout = [int(q) for q in layout]
    previous = _FIRST_LAYOUT.setdefault(instance_id, layout)
    if previous != layout:
        return False, (
            f"{instance_id}: choose_layout is not deterministic. Called twice "
            f"with identical inputs it returned {previous} and then {layout}. "
            "Seed any randomness with a fixed constant."
        )
    return True, None


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def aggregate(results: List[Any]) -> Dict[str, Any]:
    """Turn raw candidate returns into the metrics dict Shinka consumes."""
    baselines = load_baselines()

    # The harness runs every instance twice; identical layouts both times is the
    # determinism check.
    by_instance: Dict[str, List[List[int]]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        by_instance.setdefault(result.get("instance_id"), []).append(
            list(result.get("layout", []))
        )

    per_instance: Dict[str, Dict[str, Any]] = {}
    ratios: List[float] = []
    feedback_lines: List[str] = []
    problems: List[str] = []

    for inst in PANEL:
        layouts = by_instance.get(inst.instance_id, [])
        if not layouts:
            problems.append(f"{inst.instance_id}: no result returned.")
            continue
        if any(layout != layouts[0] for layout in layouts[1:]):
            problems.append(
                f"{inst.instance_id}: choose_layout is not deterministic -- "
                "it returned different layouts for identical inputs."
            )
            continue

        layout = layouts[0]
        circuit = inst.circuit()
        coupling = inst.coupling_map()

        started = time.time()
        try:
            count = routed_two_qubit_count(circuit, coupling, layout)
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM as feedback
            problems.append(f"{inst.instance_id}: routing failed ({exc}).")
            continue
        elapsed = time.time() - started

        trivial = baselines[inst.instance_id]["trivial"]
        sabre = baselines[inst.instance_id]["sabre"]
        ratio = trivial / count if count > 0 else 0.0
        ratios.append(ratio)

        per_instance[inst.instance_id] = {
            "two_qubit_gates": count,
            "trivial_baseline": trivial,
            "sabre_baseline": sabre,
            "speedup_vs_trivial": round(ratio, 4),
            "speedup_vs_sabre": round(sabre / count, 4) if count > 0 else 0.0,
            "beats_sabre": bool(count < sabre),
            "route_seconds": round(elapsed, 3),
        }
        feedback_lines.append(
            f"{inst.instance_id}: {count} two-qubit gates "
            f"(trivial {trivial}, Qiskit SabreLayout {sabre}) "
            f"-> {ratio:.3f}x vs trivial, {sabre / count:.3f}x vs Sabre"
            if count > 0
            else f"{inst.instance_id}: degenerate count"
        )

    if problems or not ratios:
        return {
            "combined_score": FAILURE_SCORE,
            "public": {"valid": False, "instances_scored": len(ratios)},
            "private": {"problems": problems},
            "text_feedback": "The candidate was rejected:\n" + "\n".join(problems),
        }

    combined = statistics.fmean(ratios)
    beat_sabre = [
        name for name, data in per_instance.items() if data["beats_sabre"]
    ]

    summary = [
        f"Mean speedup vs trivial layout: {combined:.4f}x "
        f"across {len(ratios)} instances.",
        "",
        "Per instance:",
    ]
    summary.extend(f"  {line}" for line in feedback_lines)
    summary.append("")
    if beat_sabre:
        summary.append(
            "Beat Qiskit's own SabreLayout on: " + ", ".join(sorted(beat_sabre)) + "."
        )
    else:
        summary.append(
            "Did not beat Qiskit's SabreLayout on any instance yet -- that is "
            "the bar worth aiming at."
        )
    worst = min(per_instance.items(), key=lambda kv: kv[1]["speedup_vs_trivial"])
    summary.append(
        f"Weakest instance is {worst[0]} at "
        f"{worst[1]['speedup_vs_trivial']:.3f}x vs trivial."
    )

    return {
        "combined_score": combined,
        "public": {
            "valid": True,
            "mean_speedup_vs_trivial": round(combined, 4),
            "instances_beating_sabre": len(beat_sabre),
            "instances_scored": len(ratios),
        },
        "private": {"per_instance": per_instance},
        "text_feedback": "\n".join(summary),
    }


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def experiment_kwargs(run_index: int) -> Dict[str, Any]:
    """Instance for run ``run_index``. Each instance is issued twice."""
    inst = PANEL[run_index % len(PANEL)]
    return {
        "instance_id": inst.instance_id,
        "circuit": inst.circuit(),
        "coupling_map": inst.coupling_map(),
    }


def _failure_feedback(error: Optional[str], correct: bool) -> str:
    """Actionable text for a candidate that never produced a score."""
    if correct:
        # Scored fine but the aggregator returned nothing to say. Nothing to add.
        return ""

    lines = [
        "This candidate scored nothing because it failed before it could be "
        "evaluated.",
        "",
        f"Error: {error or 'unknown'}",
        "",
        "Checklist:",
        "  - Every helper you call must exist. If you reference a helper, define "
        "it inside the EVOLVE-BLOCK -- code outside the block is not carried "
        "over from your reasoning.",
        "  - choose_layout must return a list of distinct ints, one per logical "
        "qubit, each within the device.",
        "  - It must be deterministic: identical inputs, identical output. Seed "
        "any randomness with a fixed constant.",
        "  - Only the standard library and what is already imported is "
        "available. No filesystem, no network.",
    ]
    return "\n".join(lines)


def _run_without_shinka(program_path: str) -> List[Any]:
    """Fallback path so the task can be smoke-tested before Shinka is installed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("candidate", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {program_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    results = []
    for run_index in range(2 * len(PANEL)):
        results.append(module.run_layout(**experiment_kwargs(run_index)))
    return results


def main(program_path: str, results_dir: str) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)

    try:
        from shinka.core import run_shinka_eval
    except ImportError:
        run_shinka_eval = None

    if run_shinka_eval is not None:
        metrics, correct, error = run_shinka_eval(
            program_path=program_path,
            results_dir=results_dir,
            experiment_fn_name="run_layout",
            num_runs=2 * len(PANEL),
            get_experiment_kwargs=experiment_kwargs,
            validate_fn=validate_result,
            aggregate_metrics_fn=aggregate,
        )

        # Shinka's crash path builds its metrics dict by filtering against its
        # own DEFAULT_METRICS_ON_ERROR, which has no `text_feedback` key -- so a
        # custom error message passed via `default_metrics_on_error` is dropped
        # and the model is told only "score 0", with no reason. Put the reason
        # back, and rewrite the file Shinka already emitted.
        if not metrics.get("text_feedback"):
            metrics["text_feedback"] = _failure_feedback(error, correct)
            with open(os.path.join(results_dir, "metrics.json"), "w") as fh:
                json.dump(metrics, fh, indent=2, default=str)
    else:
        # No Shinka installed: run the same panel in-process and validate by hand.
        try:
            raw = _run_without_shinka(program_path)
        except Exception as exc:  # noqa: BLE001
            metrics, correct, error = (
                {
                    "combined_score": FAILURE_SCORE,
                    "public": {"valid": False},
                    "private": {"error": str(exc)},
                    "text_feedback": f"The candidate raised: {exc}",
                },
                False,
                str(exc),
            )
        else:
            error = None
            for item in raw:
                ok, message = validate_result(item)
                if not ok:
                    error = message
                    break
            if error is not None:
                metrics, correct = (
                    {
                        "combined_score": FAILURE_SCORE,
                        "public": {"valid": False},
                        "private": {"error": error},
                        "text_feedback": f"The candidate was rejected: {error}",
                    },
                    False,
                )
            else:
                metrics = aggregate(raw)
                correct = bool(metrics["public"].get("valid"))

        with open(os.path.join(results_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        with open(os.path.join(results_dir, "correct.json"), "w") as fh:
            json.dump({"correct": correct}, fh, indent=2)

    print(f"correct = {correct}")
    if error:
        print(f"error   = {error}")
    print(f"score   = {metrics.get('combined_score')}")
    print()
    print(metrics.get("text_feedback", ""))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", default="initial.py")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
