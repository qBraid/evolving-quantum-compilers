"""Trusted evaluator for the decoder task.

Owns the circuits, the sampling and the truth. A candidate returns predictions
and nothing else; this module measures the logical error rate itself and
compares it against plain MWPM on identical shots.

Writes ``metrics.json`` and ``correct.json`` into ``results_dir``.

    python task_decoder/evaluate.py --program_path task_decoder/initial.py \
        --results_dir /tmp/out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymatching
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import benchmarks  # noqa: E402

FAILURE_SCORE = 0.0
BASELINE_CACHE = os.path.join(HERE, "baselines_decoder.json")
TIME_LIMIT_SECONDS = 600.0


class DecodeContext:
    """Everything a candidate may see. Deliberately excludes the observables."""

    def __init__(self, dem: stim.DetectorErrorModel):
        self.dem = dem
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.matching = pymatching.Matching.from_detector_error_model(dem)
        self._edges = self.matching.edges()
        self.default_weights = [attrs.get("weight", 1.0) for _, _, attrs in self._edges]
        self.edges = [
            (u, v, attrs.get("error_probability", 0.0), attrs.get("fault_ids", set()))
            for u, v, attrs in self._edges
        ]

    def new_matching(self, weights) -> pymatching.Matching:
        """A Matching with the same topology but the weights you supply."""
        weights = list(weights)
        if len(weights) != len(self._edges):
            raise ValueError(
                f"new_matching expects {len(self._edges)} weights, got {len(weights)}"
            )
        fresh = pymatching.Matching()
        for (u, v, attrs), weight in zip(self._edges, weights):
            fault_ids = attrs.get("fault_ids", set())
            if v is None:
                # A boundary edge: one endpoint is the virtual boundary node.
                fresh.add_boundary_edge(
                    u, fault_ids=fault_ids, weight=float(weight),
                    merge_strategy="replace",
                )
            else:
                fresh.add_edge(
                    u, v, fault_ids=fault_ids, weight=float(weight),
                    merge_strategy="replace",
                )
        return fresh


def logical_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    wrong = np.any(predictions.astype(bool) != observables.astype(bool), axis=1)
    return float(np.mean(wrong))


def _versions() -> str:
    return f"stim={stim.__version__},pymatching={pymatching.__version__},panel={benchmarks.SEED}"


def compute_baselines() -> Dict[str, Dict[str, float]]:
    """Plain MWPM on every instance. This is the bar the score is measured against."""
    out: Dict[str, Dict[str, float]] = {}
    for instance in benchmarks.load_panel():
        matching = pymatching.Matching.from_detector_error_model(instance.dem)
        t0 = time.time()
        predictions = matching.decode_batch(instance.detectors)
        out[instance.name] = {
            "mwpm_ler": logical_error_rate(predictions, instance.observables),
            "mwpm_seconds": time.time() - t0,
        }
    return out


def load_baselines() -> Dict[str, Dict[str, float]]:
    """Cached, but fingerprinted: stim and pymatching both move the numbers."""
    fingerprint = _versions()
    if os.path.exists(BASELINE_CACHE):
        try:
            with open(BASELINE_CACHE) as handle:
                cached = json.load(handle)
            if cached.get("fingerprint") == fingerprint:
                return cached["baselines"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    baselines = compute_baselines()
    try:
        with open(BASELINE_CACHE, "w") as handle:
            json.dump({"fingerprint": fingerprint, "baselines": baselines}, handle, indent=2)
    except OSError:
        pass
    return baselines


def load_candidate(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "decode_batch"):
        raise AttributeError("the program must define decode_batch(detectors, ctx)")
    return module


def validate(predictions: Any, shots: int, num_observables: int) -> Optional[str]:
    array = np.asarray(predictions)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] != shots:
        return f"returned {array.shape[0]} rows, expected {shots}"
    if array.shape[1] != num_observables:
        return f"returned {array.shape[1]} columns, expected {num_observables}"
    if not np.isin(array, (0, 1, True, False)).all():
        return "predictions must be 0/1"
    return None


def evaluate_program(program_path: str) -> Dict[str, Any]:
    module = load_candidate(program_path)
    baselines = load_baselines()
    panel = benchmarks.load_panel()

    ratios: List[float] = []
    per_instance: Dict[str, Any] = {}
    problems: List[str] = []
    total_seconds = 0.0

    for instance in panel:
        ctx = DecodeContext(instance.dem)
        mwpm_ler = baselines[instance.name]["mwpm_ler"]

        t0 = time.time()
        predictions = module.decode_batch(instance.detectors, ctx)
        elapsed = time.time() - t0
        total_seconds += elapsed

        problem = validate(predictions, instance.shots, instance.observables.shape[1])
        if problem:
            problems.append(f"{instance.name}: {problem}")
            continue

        # Determinism: same inputs, same answer.
        again = module.decode_batch(instance.detectors, DecodeContext(instance.dem))
        if not np.array_equal(np.asarray(predictions).astype(np.uint8).reshape(instance.shots, -1),
                              np.asarray(again).astype(np.uint8).reshape(instance.shots, -1)):
            problems.append(f"{instance.name}: not deterministic across two identical calls")
            continue

        ler = logical_error_rate(predictions, instance.observables)
        # A decoder that never errs on this many shots is suspicious, not perfect;
        # floor the ratio at the resolution the shot count can actually support.
        floor = 1.0 / instance.shots
        ratio = mwpm_ler / max(ler, floor)
        ratios.append(ratio)
        per_instance[instance.name] = {
            "ler": ler,
            "mwpm_ler": mwpm_ler,
            "ratio_vs_mwpm": ratio,
            "seconds": elapsed,
            "shots": instance.shots,
        }

    if total_seconds > TIME_LIMIT_SECONDS:
        problems.append(
            f"took {total_seconds:.0f}s over the panel, limit is {TIME_LIMIT_SECONDS:.0f}s"
        )

    if problems or not ratios:
        return {
            "combined_score": FAILURE_SCORE,
            "public": {"valid": False, "instances_scored": len(ratios)},
            "private": {"problems": problems},
            "text_feedback": "The candidate was rejected:\n" + "\n".join(problems or ["no instance scored"]),
        }

    combined = float(np.mean(ratios))
    lines = [f"Mean logical-error-rate improvement vs plain MWPM: {combined:.4f}x", "", "Per instance:"]
    for name, data in per_instance.items():
        lines.append(
            f"  {name}: LER {data['ler']:.5f} (plain MWPM {data['mwpm_ler']:.5f}) "
            f"-> {data['ratio_vs_mwpm']:.4f}x in {data['seconds']:.1f}s"
        )
    weakest = min(per_instance.items(), key=lambda kv: kv[1]["ratio_vs_mwpm"])
    lines += [
        "",
        f"Weakest instance is {weakest[0]} at {weakest[1]['ratio_vs_mwpm']:.4f}x.",
        f"Whole panel decoded in {total_seconds:.1f}s of the {TIME_LIMIT_SECONDS:.0f}s limit.",
        "Belief-matching reaches 1.55x on this panel, so there is real headroom left.",
    ]

    return {
        "combined_score": combined,
        "public": {
            "valid": True,
            "mean_ratio_vs_mwpm": combined,
            "instances_scored": len(ratios),
            "total_seconds": total_seconds,
        },
        "private": {"per_instance": per_instance},
        "text_feedback": "\n".join(lines),
    }


def _failure_feedback(error: str) -> str:
    return (
        f"The candidate raised: {error}\n\n"
        "Checklist:\n"
        "  - decode_batch(detectors, ctx) returns (shots, num_observables) of 0/1\n"
        "  - it must be deterministic across two identical calls\n"
        "  - ctx.matching / ctx.new_matching(weights) / ctx.edges are what you have;\n"
        "    the observables are not available to you\n"
        "  - the whole panel must decode within the time limit\n"
    )


def main(program_path: str, results_dir: str) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    try:
        metrics = evaluate_program(program_path)
        correct = bool(metrics["public"].get("valid"))
    except Exception as exc:  # noqa: BLE001 - the model needs to see this
        metrics = {
            "combined_score": FAILURE_SCORE,
            "public": {"valid": False},
            "private": {"error": str(exc), "traceback": traceback.format_exc()},
            "text_feedback": _failure_feedback(str(exc)),
        }
        correct = False

    with open(os.path.join(results_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    with open(os.path.join(results_dir, "correct.json"), "w") as handle:
        json.dump({"correct": correct}, handle, indent=2)

    print(f"correct = {correct}")
    print(f"score   = {metrics.get('combined_score')}")
    print()
    print(metrics.get("text_feedback", ""))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
