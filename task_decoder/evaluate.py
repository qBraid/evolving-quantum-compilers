"""Trusted evaluator for the decoder task.

Owns the circuits, the sampling and the truth. A candidate returns predictions
and nothing else; this module measures the logical error rate itself and
compares it against plain MWPM on identical shots.

Writes ``metrics.json`` and ``correct.json`` into ``results_dir``.

    python task_decoder/evaluate.py --program_path task_decoder/initial.py \
        --results_dir /tmp/out

SECURITY
    The candidate predicts the very thing it is scored against, so unlike the
    layout task in ``task/`` it cannot be allowed to run in this process. Three
    things keep it honest, in order of how much they are relied on:

    1. It runs in a subprocess (``_worker.py``) that is handed only the
       detection events and the DEM. The observables never enter its address
       space.
    2. That subprocess has the task directory removed from ``sys.path``. The
       panel is seeded, so ``benchmarks.py`` is not just a container for the
       answers, it is a recipe for regenerating them -- ``import benchmarks``
       has to fail.
    3. A plausibility gate rejects any candidate scoring better than a
       correlated decoder credibly can (``IMPLAUSIBLE_RATIO``). This is the
       backstop, and the reason 1 and 2 do not have to be airtight: a candidate
       that does find a way to the labels gets rejected rather than crowned.

    What this is not: a sandbox. The subprocess has a filesystem and could read
    ``benchmarks.py`` if it went looking for it. Closing that properly needs OS
    isolation (a container or seccomp), which is out of scope here. Gate 3 is
    what makes the residual risk tolerable -- cheating is detected, not
    prevented.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
WORKER = os.path.join(HERE, "_worker.py")

# Wall-clock ceiling for the whole panel, enforced per instance by killing the
# worker. Must stay at or below the --eval-timeout that run_evolution.py passes
# to Shinka, or the harness SIGKILLs the evaluator instead and the candidate
# gets no feedback at all. The seed decodes the panel in about 0.2s; anything
# approaching this limit is doing per-shot Python work.
TIME_LIMIT_SECONDS = 240.0

# Belief-matching -- the strongest decoder anyone has reported on this panel --
# reaches about 1.5x plain MWPM. A candidate well beyond that has almost
# certainly reached the labels rather than decoded better, so reject it and say
# so. Set deliberately loose: a genuine 2x would be a real result, and is worth
# a manual look rather than a silent score.
IMPLAUSIBLE_RATIO = 3.0

# Determinism is checked on a subsample in a second process. Two processes, not
# two calls, so a candidate that memoises its first answer is still caught.
DETERMINISM_SHOTS = 2000


def logical_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    wrong = np.any(predictions.astype(bool) != observables.astype(bool), axis=1)
    return float(np.mean(wrong))


def _versions() -> str:
    """Fingerprint for the baseline cache.

    Includes the panel definition, not just the seed: changing a distance, a
    physical error rate or a shot count moves every baseline, and none of those
    touch SEED. task/evaluate.py gets this right and this one did not."""
    panel = ";".join(
        f"{name}:{d}:{r}:{p}:{shots}" for name, d, r, p, shots in benchmarks.PANEL
    )
    return (
        f"stim={stim.__version__},pymatching={pymatching.__version__},"
        f"seed={benchmarks.SEED},panel={panel}"
    )


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


class CandidateError(RuntimeError):
    """The candidate subprocess failed, timed out, or returned nothing usable."""


def run_candidate(
    program_path: str,
    dem: stim.DetectorErrorModel,
    detectors: np.ndarray,
    timeout: float,
) -> Tuple[np.ndarray, float]:
    """Decode in a subprocess that never holds the observables.

    Returns (predictions, seconds). Raises CandidateError with a message meant
    for the model rather than for a log.
    """
    scratch = tempfile.mkdtemp(prefix="decoder_eval_")
    try:
        dem_path = os.path.join(scratch, "model.dem")
        det_path = os.path.join(scratch, "detectors.npy")
        out_path = os.path.join(scratch, "predictions.npy")
        timing_path = os.path.join(scratch, "timing.json")
        spec_path = os.path.join(scratch, "spec.json")

        with open(dem_path, "w") as handle:
            handle.write(str(dem))
        np.save(det_path, detectors)
        with open(spec_path, "w") as handle:
            json.dump(
                {
                    "program_path": os.path.abspath(program_path),
                    "task_dir": HERE,
                    "scratch_dir": scratch,
                    "dem_path": dem_path,
                    "detectors_path": det_path,
                    "output_path": out_path,
                    "timing_path": timing_path,
                },
                handle,
            )

        # cwd is the scratch dir so a bare `import benchmarks` cannot resolve
        # through '' on sys.path either.
        completed = subprocess.run(
            [sys.executable, WORKER, spec_path],
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip().splitlines()
            detail = "\n".join(tail[-12:]) if tail else "no stderr"
            raise CandidateError(f"the candidate process exited {completed.returncode}:\n{detail}")
        if not os.path.exists(out_path):
            raise CandidateError("the candidate process wrote no predictions")

        predictions = np.load(out_path)
        seconds = 0.0
        if os.path.exists(timing_path):
            with open(timing_path) as handle:
                seconds = float(json.load(handle).get("seconds", 0.0))
        return predictions, seconds
    except subprocess.TimeoutExpired:
        raise CandidateError(
            f"the candidate exceeded the {timeout:.0f}s limit and was killed"
        ) from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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
    # Baselines first, and before any candidate code has had a chance to run.
    # They are read from a cache file the candidate must never get to influence.
    baselines = load_baselines()
    panel = benchmarks.load_panel()

    ratios: List[float] = []
    per_instance: Dict[str, Any] = {}
    problems: List[str] = []
    total_seconds = 0.0

    for instance in panel:
        mwpm_ler = baselines[instance.name]["mwpm_ler"]
        remaining = max(TIME_LIMIT_SECONDS - total_seconds, 1.0)

        try:
            predictions, elapsed = run_candidate(
                program_path, instance.dem, instance.detectors, remaining
            )
        except CandidateError as exc:
            problems.append(f"{instance.name}: {exc}")
            continue
        total_seconds += elapsed

        problem = validate(predictions, instance.shots, instance.observables.shape[1])
        if problem:
            problems.append(f"{instance.name}: {problem}")
            continue

        # Determinism: a second, independent process on a subsample. Separate
        # processes rather than a second call, so memoising the first answer
        # does not pass.
        subsample = instance.detectors[:DETERMINISM_SHOTS]
        try:
            again, _ = run_candidate(program_path, instance.dem, subsample, remaining)
        except CandidateError as exc:
            problems.append(f"{instance.name}: re-running the candidate failed: {exc}")
            continue
        first = np.asarray(predictions).astype(np.uint8).reshape(instance.shots, -1)
        if not np.array_equal(first[:DETERMINISM_SHOTS],
                              np.asarray(again).astype(np.uint8).reshape(len(subsample), -1)):
            problems.append(f"{instance.name}: not deterministic across two identical calls")
            continue

        ler = logical_error_rate(predictions, instance.observables)
        # A decoder that never errs on this many shots is suspicious, not perfect;
        # floor the ratio at the resolution the shot count can actually support.
        floor = 1.0 / instance.shots
        ratio = mwpm_ler / max(ler, floor)

        if ratio > IMPLAUSIBLE_RATIO:
            problems.append(
                f"{instance.name}: {ratio:.1f}x plain MWPM is past what any known "
                f"decoder achieves here (belief-matching is ~1.5x). Rejected as "
                f"implausible -- predictions must come from the syndrome, not "
                f"from recovering the observables."
            )
            continue

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
        "Belief-matching reaches ~1.5x on this panel (measured: 1.52x on "
        "surface-d5-p005, 1.48x on surface-d5-p008), so there is real headroom "
        "left. It is far too slow to run inside the time limit, so matching that "
        "quality cheaply is the actual problem.",
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
        "  - it runs in a subprocess with no access to the benchmark module\n"
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
