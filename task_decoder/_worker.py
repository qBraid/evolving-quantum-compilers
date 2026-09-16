"""Runs one candidate's ``decode_batch`` in a subprocess. Never sees the truth.

The parent (``evaluate.py``) holds the observables. This process is handed only
the detection events and the detector error model, both of which a real decoder
would have on hardware. It writes predictions to a file and exits.

Invoked as::

    python _worker.py <spec.json>

Not part of the task surface -- candidates never import this.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def _sanitised_path(task_dir: str) -> list:
    """sys.path with the task directory removed.

    The panel is seeded, so ``benchmarks.py`` is not merely a container for the
    answers -- it is a recipe for regenerating them exactly. Keeping its
    directory off the path means the obvious ``import benchmarks`` fails.

    This is a guard rail, not a sandbox: this process still has a filesystem and
    could read that file if it went looking. The plausibility gate in the parent
    is what makes doing so unrewarding. See SECURITY in evaluate.py.
    """
    task_dir = os.path.realpath(task_dir)
    return [p for p in sys.path if p and os.path.realpath(p) != task_dir]


def main(spec_path: str) -> int:
    with open(spec_path) as handle:
        spec = json.load(handle)

    sys.path[:] = _sanitised_path(spec["task_dir"])
    os.chdir(spec["scratch_dir"])

    # Imported after the path is cleaned so the candidate inherits the clean one.
    import stim  # noqa: E402
    import importlib.util  # noqa: E402

    # By explicit file path, so the task directory stays off sys.path entirely.
    here = os.path.dirname(os.path.abspath(__file__))
    ctx_spec = importlib.util.spec_from_file_location(
        "_decoder_context", os.path.join(here, "_context.py")
    )
    ctx_module = importlib.util.module_from_spec(ctx_spec)
    ctx_spec.loader.exec_module(ctx_module)
    DecodeContext = ctx_module.DecodeContext

    dem = stim.DetectorErrorModel(open(spec["dem_path"]).read())
    detectors = np.load(spec["detectors_path"])

    module_spec = importlib.util.spec_from_file_location(
        "candidate_program", spec["program_path"]
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load {spec['program_path']}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    if not hasattr(module, "decode_batch"):
        raise AttributeError("the program must define decode_batch(detectors, ctx)")

    ctx = DecodeContext(dem)
    start = time.time()
    predictions = module.decode_batch(detectors, ctx)
    elapsed = time.time() - start

    array = np.asarray(predictions)
    if array.ndim == 1:
        array = array[:, None]
    np.save(spec["output_path"], array)
    with open(spec["timing_path"], "w") as handle:
        json.dump({"seconds": elapsed}, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
