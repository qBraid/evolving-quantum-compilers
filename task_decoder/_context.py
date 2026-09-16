"""The object a candidate decodes against.

Lives in its own module so the worker subprocess can import it by file path
without putting the task directory (and therefore ``benchmarks.py``) on
``sys.path``.
"""

from __future__ import annotations

import pymatching
import stim


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
