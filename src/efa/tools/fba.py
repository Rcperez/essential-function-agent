"""Flux Balance Analysis (FBA) essentiality channel via COBRApy.

Given a genome-scale metabolic model and a gene identifier, computes
single-gene-deletion essentiality: simulates wildtype growth, simulates
knockout growth, and classifies essentiality based on the ratio of
knockout to wildtype growth against a configurable threshold (default:
knockout growth < 1% of wildtype -> essential).

This is the public-prototype FBA wrapper: a clean general adapter with no
organism-specific calibration. The LLM reasoner consumes raw growth-rate
numbers and the threshold-based binary call directly.

FBA reference: Orth, Thiele, Palsson 2010 (Nature Biotechnology).
COBRApy reference: Ebrahim, Lerman, Palsson, Hyduke 2013 (BMC Systems Biology).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cobra


DEFAULT_THRESHOLD_RATIO = 0.01  # KO < 1% of WT -> essential


@dataclass
class GrowthRateResult:
    """Result of single-gene-deletion FBA simulation."""

    gene_id: str
    model_id: str
    wildtype_growth_rate: float
    knockout_growth_rate: float
    growth_ratio: float
    is_essential: bool
    threshold_ratio: float
    objective_id: str
    status: str


class FBAEssentialityTool:
    """FBA-based single-gene-deletion essentiality predictor.

    Wraps COBRApy's gene-knockout simulation with a clean structured
    result and an in-memory wildtype-growth cache keyed by model.id.
    The caller is responsible for providing the cobra.Model object;
    load_model() is a convenience for common cases.
    """

    def __init__(self) -> None:
        self._wt_cache: dict[str, float] = {}

    def compute_essentiality(
        self,
        model: cobra.Model,
        gene_id: str,
        threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    ) -> GrowthRateResult:
        """Run single-gene knockout and classify essentiality.

        Returns wildtype growth, knockout growth, their ratio, and a
        boolean essentiality classification (knockout < threshold_ratio
        * wildtype -> essential).

        Raises KeyError if gene_id is not in the model.
        """
        if gene_id not in model.genes:
            raise KeyError(
                f"gene {gene_id!r} not in model {model.id!r}"
            )

        wt_growth = self._wildtype_growth(model)

        with model:
            model.genes.get_by_id(gene_id).knock_out()
            sol = model.optimize()
            ko_growth = float(sol.objective_value or 0.0)
            status = str(sol.status)

        threshold = threshold_ratio * wt_growth
        ratio = (ko_growth / wt_growth) if wt_growth > 0 else 0.0
        is_essential = ko_growth < threshold

        try:
            objective_id = str(list(model.objective.variables)[0].name)
        except (AttributeError, IndexError):
            objective_id = ""

        return GrowthRateResult(
            gene_id=gene_id,
            model_id=str(model.id),
            wildtype_growth_rate=float(wt_growth),
            knockout_growth_rate=ko_growth,
            growth_ratio=float(ratio),
            is_essential=bool(is_essential),
            threshold_ratio=float(threshold_ratio),
            objective_id=objective_id,
            status=status,
        )

    def _wildtype_growth(self, model: cobra.Model) -> float:
        """Compute wildtype growth and cache per model.id in memory."""
        mid = str(model.id)
        if mid not in self._wt_cache:
            sol = model.optimize()
            self._wt_cache[mid] = float(sol.objective_value or 0.0)
        return self._wt_cache[mid]

    @staticmethod
    def load_model(source: Union[str, Path]) -> cobra.Model:
        """Load a COBRA model from a path or by built-in name.

        Accepts:
        - Path to an SBML file (.xml, .sbml, .xml.gz)
        - COBRApy built-in name: 'textbook', 'iJO1366', 'iML1515', 'salmonella'
        """
        s = str(source)
        looks_like_path = os.path.exists(s) or s.endswith(
            (".xml", ".sbml", ".xml.gz")
        )
        if looks_like_path:
            return cobra.io.read_sbml_model(s)
        return cobra.io.load_model(s)


__all__ = ["FBAEssentialityTool", "GrowthRateResult"]
