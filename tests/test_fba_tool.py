"""Tests for src/efa/tools/fba.py.

Uses COBRApy's bundled textbook E. coli core model (no external files,
no network). Cross-checks the tool's classifications against COBRApy's
reference single_gene_deletion results so the tests stay robust to
model variant.
"""

from __future__ import annotations

import cobra
import pytest
from cobra.flux_analysis import single_gene_deletion

from efa.tools.fba import FBAEssentialityTool, GrowthRateResult


@pytest.fixture(scope="module")
def textbook_model() -> cobra.Model:
    """E. coli core model bundled with COBRApy."""
    return cobra.io.load_model("textbook")


@pytest.fixture(scope="module")
def cobra_reference_classifications(textbook_model: cobra.Model):
    """COBRApy's reference single_gene_deletion classification for the model.

    Returns a tuple (essential_gene_id, non_essential_gene_id) selected
    from COBRApy's results, so tests stay robust to model variants.
    """
    df = single_gene_deletion(textbook_model, textbook_model.genes)
    wt_max = float(df["growth"].max())
    threshold = wt_max * 0.01

    essential_rows = df[df["growth"] < threshold]
    non_essential_rows = df[df["growth"] >= wt_max * 0.99]

    essential_id = next(iter(essential_rows["ids"].iloc[0]))
    non_essential_id = next(iter(non_essential_rows["ids"].iloc[0]))
    return essential_id, non_essential_id


@pytest.fixture
def fba_tool() -> FBAEssentialityTool:
    return FBAEssentialityTool()


def test_compute_essentiality_returns_structured_result(
    fba_tool: FBAEssentialityTool,
    textbook_model: cobra.Model,
) -> None:
    gene_id = next(iter(textbook_model.genes)).id
    result = fba_tool.compute_essentiality(textbook_model, gene_id)
    assert isinstance(result, GrowthRateResult)
    assert result.gene_id == gene_id
    assert result.wildtype_growth_rate > 0
    assert result.knockout_growth_rate >= 0
    assert result.status in ("optimal", "infeasible", "feasible")


def test_unknown_gene_raises_keyerror(
    fba_tool: FBAEssentialityTool,
    textbook_model: cobra.Model,
) -> None:
    with pytest.raises(KeyError):
        fba_tool.compute_essentiality(textbook_model, "nonexistent_gene_id")


def test_tool_agrees_with_cobra_reference_on_essential_gene(
    fba_tool: FBAEssentialityTool,
    textbook_model: cobra.Model,
    cobra_reference_classifications,
) -> None:
    essential_id, _ = cobra_reference_classifications
    result = fba_tool.compute_essentiality(textbook_model, essential_id)
    assert result.is_essential, (
        f"Tool says {essential_id} not essential; COBRApy reference says it is"
    )
    assert result.growth_ratio < 0.01


def test_tool_agrees_with_cobra_reference_on_non_essential_gene(
    fba_tool: FBAEssentialityTool,
    textbook_model: cobra.Model,
    cobra_reference_classifications,
) -> None:
    _, non_essential_id = cobra_reference_classifications
    result = fba_tool.compute_essentiality(textbook_model, non_essential_id)
    assert not result.is_essential, (
        f"Tool says {non_essential_id} essential; COBRApy reference says it is not"
    )
    assert result.growth_ratio > 0.99


def test_wildtype_growth_is_cached_in_memory(
    fba_tool: FBAEssentialityTool,
    textbook_model: cobra.Model,
) -> None:
    """Second call on same model should hit the in-memory cache."""
    assert textbook_model.id not in fba_tool._wt_cache
    wt1 = fba_tool._wildtype_growth(textbook_model)
    assert textbook_model.id in fba_tool._wt_cache
    wt2 = fba_tool._wildtype_growth(textbook_model)
    assert wt1 == wt2


def test_threshold_ratio_changes_classification(
    fba_tool: FBAEssentialityTool,
    textbook_model: cobra.Model,
    cobra_reference_classifications,
) -> None:
    """A non-essential gene becomes 'essential' under an extreme threshold."""
    _, non_essential_id = cobra_reference_classifications
    r_strict = fba_tool.compute_essentiality(
        textbook_model, non_essential_id, threshold_ratio=0.01,
    )
    r_extreme = fba_tool.compute_essentiality(
        textbook_model, non_essential_id, threshold_ratio=1.5,
    )
    assert not r_strict.is_essential
    # threshold 1.5 means we'd need KO growth > 1.5 * WT to be non-essential,
    # which is impossible -> always classified essential
    assert r_extreme.is_essential
    assert r_strict.threshold_ratio == 0.01
    assert r_extreme.threshold_ratio == 1.5
