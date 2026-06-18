"""Tests for src/efa/orchestrator_b.py (B-architecture tool-use loop).

Uses a mocked Anthropic client that emits scripted tool_use then a final
verdict, and MagicMock tools. No network, no real LLM, no GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from efa.orchestrator import Case, EvidenceBundle, Verdict
from efa.orchestrator_b import (
    EssentialityOrchestratorB, SYSTEM_PROMPT_B, TOOL_SCHEMAS, dispatch_tool,
)


@pytest.fixture
def sample_case() -> Case:
    return Case(
        case_id="test_rpsB",
        gene_symbol="rpsB",
        locus_tag="b0169",
        organism_taxon=511145,
        uniprot_taxon=83333,
        organism_strain="Escherichia coli K-12 MG1655",
        kegg_gene_id="eco:b0169",
        string_species=511145,
        metabolic_model="iML1515",
        protein_sequence="MATVSMR",
        dna_sequence="ATGGCAACTGT",
        design_axis="1_universal_essential_translation",
        ground_truth_essentiality="essential",
        ground_truth_source="Baba 2006 Keio",
        design_rationale="anchor test case",
    )


def _text_block(text):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(name, tool_id):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.id = tool_id
    b.input = {}
    return b


def _response(content):
    r = MagicMock()
    r.content = content
    return r


# ---------- prompt / schema sanity ----------


def test_system_prompt_b_extends_base():
    from efa.orchestrator import SYSTEM_PROMPT
    assert SYSTEM_PROMPT in SYSTEM_PROMPT_B
    assert "Workflow" in SYSTEM_PROMPT_B
    assert "you decide which channels to query" in SYSTEM_PROMPT_B


def test_six_tool_schemas_present():
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert names == {
        "query_uniprot", "query_kegg", "query_string",
        "query_esm2", "query_evo2", "query_fba",
    }


# ---------- dispatch_tool ----------


def test_dispatch_uniprot_stores_and_renders(sample_case):
    uni = MagicMock()
    ann = MagicMock()
    ann.protein_name = "30S ribosomal protein S2"
    ann.gene_name = "rpsB"
    ann.ec_numbers = []
    ann.ec = []
    ann.function_description = "Ribosomal subunit."
    ann.function = None
    ann.go_terms = []
    ann.pfam_domains = []
    ann.pfam = None
    ann.subcellular_locations = []
    ann.subcellular_location = None
    ann.kegg_xrefs = []
    ann.eggnog_xrefs = []
    ann.pdb_xrefs = []
    uni.get_annotation.return_value = ann
    bundle = EvidenceBundle(case=sample_case)
    rendered = dispatch_tool(
        "uniprot", sample_case, {"uniprot": uni}, bundle,
        {"uniprot": "get_annotation"},
    )
    uni.get_annotation.assert_called_once_with(
        "b0169", 83333, gene_symbol="rpsB"
    )
    assert bundle.uniprot is ann
    assert "30S ribosomal protein S2" in rendered


def test_dispatch_kegg_splits_gene_id(sample_case):
    kegg = MagicMock()
    entry = MagicMock()
    entry.gene_name = "rpsB"
    entry.definition = "30S ribosomal protein S2"
    entry.description = None
    entry.orthologies = []
    entry.orthology = None
    entry.pathways = []
    entry.motif_pfam = []
    entry.pfam = None
    kegg.fetch_gene.return_value = entry
    bundle = EvidenceBundle(case=sample_case)
    dispatch_tool(
        "kegg", sample_case, {"kegg": kegg}, bundle,
        {"kegg": "fetch_gene"},
    )
    kegg.fetch_gene.assert_called_once_with("eco", "b0169")
    assert bundle.kegg is entry


def test_dispatch_fba_requires_model(sample_case):
    fba = MagicMock()
    bundle = EvidenceBundle(case=sample_case)
    # no fba_model in tools
    out = dispatch_tool(
        "fba", sample_case, {"fba": fba}, bundle,
        {"fba": "compute_essentiality"},
    )
    fba.compute_essentiality.assert_not_called()
    assert "not available" in out


def test_dispatch_captures_exception(sample_case):
    fba = MagicMock()
    fba.compute_essentiality.side_effect = KeyError("gene not in model")
    bundle = EvidenceBundle(case=sample_case)
    out = dispatch_tool(
        "fba", sample_case,
        {"fba": fba, "fba_model": MagicMock()}, bundle,
        {"fba": "compute_essentiality"},
    )
    assert "fba" in bundle.errors
    assert "KeyError" in bundle.errors["fba"]
    assert "Query failed" in out


def test_dispatch_esm2_skips_when_no_sequence():
    case = Case(
        case_id="x", gene_symbol="g", locus_tag="b1",
        organism_taxon=1, uniprot_taxon=1, organism_strain="x",
        kegg_gene_id="eco:b1", string_species=1, metabolic_model="x",
        protein_sequence="", dna_sequence="",
        design_axis="x", ground_truth_essentiality="x",
        ground_truth_source="x", design_rationale="x",
    )
    esm2 = MagicMock()
    bundle = EvidenceBundle(case=case)
    out = dispatch_tool(
        "esm2", case, {"esm2": esm2}, bundle, {"esm2": "embed"},
    )
    esm2.embed.assert_not_called()
    assert "No protein sequence" in out


# ---------- the loop ----------


def test_run_loop_calls_tool_then_verdicts(sample_case):
    """Model calls one tool, then emits a verdict; loop terminates."""
    client = MagicMock()
    uni = MagicMock()
    ann = MagicMock()
    for attr in (
        "protein_name", "gene_name", "function_description",
    ):
        setattr(ann, attr, "x")
    for attr in (
        "ec_numbers", "ec", "go_terms", "pfam_domains", "pfam",
        "subcellular_locations", "subcellular_location",
        "kegg_xrefs", "eggnog_xrefs", "pdb_xrefs", "function",
    ):
        setattr(ann, attr, [] if "xref" in attr or "term" in attr
                or "domain" in attr or "location" in attr else None)
    uni.get_annotation.return_value = ann

    # Turn 1: call query_uniprot. Turn 2: emit verdict, no tool calls.
    client.messages.create.side_effect = [
        _response([_tool_use_block("query_uniprot", "tu_1")]),
        _response([_text_block(
            'Ribosomal protein.\n```json\n'
            '{"classification": "essential", "confidence": 0.95, '
            '"reasoning": "translation machinery"}\n```'
        )]),
    ]

    orch = EssentialityOrchestratorB(
        anthropic_client=client, tools={"uniprot": uni},
    )
    bundle, verdict, order = orch.run(sample_case)

    assert order == ["uniprot"]
    assert bundle.uniprot is ann
    assert verdict.classification == "essential"
    assert verdict.confidence == 0.95
    assert client.messages.create.call_count == 2


def test_run_loop_no_tools_immediate_verdict(sample_case):
    """Model emits a verdict on turn 1 without calling any tool."""
    client = MagicMock()
    client.messages.create.side_effect = [
        _response([_text_block(
            '```json\n{"classification": "uncertain", '
            '"confidence": 0.3, "reasoning": "no queries made"}\n```'
        )]),
    ]
    orch = EssentialityOrchestratorB(anthropic_client=client, tools={})
    bundle, verdict, order = orch.run(sample_case)
    assert order == []
    assert verdict.classification == "uncertain"
    assert client.messages.create.call_count == 1


def test_run_loop_respects_max_iterations(sample_case):
    """If the model keeps calling tools forever, the loop is bounded."""
    client = MagicMock()
    uni = MagicMock()
    uni.get_annotation.return_value = MagicMock(
        protein_name="x", gene_name="x", function_description="x",
        function=None, ec_numbers=[], ec=[], go_terms=[],
        pfam_domains=[], pfam=None, subcellular_locations=[],
        subcellular_location=None, kegg_xrefs=[], eggnog_xrefs=[],
        pdb_xrefs=[],
    )
    # Always returns a tool_use -> would loop forever without the cap
    client.messages.create.return_value = _response(
        [_tool_use_block("query_uniprot", "tu_loop")]
    )
    orch = EssentialityOrchestratorB(
        anthropic_client=client, tools={"uniprot": uni}, max_iterations=4,
    )
    bundle, verdict, order = orch.run(sample_case)
    assert client.messages.create.call_count == 4
    assert len(order) == 4  # one tool call per iteration
    # No verdict parseable -> falls back to uncertain
    assert verdict.classification == "uncertain"
