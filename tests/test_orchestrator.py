"""Tests for src/efa/orchestrator.py.

Uses mocked tools and a mocked Anthropic client; no network, no LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from efa.orchestrator import (
    Case, EvidenceBundle, Verdict, EssentialityOrchestrator,
    gather_all_evidence, render_evidence_bundle, parse_verdict, load_cases,
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


# ---------- parse_verdict ----------


def test_parse_verdict_fenced_json():
    response = (
        "Reasoning text.\n\n"
        "```json\n"
        '{"classification": "essential", "confidence": 0.95, "reasoning": "ribosomal"}\n'
        "```"
    )
    v = parse_verdict(response, "case1")
    assert v.classification == "essential"
    assert v.confidence == 0.95
    assert v.reasoning == "ribosomal"
    assert v.case_id == "case1"


def test_parse_verdict_inline_json():
    response = (
        'Final: {"classification": "non_essential", "confidence": 0.8, "reasoning": "PPP bypass"}'
    )
    v = parse_verdict(response, "case2")
    assert v.classification == "non_essential"
    assert v.confidence == 0.8


def test_parse_verdict_falls_back_to_uncertain():
    v = parse_verdict("I cannot determine essentiality.", "case3")
    assert v.classification == "uncertain"
    assert v.confidence == 0.0
    assert "Could not parse" in v.reasoning


def test_parse_verdict_preserves_raw_response():
    response = (
        "preamble\n```json\n"
        '{"classification": "essential", "confidence": 0.9, "reasoning": "r"}\n'
        "```\npostamble"
    )
    v = parse_verdict(response, "case4")
    assert "preamble" in v.raw_response
    assert "postamble" in v.raw_response


def test_parse_verdict_handles_conditional():
    response = (
        '```json\n{"classification": "conditional", "confidence": 0.75, "reasoning": "stress-only"}\n```'
    )
    v = parse_verdict(response, "case5")
    assert v.classification == "conditional"


# ---------- render_evidence_bundle ----------


def test_render_empty_bundle_has_case_header(sample_case):
    bundle = EvidenceBundle(case=sample_case)
    rendered = render_evidence_bundle(bundle)
    assert "rpsB" in rendered
    assert "b0169" in rendered
    assert "511145" in rendered
    assert rendered.count("_(not queried)_") == 6


def test_render_bundle_with_uniprot(sample_case):
    uniprot = MagicMock()
    uniprot.protein_name = "30S ribosomal protein S2"
    uniprot.gene_name = "rpsB"
    uniprot.ec_numbers = []
    uniprot.ec = []
    uniprot.function_description = "Component of the 30S ribosomal subunit."
    uniprot.function = None
    uniprot.go_terms = [
        {"id": "GO:0003735", "term": "structural constituent of ribosome"}
    ]
    uniprot.pfam_domains = ["PF00318"]
    uniprot.pfam = None
    uniprot.subcellular_locations = ["Cytoplasm"]
    uniprot.subcellular_location = None
    uniprot.kegg_xrefs = ["eco:b0169"]
    uniprot.eggnog_xrefs = []
    uniprot.pdb_xrefs = []
    bundle = EvidenceBundle(case=sample_case, uniprot=uniprot)
    rendered = render_evidence_bundle(bundle)
    assert "30S ribosomal protein S2" in rendered
    assert "GO:0003735" in rendered
    assert "PF00318" in rendered
    assert "Cytoplasm" in rendered


def test_render_bundle_with_error(sample_case):
    bundle = EvidenceBundle(
        case=sample_case, errors={"uniprot": "ConnectionError: timeout"}
    )
    rendered = render_evidence_bundle(bundle)
    assert "ConnectionError" in rendered
    assert "Query failed" in rendered


def test_render_bundle_with_fba(sample_case):
    fba = MagicMock()
    fba.model_id = "iML1515"
    fba.objective_id = "BIOMASS_Ec_iML1515_core_75p37M"
    fba.wildtype_growth_rate = 0.877
    fba.knockout_growth_rate = 0.0
    fba.growth_ratio = 0.0
    fba.is_essential = True
    fba.threshold_ratio = 0.01
    fba.status = "optimal"
    bundle = EvidenceBundle(case=sample_case, fba=fba)
    rendered = render_evidence_bundle(bundle)
    assert "iML1515" in rendered
    assert "ESSENTIAL" in rendered
    assert "0.877" in rendered
    assert "optimal" in rendered


# ---------- gather_all_evidence ----------


def test_gather_with_no_tools_returns_empty_bundle(sample_case):
    bundle = gather_all_evidence(sample_case, tools={})
    assert bundle.case is sample_case
    assert bundle.uniprot is None
    assert bundle.errors == {}


def test_gather_captures_per_tool_exceptions(sample_case):
    failing = MagicMock()
    failing.get_annotation.side_effect = ConnectionError("timeout after 30s")
    bundle = gather_all_evidence(sample_case, tools={"uniprot": failing})
    assert bundle.uniprot is None
    assert "ConnectionError" in bundle.errors["uniprot"]


def test_gather_one_failure_does_not_block_others(sample_case):
    failing = MagicMock()
    failing.get_annotation.side_effect = RuntimeError("bad")
    succeeding = MagicMock()
    succeeding.fetch_interaction_partners.return_value = "partners"
    bundle = gather_all_evidence(
        sample_case,
        tools={"uniprot": failing, "string": succeeding},
    )
    assert "uniprot" in bundle.errors
    assert bundle.string == "partners"


def test_gather_calls_tools_with_expected_args(sample_case):
    u, k, s, e, v = (MagicMock() for _ in range(5))
    gather_all_evidence(sample_case, tools={
        "uniprot": u, "kegg": k, "string": s, "esm2": e, "evo2": v,
    })
    u.get_annotation.assert_called_once_with(
        "b0169", 83333, gene_symbol="rpsB"
    )
    k.fetch_gene.assert_called_once_with("eco", "b0169")
    s.fetch_interaction_partners.assert_called_once_with("rpsB", 511145)
    e.embed.assert_called_once_with("MATVSMR", "rpsB")
    v.score_sequence.assert_called_once_with("ATGGCAACTGT")


def test_gather_skips_esm2_if_protein_sequence_empty():
    case = Case(
        case_id="x", gene_symbol="g", locus_tag="b0001",
        organism_taxon=1, uniprot_taxon=1, organism_strain="x", kegg_gene_id="x",
        string_species=1, metabolic_model="x",
        protein_sequence="", dna_sequence="",
        design_axis="x", ground_truth_essentiality="x",
        ground_truth_source="x", design_rationale="x",
    )
    esm2 = MagicMock()
    bundle = gather_all_evidence(case, tools={"esm2": esm2})
    esm2.embed.assert_not_called()
    assert bundle.esm2 is None


def test_gather_fba_requires_both_tool_and_model(sample_case):
    fba_tool = MagicMock()
    bundle = gather_all_evidence(sample_case, tools={"fba": fba_tool})
    fba_tool.compute_essentiality.assert_not_called()
    assert bundle.fba is None


def test_gather_fba_with_both_works(sample_case):
    fba_tool = MagicMock()
    fba_model = MagicMock()
    fba_tool.compute_essentiality.return_value = "fba_result"
    bundle = gather_all_evidence(
        sample_case, tools={"fba": fba_tool, "fba_model": fba_model},
    )
    fba_tool.compute_essentiality.assert_called_once_with(fba_model, "b0169")
    assert bundle.fba == "fba_result"


def test_gather_tool_methods_override(sample_case):
    custom = MagicMock()
    custom.fetch_gene_entry.return_value = "kegg_result"
    bundle = gather_all_evidence(
        sample_case,
        tools={"kegg": custom},
        tool_methods={"kegg": "fetch_gene_entry"},
    )
    custom.fetch_gene_entry.assert_called_once_with("eco", "b0169")
    assert bundle.kegg == "kegg_result"


# ---------- EssentialityOrchestrator ----------


def test_orchestrator_constructs_with_explicit_client():
    client = MagicMock()
    orch = EssentialityOrchestrator(anthropic_client=client)
    assert orch._anthropic_client is client


def test_orchestrator_run_end_to_end_mocked(sample_case):
    client = MagicMock()
    block = MagicMock()
    block.text = (
        "Reasoning: ribosomal essentiality.\n\n"
        "```json\n"
        '{"classification": "essential", "confidence": 0.95, "reasoning": "translation"}\n'
        "```"
    )
    response = MagicMock()
    response.content = [block]
    client.messages.create.return_value = response
    orch = EssentialityOrchestrator(anthropic_client=client, tools={})
    bundle, verdict = orch.run(sample_case)
    assert isinstance(bundle, EvidenceBundle)
    assert isinstance(verdict, Verdict)
    assert verdict.classification == "essential"
    assert verdict.confidence == 0.95
    assert verdict.case_id == "test_rpsB"
    call = client.messages.create.call_args.kwargs
    assert call["model"] == "claude-sonnet-4-6"
    assert "rpsB" in call["messages"][0]["content"]


# ---------- load_cases ----------


def test_load_cases_from_synthetic_json(tmp_path: Path):
    data = {"schema_version": "0.1.0", "cases": [{
        "case_id": "t1", "gene_symbol": "g", "locus_tag": "b1",
        "organism_taxon": 511145, "uniprot_taxon": 83333, "organism_strain": "E. coli",
        "kegg_gene_id": "eco:b1", "string_species": 511145,
        "metabolic_model": "iML1515",
        "protein_sequence": "", "dna_sequence": "",
        "design_axis": "test", "ground_truth_essentiality": "essential",
        "ground_truth_source": "test", "design_rationale": "test",
    }]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].case_id == "t1"


def test_load_cases_real_repo_file():
    """Verify data/cases.json in this repo: 5 cases, 5 distinct axes."""
    repo_root = Path(__file__).parent.parent
    p = repo_root / "data" / "cases.json"
    if not p.is_file():
        pytest.skip("data/cases.json not found in repo")
    cases = load_cases(p)
    assert len(cases) == 5
    axes = {c.design_axis for c in cases}
    assert len(axes) == 5
