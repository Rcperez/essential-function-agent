"""Tests for src/efa/tools/uniprot.py.

Six offline parsing tests use a fixed sample JSON; one live integration
test fetches E. coli K-12 RpsB (taxon 83333) as a real-data sanity check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from efa.tools.uniprot import GOTerm, UniProtAnnotation, UniProtRetriever


SAMPLE_ENTRY = {
    "primaryAccession": "P12345",
    "proteinDescription": {
        "recommendedName": {
            "fullName": {"value": "Test enzyme"},
            "ecNumbers": [{"value": "1.2.3.4"}, {"value": "2.3.4.5"}],
        }
    },
    "genes": [
        {
            "geneName": {"value": "testA"},
            "orderedLocusNames": [{"value": "TEST_0001"}],
        }
    ],
    "organism": {
        "scientificName": "Imaginary organism",
        "taxonId": 99999,
    },
    "sequence": {
        "value": "MAGICDNAIS",
        "length": 10,
    },
    "comments": [
        {
            "commentType": "FUNCTION",
            "texts": [{"value": "Catalyzes the imaginary reaction."}],
        },
        {
            "commentType": "SUBCELLULAR LOCATION",
            "subcellularLocations": [
                {"location": {"value": "Cytoplasm"}},
            ],
        },
    ],
    "uniProtKBCrossReferences": [
        {"database": "Pfam", "id": "PF00001"},
        {"database": "InterPro", "id": "IPR000001"},
        {
            "database": "GO",
            "id": "GO:0003824",
            "properties": [
                {"key": "GoTerm", "value": "F:catalytic activity"},
                {"key": "GoEvidenceType", "value": "IEA"},
            ],
        },
        {"database": "KEGG", "id": "imag:TEST_0001"},
        {"database": "eggNOG", "id": "COG1234"},
        {"database": "PDB", "id": "1ABC"},
    ],
}


@pytest.fixture
def offline_retriever(tmp_path: Path) -> UniProtRetriever:
    """Retriever with a temp cache; no network calls expected in offline tests."""
    return UniProtRetriever(
        cache_dir=tmp_path / "uniprot_cache",
        rate_limit_s=0.0,
    )


def test_parse_extracts_basic_fields(
    offline_retriever: UniProtRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_ENTRY)
    assert a.accession == "P12345"
    assert a.protein_name == "Test enzyme"
    assert a.gene_name == "testA"
    assert a.locus_tag == "TEST_0001"
    assert a.organism_taxon == 99999
    assert a.sequence == "MAGICDNAIS"
    assert a.sequence_length_aa == 10
    assert a.raw_uniprot_url == "https://www.uniprot.org/uniprotkb/P12345"


def test_parse_extracts_function_description(
    offline_retriever: UniProtRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_ENTRY)
    assert "imaginary reaction" in a.function_description


def test_parse_extracts_ec_numbers(
    offline_retriever: UniProtRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_ENTRY)
    assert a.ec_numbers == ["1.2.3.4", "2.3.4.5"]


def test_parse_extracts_xrefs(
    offline_retriever: UniProtRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_ENTRY)
    assert a.pfam_domains == ["PF00001"]
    assert a.interpro_ids == ["IPR000001"]
    assert a.kegg_xrefs == ["imag:TEST_0001"]
    assert a.eggnog_xrefs == ["COG1234"]
    assert a.pdb_xrefs == ["1ABC"]
    assert len(a.go_terms) == 1
    go = a.go_terms[0]
    assert isinstance(go, GOTerm)
    assert go.go_id == "GO:0003824"
    assert go.aspect == "F"
    assert go.name == "catalytic activity"


def test_parse_extracts_subcellular_location(
    offline_retriever: UniProtRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_ENTRY)
    assert a.subcellular_locations == ["Cytoplasm"]


def test_search_returns_none_on_empty_results(
    offline_retriever: UniProtRetriever,
) -> None:
    """Pre-seed cache with empty results to exercise the None branch."""
    cache_key = "search__locus_NONEXISTENT__taxon_99999"
    cache_path = offline_retriever._cache_path(cache_key)
    assert cache_path is not None
    cache_path.write_text(json.dumps({"results": []}))
    result = offline_retriever.search_by_locus_tag(
        "NONEXISTENT", organism_taxon=99999
    )
    assert result is None


@pytest.mark.network
def test_live_fetch_ecoli_rpsB(tmp_path: Path) -> None:
    """End-to-end live test: E. coli K-12 RpsB (30S ribosomal protein S2)."""
    r = UniProtRetriever(cache_dir=tmp_path / "uniprot_live")
    annotation = r.get_annotation("rpsB", organism_taxon=83333)
    assert annotation is not None
    assert annotation.accession.startswith("P")
    assert annotation.organism_taxon == 83333
    assert "ribosom" in annotation.protein_name.lower()
    assert annotation.sequence_length_aa > 200
    assert len(annotation.pfam_domains) > 0
    assert any(go.aspect == "F" for go in annotation.go_terms)
