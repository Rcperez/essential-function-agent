"""Tests for src/efa/tools/kegg.py.

Offline parsing tests use a faithful sample of the modern KEGG format
captured from eco:b0169 (rpsB) on 2026-06-17; one live integration test
fetches the same entry from the public KEGG REST endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efa.tools.kegg import (
    KEGGGeneAnnotation,
    KEGGOrthology,
    KEGGRetriever,
)


SAMPLE_KEGG_ENTRY = """ENTRY       b0169             CDS       T00007
SYMBOL      rpsB
NAME        (RefSeq) 30S ribosomal subunit protein S2
ORTHOLOGY   K02967  small subunit ribosomal protein S2
ORGANISM    eco  Escherichia coli K-12 MG1655
PATHWAY     eco03010  Ribosome
BRITE       KEGG Orthology (KO) [BR:eco00001]
             09120 Genetic Information Processing
              09122 Translation
               03010 Ribosome
                b0169 (rpsB)
             09180 Brite Hierarchies
              09182 Protein families: genetic information processing
               03011 Ribosome [BR:eco03011]
                b0169 (rpsB)
            Ribosome [BR:eco03011]
             Ribosomal proteins
              Bacteria
                b0169 (rpsB)
POSITION    189874..190599
MOTIF       Pfam: Ribosomal_S2 GIP4
DBLINKS     NCBI-GeneID: 947874
            NCBI-ProteinID: NP_414711
            Pasteur: rpsB
            RegulonDB: RDBECOLIGNC00891
            ECOCYC: EG10901
            UniProt: P0A7V0
STRUCTURE   PDB
AASEQ       241
            MATVSMRDMLKAGVHFGHQTRYWNPKMKPFIFGARNKVHIINLEKTVPMFNEALAELNKI
            ASRKGKILFVGTKRAASEAVKDAALSCDQFFVNHRWLGGMLTNWKTVRQSIKRLKDLETQ
            SQDGTFDKLTKKEALMRTRELEKLENSLGGIKDMGGLPDALFVIDADHEHIAIKEANNLG
            IPVFAIVDTNSDPDGVDFVIPGNDDAIRAVTLYLGAVAATVREGRSQDLASQAEESFVEA
            E
NTSEQ       726
            atggcaactgtttccatgcgcgacatgctcaaggctggtgttcacttcggtcaccagacc
///
"""

MULTI_PATHWAY_ENTRY = """ENTRY       fake01            CDS       T00007
SYMBOL      fakeA
NAME        (RefSeq) synthetic test enzyme
PATHWAY     fakepath01  First pathway
            fakepath02  Second pathway
            fakepath03  Third pathway
///
"""


@pytest.fixture
def offline_retriever(tmp_path: Path) -> KEGGRetriever:
    return KEGGRetriever(
        cache_dir=tmp_path / "kegg_cache",
        rate_limit_s=0.0,
    )


def test_parse_flat_text_finds_all_sections(
    offline_retriever: KEGGRetriever,
) -> None:
    sections = offline_retriever._parse_flat_text(SAMPLE_KEGG_ENTRY)
    for required in [
        "ENTRY", "SYMBOL", "NAME", "ORTHOLOGY", "ORGANISM",
        "PATHWAY", "BRITE", "MOTIF", "DBLINKS", "AASEQ",
    ]:
        assert required in sections, f"missing section: {required}"


def test_parse_entry_basic_fields(
    offline_retriever: KEGGRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    assert isinstance(a, KEGGGeneAnnotation)
    assert a.kegg_gene_id == "eco:b0169"
    assert a.organism_code == "eco"
    assert a.locus_tag == "b0169"
    assert a.gene_name == "rpsB"
    assert "30S ribosomal subunit protein S2" in a.definition
    assert not a.definition.startswith("(RefSeq)")
    assert a.raw_kegg_url == "https://www.kegg.jp/entry/eco:b0169"


def test_legacy_format_definition_field_used_when_present(
    offline_retriever: KEGGRetriever,
) -> None:
    """Legacy KEGG entries with DEFINITION and NAME-as-symbol parse correctly."""
    legacy = """ENTRY       legacy01          CDS       T00007
NAME        legacyA
DEFINITION  (RefSeq) legacy test enzyme
///
"""
    a = offline_retriever._parse_entry(legacy, "fake", "legacy01")
    assert a.gene_name == "legacyA"
    assert a.definition == "legacy test enzyme"


def test_parse_orthology(offline_retriever: KEGGRetriever) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    assert len(a.orthologies) == 1
    o = a.orthologies[0]
    assert isinstance(o, KEGGOrthology)
    assert o.ko_id == "K02967"
    assert "small subunit ribosomal protein S2" in o.description


def test_parse_single_pathway(offline_retriever: KEGGRetriever) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    assert len(a.pathways) == 1
    assert a.pathways[0].pathway_id == "eco03010"
    assert a.pathways[0].name == "Ribosome"


def test_pathway_continuation_lines_all_captured(
    offline_retriever: KEGGRetriever,
) -> None:
    """PATHWAY with multiple continuation lines yields one KEGGPathway each."""
    a = offline_retriever._parse_entry(MULTI_PATHWAY_ENTRY, "fake", "fake01")
    assert len(a.pathways) == 3
    ids = {p.pathway_id for p in a.pathways}
    assert ids == {"fakepath01", "fakepath02", "fakepath03"}


def test_brite_does_not_leak_into_motif(
    offline_retriever: KEGGRetriever,
) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    assert "Ribosomal_S2" in a.motif_pfam
    assert "GIP4" in a.motif_pfam
    assert len(a.motif_pfam) == 2


def test_parse_aaseq(offline_retriever: KEGGRetriever) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    assert a.aa_length == 241
    assert a.aa_sequence.startswith("MATVSMRDMLKAG")
    assert " " not in a.aa_sequence


def test_parse_ntseq(offline_retriever: KEGGRetriever) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    # Header length reflects the real gene (726 bp); fixture truncates
    # the sequence body to 60 bp for compactness. Parser uppercases.
    assert a.nt_length == 726
    assert a.nt_sequence.startswith("ATGGCAACTGTTTCC")
    assert " " not in a.nt_sequence
    assert a.nt_sequence == a.nt_sequence.upper()


def test_parse_dblinks(offline_retriever: KEGGRetriever) -> None:
    a = offline_retriever._parse_entry(SAMPLE_KEGG_ENTRY, "eco", "b0169")
    assert "UniProt" in a.db_links
    assert "P0A7V0" in a.db_links["UniProt"]
    assert "NCBI-GeneID" in a.db_links
    assert "947874" in a.db_links["NCBI-GeneID"]


@pytest.mark.network
def test_live_fetch_ecoli_rpsB(tmp_path: Path) -> None:
    """End-to-end live test: KEGG entry for eco:b0169 (rpsB)."""
    r = KEGGRetriever(cache_dir=tmp_path / "kegg_live")
    annotation = r.fetch_gene("eco", "b0169")
    assert annotation is not None
    assert annotation.kegg_gene_id == "eco:b0169"
    assert annotation.gene_name == "rpsB"
    assert "ribosom" in annotation.definition.lower()
    assert annotation.aa_length > 200
    assert any("ribosom" in p.name.lower() for p in annotation.pathways)
    assert len(annotation.orthologies) > 0
    assert "Ribosomal_S2" in annotation.motif_pfam
