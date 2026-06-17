"""Tests for src/efa/tools/string_db.py.

Seven offline tests use a synthesized STRING interaction_partners
response shape; one live integration test fetches rpsB partners from
E. coli K-12 MG1655 (NCBI taxon 511145).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efa.tools.string_db import (
    STRINGInteraction,
    STRINGNetwork,
    STRINGRetriever,
)


# Two-partner sample mirroring the live STRING JSON response shape.
SAMPLE_STRING_RESPONSE = [
    {
        "stringId_A": "511145.b0169",
        "stringId_B": "511145.b0170",
        "ncbiTaxonId": 511145,
        "preferredName_A": "rpsB",
        "preferredName_B": "tsf",
        "score": 0.999,
        "nscore": 0.9,
        "fscore": 0.0,
        "pscore": 0.6,
        "ascore": 0.5,
        "escore": 0.8,
        "dscore": 0.9,
        "tscore": 0.7,
    },
    {
        "stringId_A": "511145.b0169",
        "stringId_B": "511145.b3340",
        "ncbiTaxonId": 511145,
        "preferredName_A": "rpsB",
        "preferredName_B": "tufA",
        "score": 0.876,
        "nscore": 0.3,
        "fscore": 0.0,
        "pscore": 0.4,
        "ascore": 0.6,
        "escore": 0.7,
        "dscore": 0.8,
        "tscore": 0.5,
    },
]


@pytest.fixture
def offline_retriever(tmp_path: Path) -> STRINGRetriever:
    return STRINGRetriever(
        cache_dir=tmp_path / "string_cache",
        rate_limit_s=0.0,
    )


def test_parse_partners_count(offline_retriever: STRINGRetriever) -> None:
    network = offline_retriever._parse_partners(
        SAMPLE_STRING_RESPONSE, "rpsB", 511145,
    )
    assert isinstance(network, STRINGNetwork)
    assert network.query_identifier == "rpsB"
    assert network.species_taxon == 511145
    assert len(network.interactions) == 2


def test_parse_partners_basic_fields(
    offline_retriever: STRINGRetriever,
) -> None:
    network = offline_retriever._parse_partners(
        SAMPLE_STRING_RESPONSE, "rpsB", 511145,
    )
    first = network.interactions[0]
    assert isinstance(first, STRINGInteraction)
    assert first.partner_string_id == "511145.b0170"
    assert first.partner_preferred_name == "tsf"


def test_score_conversion_float_to_int_1000(
    offline_retriever: STRINGRetriever,
) -> None:
    """STRING returns scores as floats 0-1; parser converts to int 0-1000."""
    network = offline_retriever._parse_partners(
        SAMPLE_STRING_RESPONSE, "rpsB", 511145,
    )
    first = network.interactions[0]
    assert first.combined_score == 999
    assert first.neighborhood_score == 900
    assert first.fusion_score == 0
    assert first.cooccurrence_score == 600
    assert first.coexpression_score == 500
    assert first.experimental_score == 800
    assert first.database_score == 900
    assert first.textmining_score == 700


def test_parse_empty_partners_list(
    offline_retriever: STRINGRetriever,
) -> None:
    """Empty list = STRING found the protein but no partners above threshold."""
    network = offline_retriever._parse_partners([], "rpsB", 511145)
    assert isinstance(network, STRINGNetwork)
    assert len(network.interactions) == 0


def test_to_int_score_handles_invalid(
    offline_retriever: STRINGRetriever,
) -> None:
    """_to_int_score returns 0 for None or non-numeric input."""
    assert STRINGRetriever._to_int_score(None) == 0
    assert STRINGRetriever._to_int_score("not a number") == 0
    assert STRINGRetriever._to_int_score(0.5) == 500
    assert STRINGRetriever._to_int_score(1.0) == 1000


def test_raw_string_url(offline_retriever: STRINGRetriever) -> None:
    network = offline_retriever._parse_partners(
        SAMPLE_STRING_RESPONSE, "rpsB", 511145,
    )
    assert "string-db.org" in network.raw_string_url
    assert "rpsB" in network.raw_string_url
    assert "511145" in network.raw_string_url


def test_fetch_returns_none_on_404(
    offline_retriever: STRINGRetriever,
) -> None:
    """Pre-seed cache with 'null' to exercise the not-found path."""
    cache_key = "partners__NOTFOUND__sp511145__lim20__rs400"
    cache_path = offline_retriever._cache_path(cache_key)
    assert cache_path is not None
    cache_path.write_text("null")
    result = offline_retriever.fetch_interaction_partners(
        "NOTFOUND", species_taxon=511145,
    )
    assert result is None


@pytest.mark.network
def test_live_fetch_ecoli_rpsB_partners(tmp_path: Path) -> None:
    """End-to-end live test: STRING partners for E. coli K-12 rpsB."""
    r = STRINGRetriever(cache_dir=tmp_path / "string_live")
    network = r.fetch_interaction_partners(
        "rpsB", species_taxon=511145, limit=20,
    )
    assert network is not None
    assert isinstance(network, STRINGNetwork)
    assert network.query_identifier == "rpsB"
    assert network.species_taxon == 511145
    assert len(network.interactions) > 0
    # At least one partner should be high-confidence (> 700)
    assert any(i.combined_score > 700 for i in network.interactions)
    # All returned partners should meet the default required_score (400)
    assert all(i.combined_score >= 400 for i in network.interactions)
    # Partners should have non-empty names
    assert all(i.partner_preferred_name for i in network.interactions)
