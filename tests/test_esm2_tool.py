"""Tests for src/efa/tools/esm2.py.

Uses esm2_t6_8M_UR50D (smallest ESM-2 variant) on CPU for fast test
execution without GPU memory contention.
"""

from __future__ import annotations

import numpy as np
import pytest

from efa.tools.esm2 import ESM2EmbeddingTool, ESM2Result


TEST_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
TEST_SEQUENCE = "MALWMRLLPLLALLALWGPDPAAA"
TEST_SEQUENCE_ID = "test_insulin_signal_24aa"


@pytest.fixture(scope="module")
def esm2_tool(tmp_path_factory):
    """Module-scoped: model is loaded once across tests in this module."""
    cache_dir = tmp_path_factory.mktemp("esm2_cache")
    return ESM2EmbeddingTool(
        model_id=TEST_MODEL_ID,
        device="cpu",
        cache_dir=cache_dir,
    )


def test_initialization_does_not_load_model() -> None:
    tool = ESM2EmbeddingTool(model_id=TEST_MODEL_ID, cache_dir=None)
    assert tool._model is None
    assert tool._tokenizer is None


def test_embed_returns_expected_shape(esm2_tool: ESM2EmbeddingTool) -> None:
    result = esm2_tool.embed(TEST_SEQUENCE, TEST_SEQUENCE_ID + "_shape")
    assert isinstance(result, ESM2Result)
    assert result.sequence_length_aa == len(TEST_SEQUENCE)
    assert result.mean_embedding.ndim == 1
    assert result.mean_embedding.shape[0] > 0
    assert result.per_residue_log_likelihood.shape == (len(TEST_SEQUENCE),)
    assert result.model_id == TEST_MODEL_ID


def test_log_likelihoods_are_non_positive(
    esm2_tool: ESM2EmbeddingTool,
) -> None:
    result = esm2_tool.embed(TEST_SEQUENCE, TEST_SEQUENCE_ID + "_logprob")
    assert (result.per_residue_log_likelihood <= 0).all()
    assert result.mean_log_likelihood <= 0


def test_variant_effect_is_non_negative(
    esm2_tool: ESM2EmbeddingTool,
) -> None:
    result = esm2_tool.embed(TEST_SEQUENCE, TEST_SEQUENCE_ID + "_ve")
    assert result.mean_variant_effect >= 0


def test_cache_returns_byte_identical_result(
    esm2_tool: ESM2EmbeddingTool,
) -> None:
    result1 = esm2_tool.embed(TEST_SEQUENCE, TEST_SEQUENCE_ID + "_cache")
    result2 = esm2_tool.embed(TEST_SEQUENCE, TEST_SEQUENCE_ID + "_cache")
    np.testing.assert_array_equal(
        result1.mean_embedding, result2.mean_embedding
    )
    np.testing.assert_array_equal(
        result1.per_residue_log_likelihood,
        result2.per_residue_log_likelihood,
    )
    assert result1.mean_log_likelihood == result2.mean_log_likelihood
    assert result1.mean_variant_effect == result2.mean_variant_effect
