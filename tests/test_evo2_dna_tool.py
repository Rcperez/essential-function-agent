"""Tests for src/efa/tools/evo2_dna.py.

The arcinstitute/evo2 package requires a CUDA GPU and a multi-gigabyte
checkpoint, so unit tests inject a mock model to validate wrapper logic
without loading the real Evo2. One gpu-marked integration test exercises
the real model when both the evo2 package and a GPU are available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from efa.tools.evo2_dna import Evo2DNATool, Evo2Result


class FakeEvo2Tokenizer:
    """Mock byte-level Evo2 tokenizer; token IDs match byte values."""

    def tokenize(self, seq: str) -> list:
        return [ord(c) for c in seq]


class FakeEvo2Model:
    """Mock Evo2 model returning deterministic logits given inputs.

    Shape contract matches arcinstitute/evo2: __call__ returns
    (logits, _) where logits is shape (batch, length, vocab_size).
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.tokenizer = FakeEvo2Tokenizer()

    def __call__(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        torch.manual_seed(int(input_ids.sum().item()))
        logits = torch.randn(B, L, self.vocab_size)
        return logits, None


@pytest.fixture
def fake_tool(tmp_path: Path) -> Evo2DNATool:
    tool = Evo2DNATool(
        model_name="fake",
        cache_dir=tmp_path / "evo2_cache",
        hf_hub_cache=None,
        device="cpu",
    )
    tool._model = FakeEvo2Model()
    return tool


def test_score_sequence_returns_structured_result(
    fake_tool: Evo2DNATool,
) -> None:
    result = fake_tool.score_sequence("ACGTACGT")
    assert isinstance(result, Evo2Result)
    assert result.sequence_length == 8
    assert result.model_name == "fake"
    assert result.device == "cpu"
    assert result.sequence_preview.startswith("ACGT")


def test_uppercase_normalization(fake_tool: Evo2DNATool) -> None:
    upper = fake_tool.score_sequence("ACGTACGT")
    lower = fake_tool.score_sequence("acgtacgt")
    assert upper.sequence_sha256 == lower.sequence_sha256


def test_whitespace_normalization(fake_tool: Evo2DNATool) -> None:
    plain = fake_tool.score_sequence("ACGTACGT")
    padded = fake_tool.score_sequence("  ACGTACGT  ")
    assert plain.sequence_sha256 == padded.sequence_sha256


def test_empty_sequence_raises_valueerror(fake_tool: Evo2DNATool) -> None:
    with pytest.raises(ValueError, match="empty"):
        fake_tool.score_sequence("")
    with pytest.raises(ValueError, match="empty"):
        fake_tool.score_sequence("   ")


def test_invalid_character_raises_valueerror(fake_tool: Evo2DNATool) -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        fake_tool.score_sequence("ACGTXYZ")


def test_n_is_valid_dna_character(fake_tool: Evo2DNATool) -> None:
    result = fake_tool.score_sequence("ACGTNACGT")
    assert result.sequence_length == 9


def test_per_position_omitted_by_default(fake_tool: Evo2DNATool) -> None:
    result = fake_tool.score_sequence(
        "ACGTACGT", return_per_position=False,
    )
    assert result.per_position_log_likelihoods is None


def test_per_position_included_on_request(fake_tool: Evo2DNATool) -> None:
    result = fake_tool.score_sequence(
        "ACGTACGT", return_per_position=True,
    )
    assert result.per_position_log_likelihoods is not None
    assert len(result.per_position_log_likelihoods) == 7


def test_cache_returns_byte_identical_result(fake_tool: Evo2DNATool) -> None:
    seq = "ACGTACGTACGT"
    first = fake_tool.score_sequence(seq, return_per_position=True)
    second = fake_tool.score_sequence(seq, return_per_position=True)
    assert first == second


def test_cache_separates_per_position_modes(fake_tool: Evo2DNATool) -> None:
    """Cache distinguishes per-position vs mean-only modes."""
    seq = "ACGTACGTACGT"
    no_pp = fake_tool.score_sequence(seq, return_per_position=False)
    yes_pp = fake_tool.score_sequence(seq, return_per_position=True)
    assert no_pp.per_position_log_likelihoods is None
    assert yes_pp.per_position_log_likelihoods is not None
    assert no_pp.sequence_sha256 == yes_pp.sequence_sha256


def test_mean_log_likelihood_is_finite_negative_real(
    fake_tool: Evo2DNATool,
) -> None:
    """Log-probabilities are negative; mean should be finite and bounded."""
    result = fake_tool.score_sequence("ACGTACGTACGTACGT")
    assert isinstance(result.mean_log_likelihood, float)
    assert result.mean_log_likelihood < 0
    assert result.mean_log_likelihood > -100


def test_missing_evo2_package_raises_clear_importerror(
    tmp_path: Path,
) -> None:
    """Without a mocked model and without evo2 installed, gives clear error."""
    tool = Evo2DNATool(
        model_name="evo2_7b_base",
        cache_dir=tmp_path / "evo2_cache",
        hf_hub_cache=None,
        device="cpu",
    )
    saved = sys.modules.get("evo2")
    sys.modules["evo2"] = None  # forces ImportError on `import evo2`
    try:
        with pytest.raises(ImportError, match="evo2"):
            tool.score_sequence("ACGT")
    finally:
        if saved is None:
            sys.modules.pop("evo2", None)
        else:
            sys.modules["evo2"] = saved


@pytest.mark.gpu
def test_real_evo2_scores_short_dna_sequence(tmp_path: Path) -> None:
    """Live test against real Evo2 7B. Requires GPU and evo2 package."""
    pytest.importorskip("evo2")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tool = Evo2DNATool(cache_dir=tmp_path / "evo2_cache")
    # First 60 bp of E. coli rpsB CDS (b0169)
    seq = "ATGGCAACTGTTAGCATGCGTGACATGCTGAAAGCAGGAGTTCATTTCGGCCATCAGACA"
    result = tool.score_sequence(seq, return_per_position=True)
    assert result.sequence_length == 60
    assert result.mean_log_likelihood < 0
    assert result.per_position_log_likelihoods is not None
    assert len(result.per_position_log_likelihoods) == 59
