"""Evo2 DNA-likelihood channel (Arc Institute genomic foundation model).

Wraps the arcinstitute/evo2 Python package to compute per-token causal
log-likelihoods for DNA sequences using the Evo2 7B base model. The agent
consumes the mean log-likelihood as a sequence-level constraint signal and
optionally per-position log-likelihoods to identify regions under strong
selective pressure within a gene's coding sequence and flanking regions.

This is the public-prototype Evo2 wrapper: no GC-content calibration, no
organism-specific normalization. Any calibration logic lives in the private
FI repository and is not appropriate for the public lane.

The `evo2` package is an optional runtime dependency. The wrapper raises a
clear ImportError on first inference call if the package is not installed;
the user can install it per https://github.com/arcinstitute/evo2.

Evo2 reference: Brixi, Durrant et al. 2025 (Arc Institute).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch


DEFAULT_MODEL_NAME = "evo2_7b_base"
DEFAULT_CACHE_DIR = Path(
    "/content/drive/MyDrive/RP_RTP_Repo_Bundles/"
    "essential-function-agent/cache/evo2"
)
DEFAULT_HF_HUB_CACHE = Path(
    "/content/drive/MyDrive/RP_RTP_Repo_Bundles/fungal_evo2/hf_cache"
)
VALID_DNA_CHARS = frozenset("ACGTN")


@dataclass
class Evo2Result:
    """Result of Evo2 likelihood scoring on a DNA sequence."""

    sequence_sha256: str
    sequence_length: int
    sequence_preview: str
    mean_log_likelihood: float
    per_position_log_likelihoods: Optional[List[float]]
    model_name: str
    device: str


class Evo2DNATool:
    """Evo2 byte-level DNA-likelihood scoring wrapper.

    Lazy-loads the Evo2 model on first scoring call. Caches scoring results
    on disk keyed by SHA256(sequence) and per-position-mode flag. Supports
    a custom HuggingFace hub cache location set lazily on first call.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
        hf_hub_cache: Optional[Path] = DEFAULT_HF_HUB_CACHE,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_hub_cache = (
            Path(hf_hub_cache) if hf_hub_cache is not None else None
        )
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None  # populated lazily by _ensure_model_loaded()

    def score_sequence(
        self,
        dna_sequence: str,
        return_per_position: bool = False,
    ) -> Evo2Result:
        """Compute Evo2 log-likelihood for a DNA sequence.

        Returns the mean causal log-likelihood (averaged over tokens) and
        optionally the per-position log-likelihoods. Raises ValueError on
        empty or non-ACGTN sequences. Raises ImportError on first call if
        the evo2 package is not installed.
        """
        sequence = self._normalize_sequence(dna_sequence)
        sha = hashlib.sha256(sequence.encode()).hexdigest()

        cache_path = self._cache_path(sha, return_per_position)
        if cache_path is not None and cache_path.is_file():
            return Evo2Result(**json.loads(cache_path.read_text()))

        self._ensure_model_loaded()
        mean_ll, per_pos = self._compute_log_likelihoods(sequence)

        result = Evo2Result(
            sequence_sha256=sha,
            sequence_length=len(sequence),
            sequence_preview=(
                sequence[:80] + ("..." if len(sequence) > 80 else "")
            ),
            mean_log_likelihood=float(mean_ll),
            per_position_log_likelihoods=(
                per_pos if return_per_position else None
            ),
            model_name=self.model_name,
            device=self.device,
        )

        if cache_path is not None:
            cache_path.write_text(json.dumps(asdict(result)))
        return result

    def _normalize_sequence(self, dna_sequence: str) -> str:
        s = dna_sequence.upper().strip()
        if not s:
            raise ValueError("dna_sequence is empty after stripping whitespace")
        invalid = set(s) - VALID_DNA_CHARS
        if invalid:
            raise ValueError(
                f"dna_sequence contains invalid characters: {sorted(invalid)}; "
                f"only A, C, G, T, N permitted"
            )
        return s

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if (
            self.hf_hub_cache is not None
            and not os.environ.get("HF_HUB_CACHE")
        ):
            os.environ["HF_HUB_CACHE"] = str(self.hf_hub_cache)
        try:
            from evo2 import Evo2
        except ImportError as exc:
            raise ImportError(
                "The 'evo2' package is required to run Evo2 inference but is "
                "not installed. Install per "
                "https://github.com/arcinstitute/evo2"
            ) from exc
        self._model = Evo2(self.model_name)

    def _compute_log_likelihoods(
        self, sequence: str,
    ) -> Tuple[float, List[float]]:
        """Forward pass + per-token causal log-likelihood computation.

        Returns (mean_log_likelihood, per_position_log_likelihoods). The
        per-position list has length L-1 for an L-token input, because
        position i predicts token i+1 under causal masking.
        """
        assert self._model is not None
        tokens = self._model.tokenizer.tokenize(sequence)
        input_ids = torch.tensor(
            [tokens], dtype=torch.long, device=self.device,
        )
        with torch.no_grad():
            # Evo2 0.5.5 returns ((logits, None), None) from model(...);
            # double-unpack to extract the logits tensor.
            (logits, _), _ = self._model(input_ids)
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        per_pos = log_probs.gather(-1, targets).squeeze(-1).squeeze(0)
        per_pos_list = per_pos.cpu().tolist()
        mean_ll = float(per_pos.mean().item())
        return mean_ll, per_pos_list

    def _cache_path(
        self, sha: str, return_per_position: bool,
    ) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        suffix = "perpos" if return_per_position else "mean"
        return self.cache_dir / f"{sha}_{suffix}.json"


__all__ = ["Evo2DNATool", "Evo2Result"]
