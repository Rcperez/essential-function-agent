"""ESM-2 protein language model wrapper for the agent's protein channel.

Provides two readouts per protein:

1. Mean-pooled per-protein embedding (for kNN similarity searches against
   labeled essential / non-essential reference sets).

2. Zero-shot variant-effect score via per-residue log-likelihoods, following
   the Frazer 2021 / Brandes 2023 formulation: at each position, the
   log-probability of the wildtype residue is compared against the maximum
   log-probability at that position. The per-protein score is the mean
   across positions.

ESM-2 reference: Lin, Akin, Rao, Hie, Zhu, Lu, Smetanin, dos Santos Costa,
Fazel-Zarandi, Sercu, Candido, and Rives 2023 (Science). The variant-effect
formulation in the protein-LM literature is from Frazer et al. 2021 (Nature,
EVE) and Brandes et al. 2023 (Nature Genetics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


DEFAULT_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_CACHE_DIR = Path(
    "/content/drive/MyDrive/RP_RTP_Repo_Bundles/"
    "essential-function-agent/cache/esm2"
)


@dataclass
class ESM2Result:
    """Result of running ESM-2 on a single protein sequence."""

    sequence_id: str
    sequence_length_aa: int
    model_id: str
    mean_embedding: np.ndarray
    per_residue_log_likelihood: np.ndarray
    mean_log_likelihood: float
    mean_variant_effect: float


class ESM2EmbeddingTool:
    """ESM-2 wrapper for embedding extraction and variant-effect scoring.

    Lazy-loads the model on first call. Caches per-sequence results as
    .npz files (one per (model_id, sequence_id) pair) to avoid
    recomputation across runs.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir is not None else None
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer on first call."""
        if self._model is not None:
            return
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_id).to(
            self.device
        )
        self._model.eval()

    def _cache_path(self, sequence_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe_model = self.model_id.replace("/", "_")
        return self.cache_dir / f"{safe_model}__{sequence_id}.npz"

    def _load_from_cache(self, sequence_id: str) -> Optional[ESM2Result]:
        cache_path = self._cache_path(sequence_id)
        if cache_path is None or not cache_path.is_file():
            return None
        cached = np.load(cache_path, allow_pickle=False)
        return ESM2Result(
            sequence_id=sequence_id,
            sequence_length_aa=int(cached["sequence_length_aa"]),
            model_id=str(cached["model_id"]),
            mean_embedding=cached["mean_embedding"],
            per_residue_log_likelihood=cached["per_residue_log_likelihood"],
            mean_log_likelihood=float(cached["mean_log_likelihood"]),
            mean_variant_effect=float(cached["mean_variant_effect"]),
        )

    def _save_to_cache(self, result: ESM2Result) -> None:
        cache_path = self._cache_path(result.sequence_id)
        if cache_path is None:
            return
        np.savez(
            cache_path,
            sequence_length_aa=result.sequence_length_aa,
            model_id=result.model_id,
            mean_embedding=result.mean_embedding,
            per_residue_log_likelihood=result.per_residue_log_likelihood,
            mean_log_likelihood=result.mean_log_likelihood,
            mean_variant_effect=result.mean_variant_effect,
        )

    def embed(self, sequence: str, sequence_id: str) -> ESM2Result:
        """Run ESM-2 on a single protein sequence.

        Returns the cached result if available; otherwise loads the model
        and computes the mean embedding plus per-residue log-likelihoods.
        """
        cached = self._load_from_cache(sequence_id)
        if cached is not None:
            return cached

        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        inputs = self._tokenizer(sequence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # ESM-2 wraps the sequence with <bos> and <eos>; trim them
        logits = outputs.logits[0, 1:-1]
        hidden = outputs.hidden_states[-1][0, 1:-1]
        input_ids = inputs["input_ids"][0, 1:-1]

        mean_embedding = hidden.mean(dim=0).cpu().numpy()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        per_residue_ll = (
            log_probs.gather(1, input_ids.unsqueeze(-1))
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        max_log_probs = log_probs.max(dim=-1).values.cpu().numpy()
        per_position_ve = max_log_probs - per_residue_ll

        result = ESM2Result(
            sequence_id=sequence_id,
            sequence_length_aa=len(sequence),
            model_id=self.model_id,
            mean_embedding=mean_embedding,
            per_residue_log_likelihood=per_residue_ll,
            mean_log_likelihood=float(per_residue_ll.mean()),
            mean_variant_effect=float(per_position_ve.mean()),
        )

        self._save_to_cache(result)
        return result


__all__ = ["ESM2EmbeddingTool", "ESM2Result", "DEFAULT_MODEL_ID"]
