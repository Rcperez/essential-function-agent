"""B-architecture essentiality orchestrator (Anthropic tool-use loop).

Where the A-architecture (orchestrator.py) eagerly calls all six channels
and makes a single LLM call over the full evidence bundle, the
B-architecture exposes the six channels as tools and lets the model decide
which to call, in what order, and when it has enough evidence to rule.

Everything that makes A and B comparable is imported verbatim from
orchestrator.py: the Case / EvidenceBundle / Verdict dataclasses, the
SYSTEM_PROMPT (reasoning principles), render_evidence_bundle (per-channel
rendering), parse_verdict (JSON extraction), and load_cases. The ONLY
difference between the two architectures is the dispatch strategy in this
file: tool schemas, the call loop, and a dispatcher that marshals arguments
from the Case using the same call patterns A uses.

This keeps the experiment controlled: same evidence channels, same per-
channel rendering, same prompt principles, same verdict parsing. The single
varied factor is eager-dispatch (A) vs model-driven tool selection (B).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from efa.orchestrator import (
    Case,
    EvidenceBundle,
    Verdict,
    SYSTEM_PROMPT,
    parse_verdict,
    load_cases,
    _render_uniprot,
    _render_kegg,
    _render_string,
    _render_esm2,
    _render_evo2,
    _render_fba,
)


# Tool-use system prompt: the A-architecture SYSTEM_PROMPT (reasoning
# principles) plus a short instruction about the tool-driven workflow. The
# reasoning principles are shared verbatim; only the workflow framing differs.
TOOL_USE_SUFFIX = """

# Workflow

You have access to the six evidence channels above as callable tools. Rather
than receiving all evidence up front, you decide which channels to query.
Call the tools you judge most informative for this gene. You do NOT need to
call all six: for example, a clearly non-metabolic gene (ribosomal protein,
cell-division protein) may not warrant an FBA call, and you may reason about
why you are skipping it. When you have gathered enough evidence to classify
the gene, stop calling tools and emit your final reasoning followed by the
JSON verdict block.

Each tool takes no arguments (the gene under analysis is fixed for this
session). Call a tool to retrieve that channel's evidence for the gene.
"""

SYSTEM_PROMPT_B = SYSTEM_PROMPT + TOOL_USE_SUFFIX


# The six channels exposed as Anthropic tool schemas. Each takes no input:
# the Case is fixed per run, and the dispatcher supplies the actual args.
TOOL_SCHEMAS = [
    {
        "name": "query_uniprot",
        "description": (
            "Retrieve curated UniProt annotation for the gene: protein name, "
            "function description, EC numbers, GO terms, Pfam domains, "
            "subcellular location."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_kegg",
        "description": (
            "Retrieve KEGG pathway and orthology data for the gene: KO "
            "assignments, pathway membership, gene definition."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_string",
        "description": (
            "Retrieve the STRING protein-protein interaction network for the "
            "gene: high-confidence partners and per-channel scores."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_esm2",
        "description": (
            "Run the ESM-2 protein language model on the gene's protein "
            "sequence: mean per-residue log-likelihood and mean variant "
            "effect (conservation/constraint signal at the protein level)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_evo2",
        "description": (
            "Run the Evo2 genomic foundation model on the gene's DNA "
            "sequence: mean causal log-likelihood (constraint signal at the "
            "nucleotide level)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_fba",
        "description": (
            "Run flux balance analysis with single-gene knockout on the "
            "genome-scale metabolic model. Returns wildtype vs knockout "
            "growth and an essentiality call. NOTE: only metabolic genes are "
            "in the model; non-metabolic genes (translation, division, "
            "transcription) will return an error indicating absence from the "
            "model, which is itself informative."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Map tool name -> (bundle attribute, channel key for errors/render)
_TOOL_TO_CHANNEL = {
    "query_uniprot": "uniprot",
    "query_kegg": "kegg",
    "query_string": "string",
    "query_esm2": "esm2",
    "query_evo2": "evo2",
    "query_fba": "fba",
}

_RENDERERS = {
    "uniprot": _render_uniprot,
    "kegg": _render_kegg,
    "string": _render_string,
    "esm2": _render_esm2,
    "evo2": _render_evo2,
    "fba": _render_fba,
}


def _default_methods() -> Dict[str, str]:
    return {
        "uniprot": "get_annotation",
        "kegg": "fetch_gene",
        "string": "fetch_interaction_partners",
        "esm2": "embed",
        "evo2": "score_sequence",
        "fba": "compute_essentiality",
    }


def dispatch_tool(
    channel: str,
    case: Case,
    tools: Dict[str, Any],
    bundle: EvidenceBundle,
    methods: Dict[str, str],
) -> str:
    """Execute one channel for the case; store result in bundle; return
    the rendered text for that channel (for the tool_result block).

    Uses the SAME argument-marshalling patterns as A's gather_all_evidence,
    so a channel populated by B is identical to one populated by A. Per-tool
    exceptions are caught, recorded in bundle.errors, and surfaced in the
    returned text so the model sees the failure (e.g. FBA KeyError for a
    non-metabolic gene).
    """
    if channel not in tools:
        return f"_Tool for channel '{channel}' is not available in this run._"

    try:
        m = getattr(tools[channel], methods[channel])
        if channel == "uniprot":
            result = m(
                case.locus_tag, case.uniprot_taxon,
                gene_symbol=case.gene_symbol,
            )
        elif channel == "kegg":
            result = m(*case.kegg_gene_id.split(":", 1))
        elif channel == "string":
            result = m(case.gene_symbol, case.string_species)
        elif channel == "esm2":
            if not case.protein_sequence:
                return "_No protein sequence available for this gene._"
            result = m(case.protein_sequence, case.gene_symbol)
        elif channel == "evo2":
            if not case.dna_sequence:
                return "_No DNA sequence available for this gene._"
            result = m(case.dna_sequence)
        elif channel == "fba":
            if "fba_model" not in tools:
                return "_FBA model not available in this run._"
            result = m(tools["fba_model"], case.locus_tag)
        else:
            return f"_Unknown channel '{channel}'._"

        setattr(bundle, channel, result)
        return _RENDERERS[channel](result, bundle.errors)

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        bundle.errors[channel] = msg
        # Surface the error to the model via the renderer (it reads errors).
        return _RENDERERS[channel](None, bundle.errors)


class EssentialityOrchestratorB:
    """B-architecture: model-driven tool-use loop over the six channels."""

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_MAX_ITERATIONS = 12

    def __init__(
        self,
        anthropic_client=None,
        model: str = DEFAULT_MODEL,
        tools: Optional[Dict[str, Any]] = None,
        system_prompt: str = SYSTEM_PROMPT_B,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tool_methods: Optional[Dict[str, str]] = None,
    ):
        self._anthropic_client = anthropic_client
        self.model = model
        self.tools = tools or {}
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.methods = _default_methods()
        if tool_methods:
            self.methods.update(tool_methods)

    def _ensure_client(self):
        if self._anthropic_client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is required for the orchestrator. "
                    "Install: pip install anthropic"
                ) from exc
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable not set; "
                    "either set it or pass an anthropic_client at construction."
                )
            self._anthropic_client = Anthropic(api_key=api_key)
        return self._anthropic_client

    def run(self, case: Case) -> Tuple[EvidenceBundle, Verdict, List[str]]:
        """Run the tool-use loop on a single case.

        Returns (bundle, verdict, tool_call_order) where tool_call_order is
        the list of channel names the model chose to call, in order — the
        artifact that distinguishes B's behavior from A's eager dispatch.
        """
        client = self._ensure_client()
        bundle = EvidenceBundle(case=case)
        tool_call_order: List[str] = []

        initial_text = (
            f"Classify the essentiality of the following gene by querying "
            f"the evidence channels you judge most informative.\n\n"
            f"Gene: {case.gene_symbol} (locus tag {case.locus_tag})\n"
            f"Organism: {case.organism_strain} (taxon {case.organism_taxon})\n"
            f"Metabolic model available: {case.metabolic_model}\n\n"
            f"Call tools to gather evidence, then emit your JSON verdict."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": initial_text}
        ]

        final_text = ""
        for _ in range(self.max_iterations):
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # Collect any text the model emitted this turn
            turn_text = "\n".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            if turn_text:
                final_text = turn_text

            tool_uses = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use"
            ]

            if not tool_uses:
                # No tool calls this turn -> model is done reasoning.
                break

            # Append the assistant turn (with tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each requested tool, build tool_result blocks
            tool_results = []
            for tu in tool_uses:
                channel = _TOOL_TO_CHANNEL.get(tu.name)
                if channel is None:
                    rendered = f"_Unknown tool '{tu.name}'._"
                else:
                    tool_call_order.append(channel)
                    rendered = dispatch_tool(
                        channel, case, self.tools, bundle, self.methods
                    )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": rendered,
                })
            messages.append({"role": "user", "content": tool_results})

        verdict = parse_verdict(final_text, case.case_id)
        return bundle, verdict, tool_call_order


__all__ = [
    "EssentialityOrchestratorB",
    "SYSTEM_PROMPT_B",
    "TOOL_SCHEMAS",
    "dispatch_tool",
]
