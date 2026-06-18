"""Essentiality reasoning orchestrator (A-architecture).

Integrates evidence from six independent channels (UniProt, KEGG, STRING,
ESM-2, Evo2, FBA) into a structured EvidenceBundle, renders it into a
prompt for claude-sonnet-4-6, and parses the structured verdict.

A-architecture: eagerly call all six tools, render the populated bundle,
make one LLM call, parse the response.

Reusable abstractions that transfer verbatim to B-architecture (tool-use
loop): Case, EvidenceBundle, Verdict, SYSTEM_PROMPT, render_evidence_bundle
(handles None channels and per-tool errors), parse_verdict (standalone JSON
extraction).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are an essentiality reasoning agent. Your task is to classify whether a given gene is essential for organism viability under standard laboratory growth conditions (log-phase growth on minimal or rich glucose-based media), by integrating evidence from six independent channels.

# Evidence channels

1. **UniProt** — curated functional annotation from the UniProtKB knowledgebase: protein name, function description, EC numbers, GO terms, Pfam domains, subcellular location.
2. **KEGG** — pathway membership and orthology: KO (KEGG Orthology) assignments, pathway participation, gene description.
3. **STRING** — protein-protein interaction network: high-confidence interaction partners and per-channel scores (experimental, database, textmining, coexpression, etc.). Combined scores >= 700 indicate high-confidence partnerships.
4. **ESM-2** — protein language model (Meta AI). Returns mean per-residue log-likelihood and mean variant effect score. Strongly conserved residues produce more positive log-likelihoods; genes under strong purifying selection show low mean variant effects.
5. **Evo2** — genomic foundation model (Arc Institute). Returns mean causal log-likelihood per nucleotide for the gene's DNA sequence. Strongly constrained DNA sequences (under selection) have less-negative log-likelihoods.
6. **FBA** — flux balance analysis using a genome-scale metabolic model (typically iML1515 for E. coli). Simulates wildtype growth and single-gene knockout growth. Classifies essential if knockout growth drops below 1% of wildtype on the specified medium.

# Key reasoning principles

**Empirical evidence dominates conservation signals.** A gene can be highly conserved (low Evo2/ESM-2 variation, central in STRING network, present in core genomes) without being essential under standard lab conditions. The "core genome" includes many quasi-essential genes that are deletable but provide growth advantages or are required only under specific conditions.

**FBA only detects metabolic essentiality.** Genes involved in translation (ribosomal proteins, aminoacyl-tRNA synthetases), DNA replication, transcription, cell division, peptidoglycan biosynthesis, or cell envelope biogenesis are typically NOT flagged by FBA — they are not modeled in the metabolic network. For such genes, weigh UniProt/KEGG annotation more heavily.

**Conservation-essentiality conflation is a known failure mode.** A gene that is bacterially conserved AND central in metabolism (e.g., central glycolysis) is NOT automatically essential. Alternative pathways (e.g., the pentose phosphate bypass around pgi) can rescue knockouts. Trust empirical FBA + single-gene-deletion data over network centrality and conservation when they conflict.

**Conditional essentiality is a real category.** Some genes are essential only under stress (rpoS for stationary phase), specific carbon sources, or nutrient limitation. When channel evidence suggests this pattern, emit the "conditional" classification rather than forcing a binary essential / non-essential call.

# Output format

In 4-8 sentences, walk through the most informative channels and explicitly note any conflicts (e.g., "STRING shows X but FBA shows Y"). Conclude with your reasoning and emit a JSON block in this exact format:

```json
{
  "classification": "essential" | "non_essential" | "conditional" | "uncertain",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one to three sentence summary of the decisive evidence>"
}
```

Classifications:
- **essential**: gene knockout would prevent organism growth under standard log-phase conditions on glucose minimal medium.
- **non_essential**: gene knockout is viable under standard conditions, possibly with reduced growth rate.
- **conditional**: essential only under specific conditions (stress, stationary phase, alternate carbon source, antibiotic challenge).
- **uncertain**: evidence is conflicting or insufficient for a confident call.

Confidence reflects how decisive the evidence is, not how confident you feel intuitively. A confidence of 0.95 should mean multiple channels agree strongly; 0.5-0.7 means meaningful conflict between channels; below 0.5 means weak or ambiguous evidence.
"""

USER_INSTRUCTION = """Based on the evidence above, classify this gene's essentiality. Walk through the channels in your reasoning, explicitly note any conflicts between channels, and weigh empirical evidence appropriately against conservation/centrality signals. End with the JSON verdict block."""


@dataclass
class Case:
    """A single gene-essentiality reasoning case."""

    case_id: str
    gene_symbol: str
    locus_tag: str
    organism_taxon: int
    uniprot_taxon: int
    organism_strain: str
    kegg_gene_id: str
    string_species: int
    metabolic_model: str
    protein_sequence: str
    dna_sequence: str
    design_axis: str
    ground_truth_essentiality: str
    ground_truth_source: str
    design_rationale: str


@dataclass
class EvidenceBundle:
    """Per-case container for results from all six tool channels.

    Each channel is Optional[ResultType]: None means "not queried", a
    populated value means the query succeeded. Per-tool failures are
    captured in the errors dict keyed by channel name.
    """

    case: Case
    uniprot: Optional[Any] = None
    kegg: Optional[Any] = None
    string: Optional[Any] = None
    esm2: Optional[Any] = None
    evo2: Optional[Any] = None
    fba: Optional[Any] = None
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class Verdict:
    """Structured essentiality classification with confidence and reasoning."""

    case_id: str
    classification: str
    confidence: float
    reasoning: str
    raw_response: str


# --------------------------------------------------------------------------
# Per-channel rendering (private; called by render_evidence_bundle)
# --------------------------------------------------------------------------


def _channel_status(result, error_msg):
    if error_msg:
        return f"_Query failed: {error_msg}_"
    if result is None:
        return "_(not queried)_"
    return None


def _safe_list(value, max_items=None):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return [value]
    items = list(value)
    if max_items is not None:
        items = items[:max_items]
    return items


def _render_uniprot(uniprot, errors):
    status = _channel_status(uniprot, errors.get("uniprot"))
    if status:
        return status
    lines = []
    name = getattr(uniprot, "protein_name", None)
    if name:
        lines.append(f"Protein name: {name}")
    gene = getattr(uniprot, "gene_name", None)
    if gene:
        lines.append(f"Gene name: {gene}")
    ec = getattr(uniprot, "ec_numbers", None) or getattr(uniprot, "ec", None)
    if ec:
        lines.append(f"EC: {', '.join(str(e) for e in _safe_list(ec))}")
    fn = (
        getattr(uniprot, "function_description", None)
        or getattr(uniprot, "function", None)
    )
    if fn:
        text = str(fn)
        if len(text) > 300:
            text = text[:300] + "..."
        lines.append(f"Function: {text}")
    go = getattr(uniprot, "go_terms", None)
    if go:
        rendered = []
        for g in _safe_list(go, 5):
            if isinstance(g, dict):
                gid = g.get("id") or g.get("go_id") or "?"
                term = (
                    g.get("term") or g.get("name") or g.get("description") or "?"
                )
                rendered.append(f"{gid} ({term})")
            else:
                rendered.append(str(g))
        lines.append(f"GO terms (top 5): {', '.join(rendered)}")
    pfam = (
        getattr(uniprot, "pfam_domains", None)
        or getattr(uniprot, "pfam", None)
    )
    if pfam:
        lines.append(
            f"Pfam domains: {', '.join(str(p) for p in _safe_list(pfam, 5))}"
        )
    subloc = (
        getattr(uniprot, "subcellular_locations", None)
        or getattr(uniprot, "subcellular_location", None)
    )
    if subloc:
        if isinstance(subloc, (list, tuple)):
            lines.append(
                f"Subcellular location: {', '.join(str(s) for s in subloc)}"
            )
        else:
            lines.append(f"Subcellular location: {subloc}")
    xrefs = []
    for attr, label in (
        ("kegg_xrefs", "KEGG"),
        ("eggnog_xrefs", "eggNOG"),
        ("pdb_xrefs", "PDB"),
    ):
        v = getattr(uniprot, attr, None)
        if v:
            xrefs.append(
                f"{label}: {', '.join(str(x) for x in _safe_list(v, 3))}"
            )
    if xrefs:
        lines.append(f"Cross-refs: {'; '.join(xrefs)}")
    return "\n".join(lines) if lines else "_(no fields populated)_"


def _render_kegg(kegg, errors):
    status = _channel_status(kegg, errors.get("kegg"))
    if status:
        return status
    lines = []
    sym = (
        getattr(kegg, "gene_name", None)
        or getattr(kegg, "gene_symbol", None)
    )
    if sym:
        lines.append(f"Gene symbol: {sym}")
    defn = (
        getattr(kegg, "definition", None)
        or getattr(kegg, "description", None)
    )
    if defn:
        lines.append(f"Definition: {defn}")
    orths = (
        getattr(kegg, "orthologies", None)
        or getattr(kegg, "orthology", None)
    )
    if orths:
        rendered = []
        for o in _safe_list(orths, 3):
            if isinstance(o, dict):
                kid = o.get("ko_id") or o.get("id") or "?"
                desc = o.get("description") or o.get("name") or ""
                if len(desc) > 80:
                    desc = desc[:80] + "..."
                rendered.append(f"{kid} ({desc})" if desc else kid)
            else:
                rendered.append(str(o))
        lines.append(f"Orthology (top 3): {', '.join(rendered)}")
    paths = getattr(kegg, "pathways", None)
    if paths:
        rendered = []
        for p in _safe_list(paths, 5):
            if isinstance(p, dict):
                pid = p.get("pathway_id") or p.get("id") or "?"
                pname = p.get("name") or ""
                if len(pname) > 60:
                    pname = pname[:60] + "..."
                rendered.append(f"{pid} ({pname})" if pname else pid)
            else:
                rendered.append(str(p))
        lines.append(f"Pathways (top 5): {', '.join(rendered)}")
    motif = (
        getattr(kegg, "motif_pfam", None) or getattr(kegg, "pfam", None)
    )
    if motif:
        lines.append(
            f"Pfam motifs: {', '.join(str(m) for m in _safe_list(motif, 5))}"
        )
    return "\n".join(lines) if lines else "_(no fields populated)_"


def _render_string(net, errors):
    status = _channel_status(net, errors.get("string"))
    if status:
        return status
    lines = []
    species = (
        getattr(net, "species_taxon", None) or getattr(net, "species", None)
    )
    if species is not None:
        lines.append(f"Species taxon: {species}")
    partners = (
        getattr(net, "partners", None)
        or getattr(net, "interactions", None)
        or []
    )
    lines.append(f"Number of partners returned: {len(partners)}")
    if partners:
        def score_key(p):
            return getattr(p, "combined_score", 0) or 0
        top = sorted(partners, key=score_key, reverse=True)[:10]
        lines.append("Top 10 high-confidence partners (by combined score):")
        for p in top:
            name = (
                getattr(p, "partner_preferred_name", None)
                or getattr(p, "partner_name", "?")
            )
            combined = getattr(p, "combined_score", "?")
            exp = getattr(p, "experimental_score", "?")
            db = getattr(p, "database_score", "?")
            txt = getattr(p, "textmining_score", "?")
            lines.append(
                f"  - {name} (combined: {combined}, experimental: {exp}, "
                f"database: {db}, textmining: {txt})"
            )
    return "\n".join(lines)


def _render_esm2(esm2, errors):
    status = _channel_status(esm2, errors.get("esm2"))
    if status:
        return status
    lines = []
    lines.append(f"Model: {getattr(esm2, 'model_name', '?')}")
    sl = (
        getattr(esm2, "sequence_length_aa", None)
        if getattr(esm2, "sequence_length_aa", None) is not None
        else getattr(esm2, "sequence_length", None)
    )
    if sl is not None:
        lines.append(f"Sequence length: {sl} aa")
    ll = getattr(esm2, "mean_log_likelihood", None)
    if ll is not None:
        lines.append(
            f"Mean log-likelihood: {ll:.4f} "
            f"(less negative = more likely under the model)"
        )
    mve = getattr(esm2, "mean_variant_effect", None)
    if mve is not None:
        lines.append(
            f"Mean variant effect: {mve:.4f} "
            f"(lower = more under purifying selection)"
        )
    return "\n".join(lines)


def _render_evo2(evo2, errors):
    status = _channel_status(evo2, errors.get("evo2"))
    if status:
        return status
    lines = []
    lines.append(f"Model: {getattr(evo2, 'model_name', '?')}")
    sl = getattr(evo2, "sequence_length", None)
    if sl is not None:
        lines.append(f"Sequence length: {sl} bp")
    ll = getattr(evo2, "mean_log_likelihood", None)
    if ll is not None:
        lines.append(
            f"Mean causal log-likelihood: {ll:.4f} "
            f"(less negative = more constrained)"
        )
    return "\n".join(lines)


def _render_fba(fba, errors):
    status = _channel_status(fba, errors.get("fba"))
    if status:
        return status
    lines = []
    lines.append(f"Model: {getattr(fba, 'model_id', '?')}")
    obj = getattr(fba, "objective_id", None)
    if obj:
        lines.append(f"Objective: {obj}")
    wt = getattr(fba, "wildtype_growth_rate", None)
    ko = getattr(fba, "knockout_growth_rate", None)
    ratio = getattr(fba, "growth_ratio", None)
    if wt is not None:
        lines.append(f"Wildtype growth rate: {wt:.4f} h^-1")
    if ko is not None:
        lines.append(f"Knockout growth rate: {ko:.4f} h^-1")
    if ratio is not None:
        lines.append(f"Growth ratio: {ratio:.4f}")
    is_ess = getattr(fba, "is_essential", None)
    thr = getattr(fba, "threshold_ratio", None)
    if is_ess is not None:
        thr_str = (
            f" (threshold: KO < {thr:.2%} of WT)" if thr is not None else ""
        )
        cls = "ESSENTIAL" if is_ess else "non-essential"
        lines.append(f"FBA classification{thr_str}: {cls}")
    st = getattr(fba, "status", None)
    if st:
        lines.append(f"Solver status: {st}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def render_evidence_bundle(bundle: EvidenceBundle) -> str:
    """Format an EvidenceBundle into structured markdown for the LLM."""
    case = bundle.case
    sections = [
        f"## Case: {case.gene_symbol} ({case.locus_tag})",
        f"Organism: {case.organism_strain} (taxon {case.organism_taxon})",
        f"Metabolic model: {case.metabolic_model}",
        f"Design axis: {case.design_axis}",
        "",
        "### UniProt (curated annotation)",
        _render_uniprot(bundle.uniprot, bundle.errors),
        "",
        "### KEGG (pathway and orthology)",
        _render_kegg(bundle.kegg, bundle.errors),
        "",
        "### STRING (protein-protein interactions)",
        _render_string(bundle.string, bundle.errors),
        "",
        "### ESM-2 (protein language model)",
        _render_esm2(bundle.esm2, bundle.errors),
        "",
        "### Evo2 (genomic foundation model)",
        _render_evo2(bundle.evo2, bundle.errors),
        "",
        "### FBA (flux balance analysis)",
        _render_fba(bundle.fba, bundle.errors),
        "",
    ]
    return "\n".join(sections)


def gather_all_evidence(
    case: Case,
    tools: Dict[str, Any],
    tool_methods: Optional[Dict[str, str]] = None,
) -> EvidenceBundle:
    """Eagerly call all six tools; return populated EvidenceBundle.

    `tools` maps channel name to tool instance. Pass an `fba_model` (a
    cobra.Model) alongside the 'fba' tool. Per-tool exceptions are caught
    and recorded in bundle.errors[channel] without halting the bundle.
    `tool_methods` (optional) overrides the default method name for any
    channel.
    """
    methods = {
        "uniprot": "get_annotation",
        "kegg": "fetch_gene",
        "string": "fetch_interaction_partners",
        "esm2": "embed",
        "evo2": "score_sequence",
        "fba": "compute_essentiality",
    }
    if tool_methods:
        methods.update(tool_methods)

    bundle = EvidenceBundle(case=case)

    if "uniprot" in tools:
        try:
            m = getattr(tools["uniprot"], methods["uniprot"])
            bundle.uniprot = m(
                case.locus_tag,
                case.uniprot_taxon,
                gene_symbol=case.gene_symbol,
            )
        except Exception as e:
            bundle.errors["uniprot"] = f"{type(e).__name__}: {e}"

    if "kegg" in tools:
        try:
            m = getattr(tools["kegg"], methods["kegg"])
            bundle.kegg = m(*case.kegg_gene_id.split(":", 1))
        except Exception as e:
            bundle.errors["kegg"] = f"{type(e).__name__}: {e}"

    if "string" in tools:
        try:
            m = getattr(tools["string"], methods["string"])
            bundle.string = m(case.gene_symbol, case.string_species)
        except Exception as e:
            bundle.errors["string"] = f"{type(e).__name__}: {e}"

    if "esm2" in tools and case.protein_sequence:
        try:
            m = getattr(tools["esm2"], methods["esm2"])
            bundle.esm2 = m(case.protein_sequence, case.gene_symbol)
        except Exception as e:
            bundle.errors["esm2"] = f"{type(e).__name__}: {e}"

    if "evo2" in tools and case.dna_sequence:
        try:
            m = getattr(tools["evo2"], methods["evo2"])
            bundle.evo2 = m(case.dna_sequence)
        except Exception as e:
            bundle.errors["evo2"] = f"{type(e).__name__}: {e}"

    if "fba" in tools and "fba_model" in tools and case.locus_tag:
        try:
            m = getattr(tools["fba"], methods["fba"])
            bundle.fba = m(tools["fba_model"], case.locus_tag)
        except Exception as e:
            bundle.errors["fba"] = f"{type(e).__name__}: {e}"

    return bundle


def parse_verdict(claude_response: str, case_id: str) -> Verdict:
    """Extract a structured Verdict from Claude's response text.

    Tries (1) a fenced ```json``` block; (2) any standalone JSON object
    containing a "classification" key. Falls back to 'uncertain' with
    confidence 0.0 if no parseable JSON is found.
    """
    candidates: List[str] = []

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        claude_response, re.DOTALL,
    )
    if fenced:
        candidates.append(fenced.group(1))

    inline = re.findall(
        r'\{[^{}]*"classification"\s*:\s*"[^"]+",[^{}]*\}',
        claude_response, re.DOTALL,
    )
    candidates.extend(inline)

    for cand in candidates:
        try:
            data = json.loads(cand)
            if "classification" in data:
                return Verdict(
                    case_id=case_id,
                    classification=str(data.get("classification", "uncertain")),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=str(data.get("reasoning", "")),
                    raw_response=claude_response,
                )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue

    return Verdict(
        case_id=case_id,
        classification="uncertain",
        confidence=0.0,
        reasoning="Could not parse structured verdict from response",
        raw_response=claude_response,
    )


class EssentialityOrchestrator:
    """A-architecture orchestrator: gather -> render -> reason -> parse."""

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 2000

    def __init__(
        self,
        anthropic_client=None,
        model: str = DEFAULT_MODEL,
        tools: Optional[Dict[str, Any]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        user_instruction: str = USER_INSTRUCTION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tool_methods: Optional[Dict[str, str]] = None,
    ):
        self._anthropic_client = anthropic_client
        self.model = model
        self.tools = tools or {}
        self.system_prompt = system_prompt
        self.user_instruction = user_instruction
        self.max_tokens = max_tokens
        self.tool_methods = tool_methods

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

    def run(self, case: Case) -> Tuple[EvidenceBundle, Verdict]:
        """Run the orchestrator on a single case."""
        bundle = gather_all_evidence(case, self.tools, self.tool_methods)
        prompt = render_evidence_bundle(bundle) + "\n\n" + self.user_instruction

        client = self._ensure_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        claude_text = "\n".join(text_blocks)
        verdict = parse_verdict(claude_text, case.case_id)
        return bundle, verdict


def load_cases(path) -> List[Case]:
    """Load cases from a JSON file matching the data/cases.json schema."""
    data = json.loads(Path(path).read_text())
    return [Case(**c) for c in data.get("cases", [])]


__all__ = [
    "Case", "EvidenceBundle", "Verdict",
    "SYSTEM_PROMPT", "USER_INSTRUCTION",
    "render_evidence_bundle", "gather_all_evidence", "parse_verdict",
    "EssentialityOrchestrator", "load_cases",
]
