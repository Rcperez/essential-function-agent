# Neurohybrid Agent for Essential Function Prediction

A prototype agent for predicting the functions of essential genes in minimal
cells, combining:

- **Knowledge-graph retrieval** over UniProt, KEGG, STRING, and (forthcoming)
  a custom JCVI-syn3A knowledge graph integrating the iJCVIsyn3A metabolic
  model, Breuer 2019 essentiality data, eggNOG functional annotations, and
  PROST-syn3A structural annotations
- **Genomic language model** (Evo2) for sequence-level evolutionary-constraint
  signals, with organism-specific GC calibration
- **Protein language model** (ESM-2) for protein-level variant-effect signals
- **Mechanistic simulation** via flux balance analysis on curated genome-scale
  metabolic models (iJCVIsyn3A, iML1515)
- **LLM reasoning** (Claude Sonnet 4.6) over structured multi-channel evidence
  with explicit reasoning-chain capture

The agent generates verdicts and structured reasoning chains for known-essential
and unknown-function essential genes.

## Status

Prototype, June 2026. Establishes the architecture and demonstrates on a
curated case set of JCVI-syn3A and E. coli genes spanning known-essential,
known-non-essential, conservation-confounded, and dark-essential cases.

## Motivation

JCVI-syn3A is the smallest viable synthetic cell with 452 protein-coding genes.
Of these, approximately 30 essential genes have no known function. The
longer-arc goal of this work is to generate biologically plausible functional
hypotheses for these dark essential genes. This prototype validates the
architectural approach on genes whose essentiality is independently known.

## Architecture
+----------------------------+
                |   structured prompt with   |
                |   per-channel evidence     |
                +-------------+--------------+
                              |
                              v
                      [ Claude Sonnet 4.6 ]
                              |
                              v
                +----------------------------+
                |   verdict + reasoning      |
                |   chain + function         |
                |   hypothesis               |
                +----------------------------+
                [FBA]         single-gene-deletion against iJCVIsyn3A / iML1515
[Evo2]        GC-corrected mean log-likelihood (and variant-effect
              readouts forthcoming)
[ESM-2]       mean embedding + zero-shot variant effects via masked-LM
[KG-REST]     UniProt + KEGG + STRING REST retrieval
[KG-syn3A]    custom RDF knowledge graph for syn3A (under construction)
## Quick start

[TBD — coming with v0.1 release]

## Demonstration case set

| Gene | Organism | Class | Purpose |
|---|---|---|---|
| JCVISYN3A_0129 (PyrG) | syn3A | known essential | sanity check |
| JCVISYN3A_0133 (IetS) | syn3A | known non-essential | sanity check |
| JCVISYN3A_0522 (FtsZ) | syn3A | non-essential in syn3A, essential elsewhere | conservation-confounded |
| JCVISYN3A_0930 (RpmG) | syn3A | option_a non-essential | replicable across seeds |
| b3189 (rpsB) | E. coli | known essential | cross-organism |
| TBD | syn3A | dark essential | hypothesis generation, well-annotated |
| TBD | syn3A | dark essential | hypothesis generation, sparse-annotation |

## License

MIT

## Citation

If you use this prototype or the JCVI-syn3A knowledge graph derived from it,
please cite:

> Perez, R. (2026). Neurohybrid Knowledge-Graph and Language-Model Agents for
> Essential Gene Function Prediction in Minimal Cells. Preprint forthcoming.

## Related work

This prototype is part of a four-paper arc on agentic biological reasoning.
See [docs/related_work.md](docs/related_work.md) for context on SynWiki,
PROST-syn3A, iJCVIsyn3A, and the Breuer 2019 essentiality data that this
work integrates.
