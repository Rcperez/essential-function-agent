# Neurohybrid Agent for Essential Function Prediction

A prototype agent for predicting the functions of essential genes in minimal
cells, combining:

- **Knowledge-graph retrieval** over UniProt, KEGG, STRING, plus a custom
  JCVI-syn3A knowledge graph serialized as an
  [Open Knowledge Format (OKF)](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
  bundle (an LLM-readable, vendor-neutral markdown-and-YAML representation)
- **Genomic language model** (Evo2) for sequence-level evolutionary-constraint
  signals, with organism-specific GC calibration
- **Protein language model** (ESM-2) for protein-level variant-effect signals
- **Mechanistic simulation** via flux balance analysis on curated genome-scale
  metabolic models (iJCVIsyn3A, iML1515)
- **LLM reasoning** (Claude Sonnet 4.6) over structured multi-channel evidence
  with explicit reasoning-chain capture

The agent generates verdicts and structured reasoning chains for
known-essential and unknown-function essential genes.

## The syn3A OKF bundle

This repository ships, to our knowledge, the first biological-domain knowledge
graph published as an Open Knowledge Format (OKF) bundle. The bundle
integrates the iJCVIsyn3A metabolic model, Breuer 2019 essentiality data,
eggNOG functional annotations, PROST-syn3A structural annotations, and
SynWiki curated knowledge into one concept-per-file markdown directory at
`src/efa/kg/syn3a_okf/`. See [docs/okf_bundle_spec.md](docs/okf_bundle_spec.md)
for the bundle conformance specification.

OKF v0.1 was introduced by Google Cloud on June 12, 2026, shortly before the
initial commit of this repository. Adopting it early lets the syn3A KG be
read directly by any OKF-aware agent without translation, and lets the bundle
live in version control alongside the agent code that consumes it.

## Status

Prototype, June 2026. v0.1 establishes the architecture and demonstrates on a
curated case set of JCVI-syn3A and E. coli genes spanning known-essential,
known-non-essential, conservation-confounded, and dark-essential cases.

## Motivation

JCVI-syn3A is the smallest viable synthetic cell with 452 protein-coding
genes. Of these, approximately 30 essential genes have no known function. The
longer-arc goal of this work is to generate biologically plausible functional
hypotheses for these dark essential genes. This prototype validates the
architectural approach on genes whose essentiality is independently known.

## Architecture

~~~
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

  Evidence channels (called per gene):

    [FBA]           single-gene-deletion against iJCVIsyn3A / iML1515
    [Evo2]          GC-corrected mean log-likelihood; saturation
                    mutagenesis and premature-stop perturbation
                    forthcoming
    [ESM-2]         mean embedding + zero-shot variant effects via
                    masked-LM likelihoods
    [REST KG]       UniProt + KEGG + STRING REST retrieval
    [syn3A OKF KG]  custom OKF bundle integrating iJCVIsyn3A SBML +
                    Breuer 2019 + eggNOG + PROST-syn3A + SynWiki
~~~

## Quick start

Coming with v0.1 release.

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

If you use this prototype or the JCVI-syn3A OKF bundle derived from it,
please cite:

> Perez, R. (2026). Neurohybrid Knowledge-Graph and Language-Model Agents for
> Essential Gene Function Prediction in Minimal Cells. Preprint forthcoming.

## Related work

See [docs/related_work.md](docs/related_work.md) for context on SynWiki,
PROST-syn3A, iJCVIsyn3A, Breuer 2019, OKF, and adjacent agentic-AI tooling.
