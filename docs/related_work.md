# Related work and integrated data sources

This prototype builds on and integrates the following prior work.

## JCVI-syn3A as a biological substrate

- **Hutchison et al. 2016 (Science)**: design and synthesis of the syn3.0
  minimal bacterial genome, the predecessor of syn3A.
- **Breuer et al. 2019 (eLife 36842)**: essentiality classification by
  transposon insertion sequencing for the 452 protein-coding genes of syn3A.
  The canonical ground-truth essentiality dataset used throughout this work.
- **Pedreira et al. 2022 (Protein Science)**: SynWiki database, the existing
  curated relational database for syn3A, including protein-protein
  interaction data. Available at synwiki.uni-goettingen.de.

## Annotation and structural resources

- **Kilinc, Jia, and Jernigan 2025 (Methods in Molecular Biology)**:
  PROST-syn3A, improved structural annotations for syn3A proteins via
  TM-align. Available at bit.ly/prost-syn3a.

## Metabolic model

- **iJCVIsyn3A** genome-scale metabolic model: covers 155 metabolic genes of
  the 452 protein-coding genes in syn3A. Used here for flux balance analysis
  via COBRApy.

## Foundation models

- **Evo2 7B base** (Arc Institute): genomic language model used for
  sequence-level constraint signals, with organism-specific GC calibration
  applied.
- **ESM-2** (Meta FAIR / EvolutionaryScale): protein language model used for
  protein-level variant-effect signals.

## Knowledge representation

- **Open Knowledge Format (OKF v0.1)** (Google Cloud, June 12, 2026):
  vendor-neutral markdown-and-YAML-frontmatter representation for knowledge
  atoms intended for consumption by AI agents. This prototype ships the
  syn3A knowledge graph as an OKF bundle, integrating the data sources above
  into a directory of concept-per-file documents linked by markdown
  cross-references. See `docs/okf_bundle_spec.md` for the bundle conformance
  specification. OKF specification at
  https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf and
  introductory blog at
  https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing.

## Knowledge graph integration

The custom syn3A knowledge graph integrates iJCVIsyn3A, Breuer 2019,
eggNOG-mapper functional annotations, PROST-syn3A structural annotations, and
SynWiki curated annotations into one OKF bundle. This addresses the gap
identified in our survey of existing syn3A resources: SynWiki is a relational
database with a web interface; iJCVIsyn3A is an SBML metabolic model;
PROST-syn3A is a structural-annotation static site. No prior work integrates
these into a single, formally-specified, agent-readable knowledge
representation.

## Future integrations (under evaluation)

- **Hyper-Extract** (Feng et al., 2026, Apache-2.0): LLM-driven knowledge
  extraction CLI supporting Auto-Graph and Auto-Hypergraph output types with
  eighty-plus domain templates. Relevant for Paper 2's dark-gene literature
  mining once we move beyond the curated structured-database sources
  currently integrated. https://github.com/yifanfeng97/Hyper-Extract

- **ERA / Flat UCB Tree Search** (Aygun et al. 2025, arXiv 2509.06503):
  iterative LLM-plus-search algorithm for generating empirical software
  against scoring functions. Relevant for future configuration- or
  prompt-search work on the agent itself once an appropriate scoring
  function is identified. https://github.com/google-research/era
