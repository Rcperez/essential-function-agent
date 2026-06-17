# Related work and integrated data sources

This prototype builds on and integrates the following prior work.

## JCVI-syn3A as a biological substrate

- **Hutchison et al. 2016 (Science)**: design and synthesis of the syn3.0 minimal
  bacterial genome, the predecessor of syn3A.
- **Breuer et al. 2019 (eLife 36842)**: essentiality classification by transposon
  insertion sequencing for the 452 protein-coding genes of syn3A. The
  canonical ground-truth essentiality dataset used throughout this work.
- **Pedreira et al. 2022 (Protein Science)**: SynWiki database, the existing
  curated relational database for syn3A, including protein-protein interaction
  data. Available at synwiki.uni-goettingen.de.

## Annotation and structural resources

- **Kilinc, Jia, and Jernigan 2025 (Methods in Molecular Biology)**: PROST-syn3A,
  improved structural annotations for syn3A proteins via TM-align. Available at
  bit.ly/prost-syn3a.

## Metabolic model

- **iJCVIsyn3A** genome-scale metabolic model: covers 155 metabolic genes of the
  452 protein-coding genes in syn3A. Used here for flux balance analysis via
  COBRApy.

## Foundation models

- **Evo2 7B base** (Arc Institute): genomic language model used for sequence-level
  constraint signals, with organism-specific GC calibration applied.
- **ESM-2** (Meta FAIR / EvolutionaryScale): protein language model used for
  protein-level variant-effect signals.

## Knowledge integration

The custom syn3A knowledge graph (under construction) integrates the above
sources into an RDF triplestore queryable via SPARQL, with formal schema
linking to the Gene Ontology, Sequence Ontology, Systems Biology Ontology,
Evidence and Conclusion Ontology, ChEBI, and PSI-MI ontologies.

This addresses the gap identified in our survey of existing syn3A resources:
SynWiki is a relational database; iJCVIsyn3A is an SBML metabolic model;
PROST-syn3A is a structural-annotation static site. No prior work integrates
these into a formal knowledge graph with machine-queryable structure.
