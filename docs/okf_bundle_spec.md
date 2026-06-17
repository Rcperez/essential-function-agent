# JCVI-syn3A OKF Bundle Specification

This document specifies the conformance of the JCVI-syn3A knowledge graph
bundle to the
[Open Knowledge Format (OKF) v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
specification, plus the syn3A-specific extensions used in the bundle.

## Bundle location

The bundle lives at `src/efa/kg/syn3a_okf/` in this repository. The bundle is
the canonical knowledge representation for the JCVI-syn3A organism in this
prototype; the file system layout is the primary index.

## Bundle structure

~~~
syn3a_okf/
├── index.md
├── genes/
│   ├── index.md
│   ├── JCVISYN3A_0001.md
│   ├── JCVISYN3A_0002.md
│   └── ... (one file per protein-coding gene, 452 files)
├── proteins/
│   ├── index.md
│   └── ... (one file per protein product)
├── reactions/
│   ├── index.md
│   └── ... (155 metabolic reactions from iJCVIsyn3A)
├── metabolites/
│   ├── index.md
│   └── ... (metabolites from iJCVIsyn3A)
├── pathways/
│   ├── index.md
│   └── ... (subsystem assignments from iJCVIsyn3A)
├── ogs/
│   ├── index.md
│   └── ... (eggNOG orthologous groups)
└── log.md
~~~

## Concept types

Each concept document declares its type via the OKF-required `type`
frontmatter field. Bundle-specific types:

- `Gene`: a syn3A protein-coding gene
- `Protein`: a syn3A protein (one per gene)
- `Reaction`: a metabolic reaction from iJCVIsyn3A
- `Metabolite`: a small molecule from iJCVIsyn3A
- `Pathway`: a subsystem from iJCVIsyn3A
- `OrthologGroup`: an eggNOG-assigned orthologous group

## syn3A-specific frontmatter fields

In addition to the OKF-standard fields (`type`, `title`, `description`,
`resource`, `tags`, `timestamp`), the bundle uses the following biology-
specific frontmatter fields.

### For `Gene` concepts

| Field | Type | Description |
|---|---|---|
| `locus_tag` | string | JCVISYN3A_XXXX identifier |
| `organism` | string | always `syn3A` for this bundle |
| `essentiality_class` | enum | `essential`, `quasi_essential`, `non_essential` |
| `in_gsmm` | bool | whether the gene is in the iJCVIsyn3A model |
| `function_class` | string | functional category per Breuer 2019 |
| `protein_length_aa` | int | protein length in amino acids |
| `nucleotide_length_bp` | int | CDS length in base pairs |

### For `Protein` concepts

| Field | Type | Description |
|---|---|---|
| `uniprot_id` | string | UniProt accession |
| `pfam_domains` | list[string] | Pfam domain identifiers |
| `cog_category` | string | COG functional category letter from eggNOG |
| `og_taxonomic_level` | string | broadest OG level (LUCA, Bacteria, etc.) |
| `prost_annotation` | string | function predicted by PROST-syn3A, if any |

### For `Reaction` concepts

| Field | Type | Description |
|---|---|---|
| `reaction_id` | string | iJCVIsyn3A reaction identifier (R_XXX) |
| `ec_number` | string | EC number if assigned |
| `reversible` | bool | reaction reversibility |
| `objective_coefficient` | float | nonzero if part of the biomass reaction |

## Example concept document

~~~markdown
---
type: Gene
title: JCVISYN3A_0129 (PyrG)
description: CTP synthase; only route from UTP to CTP in syn3A
tags: [nucleotide_biosynthesis, essential]
timestamp: 2026-06-15T18:00:00Z
locus_tag: JCVISYN3A_0129
organism: syn3A
essentiality_class: essential
in_gsmm: true
function_class: nucleotide_biosynthesis
protein_length_aa: 532
nucleotide_length_bp: 1599
---

# PyrG (CTP synthase)

The only route from UTP to CTP in syn3A, which lacks cytidine kinase salvage
activity. Classified essential by Breuer 2019.

## Mechanism

CTP synthase converts UTP plus glutamine to CTP plus glutamate via an
ATP-dependent amidation. Single copy in syn3A; no isozyme present.

## Linked concepts

- Encoded protein: [PyrG protein](../proteins/JCVISYN3A_0129_protein.md)
- Catalyzes: [R_CTPS](../reactions/R_CTPS.md)
- Member of pathway: [pyrimidine biosynthesis](../pathways/pyrimidine_biosynthesis.md)

## Evidence

- Breuer et al. 2019 (eLife 36842), supplementary table 3
- iJCVIsyn3A SBML model
~~~

## Cross-linking conventions

Cross-references between concepts use standard markdown links with relative
paths from the linking document. The orchestrator follows links during
context assembly to expand the neighborhood of a queried gene.

## Versioning

This bundle is versioned as `syn3a_okf v0.1`. Material changes to schema or
contents bump the version and are recorded in `log.md`.

## Conformance claims

The bundle conforms to OKF v0.1 in that:

- Every concept document is a markdown file with YAML frontmatter
- Every frontmatter block declares a `type` field
- Cross-references use markdown links
- The bundle is shippable as a directory tree under version control with no
  required runtime or SDK

The bundle extends OKF v0.1 with biology-specific frontmatter fields
described above, all of which are producer-defined (within OKF's design) and
remain ignorable by generic OKF consumers.
