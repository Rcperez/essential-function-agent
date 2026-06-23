# syn3A Essentiality Analysis: Safety-Layer Refusal (measured finding)

## Summary

Gene-essentiality analysis of **JCVI-syn3A** (a published minimal synthetic
bacterium) is **hard-refused by Anthropic's safety layer** — near-
deterministically, on both Sonnet 4.6 and Opus 4.8, independent of
architecture (analytical or agentic) and gene function. The identical task
on *E. coli* K-12 passes cleanly. Among three major providers, **only
Anthropic refuses**: OpenAI (gpt-4o) and Google (gemini-2.5-pro) both comply
~100% with genuine classifications. The reproducible syn3A "result" of this
demonstrator is therefore a characterization of an Anthropic-specific
over-refusal, not a cross-organism accuracy number.

## Measured evidence (see `syn3a_refusal_matrix.json`)

| provider | model(s) | syn3A | E. coli |
|----------|----------|-------|---------|
| **Anthropic** | Sonnet 4.6, Opus 4.8 | **HARD refusal 30/30** | 0/12 |
| OpenAI | gpt-4o | complied 15/15 | complied 6/6 |
| Google | gemini-2.5-pro | complied 15/15 | complied 6/6 |

Within Anthropic (Sonnet 4.6), analytical (A) and agentic (B) framings
refused equally (15/15 and 14/15) — architecture is not the trigger.
Metabolic (pyrG/adk/eno) and annotation-driven (ftsZ/rpsB) genes both
refused at ceiling — gene function is not the trigger. The trigger is the
synthetic-minimal-organism identity (JCVI-syn3A, *M. mycoides* lineage).

## Why it is a false positive

The work is reproduction of published science: JCVI-syn3A's genome
(CP016816.2), its genome-scale metabolic model (iJCVIsyn3A), and its
experimentally determined essential genes (Breuer et al. 2019, *eLife*) are
all public. Classifying whether a *known* gene is essential provides no
uplift toward designing or modifying any organism. An explicit, truthful
research-context preamble did not clear the refusal.

## Notable

The complying models cited syn3A's minimal-synthetic-genome status as their
*strongest* evidence of essentiality. The property that triggers Anthropic's
refusal is the property that makes the question answerable.

## What still holds

The E. coli cross-organism bridge genes (ftsZ, rpsB) classified essential in
both organisms in an earlier run; the E. coli side is reliable. The syn3A
side cannot be reliably run under the current Anthropic safety layer.

## Correction history

An earlier committed finding claimed the refusal was specific to the agentic
(B) architecture and that the analytical (A) architecture ran clean.
Controlled measurement falsified that — A refuses at least as often as B.
The corrected, measured finding is the one above. Reported to Anthropic as a
false positive (`ANTHROPIC_FALSE_POSITIVE_REPORT.md`).
