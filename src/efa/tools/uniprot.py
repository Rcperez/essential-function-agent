"""UniProt REST API retriever for the protein-annotation channel.

Resolves organism + locus tag to UniProt accession and fetches structured
protein annotations: function description, EC numbers, GO terms, Pfam
domains, subcellular locations, and cross-references to KEGG, eggNOG, PDB.

Uses a requests.Session with a urllib3 Retry adapter (retries connect,
read, and status errors with exponential backoff) and a polite
User-Agent header per UniProt API etiquette.

UniProt REST documentation: https://www.uniprot.org/help/api_queries
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from ._http import make_session


UNIPROT_BASE = "https://rest.uniprot.org"
DEFAULT_TIMEOUT_S = 30
DEFAULT_RATE_LIMIT_S = 0.5
DEFAULT_CACHE_DIR = Path(
    "/content/drive/MyDrive/RP_RTP_Repo_Bundles/"
    "essential-function-agent/cache/uniprot"
)


@dataclass
class GOTerm:
    """A single Gene Ontology term annotation.

    aspect is the GO ontology branch letter: P (biological process),
    F (molecular function), C (cellular component).
    """

    go_id: str
    name: str
    aspect: str


@dataclass
class UniProtAnnotation:
    """Structured UniProt annotation for a single protein."""

    accession: str
    locus_tag: Optional[str]
    protein_name: str
    gene_name: Optional[str]
    organism_name: str
    organism_taxon: int
    sequence: str
    sequence_length_aa: int
    function_description: str
    ec_numbers: list[str]
    pfam_domains: list[str]
    interpro_ids: list[str]
    go_terms: list[GOTerm]
    subcellular_locations: list[str]
    kegg_xrefs: list[str]
    eggnog_xrefs: list[str]
    pdb_xrefs: list[str]
    raw_uniprot_url: str


class UniProtRetriever:
    """Retriever for UniProt protein annotations.

    Caches raw JSON responses to disk and re-parses on each call (parsing
    is cheap relative to network roundtrip). Throttles requests to one
    every rate_limit_s seconds to respect UniProt's fair-use guidelines.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
    ) -> None:
        self.cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir is not None else None
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.rate_limit_s = rate_limit_s
        self._last_request_t = 0.0
        self._session = make_session("application/json")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_t
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_t = time.monotonic()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe_key = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_key}.json"

    def search_by_locus_tag(
        self,
        locus_tag: str,
        organism_taxon: Optional[int] = None,
        gene_symbol: Optional[str] = None,
        taxon_fallbacks: Optional[list[int]] = None,
    ) -> Optional[str]:
        """Search UniProt for a gene; return preferred accession or None.

        Tries a sequence of queries from most to least specific and returns
        the first that yields a hit. Among hits, prefers reviewed
        (Swiss-Prot) entries, then an exact gene-symbol match.

        Query order, for each taxon in [organism_taxon, *taxon_fallbacks]:
          1. gene:{gene_symbol}  (if a symbol is supplied)
          2. gene:{locus_tag}    (some organisms index locus in the gene field)
          3. {locus_tag}         (bare full-text)
        Searches are confined to the supplied taxa; the organism filter
        is only dropped if no taxon (and no fallback) was supplied at all.

        Note on taxonomy: UniProt files reviewed E. coli K-12 entries under
        taxon 83333 (species-level K-12), whereas NCBI/KEGG/BiGG use 511145
        (the MG1655 substrain). Callers should pass taxon_fallbacks=[83333]
        (or set organism_taxon directly) so the reviewed entries are found.
        """
        taxa: list[Optional[int]] = []
        if organism_taxon is not None:
            taxa.append(organism_taxon)
        if taxon_fallbacks:
            for t in taxon_fallbacks:
                if t not in taxa:
                    taxa.append(t)
        # If no taxon at all was supplied, allow a single unfiltered search
        # (caller explicitly opted out of organism scoping). Otherwise we
        # never strip the organism filter: an organism-scoped lookup should
        # return None when the gene isn't in the requested taxon rather than
        # returning a spurious cross-organism hit.
        if not taxa:
            taxa.append(None)

        terms: list[str] = []
        if gene_symbol:
            terms.append(f"gene:{gene_symbol}")
        terms.append(f"gene:{locus_tag}")
        terms.append(locus_tag)

        for taxon in taxa:
            for term in terms:
                if taxon is not None:
                    query = f"({term}) AND (organism_id:{taxon})"
                else:
                    query = term
                accession = self._run_search(
                    query, gene_symbol=gene_symbol, locus_tag=locus_tag
                )
                if accession is not None:
                    return accession
        return None

    def _run_search(
        self,
        query: str,
        gene_symbol: Optional[str] = None,
        locus_tag: Optional[str] = None,
    ) -> Optional[str]:
        """Execute one UniProt search query; return preferred accession.

        Caches raw JSON per query string. Among results, prefers reviewed
        entries, then an entry whose gene symbol or ordered-locus-name
        exactly matches the supplied hints.
        """
        cache_key = "search__" + query.replace(" ", "_")
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.is_file():
            data = json.loads(cache_path.read_text())
        else:
            params = {
                "query": query,
                "format": "json",
                "size": "10",
                "fields": (
                    "accession,reviewed,organism_id,gene_names,protein_name"
                ),
            }
            self._throttle()
            r = self._session.get(
                f"{UNIPROT_BASE}/uniprotkb/search",
                params=params,
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            if cache_path is not None:
                cache_path.write_text(json.dumps(data, indent=2))

        results = data.get("results", [])
        if not results:
            return None

        def is_reviewed(x: dict) -> bool:
            return "reviewed" in x.get("entryType", "").lower()

        def symbol_matches(x: dict) -> bool:
            if not (gene_symbol or locus_tag):
                return False
            for g in x.get("genes", []):
                gn = g.get("geneName", {})
                if isinstance(gn, dict) and gene_symbol:
                    if gn.get("value", "").lower() == gene_symbol.lower():
                        return True
                for oln in g.get("orderedLocusNames", []):
                    if locus_tag and oln.get("value", "").lower() == locus_tag.lower():
                        return True
            return False

        # Preference order: reviewed+symbol_match, reviewed, symbol_match, first
        reviewed_and_match = [
            x for x in results if is_reviewed(x) and symbol_matches(x)
        ]
        if reviewed_and_match:
            return reviewed_and_match[0].get("primaryAccession")
        reviewed = [x for x in results if is_reviewed(x)]
        if reviewed:
            return reviewed[0].get("primaryAccession")
        matches = [x for x in results if symbol_matches(x)]
        if matches:
            return matches[0].get("primaryAccession")
        return results[0].get("primaryAccession")

    def fetch_by_accession(self, accession: str) -> UniProtAnnotation:
        """Fetch and parse the full UniProt entry for an accession."""
        cache_key = f"entry__{accession}"
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.is_file():
            data = json.loads(cache_path.read_text())
        else:
            self._throttle()
            r = self._session.get(
                f"{UNIPROT_BASE}/uniprotkb/{accession}.json",
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            if cache_path is not None:
                cache_path.write_text(json.dumps(data, indent=2))
        return self._parse_entry(data)

    def get_annotation(
        self,
        locus_tag: str,
        organism_taxon: Optional[int] = None,
        gene_symbol: Optional[str] = None,
        taxon_fallbacks: Optional[list[int]] = None,
    ) -> Optional[UniProtAnnotation]:
        """Convenience: search for the gene, then fetch the full annotation.

        Pass gene_symbol when known (UniProt's gene: field matches symbols,
        not ordered-locus-names). Pass taxon_fallbacks to cover the case
        where the caller's taxon id differs from UniProt's (e.g. E. coli
        K-12: NCBI 511145 vs UniProt 83333).
        """
        accession = self.search_by_locus_tag(
            locus_tag,
            organism_taxon,
            gene_symbol=gene_symbol,
            taxon_fallbacks=taxon_fallbacks,
        )
        if accession is None:
            return None
        annotation = self.fetch_by_accession(accession)
        annotation.locus_tag = annotation.locus_tag or locus_tag
        return annotation

    def _parse_entry(self, data: dict) -> UniProtAnnotation:
        """Parse a UniProt JSON entry into a UniProtAnnotation."""
        accession = data["primaryAccession"]

        protein_desc = data.get("proteinDescription", {})
        recommended = protein_desc.get("recommendedName", {})
        if recommended:
            protein_name = (
                recommended.get("fullName", {}).get("value", "") or accession
            )
        else:
            subs = protein_desc.get("submissionNames", [])
            if subs:
                protein_name = subs[0].get("fullName", {}).get("value", accession)
            else:
                protein_name = accession

        gene_name: Optional[str] = None
        locus_tag: Optional[str] = None
        genes = data.get("genes", [])
        if genes:
            g = genes[0]
            gn = g.get("geneName")
            if isinstance(gn, dict):
                gene_name = gn.get("value")
            oln = g.get("orderedLocusNames", [])
            if oln:
                locus_tag = oln[0].get("value")

        organism = data.get("organism", {})
        organism_name = organism.get("scientificName", "")
        organism_taxon = int(organism.get("taxonId", 0))

        seq_obj = data.get("sequence", {})
        sequence = seq_obj.get("value", "")
        sequence_length_aa = int(seq_obj.get("length", len(sequence)))

        function_description = ""
        subcellular_locations: list[str] = []
        for comment in data.get("comments", []):
            ctype = comment.get("commentType", "")
            if ctype == "FUNCTION" and not function_description:
                texts = comment.get("texts", [])
                if texts:
                    function_description = texts[0].get("value", "")
            elif ctype == "SUBCELLULAR LOCATION":
                for loc_obj in comment.get("subcellularLocations", []):
                    loc = loc_obj.get("location", {}).get("value", "")
                    if loc:
                        subcellular_locations.append(loc)

        ec_numbers: list[str] = []
        if recommended:
            for ec in recommended.get("ecNumbers", []):
                v = ec.get("value", "")
                if v:
                    ec_numbers.append(v)

        pfam_domains: list[str] = []
        interpro_ids: list[str] = []
        go_terms: list[GOTerm] = []
        kegg_xrefs: list[str] = []
        eggnog_xrefs: list[str] = []
        pdb_xrefs: list[str] = []
        for xref in data.get("uniProtKBCrossReferences", []):
            db = xref.get("database", "")
            xid = xref.get("id", "")
            if db == "Pfam":
                pfam_domains.append(xid)
            elif db == "InterPro":
                interpro_ids.append(xid)
            elif db == "GO":
                name = ""
                aspect = ""
                for prop in xref.get("properties", []):
                    if prop.get("key") == "GoTerm":
                        val = prop.get("value", "")
                        if ":" in val:
                            aspect, name = val.split(":", 1)
                go_terms.append(GOTerm(go_id=xid, name=name, aspect=aspect))
            elif db == "KEGG":
                kegg_xrefs.append(xid)
            elif db == "eggNOG":
                eggnog_xrefs.append(xid)
            elif db == "PDB":
                pdb_xrefs.append(xid)

        return UniProtAnnotation(
            accession=accession,
            locus_tag=locus_tag,
            protein_name=protein_name,
            gene_name=gene_name,
            organism_name=organism_name,
            organism_taxon=organism_taxon,
            sequence=sequence,
            sequence_length_aa=sequence_length_aa,
            function_description=function_description,
            ec_numbers=ec_numbers,
            pfam_domains=pfam_domains,
            interpro_ids=interpro_ids,
            go_terms=go_terms,
            subcellular_locations=subcellular_locations,
            kegg_xrefs=kegg_xrefs,
            eggnog_xrefs=eggnog_xrefs,
            pdb_xrefs=pdb_xrefs,
            raw_uniprot_url=f"https://www.uniprot.org/uniprotkb/{accession}",
        )


__all__ = ["UniProtRetriever", "UniProtAnnotation", "GOTerm"]
