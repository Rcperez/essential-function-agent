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
    ) -> Optional[str]:
        """Search UniProt for a locus tag; return preferred accession or None.

        Prefers reviewed (Swiss-Prot) entries over unreviewed (TrEMBL).
        """
        taxon_part = str(organism_taxon) if organism_taxon is not None else "any"
        cache_key = f"search__locus_{locus_tag}__taxon_{taxon_part}"
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.is_file():
            data = json.loads(cache_path.read_text())
        else:
            query = f"gene:{locus_tag}"
            if organism_taxon is not None:
                query = f"({query}) AND (organism_id:{organism_taxon})"
            params = {
                "query": query,
                "format": "json",
                "size": "5",
                "fields": "accession,reviewed,organism_id,gene_names,protein_name",
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
        reviewed = [
            x for x in results
            if "reviewed" in x.get("entryType", "").lower()
        ]
        chosen = reviewed[0] if reviewed else results[0]
        return chosen.get("primaryAccession")

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
    ) -> Optional[UniProtAnnotation]:
        """Convenience: search by locus tag, then fetch the full annotation."""
        accession = self.search_by_locus_tag(locus_tag, organism_taxon)
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
