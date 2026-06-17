"""KEGG REST API retriever for the pathway/orthology channel.

Resolves (organism_code, locus_tag) to a KEGG gene entry and parses the
flat-text response for orthology (KO) assignment, pathway membership,
Pfam motifs, amino-acid sequence, and cross-references to other databases.

KEGG REST documentation: https://www.kegg.jp/kegg/rest/keggapi.html

Modern KEGG flat-text format (observed 2026-06) uses SYMBOL for the gene
symbol and NAME for the RefSeq description (prefixed "(RefSeq) ...").
Legacy entries use NAME for the gene symbol and DEFINITION for the
description. The parser handles both.

TODO: extract _make_session to a shared src/efa/tools/_http.py once a
third REST retriever (STRING) is added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


KEGG_BASE = "https://rest.kegg.jp"
DEFAULT_TIMEOUT_S = 30
DEFAULT_RATE_LIMIT_S = 0.5
DEFAULT_CACHE_DIR = Path(
    "/content/drive/MyDrive/RP_RTP_Repo_Bundles/"
    "essential-function-agent/cache/kegg"
)
USER_AGENT = (
    "essential-function-agent/0.1.0 "
    "(https://github.com/Rcperez/essential-function-agent)"
)
REFSEQ_PREFIX = "(RefSeq)"


@dataclass
class KEGGPathway:
    """A KEGG pathway the gene participates in."""

    pathway_id: str
    name: str


@dataclass
class KEGGOrthology:
    """KEGG Orthology (KO) assignment for a gene."""

    ko_id: str
    description: str


@dataclass
class KEGGGeneAnnotation:
    """Structured KEGG gene annotation."""

    kegg_gene_id: str
    organism_code: str
    locus_tag: str
    gene_name: Optional[str]
    definition: str
    orthologies: list[KEGGOrthology]
    pathways: list[KEGGPathway]
    motif_pfam: list[str]
    aa_sequence: str
    aa_length: int
    db_links: dict[str, list[str]]
    raw_kegg_url: str


def _make_session() -> requests.Session:
    """Build a requests.Session with retry adapter and polite headers."""
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/plain",
    })
    return s


class KEGGRetriever:
    """Retriever for KEGG gene annotations.

    Caches raw flat-text responses to disk and re-parses on each call.
    Throttles requests to one every rate_limit_s seconds per KEGG's
    fair-use guidance.
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
        self._session = _make_session()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_t
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_t = time.monotonic()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.txt"

    def fetch_gene(
        self,
        organism_code: str,
        locus_tag: str,
    ) -> Optional[KEGGGeneAnnotation]:
        """Fetch and parse a KEGG gene entry.

        Returns None if KEGG returns HTTP 404 or an empty response.
        """
        kegg_id = f"{organism_code}:{locus_tag}"
        cache_path = self._cache_path(f"gene__{kegg_id}")
        if cache_path is not None and cache_path.is_file():
            text = cache_path.read_text()
        else:
            self._throttle()
            r = self._session.get(
                f"{KEGG_BASE}/get/{kegg_id}",
                timeout=self.timeout_s,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            text = r.text
            if not text.strip():
                return None
            if cache_path is not None:
                cache_path.write_text(text)
        return self._parse_entry(text, organism_code, locus_tag)

    @staticmethod
    def _parse_flat_text(text: str) -> dict[str, list[str]]:
        """Parse KEGG flat text into a dict of section -> list of value lines."""
        sections: dict[str, list[str]] = {}
        current: Optional[str] = None
        for line in text.splitlines():
            if line.startswith("///"):
                break
            if not line.strip():
                continue
            if line[0].isalpha():
                parts = line.split(None, 1)
                current = parts[0]
                sections.setdefault(current, [])
                if len(parts) == 2:
                    sections[current].append(parts[1])
            else:
                if current is not None:
                    sections.setdefault(current, []).append(line.strip())
        return sections

    @staticmethod
    def _extract_gene_symbol(
        sections: dict[str, list[str]],
    ) -> Optional[str]:
        """Extract gene symbol.

        Modern KEGG: SYMBOL field contains comma-separated gene symbols.
        Legacy KEGG: NAME field contains symbols (only when NAME does not
        start with the "(RefSeq)" prefix that marks it as a description).
        """
        sym_lines = sections.get("SYMBOL", [])
        if sym_lines:
            first = sym_lines[0].split(",")[0].strip()
            if first:
                return first
        name_lines = sections.get("NAME", [])
        if name_lines:
            first_line = name_lines[0].strip()
            if not first_line.startswith(REFSEQ_PREFIX):
                first = first_line.split(",")[0].strip()
                if first:
                    return first
        return None

    @staticmethod
    def _extract_description(sections: dict[str, list[str]]) -> str:
        """Extract function description.

        Legacy KEGG: DEFINITION holds the description (prefixed "(RefSeq) ").
        Modern KEGG: NAME holds the description prefixed "(RefSeq) ".
        Either way, strip the prefix if present.
        """
        def_lines = sections.get("DEFINITION", [])
        if def_lines:
            joined = " ".join(def_lines).strip()
            if joined.startswith(REFSEQ_PREFIX):
                joined = joined[len(REFSEQ_PREFIX):].strip()
            if joined:
                return joined
        name_lines = sections.get("NAME", [])
        if name_lines:
            first_line = name_lines[0].strip()
            if first_line.startswith(REFSEQ_PREFIX):
                return first_line[len(REFSEQ_PREFIX):].strip()
        return ""

    def _parse_entry(
        self,
        text: str,
        organism_code: str,
        locus_tag: str,
    ) -> KEGGGeneAnnotation:
        """Parse KEGG flat-text entry into a KEGGGeneAnnotation."""
        sections = self._parse_flat_text(text)

        gene_name = self._extract_gene_symbol(sections)
        definition = self._extract_description(sections)

        orthologies: list[KEGGOrthology] = []
        for line in sections.get("ORTHOLOGY", []):
            parts = line.split(None, 1)
            if len(parts) == 2:
                orthologies.append(
                    KEGGOrthology(ko_id=parts[0], description=parts[1].strip())
                )

        pathways: list[KEGGPathway] = []
        for line in sections.get("PATHWAY", []):
            parts = line.split(None, 1)
            if len(parts) == 2:
                pathways.append(
                    KEGGPathway(pathway_id=parts[0], name=parts[1].strip())
                )

        motif_pfam: list[str] = []
        for line in sections.get("MOTIF", []):
            if line.startswith("Pfam:"):
                pfams = line[len("Pfam:"):].strip().split()
                motif_pfam.extend(pfams)

        aaseq_lines = sections.get("AASEQ", [])
        aa_length = 0
        aa_sequence = ""
        if aaseq_lines:
            try:
                aa_length = int(aaseq_lines[0])
            except (ValueError, IndexError):
                aa_length = 0
            aa_sequence = "".join(
                ln.replace(" ", "") for ln in aaseq_lines[1:]
            )

        db_links: dict[str, list[str]] = {}
        for line in sections.get("DBLINKS", []):
            if ":" in line:
                key, val = line.split(":", 1)
                ids = val.strip().split()
                db_links.setdefault(key.strip(), []).extend(ids)

        return KEGGGeneAnnotation(
            kegg_gene_id=f"{organism_code}:{locus_tag}",
            organism_code=organism_code,
            locus_tag=locus_tag,
            gene_name=gene_name,
            definition=definition,
            orthologies=orthologies,
            pathways=pathways,
            motif_pfam=motif_pfam,
            aa_sequence=aa_sequence,
            aa_length=aa_length,
            db_links=db_links,
            raw_kegg_url=f"https://www.kegg.jp/entry/{organism_code}:{locus_tag}",
        )


__all__ = [
    "KEGGRetriever",
    "KEGGGeneAnnotation",
    "KEGGPathway",
    "KEGGOrthology",
]
