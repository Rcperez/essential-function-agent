"""STRING REST API retriever for the protein-protein interaction channel.

Fetches a protein's STRING-curated interaction partners with channel-
decomposed confidence scores (neighborhood, fusion, cooccurrence,
coexpression, experimental, database, textmining, and combined).

STRING REST documentation: https://string-db.org/help/api/

STRING returns scores as floats 0-1; this module converts them to ints
0-1000 to match STRING's display convention.

TODO: extract _make_session to a shared src/efa/tools/_http.py. Currently
duplicated across uniprot.py, kegg.py, and this module (three copies).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STRING_BASE = "https://string-db.org/api"
DEFAULT_TIMEOUT_S = 30
DEFAULT_RATE_LIMIT_S = 1.0
DEFAULT_CACHE_DIR = Path(
    "/content/drive/MyDrive/RP_RTP_Repo_Bundles/"
    "essential-function-agent/cache/string"
)
USER_AGENT = (
    "essential-function-agent/0.1.0 "
    "(https://github.com/Rcperez/essential-function-agent)"
)


@dataclass
class STRINGInteraction:
    """A single protein-protein interaction edge from STRING.

    All scores are integers on STRING's 0-1000 display scale (the API
    returns floats 0-1; this class stores the rounded 0-1000 form).
    Higher = more confident. 0 in a channel-specific score means
    "no evidence in this channel".
    """

    partner_string_id: str
    partner_preferred_name: str
    combined_score: int
    neighborhood_score: int
    fusion_score: int
    cooccurrence_score: int
    coexpression_score: int
    experimental_score: int
    database_score: int
    textmining_score: int


@dataclass
class STRINGNetwork:
    """A protein's STRING interaction network."""

    query_identifier: str
    species_taxon: int
    interactions: list[STRINGInteraction]
    raw_string_url: str


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
        "Accept": "application/json",
    })
    return s


class STRINGRetriever:
    """Retriever for STRING protein-protein interaction networks.

    Caches raw JSON responses to disk and re-parses on each call.
    Throttles requests to one every rate_limit_s seconds per STRING's
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
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.json"

    def fetch_interaction_partners(
        self,
        identifier: str,
        species_taxon: int,
        limit: int = 20,
        required_score: int = 400,
    ) -> Optional[STRINGNetwork]:
        """Fetch interaction partners for a protein from STRING.

        Args:
            identifier: gene name, UniProt accession, or STRING ID
            species_taxon: NCBI taxon ID (STRING species code matches)
            limit: max partners to return (1-1000)
            required_score: minimum combined score (0-1000); 400 = medium
                confidence per STRING convention, 700 = high, 900 = highest

        Returns:
            STRINGNetwork (possibly with empty interactions) or None if
            STRING could not resolve the identifier (400 or 404 response).
        """
        cache_key = (
            f"partners__{identifier}__sp{species_taxon}__"
            f"lim{limit}__rs{required_score}"
        )
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.is_file():
            data = json.loads(cache_path.read_text())
            if data is None:
                return None
        else:
            params = {
                "identifiers": identifier,
                "species": str(species_taxon),
                "limit": str(limit),
                "required_score": str(required_score),
                "caller_identity": "essential-function-agent",
            }
            self._throttle()
            r = self._session.get(
                f"{STRING_BASE}/json/interaction_partners",
                params=params,
                timeout=self.timeout_s,
            )
            if r.status_code in (400, 404):
                if cache_path is not None:
                    cache_path.write_text("null")
                return None
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return None
            if cache_path is not None:
                cache_path.write_text(json.dumps(data, indent=2))

        return self._parse_partners(data, identifier, species_taxon)

    def _parse_partners(
        self,
        data: list[dict],
        query_identifier: str,
        species_taxon: int,
    ) -> STRINGNetwork:
        """Parse a STRING interaction_partners response."""
        interactions: list[STRINGInteraction] = []
        for row in data:
            interactions.append(STRINGInteraction(
                partner_string_id=row.get("stringId_B", ""),
                partner_preferred_name=row.get("preferredName_B", ""),
                combined_score=self._to_int_score(row.get("score", 0)),
                neighborhood_score=self._to_int_score(row.get("nscore", 0)),
                fusion_score=self._to_int_score(row.get("fscore", 0)),
                cooccurrence_score=self._to_int_score(row.get("pscore", 0)),
                coexpression_score=self._to_int_score(row.get("ascore", 0)),
                experimental_score=self._to_int_score(row.get("escore", 0)),
                database_score=self._to_int_score(row.get("dscore", 0)),
                textmining_score=self._to_int_score(row.get("tscore", 0)),
            ))
        return STRINGNetwork(
            query_identifier=query_identifier,
            species_taxon=species_taxon,
            interactions=interactions,
            raw_string_url=(
                f"https://string-db.org/cgi/network?identifiers="
                f"{query_identifier}&species={species_taxon}"
            ),
        )

    @staticmethod
    def _to_int_score(v: Any) -> int:
        """Convert STRING float score (0-1) to int on 0-1000 scale."""
        try:
            return int(round(float(v) * 1000))
        except (TypeError, ValueError):
            return 0


__all__ = ["STRINGRetriever", "STRINGNetwork", "STRINGInteraction"]
