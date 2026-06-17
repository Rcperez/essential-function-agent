"""Shared HTTP session factory for REST retrievers.

All REST retrievers in this package (UniProt, KEGG, STRING, and any
future addition) use the same Session-with-Retry-adapter and polite
User-Agent pattern. This module is the single source of truth for that
pattern.

The Accept header is parameterized because UniProt and STRING return
JSON while KEGG returns text/plain.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


USER_AGENT = (
    "essential-function-agent/0.1.0 "
    "(https://github.com/Rcperez/essential-function-agent)"
)


def make_session(accept: str = "application/json") -> requests.Session:
    """Build a requests.Session with retry adapter and polite headers.

    Retries connect, read, and status errors (429 plus 5xx) up to four
    times with exponential backoff. The polite User-Agent identifies the
    project and provides a contact URL per public-API etiquette.

    Args:
        accept: Value for the Accept header. JSON for UniProt and STRING,
            text/plain for KEGG.
    """
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
        "Accept": accept,
    })
    return s


__all__ = ["make_session", "USER_AGENT"]
