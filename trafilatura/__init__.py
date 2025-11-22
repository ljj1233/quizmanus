"""Minimal fetch-only subset of Trafilatura for offline environments."""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def fetch_url(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except requests.RequestException as exc:  # pragma: no cover - network may be blocked
        logger.error("Failed to fetch %s via fallback fetch_url: %s", url, exc)
        return None
