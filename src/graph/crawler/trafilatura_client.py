import logging
from typing import Optional

import trafilatura

logger = logging.getLogger(__name__)


class TrafilaturaClient:
    def crawl(self, url: str) -> str:
        """Fetch raw HTML using Trafilatura.

        Trafilatura internally handles polite fetching and content
        normalization; we only need the downloaded HTML to pass into our
        readability extractor. A missing response triggers a clear error so
        callers can fall back or alert users without silent failures.
        """

        downloaded: Optional[str] = trafilatura.fetch_url(url)
        if not downloaded:
            logger.error("Trafilatura failed to fetch content from %s", url)
            raise ValueError(f"Unable to fetch content from {url}")
        return downloaded
