import sys

from .article import Article
from .readability_extractor import ReadabilityExtractor
from .trafilatura_client import TrafilaturaClient


class Crawler:
    def crawl(self, url: str) -> Article:
        # To help LLMs better understand content, we extract clean
        # articles from HTML, convert them to markdown, and split
        # them into text and image blocks for one single and unified
        # LLM message.
        #
        # To avoid rate limits and external dependencies, use a local
        # Trafilatura fetch to pull the raw HTML before applying our
        # own readability extraction pipeline.
        trafilatura_client = TrafilaturaClient()
        html = trafilatura_client.crawl(url)
        extractor = ReadabilityExtractor()
        article = extractor.extract_article(html)
        article.url = url
        return article


if __name__ == "__main__":
    if len(sys.argv) == 2:
        url = sys.argv[1]
    else:
        url = "https://fintel.io/zh-hant/s/br/nvdc34"
    crawler = Crawler()
    article = crawler.crawl(url)
    print(article.to_markdown())
