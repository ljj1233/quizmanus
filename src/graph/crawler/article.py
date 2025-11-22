import re
from typing import Optional
from urllib.parse import urljoin

from markdownify import markdownify as md


class Article:
    """Lightweight container for crawled page content."""

    def __init__(self, title: str, url: str = "", content: Optional[str] = None, html_content: Optional[str] = None):
        if content is None and html_content is None:
            raise ValueError("Either markdown content or html content must be provided.")

        self.title = title
        self.url = url
        self.raw_html = html_content
        # Prefer explicit markdown content; otherwise convert provided HTML.
        self.content = content if content is not None else md(html_content)

    def to_markdown(self, including_title: bool = True) -> str:
        markdown_parts = []
        if including_title:
            markdown_parts.append(f"# {self.title}")
        markdown_parts.append(self.content)
        return "\n\n".join(markdown_parts).strip()

    def to_message(self) -> list[dict]:
        image_pattern = r"!\[.*?\]\((.*?)\)"

        content: list[dict[str, str]] = []
        parts = re.split(image_pattern, self.to_markdown())

        for i, part in enumerate(parts):
            if i % 2 == 1:
                image_url = urljoin(self.url, part.strip())
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                content.append({"type": "text", "text": part.strip()})

        return content
