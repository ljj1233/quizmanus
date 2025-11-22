from __future__ import annotations
import inspect
import logging
import os
from typing import Optional
from urllib.parse import urlparse

try:
    from curl_cffi import requests
except ImportError:
    raise ImportError("请先安装 curl_cffi: pip install curl_cffi")

import trafilatura
from .article import Article

logger = logging.getLogger(__name__)

ENV_COOKIE_NAME='ENV_COOKIE_NAME'


class Crawler:
    # 1. 修改初始化：增加 cookie 参数
    def __init__(self, cookie: Optional[str] = None, timeout: int = 30):
        # 默认从环境变量读取，避免硬编码
        self.cookie = cookie if cookie is not None else os.getenv(ENV_COOKIE_NAME, "").strip()
        self.timeout = timeout

    def crawl(self, url: str) -> Article:
        """Fetch and parse a URL into an Article using trafilatura."""
        logger.info("Starting crawl for: %s", url)

        try:
            downloaded = self._fetch(url)
        except Exception as exc:
            logger.error("Fetch failed for %s: %s", url, exc, exc_info=True)
            return Article(title="Error", url=url, content=f"Failed to fetch content: {exc}")

        if not downloaded:
            logger.error("Empty response for url: %s", url)
            return Article(title="Error", url=url, content="Failed to fetch content: empty response.")

        # Trafilatura 参数
        extract_kwargs = {
            "output_format": "markdown",
            "include_tables": True,
            "include_formatting": True,
            "include_images": True,
            "include_comments": False,
            "url": url,
        }

        extract_params = inspect.signature(trafilatura.extract).parameters
        if "deduplicate_matches" in extract_params:
            extract_kwargs["deduplicate_matches"] = True
        elif "deduplicate" in extract_params:
            extract_kwargs["deduplicate"] = True

        try:
            logger.debug("Starting extraction for %s", url)
            content = trafilatura.extract(downloaded, **extract_kwargs)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", url, exc, exc_info=True)
            return Article(title="Error", url=url, content=f"Failed to extract content: {exc}")

        if not content:
            # 尝试看下是否被反爬拦截页面（通常内容很短）
            logger.warning("No content extracted for url: %s. Raw length: %d", url, len(downloaded))
            content = "No readable content found on this page."

        try:
            metadata = trafilatura.extract_metadata(downloaded)
        except Exception as exc:
            logger.warning("Metadata extraction failed for %s: %s", url, exc, exc_info=True)
            metadata = None
            
        title = "No Title"
        if metadata and metadata.title:
            title = metadata.title

        article = Article(
            title=title,
            url=url,
            content=content,
        )

        logger.info("Successfully crawled: %s (%d markdown chars)", title, len(content))
        return article

    def _fetch(self, url: str) -> str:
        """Download using curl_cffi with specific fingerprint."""
        
        # 2. 修改指纹：使用 chrome110 避免 TLS 握手错误 (OPENSSL_internal)
        impersonate_ver = "chrome110"
        logger.debug(f"Fetching {url} using curl_cffi (impersonate='{impersonate_ver}')")
        
        # 3. 构造 Headers：加入 Cookie 和 Referer
        headers = {
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            # 知乎非常看重 Referer，必须加上
            "Referer": "https://www.zhihu.com/", 
        }
        
        # 如果初始化时传入了 cookie，则添加到 headers 中
        if self.cookie:
            headers["Cookie"] = self.cookie
            logger.debug("Sending Cookie header (%s)", ENV_COOKIE_NAME)

        response = requests.get(
            url, 
            impersonate=impersonate_ver, 
            headers=headers,
            timeout=self.timeout
        )
        
        logger.debug("Fetch status %s", response.status_code)
        
        # 如果遇到 403，打印一下返回内容，看看是不是验证码页面
        if response.status_code == 403:
            logger.error("403 Forbidden. Body preview: %s", response.text[:200])
            
        response.raise_for_status()
        
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding or 'utf-8'
            
        return response.text
