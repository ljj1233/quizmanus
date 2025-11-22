from __future__ import annotations
import inspect
import logging
import os
from typing import Optional

from curl_cffi import requests

import trafilatura
from .article import Article
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

# 定义环境变量的名称
ENV_COOKIE_NAME='ENV_COOKIE_NAME'

load_dotenv(override=True)
class Crawler:
    def __init__(self, cookie: Optional[str] = None, timeout: int = 30):
        self.timeout = timeout
        
        # 逻辑：优先使用传入的 cookie 参数 -> 其次读取环境变量 -> 如果都没有，直接报错
        self.cookie_header = cookie if cookie is not None else os.getenv(ENV_COOKIE_NAME, "").strip()

        # 严检查：如果 cookie 依然为空，抛出异常阻止程序运行
        if not self.cookie_header or not self.cookie_header.strip():
            error_msg = (
                f"❌ 初始化失败：未提供 Cookie，且环境变量 '{ENV_COOKIE_NAME}' 未设置或为空。\n"
                f"请在终端运行: export {ENV_COOKIE_NAME}='你的cookie值' \n"
                f"或者在 .env 文件中配置。"
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

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
            logger.warning("No content extracted for url: %s", url)
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
        """Download using curl_cffi to bypass TLS fingerprinting."""
        
        # 保持之前的成功配置：chrome110
        impersonate_ver = "chrome110"
        logger.debug(f"Fetching {url} using curl_cffi (impersonate='{impersonate_ver}')")

        headers = {
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.zhihu.com/",
            # 这里确保 Cookie 一定存在，因为 __init__ 已经检查过了
            "Cookie": self.cookie_header
        }
        
        response = requests.get(
            url, 
            impersonate=impersonate_ver, 
            headers=headers,
            timeout=self.timeout
        )
        
        logger.debug("Fetch status %s", response.status_code)
        
        if response.status_code == 403:
            logger.error("403 Forbidden. Cookie 可能过期或被风控。")
            logger.error("Response preview: %s", response.text[:100])
            
        response.raise_for_status()
        
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding or 'utf-8'
            
        return response.text