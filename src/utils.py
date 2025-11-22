from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import httpx
import json_repair
import requests
from openai import OpenAI
from tqdm import tqdm

from src.config.env import get_common_openai_settings, get_hkust_settings

logger = logging.getLogger(__name__)


def getData(path: str) -> list | str:
    """Load structured text data from supported file types."""

    if not os.path.exists(path):
        return []

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as file:
            data = [line.strip() for line in file]
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in tqdm(file) if line.strip()]
    elif path.endswith(".md"):
        md_file = Path(path)
        data = md_file.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return data


def saveData(data: list, path: str) -> None:
    """Persist structured data to json/jsonl/txt files."""

    if path.endswith("json"):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    elif path.endswith("jsonl"):
        with open(path, "w", encoding="utf-8") as file:
            if isinstance(data, (list, dict)):
                for item in data if isinstance(data, list) else data.values():
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif path.endswith("txt"):
        with open(path, "w", encoding="utf-8") as file:
            for item in data:
                file.write(f"{item}\n")
    else:
        raise ValueError(f"Unsupported file type: {path}")


def removeDuplicates(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate items by id while keeping the last occurrence."""

    ids = []
    for item in data:
        if item["id"] not in ids:
            ids.append(item["id"])
    tmp_dict: Dict[str, Dict[str, Any]] = {}
    result_list: List[Dict[str, Any]] = []
    for item in data:
        tmp_dict[item["id"]] = item
    for id_value in ids:
        result_list.append(tmp_dict[id_value])
    return result_list


def getHkustClient(api_type: str = "DeepSeek-R1-671B") -> OpenAI:
    """Build an OpenAI-compatible client for HKUST endpoints."""

    settings = get_hkust_settings()
    if not settings.base_url or not settings.api_key:
        raise ValueError("HKUST_OPENAI_BASE_URL and HKUST_OPENAI_KEY must be configured.")
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        http_client=httpx.Client(
            base_url=settings.base_url,
            follow_redirects=True,
        ),
    )
    return client


def call_Hkust_api(
    prompt: str,
    messages: Sequence[Dict[str, str]] | None = None,
    remain_reasoning: bool = False,
    api_type: str = "DeepSeek-R1-671B",
    config: Dict[str, Any] | None = None,
) -> str:
    payload_messages = [{"role": "user", "content": prompt}] if not messages else list(messages)
    merged_config: Dict[str, Any] = {"temperature": 0.7, **(config or {})}
    settings = get_hkust_settings()
    if not settings.base_url or not settings.api_key:
        logger.error("HKUST OpenAI endpoint is not configured.")
        return ""
    try:
        response = requests.post(
            settings.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.authorization or settings.api_key}",
            },
            data=json.dumps(
                {
                    "model": api_type,
                    "messages": payload_messages,
                    **merged_config,
                }
            ),
            timeout=30,
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not remain_reasoning:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as exc:
        logger.error("HKUST API call failed: %s", exc)
        return ""


def getClient(api_type: str = "gpt-4o-mini") -> OpenAI:
    """Build an OpenAI client using shared COMMON settings."""

    settings = get_common_openai_settings()
    if not settings.base_url or not settings.api_key:
        raise ValueError("COMMON_OPENAI_BASE_URL and COMMON_OPENAI_KEY must be configured.")
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        http_client=httpx.Client(
            base_url=settings.base_url,
            follow_redirects=True,
        ),
    )
    return client


def call_api(prompt: str, api_type: str = "gpt-4o-mini") -> object | str:
    """Simple wrapper to invoke chat completions."""

    try:
        response = getClient(api_type).chat.completions.create(
            model=api_type,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response
    except Exception as exc:
        logger.error("OpenAI API call failed: %s", exc)
        return ""


def get_list_text(text: str) -> str | int:
    match = re.search(r"\[(.*?)\]", text)
    if match:
        full_match = match.group(0)
        return full_match
    return -1


def get_json_text(text: str) -> str | int:
    match = re.search(r"\{(.*?)\}", text)
    if match:
        full_match = match.group(0)
        return full_match
    return -1


def get_absolute_file_paths(absolute_dir: str, file_type: str) -> List[str]:
    """
    absolute_dir: 文件夹
    file_type: "md","json"...
    """

    json_files = [os.path.join(absolute_dir, f) for f in os.listdir(absolute_dir) if f.endswith(f".{file_type}")]
    return json_files


def get_json_result(text: str) -> Any:
    parsed_response = json_repair.loads(text)
    if isinstance(parsed_response, list):
        parsed_response = parsed_response[-1]
    return parsed_response
