from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


def _load_all_keys() -> Optional[Any]:
    """Attempt to import ALL_KEYS without polluting sys.path."""

    try:
        import ALL_KEYS  # type: ignore
    except Exception:
        return None
    return ALL_KEYS


_ALL_KEYS = _load_all_keys()


def _get_from_all_keys(attr: str) -> str:
    if _ALL_KEYS is None:
        return ""
    value = getattr(_ALL_KEYS, attr, "")
    return str(value) if value else ""


def env_or_all_keys(env_var: str, all_keys_attr: str | None = None, default: str = "") -> str:
    """Prefer environment variables, fall back to optional ALL_KEYS attributes."""

    env_value = os.getenv(env_var, "")
    if env_value:
        return env_value
    if all_keys_attr:
        fallback = _get_from_all_keys(all_keys_attr)
        if fallback:
            return fallback
    return default


def get_path_from_env(env_var: str, default: str = "") -> str:
    """Read a filesystem path from environment variables with a safe default."""

    env_value = os.getenv(env_var, "")
    if env_value:
        return env_value
    return default


@dataclass(frozen=True)
class ApiSettings:
    base_url: str
    api_key: str
    authorization: str = ""


@lru_cache(maxsize=1)
def get_hkust_settings() -> ApiSettings:
    return ApiSettings(
        base_url=env_or_all_keys("HKUST_OPENAI_BASE_URL", "hkust_openai_base_url"),
        api_key=env_or_all_keys("HKUST_OPENAI_KEY", "hkust_openai_key"),
        authorization=env_or_all_keys("HKUST_AUTHORIZATION_KEY", "Authorization_hkust_key"),
    )


@lru_cache(maxsize=1)
def get_common_openai_settings() -> ApiSettings:
    return ApiSettings(
        base_url=env_or_all_keys("COMMON_OPENAI_BASE_URL", "common_openai_base_url"),
        api_key=env_or_all_keys("COMMON_OPENAI_KEY", "common_openai_key"),
    )
