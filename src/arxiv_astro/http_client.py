from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx


PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def proxy_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    env = environ or os.environ
    for key in PROXY_ENV_KEYS:
        value = env.get(key)
        if value:
            return value
    return None


def create_http_client(**kwargs: Any) -> httpx.Client:
    proxy = proxy_from_env()
    if proxy:
        kwargs.setdefault("proxy", proxy)
    kwargs["trust_env"] = False
    return httpx.Client(**kwargs)
