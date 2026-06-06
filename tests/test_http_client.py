from __future__ import annotations

from arxiv_astro.http_client import create_http_client, proxy_from_env


def test_proxy_from_env_prefers_https_proxy() -> None:
    proxy = proxy_from_env(
        {
            "HTTP_PROXY": "http://127.0.0.1:10080",
            "HTTPS_PROXY": "http://127.0.0.1:10081",
        }
    )

    assert proxy == "http://127.0.0.1:10081"


def test_proxy_from_env_returns_none_without_proxy() -> None:
    assert proxy_from_env({"NO_PROXY": "[ff00::*]"}) is None


def test_create_http_client_uses_proxy_but_ignores_no_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10081")
    monkeypatch.setenv("NO_PROXY", "[ff00::*]")

    client = create_http_client()

    assert client.trust_env is False
    client.close()
