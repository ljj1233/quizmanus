import importlib
import sys
import types

import pytest


def _reload_client_with_fetch(fake_fetch):
    fake_module = types.SimpleNamespace(fetch_url=fake_fetch)
    sys.modules["trafilatura"] = fake_module
    import src.graph.crawler.trafilatura_client as client_module

    importlib.reload(client_module)
    return client_module


def test_trafilatura_client_returns_html(monkeypatch):
    client_module = _reload_client_with_fetch(lambda url: "<html>ok</html>")

    client = client_module.TrafilaturaClient()
    html = client.crawl("http://example.com")

    assert html == "<html>ok</html>"


def test_trafilatura_client_raises_on_missing(monkeypatch):
    client_module = _reload_client_with_fetch(lambda url: None)
    client = client_module.TrafilaturaClient()

    with pytest.raises(ValueError):
        client.crawl("http://invalid")
