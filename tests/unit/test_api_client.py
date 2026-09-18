import pytest

from football_predictor.data.providers.base import ApiClient, ProviderError


def test_get_returns_json():
    client = ApiClient(
        base_url="https://example.test",
        api_key="sekret",
        transport=__import__("httpx").MockTransport(
            lambda r: __import__("httpx").Response(200, json={"ok": 1})
        ),
    )
    assert client.get("/x") == {"ok": 1}


def test_sends_bearer_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return __import__("httpx").Response(200, json={})

    client = ApiClient(
        base_url="https://example.test",
        api_key="sekret",
        transport=__import__("httpx").MockTransport(handler),
    )
    client.get("/x")
    assert seen["auth"] == "Bearer sekret"


def test_sends_raw_header_when_no_scheme():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("X-Auth-Token")
        return __import__("httpx").Response(200, json={})

    client = ApiClient(
        base_url="https://example.test",
        api_key="sekret",
        auth_header="X-Auth-Token",
        auth_scheme="",
        transport=__import__("httpx").MockTransport(handler),
    )
    client.get("/x")
    assert seen["auth"] == "sekret"


def test_extra_headers_always_sent():
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("x-rapidapi-key")
        return __import__("httpx").Response(200, json={})

    client = ApiClient(
        base_url="https://example.test",
        extra_headers={"x-rapidapi-key": "rk", "x-rapidapi-host": "h"},
        transport=__import__("httpx").MockTransport(handler),
    )
    client.get("/x")
    assert seen["key"] == "rk"


def test_non_2xx_raises_provider_error():
    def handler(request):
        return __import__("httpx").Response(403, text="quota exceeded")

    client = ApiClient(
        base_url="https://example.test",
        transport=__import__("httpx").MockTransport(handler),
    )
    with pytest.raises(ProviderError, match="credencial inválida"):
        client.get("/x")


def test_429_reports_rate_limit():
    def handler(request):
        return __import__("httpx").Response(429, text="too many")

    client = ApiClient(
        base_url="https://example.test",
        transport=__import__("httpx").MockTransport(handler),
    )
    with pytest.raises(ProviderError, match="rate limit"):
        client.get("/x")


def test_bad_json_raises():
    def handler(request):
        return __import__("httpx").Response(200, text="<html>not json")

    client = ApiClient(
        base_url="https://example.test",
        transport=__import__("httpx").MockTransport(handler),
    )
    with pytest.raises(ProviderError, match="no-JSON"):
        client.get("/x")