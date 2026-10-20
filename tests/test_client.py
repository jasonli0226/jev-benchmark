import httpx
import pytest

from jev_bench.jev import client as client_mod
from jev_bench.jev.client import JevClient, JevError


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(client_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(client_mod, "BACKOFF_BASE_S", 0.0)

    def factory(handler):
        c = JevClient(concurrency=2)
        c._http = httpx.Client(transport=httpx.MockTransport(handler))
        return c

    return factory


OK = {"answers": {"answer": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 10, "cost": 4.2e-7}}


def test_retries_rate_limit_then_succeeds(make_client):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {}}) if len(calls) < 3 else httpx.Response(200, json=OK)

    res = make_client(handler).decide({"q": 1})
    assert res.attempts == 3 and res.body == OK and not res.cached


def test_auth_error_is_not_retried(make_client):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "API key expired."}})

    with pytest.raises(JevError, match="401"):
        make_client(handler).decide({"q": 1})
    assert len(calls) == 1


def test_cache_hit_skips_network_but_new_repeat_does_not(make_client):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=OK)

    c = make_client(handler)
    c.decide({"q": 1})
    assert c.decide({"q": 1}).cached
    c.decide({"q": 1}, repeat=1)
    assert len(calls) == 2


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JevError, match="OPENROUTER_API_KEY"):
        JevClient()
