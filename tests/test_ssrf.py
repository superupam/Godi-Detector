"""Tests for the scraper's SSRF guard and the redirect/streaming behaviour of
_safe_get.

Everything here is offline: literal-IP cases never resolve, and hostname cases
monkeypatch the resolver.
"""

import pytest

import scraper
from scraper import UnsafeURLError, _safe_get, validate_public_url


BLOCKED = scraper.BLOCKED_URL_MESSAGE
INVALID = "Invalid URL. Please provide a link starting with http:// or https://"


# --- Literal addresses the guard must refuse -------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8000/api/status",
        "https://localhost/",
        "http://0.0.0.0/",
        "http://10.1.2.3/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",          # carrier-grade NAT
        "http://192.0.2.10/",          # documentation range
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "http://[64:ff9b::a9fe:a9fe]/",  # NAT64-wrapped 169.254.169.254
        "http://[2002:7f00:1::]/",       # 6to4-wrapped 127.0.0.1
        "http://2130706433/",          # decimal 127.0.0.1
        "http://0x7f.1/",              # hex 127.0.0.1
        "http://0177.0.0.1/",          # octal 127.0.0.1
        "http://127.1/",               # short form 127.0.0.1
    ],
)
def test_internal_literal_addresses_are_blocked(url):
    assert validate_public_url(url) == BLOCKED


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",
        "http://LOCALHOST./",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata/",
        "http://instance-data/",
        "http://printer.local/",
        "http://service.internal/",
        "http://router.home.arpa/",
    ],
)
def test_internal_hostnames_are_blocked_without_resolving(url):
    assert validate_public_url(url) == BLOCKED


# --- Public destinations --------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://93.184.216.34/",
        "https://8.8.8.8/",
        "http://[2606:4700::1111]/",
    ],
)
def test_public_literal_addresses_are_allowed(url):
    assert validate_public_url(url) is None


# --- Schemes and malformed input ------------------------------------------


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", "example.com"])
def test_non_http_schemes_are_rejected(url):
    assert validate_public_url(url) in (INVALID, "Only http:// and https:// links can be scraped.")


@pytest.mark.parametrize("url", ["", "   "])
def test_empty_input_is_rejected_as_not_http(url):
    assert validate_public_url(url) == "Only http:// and https:// links can be scraped."


@pytest.mark.parametrize("url", ["http://", "https:///path"])
def test_missing_host_is_rejected(url):
    assert validate_public_url(url) == INVALID


def test_bad_port_is_rejected():
    assert validate_public_url("http://example.com:notaport/") == "Invalid port in URL."


# --- Hostname resolution -------------------------------------------------


def _resolver(*addresses):
    """Fake socket.getaddrinfo returning one SOCK_STREAM tuple per address."""
    def fake_getaddrinfo(host, port, **kwargs):
        return [(2, 1, 6, "", (address, port)) for address in addresses]
    return fake_getaddrinfo


def test_hostname_resolving_to_a_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(scraper.socket, "getaddrinfo", _resolver("93.184.216.34"))
    assert validate_public_url("https://www.bbc.com/news") is None


def test_mixed_public_and_private_answers_are_blocked(monkeypatch):
    monkeypatch.setattr(scraper.socket, "getaddrinfo", _resolver("93.184.216.34", "10.0.0.7"))
    assert validate_public_url("https://evil.example.com/") == BLOCKED


def test_ipv6_only_private_answer_is_blocked(monkeypatch):
    monkeypatch.setattr(scraper.socket, "getaddrinfo", _resolver("::1"))
    assert validate_public_url("https://evil.example.com/") == BLOCKED


def test_unresolvable_host_is_reported(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr(scraper.socket, "getaddrinfo", boom)
    assert validate_public_url("https://does-not-exist.example/") == (
        "Could not resolve the host 'does-not-exist.example'."
    )


def test_empty_resolution_is_reported(monkeypatch):
    monkeypatch.setattr(scraper.socket, "getaddrinfo", _resolver())
    assert validate_public_url("https://empty.example/") == "Could not resolve the host 'empty.example'."


# --- _safe_get: redirects are re-validated and bodies are capped ----------


class FakeResponse:
    def __init__(self, status_code=200, headers=None, chunks=(), encoding="utf-8"):
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = encoding
        self._chunks = list(chunks)
        self.closed = False

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, chunk_size=1024):
        return iter(self._chunks)

    def close(self):
        self.closed = True


def test_redirect_to_a_private_address_is_blocked_before_fetching_it(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(302, {"Location": "http://127.0.0.1/admin"})

    monkeypatch.setattr(scraper.requests, "get", fake_get)

    with pytest.raises(UnsafeURLError) as excinfo:
        _safe_get("http://93.184.216.34/start", timeout=1)

    assert str(excinfo.value) == BLOCKED
    assert calls == ["http://93.184.216.34/start"]


def test_redirect_loop_gives_up(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(302, {"Location": "/again"})

    monkeypatch.setattr(scraper.requests, "get", fake_get)

    with pytest.raises(UnsafeURLError) as excinfo:
        _safe_get("http://93.184.216.34/start", timeout=1)

    assert str(excinfo.value) == "Too many redirects."
    assert len(calls) == scraper.MAX_REDIRECTS + 1


def test_response_body_is_truncated_to_the_byte_cap(monkeypatch):
    body = b"a" * 400 + b"b" * 400

    def fake_get(url, **kwargs):
        return FakeResponse(200, chunks=[body[:300], body[300:600], body[600:]])

    monkeypatch.setattr(scraper.requests, "get", fake_get)

    response, page_text = _safe_get("http://93.184.216.34/", timeout=1, max_bytes=500)
    assert response.status_code == 200
    assert page_text == "a" * 400 + "b" * 100


def test_non_200_body_is_not_read(monkeypatch):
    class ExplodingBody(FakeResponse):
        def iter_content(self, chunk_size=1024):
            raise AssertionError("body should not be read for a non-200 response")

    def fake_get(url, **kwargs):
        return ExplodingBody(403)

    monkeypatch.setattr(scraper.requests, "get", fake_get)

    response, page_text = _safe_get("http://93.184.216.34/", timeout=1)
    assert response.status_code == 403
    assert page_text == ""


def test_unknown_charset_falls_back_to_utf8(monkeypatch):
    def fake_get(url, **kwargs):
        return FakeResponse(200, chunks=["café".encode("utf-8")], encoding="x-not-a-codec")

    monkeypatch.setattr(scraper.requests, "get", fake_get)

    _, page_text = _safe_get("http://93.184.216.34/", timeout=1)
    assert page_text == "café"
