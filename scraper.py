# ==========================================
# GODI DETECTOR
# Web News Link Scraper
# Extracts article headline, body, and metadata
# ==========================================

import ipaddress
import re
import socket
import urllib.parse

import requests
from bs4 import BeautifulSoup

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1"
}


# --- SSRF protection -------------------------------------------------------
# /api/scrape fetches a URL chosen by the caller, so the destination has to be
# checked before every request. Without this the endpoint can be pointed at
# localhost, private networks, or cloud metadata services on 169.254.169.254.

MAX_REDIRECTS = 5

# A news article needs a few hundred KB at most. Stop reading after 5 MB so a
# hostile or accidental giant response cannot exhaust the server's memory.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

BLOCKED_URL_MESSAGE = (
    "That URL points to a local, private, or internal address. "
    "Only public news sites can be scraped."
)

# Internal hostnames that would otherwise look like ordinary names.
BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
    "instance-data.ec2.internal",
}
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")


class UnsafeURLError(ValueError):
    """Raised when a URL resolves to a destination we refuse to fetch."""


# IPv6 forms that are routable but mean "reach this IPv4 address", so the
# address inside them is the one that has to be checked.
NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
SIX_TO_FOUR_PREFIX = ipaddress.ip_network("2002::/16")


def _embedded_ipv4(ip):
    """Return the IPv4 address carried inside an IPv6 wrapper, if any."""
    if ip.version != 6:
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in NAT64_PREFIX:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in SIX_TO_FOUR_PREFIX:
        return ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
    return None


def _is_public_ip(ip) -> bool:
    """True only for globally reachable unicast addresses."""
    while (inner := _embedded_ipv4(ip)) is not None:
        ip = inner
    # is_global rules out reserved and test ranges too, but not 100.64/10
    # (carrier-grade NAT), so the explicit checks stay as a backstop.
    return ip.is_global and not (
        ip.is_private        # 10/8, 172.16/12, 192.168/16, fc00::/7
        or ip.is_loopback
        or ip.is_link_local  # 169.254/16 — cloud metadata lives here
        or ip.is_multicast
        or ip.is_unspecified
    )


def _host_literal_ip(host: str):
    """
    Parse the host as an IP address when it is one, including the shorthand
    forms a resolver accepts but ipaddress does not: 2130706433, 0x7f.1,
    0177.0.0.1 all mean 127.0.0.1.
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        return ipaddress.ip_address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def validate_public_url(url: str):
    """
    Check that a URL may be fetched.

    Returns:
        None when the destination is a public http(s) host, otherwise a
        human-readable reason explaining why it was refused.
    """
    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return "Invalid URL. Please provide a link starting with http:// or https://"

    if parsed.scheme not in ("http", "https"):
        return "Only http:// and https:// links can be scraped."

    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host:
        return "Invalid URL. Please provide a link starting with http:// or https://"

    if host in BLOCKED_HOSTNAMES or host.endswith(tuple(BLOCKED_HOST_SUFFIXES)):
        return BLOCKED_URL_MESSAGE

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return "Invalid port in URL."

    literal = _host_literal_ip(host)
    if literal is not None:
        return None if _is_public_ip(literal) else BLOCKED_URL_MESSAGE

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return f"Could not resolve the host '{host}'."

    addresses = {info[4][0].split("%")[0] for info in infos}
    if not addresses:
        return f"Could not resolve the host '{host}'."

    for address in addresses:
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError:
            return f"Could not verify the host '{host}'."
        # Reject if *any* answer is internal, so mixed A/AAAA replies cannot
        # be used to slip past the check.
        if not _is_public_ip(resolved):
            return BLOCKED_URL_MESSAGE

    return None


def _safe_get(url: str, timeout: float, max_bytes: int = MAX_RESPONSE_BYTES):
    """
    GET a URL after validating it and every redirect target.

    requests follows redirects internally, which would let a public URL
    redirect to a private one, so each hop is requested explicitly.

    Returns:
        (response, page_text) — the body is decoded and truncated to max_bytes.
        page_text is empty for non-200 responses, whose body is never needed.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        reason = validate_public_url(current)
        if reason:
            raise UnsafeURLError(reason)

        response = requests.get(
            current,
            headers=BROWSER_HEADERS,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )

        if response.is_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                return response, ""
            current = urllib.parse.urljoin(current, location)
            continue

        if response.status_code != 200:
            response.close()
            return response, ""

        chunks = []
        read = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                chunks.append(chunk)
                read += len(chunk)
                if read >= max_bytes:
                    break
        finally:
            response.close()

        raw = b"".join(chunks)[:max_bytes]
        encoding = response.encoding or "utf-8"
        try:
            page_text = raw.decode(encoding, errors="replace")
        except LookupError:
            # Unknown charset in the header; UTF-8 with replacement is close enough.
            page_text = raw.decode("utf-8", errors="replace")
        return response, page_text

    raise UnsafeURLError("Too many redirects.")


def is_valid_url(url_string: str) -> bool:
    """Check if the provided text is a valid web URL with a real host."""
    if not isinstance(url_string, str):
        return False
    try:
        parsed = urllib.parse.urlparse(url_string.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def scrape_news_article(url: str):
    """
    Scrape article title, body text, and source domain from a news URL.
    Returns:
        dict: {
            "success": bool,
            "url": str,
            "domain": str,
            "title": str,
            "content": str,
            "preview": str,
            "word_count": int,
            "error": str | None
        }
    """
    url = url.strip()
    if not is_valid_url(url):
        return {
            "success": False,
            "url": url,
            "domain": "",
            "title": "",
            "content": "",
            "preview": "",
            "word_count": 0,
            "error": "Invalid URL. Please provide a link starting with http:// or https://"
        }

    domain = urllib.parse.urlparse(url).netloc.replace("www.", "")

    try:
        response, page_text = _safe_get(url, timeout=8)
        if response.status_code != 200:
            return {
                "success": False,
                "url": url,
                "domain": domain,
                "title": "",
                "content": "",
                "preview": "",
                "word_count": 0,
                "error": f"Website returned HTTP status {response.status_code}. The site may be blocking automated requests."
            }

        soup = BeautifulSoup(page_text, "html.parser")

        # 1. Extract Article Title
        title = ""
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        # 2. Clean out junk tags
        for unwanted in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]):
            unwanted.decompose()

        # 3. Target main article container
        article_container = (
            soup.find("article") or
            soup.find("div", class_=re.compile(r"article|story|content|post-body|main-content", re.I)) or
            soup.find("main") or
            soup
        )

        # 4. Extract paragraphs
        paragraphs = []
        skip_phrases = [
            "cookie", "subscribe", "newsletter", "sign in", "all rights reserved",
            "advertisement", "terms of use", "privacy policy", "click here", "follow us"
        ]

        for p in article_container.find_all("p"):
            p_text = p.get_text(separator=" ", strip=True)
            # Require minimum length and filter out boilerplate
            if len(p_text) > 40 and not any(skip in p_text.lower() for skip in skip_phrases):
                paragraphs.append(p_text)

        # Combine paragraphs (up to first 12 paragraphs for verification)
        combined_text = " ".join(paragraphs[:12]).strip()
        
        # Fallback if no <p> tags matched
        if not combined_text and title:
            combined_text = title

        words = combined_text.split()
        word_count = len(words)
        preview = " ".join(words[:50]) + "..." if len(words) > 50 else combined_text

        if not title and not combined_text:
            return {
                "success": False,
                "url": url,
                "domain": domain,
                "title": "",
                "content": "",
                "preview": "",
                "word_count": 0,
                "error": "Could not extract article text from this page. The site might require login or use heavy client-side JavaScript."
            }

        return {
            "success": True,
            "url": url,
            "domain": domain,
            "title": title,
            "content": combined_text,
            "preview": preview,
            "word_count": word_count,
            "error": None
        }

    except UnsafeURLError as e:
        return {
            "success": False,
            "url": url,
            "domain": domain,
            "title": "",
            "content": "",
            "preview": "",
            "word_count": 0,
            "error": str(e)
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "url": url,
            "domain": domain,
            "title": "",
            "content": "",
            "preview": "",
            "word_count": 0,
            "error": "Request timed out while connecting to the news website."
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "domain": domain,
            "title": "",
            "content": "",
            "preview": "",
            "word_count": 0,
            "error": f"Failed to scrape article: {str(e)}"
        }


if __name__ == "__main__":
    import sys
    test_link = sys.argv[1] if len(sys.argv) > 1 else "https://www.bbc.com/news"
    print(f"Testing scraper on: {test_link}")
    result = scrape_news_article(test_link)
    print("Success:", result["success"])
    print("Domain:", result["domain"])
    print("Title:", result["title"])
    print("Word Count:", result["word_count"])
    print("Preview:", result["preview"])
    if result["error"]:
        print("Error:", result["error"])
