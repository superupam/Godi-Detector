# ==========================================
# GODI DETECTOR
# Web News Link Scraper
# Extracts article headline, body, and metadata
# ==========================================

import re
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
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=8)
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

        soup = BeautifulSoup(response.text, "html.parser")

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
