# ==========================================
# GODI DETECTOR
# Real-Time Fact Retrieval & Evidence Engine
# Upgraded: Sentence-Level Proximity & Anchor Verification
# ==========================================

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
import requests

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "out", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "yesterday", "today", "tomorrow"
}

WIKI_HEADERS = {
    "User-Agent": "GodiDetector/1.0 (Educational Fact Verification System; contact@godidetector.ai)"
}

NEWS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def clean_html(text):
    """Strip HTML tags and unescape entities."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html.unescape(clean).strip()


def extract_keywords(text):
    """Extract significant keywords from a news claim."""
    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text)
    keywords = [w for w in words if w.lower() not in STOPWORDS]
    return keywords if keywords else words


def extract_anchors(text):
    """Extract temporal and numerical anchors (e.g. 2022, 2021) that must match."""
    years = re.findall(r"\b(19\d\d|20\d\d)\b", text)
    return set(years)


def search_wikipedia(query_keywords, max_results=4):
    """Search Wikipedia for authoritative background evidence."""
    evidence = []
    if not query_keywords:
        return evidence

    search_query = " ".join(query_keywords[:8])
    encoded_query = urllib.parse.quote(search_query)
    search_url = (
        f"https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={encoded_query}&format=json"
    )

    try:
        resp = requests.get(search_url, headers=WIKI_HEADERS, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            search_items = data.get("query", {}).get("search", [])

            for item in search_items[:max_results]:
                title = item.get("title", "")
                raw_snippet = item.get("snippet", "")
                clean_search_snippet = clean_html(raw_snippet)

                if not title:
                    continue

                title_encoded = urllib.parse.quote(title.replace(" ", "_"))
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_encoded}"
                sum_resp = requests.get(summary_url, headers=WIKI_HEADERS, timeout=5)

                extract = ""
                page_url = f"https://en.wikipedia.org/wiki/{title_encoded}"
                if sum_resp.status_code == 200:
                    sum_data = sum_resp.json()
                    extract = sum_data.get("extract", "")
                    page_url = sum_data.get("content_urls", {}).get("desktop", {}).get("page", page_url)

                combined_text = f"{clean_search_snippet}. {extract}".strip()

                if combined_text:
                    evidence.append({
                        "source_type": "Wikipedia",
                        "source_name": f"Wikipedia ({title})",
                        "title": title,
                        "text": combined_text,
                        "url": page_url
                    })
    except Exception:
        pass

    return evidence


def search_google_news(query_keywords, max_results=5):
    """Search Google News RSS for matching news coverage."""
    evidence = []
    if not query_keywords:
        return evidence

    search_query = " ".join(query_keywords[:8])
    encoded_query = urllib.parse.quote(search_query)
    news_url = (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = requests.get(news_url, headers=NEWS_HEADERS, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")

            for item in items[:max_results]:
                title_elem = item.find("title")
                link_elem = item.find("link")
                source_elem = item.find("source")

                title = title_elem.text if title_elem is not None else ""
                link = link_elem.text if link_elem is not None else ""
                source_name = source_elem.text if source_elem is not None else "News Source"

                if title:
                    evidence.append({
                        "source_type": "News Media",
                        "source_name": source_name,
                        "title": title,
                        "text": title,
                        "url": link
                    })
    except Exception:
        pass

    return evidence


_AUX_VERBS = r"(?:(?:has|have|had|is|was|are|were|will|would|also)\s+){0,3}"

_WINNER_PATTERNS = (
    # "Argentina won the ensuing penalty shoot-out ..."
    r"([A-Z][\w'’’.\-]*(?:\s+[A-Z][\w'’’.\-]*)*)\s+" + _AUX_VERBS +
    r"(?:won|beat|defeated)\s+(?:the\s+)?(?:ensuing\s+)?"
    r"(?:penalty\s+shoot-?out|match|final|tournament|title)",
    # "Cristiano Ronaldo has won ..." / plain "X won ..."
    r"([A-Z][\w'’’.\-]*(?:\s+[A-Z][\w'’’.\-]*)*)\s+" + _AUX_VERBS + r"won\b",
)


def extract_asserted_winner(text):
    """Return the entity a sentence says won, or None."""
    for pattern in _WINNER_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def fetch_wikipedia_plaintext(title):
    """Fetch the full plain-text extract for a page (the summary often omits the result)."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    api_url = (
        f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
        f"&explaintext=1&redirects=1&format=json&titles={encoded}"
    )
    fallback_url = f"https://en.wikipedia.org/wiki/{encoded}"

    resp = requests.get(api_url, headers=WIKI_HEADERS, timeout=5)
    if resp.status_code != 200:
        return "", fallback_url

    pages = resp.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    resolved_title = page.get("title", title)
    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(resolved_title.replace(' ', '_'))}"
    return page.get("extract", "") or "", page_url


def extract_participants(plaintext):
    """Pull the competitors out of '... was contested by Argentina and ... France.'"""
    match = re.search(r"was contested by ([A-Z][^.]{0,140}?)(?:\.|,|\n)", plaintext)
    if not match:
        return []
    return [part.strip() for part in re.split(r",|\band\b", match.group(1)) if part.strip()]


def check_direct_event_contradiction(claim):
    """
    Generic event-winner contradiction check.
    If the claim asserts someone won a tournament/award/election, fetch the
    official Wikipedia page for that event and compare the actual winner
    against the entity named in the claim.
    Currently handles: FIFA World Cup (any year where Wikipedia has results).
    """
    claim_lower = claim.lower()
    win_terms = ["won", "winner", "champion", "victory", "title", "cup"]
    if not any(term in claim_lower for term in win_terms):
        return None

    # --- FIFA World Cup: generic year-based check ---
    if "world cup" in claim_lower and ("fifa" in claim_lower or "football" in claim_lower or "soccer" in claim_lower or "messi" in claim_lower or "ronaldo" in claim_lower):
        years = re.findall(r"\b(19\d\d|20\d\d)\b", claim)
        target_year = years[0] if years else "2022"

        final_query = f"{target_year} FIFA World Cup final winner"
        encoded = urllib.parse.quote(final_query)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json"

        try:
            resp = requests.get(search_url, headers=WIKI_HEADERS, timeout=5)
            if resp.status_code != 200:
                return None

            items = resp.json().get("query", {}).get("search", [])
            if not items:
                return None

            # Prefer the article about the final itself, then any article for that year
            titles = [item.get("title", "") for item in items]
            title = next(
                (t for t in titles if target_year in t and "final" in t.lower()),
                next((t for t in titles if target_year in t), titles[0] if titles else ""),
            )
            if not title:
                return None

            # Use the full article text: the short page summary frequently omits the result
            plaintext, page_url = fetch_wikipedia_plaintext(title)
            if not plaintext:
                return None

            text_lower = plaintext.lower()
            actual_winner = extract_asserted_winner(plaintext)
            if not actual_winner:
                return None

            participants = extract_participants(plaintext)
            winner_lower = actual_winner.lower()

            # Who does the claim say won? ("Cristiano Ronaldo has won ...")
            claimed_winner = extract_asserted_winner(claim)
            contradicted_entities = []

            if claimed_winner:
                claimed_lower = claimed_winner.lower()
                if claimed_lower != winner_lower:
                    claims_participant = any(claimed_lower in p.lower() for p in participants)
                    mentions_winner_only = claimed_lower not in text_lower
                    if claims_participant or mentions_winner_only:
                        contradicted_entities = [claimed_winner]
            else:
                # Fall back to checking every proper noun named in the claim
                claim_entities = re.findall(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b", claim)
                contradicted_entities = [
                    e for e in claim_entities
                    if len(e) > 3 and e.lower() not in text_lower and e.lower() != winner_lower
                ]

            if contradicted_entities:
                    contradiction_detail = (
                        f"According to Wikipedia ({title}), the {target_year} FIFA World Cup was won by "
                        f"{actual_winner}. The claim mentions {', '.join(contradicted_entities)}, "
                        f"which is not supported by the official record."
                    )
                    return {
                        "status": "CONTRADICTED",
                        "confidence": 0.97,
                        "top_source": f"Wikipedia ({title})",
                        "top_snippet": contradiction_detail,
                        "top_url": page_url,
                        "matched_keywords": [actual_winner],
                        "missing_keywords": contradicted_entities,
                        "match_ratio": 0.0,
                        "total_evidence_count": 1,
                    }
        except Exception:
            pass

    return None


def evaluate_sentence_level_evidence(claim, evidence_items):
    """
    Score evidence by analyzing sentence-level co-occurrence.
    Prevents false positives where keywords match across separate, unrelated paragraphs.
    """
    keywords = extract_keywords(claim)
    lower_keywords = [w.lower() for w in keywords]
    anchors = extract_anchors(claim)

    best_score = 0.0
    best_item = None
    best_sentence = ""
    best_matched = []
    best_missing = lower_keywords

    for item in evidence_items:
        text = item["text"]
        # Split into distinct sentences
        raw_sentences = re.split(r"[.!?\n]+", text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

        for sentence in sentences:
            s_lower = sentence.lower()

            # Anchor constraint: If claim has a specific year, the sentence MUST contain it
            if anchors:
                has_anchor = any(a in s_lower for a in anchors)
                if not has_anchor:
                    continue

            # Check matching keywords in THIS specific sentence
            matched = [kw for kw in lower_keywords if kw in s_lower]
            missing = [kw for kw in lower_keywords if kw not in s_lower]
            ratio = len(matched) / len(lower_keywords) if lower_keywords else 0.0

            if ratio > best_score:
                best_score = ratio
                best_item = item
                best_sentence = sentence
                best_matched = matched
                best_missing = missing

    return best_score, best_item, best_sentence, best_matched, best_missing


def verify_claim_evidence(claim):
    """
    Main verification pipeline:
    1. Checks direct historical contradictions (e.g. World Cup winners, election winners).
    2. Performs sentence-level co-occurrence verification.
    """
    # Step 1: Check direct contradiction on authoritative event records
    contradiction = check_direct_event_contradiction(claim)
    if contradiction:
        return contradiction

    keywords = extract_keywords(claim)
    if not keywords:
        return {
            "status": "NO_EVIDENCE",
            "confidence": 0.0,
            "top_source": None,
            "top_snippet": "No actionable keywords found in input.",
            "top_url": None,
            "matched_keywords": [],
            "missing_keywords": [],
            "match_ratio": 0.0,
            "total_evidence_count": 0
        }

    lower_keywords = [w.lower() for w in keywords]

    # Step 2: Retrieve evidence
    wiki_evidence = search_wikipedia(keywords, max_results=4)
    news_evidence = search_google_news(keywords, max_results=5)
    all_evidence = wiki_evidence + news_evidence

    if not all_evidence:
        return {
            "status": "NO_EVIDENCE",
            "confidence": 0.0,
            "top_source": None,
            "top_snippet": "No matching records found in knowledge base or news archives.",
            "top_url": None,
            "matched_keywords": [],
            "missing_keywords": lower_keywords,
            "match_ratio": 0.0,
            "total_evidence_count": 0
        }

    # Step 3: Sentence-level evaluation
    best_score, best_item, best_sentence, best_matched, best_missing = evaluate_sentence_level_evidence(
        claim, all_evidence
    )

    # Corroboration Thresholds (sentence-level requires >= 0.55 match in a single coherent sentence)
    if best_score >= 0.55:
        status = "CORROBORATED"
        confidence = min(0.78 + (best_score * 0.21), 0.99)
        top_snippet = best_sentence
    elif best_score >= 0.35:
        status = "PARTIALLY_CORROBORATED"
        confidence = 0.50 + (best_score * 0.15)
        top_snippet = best_sentence
    else:
        status = "NO_EVIDENCE"
        confidence = 0.15
        top_snippet = f"No single authoritative sentence confirmed the connection between {', '.join(keywords[:3])}."

    top_source = best_item["source_name"] if best_item else (all_evidence[0]["source_name"] if all_evidence else None)
    top_url = best_item["url"] if best_item else (all_evidence[0]["url"] if all_evidence else None)

    return {
        "status": status,
        "confidence": round(confidence, 2),
        "top_source": top_source,
        "top_snippet": top_snippet,
        "top_url": top_url,
        "matched_keywords": best_matched,
        "missing_keywords": best_missing,
        "match_ratio": round(best_score, 2),
        "total_evidence_count": len(all_evidence)
    }


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("      GODI DETECTOR - FACT RETRIEVAL ENGINE")
    print("=" * 60)

    test_claims = [
        "Cristiano Ronaldo has won the FIFA World Cup in 2022.",
        "Lionel Messi won the FIFA World Cup in December, 2022",
        "NASA rover Perseverance landed on Mars in February 2021"
    ]

    claims_to_run = [sys.argv[1]] if len(sys.argv) > 1 else test_claims

    for claim in claims_to_run:
        print(f"\nClaim: \"{claim}\"")
        res = verify_claim_evidence(claim)
        print(f"Status: [{res['status']}] (Confidence: {res['confidence']*100:.0f}%)")
        print(f"Top Source: {res['top_source']}")
        print(f"Matched Words: {res['matched_keywords']}")
        print(f"Snippet: {res['top_snippet']}")
        if res.get('top_url'):
            print(f"Source URL: {res['top_url']}")
        print("-" * 60)
