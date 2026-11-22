// ==========================================
// GODI DETECTOR - CLIENT SCRIPT
// Dual-Engine Verification & Link Scraper
// ==========================================

document.addEventListener("DOMContentLoaded", () => {
    // Mode Switching Elements
    const tabText = document.getElementById("tab-text");
    const tabUrl = document.getElementById("tab-url");
    const viewText = document.getElementById("view-text");
    const viewUrl = document.getElementById("view-url");

    // Inputs
    const newsInput = document.getElementById("news-input");
    const urlInput = document.getElementById("url-input");
    const btnScrape = document.getElementById("btn-scrape");
    const charCount = document.getElementById("char-count");
    const toggleFactCheck = document.getElementById("toggle-fact-check");
    const btnVerify = document.getElementById("btn-verify");
    const btnText = btnVerify.querySelector(".btn-text");
    const loadingBox = document.getElementById("loading-box");
    const loadingSubText = document.getElementById("loading-sub-text");
    const resultsSection = document.getElementById("results-section");

    // Scraped Preview Elements
    const scrapedPreviewCard = document.getElementById("scraped-preview-card");
    const previewDomain = document.getElementById("preview-domain");
    const previewWordcount = document.getElementById("preview-wordcount");
    const previewTitle = document.getElementById("preview-title");
    const previewSnippet = document.getElementById("preview-snippet");

    // Scraped Source Banner in Results
    const scrapedSourceBanner = document.getElementById("scraped-source-banner");
    const scrapedBannerDomain = document.getElementById("scraped-banner-domain");
    const scrapedBannerTitle = document.getElementById("scraped-banner-title");

    // Verdict Banner Elements
    const verdictBanner = document.getElementById("verdict-banner");
    const verdictIcon = document.getElementById("verdict-icon");
    const verdictTitle = document.getElementById("verdict-title");
    const verdictBadge = document.getElementById("verdict-badge");
    const verdictExplanation = document.getElementById("verdict-explanation");

    // Engine 1 Elements
    const styleRealPercent = document.getElementById("style-real-percent");
    const styleRealBar = document.getElementById("style-real-bar");
    const styleFakePercent = document.getElementById("style-fake-percent");
    const styleFakeBar = document.getElementById("style-fake-bar");
    const signalsContainer = document.getElementById("signals-container");

    // Engine 2 Elements
    const factStatusBadge = document.getElementById("fact-status-badge");
    const factSource = document.getElementById("fact-source");
    const factMatch = document.getElementById("fact-match");
    const factSnippet = document.getElementById("fact-snippet");
    const factUrl = document.getElementById("fact-url");
    const factLinkContainer = document.getElementById("fact-link-container");

    let currentMode = "text"; // "text" or "url"

    // Samples
    const samples = {
        real: "The Senate on Wednesday passed a bipartisan $1.2 trillion infrastructure bill with overwhelming support from both parties.",
        fake: "BREAKING: Shocking video exposes secret plot by corrupt politicians to rig the election! WATCH HERE!",
        messi: "Lionel Messi won the FIFA World Cup in December, 2022",
        ronaldo: "Cristiano Ronaldo has won the FIFA World Cup in 2022.",
        "bbc-url": "https://www.bbc.com/news/world"
    };

    // Tab Switching
    tabText.addEventListener("click", () => {
        currentMode = "text";
        tabText.classList.add("active");
        tabUrl.classList.remove("active");
        viewText.style.display = "block";
        viewUrl.style.display = "none";
        newsInput.focus();
    });

    tabUrl.addEventListener("click", () => {
        currentMode = "url";
        tabUrl.classList.add("active");
        tabText.classList.remove("active");
        viewUrl.style.display = "block";
        viewText.style.display = "none";
        urlInput.focus();
    });

    // Character Count Update
    newsInput.addEventListener("input", () => {
        const len = newsInput.value.length;
        charCount.textContent = `${len} character${len === 1 ? "" : "s"}`;
    });

    // Handle Sample Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const sampleKey = chip.getAttribute("data-sample");
            if (sampleKey === "bbc-url") {
                tabUrl.click();
                urlInput.value = samples[sampleKey];
                btnScrape.click();
            } else if (samples[sampleKey]) {
                tabText.click();
                newsInput.value = samples[sampleKey];
                newsInput.dispatchEvent(new Event("input"));
                newsInput.focus();
            }
        });
    });

    // Handle Scrape Button
    btnScrape.addEventListener("click", async () => {
        const url = urlInput.value.trim();
        if (!url || (!url.startsWith("http://") && !url.startsWith("https://"))) {
            alert("Please enter a valid web URL starting with http:// or https://");
            urlInput.focus();
            return;
        }

        btnScrape.disabled = true;
        btnScrape.textContent = "Scraping...";

        try {
            const response = await fetch("/api/scrape", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Failed to extract article");
            }

            const article = await response.json();
            previewDomain.textContent = article.domain;
            previewWordcount.textContent = `${article.word_count} words extracted`;
            previewTitle.textContent = article.title || "No Title Found";
            previewSnippet.textContent = article.preview || "No content snippet available.";
            scrapedPreviewCard.style.display = "block";

        } catch (error) {
            alert(`Scraping Error: ${error.message}`);
            scrapedPreviewCard.style.display = "none";
        } finally {
            btnScrape.disabled = false;
            btnScrape.textContent = "Extract Article";
        }
    });

    // Handle Verification
    btnVerify.addEventListener("click", async () => {
        let textToVerify = "";

        if (currentMode === "url") {
            const url = urlInput.value.trim();
            if (!url) {
                alert("Please paste a news article URL to verify.");
                urlInput.focus();
                return;
            }
            textToVerify = url;
            loadingSubText.textContent = "Connecting to website, scraping article, and verifying...";
        } else {
            textToVerify = newsInput.value.trim();
            if (!textToVerify) {
                alert("Please enter a news headline or claim to verify.");
                newsInput.focus();
                return;
            }
            loadingSubText.textContent = "Evaluating vocabulary tone & cross-referencing live databases...";
        }

        // UI Loading State
        btnVerify.disabled = true;
        btnText.textContent = "Verifying with Dual Engines...";
        loadingBox.style.display = "block";
        resultsSection.style.display = "none";

        try {
            const response = await fetch("/api/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: textToVerify,
                    check_facts: toggleFactCheck.checked
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Verification failed");
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btnVerify.disabled = false;
            btnText.textContent = "🔍 Verify News Content";
            loadingBox.style.display = "none";
        }
    });

    function renderResults(data) {
        // 0. Scraped Source Banner
        if (data.scraped_article) {
            scrapedBannerDomain.textContent = data.scraped_article.domain;
            scrapedBannerTitle.textContent = data.scraped_article.title;
            scrapedSourceBanner.style.display = "flex";
        } else {
            scrapedSourceBanner.style.display = "none";
        }

        // 1. Verdict Banner
        verdictBanner.className = "verdict-banner";
        if (data.category === "real") {
            verdictBanner.classList.add("verdict-real");
            verdictIcon.textContent = "✅";
        } else if (data.category === "fake") {
            verdictBanner.classList.add("verdict-fake");
            verdictIcon.textContent = "🚨";
        } else {
            verdictBanner.classList.add("verdict-warn");
            verdictIcon.textContent = "⚠️";
        }

        verdictTitle.textContent = data.verdict;
        verdictBadge.textContent = `${data.confidence}% Confidence`;
        verdictExplanation.textContent = data.explanation;

        // 2. Engine 1: Style Analysis
        const style = data.engine_1_style;
        styleRealPercent.textContent = `${style.real_probability}%`;
        styleRealBar.style.width = `${style.real_probability}%`;

        styleFakePercent.textContent = `${style.fake_probability}%`;
        styleFakeBar.style.width = `${style.fake_probability}%`;

        signalsContainer.innerHTML = "";

        // Show BOTH signal groups with their own labels
        const groups = [
            { tokens: style.real_signals, cssClass: "badge-signal-real", prefix: "✓", label: "Formal:" },
            { tokens: style.fake_signals, cssClass: "badge-signal-fake", prefix: "⚠", label: "Sensational:" }
        ];

        groups.forEach(({ tokens, cssClass, prefix, label }) => {
            if (!tokens || tokens.length === 0) return;
            const labelEl = document.createElement("span");
            labelEl.style.cssText = "font-size:0.78rem;font-weight:700;color:#94A3B8;margin-right:0.3rem;";
            labelEl.textContent = label;
            signalsContainer.appendChild(labelEl);
            tokens.forEach(token => {
                const badge = document.createElement("span");
                badge.className = `badge-signal ${cssClass}`;
                badge.textContent = `${prefix} ${token}`;
                signalsContainer.appendChild(badge);
            });
            signalsContainer.appendChild(document.createElement("br"));
        });

        if (signalsContainer.innerHTML === "") {
            signalsContainer.innerHTML = '<span style="font-size:0.85rem; color:#94A3B8;">No strong distinctive keywords detected.</span>';
        }

        // 3. Engine 2: Fact Retrieval
        const facts = data.engine_2_facts;
        if (facts) {
            factStatusBadge.className = "badge-status";
            if (facts.status === "CORROBORATED") {
                factStatusBadge.classList.add("status-corroborated");
                factStatusBadge.textContent = "CORROBORATED";
            } else if (facts.status === "CONTRADICTED") {
                factStatusBadge.classList.add("status-none");
                factStatusBadge.textContent = "CONTRADICTED / DEBUNKED";
            } else if (facts.status === "PARTIALLY_CORROBORATED") {
                factStatusBadge.classList.add("status-partial");
                factStatusBadge.textContent = "PARTIALLY CORROBORATED";
            } else {
                factStatusBadge.classList.add("status-none");
                factStatusBadge.textContent = "NO EVIDENCE FOUND";
            }

            factSource.textContent = facts.top_source || "None";
            factMatch.textContent = `${Math.round(facts.match_ratio * 100)}% of claim keywords verified`;
            factSnippet.textContent = facts.top_snippet ? `"${facts.top_snippet.slice(0, 240)}..."` : "No snippet available.";

            if (facts.top_url) {
                factUrl.href = facts.top_url;
                factLinkContainer.style.display = "block";
            } else {
                factLinkContainer.style.display = "none";
            }
        } else {
            factStatusBadge.className = "badge-status status-partial";
            factStatusBadge.textContent = "DISABLED";
            factSource.textContent = "Live checking was disabled.";
            factMatch.textContent = "N/A";
            factSnippet.textContent = "Enable live fact checking in the options above.";
            factLinkContainer.style.display = "none";
        }

        resultsSection.style.display = "block";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
});
