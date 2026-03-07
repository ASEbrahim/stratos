#!/usr/bin/env python3
"""
V3 Training Scoring — Batch 2
Profiles: journalist_dc, ecommerce_istanbul, env_consultant_amsterdam, hr_director_sydney
813 articles x 4 profiles = 3,252 scores.
"""

import json
import os
import re

ARTICLES_PATH = "/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/articles_v2.json"
OUTPUT_PATH = "/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/v3_scores_batch2.json"

# Sports keywords that indicate pure sports content (not sports business)
SPORTS_NOISE = [
    "nrl", "nfl", "nba", "afl", "rugby", "cricket", "football score",
    "super rugby", "premier league", "la liga", "serie a", "bundesliga",
    "champions league", "world cup", "olympic medal", "batting", "bowling",
    "goalkeeper", "striker", "midfielder", "quarterback", "touchdown",
    "slam dunk", "grand slam", "wicket", "innings", "home run",
    "playoff", "semifinals", "quarterfinal", "penalty kick",
    "tennis", "golf tournament", "boxing", "mma", "ufc",
    "motorsport", "grand prix", "formula 1", "f1 ", "race winner",
    "transfer window", "signings tracker", "free agent",
    "brumbies", "wallabies", "panthers", "rabbitohs", "roosters",
    "hurricanes", "crusaders", "waratahs", "reds", "rebels",
    "broncos", "cowboys", "knights", "raiders", "eels", "sharks",
    "bulldogs", "dragons", "tigers", "warriors", "storm", "titans",
    "ferrari", "red bull racing", "mclaren",
]

GENERAL_NOISE = [
    "horoscope", "zodiac", "astrology", "lottery", "sweepstakes",
    "obituar", "crossword", "sudoku", "puzzle answer",
    "recipe of the day", "daily horoscope", "weather forecast",
    "box office day", "bollywood", "movie review", "film review",
    "album review", "concert review",
]


def _is_non_english(title, summary):
    combined = f"{title} {summary}"
    non_latin = sum(1 for c in combined if ord(c) > 0x024F and c not in '–—''""…•·€£¥₹°±²³')
    total = max(len(combined), 1)
    return non_latin / total > 0.3


def _is_sports(text):
    """Detect pure sports content."""
    hits = sum(1 for k in SPORTS_NOISE if k in text)
    return hits >= 2  # Need at least 2 sports keywords to flag


def _is_noise(text):
    return any(p in text for p in GENERAL_NOISE)


def _word_in(word, text):
    """Check if word appears as a whole word (not substring) in text."""
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


def _any_in(keywords, text, whole_word=False):
    """Check if any keyword is in text."""
    if whole_word:
        return any(_word_in(k, text) for k in keywords)
    return any(k in text for k in keywords)


def _count_in(keywords, text, whole_word=False):
    """Count matching keywords."""
    if whole_word:
        return sum(1 for k in keywords if _word_in(k, text))
    return sum(1 for k in keywords if k in text)


def _matched(keywords, text, limit=3, whole_word=False):
    """Return matched keywords."""
    if whole_word:
        return [k for k in keywords if _word_in(k, text)][:limit]
    return [k for k in keywords if k in text][:limit]


# ════════════════════════════════════════
# JOURNALIST_DC
# ════════════════════════════════════════

def score_journalist_dc(title, summary, text, article):
    score = 0.0
    signals = []

    # Tier 1: directly actionable (base 8.5-10)
    t1_kw = [
        "antitrust", "anti-trust", "tech regulation", "ai regulation",
        "ai policy", "ai governance", "ai act", "ai bill",
        "ai executive order", "section 230", "congressional hearing",
        "senate hearing", "tech lobbying", "data privacy law",
        "privacy regulation", "monopoly", "platform liability",
        "tiktok ban", "content moderation",
    ]
    t1_inst = ["ftc", "fcc"]  # whole-word match needed

    t1 = _count_in(t1_kw, text) + _count_in(t1_inst, text, whole_word=True)
    if t1 > 0:
        score += 8.5 + min(t1 - 1, 3) * 0.5
        signals.append(f"core: {', '.join(_matched(t1_kw, text) + _matched(t1_inst, text, whole_word=True))}")

    # Tier 2: highly relevant (base 5.5-8.5)
    t2_kw = [
        "openai", "artificial intelligence", "data privacy", "gdpr",
        "surveillance", "facial recognition", "deepfake",
        "misinformation", "disinformation", "algorithmic bias",
        "tech policy", "digital regulation", "cybersecurity policy",
        "encryption", "lobbying", "lobbyist", "whistleblower",
        "tech antitrust", "big tech", "silicon valley",
        "senate commerce", "tech worker",
    ]
    t2_co = ["google", "meta", "apple", "microsoft"]  # tracked companies
    t2_re = [r'\bai\b']  # regex patterns

    t2 = _count_in(t2_kw, text)
    t2 += sum(1 for c in t2_co if _word_in(c, text))
    t2 += sum(1 for p in t2_re if re.search(p, text))

    if t2 > 0:
        if score == 0:
            score = 5.5 + min(t2 - 1, 7) * 0.4
        else:
            score += min(t2, 4) * 0.2
        matched = _matched(t2_kw, text) + [c for c in t2_co if _word_in(c, text)]
        signals.append(f"relevant: {', '.join(matched[:3])}")

    # Tier 3: somewhat relevant (base 2.5-5.5)
    t3_kw = [
        "regulation", "policy", "government", "congress", "senate",
        "legislation", "cybersecurity", "hack", "breach",
        "internet", "social media", "journalism", "press freedom",
        "free speech", "censorship", "transparency",
        "startup", "venture capital", "tech company",
        "smartphone", "cloud computing", "digital",
    ]
    t3 = _count_in(t3_kw, text)
    if t3 > 0:
        if score == 0:
            score = 2.5 + min(t3 - 1, 8) * 0.3
        else:
            score += min(t3, 5) * 0.1
        if not signals:
            signals.append(f"tangential: {', '.join(_matched(t3_kw, text, 2))}")

    # Tier 4: weak relevance (base 1.0-2.5)
    t4_kw = [
        "technology", "innovation", "data", "machine learning",
        "robot", "automat", "semiconductor", "blockchain", "crypto",
        "5g", "telecom",
    ]
    t4 = _count_in(t4_kw, text)
    if t4 > 0 and score == 0:
        score = 1.0 + min(t4 - 1, 5) * 0.2
        signals.append(f"weak: {', '.join(_matched(t4_kw, text, 2))}")

    # DC / federal boost (small)
    if _any_in(["washington dc", "capitol hill", "white house", "federal agency"], text):
        score += 0.5
        signals.append("DC angle")

    return min(round(score, 1), 10.0), signals


# ════════════════════════════════════════
# ECOMMERCE_ISTANBUL
# ════════════════════════════════════════

def score_ecommerce_istanbul(title, summary, text, article):
    score = 0.0
    signals = []

    # Tier 1
    t1_kw = [
        "trendyol", "hepsiburada", "getir", "iyzico",
        "turkish e-commerce", "turkey e-commerce", "turkish export",
        "cross-border e-commerce", "cross-border commerce",
        "turkish trade", "istanbul commerce",
    ]
    t1 = _count_in(t1_kw, text)
    if t1 > 0:
        score += 8.5 + min(t1 - 1, 3) * 0.5
        signals.append(f"core: {', '.join(_matched(t1_kw, text))}")

    # Tier 2
    t2_kw = [
        "e-commerce", "ecommerce", "online retail", "online marketplace",
        "marketplace platform", "cross-border", "payment gateway",
        "last-mile delivery", "last mile", "fulfillment",
        "shopify", "alibaba",
        "logistics company", "shipping company", "supply chain",
        "digital payment", "checkout",
    ]
    t2_geo = ["turkey", "turkish", "istanbul", "ankara"]

    t2 = _count_in(t2_kw, text)
    geo_match = _count_in(t2_geo, text)

    # Turkey + commerce = boost
    if geo_match > 0 and _any_in(["commerce", "retail", "trade", "export", "import", "business", "startup", "market"], text):
        score += 3.0
        signals.append(f"Turkey + commerce context")

    if t2 > 0:
        if score == 0:
            score = 5.0 + min(t2 - 1, 6) * 0.4
        else:
            score += min(t2, 4) * 0.3
        signals.append(f"relevant: {', '.join(_matched(t2_kw, text))}")
    elif geo_match > 0:
        score += 2.0
        signals.append("Turkey mention")

    # Tier 3
    t3_kw = [
        "retail", "online shopping", "consumer", "delivery", "courier",
        "freight", "cargo", "tariff", "customs", "import", "export",
        "trade agreement", "free trade", "trade policy",
        "payment", "stripe", "paypal", "fintech",
        "mobile commerce", "social commerce",
        "startup", "founder", "venture capital", "funding",
    ]
    t3 = _count_in(t3_kw, text)
    if t3 > 0:
        if score == 0:
            score = 2.0 + min(t3 - 1, 8) * 0.3
        else:
            score += min(t3, 5) * 0.1
        if not signals:
            signals.append(f"tangential: {', '.join(_matched(t3_kw, text, 2))}")

    # Tier 4
    t4_kw = [
        "business", "entrepreneur", "small business", "growth",
        "marketing", "advertising", "customer",
        "middle east", "emerging market",
    ]
    t4 = _count_in(t4_kw, text)
    if t4 > 0 and score == 0:
        score = 0.8 + min(t4 - 1, 5) * 0.2
        signals.append(f"weak: {', '.join(_matched(t4_kw, text, 2))}")

    return min(round(score, 1), 10.0), signals


# ════════════════════════════════════════
# ENV_CONSULTANT_AMSTERDAM
# ════════════════════════════════════════

def score_env_consultant(title, summary, text, article):
    score = 0.0
    signals = []

    # Tier 1
    t1_kw = [
        "carbon market", "carbon trading", "carbon credit", "carbon offset",
        "carbon price", "emission trading", "eu ets", "carbon tax",
        "csrd", "sustainability reporting", "esg reporting",
        "corporate sustainability", "climate disclosure",
        "climeworks", "south pole", "carbon capture",
        "net zero", "net-zero", "green deal", "eu taxonomy",
        "fit for 55", "greenwashing",
    ]
    t1 = _count_in(t1_kw, text)
    if t1 > 0:
        score += 8.5 + min(t1 - 1, 3) * 0.5
        signals.append(f"core: {', '.join(_matched(t1_kw, text))}")

    # Tier 2
    t2_kw = [
        "carbon", "emission", "greenhouse gas", "co2",
        "climate change", "climate risk", "global warming",
        "paris agreement", "cop28", "cop29", "cop30",
        "circular economy", "waste reduction",
        "renewable energy", "solar energy", "wind energy", "wind farm",
        "clean energy", "green energy", "energy transition",
        "fossil fuel", "methane", "deforestation", "biodiversity",
        "sustainability", "sustainable development",
        "direct air capture", "netherlands", "dutch", "amsterdam",
        "eu regulation", "eu commission",
    ]
    t2_co = ["arcadis"]  # Shell needs context check

    t2 = _count_in(t2_kw, text)
    t2 += _count_in(t2_co, text)

    # Shell with environmental context
    if _word_in("shell", text) and _any_in(["oil", "energy", "emission", "carbon", "climate", "gas", "lng", "fossil", "transition"], text):
        t2 += 1
        signals.append("Shell (energy context)")

    if t2 > 0:
        if score == 0:
            score = 5.0 + min(t2 - 1, 7) * 0.4
        else:
            score += min(t2, 5) * 0.2
        signals.append(f"relevant: {', '.join(_matched(t2_kw, text, 3))}")

    # Tier 3
    t3_kw = [
        "climate", "pollution", "environmental", "green",
        "renewable", "solar", "wind", "hydrogen",
        "electric vehicle", "battery", "energy efficiency",
        "heat pump", "insulation",
        "regulation", "compliance", "reporting",
        "oil and gas", "petroleum", "mining",
        "conservation", "ecosystem", "wildfire",
        "drought", "flood", "extreme weather",
    ]
    t3 = _count_in(t3_kw, text)
    if t3 > 0:
        if score == 0:
            score = 2.5 + min(t3 - 1, 8) * 0.3
        else:
            score += min(t3, 5) * 0.1
        if not signals:
            signals.append(f"tangential: {', '.join(_matched(t3_kw, text, 2))}")

    # Tier 4
    t4_kw = [
        "energy", "power", "electricity", "grid", "utility",
        "infrastructure", "agriculture", "farming",
        "supply chain", "manufacturing",
    ]
    t4 = _count_in(t4_kw, text)
    if t4 > 0 and score == 0:
        score = 1.0 + min(t4 - 1, 5) * 0.2
        signals.append(f"weak: {', '.join(_matched(t4_kw, text, 2))}")

    return min(round(score, 1), 10.0), signals


# ════════════════════════════════════════
# HR_DIRECTOR_SYDNEY
# ════════════════════════════════════════

def score_hr_director(title, summary, text, article):
    score = 0.0
    signals = []

    # FIRST: sports filter — if this is sports content, heavily discount Australia matches
    is_sport = _is_sports(text)

    # Tier 1
    t1_kw = [
        "talent acquisition", "recruitment strategy",
        "remote work policy", "return to office", "rto mandate",
        "australian employment", "australian labour", "fair work",
        "people analytics", "diversity equity inclusion",
    ]
    t1_co = ["atlassian", "canva", "culture amp", "employment hero"]

    t1 = _count_in(t1_kw, text) + _count_in(t1_co, text)
    if t1 > 0 and not is_sport:
        score += 8.5 + min(t1 - 1, 3) * 0.5
        signals.append(f"core: {', '.join(_matched(t1_kw, text) + _matched(t1_co, text))}")

    # Tier 2
    t2_kw = [
        "hiring", "recruitment", "recruiter", "job market",
        "labor market", "labour market", "talent shortage",
        "employee retention", "turnover", "attrition",
        "remote work", "work from home", "hybrid work",
        "employment law", "labor law", "labour law",
        "workplace regulation", "industrial relations",
        "performance management", "performance review",
        "employee engagement", "onboarding",
        "layoff", "laid off", "job cut", "downsiz", "retrench",
        "tech worker", "tech talent", "software engineer",
        "compensation", "salary", "pay gap",
        "burnout", "mental health",
        "ai hiring", "ai recruitment", "automated hiring",
        "hr tech", "hr software", "hr platform",
    ]
    t2_geo_kw = ["dei", "diversity inclusion"]  # these are HR-specific enough

    t2 = _count_in(t2_kw, text) + _count_in(t2_geo_kw, text, whole_word=True)

    # Australia angle — only if not sports
    au_match = 0
    if not is_sport:
        au_kw = ["australia", "australian", "sydney"]
        au_match = _count_in(au_kw, text)
        if au_match > 0 and _any_in(["employment", "workplace", "labour", "labor", "work", "hiring", "job", "tech", "company", "business"], text):
            score += 2.0
            signals.append("Australian business context")

    if t2 > 0:
        if is_sport:
            pass  # ignore t2 if it's sports
        elif score == 0:
            score = 5.0 + min(t2 - 1, 7) * 0.4
        else:
            score += min(t2, 4) * 0.3
        if not is_sport:
            signals.append(f"relevant: {', '.join(_matched(t2_kw, text, 3))}")

    # Tier 3 (skip if sports)
    if not is_sport:
        t3_kw = [
            "employee", "employer", "workforce", "staff",
            "human resources", "workplace", "career development",
            "training", "upskilling", "reskilling", "learning",
            "leadership", "management", "team building", "culture",
            "startup culture", "tech company", "tech industry",
            "interview", "resume", "linkedin",
            "contractor", "freelance", "gig economy", "outsourc",
            "immigration", "visa", "skilled worker",
            "gender gap", "equal pay", "harassment", "discrimination",
        ]
        t3 = _count_in(t3_kw, text)
        if t3 > 0:
            if score == 0:
                score = 2.5 + min(t3 - 1, 8) * 0.3
            else:
                score += min(t3, 5) * 0.1
            if not signals:
                signals.append(f"tangential: {', '.join(_matched(t3_kw, text, 2))}")

    # Tier 4
    if not is_sport:
        t4_kw = [
            "technology", "software", "business", "corporate",
            "office", "productivity", "collaboration",
        ]
        t4 = _count_in(t4_kw, text)
        if t4 > 0 and score == 0:
            score = 0.8 + min(t4 - 1, 5) * 0.2
            signals.append(f"weak: {', '.join(_matched(t4_kw, text, 2))}")

    if is_sport:
        if score > 0:
            score *= 0.1  # Near-zero for sports
        signals = ["sports content - not relevant to HR"]

    return min(round(score, 1), 10.0), signals


# ════════════════════════════════════════
# MAIN ORCHESTRATOR
# ════════════════════════════════════════

SCORERS = {
    "journalist_dc": score_journalist_dc,
    "ecommerce_istanbul": score_ecommerce_istanbul,
    "env_consultant_amsterdam": score_env_consultant,
    "hr_director_sydney": score_hr_director,
}

PROFILE_CONTEXTS = {
    "journalist_dc": "tech regulation, AI policy, and corporate lobbying in Washington DC",
    "ecommerce_istanbul": "cross-border e-commerce, Turkish exports, logistics, and payments",
    "env_consultant_amsterdam": "carbon markets, EU ETS compliance, CSRD reporting, and climate risk",
    "hr_director_sydney": "talent acquisition, remote work, DEI, and Australian employment law in tech",
}


def score_article(article, profile_id):
    title = (article.get("title") or "").lower()
    summary = (article.get("summary") or "").lower()
    text = f"{title} {summary}"

    # Non-English check
    if _is_non_english(title, summary):
        # Check if tier1 still matches
        scorer = SCORERS[profile_id]
        s, sig = scorer(title, summary, text, article)
        if s >= 7:
            s *= 0.5  # Penalize but don't zero out
            sig.append("non-English penalty")
        else:
            return (0.0, "Non-English content",
                    f"Article in non-English language, not relevant to {profile_id}.")

        s = round(min(s, 10.0), 1)
        reason = "; ".join(sig[:3]) if sig else f"Not relevant to {profile_id}"
        think = _build_think(article, profile_id, sig, s)
        return (s, reason, think)

    # General noise check
    if _is_noise(text):
        return (0.0, "Noise content",
                f"Article is noise (horoscope/recipe/etc), irrelevant to {profile_id}.")

    # Score
    scorer = SCORERS[profile_id]
    s, sig = scorer(title, summary, text, article)
    s = round(min(s, 10.0), 1)

    reason = "; ".join(sig[:3]) if sig else f"Not relevant to {profile_id}"
    think = _build_think(article, profile_id, sig, s)
    return (s, reason, think)


def _build_think(article, profile_id, signals, score):
    title_short = (article.get("title") or "")[:60]
    ctx = PROFILE_CONTEXTS[profile_id]
    if score >= 7:
        level = "highly relevant"
    elif score >= 4:
        level = "moderately relevant"
    elif score > 0:
        level = "weakly relevant"
    else:
        level = "irrelevant"

    sig_text = "; ".join(signals[:4]) if signals else "No matching signals"
    return (f"Article '{title_short}' scored {score}/10 for {profile_id}. "
            f"Signals: {sig_text}. "
            f"Content is {level} to the profile's focus on {ctx}.")


def main():
    with open(ARTICLES_PATH) as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles")
    print(f"Profiles: {list(SCORERS.keys())}")
    print(f"Total scores: {len(articles) * len(SCORERS)}")

    all_scores = []

    for pid in SCORERS:
        print(f"\n{'='*60}")
        print(f"Scoring: {pid}")
        print(f"{'='*60}")

        for article in articles:
            s, reason, think = score_article(article, pid)
            all_scores.append({
                "article_id": article["id"],
                "profile_id": pid,
                "score": s,
                "reason": reason,
                "think_text": think,
                "think_tokens": 100,
            })

        # Profile stats
        ps = [e["score"] for e in all_scores if e["profile_id"] == pid]
        avg = sum(ps) / len(ps)
        high = sum(1 for s in ps if s >= 7.0)
        mid = sum(1 for s in ps if 4.0 <= s < 7.0)
        low_nz = sum(1 for s in ps if 0 < s < 4.0)
        zero = sum(1 for s in ps if s == 0)
        print(f"  avg={avg:.2f} | >=7: {high} | 4-7: {mid} | 0.1-4: {low_nz} | 0: {zero}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_scores)} scores to {OUTPUT_PATH}")

    # Distribution summary
    print("\n=== SCORE DISTRIBUTION ===")
    for pid in SCORERS:
        ps = [e["score"] for e in all_scores if e["profile_id"] == pid]
        buckets = {}
        for s in ps:
            b = int(s)
            buckets[b] = buckets.get(b, 0) + 1
        print(f"\n{pid}:")
        for b in sorted(buckets.keys()):
            bar = "#" * (buckets[b] // 5)
            print(f"  {b:>2}: {buckets[b]:>4} {bar}")


if __name__ == "__main__":
    main()
