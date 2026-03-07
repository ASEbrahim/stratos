#!/usr/bin/env python3
"""
V3 Training Data Scorer — Batch 0 (profiles 0-2 of V3_NEW_PROFILES)
Profiles: math_teacher_texas, backend_engineer_singapore, real_estate_dubai

Improved scoring with word-boundary matching and context-aware heuristics.
"""

import json
import re
import os

OUTPUT_FILE = "/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/v3_scores_batch0.json"
ARTICLES_FILE = "/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/articles_v2.json"

with open(ARTICLES_FILE) as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")


def kw_match(text, keywords):
    """Count keyword matches using word-boundary-aware matching."""
    count = 0
    for kw in keywords:
        kw_clean = kw.strip()
        if len(kw_clean) <= 3:
            # Short keywords need word boundaries
            if re.search(r'\b' + re.escape(kw_clean) + r'\b', text, re.IGNORECASE):
                count += 1
        else:
            if kw_clean.lower() in text:
                count += 1
    return count


def kw_which(text, keywords):
    """Return which keywords matched."""
    matched = []
    for kw in keywords:
        kw_clean = kw.strip()
        if len(kw_clean) <= 3:
            if re.search(r'\b' + re.escape(kw_clean) + r'\b', text, re.IGNORECASE):
                matched.append(kw_clean)
        else:
            if kw_clean.lower() in text:
                matched.append(kw_clean)
    return matched


# ═══════════════════════════════════════════════════════════════
# PROFILE SCORING FUNCTIONS — one per profile for nuanced control
# ═══════════════════════════════════════════════════════════════

def score_math_teacher(title, summary, text, source, article):
    """Score for math_teacher_texas profile."""

    # Tier 1: Directly actionable for a HS math teacher in Austin TX
    t1 = [
        "ap calculus", "ap statistics", "math teacher", "math education",
        "texas education agency", "math curriculum", "stem curriculum",
        "desmos", "khan academy", "texas instruments",
        "nctm", "edtech", "education technology",
        "college board", "sat math", "act math",
        "austin isd", "texas school", "texas teacher",
        "math pedagogy", "math instruction", "teaching math",
        "ap exam", "calculus curriculum", "statistics course",
    ]

    # Tier 2: Relevant to education field
    t2 = [
        "education policy", "teacher salary", "teacher shortage",
        "stem education", "stem outreach", "coding in school",
        "standardized testing", "student achievement", "learning outcomes",
        "online learning", "remote learning", "tutoring",
        "school funding", "education budget", "public school",
        "high school", "secondary education", "k-12",
        "math olympiad", "statistics education",
        "ai in education", "chatgpt school", "ai tutor",
        "graphing calculator", "classroom technology",
        "pedagogy", "curriculum development",
        "texas legislature", "texas policy",
        "university of texas", "ut austin",
        "school district", "state education",
        "teaching career", "teaching profession",
    ]

    # Tier 3: Tangentially relevant
    t3 = [
        "education", "teaching", "school", "teacher", "curriculum",
        "texas", "austin", "mathematics", "calculus",
        "statistics", "algebra", "geometry", "stem",
        "learning", "academic", "student",
    ]

    t1_hits = kw_match(text, t1)
    t2_hits = kw_match(text, t2)
    t3_hits = kw_match(text, t3)

    # Check for noise topics
    noise_topics = [
        "oil and gas", "petroleum", "drilling", "pipeline",
        "cryptocurrency", "bitcoin", "ethereum",
        "real estate", "property market", "mortgage",
        "k-pop", "gaming", "esports",
        "mining", "welding", "hvac",
        "military", "defense", "weapons",
        "stock market", "earnings report", "quarterly profit",
        "recipe", "cooking", "restaurant",
    ]
    noise_hits = kw_match(text, noise_topics)

    # Calculate base score
    if t1_hits >= 3:
        score = 9.0 + min(t1_hits - 3, 2) * 0.3
    elif t1_hits >= 2:
        score = 8.0 + min(t2_hits, 3) * 0.2
    elif t1_hits >= 1:
        score = 6.5 + min(t2_hits, 4) * 0.3
    elif t2_hits >= 4:
        score = 6.0 + min(t2_hits - 4, 3) * 0.2
    elif t2_hits >= 3:
        score = 5.0 + t3_hits * 0.1
    elif t2_hits >= 2:
        score = 4.0 + min(t3_hits, 4) * 0.15
    elif t2_hits >= 1:
        score = 2.5 + min(t3_hits, 5) * 0.2
    elif t3_hits >= 4:
        score = 2.5 + min(t3_hits - 4, 3) * 0.15
    elif t3_hits >= 2:
        score = 1.5 + (t3_hits - 2) * 0.2
    elif t3_hits >= 1:
        score = 1.0
    else:
        score = 0.5

    # Noise penalty
    if noise_hits > 0 and t1_hits == 0:
        score = min(score, max(score - noise_hits * 0.5, 0.5))

    return score


def score_backend_engineer(title, summary, text, source, article):
    """Score for backend_engineer_singapore profile."""

    t1 = [
        "kubernetes", "docker container", "container orchestration",
        "distributed system", "microservice", "golang",
        "cloud infrastructure", "cloud native", "devops",
        "aws lambda", "google cloud platform", "gcp ",
        "singapore tech", "singapore software", "grab tech",
        "sea group", "shopee engineering",
        "backend engineer", "backend developer",
        "api gateway", "service mesh", "istio", "envoy proxy",
        "serverless", "cloud function",
        "terraform", "infrastructure as code",
        "kafka", "message queue", "event driven",
        "ci/cd pipeline", "continuous integration",
    ]

    t2 = [
        "software engineering", "software developer",
        "python programming", "java developer", "rust programming",
        "postgresql", "redis", "mongodb", "database",
        "jenkins", "github actions",
        "prometheus", "grafana", "datadog", "observability",
        "load balancing", "caching", "scalability",
        "tech hiring", "software job", "tech layoff",
        "open source", "linux", "system design",
        "grpc", "rest api", "graphql",
        "southeast asia tech", "asean tech", "singapore startup",
        "fintech singapore", "digital bank",
        "ai infrastructure", "mlops",
        "cloud security", "zero trust",
        "node.js", "typescript",
        "kubernetes", "docker",
        "aws", "azure", "cloud computing",
        "site reliability", "sre ",
        "tech startup", "series a", "series b",
    ]

    t3 = [
        "technology", "software", "engineer", "developer",
        "cloud", "infrastructure", "deploy",
        "singapore", "southeast asia",
        "startup", "tech company",
        "artificial intelligence", "machine learning",
        "data engineering", "algorithm", "computing",
        "programming", "code", "coding",
        "cybersecurity", "network security",
    ]

    t1_hits = kw_match(text, t1)
    t2_hits = kw_match(text, t2)
    t3_hits = kw_match(text, t3)

    noise_topics = [
        "oil and gas", "petroleum", "drilling",
        "real estate", "property market", "construction project",
        "k-pop", "sports", "football", "basketball",
        "farming", "agriculture", "mining",
        "cooking", "restaurant", "food recipe",
        "fashion", "beauty", "lifestyle",
        "celebrity", "entertainment", "movie",
    ]
    noise_hits = kw_match(text, noise_topics)

    if t1_hits >= 3:
        score = 9.0 + min(t1_hits - 3, 2) * 0.3
    elif t1_hits >= 2:
        score = 8.0 + min(t2_hits, 3) * 0.2
    elif t1_hits >= 1:
        score = 6.5 + min(t2_hits, 4) * 0.3
    elif t2_hits >= 4:
        score = 6.0 + min(t2_hits - 4, 3) * 0.2
    elif t2_hits >= 3:
        score = 5.0 + t3_hits * 0.1
    elif t2_hits >= 2:
        score = 4.0 + min(t3_hits, 4) * 0.15
    elif t2_hits >= 1:
        score = 2.5 + min(t3_hits, 5) * 0.2
    elif t3_hits >= 4:
        score = 2.5
    elif t3_hits >= 2:
        score = 1.5
    elif t3_hits >= 1:
        score = 1.0
    else:
        score = 0.5

    if noise_hits > 0 and t1_hits == 0:
        score = min(score, max(score - noise_hits * 0.5, 0.5))

    # Special: if source is about pets/cats/animals and no real tech content, cap low
    animal_terms = ["cat ", "dog ", "pet ", "animal", "puppy", "kitten"]
    if any(t in text for t in animal_terms) and t1_hits == 0 and t2_hits < 2:
        score = min(score, 1.0)

    return score


def score_real_estate_dubai(title, summary, text, source, article):
    """Score for real_estate_dubai profile."""

    t1 = [
        "dubai real estate", "dubai property", "dubai housing",
        "emaar", "damac", "nakheel", "aldar properties", "meraas",
        "rera dubai", "dubai land department",
        "off-plan", "off plan", "dubai apartment", "dubai villa",
        "palm jumeirah", "dubai marina", "downtown dubai",
        "dubai creek", "dubai hills", "jumeirah village",
        "difc real estate", "luxury property dubai",
        "uae real estate", "uae property market",
        "golden visa uae", "dubai developer",
        "dubai construction", "abu dhabi real estate",
    ]

    t2 = [
        "real estate investment", "property investment",
        "property development", "real estate market",
        "commercial property", "residential property",
        "property price", "housing market",
        "gcc real estate", "gulf property",
        "uae economy", "dubai economy",
        "dubai tourism", "dubai infrastructure",
        "construction industry", "building material",
        "smart city dubai", "sustainable building",
        "property fund", "real estate fund", "reit",
        "foreign direct investment", "fdi uae",
        "interest rate", "mortgage rate",
        "wealth management", "high net worth",
        "luxury market", "premium property",
        "dubai expo", "abu dhabi", "sharjah",
        "dubai mall", "dubai airport",
        "uae regulation", "uae policy",
    ]

    t3 = [
        "real estate", "property", "housing", "construction",
        "dubai", "uae", "emirates", "abu dhabi",
        "gulf", "gcc", "middle east",
        "investment", "developer", "building",
        "rental", "lease", "tenant",
        "architecture", "urban development",
        "luxury", "premium",
        "qatar", "saudi arabia", "bahrain",
    ]

    t1_hits = kw_match(text, t1)
    t2_hits = kw_match(text, t2)
    t3_hits = kw_match(text, t3)

    noise_topics = [
        "k-pop", "gaming", "esports",
        "farming", "agriculture", "crop",
        "education", "school", "teacher",
        "medical", "hospital", "healthcare",
        "cryptocurrency", "bitcoin", "ethereum",
        "football", "basketball", "cricket",
        "recipe", "cooking", "restaurant menu",
    ]
    noise_hits = kw_match(text, noise_topics)

    if t1_hits >= 3:
        score = 9.0 + min(t1_hits - 3, 2) * 0.3
    elif t1_hits >= 2:
        score = 8.0 + min(t2_hits, 3) * 0.2
    elif t1_hits >= 1:
        score = 6.5 + min(t2_hits, 4) * 0.3
    elif t2_hits >= 4:
        score = 6.0 + min(t2_hits - 4, 3) * 0.2
    elif t2_hits >= 3:
        score = 5.0 + t3_hits * 0.1
    elif t2_hits >= 2:
        score = 4.0 + min(t3_hits, 4) * 0.15
    elif t2_hits >= 1:
        score = 2.5 + min(t3_hits, 5) * 0.2
    elif t3_hits >= 4:
        score = 2.5 + min(t3_hits - 4, 3) * 0.15
    elif t3_hits >= 2:
        score = 1.5 + (t3_hits - 2) * 0.2
    elif t3_hits >= 1:
        score = 1.0
    else:
        score = 0.5

    if noise_hits > 0 and t1_hits == 0:
        score = min(score, max(score - noise_hits * 0.5, 0.5))

    return score


PROFILE_SCORERS = {
    "math_teacher_texas": score_math_teacher,
    "backend_engineer_singapore": score_backend_engineer,
    "real_estate_dubai": score_real_estate_dubai,
}

PROFILE_DEFS = {
    "math_teacher_texas": {
        "role": "High school math teacher",
        "location": "Austin, Texas, USA",
    },
    "backend_engineer_singapore": {
        "role": "Backend software engineer at a cloud infrastructure company",
        "location": "Singapore",
    },
    "real_estate_dubai": {
        "role": "Real estate developer and investment manager",
        "location": "Dubai, UAE",
    },
}


def is_foreign_language(text):
    """Check if text is primarily non-English."""
    has_arabic = bool(re.search(r'[\u0600-\u06FF]{3,}', text))
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]{2,}', text))
    has_heavy_german = sum(1 for w in ['über', 'für', 'können', 'nicht', 'auch', 'noch', 'schon', 'einen', 'werden'] if w in text.lower()) >= 3
    return has_arabic or has_cjk or has_heavy_german


def is_sports_noise(text):
    """Detect sports content."""
    sports = ["nfl", "nba", "mlb", "nhl", "premier league", "la liga",
              "quarterback", "touchdown", "slam dunk", "home run",
              "aaron rodgers", "playoff", "super bowl", "world cup",
              "cricket match", "rugby match", "tennis open", "golf tournament",
              "transfer window", "free agent", "draft pick", "fantasy football"]
    return sum(1 for s in sports if s in text.lower()) >= 1


def score_article(article, profile_id):
    """Score a single article against a profile."""
    title = article.get("title") or ""
    summary = article.get("summary") or ""
    source = article.get("source") or ""
    text = f"{title} {summary}".lower()

    scorer = PROFILE_SCORERS[profile_id]
    score = scorer(title.lower(), summary.lower(), text, source.lower(), article)

    # Foreign language penalty
    if is_foreign_language(f"{title} {summary}"):
        if not (profile_id == "real_estate_dubai" and bool(re.search(r'[\u0600-\u06FF]{3,}', f"{title} {summary}"))):
            score = min(score, max(score - 2.0, 0.5))

    # Sports noise cap
    if is_sports_noise(text) and score < 5.0:
        score = min(score, 1.0)

    # Clamp
    score = max(0.0, min(10.0, round(score, 1)))

    # Build reason and think_text
    pdef = PROFILE_DEFS[profile_id]
    reason = build_reason(title, text, profile_id, score)
    think_text = build_think(title, summary, source, pdef, profile_id, score)
    think_tokens = len(think_text.split())

    return score, reason, think_text, think_tokens


def build_reason(title, text, profile_id, score):
    """Build a brief reason string."""
    if score >= 8.0:
        return "Directly relevant to profile's core professional domain and interests"
    elif score >= 6.0:
        return "Strong relevance to profile's field with multiple topic matches"
    elif score >= 4.0:
        return "Moderate relevance - touches on adjacent topics to profile's interests"
    elif score >= 2.0:
        return "Weak relevance - tangential connection to profile's domain"
    else:
        return "Not relevant to this profile's professional focus or interests"


def build_think(title, summary, source, pdef, profile_id, score):
    """Build detailed think_text."""
    parts = []
    title_short = title[:60] if len(title) > 60 else title
    parts.append(f"Article '{title_short}' from {source}.")
    parts.append(f"Profile is a {pdef['role']} in {pdef['location']}.")

    if score >= 7.0:
        parts.append("This article directly addresses topics within the profile's professional domain, making it highly actionable.")
    elif score >= 4.0:
        parts.append("The article has some overlap with the profile's interests or industry but is not a core match.")
    elif score >= 2.0:
        parts.append("Only tangential relevance detected through general topic overlap; not directly useful.")
    else:
        parts.append("The article's subject matter falls outside this profile's professional scope and personal interests.")

    if is_foreign_language(f"{title} {summary}"):
        parts.append("Foreign language content reduces practical relevance for this profile.")

    return " ".join(parts)


def main():
    results = []

    for profile_id in ["math_teacher_texas", "backend_engineer_singapore", "real_estate_dubai"]:
        print(f"\n{'='*60}")
        print(f"Scoring profile: {profile_id}")
        print(f"{'='*60}")

        profile_scores = []
        for ai, article in enumerate(articles):
            score, reason, think_text, think_tokens = score_article(article, profile_id)
            result = {
                "article_id": article["id"],
                "profile_id": profile_id,
                "score": score,
                "reason": reason,
                "think_text": think_text,
                "think_tokens": think_tokens,
            }
            profile_scores.append(result)

            if (ai + 1) % 200 == 0:
                print(f"  Scored {ai+1}/{len(articles)} articles...")

        results.extend(profile_scores)

        # Score distribution
        buckets = {"0-2": 0, "2-5": 0, "5-7": 0, "7-9": 0, "9-10": 0}
        for r in profile_scores:
            s = r["score"]
            if s < 2: buckets["0-2"] += 1
            elif s < 5: buckets["2-5"] += 1
            elif s < 7: buckets["5-7"] += 1
            elif s < 9: buckets["7-9"] += 1
            else: buckets["9-10"] += 1
        print(f"  Distribution: {buckets}")
        print(f"  Mean: {sum(r['score'] for r in profile_scores)/len(profile_scores):.2f}")

        # Top 10 articles
        top10 = sorted(profile_scores, key=lambda x: x["score"], reverse=True)[:10]
        print(f"  Top 10 scores:")
        for t in top10:
            art = next(a for a in articles if a["id"] == t["article_id"])
            print(f"    {t['score']:5.1f} — {art['title'][:70]}")

    # Save all results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: Saved {len(results)} scores to {OUTPUT_FILE}")
    print(f"Expected: {len(articles)} * 3 = {len(articles)*3}")


if __name__ == "__main__":
    main()
