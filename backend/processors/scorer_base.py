"""
STRAT_OS — Scorer Base Module
==============================
Shared constants, pattern lists, utility functions, and ScoringMemory
used by both scorers and fetchers. Extracted from scorer.py as part of
the B3.3 scorer retirement (Sprint 4).

Contains: pattern lists, utility functions, ScoringMemory, and the
ScorerBase class with shared Ollama client, score parsing, calibration.
"""

import collections
import json
import logging
import re
import time as _time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SHARED: Noise & staleness detection (applies to ALL categories)
# ═══════════════════════════════════════════════════════════════════

NOISE_EXACT = [
    'horoscope', 'lottery', 'astrology', 'gossip', 'kardashian',
    'reality tv', 'bachelor ', 'love island', 'bollywood',
    'hollywood', 'cricket score', 'football score',
    'recipe', 'weight loss', 'diet plan',
]
NOISE_PATTERNS = [
    r'jobs services career solutions',
    r'featured profile.*career booster',
    r'create job alert',
    r'sign up.*job alerts',
    r'browse.*open jobs',
    r'cookie policy|privacy policy|terms of service',
    r'page not found|404|access denied',
    # Job aggregator pages with big vacancy counts
    r'\d{3,}\s*(?:job|vacanc|position)',  # "159074 Knpc Job Vacancies"
    r'(?:job|vacanc|position).*\d{3,}',  # "Jobs, Employment 31..." (reverse)
    r'search \d+ ',  # "Search 978 Safety jobs"
]
# Instagram / social media noise (source-based)
SOCIAL_NOISE_PATTERNS = [
    r'#\w+love\b.*#\w+',  # "#kuwaitlove #kuwaitkw" = social spam
    r'#viral\b.*#insta',  # "#viral #insta #instamood"
    r'#insta\w*\s*#',  # "#instamood #"
    r'photo by .* \(@\w+\)',  # "Photo by XXX (@handle)" = Instagram post
    r'may be an image of',  # Instagram image description
    r'10k love you',  # Follower bait
]
# LinkedIn company profile pages (not news, not hiring)
COMPANY_PROFILE_PATTERNS = [
    r'^[\w\s]+ (?:corporation|company|group|bank)$',  # Exact: "Kuwait Petroleum Corporation"
    r'about\s*;\s*website',  # "About ; Website: http://..."
    r'about us.*website.*external link',  # LinkedIn about section
    r'company size.*\d+.*employees',  # "Company size: 10,001+ employees"
    r'industry.*oil and gas',  # LinkedIn "Industry: Oil and Gas" profile
    r'headquarters\s*;',  # LinkedIn profile section markers
    r'is considered a pioneer in.*banking',  # Wikipedia-style bank description
    r'is an? (?:international|kuwaiti|leading).*(?:company|bank|corporation).*(?:engaged|established|founded)',  # About page
]
# Stock price / finance data pages (not deals, not hiring)
STOCK_PAGE_PATTERNS = [
    r'stock price.*news.*quote.*history',  # "Stock Price, News, Quote & History"
    r'stock price.*news',  # "Stock Price, News..."
    r'trailing total returns',  # Yahoo Finance metric
    r'profit margin.*return on assets.*return on equity',  # Financial metrics
    r'diluted eps',  # Earnings per share — finance data page
    r'\bk\.s\.c\.p\b',  # "K.S.C.P." Kuwaiti stock listing
    r'\bsakp\b',  # "SAKP" Kuwaiti stock listing
    r'\b\w+\.kw\b.*stock',  # "NBK.KW stock"
    r'stock quote.*history.*news',  # Stock data page variants
    r'market cap.*\d+[mb]',  # Market cap listing
    r'share price monitor',  # Stock price monitoring tools
    r'stock screener|stock scanner',  # Stock screening tools
]
# Flight pages that match bank abbreviations (PR-NBK = flight callsign, not bank)
FLIGHT_PAGE_PATTERNS = [
    r'flight.*(?:status|tracking|history|departure|arrival)',
    r'(?:status|tracking|history).*flight',
    r'estimated.*(?:departure|arrival).*(?:time|gate)',
    r'(?:departure|arrival)\s*(?:time|gate)',
]
# Technical data pages that mention company names as ISP/AS numbers
TECHNICAL_DATA_PATTERNS = [
    r'aggregation.*(?:as\s*prepended|bgp|routing)',  # BGP routing tables
    r'as\d{4,}',  # AS number references (AS205787)
    r'(?:ipv4|ipv6).*(?:prefix|subnet|allocation)',
    r'points exchange.*(?:earn|miles|rewards)',  # Airline loyalty pages listing bank partners
    r'corporate actions? tracker',  # Investment platform corporate actions
]
# Non-bank businesses that mention banks in financing context
NON_BANK_BUSINESS_PATTERNS = [
    r'dent(?:al|el)\s*(?:center|clinic|care|plan)',  # Dental center/clinic
    r'hospital|medical\s*center|health\s*center',  # Medical facilities
    r'real\s*estate.*(?:financ|loan|mortgage)',  # Real estate financing
    r'car\s*(?:dealer|showroom|loan)',  # Car dealerships
    r'(?:furnitur|applianc).*(?:financ|installment)',  # Furniture stores with bank financing
]
STALE_PATTERNS = [
    r'posted.*201[0-8]', r'date.*201[0-8]', r'published.*201[0-8]',
    r'posted.*2019', r'date.*2019', r'posted.*202[0-3]',
    r'28-02-2018',
]
GENERIC_PAGE_PATTERNS = [
    r'^vacancy search$', r'^job search$', r'^careers?$',
    r'^home page$', r'^about us$', r'^contact us$',
    r'all rights reserved', r'copyright \d{4}',
    r'^\w+ group$', r'^\w+ corporate$',
]

# ═══════════════════════════════════════════════════════════════════
# GEOGRAPHIC VALIDATION — kill non-Kuwait items in Kuwait tab
# ═══════════════════════════════════════════════════════════════════

# Domains that are NEVER relevant to Kuwait career/finance intel
GARBAGE_DOMAINS = [
    'rightmove.co.uk', '6figr.com', 'ku.ac.ke', 'centralbank.go.ke',
    'trabajo.org', 'ph.trabajo.org', 'kfh.co.uk',  # KFH UK real estate
    'etihad.com/en-au', 'etihad.com/en-be',  # Non-Kuwait Etihad pages
    'fundsforngos.org', 'digitaldefynd.com',
    'naukri.com', 'shine.com', 'timesjobs.com',  # Indian job sites
    'reed.co.uk', 'totaljobs.com', 'cv-library.co.uk',  # UK job sites
    'jooble.org', 'careerjet.com', 'adzuna.com',  # Generic aggregators
    'salary.com', 'payscale.com', 'glassdoor.com/Salaries',  # Salary comparison
    'wattpad.com', 'steamdb.info', 'boards.4chan.org',  # Fiction/gaming/chan
    'tiktok.com', 'zoopla.co.uk',  # Social/UK property
    'muscogeenation.com', 'kdhnews.com',  # US local news
    'events.umich.edu', 'newsroom.lmu.edu',  # US university sites
    'ffnews.com', 'bebee.com',  # Irrelevant news/job aggregators
    'investorshub.advfn.com',  # Stock message boards
    # Indian classifieds / communities (not real job sources)
    'indiansinkuwait.com', 'expatriates.com', 'dubizzle.com',
    'q8india.com', 'kuwaitlocal.com', 'expat.com',
    # Reddit (not a news source)
    'reddit.com', 'old.reddit.com',
    # Flight tracking (false positive on bank abbreviations like PR-NBK)
    'flightaware.com', 'flightradar24.com', 'flightview.com',
    # Airline loyalty / points programs (list bank names as partners)
    'etihadguest.com', 'qmiles.com', 'miles-and-more.com',
    # Stock/share monitoring (not bank deals)
    'sharepricemonitor.com', 'tradingview.com', 'stockanalysis.com',
    # Wealth management / investment platforms (not Kuwait bank deals)
    'wealthsimple.com', 'etoro.com', 'robinhood.com',
    # Technical data pages (BGP routing tables mention company names as ISPs)
    'bgp.potaroo.net', 'bgp.he.net', 'peeringdb.com',
]

# Geographic signals that indicate non-Kuwait content
NON_KUWAIT_LOCATIONS = [
    # India
    'in india', 'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai',
    'hyderabad', 'pune', 'kolkata', 'lakhs', 'lakh', '₹', 'inr ',
    'naukri', 'pradhan',
    # Philippines
    'mandaluyong', 'manila', 'cebu', 'philippines', 'makati',
    # UK (except Kuwait-related UK news)
    'london bridge', 'london w1', 'london e1', 'london n1', 'london se',
    'lettings agent', 'estate agent', 'flat for rent',
    'leasehold', 'freehold', 'council tax',
    # Kenya
    'kenyatta', 'nairobi', 'kenya shilling',
    # Egypt (unless GCC context)
    'egyptian pound', 'cairo university',
    # Pakistan
    'karachi', 'lahore', 'islamabad', 'rawalpindi', 'pkr ',
    # Generic non-Gulf
    'work permit canada', 'green card', 'h1b visa',
]

# Instagram-specific: these signals must be present for Instagram items to score high
INSTAGRAM_STRONG_SIGNALS = [
    'hiring', 'vacancy', 'apply now', 'job opening', 'career',
    'fresh graduate', 'graduate program', 'internship',
    'kd ', 'cash gift', 'cashback', 'student offer', 'allowance transfer',
    'promotion', 'offer', 'deal', 'discount',
]


# ═══════════════════════════════════════════════════════════════════
# Language / script detection
# ═══════════════════════════════════════════════════════════════════

def _count_script_chars(text: str) -> dict:
    """Count characters by script type in a string."""
    counts = {'latin': 0, 'arabic': 0, 'cjk': 0, 'cyrillic': 0,
              'devanagari': 0, 'thai': 0, 'hangul': 0, 'other': 0}
    for ch in text:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            counts['arabic'] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or 0x31F0 <= cp <= 0x31FF:
            counts['cjk'] += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            counts['hangul'] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts['cyrillic'] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts['devanagari'] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts['thai'] += 1
        elif cp > 0x024F and cp not in range(0x2000, 0x3000):
            counts['other'] += 1
        elif ch.isalpha():
            counts['latin'] += 1
    return counts


# ── Location → language/script mapping ──

# Maps location keywords to: (allowed_scripts, ddg_region, language_label)
# 'latin' is always allowed (English uses Latin script)
_LOCATION_LANG_MAP = {
    # GCC / Middle East
    'kuwait':    ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'saudi':     ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'uae':       ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'emirates':  ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'dubai':     ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'abu dhabi': ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'qatar':     ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'bahrain':   ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'oman':      ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'iraq':      ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'jordan':    ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'lebanon':   ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'egypt':     ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'morocco':   ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'tunisia':   ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'algeria':   ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    'libya':     ({'latin', 'arabic'}, 'xa-ar', 'English or Arabic'),
    # East Asia
    'japan':     ({'latin', 'cjk'},      'jp-jp', 'English or Japanese'),
    'tokyo':     ({'latin', 'cjk'},      'jp-jp', 'English or Japanese'),
    'china':     ({'latin', 'cjk'},      'cn-zh', 'English or Chinese'),
    'beijing':   ({'latin', 'cjk'},      'cn-zh', 'English or Chinese'),
    'shanghai':  ({'latin', 'cjk'},      'cn-zh', 'English or Chinese'),
    'taiwan':    ({'latin', 'cjk'},      'tw-tzh', 'English or Chinese'),
    'korea':     ({'latin', 'hangul'},   'kr-kr', 'English or Korean'),
    'seoul':     ({'latin', 'hangul'},   'kr-kr', 'English or Korean'),
    # South Asia
    'india':     ({'latin', 'devanagari'}, 'in-en', 'English or Hindi'),
    'mumbai':    ({'latin', 'devanagari'}, 'in-en', 'English or Hindi'),
    'delhi':     ({'latin', 'devanagari'}, 'in-en', 'English or Hindi'),
    'bangalore': ({'latin', 'devanagari'}, 'in-en', 'English or Hindi'),
    # Russia / CIS
    'russia':    ({'latin', 'cyrillic'}, 'ru-ru', 'English or Russian'),
    'moscow':    ({'latin', 'cyrillic'}, 'ru-ru', 'English or Russian'),
    # SE Asia
    'thailand':  ({'latin', 'thai'},     'th-th', 'English or Thai'),
    'bangkok':   ({'latin', 'thai'},     'th-th', 'English or Thai'),
    # Europe + Americas + Africa (Latin script — same as English)
    'germany':   ({'latin'},  'de-de', 'English or German'),
    'berlin':    ({'latin'},  'de-de', 'English or German'),
    'munich':    ({'latin'},  'de-de', 'English or German'),
    'france':    ({'latin'},  'fr-fr', 'English or French'),
    'paris':     ({'latin'},  'fr-fr', 'English or French'),
    'spain':     ({'latin'},  'es-es', 'English or Spanish'),
    'uk':        ({'latin'},  'uk-en', 'English'),
    'london':    ({'latin'},  'uk-en', 'English'),
    'singapore': ({'latin'},  'sg-en', 'English'),
    'australia': ({'latin'},  'au-en', 'English'),
    'canada':    ({'latin'},  'ca-en', 'English'),
    'us':        ({'latin'},  'us-en', 'English'),
    'usa':       ({'latin'},  'us-en', 'English'),
    'brazil':    ({'latin'},  'br-pt', 'English or Portuguese'),
    'nigeria':   ({'latin'},  'xa-en', 'English'),
    'kenya':     ({'latin'},  'xa-en', 'English'),
    'south africa': ({'latin'}, 'xa-en', 'English'),
}

# Default when location doesn't match any key
_DEFAULT_LANG = ({'latin'}, 'wt-wt', 'English')


def location_to_lang(location: str) -> tuple:
    """Map a profile location string to (allowed_scripts, ddg_region, language_label).

    Returns the best match by checking if any known keyword appears in the location.
    Always includes 'latin' (English). Falls back to English-only worldwide.
    """
    if not location:
        return _DEFAULT_LANG
    loc = location.lower()
    for keyword, lang_tuple in _LOCATION_LANG_MAP.items():
        if keyword in loc:
            return lang_tuple
    return _DEFAULT_LANG


def _is_non_latin_title(title: str, allowed_scripts: set = None) -> bool:
    """Detect titles primarily in non-target scripts.
    allowed_scripts defaults to {'latin'} if not provided."""
    if not title:
        return False
    if allowed_scripts is None:
        allowed_scripts = {'latin'}
    counts = _count_script_chars(title)
    total = sum(counts.values())
    if total == 0:
        return False
    allowed_count = sum(counts.get(s, 0) for s in allowed_scripts)
    non_relevant = total - allowed_count
    return non_relevant / total > 0.25


def _is_non_target_language(title: str, body: str = '', allowed_scripts: set = None) -> bool:
    """Check if article is in a non-target language by examining title + body.
    allowed_scripts defaults to {'latin'} if not provided.
    More aggressive than _is_non_latin_title — checks body too, 15% threshold."""
    if allowed_scripts is None:
        allowed_scripts = {'latin'}
    text = f"{title} {body[:300]}"
    if not text.strip():
        return False
    counts = _count_script_chars(text)
    total = sum(counts.values())
    if total == 0:
        return False
    allowed_count = sum(counts.get(s, 0) for s in allowed_scripts)
    non_relevant = total - allowed_count
    return non_relevant / total > 0.15


# ═══════════════════════════════════════════════════════════════════
# Shared noise check — used by fetchers as pre-filter
# ═══════════════════════════════════════════════════════════════════

def _shared_noise_check(title: str, text: str, source: str = '', url: str = '', root: str = '',
                         allowed_scripts: set = None) -> Optional[Tuple[float, str]]:
    """Universal noise/staleness check. Returns (score, reason) or None."""
    title_lower = title.lower()
    source_lower = source.lower() if source else ''
    url_lower = url.lower() if url else ''

    # ── URL-based garbage domain filter ──
    for domain in GARBAGE_DOMAINS:
        if domain in url_lower:
            return 2.0, f"Garbage domain: {domain}"

    # ── Non-target language check (title + body) — allowed scripts from profile location ──
    if _is_non_latin_title(title, allowed_scripts=allowed_scripts):
        return 2.0, "Non-target language title"
    if _is_non_target_language(title, text, allowed_scripts=allowed_scripts):
        return 2.0, "Non-target language content"

    # ── Geographic validation for Kuwait items ──
    if root == 'kuwait':
        for location in NON_KUWAIT_LOCATIONS:
            if location in title_lower or location in text[:500].lower():
                return 2.5, f"Non-Kuwait content: '{location}'"

    for noise in NOISE_EXACT:
        if noise in text:
            return 1.0, f"Noise: '{noise}'"

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.0, "Noise: generic/aggregator page"

    for pattern in STALE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.5, "Stale: old or outdated content"

    for pattern in GENERIC_PAGE_PATTERNS:
        if re.search(pattern, title_lower):
            return 3.0, f"Generic page: {title[:40]}"

    # Junk summary detection — "no photo description" + keyword salad from social media scrapes
    if 'no photo description available' in text or 'no description available' in text:
        has_strong = any(s in text for s in INSTAGRAM_STRONG_SIGNALS)
        if not has_strong:
            return 3.0, "Junk content: scraped social media with no real description"

    # LinkedIn personal posts (not job listings or company announcements)
    if 'linkedin.com/posts/' in url_lower or 'linkedin.com/pulse/' in url_lower:
        has_strong_signal = any(s in text.lower() for s in ['hiring', 'job opening', 'fresh graduate', 'we are looking', 'join our team', 'apply now'])
        if not has_strong_signal:
            return 3.5, "LinkedIn personal post, not actionable"

    # LinkedIn company profile pages — not news, not hiring
    if 'linkedin.com/company/' in url_lower:
        has_hiring = any(s in text.lower() for s in ['hiring', 'job opening', 'fresh graduate', 'we are looking', 'join our team', 'apply now', 'open position'])
        if not has_hiring:
            return 3.5, "LinkedIn company profile page, not actionable"

    # Reddit posts — rarely actionable career/finance intel
    if 'reddit.com/' in url_lower:
        has_strong_signal = any(s in text.lower() for s in ['hiring', 'job opening', 'kuwait', 'salary', 'offer'])
        if not has_strong_signal:
            return 3.0, "Reddit post, not actionable intel"

    # Title ending with "'s Post" — social media reshare
    if title_lower.endswith("'s post") or "'s post" in title_lower:
        return 3.5, "Social media personal post"

    # Non-engineering / manual labor job postings (not relevant to CPEG student)
    non_eng_roles = r'(?:clerk|driver|technician|operator|helper|laborer|welder|plumber|carpenter|electrician|dispatch|warehouse|delivery|packing|cleaning|security guard|cook|cashier|receptionist|secretary|nurse|accountant)'
    if re.search(non_eng_roles, title_lower) and re.search(r'(?:hiring|job|vacancy|position|apply|opening)', title_lower):
        return 2.5, "Non-engineering job posting"

    # Yahoo Finance / stock quote pages — data pages, not news
    if 'finance.yahoo.com/quote/' in url_lower:
        return 3.0, "Stock quote page, not news"

    # Date-only titles (e.g. "2026-02-06") — no actual content
    if re.match(r'^\d{4}-\d{2}-\d{2}$', title.strip()):
        return 2.0, "Date-only title, no content"

    # "URGENT HIRING" style posts — usually non-CPEG trades
    if re.search(r'urgent(?:ly)?\s+hiring', title_lower):
        cpeg_roles = ['computer', 'software', 'it ', 'data', 'network', 'cloud', 'devops', 'engineer']
        if not any(r in text.lower() for r in cpeg_roles):
            return 3.0, "Urgent hiring for non-CPEG roles"

    # Facebook posts — rarely actionable intel
    if 'facebook.com/posts/' in url_lower or 'facebook.com/' in url_lower:
        has_strong = any(s in text.lower() for s in ['hiring', 'job opening', 'fresh graduate', 'kuwait'])
        if not has_strong:
            return 3.0, "Facebook post without actionable content"

    # Instagram social noise
    if source_lower == 'instagram' or 'instagram' in source_lower or 'instagram.com' in url_lower:
        for pattern in SOCIAL_NOISE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 2.5, "Instagram social spam"
        has_strong_signal = any(s in text.lower() for s in INSTAGRAM_STRONG_SIGNALS)
        text_l = text.lower()
        has_relevance = any(s in text_l for s in [
            'kuwait', 'computer engineer', 'software', 'cpeg', 'fresh graduate',
            'student offer', 'allowance transfer', 'cash gift', 'student deal',
            'kd ', 'nbk', 'boubyan', 'kfh', 'warba', 'kib',
        ])
        if not (has_strong_signal and has_relevance):
            return 3.5, "Instagram post without Kuwait/CPEG relevance"
        return 6.0, "Instagram (capped): relevant but unverified social source"

    # Facebook group posts
    if 'facebook.com/groups/' in url_lower:
        has_strong_signal = any(s in text.lower() for s in INSTAGRAM_STRONG_SIGNALS)
        if not has_strong_signal:
            return 3.0, "Facebook group post without actionable content"

    # Social spam regardless of source
    for pattern in SOCIAL_NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.5, "Social media spam"

    # LinkedIn company profile pages
    for pattern in COMPANY_PROFILE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 3.5, "Company profile page, not news"

    # Stock price / finance data pages
    for pattern in STOCK_PAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 3.0, "Stock/finance data page, not actionable"

    # Flight tracking pages (false positive on bank abbreviations like PR-NBK)
    for pattern in FLIGHT_PAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.0, "Flight tracking page, not relevant"

    # Technical data pages (BGP routing tables, airline loyalty partner lists)
    for pattern in TECHNICAL_DATA_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.0, "Technical data page, not news"

    # Non-bank businesses that mention banks only for financing
    for pattern in NON_BANK_BUSINESS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 3.5, "Non-bank business (dental/medical/real estate)"

    # Corporate M&A / oil deals (not career intel)
    if re.search(r'(?:sells?|divests?|acquir|stake|interest).*(?:offshore|brazil|deepwater|pre-salt)', text, re.IGNORECASE):
        return 3.5, "Corporate M&A deal, not career intel"
    if re.search(r'(?:shell|chevron|total|bp)\s+(?:sells?|divests?|sheds?)\s+.*(?:stake|interest)', text, re.IGNORECASE):
        return 3.5, "Oil company M&A, not career intel"

    # Salary survey / comparison pages (not job listings)
    if re.search(r'(?:average\s+)?salary.*(?:in india|₹|lakhs|inr)', text, re.IGNORECASE):
        return 3.0, "Indian salary survey, not Kuwait career"
    if re.search(r'salaries\s+\d{4}.*(?:average|range|lakh)', text, re.IGNORECASE):
        return 3.0, "Salary survey page, not job listing"
    if re.search(r'salaries?\s*$', title_lower) or 'salary' in title_lower and 'survey' in text.lower():
        if not any(s in text.lower() for s in ['kuwait', 'kwd', 'kd ']):
            return 3.0, "Non-Kuwait salary page"

    # Tender / procurement pages (not career)
    if re.search(r'tender\s*(?:no|number|#)?\s*\d|carrying out task|supply of\b', text, re.IGNORECASE):
        return 3.5, "Tender/procurement page, not career"

    # Empty form pages
    if re.search(r'first name.*last name.*(?:phone|email).*required.*submit', text, re.IGNORECASE):
        return 3.0, "Empty form page, no content"

    return None


# ═══════════════════════════════════════════════════════════════════
# ScoringMemory — feedback loop + high-score examples
# ═══════════════════════════════════════════════════════════════════

class ScoringMemory:
    """
    Scoring memory with user feedback loop (Tier 1).

    Combines two sources:
    1. memory.json — high-scoring items from past scans (automatic)
    2. user_feedback DB table — clicks, saves, dismissals, ratings (user behavior)

    The feedback loop injects personalized signals into the LLM prompt:
    - "User found these useful" (saves, high ratings, clicks)
    - "User dismissed these as noise" (dismissals, low ratings)
    - "Scoring corrections" (where user disagreed with AI score)
    """
    DEFAULT_PATH = Path(__file__).parent.parent / "data" / "memory.json"

    def __init__(self, memory_path=None, max_examples=10, min_score=8.5, db=None):
        self.memory_path = memory_path or self.DEFAULT_PATH
        self.max_examples = max_examples
        self.min_score = min_score
        self.db = db  # Database reference for feedback loop
        self._feedback_cache = None
        self._feedback_cache_time = None
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.memory_path.exists():
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self._save({"version": "1.0", "last_updated": "", "examples": []})

    def _load(self):
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"version": "1.0", "last_updated": "", "examples": []}

    def _save(self, data):
        data["last_updated"] = datetime.now().isoformat()
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_examples(self, category=None, count=3):
        examples = self._load().get("examples", [])
        if category:
            examples = [e for e in examples if e.get("category") == category]
        return examples[:count]

    def add_example(self, item, category=None):
        score = item.get("score", 0)
        url = item.get("url", "")
        if score < self.min_score or not url:
            return False
        data = self._load()
        examples = data.get("examples", [])
        if any(e.get("url") == url for e in examples):
            return False
        content = item.get("content", "") or item.get("summary", "")
        examples.insert(0, {
            "title": item.get("title", "")[:100],
            "content_preview": content[:200],
            "score": score,
            "reason": item.get("score_reason", "")[:100],
            "url": url,
            "category": category or "",
            "added_at": datetime.now().isoformat()
        })
        data["examples"] = examples[:self.max_examples]
        self._save(data)
        return True

    def format_for_prompt(self, count=3):
        examples = self.get_examples(count)
        if not examples:
            return ""
        lines = ["\n\n=== PAST HIGH-SCORING ITEMS ==="]
        for i, ex in enumerate(examples, 1):
            lines.append(f"\nExample {i}: \"{ex.get('title', '')}\" -> {ex.get('score', 0):.1f}")
        lines.append("\n=== END ===\n")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    # TIER 1: Feedback Loop
    # ═══════════════════════════════════════════════════════════════

    def _get_feedback(self):
        """Get feedback from DB with 10-minute cache to avoid repeated queries during a scoring run."""
        now = datetime.now()
        if (self._feedback_cache is not None
                and self._feedback_cache_time
                and (now - self._feedback_cache_time).total_seconds() < 600):
            return self._feedback_cache

        if not self.db:
            return {'positive': [], 'negative': [], 'corrections': []}

        try:
            self._feedback_cache = self.db.get_feedback_for_scoring(days=30, limit=15)
            self._feedback_cache_time = now
            return self._feedback_cache
        except Exception as e:
            logger.warning(f"Failed to load feedback for scoring: {e}")
            return {'positive': [], 'negative': [], 'corrections': []}

    def format_feedback_for_prompt(self, max_each=5):
        """
        Format user feedback as prompt context for LLM scoring.

        Returns a string to inject into the scoring prompt that teaches the LLM
        what this specific user considers useful vs noise.
        """
        feedback = self._get_feedback()
        positive = feedback.get('positive', [])[:max_each]
        negative = feedback.get('negative', [])[:max_each]
        corrections = feedback.get('corrections', [])[:max_each]

        if not positive and not negative and not corrections:
            return ""

        lines = ["\n=== USER FEEDBACK (learn from this) ==="]

        if positive:
            lines.append("\nItems the user found USEFUL (score similar items higher):")
            for p in positive:
                title = p.get('title', '')[:80]
                cat = p.get('category', '')
                action = p.get('action', '')
                lines.append(f'  + "{title}" [{cat}] (user {action}d)')

        if negative:
            lines.append("\nItems the user DISMISSED as noise (score similar items lower):")
            for n in negative:
                title = n.get('title', '')[:80]
                cat = n.get('category', '')
                ai = n.get('ai_score')
                ai_str = f", was scored {ai:.1f}" if ai else ""
                lines.append(f'  - "{title}" [{cat}]{ai_str}')

        if corrections:
            lines.append("\nScoring CORRECTIONS (user disagreed with AI):")
            for c in corrections:
                title = c.get('title', '')[:80]
                ai = c.get('ai_score', 0)
                user = c.get('user_score', 0)
                direction = "↑" if user > ai else "↓"
                lines.append(f'  {direction} "{title}" AI:{ai:.1f} → User:{user:.1f}')

        lines.append("=== END FEEDBACK ===\n")
        return "\n".join(lines)

    def get_feedback_summary(self):
        """Quick stats for logging."""
        feedback = self._get_feedback()
        return {
            'positive': len(feedback.get('positive', [])),
            'negative': len(feedback.get('negative', [])),
            'corrections': len(feedback.get('corrections', []))
        }


# ═══════════════════════════════════════════════════════════════════
# ScoringTimer — rolling average for dynamic timeouts (two-pass scoring)
# ═══════════════════════════════════════════════════════════════════

class ScoringTimer:
    """Tracks rolling average of LLM scoring call durations for dynamic timeouts.

    Used by the two-pass scoring pipeline:
    - Pass 1 (fast): rolling_avg + buffer, minimum floor
    - Pass 2 (slow): rolling_avg × multiplier + buffer
    Only successful completions are recorded (timeouts excluded).
    """

    def __init__(self, window: int = 20, seed_avg: float = 10.0, avg_cap: float = 60.0):
        self._times = collections.deque(maxlen=window)
        self._seed_avg = seed_avg
        self._avg_cap = avg_cap

    def record(self, elapsed: float):
        """Record a successful scoring call duration (seconds)."""
        self._times.append(elapsed)

    @property
    def rolling_avg(self) -> float:
        """Current rolling average, capped at avg_cap. Returns seed_avg if <5 samples."""
        if len(self._times) < 5:
            return self._seed_avg
        avg = sum(self._times) / len(self._times)
        if avg > self._avg_cap:
            logger.warning(f"ScoringTimer avg {avg:.1f}s exceeds cap {self._avg_cap}s — model may be too slow")
        return min(avg, self._avg_cap)

    @property
    def sample_count(self) -> int:
        return len(self._times)

    def fast_timeout(self, buffer: float = 30.0, minimum: float = 45.0) -> float:
        """Pass 1 timeout: rolling_avg + buffer, at least minimum."""
        return max(self.rolling_avg + buffer, minimum)

    def slow_timeout(self, multiplier: float = 3.0, buffer: float = 60.0) -> float:
        """Pass 2 timeout: rolling_avg × multiplier + buffer."""
        return self.rolling_avg * multiplier + buffer


# ═══════════════════════════════════════════════════════════════════
# ScorerBase — shared Ollama client, score parsing, calibration
# ═══════════════════════════════════════════════════════════════════

class ScorerBase:
    """Base class with shared scoring infrastructure.

    Provides: Ollama communication, availability check, score calibration,
    forbidden-5.0 guard, score category classification, and stat tracking.
    Subclasses implement the actual scoring logic (rule engine + prompt building).
    """

    def __init__(self, config: Dict[str, Any], db=None):
        scoring_config = config.get("scoring", config) if "scoring" in config else config

        self.model = scoring_config.get("model", "stratos-scorer-v1")
        self.host = scoring_config.get("ollama_host", "http://localhost:11434")
        self.inference_model = scoring_config.get("inference_model", "qwen3:30b-a3b")
        self.forbidden_score = scoring_config.get("forbidden_score", 5.0)

        self.critical_min = scoring_config.get("critical_min", 9.0)
        self.high_min = scoring_config.get("high_min", 7.0)
        self.medium_min = scoring_config.get("medium_min", 5.0)

        self.memory = ScoringMemory(
            max_examples=scoring_config.get("memory_max_examples", 10),
            min_score=scoring_config.get("memory_min_score", 8.5),
            db=db
        )
        self.few_shot_count = scoring_config.get("few_shot_examples", 3)
        self._available = None
        self._stats = {'rule': 0, 'llm': 0, 'truncated': 0}

        # Two-pass scoring timer
        timeout_cfg = scoring_config.get("timeout", {})
        self.scoring_timer = ScoringTimer(
            window=timeout_cfg.get("rolling_window", 20),
            seed_avg=timeout_cfg.get("seed_avg", 10),
            avg_cap=timeout_cfg.get("avg_cap", 60),
        )
        self._timeout_cfg = timeout_cfg

        # V1 calibration corrects +1.33 inflation in the base Qwen3-8B model.
        # V2+ models are trained with proper calibration and MUST NOT have this applied.
        self._calibration_table = None
        is_v2_plus = "v2" in self.model.lower() or "scorer-v2" in self.model.lower()
        if not is_v2_plus:
            cal_path = Path(__file__).parent.parent / "data" / "v1_calibration_table.json"
            if cal_path.exists():
                try:
                    with open(cal_path) as f:
                        self._calibration_table = json.load(f)
                    logger.info(f"V1 calibration table loaded ({len(self._calibration_table)} entries) for model '{self.model}'")
                except Exception as e:
                    logger.warning(f"Failed to load calibration table: {e}")
        else:
            logger.info(f"V2+ model detected ('{self.model}') — skipping V1 calibration table")

    def is_available(self) -> bool:
        """Check if Ollama LLM is available."""
        if self._available is True:
            return True
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                full_names = [m.get("name", "") for m in models]
                base_names = [n.split(":")[0] for n in full_names]
                model_base = self.model.split(":")[0]
                self._available = (
                    self.model in full_names or
                    model_base in base_names or
                    f"{self.model}:latest" in full_names
                )
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def _call_ollama(self, prompt, system_prompt=None, cancel_check=None, timeout_seconds=None):
        """Call Ollama LLM with streaming, cancellation, and dynamic timeout.

        Returns (clean_text, think_block) tuple.
        - clean_text: response with <think> stripped (for SCORE/REASON parsing)
        - think_block: the raw <think>...</think> content (for logging/debugging)
        - "__TIMEOUT__" as clean_text signals the call exceeded timeout_seconds
        """
        call_start = _time.time()
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_predict": 512}
            }
            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=(10, 30),  # (connect_timeout, read_timeout_per_chunk)
                stream=True
            )
            if response.status_code != 200:
                return None, ""

            full_text = []
            chunk_count = 0
            done_received = False
            _timed_out = False
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk_count += 1
                    # Cancel check every 10 chunks
                    if cancel_check and chunk_count % 10 == 0 and cancel_check():
                        response.close()
                        logger.debug("Ollama call cancelled mid-stream")
                        return None, ""
                    # Total timeout check every 5 chunks
                    if timeout_seconds and chunk_count % 5 == 0:
                        elapsed = _time.time() - call_start
                        if elapsed > timeout_seconds:
                            response.close()
                            _timed_out = True
                            logger.info(f"[TIMEOUT] Ollama call exceeded {timeout_seconds:.0f}s (elapsed={elapsed:.1f}s, {chunk_count} chunks)")
                            break
                    try:
                        data = json.loads(line)
                        full_text.append(data.get("response", ""))
                        if data.get("done", False):
                            done_received = True
                            break
                    except json.JSONDecodeError:
                        continue
            except requests.exceptions.ReadTimeout:
                elapsed = _time.time() - call_start
                logger.warning(f"[TIMEOUT] Ollama read timeout — no chunk for 30s (elapsed={elapsed:.1f}s)")
                self._stats['truncated'] = self._stats.get('truncated', 0) + 1
                _timed_out = True
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as stream_err:
                logger.warning(f"[TRUNCATED] Ollama stream interrupted: {stream_err}")
                self._stats['truncated'] = self._stats.get('truncated', 0) + 1

            # Record timing for successful completions only
            call_elapsed = _time.time() - call_start
            if done_received:
                self.scoring_timer.record(call_elapsed)

            if _timed_out:
                return "__TIMEOUT__", ""

            if chunk_count > 0 and not done_received:
                logger.warning(f"[TRUNCATED] Ollama stream ended without done signal ({chunk_count} chunks)")
                self._stats['truncated'] = self._stats.get('truncated', 0) + 1

            text = "".join(full_text).strip()

            think_block = ""
            think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
            if think_match:
                think_block = think_match.group(1).strip()

            clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            if chunk_count > 0 and not clean and think_block:
                logger.warning("[TRUNCATED] Response consumed by think block — no SCORE/REASON output")
                self._stats['truncated'] = self._stats.get('truncated', 0) + 1

            return clean, think_block
        except Exception as e:
            logger.debug(f"Ollama call failed: {e}")
        return None, ""

    def _calibrate_score(self, raw_score: float) -> float:
        """Apply isotonic calibration to correct systematic score inflation."""
        if not self._calibration_table:
            return raw_score

        key = str(round(raw_score * 10) / 10)
        if key in self._calibration_table:
            return self._calibration_table[key]

        lo = round(raw_score * 10) / 10
        hi = lo + 0.1
        lo_key, hi_key = str(round(lo, 1)), str(round(hi, 1))
        if lo_key in self._calibration_table and hi_key in self._calibration_table:
            lo_val = self._calibration_table[lo_key]
            hi_val = self._calibration_table[hi_key]
            frac = (raw_score - lo) / 0.1 if 0.1 > 0 else 0
            return lo_val + frac * (hi_val - lo_val)

        return raw_score

    @staticmethod
    def apply_forbidden_50(score: float) -> float:
        """Nudge scores away from the forbidden 5.0 boundary."""
        if 4.8 <= score <= 5.2:
            return 5.3 if score >= 5.0 else 4.8
        return score

    def get_score_category(self, score: float) -> str:
        """Classify a score into critical/high/medium/noise."""
        if score >= self.critical_min:
            return 'critical'
        elif score >= self.high_min:
            return 'high'
        elif score >= self.medium_min:
            return 'medium'
        return 'noise'
