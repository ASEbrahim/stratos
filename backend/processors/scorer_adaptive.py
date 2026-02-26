"""
STRAT_OS - Adaptive Profile Scorer
====================================
A profile-adaptive scorer that builds relevance rules dynamically from the
user's categories, keywords, role, and context — rather than using hardcoded
Kuwait/CPEG-specific patterns.

Activated when the user's profile doesn't match the default CPEG-in-Kuwait
setup. The original scorer.py is preserved for that highly-tuned use case.

Scoring bands are identical to the original:
  9-10:  Critical / actionable
  7-8.9: High importance
  5-6.9: Medium (visible but not urgent)
  0-4.9: Noise (filtered from main view)

Interface matches AIScorer exactly: score_items(), score_item(), get_score_category()
"""

import json
import logging
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Import truly universal patterns from the original scorer
# These are domain-agnostic (spam, garbage sites, empty pages)
# ═══════════════════════════════════════════════════════════════════
from processors.scorer_base import (
    GARBAGE_DOMAINS, NOISE_EXACT, NOISE_PATTERNS, STALE_PATTERNS,
    GENERIC_PAGE_PATTERNS, SOCIAL_NOISE_PATTERNS,
    COMPANY_PROFILE_PATTERNS, STOCK_PAGE_PATTERNS,
    FLIGHT_PAGE_PATTERNS, TECHNICAL_DATA_PATTERNS,
    ScoringMemory, ScorerBase,
    _is_non_latin_title, _is_non_target_language,
    location_to_lang,
)


# ═══════════════════════════════════════════════════════════════════
# Universal action signals — domain-agnostic concepts
# ═══════════════════════════════════════════════════════════════════

HIRING_SIGNALS = [
    'hiring', 'vacancy', 'vacancies', 'job opening', 'apply now',
    'we are recruiting', 'career opportunity', 'job opportunity',
    'open position', 'now hiring', 'recruitment', 'job post',
    'accepting applications', 'applications open', 'apply before',
    'submit your cv', 'submit your resume', 'join our team',
    'looking for', 'seeking', 'positions available', 'walk-in interview',
]

ENTRY_LEVEL_SIGNALS = [
    'fresh graduate', 'graduate program', 'graduate trainee',
    'recent graduate', 'entry level', 'entry-level', 'junior',
    'trainee', 'new graduate', 'campus recruitment',
    'graduate recruitment', 'internship', 'no experience required',
    'early career', 'co-op', 'work placement',
    '0-2 years', '0-1 years', 'zero experience',
]

SENIOR_LEVEL_SIGNALS = [
    'senior ', 'sr. ', 'lead ', 'principal ', 'staff ',
    'manager ', 'director ', 'head of ', 'vp ',
    'chief ', '5+ years', '7+ years', '10+ years',
]

OFFER_DEAL_SIGNALS = [
    'offer', 'promotion', 'deal', 'discount', 'cashback', 'cash back',
    'bonus', 'free ', 'gift', 'reward', 'savings',
    'sign-up', 'transfer bonus', 'student offer', 'student deal',
    'scholarship', 'financial aid', 'grant', 'bursary',
]

BREAKTHROUGH_SIGNALS = [
    'breakthrough', 'discovered', 'invented', 'paradigm',
    'first ever', 'revolutionary', 'groundbreaking',
    'landmark study', 'major finding', 'clinical trial results',
    'fda approved', 'new treatment', 'cure for',
]

NEWS_ACTION_SIGNALS = [
    'new release', 'launched', 'introduces', 'announces',
    'published', 'study finds', 'research shows', 'report reveals',
    'regulation', 'policy change', 'law passed', 'guidelines',
    'partnership', 'acquisition', 'merger', 'expansion',
]

# Experience requirement regex patterns (universal)
EXPERIENCE_REGEX = [
    r'\b\d+\s*[\-–—]\s*\d+\s*years?\b',
    r'\b\d+\+?\s*years?\s*(?:of\s+)?experience',
    r'\bminimum\s+\d+\s*years?',
    r'\b(?:at least|min\.?)\s+\d+\s*years?',
    r'\b[5-9]\+?\s*(?:yrs?|years?)\b',
    r'\b\d{2}\+?\s*(?:yrs?|years?)\b',
]


# ═══════════════════════════════════════════════════════════════════
# Keyword Index Builder
# ═══════════════════════════════════════════════════════════════════

class KeywordIndex:
    """
    Builds a weighted keyword dictionary from the user's profile and categories.
    Used for relevance scoring — items matching more keywords score higher.
    """

    def __init__(self, config: Dict[str, Any]):
        self.profile = config.get('profile', {})
        self.categories = config.get('dynamic_categories', [])
        self.location = self.profile.get('location', '').strip().lower()
        self.role = self.profile.get('role', '').strip().lower()
        self.context = self.profile.get('context', '').strip().lower()

        # category_id -> set of keywords (lowercased)
        self.category_keywords: Dict[str, Set[str]] = {}
        # All keywords merged (for universal matching)
        self.all_keywords: Set[str] = set()
        # Category metadata
        self.category_meta: Dict[str, Dict] = {}

        self._build()

    def _build(self):
        """Build keyword sets from categories and profile."""
        STOP_WORDS = {
            'and', 'or', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'is',
            'at', 'by', 'on', 'with', 'as', 'it', 'be', 'this', 'that',
            'from', 'are', 'was', 'were', 'been', 'has', 'have', 'had',
            'not', 'but', 'if', 'they', 'we', 'you', 'he', 'she',
            'news', 'jobs', 'deals', '&', 'new', 'top', 'best',
            # Generic domain words that match too broadly when split from labels
            'trends', 'methods', 'strategies', 'techniques', 'services',
            'studies', 'research', 'opportunities', 'programs', 'development',
            'management', 'systems', 'practices', 'approaches', 'solutions',
            'training', 'current', 'latest', 'recent', 'modern', 'advanced',
            'job', 'career', 'work', 'hiring', 'positions',
        }

        for cat in self.categories:
            if cat.get('enabled') is False:
                continue
            cat_id = cat.get('id', '').lower()
            items = [i.strip().lower() for i in cat.get('items', []) if i.strip()]
            label = cat.get('label', '').strip().lower()

            keywords = set()

            # Category items are the PRIMARY keywords (highest signal)
            for item in items:
                keywords.add(item)
                # Also add individual words from multi-word items
                for word in item.split():
                    if word not in STOP_WORDS and len(word) > 2:
                        keywords.add(word)

            # Label words as secondary keywords
            for word in label.split():
                if word not in STOP_WORDS and len(word) > 2:
                    keywords.add(word)

            self.category_keywords[cat_id] = keywords
            self.all_keywords.update(keywords)
            self.category_meta[cat_id] = {
                'label': label,
                'scorer_type': cat.get('scorer_type', 'auto'),
                'root': cat.get('root', 'global'),
                'items': items,
            }

        # Add role and context words as universal keywords
        for text in [self.role, self.context]:
            for word in text.split():
                word_clean = word.strip('.,;:!?()[]{}"\'-')
                if word_clean not in STOP_WORDS and len(word_clean) > 2:
                    self.all_keywords.add(word_clean)

        # Extract location parts for geographic matching
        self.location_parts = set()
        for part in re.split(r'[,\s]+', self.location):
            part = part.strip().lower()
            if part and len(part) > 2:
                self.location_parts.add(part)

        logger.info(f"AdaptiveScorer keyword index: {len(self.all_keywords)} keywords "
                    f"across {len(self.category_keywords)} categories")

    def match_category(self, text: str, category_id: str) -> Tuple[int, List[str]]:
        """
        Count keyword matches for a specific category.
        Returns (match_count, matched_keywords).
        """
        text_lower = text.lower()
        keywords = self.category_keywords.get(category_id, set())
        matched = []
        for kw in keywords:
            if len(kw) <= 4:
                # Short keywords need word boundary matching
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    matched.append(kw)
            else:
                if kw in text_lower:
                    matched.append(kw)
        return len(matched), matched

    def match_any(self, text: str) -> Tuple[int, List[str]]:
        """Count matches against ALL keywords."""
        text_lower = text.lower()
        matched = []
        for kw in self.all_keywords:
            if len(kw) <= 4:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    matched.append(kw)
            else:
                if kw in text_lower:
                    matched.append(kw)
        return len(matched), matched

    def match_location(self, text: str) -> bool:
        """Check if text mentions user's location."""
        text_lower = text.lower()
        return any(loc in text_lower for loc in self.location_parts if loc)


# ═══════════════════════════════════════════════════════════════════
# Universal Noise Check (domain-agnostic subset of _shared_noise_check)
# ═══════════════════════════════════════════════════════════════════

def _universal_noise_check(title: str, text: str, source: str = '',
                            url: str = '', root: str = '',
                            allowed_scripts: set = None) -> Optional[Tuple[float, str]]:
    """
    Domain-agnostic noise detection. Only fires on content that is UNIVERSALLY
    garbage regardless of the user's field — spam, broken pages, data tables, etc.

    DOES NOT include: non-engineering role filter, Kuwait geographic checks,
    CPEG-specific patterns, or any domain-specific content gates.
    """
    title_lower = title.lower()
    source_lower = source.lower() if source else ''
    url_lower = url.lower() if url else ''

    # ── Garbage domains ──
    for domain in GARBAGE_DOMAINS:
        if domain in url_lower:
            return 2.0, f"Garbage domain: {domain}"

    # ── Exact noise terms (horoscopes, gossip, etc.) ──
    for noise in NOISE_EXACT:
        if noise in text:
            return 1.0, f"Noise: '{noise}'"

    # ── Aggregator / form / error pages ──
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.0, "Noise: generic/aggregator page"

    # ── Stale content ──
    for pattern in STALE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.5, "Stale: old or outdated content"

    # ── Generic page titles ──
    for pattern in GENERIC_PAGE_PATTERNS:
        if re.search(pattern, title_lower):
            return 3.0, f"Generic page: {title[:40]}"

    # ── Junk summary (scraped social with no content) ──
    if 'no photo description available' in text or 'no description available' in text:
        return 3.0, "Junk content: no real description"

    # ── Social media personal posts (not news) ──
    if title_lower.endswith("'s post") or "'s post" in title_lower:
        return 3.5, "Social media personal post"

    # ── LinkedIn personal posts ──
    if 'linkedin.com/posts/' in url_lower or 'linkedin.com/pulse/' in url_lower:
        has_signal = any(s in text.lower() for s in HIRING_SIGNALS[:10])
        if not has_signal:
            return 3.5, "LinkedIn personal post, not actionable"

    # ── Social spam ──
    for pattern in SOCIAL_NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.5, "Social media spam"

    # ── Stock quote pages ──
    for pattern in STOCK_PAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 3.0, "Stock/finance data page"

    # ── Flight tracking pages ──
    for pattern in FLIGHT_PAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.0, "Flight tracking page"

    # ── Technical data pages (BGP, etc.) ──
    for pattern in TECHNICAL_DATA_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 2.0, "Technical data page"

    # ── Yahoo Finance quote pages ──
    if 'finance.yahoo.com/quote/' in url_lower:
        return 3.0, "Stock quote page, not news"

    # ── Date-only titles ──
    if re.match(r'^\d{4}-\d{2}-\d{2}$', title.strip()):
        return 2.0, "Date-only title, no content"

    # ── Empty form pages ──
    if re.search(r'first name.*last name.*(?:phone|email).*required.*submit', text, re.IGNORECASE):
        return 3.0, "Empty form page, no content"

    # ── Company profile page (not news) ──
    for pattern in COMPANY_PROFILE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 3.5, "Company profile page, not news"

    # ── Non-target language (title + body) — allowed scripts derived from profile location ──
    if _is_non_latin_title(title, allowed_scripts=allowed_scripts):
        return 2.0, "Non-target language title"
    if _is_non_target_language(title, text, allowed_scripts=allowed_scripts):
        return 2.0, "Non-target language content"

    # ── Job aggregator listing pages ──
    if re.search(r'showing\s+\d+-\d+\s+of\s+\d+\s+job', text, re.IGNORECASE):
        return 4.0, "Job aggregator listing page"
    if re.search(r'\d{3,}\s*(?:job|vacanc|position)', title_lower):
        return 3.5, "Job aggregator page"
    # Catch "21 Part Time Psychologist Jobs", "69 Clinical psychologist jobs", "50+ offres"
    if re.search(r'^\d+\+?\s+(?:part.time|full.time|remote|clinical|research|federal|freelance)?\s*\w+\s+jobs?\b', title_lower):
        return 3.5, "Job aggregator listing (N jobs pattern)"
    if re.search(r'\bhiring now:?\s*\d+\s', title_lower):
        return 3.5, "Job aggregator listing (Hiring Now: N)"
    if re.search(r'\d+\+?\s+(?:offres?|emplois?|postes?|vagas?|stellenangebote)\b', title_lower):
        return 3.5, "Foreign-language job aggregator"
    # Generic title: just "X jobs in Y" with no specificity
    if re.search(r'^[\w\s]*jobs?\s+in\s+\w', title_lower) and len(title) < 60:
        return 3.5, "Generic job search page"

    return None


# ═══════════════════════════════════════════════════════════════════
# Profile-Adaptive Scoring Functions
# ═══════════════════════════════════════════════════════════════════

def _has_experience_requirement(text: str) -> bool:
    """Check if text contains experience requirements (universal)."""
    for pattern in EXPERIENCE_REGEX:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _detect_student(profile: Dict[str, Any]) -> bool:
    """Detect if the user is a student based on profile signals.
    
    Role field takes priority over context, because context is AI-generated
    and may contain words like 'professional' or 'specialist' that would
    falsely override a clear 'student' in the role.
    """
    role = profile.get('role', '').lower()
    context = profile.get('context', '').lower()
    exp = profile.get('experience_level', '').lower()

    student_signals = ['student', 'bachelor', 'undergraduate', 'graduate student',
                       'master student', 'phd student', 'university', 'college',
                       'studying', 'freshman', 'sophomore', 'junior year', 'senior year']
    experienced_signals = ['head', 'manager', 'director', 'lead', 'senior', 'principal',
                          'specialist', 'professional', 'years experience', 'engineer at',
                          'working at', 'employed at', 'consultant']

    # Check if role explicitly says student — this is the strongest signal
    role_says_student = any(s in role for s in student_signals)
    role_says_experienced = any(s in role for s in experienced_signals)
    
    # If role explicitly says "student", trust it even if AI-generated context
    # contains experienced-sounding words like "professional certifications"
    if role_says_student and not role_says_experienced:
        return True
    
    # Otherwise fall back to checking all fields
    is_student = any(s in role or s in context or s in exp for s in student_signals)
    is_experienced = any(s in role for s in experienced_signals)  # Only check role for experienced

    return is_student and not is_experienced


def score_career_adaptive(title: str, text: str, url: str,
                          keyword_index: KeywordIndex, category_id: str,
                          is_student: bool) -> Tuple[float, str, bool]:
    """
    Profile-adaptive career scoring.
    Uses the user's category keywords instead of hardcoded CPEG patterns.
    """
    text_lower = text.lower()
    title_lower = title.lower()

    # Match against category's own keywords
    cat_matches, cat_matched = keyword_index.match_category(text, category_id)
    title_cat_matches = sum(1 for kw in keyword_index.category_keywords.get(category_id, set())
                       if kw in title_lower)

    has_hiring = any(s in text_lower for s in HIRING_SIGNALS)
    has_entry = any(s in text_lower for s in ENTRY_LEVEL_SIGNALS)
    has_senior_title = any(s in title_lower for s in SENIOR_LEVEL_SIGNALS)
    has_senior_content = any(s in text_lower for s in SENIOR_LEVEL_SIGNALS)
    has_experience = _has_experience_requirement(text_lower)
    has_location = keyword_index.match_location(text)

    # ── Student path ──
    if is_student:
        # Senior/Lead/VP/Director in TITLE → hard kill regardless of keywords
        if has_senior_title:
            return 3.5, f"Senior-level title (user is student): {title[:50]}", True

        # Experience required in content (without entry-level counterbalance)
        if has_experience and not has_entry:
            if cat_matches >= 2:
                return 3.5, f"Relevant but experience required ({', '.join(cat_matched[:3])})", True
            return 3.5, "Experience required (user is student)", True

        # Critical: keyword match + hiring + entry level + location
        if cat_matches >= 2 and has_hiring and has_entry and has_location:
            label = ', '.join(cat_matched[:3])
            return 9.5, f"Entry-level hiring in target location: {label}", True

        # Critical: keyword match + hiring + entry level
        if cat_matches >= 2 and has_hiring and has_entry:
            label = ', '.join(cat_matched[:3])
            return 9.0, f"Entry-level hiring match: {label}", True

        # High: keyword match + hiring + location
        if cat_matches >= 2 and has_hiring and has_location:
            if has_experience:
                return 3.5, "Hiring in location but experience required", True
            label = ', '.join(cat_matched[:3])
            return 8.5, f"Relevant hiring in target location: {label}", True

        # High: keyword match + entry level (no explicit hiring signal)
        if cat_matches >= 2 and has_entry:
            return 8.0, f"Entry-level opportunity: {', '.join(cat_matched[:3])}", True

        # High: keyword match + hiring (general)
        if cat_matches >= 2 and has_hiring:
            if has_experience:
                return 3.5, "Hiring but experience required", True
            # Only penalize for senior signals if TITLE also looks like a job posting
            has_hiring_title = any(s in title_lower for s in ['hiring', 'vacancy', 'job opening',
                                                               'apply now', 'open position', 'join our'])
            if has_senior_content and has_hiring_title:
                return 4.5, f"Senior-level job posting ({', '.join(cat_matched[:3])})", True
            return 7.5, f"Hiring with relevance: {', '.join(cat_matched[:3])}", True

    else:
        # ── Experienced user path ──
        if has_entry and not has_experience:
            return 3.5, "Entry-level role (user is experienced)", True

        if cat_matches >= 2 and has_hiring:
            return 9.0, f"Relevant hiring: {', '.join(cat_matched[:3])}", True

        if cat_matches >= 2 and has_location:
            return 8.0, f"Relevant opportunity in target location: {', '.join(cat_matched[:3])}", True

    # ── Common path (both student and experienced) ──

    # Medium-high: strong keyword match in title
    if title_cat_matches >= 2:
        return 7.0, f"Strong title match: {', '.join(cat_matched[:3])}", True

    # Medium: decent keyword match
    if cat_matches >= 3:
        return 6.5, f"Multiple keyword matches: {', '.join(cat_matched[:3])}", False

    if cat_matches >= 2:
        if has_hiring or has_location:
            return 6.0, f"Keyword + context match: {', '.join(cat_matched[:3])}", False
        return 5.5, f"Moderate relevance: {', '.join(cat_matched[:3])}", False

    if cat_matches == 1:
        return 4.5, f"Weak match: {cat_matched[0]}", False

    return 3.5, "No keyword relevance to category", True


def score_finance_adaptive(title: str, text: str, url: str,
                           keyword_index: KeywordIndex, category_id: str) -> Tuple[float, str, bool]:
    """Profile-adaptive finance/banking scoring."""
    text_lower = text.lower()

    cat_matches, cat_matched = keyword_index.match_category(text, category_id)
    has_offer = any(s in text_lower for s in OFFER_DEAL_SIGNALS)
    has_hiring = any(s in text_lower for s in HIRING_SIGNALS)
    has_location = keyword_index.match_location(text)

    # Critical: tracked institution + offer/deal
    if cat_matches >= 2 and has_offer:
        return 9.5, f"Offer from tracked entity: {', '.join(cat_matched[:3])}", True

    if cat_matches >= 1 and has_offer and has_location:
        return 9.0, f"Deal in target location: {', '.join(cat_matched[:3])}", True

    # High: tracked institution + hiring (tech roles at banks)
    if cat_matches >= 2 and has_hiring:
        return 8.0, f"Hiring at tracked entity: {', '.join(cat_matched[:3])}", True

    if cat_matches >= 1 and has_offer:
        return 7.5, f"Relevant offer: {', '.join(cat_matched[:3])}", True

    # Medium: keyword match
    if cat_matches >= 2:
        return 6.5, f"Finance relevance: {', '.join(cat_matched[:3])}", False

    if cat_matches == 1:
        if has_offer or has_hiring:
            return 6.0, f"Partial match + signal: {cat_matched[0]}", False
        return 4.5, f"Weak finance match: {cat_matched[0]}", False

    return 3.5, "No finance relevance", True


def score_tech_adaptive(title: str, text: str,
                        keyword_index: KeywordIndex, category_id: str) -> Tuple[float, str, bool]:
    """Profile-adaptive tech/research scoring."""
    text_lower = text.lower()
    title_lower = title.lower()

    cat_matches, cat_matched = keyword_index.match_category(text, category_id)
    has_breakthrough = any(s in text_lower for s in BREAKTHROUGH_SIGNALS)
    has_news_action = any(s in text_lower for s in NEWS_ACTION_SIGNALS)

    # Critical: keyword + breakthrough
    if cat_matches >= 2 and has_breakthrough:
        return 9.5, f"Breakthrough in tracked area: {', '.join(cat_matched[:3])}", True

    # High: keyword + notable news
    if cat_matches >= 2 and has_news_action:
        return 8.0, f"Notable development: {', '.join(cat_matched[:3])}", True

    # High: strong title match with any context
    title_matches = sum(1 for kw in keyword_index.category_keywords.get(category_id, set())
                       if kw in title_lower)
    if title_matches >= 2:
        if has_breakthrough or has_news_action:
            return 8.5, f"Strong title: {', '.join(cat_matched[:3])}", True
        return 7.0, f"Relevant topic: {', '.join(cat_matched[:3])}", True

    # Medium: decent keyword match
    if cat_matches >= 3:
        return 7.0, f"Multiple matches: {', '.join(cat_matched[:3])}", False

    if cat_matches >= 2:
        return 6.0, f"Moderate relevance: {', '.join(cat_matched[:3])}", False

    if cat_matches == 1:
        if has_breakthrough or has_news_action:
            return 5.5, f"Weak match + action: {cat_matched[0]}", False
        return 4.5, f"Weak match: {cat_matched[0]}", False

    return 3.5, "No topic relevance", True


def score_generic_adaptive(title: str, text: str,
                           keyword_index: KeywordIndex, category_id: str) -> Tuple[float, str, bool]:
    """Fallback scoring for 'auto' scorer_type — pure keyword relevance."""
    text_lower = text.lower()
    title_lower = title.lower()

    cat_matches, cat_matched = keyword_index.match_category(text, category_id)
    has_location = keyword_index.match_location(text)
    has_action = any(s in text_lower for s in NEWS_ACTION_SIGNALS + HIRING_SIGNALS + OFFER_DEAL_SIGNALS)

    # Title match is a strong signal
    title_matches = sum(1 for kw in keyword_index.category_keywords.get(category_id, set())
                       if kw in title_lower)

    if title_matches >= 2 and has_action:
        return 8.5, f"Strong match + actionable: {', '.join(cat_matched[:3])}", True

    if cat_matches >= 3 and has_action:
        return 8.0, f"High relevance + action: {', '.join(cat_matched[:3])}", True

    if cat_matches >= 3 and has_location:
        return 7.5, f"Relevant + local: {', '.join(cat_matched[:3])}", True

    if title_matches >= 2:
        return 7.0, f"Title relevance: {', '.join(cat_matched[:3])}", True

    if cat_matches >= 3:
        return 6.5, f"Good keyword coverage: {', '.join(cat_matched[:3])}", False

    if cat_matches >= 2:
        if has_action or has_location:
            return 6.0, f"Moderate + context: {', '.join(cat_matched[:3])}", False
        return 5.5, f"Moderate relevance: {', '.join(cat_matched[:3])}", False

    if cat_matches == 1:
        return 4.5, f"Weak match: {cat_matched[0]}", False

    return 3.5, "No relevance to category", True


# ═══════════════════════════════════════════════════════════════════
# Main Adaptive Scorer Class
# ═══════════════════════════════════════════════════════════════════

class AdaptiveScorer(ScorerBase):
    """
    Profile-adaptive scorer that builds relevance rules dynamically
    from the user's categories, keywords, role, and context.

    Inherits Ollama client, calibration, and score classification from ScorerBase.
    """

    def __init__(self, config: Dict[str, Any], db=None):
        super().__init__(config, db=db)

        # Build keyword index from user's profile and categories
        self._full_config = config
        self._profile = config.get("profile", {})
        self._keyword_index = KeywordIndex(config)
        self._is_student = _detect_student(self._profile)

        # Build category → scorer_type map
        self._scorer_type_map: Dict[str, str] = {}
        for cat in config.get("dynamic_categories", []):
            cat_id = cat.get("id", "").lower()
            self._scorer_type_map[cat_id] = cat.get("scorer_type", "auto")

        role = self._profile.get('role', 'unknown')
        location = self._profile.get('location', 'unknown')

        # Location-aware language filtering
        self._allowed_scripts, self._ddg_region, self._lang_label = location_to_lang(location)

        logger.info(f"AdaptiveScorer initialized for {role} in {location} "
                    f"(student={self._is_student}, categories={len(self._scorer_type_map)}, "
                    f"lang={self._lang_label}, ddg_region={self._ddg_region})")

    def _tracked_fields_block(self) -> str:
        """Build the tracked-fields block for system prompts.
        Must match export_training.py's build_system_prompt exactly."""
        companies = ', '.join(self._profile.get('tracked_companies', []))
        institutions = ', '.join(self._profile.get('tracked_institutions', []))
        interests = ', '.join(self._profile.get('interests', []))
        industries = ', '.join(self._profile.get('tracked_industries', []))
        return (
            f"Tracked companies: {companies if companies else 'None specified'}\n"
            f"Tracked institutions: {institutions if institutions else 'None specified'}\n"
            f"Tracked interests: {interests if interests else 'None specified'}\n"
            f"Tracked industries: {industries if industries else 'None specified'}"
        )

    # ─── Routing ──────────────────────────────────────────────────

    def _route_item(self, item: Dict[str, Any]) -> str:
        """Determine which scoring path to use."""
        category = item.get('category', '').lower()

        # Check dynamic category scorer_type
        if category in self._scorer_type_map:
            st = self._scorer_type_map[category]
            if st in ('career', 'banks', 'tech', 'regional'):
                return st
            return 'auto'

        # Fallback based on root
        root = item.get('root', 'global').lower()
        if root in ('ai', 'global'):
            return 'tech'
        return 'auto'

    # ─── Rule-based scoring ───────────────────────────────────────

    def _rule_score(self, item: Dict[str, Any]) -> Tuple[float, str, bool]:
        """Adaptive rule-based scoring using keyword index."""
        title = item.get('title', '')
        content = item.get('content', '') or item.get('summary', '')
        source = item.get('source', '')
        url = item.get('url', '')
        root = item.get('root', 'global').lower()
        text = f"{title} {content}".lower()
        category = item.get('category', '').lower()
        title_lower = title.lower()

        # Universal noise check (domain-agnostic, location-aware language filter)
        noise = _universal_noise_check(title, text, source, url=url, root=root,
                                       allowed_scripts=self._allowed_scripts)
        if noise:
            return noise[0], noise[1], True

        # ── Universal student guard: senior-level JOB POSTINGS across ALL categories ──
        if self._is_student:
            has_senior_title = any(s in title_lower for s in SENIOR_LEVEL_SIGNALS)
            has_hiring_signal = any(s in text for s in HIRING_SIGNALS)
            if has_senior_title and has_hiring_signal:
                return 3.5, f"Senior-level job posting (user is student): {title[:50]}", True

        # Route to appropriate scoring function
        route = self._route_item(item)

        if route == 'career':
            return score_career_adaptive(
                title.lower(), text, url,
                self._keyword_index, category, self._is_student)
        elif route == 'banks':
            return score_finance_adaptive(
                title.lower(), text, url,
                self._keyword_index, category)
        elif route in ('tech', 'regional'):
            return score_tech_adaptive(
                title.lower(), text,
                self._keyword_index, category)
        else:  # auto
            return score_generic_adaptive(
                title.lower(), text,
                self._keyword_index, category)

    # ─── LLM scoring ────────────────────────────────────────────

    def _build_llm_prompt(self, item: Dict[str, Any], route: str) -> str:
        """Build an LLM scoring prompt from the user's profile."""
        role = self._profile.get('role', 'professional')
        location = self._profile.get('location', 'unspecified')
        context = self._profile.get('context', '')
        category = item.get('category', 'general')
        cat_meta = self._keyword_index.category_meta.get(category, {})
        cat_label = cat_meta.get('label', category)
        cat_items = ', '.join(cat_meta.get('items', [])[:8])

        title = item.get('title', '')
        content = (item.get('content', '') or item.get('summary', ''))[:500]

        level_note = ""
        if self._is_student:
            level_note = "\nIMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
        else:
            level_note = "\nIMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

        tracked = self._tracked_fields_block()

        return f"""Score this article's relevance for a {role} in {location}.

User context: {context if context else 'Not specified'}
{tracked}
Category: {cat_label}
Tracked keywords: {cat_items}{level_note}
{self.memory.format_feedback_for_prompt(max_each=3)}
Article title: {title}
Article content: {content}

LANGUAGE: Articles must be in {self._lang_label}. If the article is primarily in another language, score 0.0-2.0 regardless of topic relevance.

Score 0.0-10.0 where:
9-10: Directly actionable (e.g., hiring match at right level, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong experience level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

    def _build_llm_prompt_v2(self, item: Dict[str, Any], route: str) -> Tuple[str, str]:
        """Build system + user prompt pair — matches training data format.
        
        Training data uses: system (profile + instructions) + user (article).
        This format aligns with _build_batch_prompt and with export_training.py,
        ensuring the model sees the same prompt structure at training and inference.
        
        Returns (system_prompt, user_prompt).
        """
        role = self._profile.get('role', 'professional')
        location = self._profile.get('location', 'unspecified')
        context = self._profile.get('context', '')
        category = item.get('category', 'general')
        cat_meta = self._keyword_index.category_meta.get(category, {})
        cat_label = cat_meta.get('label', category)
        cat_items = ', '.join(cat_meta.get('items', [])[:8])

        title = item.get('title', '')
        content = (item.get('content', '') or item.get('summary', ''))[:500]

        level_note = ""
        if self._is_student:
            level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
        else:
            level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

        feedback_text = self.memory.format_feedback_for_prompt(max_each=3)
        tracked = self._tracked_fields_block()

        # Core template below MUST match export_training.build_system_prompt character-for-character.
        # feedback_text and LANGUAGE line are runtime-only additions not present in training data.
        system = f"""You are a relevance scorer for a {role} in {location}.
User context: {context if context else 'Not specified'}
{tracked}
{level_note}
{feedback_text}
LANGUAGE: Articles must be in {self._lang_label}. Other languages score 0.0-2.0.

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

        user = f"""Score this article:
Category: {cat_label}
Keywords: {cat_items}
Title: {title}
Content: {content}"""

        return system, user

    def _prompt_rescore(self, item: Dict[str, Any], first_score: float, route: str, cancel_check=None, timeout_seconds=None) -> Tuple[float, str]:
        """Prompt engineering re-score for uncertain items (score 4.5-6.5).
        
        When the LoRA model outputs a score in the ambiguous zone, this method
        provides a second opinion using a richer prompt with:
        - Explicit structured rubric forcing step-by-step reasoning
        - Category-specific context from the keyword index
        
        This is the 'jury deliberation' layer — slower but more accurate.
        Returns (score, reason).
        """
        if not self.is_available():
            return first_score, f"LLM: rescore unavailable, keeping {first_score:.1f}"

        role = self._profile.get('role', 'professional')
        location = self._profile.get('location', 'unspecified')
        context = self._profile.get('context', '')
        category = item.get('category', 'general')
        cat_meta = self._keyword_index.category_meta.get(category, {})
        cat_label = cat_meta.get('label', category)
        cat_items = ', '.join(cat_meta.get('items', [])[:8])

        title = item.get('title', '')
        content = (item.get('content', '') or item.get('summary', ''))[:500]

        level_note = ""
        if self._is_student:
            level_note = "The user is a STUDENT — entry-level opportunities score highest, senior roles requiring years of experience score 0-3."
        else:
            level_note = "The user is an EXPERIENCED PROFESSIONAL — senior specialist roles score highest, entry-level/intern roles score low."

        tracked = self._tracked_fields_block()

        system = f"""You are an expert relevance scorer. Carefully evaluate this article for:
User: {role} in {location}
Context: {context if context else 'Not specified'}
{tracked}
{level_note}

EVALUATION RUBRIC — work through each dimension:
1. PROFILE MATCH: Does this article's topic relate to the user's profession or stated interests?
2. LOCATION MATCH: Is this geographically relevant? (local > regional > global)
3. ACTIONABILITY: Can the user DO something with this? (apply for job, invest, learn skill, save money)
4. LEVEL MATCH: Does the experience/seniority level fit the user?
5. NOISE CHECK: Is this real content or a generic page, clickbait, login wall, or aggregator?

Score 0.0-10.0. Be decisive — if you're unsure, lean toward the user's benefit.
Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

        user = f"""Carefully evaluate this article:
Category: {cat_label}
Keywords: {cat_items}
Title: {title}
Content: {content}

Previous automated score was {first_score:.1f} (uncertain). Please re-evaluate carefully."""

        clean, think_block = self._call_ollama(user, system_prompt=system, cancel_check=cancel_check, timeout_seconds=timeout_seconds)
        if clean == "__TIMEOUT__":
            return first_score, "__DEFERRED__"
        if not clean:
            return first_score, f"LLM: rescore failed, keeping {first_score:.1f}"

        try:
            score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+)', clean, re.IGNORECASE)
            if score_match:
                rescore = float(score_match.group(1))
                rescore = max(0.0, min(10.0, rescore))
                if 4.8 <= rescore <= 5.2:
                    rescore = 5.3 if rescore >= 5.0 else 4.8
                reason = reason_match.group(1).strip() if reason_match else "rescored"
                return rescore, f"LLM-rescore: {reason}"
        except (ValueError, AttributeError):
            pass

        return first_score, f"LLM: rescore parse failed, keeping {first_score:.1f}"

    def _llm_score(self, item: Dict[str, Any], rule_score: float, route: str, cancel_check=None, timeout_seconds=None) -> Tuple[float, str]:
        """Score with LLM for ambiguous items.

        Uses system + user prompt format matching the training data layout.
        Think mode generates structured reasoning before the SCORE/REASON output.
        """
        if not self.is_available():
            return rule_score, "LLM unavailable, using rule score"

        system, prompt = self._build_llm_prompt_v2(item, route)
        clean, think_block = self._call_ollama(prompt, system_prompt=system, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

        if clean == "__TIMEOUT__":
            return rule_score, "__DEFERRED__"
        if not clean:
            return rule_score, "LLM no response, using rule score"

        try:
            score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+)', clean, re.IGNORECASE)

            if score_match:
                llm_score = float(score_match.group(1))
                llm_score = max(0.0, min(10.0, llm_score))

                # Forbidden 5.0
                if 4.8 <= llm_score <= 5.2:
                    llm_score = 5.3 if llm_score >= 5.0 else 4.8

                reason = reason_match.group(1).strip() if reason_match else "LLM scored"
                return llm_score, f"LLM: {reason}"
        except (ValueError, AttributeError):
            pass

        return rule_score, "LLM parse failed, using rule score"

    # ─── Batch LLM scoring (3-4 items per call) ──────────────────

    BATCH_SIZE = 4

    def _build_batch_prompt(self, batch: List[Tuple[Dict, float, str]]) -> Tuple[str, str]:
        """Build a single prompt that scores multiple items at once.
        Returns (system_prompt, user_prompt)."""
        role = self._profile.get('role', 'professional')
        location = self._profile.get('location', 'unspecified')
        context = self._profile.get('context', '')

        level_note = ""
        if self._is_student:
            level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
        else:
            level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

        # Build feedback context for batch scoring
        feedback_text = self.memory.format_feedback_for_prompt(max_each=3)
        tracked = self._tracked_fields_block()

        system = f"""You are a relevance scorer for a {role} in {location}.
User context: {context if context else 'Not specified'}
{tracked}
{level_note}
{feedback_text}
Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply with EXACTLY one line per article, numbered:
[1] SCORE: X.X | REASON: brief explanation
[2] SCORE: X.X | REASON: brief explanation
...and so on. Nothing else."""

        articles = []
        for idx, (item, rule_score, route) in enumerate(batch, 1):
            category = item.get('category', 'general')
            cat_meta = self._keyword_index.category_meta.get(category, {})
            cat_label = cat_meta.get('label', category)
            cat_items = ', '.join(cat_meta.get('items', [])[:6])
            title = item.get('title', '')
            content = (item.get('content', '') or item.get('summary', ''))[:300]
            articles.append(f"[{idx}] Category: {cat_label} | Keywords: {cat_items}\n    Title: {title}\n    Content: {content}")

        user_prompt = f"Score these {len(batch)} articles:\n\n" + "\n\n".join(articles)
        return system, user_prompt

    def _parse_batch_response(self, response: str, batch_size: int) -> List[Optional[Tuple[float, str]]]:
        """Parse numbered SCORE/REASON lines from batch response.
        Returns list of (score, reason) or None for unparseable entries."""
        results = [None] * batch_size
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Match patterns like "[1] SCORE: 7.5 | REASON: ..." or "1. SCORE: 7.5 | REASON: ..."
            m = re.match(r'[\[\(]?(\d+)[\]\).]?\s*SCORE:\s*(\d+\.?\d*)\s*\|?\s*REASON:\s*(.*)', line, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < batch_size:
                    score = max(0.0, min(10.0, float(m.group(2))))
                    if 4.8 <= score <= 5.2:
                        score = 5.3 if score >= 5.0 else 4.8
                    results[idx] = (score, f"LLM: {m.group(3).strip()}")
        return results

    def _llm_score_batch(self, batch: List[Tuple[Dict, float, str]], cancel_check=None, timeout_seconds=None) -> List[Tuple[float, str]]:
        """Score a batch of items with a single Ollama call.
        Falls back to rule scores for any items that fail to parse."""
        if not self.is_available() or not batch:
            return [(rs, "LLM unavailable, using rule score") for _, rs, _ in batch]

        system, prompt = self._build_batch_prompt(batch)
        clean, think_block = self._call_ollama(prompt, system_prompt=system, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

        if clean == "__TIMEOUT__":
            return [(rs, "__DEFERRED__") for _, rs, _ in batch]
        if not clean:
            return [(rs, "LLM no response, using rule score") for _, rs, _ in batch]

        parsed = self._parse_batch_response(clean, len(batch))
        results = []
        for i, (item, rule_score, route) in enumerate(batch):
            if parsed[i] is not None:
                results.append(parsed[i])
            else:
                # This item failed to parse — fall back to individual call
                logger.debug(f"Batch parse miss for item {i+1}, falling back to single score")
                results.append(self._llm_score(item, rule_score, route, cancel_check=cancel_check))
        return results

    # ─── Main entry points ────────────────────────────────────────

    def score_item(self, item: Dict[str, Any]) -> Tuple[float, str]:
        """Score a single item."""
        route = self._route_item(item)
        rule_score, rule_reason, is_confident = self._rule_score(item)

        if is_confident:
            self._stats['rule'] += 1
            return rule_score, rule_reason

        # Ambiguous → try LLM
        llm_score, llm_reason = self._llm_score(item, rule_score, route)
        self._stats['llm'] += 1
        llm_score = self._calibrate_score(llm_score)
        return round(llm_score, 1), llm_reason

    def score_items(self, items: List[Dict[str, Any]],
                    max_workers: int = 1,
                    progress_callback=None,
                    cancel_check=None,
                    timeout_seconds=None) -> tuple:
        """Score multiple items. Batches ambiguous items for fewer LLM calls.

        Args:
            cancel_check: Optional callable returning True if scan should stop.
                          Checked between articles; current article finishes first.
            timeout_seconds: Optional max seconds per LLM call. Items that exceed
                           this are returned in the deferred list.

        Returns:
            (scored_items, deferred_items) — deferred items timed out and need retry.
        """
        total = len(items)
        logger.info(f"Scoring {total} items (adaptive profile scorer)...")

        # Log feedback loop status
        fb_stats = self.memory.get_feedback_summary()
        if any(fb_stats.values()):
            logger.info(f"Feedback loop active: {fb_stats['positive']} positive, "
                       f"{fb_stats['negative']} negative, {fb_stats['corrections']} corrections")

        self._stats = {'rule': 0, 'llm': 0, 'truncated': 0}
        scored_items = []
        deferred = []  # Items that timed out during LLM scoring — returned for Pass 2
        new_examples = 0
        seen_titles = {}

        # Phase 1: Rule-score everything, collect ambiguous items
        ambiguous = []  # [(index_in_scored_items, item, rule_score, route)]
        uncertain_items = []  # [(index, item, first_score, route)] — for prompt engineering cascade

        _cancelled = False
        for i, item in enumerate(items):
            if cancel_check and cancel_check():
                logger.info(f"Scan cancelled during rule-scoring at item {i}/{total}")
                _cancelled = True
                break
            try:
                current = i + 1
                if progress_callback:
                    progress_callback(current, total)

                # Pre-filtered items already have a score
                if 'pre_filter_score' in item:
                    item['score'] = item['pre_filter_score']
                    item['score_reason'] = f"Pre-filtered: {item.get('pre_filter_reason', '')}"
                    scored_items.append(item)
                    self._stats['rule'] += 1
                    continue

                # Title deduplication
                title_key = item.get('title', '').strip().lower()
                if title_key and title_key in seen_titles:
                    item['score'] = 2.0
                    item['score_reason'] = f"Duplicate title (same as item #{seen_titles[title_key]})"
                    scored_items.append(item)
                    self._stats['rule'] += 1
                    continue
                if title_key:
                    seen_titles[title_key] = current

                # Rule scoring
                route = self._route_item(item)
                rule_score, rule_reason, is_confident = self._rule_score(item)

                if is_confident:
                    if 4.8 <= rule_score <= 5.2:
                        rule_score = 5.3 if rule_score >= 5.0 else 4.8
                    item['score'] = rule_score
                    item['score_reason'] = rule_reason
                    scored_items.append(item)
                    self._stats['rule'] += 1
                else:
                    # Placeholder — will be filled by batch LLM scoring
                    idx = len(scored_items)
                    scored_items.append(item)
                    ambiguous.append((idx, item, rule_score, route))

            except Exception as e:
                logger.error(f"Scoring failed: {e}")
                item['score'] = 3.0
                item['score_reason'] = "Error"
                scored_items.append(item)

        # On cancel, assign rule_scores to any ambiguous items that won't get LLM-scored
        if _cancelled and ambiguous:
            for idx, item, rule_score, route in ambiguous:
                if 'score' not in item:
                    item['score'] = rule_score
                    item['score_reason'] = f"Rule score (scan cancelled)"
                    scored_items[idx] = item
                    self._stats['rule'] += 1

        # Phase 2: Batch LLM scoring for ambiguous items
        if ambiguous and not _cancelled:
            logger.info(f"Batch-scoring {len(ambiguous)} ambiguous items in groups of {self.BATCH_SIZE}..."
                        + (f" (timeout={timeout_seconds:.0f}s)" if timeout_seconds else ""))
            for batch_start in range(0, len(ambiguous), self.BATCH_SIZE):
                if cancel_check and cancel_check():
                    logger.info(f"Scan cancelled during LLM batch-scoring at batch {batch_start}/{len(ambiguous)}")
                    _cancelled = True
                    break
                batch_slice = ambiguous[batch_start:batch_start + self.BATCH_SIZE]
                batch_tuples = [(item, rs, route) for _, item, rs, route in batch_slice]

                results = self._llm_score_batch(batch_tuples, cancel_check=cancel_check, timeout_seconds=timeout_seconds)
                self._stats['llm'] += len(batch_slice)

                for j, (idx, item, rule_score, route) in enumerate(batch_slice):
                    score, reason = results[j]

                    # Deferred item — timed out, will be retried in Pass 2
                    if reason == "__DEFERRED__":
                        import copy as _copy
                        deferred.append(_copy.deepcopy(item))
                        item['score'] = rule_score
                        item['score_reason'] = f"Deferred (timeout, rule={rule_score:.1f})"
                        scored_items[idx] = item
                        continue

                    if 4.8 <= score <= 5.2:
                        score = 5.3 if score >= 5.0 else 4.8
                    item['score'] = score
                    item['score_reason'] = reason
                    scored_items[idx] = item

                    # Confidence cascade: items in the uncertain zone get re-scored
                    # with prompt engineering (structured rubric + richer context)
                    if 4.5 <= score <= 6.5:
                        uncertain_items.append((idx, item, score, route))

                    if self.memory.add_example(item, category=route):
                        new_examples += 1

                if progress_callback:
                    done = min(batch_start + self.BATCH_SIZE, len(ambiguous))
                    progress_callback(total - len(ambiguous) + done, total)

        # If cancelled mid-Phase-2, fill remaining unscored ambiguous items with rule scores
        if _cancelled:
            for idx, item, rule_score, route in ambiguous:
                if 'score' not in item:
                    item['score'] = rule_score
                    item['score_reason'] = f"Rule score (scan cancelled)"
                    scored_items[idx] = item

        # Phase 3: Prompt engineering re-score for uncertain items (4.5-6.5)
        # These items were ambiguous even after LoRA scoring — a richer prompt
        # with structured rubric often resolves them decisively.
        # Fix 2B: Re-check live cancel flag at Phase 3 boundary
        if uncertain_items and not _cancelled and cancel_check and cancel_check():
            logger.info("Scan cancelled at Phase 3 boundary (prompt re-scoring skipped)")
            _cancelled = True
        if uncertain_items and not _cancelled:
            # Cap re-scores to avoid excessive Ollama calls
            rescore_batch = uncertain_items[:12]
            logger.info(f"Re-scoring {len(rescore_batch)} uncertain items (score 4.5-6.5) with prompt engineering...")
            rescored = 0
            for idx, item, first_score, route in rescore_batch:
                new_score, new_reason = self._prompt_rescore(item, first_score, route, cancel_check=cancel_check, timeout_seconds=timeout_seconds)
                if abs(new_score - first_score) >= 1.0:
                    # Meaningful change — accept the re-score
                    if 4.8 <= new_score <= 5.2:
                        new_score = 5.3 if new_score >= 5.0 else 4.8
                    item['score'] = new_score
                    item['score_reason'] = new_reason
                    scored_items[idx] = item
                    rescored += 1
                    self._stats['rescore'] = self._stats.get('rescore', 0) + 1
            if rescored:
                logger.info(f"  → {rescored}/{len(rescore_batch)} items rescored with meaningful change")

        # Also add memory examples for rule-scored items
        for item in scored_items:
            if item.get('score_reason', '').startswith('Pre-filtered') or item.get('score_reason', '').startswith('Duplicate'):
                continue
            if 'score' in item and not item.get('score_reason', '').startswith('LLM'):
                route = self._route_item(item)
                if self.memory.add_example(item, category=route):
                    new_examples += 1

        # V1 calibration: correct systematic score inflation on LLM-scored items
        if self._calibration_table:
            calibrated_count = 0
            for item in scored_items:
                reason = item.get('score_reason', '')
                # Only calibrate LLM-scored items (skip rule-based, pre-filtered, duplicates)
                if reason.startswith('Pre-filtered') or reason.startswith('Duplicate') or reason == 'Error':
                    continue
                raw = item.get('score', 0)
                cal = self._calibrate_score(raw)
                if abs(cal - raw) > 0.01:
                    item['score'] = round(cal, 1)
                    calibrated_count += 1
            if calibrated_count:
                logger.info(f"V1 calibration applied to {calibrated_count} items")

        scored_items.sort(key=lambda x: x.get('score', 0), reverse=True)

        critical = sum(1 for i in scored_items if i.get('score', 0) >= 9.0)
        high = sum(1 for i in scored_items if 7.0 <= i.get('score', 0) < 9.0)
        medium = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
        noise = sum(1 for i in scored_items if i.get('score', 0) <= 5.0)

        deferred_note = f", Deferred: {len(deferred)}" if deferred else ""
        logger.info(
            f"Scoring complete: {critical} critical, {high} high, "
            f"{medium} medium, {noise} noise | "
            f"Rules: {self._stats['rule']}, LLM: {self._stats['llm']}{deferred_note} | "
            f"New examples: {new_examples}"
        )
        return scored_items, deferred


