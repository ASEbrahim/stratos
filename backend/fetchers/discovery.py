"""
STRAT_OS - Dynamic Entity Discovery
Automatically discovers and tracks rising entities/topics.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import Counter

logger = logging.getLogger(__name__)


class EntityDiscovery:
    """
    Discovers new entities by tracking keyword frequency changes.
    
    The algorithm:
    1. Extract potential entities (proper nouns, company names) from news
    2. Compare current frequency vs historical baseline
    3. Flag entities with significant frequency increase as "rising"
    4. Optionally auto-add rising entities to tracking list
    """
    
    # Words to ignore (common but not useful)
    STOPWORDS = {
        # Days/months
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
        'september', 'october', 'november', 'december',
        # Common words that get capitalized
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'new', 'first', 'last',
        'next', 'year', 'month', 'week', 'day', 'today', 'yesterday', 'tomorrow',
        'breaking', 'update', 'report', 'news', 'exclusive', 'official', 'top',
        'best', 'worst', 'latest', 'recent', 'just', 'now', 'says', 'said',
        # Generic tech/business words
        'company', 'market', 'stock', 'price', 'business', 'industry', 'sector',
        'technology', 'tech', 'digital', 'online', 'global', 'world', 'international',
        # Places (too broad)
        'united', 'states', 'america', 'europe', 'asia', 'africa',
    }
    
    # Categories for discovered entities
    CATEGORY_PATTERNS = {
        'kuwait_companies': [
            r'\b(bank|oil|petroleum|telecom|investment|holding)\b',
            r'\b(kuwait|kuwaiti)\b'
        ],
        'tech_terms': [
            r'\b(ai|ml|quantum|blockchain|crypto|semiconductor|chip)\b',
            r'\b(computing|intelligence|neural|algorithm)\b'
        ],
        'financial_products': [
            r'\b(loan|mortgage|savings|account|offer|promotion|deal)\b',
            r'\b(student|allowance|transfer|bonus)\b'
        ]
    }
    
    def __init__(self, config: Dict[str, Any], database):
        """
        Initialize entity discovery.
        
        Args:
            config: Discovery config section
            database: Database instance for persistence
        """
        self.config = config
        self.db = database
        self.enabled = config.get("enabled", True)
        self.rising_multiplier = config.get("rising_multiplier", 3.0)
        self.min_mentions = config.get("min_mentions", 3)
        self.baseline_hours = config.get("baseline_window_hours", 168)
        self.auto_track = config.get("auto_track", True)
        
        # Load existing tracked entities
        self._tracked_entities: Set[str] = set()
        self._load_tracked_entities()
    
    def _load_tracked_entities(self):
        """Load existing tracked entities from database."""
        try:
            entities = self.db.get_tracked_entities(active_only=True)
            self._tracked_entities = {e['name'].lower() for e in entities}
            logger.info(f"Loaded {len(self._tracked_entities)} tracked entities")
        except Exception as e:
            logger.error(f"Failed to load tracked entities: {e}")
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract potential entity names from text.
        
        Uses simple pattern matching for:
        - Capitalized words/phrases (proper nouns)
        - Known patterns (Bank, Corp, Inc, etc.)
        
        Args:
            text: Input text
            
        Returns:
            List of potential entity names
        """
        entities = []
        
        # Pattern 1: Capitalized word sequences (2-4 words)
        # e.g., "Kuwait Finance House", "National Bank"
        pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
        matches = re.findall(pattern1, text)
        entities.extend(matches)
        
        # Pattern 2: All-caps acronyms (2-5 chars)
        # e.g., "KOC", "KNPC", "NBK"
        pattern2 = r'\b([A-Z]{2,5})\b'
        matches = re.findall(pattern2, text)
        entities.extend(matches)
        
        # Pattern 3: Company suffixes
        # e.g., "Equate Corp", "Warba Bank"
        pattern3 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Bank|Corp|Inc|Ltd|Company|Co|Group))\b'
        matches = re.findall(pattern3, text)
        entities.extend(matches)
        
        # Filter and clean
        cleaned = []
        for entity in entities:
            entity = entity.strip()
            # Skip stopwords and too short
            if entity.lower() in self.STOPWORDS:
                continue
            if len(entity) < 3:
                continue
            # Skip if all caps and too short (likely not meaningful)
            if entity.isupper() and len(entity) < 2:
                continue
            cleaned.append(entity)
        
        return cleaned
    
    def categorize_entity(self, entity: str, context: str = "") -> str:
        """
        Determine the category of an entity based on patterns.
        
        Args:
            entity: Entity name
            context: Surrounding text for additional context
            
        Returns:
            Category string
        """
        combined = f"{entity} {context}".lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return category
        
        return "general"
    
    def process_news_items(self, items: List[Dict]) -> Dict[str, int]:
        """
        Extract and count entities from news items.
        
        Args:
            items: List of news item dicts
            
        Returns:
            Dict of entity -> count
        """
        all_entities = []
        
        for item in items:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            entities = self.extract_entities(text)
            all_entities.extend(entities)
        
        # Count occurrences
        return Counter(all_entities)
    
    def find_rising_entities(self, current_counts: Dict[str, int]) -> List[Dict]:
        """
        Identify entities with significantly increased mention frequency.
        
        Args:
            current_counts: Current entity counts from latest fetch
            
        Returns:
            List of rising entity dicts with metadata
        """
        if not self.enabled:
            return []
        
        rising = []
        
        for entity, count in current_counts.items():
            # Skip if below minimum threshold
            if count < self.min_mentions:
                continue
            
            # Skip if already tracked
            if entity.lower() in self._tracked_entities:
                continue
            
            # Get baseline frequency
            baseline = self.db.get_entity_baseline(entity, self.baseline_hours)
            
            # Calculate if "rising"
            # If no baseline, and count >= min_mentions, it's potentially new
            if baseline == 0:
                if count >= self.min_mentions * 2:
                    is_rising = True
                    increase_factor = float('inf')
                else:
                    is_rising = False
                    increase_factor = 0
            else:
                increase_factor = count / baseline
                is_rising = increase_factor >= self.rising_multiplier
            
            if is_rising:
                category = self.categorize_entity(entity)
                rising.append({
                    "name": entity,
                    "current_count": count,
                    "baseline_avg": baseline,
                    "increase_factor": increase_factor if increase_factor != float('inf') else 999,
                    "category": category
                })
        
        # Sort by increase factor
        rising.sort(key=lambda x: x["increase_factor"], reverse=True)
        
        logger.info(f"Found {len(rising)} rising entities")
        return rising
    
    def record_mentions(self, counts: Dict[str, int]):
        """
        Record entity mentions for future baseline calculation.
        
        Args:
            counts: Dict of entity -> count
        """
        for entity, count in counts.items():
            if count >= 2:  # Only record if mentioned at least twice
                self.db.record_entity_mention(entity, count)
    
    def auto_track_entity(self, entity: Dict) -> bool:
        """
        Automatically add a rising entity to tracking.
        
        Args:
            entity: Entity dict from find_rising_entities
            
        Returns:
            True if successfully added
        """
        if not self.auto_track:
            return False
        
        success = self.db.add_tracked_entity(
            name=entity["name"],
            category=entity["category"],
            is_discovered=True
        )
        
        if success:
            self._tracked_entities.add(entity["name"].lower())
            logger.info(f"Auto-tracked new entity: {entity['name']}")
        
        return success
    
    def discover(self, news_items: List[Dict]) -> List[Dict]:
        """
        Main discovery pipeline.
        
        Args:
            news_items: List of news item dicts
            
        Returns:
            List of newly discovered rising entities
        """
        if not self.enabled:
            return []
        
        # 1. Extract and count entities
        counts = self.process_news_items(news_items)
        
        # 2. Record for baseline tracking
        self.record_mentions(counts)
        
        # 3. Find rising entities
        rising = self.find_rising_entities(counts)
        
        # 4. Auto-track if enabled
        for entity in rising:
            self.auto_track_entity(entity)
        
        return rising


def run_discovery(config: Dict[str, Any], database, news_items: List[Dict]) -> List[Dict]:
    """
    Convenience function to run entity discovery.
    
    Args:
        config: Full configuration dict
        database: Database instance
        news_items: List of news items to analyze
        
    Returns:
        List of discovered rising entities
    """
    discovery_config = config.get("discovery", {})
    discovery = EntityDiscovery(discovery_config, database)
    return discovery.discover(news_items)
