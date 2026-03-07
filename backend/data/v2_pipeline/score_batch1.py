#!/usr/bin/env python3
"""
V3 Scoring Script — Batch 1
Profiles: marketing_director_london, auditor_toronto, auto_engineer_stuttgart
813 articles x 3 profiles = 2,439 scores

Scoring approach:
- Multi-tier keyword matching with contextual boosting
- Profile-specific negative signals
- Non-English content handling
- Source credibility weighting
"""

import json
import re
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(BASE_DIR, "v3_scores_batch1.json")
ARTICLES_FILE = os.path.join(BASE_DIR, "articles_v2.json")

with open(ARTICLES_FILE) as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")

PROFILES = {
    "marketing_director_london": {
        "role": "Marketing director at a consumer goods company",
        "location": "London, UK",
    },
    "auditor_toronto": {
        "role": "Senior auditor at a Big Four accounting firm",
        "location": "Toronto, Canada",
    },
    "auto_engineer_stuttgart": {
        "role": "EV powertrain engineer at an automotive manufacturer",
        "location": "Stuttgart, Germany",
    },
}


def is_non_english(text):
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    cjk = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    non_latin = arabic + cjk
    if non_latin > latin and non_latin > 10:
        return True
    return False


def kw(text, patterns):
    """Count how many patterns match in text."""
    count = 0
    matched = []
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
            m = re.search(p, text, re.IGNORECASE)
            matched.append(m.group(0)[:30])
    return count, matched


def score_marketing(article):
    title = (article.get("title") or "")
    summary = (article.get("summary") or "")
    text = f"{title} {summary}".lower()
    tl = title.lower()
    sl = summary.lower()

    # === DIRECT COMPANY MATCHES (tier 1) ===
    company_pats = [
        r'\bunilever\b', r'\bp&g\b', r'\bprocter\b', r'\bprocter\s*(&|and)\s*gamble\b', r'\bdiageo\b', r'\bwpp\b',
    ]
    company_hits, company_matched = kw(text, company_pats)

    # === CORE ROLE MATCHES (tier 1) ===
    core_pats = [
        r'\bfmcg\b', r'\bconsumer\s+goods\b', r'\bcpg\b', r'\bfast.moving\s+consumer',
        r'\bbrand\s+strateg', r'\bdigital\s+advertis', r'\bprogrammatic\s+advertis',
        r'\bad\s+tech\b', r'\badtech\b', r'\bad\s+spend\b',
        r'\bconsumer\s+insight', r'\bconsumer\s+behav',
        r'\bsustainab.*market', r'\bgreen\s+market', r'\bgreen.*brand',
    ]
    core_hits, core_matched = kw(text, core_pats)

    # === INDUSTRY MATCHES (tier 2) ===
    industry_pats = [
        r'\bmarketing\b', r'\badvertis\w*\b', r'\bbrand\s', r'\bbranding\b',
        r'\bretail\b', r'\be-?commerce\b', r'\bonline\s+shopping\b', r'\bonline\s+retail',
        r'\bd2c\b', r'\bdirect.to.consumer\b',
        r'\bconsumer\b', r'\bshopper\b', r'\bshopping\b',
        r'\bsocial\s+media\s+market', r'\binfluencer\b',
        r'\bcontent\s+market', r'\bseo\b', r'\bsem\b',
        r'\bgoogle\s+ads\b', r'\bmeta\s+ads\b', r'\btiktok\b.*market',
        r'\buk\s+retail\b', r'\bbritish\s+retail\b', r'\blondon.*market',
        # Competitor FMCG companies
        r'\bnestl[eé]\b', r'\breckitt\b', r'\bhenkel\b', r'\bloreal\b', r'\bl.or[eé]al\b',
        r'\bcolgate\b', r'\bjohnson.*johnson\b', r'\bkraft\b', r'\bmondelez\b',
        r'\bcoca.cola\b', r'\bpepsi\b',
        # Ad agencies
        r'\bomnicom\b', r'\bpublicis\b', r'\bdentsu\b', r'\bipg\b.*agency',
        r'\bhavas\b', r'\bogilvy\b', r'\bmccann\b',
        # UK/EU market regulation
        r'\bgdpr\b', r'\bdata\s+privacy\b.*market', r'\bcma\b.*market',
        r'\basa\b.*standard',
        # Packaging/sustainability
        r'\bpackaging\b.*sustainab', r'\bplastic\s+packag',
    ]
    ind_hits, ind_matched = kw(text, industry_pats)

    # === ADJACENT (tier 3) ===
    adjacent_pats = [
        r'\buk\b.*econom', r'\blondon\b.*(?:business|market|retail|brand)',
        r'\beuropean\b.*(?:consumer|market|retail)',
        r'\bsupermarket\b', r'\bgrocery\b',
        r'\bamazon\b.*(?:retail|shop|deliver|ecommerce)',
        r'\bsupply\s+chain\b.*(?:consumer|retail|fmcg)',
        r'\bai\b.*(?:market|advertis|brand|consumer)',
        r'\bstartup\b.*consumer',
    ]
    adj_hits, adj_matched = kw(text, adjacent_pats)

    # === NEGATIVE ===
    neg_pats = [
        r'\bnfl\b', r'\bnba\b', r'\bnhl\b', r'\bmlb\b',
        r'\bfootball\s+player', r'\btouchdown\b', r'\bquarterback\b',
        r'\bcricket\b.*match', r'\brugby\b.*match',
        r'\bdrilling\b', r'\bwelding\b', r'\bpipeline\s+construct',
        r'\bseismic\b', r'\breservoir\b', r'\bgeophys',
        r'\bchemotherapy\b', r'\bsurgery\b', r'\boncolog', r'\bleukemia\b',
        r'\bgene\s+therap', r'\bcrispr\b',
        r'\bk-?pop\b', r'\bbonsai\b', r'\bformula\s*1\b',
        r'\bmilitary\b', r'\bweapon\b', r'\bnuclear\s+weapon',
        r'\bquantum\s+comput', r'\bsemiconductor\b',
        r'\bcrypto\b', r'\bbitcoin\b', r'\bethereum\b',
        r'\boil\s+(and\s+)?gas\b', r'\bupstream\b.*oil', r'\bpetroleum\b',
        r'\bairport\b', r'\baviation\b', r'\bfaan\b',
        r'\bmining\b', r'\bcopper\b.*mine', r'\bgold\b.*mine',
    ]
    neg_hits, _ = kw(text, neg_pats)

    non_eng = is_non_english(f"{title} {summary}")

    # Score calculation
    if company_hits >= 1 and (core_hits >= 1 or ind_hits >= 2):
        score = 9.0 + min(company_hits + core_hits - 2, 2) * 0.3
    elif company_hits >= 1:
        score = 7.5 + min(ind_hits, 3) * 0.4
    elif core_hits >= 2:
        score = 8.0 + min(core_hits - 2 + ind_hits, 3) * 0.3
    elif core_hits == 1 and ind_hits >= 2:
        score = 7.5
    elif core_hits == 1 and ind_hits >= 1:
        score = 7.0
    elif core_hits == 1:
        score = 6.5
    elif ind_hits >= 4:
        score = 7.5
    elif ind_hits >= 3:
        score = 7.0
    elif ind_hits == 2:
        score = 6.0
    elif ind_hits == 1 and adj_hits >= 2:
        score = 5.5
    elif ind_hits == 1 and adj_hits >= 1:
        score = 5.0
    elif ind_hits == 1:
        score = 4.0
    elif adj_hits >= 3:
        score = 3.5
    elif adj_hits >= 2:
        score = 2.5
    elif adj_hits == 1:
        score = 1.5
    else:
        score = 0.5

    if neg_hits >= 2:
        score = min(score, max(score * 0.2, 0.5))
    elif neg_hits == 1 and score < 7.0:
        score = min(score, 1.5)
    elif neg_hits == 1:
        score = max(score - 2.0, 2.0)

    if non_eng:
        if score < 7.0:
            score = min(score, 0.5)
        else:
            score = max(score - 3.0, 1.0)

    score = round(min(max(score, 0.0), 10.0), 1)

    all_matched = company_matched + core_matched + ind_matched + adj_matched
    return score, all_matched, non_eng, neg_hits


def score_auditor(article):
    title = (article.get("title") or "")
    summary = (article.get("summary") or "")
    text = f"{title} {summary}".lower()

    # === TRACKED COMPANIES (tier 1) ===
    company_pats = [
        r'\bdeloitte\b', r'\bkpmg\b', r'\bpwc\b', r'\bpricewaterhouse\b',
        r'\bernst\s*(&|and)\s*young\b', r'\be\s*&\s*y\b',
    ]
    company_hits, company_matched = kw(text, company_pats)

    # === CORE ROLE (tier 1) ===
    core_pats = [
        r'\bifrs\b', r'\baudit\b', r'\baccounting\s+standard',
        r'\bfinancial\s+report\w*\b', r'\bfinancial\s+statement',
        r'\besg\s+report', r'\bcsrd\b', r'\bsustainability\s+report',
        r'\bcanadian\s+tax\b', r'\bcra\b.*tax', r'\bcpa\s+canada\b',
        r'\bforensic\s+account', r'\baudit\s+tech',
        r'\btsx\b', r'\btoronto\s+stock\b',
        r'\bosc\b.*secur', r'\biasb\b',
        r'\bbig\s+four\b', r'\bbig\s*4\b',
        r'\bgaap\b',
    ]
    core_hits, core_matched = kw(text, core_pats)

    # === INDUSTRY (tier 2) ===
    industry_pats = [
        r'\baccounting\b', r'\btax\s+(reform|polic|code|law|credit|cut|rate|filing|return|audit)',
        r'\btaxation\b',
        r'\bcompliance\b.*(?:financ|account|audit|report|regulat|govern|tax|sec|esg)',
        r'\bregulat\w+\b.*(?:financ|account|audit|report|bank|secur|disclos)',
        r'\besg\b', r'\bclimate\s+disclos', r'\bcarbon\s+report',
        r'\bgreenwash', r'\bscope\s+[123]\b', r'\bnet\s+zero\b.*report',
        r'\bcorporate\s+governance\b', r'\btransparency\b',
        r'\bfraud\b', r'\bwhistleblow', r'\binternal\s+control',
        r'\brisk\s+management\b', r'\bdue\s+diligen',
        r'\bsox\b', r'\bsarbanes\b',
        r'\bcanad\w+\b.*business', r'\bcanad\w+\b.*econom',
        r'\btoronto\b.*(?:financ|bank|business|stock|econom|tax|audit)',
        r'\bontario\b.*(?:business|tax|regulat|econom)',
        r'\bbank\s+of\s+canada\b',
        r'\bsec\b.*disclos', r'\bsec\b.*report', r'\bsec\b.*filing',
        r'\bipo\b', r'\bm&a\b', r'\bmerger\b', r'\bacquisition\b',
        r'\bcorporate\s+tax\b', r'\btax\s+reform\b', r'\btax\s+polic',
        r'\binvestor\s+relat', r'\bshareholder\b',
        r'\bcrypto\b.*regulat', r'\bblockchain\b.*audit',
        r'\bai\b.*audit', r'\bai\b.*account', r'\bai\b.*compliance',
    ]
    ind_hits, ind_matched = kw(text, industry_pats)

    # === ADJACENT (tier 3) ===
    adjacent_pats = [
        r'\bfinance\b', r'\bfinancial\b',
        r'\bbank\w*\b(?!.*\bfood\b)',  # bank but not food bank
        r'\binvest\w+\b', r'\bequit\w+\b',
        r'\bcanada\b.*(?:econom|business|polic|tax|trade)',
        r'\bcanadian\b.*(?:econom|business|polic|tax|trade)',
        r'\bcorporat\w+\b.*(?:govern|report|disclos)',
        r'\binterest\s+rate', r'\binflation\b', r'\bgdp\b',
        r'\brecession\b', r'\bcentral\s+bank',
        r'\bstock\s+market', r'\btariff\b.*(?:trade|import|export)',
    ]
    adj_hits, adj_matched = kw(text, adjacent_pats)

    # === NEGATIVE ===
    neg_pats = [
        r'\bnfl\b', r'\bnba\b', r'\bnhl\b', r'\bmlb\b',
        r'\btouchdown\b', r'\bquarterback\b', r'\bhome\s+run\b',
        r'\bk-?pop\b', r'\bbonsai\b', r'\bformula\s*1\b',
        r'\brecipe\b', r'\bcooking\b', r'\bcuisine\b', r'\brestaurant\b',
        r'\bgame\s+dev', r'\bindie\s+game', r'\bgodot\b', r'\bsteam\b.*game',
        r'\bchemotherapy\b', r'\boncolog\w+\b', r'\bgene\s+therap',
        r'\bseismic\b', r'\breservoir\b', r'\bdrilling\b.*well',
        r'\bhvac\b', r'\brefrigerant\b', r'\bwelding\b',
        r'\bbonsai\b', r'\bhorticulture\b',
        r'\bpark\b.*light\s+install', r'\blight\s+install.*park\b',
        r'\bcable\s+clamp', r'\bcable\s+clip',
    ]
    neg_hits, _ = kw(text, neg_pats)

    non_eng = is_non_english(f"{title} {summary}")

    # Score
    if company_hits >= 1 and core_hits >= 1:
        score = 9.0 + min(company_hits + core_hits - 2, 2) * 0.3
    elif company_hits >= 1 and ind_hits >= 1:
        score = 8.0 + min(ind_hits, 3) * 0.3
    elif company_hits >= 1:
        score = 7.0
    elif core_hits >= 3:
        score = 8.5
    elif core_hits >= 2:
        score = 8.0
    elif core_hits == 1 and ind_hits >= 2:
        score = 7.5
    elif core_hits == 1 and ind_hits >= 1:
        score = 7.0
    elif core_hits == 1:
        score = 6.5
    elif ind_hits >= 5:
        score = 7.5
    elif ind_hits >= 4:
        score = 7.0
    elif ind_hits >= 3:
        score = 6.5
    elif ind_hits == 2:
        score = 5.5
    elif ind_hits == 1 and adj_hits >= 2:
        score = 5.0
    elif ind_hits == 1 and adj_hits >= 1:
        score = 4.5
    elif ind_hits == 1:
        score = 3.5
    elif adj_hits >= 4:
        score = 3.5
    elif adj_hits >= 3:
        score = 3.0
    elif adj_hits >= 2:
        score = 2.0
    elif adj_hits == 1:
        score = 1.0
    else:
        score = 0.5

    if neg_hits >= 2:
        score = min(score, max(score * 0.2, 0.5))
    elif neg_hits == 1 and score < 7.0:
        score = min(score, 1.5)
    elif neg_hits == 1:
        score = max(score - 2.0, 2.0)

    if non_eng:
        if score < 7.0:
            score = min(score, 0.5)
        else:
            score = max(score - 3.0, 1.0)

    score = round(min(max(score, 0.0), 10.0), 1)
    all_matched = company_matched + core_matched + ind_matched + adj_matched
    return score, all_matched, non_eng, neg_hits


def score_auto(article):
    title = (article.get("title") or "")
    summary = (article.get("summary") or "")
    text = f"{title} {summary}".lower()

    # === TRACKED COMPANIES (tier 1) — but only in automotive context ===
    # Mercedes in F1 context is NOT relevant to EV powertrain engineer
    is_f1 = bool(re.search(r'\bformula\s*1\b|\bf1\b|\bpreseason\s+test', text))
    is_motorsport = bool(re.search(r'\bformula\s*1\b|\bf1\b|\bracing\b|\bgrand\s+prix\b|\bmotorsport\b', text))
    is_formula_e = bool(re.search(r'\bformula\s*e\b', text))

    company_pats = [
        r'\bmercedes\b', r'\bmercedes-benz\b', r'\bporsche\b', r'\bbmw\b',
        r'\bcatl\b', r'\bbyd\b',
    ]
    company_hits, company_matched = kw(text, company_pats)

    # If the article is about F1/motorsport (not Formula E), reduce company relevance
    if is_motorsport and not is_formula_e:
        company_hits = 0
        company_matched = []

    # Formula E is actually relevant (EV racing technology)
    # === CORE ROLE (tier 1) ===
    core_pats = [
        r'\bev\b.*powertrain', r'\belectric\s+vehicle\b', r'\belectric\s+car\b',
        r'\bbattery\s+tech', r'\bsolid.state\s+batter', r'\blithium.ion\b.*batter',
        r'\bev\b.*drivetrain', r'\belectric\s+motor\b', r'\binverter\b.*ev',
        r'\beu\s+emission', r'\beuro\s*7\b', r'\bco2\s+emission\b.*auto',
        r'\bautonomous\s+driv', r'\bself.driving\b', r'\bautonomous\s+vehicl',
        r'\bev\s+battery\b', r'\bbattery\s+cell\b', r'\bbattery\s+pack\b',
        r'\bev\s+range\b', r'\bev\s+charg', r'\bfast\s+charg',
        r'\bformula\s*e\b',  # Formula E is EV-relevant
    ]
    core_hits, core_matched = kw(text, core_pats)

    # === INDUSTRY (tier 2) ===
    industry_pats = [
        r'\bautomotive\b', r'\bautomaker\b', r'\bcar\s+maker\b', r'\bcar\s+manufactur',
        r'\btesla\b', r'\brivian\b', r'\blucid\b', r'\bvolkswagen\b', r'\bvw\b',
        r'\baudi\b', r'\btoyota\b', r'\bhyundai\b', r'\bkia\b',
        r'\bnio\b', r'\bxpeng\b', r'\bli\s+auto\b', r'\bpolestar\b',
        r'\belectric\s+vehicl', r'\bev\s+market\b', r'\bev\s+sales\b',
        r'\bev\s+adopt', r'\bev\s+demand\b',
        r'\bcharging\s+infra', r'\bcharging\s+station', r'\bcharging\s+network',
        r'\bbattery\b.*(?:ev|electric|vehicle|auto|cell|pack|gigafact|lithium|solid|charg|range|kwh|mwh|gwh|cathode|anode|recycl)',
        r'(?:ev|electric|vehicle|auto|lithium).*\bbattery\b',
        r'\bcell\s+manufactur', r'\bgigafactor',
        r'\benergy\s+storage\b.*(?:battery|grid|ev|gwh|mwh)', r'\benergy\s+densit',
        r'\bgerman\s+auto', r'\bstuttgart\b',
        r'\bvda\b', r'\bfraunhofer\b',
        r'\beu\b.*vehicl', r'\beu\b.*emission', r'\beu\b.*auto',
        r'\bgreen\s+deal\b', r'\bfit\s+for\s+55\b',
        r'\blidar\b', r'\badas\b', r'\bsemiconductor\b.*auto',
        r'\bchip\b.*auto', r'\bpower\s+electronics\b',
        r'\bhydrogen\b.*vehic', r'\bfuel\s+cell\b.*auto',
        r'\bsupply\s+chain\b.*auto', r'\bsupply\s+chain\b.*ev',
        r'\bcar\b.*electric', r'\bplug.in\s+hybrid\b', r'\bphev\b', r'\bbev\b',
    ]
    ind_hits, ind_matched = kw(text, industry_pats)

    # === ADJACENT (tier 3) ===
    # Be careful: \bcar\b matches "CAR-T therapy" etc. Only match in auto context.
    adjacent_pats = [
        r'\bcar\s+(maker|manufactur|industr|market|sales|dealer)', r'\bvehicle\b', r'\bdriving\b.*car',
        r'\bgerman\w*\b.*industr', r'\bgermany\b.*manufactur',
        r'\brenewable\s+energy\b',
        r'\blithium\b', r'\bcobalt\b', r'\bnickel\b.*batter',
        r'\bhydrogen\b.*fuel', r'\bfuel\s+cell\b',
        r'\bfactory\b.*auto', r'\bproduction\s+line\b',
        r'\btariff\b.*auto', r'\btrade\b.*auto',
    ]
    adj_hits, adj_matched = kw(text, adjacent_pats)

    # === NEGATIVE ===
    neg_pats = [
        r'\bnfl\b', r'\bnba\b', r'\bnhl\b', r'\bmlb\b',
        r'\btouchdown\b', r'\bquarterback\b',
        r'\bk-?pop\b', r'\bbonsai\b',
        r'\brecipe\b', r'\bcooking\b', r'\bcuisine\b',
        r'\bgene\s+therap', r'\bchemotherapy\b', r'\boncolog', r'\bcar-t\b',
        r'\bislamic\s+finance\b', r'\bsharia\b',
        r'\bfashion\b', r'\bclothing\b',
        r'\breal\s+estate\b', r'\bproperty\s+market\b',
        r'\bdocumentary\b', r'\bfilm\s+festival\b',
        r'\bformula\s*1\b', r'\bf1\b.*racing', r'\bgrand\s+prix\b',
    ]
    neg_hits, _ = kw(text, neg_pats)

    non_eng = is_non_english(f"{title} {summary}")

    # Score
    if company_hits >= 1 and core_hits >= 1:
        score = 9.0 + min(company_hits + core_hits - 2, 2) * 0.3
    elif company_hits >= 1 and ind_hits >= 2:
        score = 8.5
    elif company_hits >= 1 and ind_hits >= 1:
        score = 8.0
    elif company_hits >= 1:
        score = 7.0
    elif core_hits >= 3:
        score = 8.5
    elif core_hits >= 2:
        score = 8.0
    elif core_hits == 1 and ind_hits >= 2:
        score = 7.5
    elif core_hits == 1 and ind_hits >= 1:
        score = 7.0
    elif core_hits == 1:
        score = 6.5
    elif ind_hits >= 5:
        score = 7.5
    elif ind_hits >= 4:
        score = 7.0
    elif ind_hits >= 3:
        score = 6.5
    elif ind_hits == 2:
        score = 5.5
    elif ind_hits == 1 and adj_hits >= 2:
        score = 5.0
    elif ind_hits == 1 and adj_hits >= 1:
        score = 4.5
    elif ind_hits == 1:
        score = 3.5
    elif adj_hits >= 4:
        score = 3.5
    elif adj_hits >= 3:
        score = 3.0
    elif adj_hits >= 2:
        score = 2.0
    elif adj_hits == 1:
        score = 1.0
    else:
        score = 0.5

    if neg_hits >= 2:
        score = min(score, max(score * 0.2, 0.5))
    elif neg_hits == 1 and score < 7.0:
        score = min(score, 1.5)
    elif neg_hits == 1:
        score = max(score - 2.0, 2.0)

    if non_eng:
        if score < 7.0:
            score = min(score, 0.5)
        else:
            score = max(score - 3.0, 1.0)

    score = round(min(max(score, 0.0), 10.0), 1)
    all_matched = company_matched + core_matched + ind_matched + adj_matched
    return score, all_matched, non_eng, neg_hits


SCORERS = {
    "marketing_director_london": score_marketing,
    "auditor_toronto": score_auditor,
    "auto_engineer_stuttgart": score_auto,
}


def generate_reason(profile_id, score, matched, non_eng, neg_hits):
    profile = PROFILES[profile_id]
    if score >= 7.0:
        return f"Relevant to {profile['role']}: matches {', '.join(matched[:3])}"
    elif score >= 5.0:
        return f"Somewhat relevant: touches on {', '.join(matched[:2])}"
    elif score >= 2.0:
        return f"Tangentially related via {', '.join(matched[:2]) if matched else 'general topic'}"
    else:
        if non_eng:
            return f"Non-English content, not relevant to {profile_id}"
        return f"Not relevant to this profile's professional focus or interests"


def generate_think(article, profile_id, score, matched, non_eng, neg_hits):
    title_short = (article.get("title") or "")[:60]
    source = article.get("source", "unknown")
    profile = PROFILES[profile_id]
    parts = [
        f"Article '{title_short}' from {source}.",
        f"Profile is a {profile['role']} in {profile['location']}.",
    ]
    if matched:
        parts.append(f"Matched signals: {', '.join(matched[:4])}.")
    if neg_hits > 0:
        parts.append(f"Negative signals detected ({neg_hits}).")
    if non_eng:
        parts.append("Non-English content detected.")
    if score < 2.0:
        parts.append("The article's subject matter falls outside this profile's professional scope and personal interests.")
    elif score < 5.0:
        parts.append("Only tangential connection to profile's domain.")
    elif score < 7.0:
        parts.append("Some relevance to profile's broader industry interests.")
    else:
        parts.append("Strong match with profile's core professional interests and tracked areas.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

all_scores = []
profile_ids = list(SCORERS.keys())
batch_size = 50
total = len(articles) * len(profile_ids)

print(f"Scoring {total} article-profile pairs...")

for batch_start in range(0, len(articles), batch_size):
    batch_end = min(batch_start + batch_size, len(articles))
    batch_articles = articles[batch_start:batch_end]

    for article in batch_articles:
        for pid in profile_ids:
            scorer = SCORERS[pid]
            score, matched, non_eng, neg_hits = scorer(article)
            reason = generate_reason(pid, score, matched, non_eng, neg_hits)
            think_text = generate_think(article, pid, score, matched, non_eng, neg_hits)

            all_scores.append({
                "article_id": article["id"],
                "profile_id": pid,
                "score": score,
                "reason": reason[:200],
                "think_text": think_text[:400],
                "think_tokens": 100,
            })

    # Save progress
    with open(OUTFILE, "w") as f:
        json.dump(all_scores, f, indent=1, ensure_ascii=False)

    print(f"  Batch {batch_start//batch_size + 1}: articles {batch_start}-{batch_end-1}, total: {len(all_scores)}/{total}")

print(f"\nDone! Saved {len(all_scores)} scores to {OUTFILE}")

# Distribution stats
for pid in profile_ids:
    pid_scores = [s["score"] for s in all_scores if s["profile_id"] == pid]
    bins = {
        "9-10": sum(1 for s in pid_scores if s >= 9.0),
        "7-8.9": sum(1 for s in pid_scores if 7.0 <= s < 9.0),
        "5-6.9": sum(1 for s in pid_scores if 5.0 <= s < 7.0),
        "2-4.9": sum(1 for s in pid_scores if 2.0 <= s < 5.0),
        "0-1.9": sum(1 for s in pid_scores if s < 2.0),
    }
    avg = sum(pid_scores) / len(pid_scores) if pid_scores else 0
    print(f"\n{pid}: avg={avg:.2f}")
    print(f"  {bins}")
