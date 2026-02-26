"""
V2 Training Pipeline — Profile Definitions
============================================
30 diverse profiles for multi-profile contrastive training data generation.
10 existing V1 profiles (from autopilot.py) + 20 new profiles.
"""

# ══════════════════════════════════════════════════════════════
# V1 PROFILES (existing — from autopilot.py PROFILE_TEMPLATES)
# ══════════════════════════════════════════════════════════════

V1_PROFILES = [
    {
        "id": "cpeg_student_kw",
        "role": "Computer Engineering student at American University of Kuwait",
        "location": "Kuwait",
        "context": "Seeking first engineering position in Kuwait oil/gas or tech sector",
        "interests": ["AI", "quantum computing", "semiconductors", "fintech", "embedded systems"],
        "tracked_companies": "Equate, SLB, Halliburton, KOC, KNPC",
        "tracked_institutions": "Warba Bank, Boubyan Bank, NBK",
        "tracked_industries": "oil and gas, fintech",
        "diversity_tag": "student/GCC",
    },
    {
        "id": "petrol_eng_kw",
        "role": "Petroleum Engineering student at Kuwait University",
        "location": "Kuwait",
        "context": "Final-year petroleum engineering student interested in upstream operations and reservoir simulation",
        "interests": ["reservoir simulation", "drilling technology", "upstream operations", "seismic interpretation"],
        "tracked_companies": "KOC, KNPC, KIPIC, SLB, Halliburton",
        "tracked_institutions": "Kuwait University",
        "tracked_industries": "oil and gas, energy",
        "diversity_tag": "student/GCC",
    },
    {
        "id": "finance_student_kw",
        "role": "Finance & Accounting student at GUST Kuwait",
        "location": "Kuwait",
        "context": "Finance student tracking GCC markets, banking sector, and fintech disruption",
        "interests": ["Islamic finance", "portfolio management", "equity research", "blockchain", "banking"],
        "tracked_companies": "NBK, KFH, Boursa Kuwait, Agility",
        "tracked_institutions": "Boursa Kuwait, CMA Kuwait",
        "tracked_industries": "banking, fintech, capital markets",
        "diversity_tag": "student/GCC",
    },
    {
        "id": "cybersecurity_kw",
        "role": "Cybersecurity analyst at a Kuwaiti bank",
        "location": "Kuwait",
        "context": "SOC analyst protecting banking infrastructure, tracking threat intelligence and compliance",
        "interests": ["threat intelligence", "SIEM", "zero trust", "cloud security", "compliance"],
        "tracked_companies": "CrowdStrike, Palo Alto Networks, Fortinet",
        "tracked_institutions": "CITRA Kuwait, CBK",
        "tracked_industries": "cybersecurity, banking",
        "diversity_tag": "mid-career/GCC",
    },
    {
        "id": "geophysicist_koc_kw",
        "role": "Senior geophysicist at KOC",
        "location": "Kuwait",
        "context": "Handles acquisition & processing of seismic data, also manages legal tenders for geophysical services",
        "interests": ["seismic processing", "velocity modeling", "subsurface imaging", "geophysical tenders"],
        "tracked_companies": "KOC, CGG, PGS, TGS, ION Geophysical",
        "tracked_institutions": "SEG, EAGE",
        "tracked_industries": "oil and gas, geoscience",
        "diversity_tag": "senior/GCC",
    },
    {
        "id": "meche_grad_sa",
        "role": "Mechanical Engineering fresh graduate seeking NEOM/Aramco roles",
        "location": "Saudi Arabia",
        "context": "Recent graduate targeting Vision 2030 mega-projects, NEOM construction, and Aramco engineering roles",
        "interests": ["NEOM", "Saudi Aramco", "HVAC design", "project management", "renewable energy"],
        "tracked_companies": "Saudi Aramco, NEOM, SABIC, ACWA Power",
        "tracked_institutions": "KFUPM, Saudi Engineers Authority",
        "tracked_industries": "oil and gas, construction, renewable energy",
        "diversity_tag": "student/GCC",
    },
    {
        "id": "data_scientist_dubai",
        "role": "Data Scientist at a Dubai fintech startup",
        "location": "Dubai, UAE",
        "context": "Building ML models for fraud detection and credit scoring in the GCC financial ecosystem",
        "interests": ["machine learning", "fraud detection", "NLP", "MLOps", "Python"],
        "tracked_companies": "Careem, Tabby, Sarwa, Mashreq Neo",
        "tracked_institutions": "DIFC, DFSA",
        "tracked_industries": "fintech, AI",
        "diversity_tag": "mid-career/GCC",
    },
    {
        "id": "supply_chain_bahrain",
        "role": "Supply chain analyst at a logistics company",
        "location": "Manama, Bahrain",
        "context": "Managing GCC supply chain operations, tracking port logistics and trade route disruptions",
        "interests": ["supply chain analytics", "port logistics", "trade compliance", "SAP", "inventory optimization"],
        "tracked_companies": "DHL, Maersk, DP World, Gulf Air Cargo",
        "tracked_institutions": "Bahrain Economic Board",
        "tracked_industries": "logistics, trade",
        "diversity_tag": "mid-career/GCC",
    },
    {
        "id": "biotech_researcher_sa",
        "role": "Biotech researcher at KAUST studying gene therapy",
        "location": "Jeddah, Saudi Arabia",
        "context": "Postdoctoral researcher in gene therapy and CRISPR applications, publishing in high-impact journals",
        "interests": ["CRISPR", "gene therapy", "bioinformatics", "clinical trials", "molecular biology"],
        "tracked_companies": "Illumina, Novartis, CRISPR Therapeutics",
        "tracked_institutions": "KAUST, SFDA",
        "tracked_industries": "biotechnology, pharmaceuticals",
        "diversity_tag": "senior/GCC",
    },
    {
        "id": "architect_qatar",
        "role": "Architect working on smart city infrastructure projects",
        "location": "Doha, Qatar",
        "context": "Designing sustainable urban developments for Qatar's post-World Cup infrastructure expansion",
        "interests": ["smart cities", "sustainable architecture", "BIM", "parametric design", "urban planning"],
        "tracked_companies": "Msheireb Properties, Qatari Diar, Foster + Partners",
        "tracked_institutions": "Qatar Foundation, QSTP",
        "tracked_industries": "architecture, construction, urban development",
        "diversity_tag": "mid-career/GCC",
    },
]

# ══════════════════════════════════════════════════════════════
# V2 NEW PROFILES (20 new — maximum diversity)
# ══════════════════════════════════════════════════════════════

V2_NEW_PROFILES = [
    # ── Non-English Markets (5 profiles) ──
    {
        "id": "ux_designer_tokyo",
        "role": "UX Designer at a consumer electronics company",
        "location": "Tokyo, Japan",
        "context": "Designing mobile interfaces for Japanese market, tracking design trends and tech product launches",
        "interests": ["mobile UX", "design systems", "accessibility", "user research", "Japanese consumer tech"],
        "tracked_companies": "Sony, Nintendo, LINE, Rakuten",
        "tracked_institutions": "Japan Design Foundation",
        "tracked_industries": "consumer electronics, design, mobile apps",
        "diversity_tag": "mid-career/non-English/creative",
    },
    {
        "id": "agribiz_sao_paulo",
        "role": "Agricultural commodities trader",
        "location": "São Paulo, Brazil",
        "context": "Trading soy, corn, and coffee futures on B3 exchange, monitoring Brazilian crop yields and export policy",
        "interests": ["soybean futures", "Brazilian agriculture", "commodity markets", "weather patterns", "USDA reports"],
        "tracked_companies": "Cargill, Bunge, JBS, BRF",
        "tracked_institutions": "B3 Exchange, Embrapa, CONAB",
        "tracked_industries": "agriculture, commodities trading",
        "diversity_tag": "mid-career/non-English",
    },
    {
        "id": "pharmacist_paris",
        "role": "Hospital pharmacist specializing in oncology",
        "location": "Paris, France",
        "context": "Managing chemotherapy protocols at a Paris teaching hospital, tracking drug approvals and clinical trials",
        "interests": ["oncology drugs", "clinical pharmacology", "drug interactions", "EU drug regulation", "immunotherapy"],
        "tracked_companies": "Sanofi, Roche, Novartis, AstraZeneca",
        "tracked_institutions": "ANSM, EMA, AP-HP",
        "tracked_industries": "pharmaceuticals, healthcare",
        "diversity_tag": "mid-career/non-English/healthcare",
    },
    {
        "id": "kpop_marketer_seoul",
        "role": "Digital marketing manager at a K-pop entertainment agency",
        "location": "Seoul, South Korea",
        "context": "Managing global fan engagement campaigns, social media strategy for K-pop artists and merchandise",
        "interests": ["K-pop industry", "social media marketing", "fan engagement", "content strategy", "live streaming"],
        "tracked_companies": "HYBE, SM Entertainment, JYP Entertainment, YG Entertainment",
        "tracked_institutions": "KOCCA, Korea Creative Content Agency",
        "tracked_industries": "entertainment, digital marketing, music",
        "diversity_tag": "mid-career/non-English/creative",
    },
    {
        "id": "pediatric_oncologist_riyadh",
        "role": "Pediatric oncologist at King Faisal Specialist Hospital",
        "location": "Riyadh, Saudi Arabia",
        "context": "Treating childhood cancers, running clinical trials, and tracking Saudi healthcare policy changes",
        "interests": ["pediatric leukemia", "CAR-T therapy", "clinical trials", "Saudi health policy", "precision oncology"],
        "tracked_companies": "Novartis (Kymriah), Gilead (Yescarta), Roche",
        "tracked_institutions": "King Faisal Specialist Hospital, SCFHS, Saudi MOH",
        "tracked_industries": "healthcare, oncology, pharmaceuticals",
        "diversity_tag": "senior/non-English/healthcare",
    },

    # ── Blue-Collar / Trades (3 profiles) ──
    {
        "id": "hvac_tech_texas",
        "role": "HVAC technician and small business owner",
        "location": "Houston, Texas, USA",
        "context": "Running a 5-person HVAC company, tracking EPA regulations, refrigerant changes, and Texas building codes",
        "interests": ["HVAC systems", "EPA refrigerant regulations", "heat pump technology", "building codes", "small business"],
        "tracked_companies": "Carrier, Trane, Daikin, Lennox",
        "tracked_institutions": "EPA, ACCA, Texas TDLR",
        "tracked_industries": "HVAC, construction, home services",
        "diversity_tag": "mid-career/blue-collar",
    },
    {
        "id": "marine_electrician_norway",
        "role": "Marine electrician on offshore wind vessels",
        "location": "Stavanger, Norway",
        "context": "Working on offshore wind installation vessels, maintaining high-voltage electrical systems in harsh marine environments",
        "interests": ["offshore wind", "marine electrical systems", "high-voltage safety", "vessel maintenance", "North Sea operations"],
        "tracked_companies": "Equinor, Orsted, Siemens Gamesa, Fred. Olsen Windcarrier",
        "tracked_institutions": "Norwegian Maritime Authority, DNV",
        "tracked_industries": "offshore wind, maritime, energy",
        "diversity_tag": "mid-career/blue-collar/non-English",
    },
    {
        "id": "welder_pipeline_alberta",
        "role": "Pipeline welder and CWB-certified inspector",
        "location": "Edmonton, Alberta, Canada",
        "context": "Welding on oil sands pipeline projects, tracking Alberta energy policy, carbon capture projects, and welding certifications",
        "interests": ["pipeline welding", "CWB certifications", "carbon capture", "Alberta oil sands", "hydrogen pipelines"],
        "tracked_companies": "TC Energy, Enbridge, Suncor, CNRL",
        "tracked_institutions": "CWB Group, Alberta Energy Regulator",
        "tracked_industries": "oil and gas, pipeline construction",
        "diversity_tag": "mid-career/blue-collar",
    },

    # ── Creative Industries (2 profiles) ──
    {
        "id": "indie_gamedev_berlin",
        "role": "Indie game developer and studio founder",
        "location": "Berlin, Germany",
        "context": "Building narrative-driven indie games in Godot, tracking game industry trends, funding, and Steam marketplace",
        "interests": ["Godot engine", "indie games", "game design", "Steam marketplace", "game funding"],
        "tracked_companies": "Valve, Epic Games, Devolver Digital, Raw Fury",
        "tracked_institutions": "German Games Industry Association, Medienboard Berlin",
        "tracked_industries": "gaming, entertainment, software",
        "diversity_tag": "mid-career/creative",
    },
    {
        "id": "filmmaker_mumbai",
        "role": "Independent documentary filmmaker",
        "location": "Mumbai, India",
        "context": "Producing social-impact documentaries, applying to international film festivals, tracking streaming platform acquisitions",
        "interests": ["documentary filmmaking", "film festivals", "streaming platforms", "social impact", "cinematography"],
        "tracked_companies": "Netflix, Amazon Prime Video, Zee Studios, MUBI",
        "tracked_institutions": "NFDC, Sundance Institute, TIFF",
        "tracked_industries": "film, streaming, media",
        "diversity_tag": "mid-career/creative/non-English",
    },

    # ── Healthcare (2 additional profiles) ──
    {
        "id": "physical_therapist_sydney",
        "role": "Sports physiotherapist at a professional rugby club",
        "location": "Sydney, Australia",
        "context": "Managing injury rehabilitation for elite rugby players, tracking sports science research and physiotherapy techniques",
        "interests": ["sports rehabilitation", "ACL recovery", "load management", "physiotherapy research", "concussion protocols"],
        "tracked_companies": "Australian Rugby, Physio Inq",
        "tracked_institutions": "APA, Sports Medicine Australia",
        "tracked_industries": "healthcare, sports medicine",
        "diversity_tag": "mid-career/healthcare",
    },
    {
        "id": "nurse_practitioner_toronto",
        "role": "Emergency department nurse practitioner",
        "location": "Toronto, Canada",
        "context": "Senior NP in a Level 1 trauma center, tracking Canadian healthcare policy, nursing staffing crisis, and emergency medicine research",
        "interests": ["emergency medicine", "nursing workforce", "telehealth", "Canadian health policy", "trauma care"],
        "tracked_companies": "Ontario Health, Sunnybrook Hospital",
        "tracked_institutions": "RNAO, Canadian Nurses Association",
        "tracked_industries": "healthcare, nursing",
        "diversity_tag": "senior/healthcare",
    },

    # ── Pure Hobbyist (1 profile) ──
    {
        "id": "hobbyist_crypto_f1",
        "role": "Retired IT consultant",
        "location": "Lisbon, Portugal",
        "context": "No career angle — purely tracking personal interests in crypto markets, Formula 1 racing, whisky investing, and vintage watches",
        "interests": ["cryptocurrency", "Formula 1", "whisky collecting", "vintage watches", "travel"],
        "tracked_companies": "Coinbase, Binance, Ferrari, Rolex",
        "tracked_institutions": "None",
        "tracked_industries": "None",
        "diversity_tag": "retired/hobbyist",
    },

    # ── Edge Cases (3 profiles) ──
    {
        "id": "lawyer_ai_disruption",
        "role": "Corporate lawyer at a major law firm",
        "location": "London, UK",
        "context": "Paradox profile: practices corporate M&A law while actively tracking how AI is replacing legal work. Interested in both defending the profession and preparing for its transformation.",
        "interests": ["AI in law", "legal tech", "M&A transactions", "AI regulation", "contract automation"],
        "tracked_companies": "Harvey AI, Clio, Thomson Reuters, Allen & Overy",
        "tracked_institutions": "Law Society, SRA, EU AI Act regulators",
        "tracked_industries": "legal, AI, technology",
        "diversity_tag": "senior/edge-case",
    },
    {
        "id": "bonsai_competitor_kyoto",
        "role": "Professional bonsai artist and competition judge",
        "location": "Kyoto, Japan",
        "context": "Extremely niche: competitive bonsai grower, travels internationally for exhibitions, tracks rare species trading and Japanese horticultural traditions",
        "interests": ["bonsai cultivation", "Japanese horticulture", "rare plant species", "bonsai exhibitions", "suiseki"],
        "tracked_companies": "None specific",
        "tracked_institutions": "Nippon Bonsai Association, World Bonsai Friendship Federation",
        "tracked_industries": "horticulture, art",
        "diversity_tag": "senior/edge-case/non-English",
    },
    {
        "id": "undeclared_student_chicago",
        "role": "Undeclared sophomore at University of Chicago",
        "location": "Chicago, Illinois, USA",
        "context": "Broadly curious student who hasn't picked a major — following economics, philosophy, climate science, and startup culture equally",
        "interests": ["economics", "philosophy", "climate change", "startup culture", "behavioral science"],
        "tracked_companies": "Y Combinator, OpenAI",
        "tracked_institutions": "University of Chicago, Federal Reserve",
        "tracked_industries": "varied",
        "diversity_tag": "student/edge-case",
    },

    # ── Seniority / Geography Gap-Fillers (4 profiles) ──
    {
        "id": "cto_fintech_lagos",
        "role": "CTO of a mobile payments startup",
        "location": "Lagos, Nigeria",
        "context": "Leading engineering at a Series B fintech, building mobile money infrastructure for West African markets",
        "interests": ["mobile payments", "USSD technology", "African fintech", "API design", "regulatory compliance"],
        "tracked_companies": "Flutterwave, Paystack, OPay, MTN MoMo",
        "tracked_institutions": "CBN, NITDA",
        "tracked_industries": "fintech, mobile payments, banking",
        "diversity_tag": "executive/Africa",
    },
    {
        "id": "mining_engineer_santiago",
        "role": "Mining engineer at a copper mine",
        "location": "Santiago, Chile",
        "context": "Managing open-pit copper extraction operations, tracking lithium demand, Chilean mining policy, and ESG regulations",
        "interests": ["copper mining", "lithium extraction", "mine safety", "ESG compliance", "Chilean mining law"],
        "tracked_companies": "Codelco, BHP, Antofagasta Minerals, SQM",
        "tracked_institutions": "Sernageomin, Cochilco",
        "tracked_industries": "mining, metals, energy transition",
        "diversity_tag": "mid-career/non-English",
    },
    {
        "id": "retired_diplomat_vienna",
        "role": "Retired UN diplomat and policy consultant",
        "location": "Vienna, Austria",
        "context": "Former IAEA and UN official now consulting on nuclear non-proliferation and Middle East policy. Tracks geopolitics, arms control, and international law.",
        "interests": ["nuclear non-proliferation", "Middle East geopolitics", "international law", "arms control", "UN reform"],
        "tracked_companies": "None",
        "tracked_institutions": "IAEA, UN Security Council, OPCW",
        "tracked_industries": "diplomacy, international relations",
        "diversity_tag": "retired/edge-case",
    },
    {
        "id": "chef_mexico_city",
        "role": "Executive chef and restaurant owner",
        "location": "Mexico City, Mexico",
        "context": "Running a modern Mexican restaurant, tracking food industry trends, supply chain issues, and culinary awards",
        "interests": ["Mexican cuisine", "restaurant management", "food supply chain", "culinary awards", "sustainable sourcing"],
        "tracked_companies": "Sysco, US Foods",
        "tracked_institutions": "James Beard Foundation, Michelin Guide",
        "tracked_industries": "food service, hospitality, agriculture",
        "diversity_tag": "mid-career/non-English/creative",
    },
]

ALL_PROFILES = V1_PROFILES + V2_NEW_PROFILES


def print_profile_table():
    """Print all 30 profiles in a table format."""
    print(f"\n{'#':<3} {'ID':<28} {'Role':<55} {'Location':<25} {'Diversity Tag'}")
    print(f"{'─'*3} {'─'*28} {'─'*55} {'─'*25} {'─'*30}")
    for i, p in enumerate(ALL_PROFILES, 1):
        role_short = p['role'][:53] + '..' if len(p['role']) > 55 else p['role']
        loc_short = p['location'][:23] + '..' if len(p['location']) > 25 else p['location']
        print(f"{i:<3} {p['id']:<28} {role_short:<55} {loc_short:<25} {p.get('diversity_tag', '')}")

    # Diversity audit
    print(f"\n{'='*120}")
    print("DIVERSITY AUDIT")
    print(f"{'='*120}")

    countries = set()
    for p in ALL_PROFILES:
        loc = p['location'].lower()
        for country in ['kuwait', 'saudi', 'uae', 'dubai', 'bahrain', 'oman', 'qatar',
                        'japan', 'brazil', 'france', 'south korea', 'korea', 'usa', 'texas', 'chicago', 'illinois',
                        'norway', 'canada', 'alberta', 'toronto', 'germany', 'berlin', 'india', 'mumbai',
                        'australia', 'sydney', 'portugal', 'lisbon', 'uk', 'london',
                        'nigeria', 'lagos', 'chile', 'santiago', 'austria', 'vienna', 'mexico']:
            if country in loc:
                # Normalize
                norm = {'texas': 'USA', 'chicago': 'USA', 'illinois': 'USA', 'houston': 'USA',
                        'dubai': 'UAE', 'berlin': 'Germany', 'mumbai': 'India', 'sydney': 'Australia',
                        'lisbon': 'Portugal', 'london': 'UK', 'lagos': 'Nigeria',
                        'santiago': 'Chile', 'vienna': 'Austria', 'toronto': 'Canada',
                        'alberta': 'Canada', 'seoul': 'South Korea', 'korea': 'South Korea',
                        'saudi': 'Saudi Arabia', 'bahrain': 'Bahrain', 'oman': 'Oman',
                        'qatar': 'Qatar', 'kuwait': 'Kuwait', 'japan': 'Japan',
                        'brazil': 'Brazil', 'france': 'France', 'norway': 'Norway',
                        'germany': 'Germany', 'india': 'India', 'australia': 'Australia',
                        'portugal': 'Portugal', 'nigeria': 'Nigeria', 'chile': 'Chile',
                        'austria': 'Austria', 'mexico': 'Mexico', 'usa': 'USA',
                        'uae': 'UAE', 'canada': 'Canada', 'uk': 'UK'}.get(country, country.title())
                countries.add(norm)

    tags = [p.get('diversity_tag', '') for p in ALL_PROFILES]
    print(f"Total profiles: {len(ALL_PROFILES)}")
    print(f"Distinct countries: {len(countries)} — {sorted(countries)}")
    print(f"Non-English: {sum(1 for t in tags if 'non-English' in t)}")
    print(f"Blue-collar: {sum(1 for t in tags if 'blue-collar' in t)}")
    print(f"Creative: {sum(1 for t in tags if 'creative' in t)}")
    print(f"Healthcare: {sum(1 for t in tags if 'healthcare' in t)}")
    print(f"Hobbyist: {sum(1 for t in tags if 'hobbyist' in t)}")
    print(f"Edge-case: {sum(1 for t in tags if 'edge-case' in t)}")
    print(f"Student: {sum(1 for t in tags if 'student' in t)}")
    print(f"Mid-career: {sum(1 for t in tags if 'mid-career' in t)}")
    print(f"Senior: {sum(1 for t in tags if 'senior' in t)}")
    print(f"Executive: {sum(1 for t in tags if 'executive' in t)}")
    print(f"Retired: {sum(1 for t in tags if 'retired' in t)}")


if __name__ == "__main__":
    print_profile_table()
