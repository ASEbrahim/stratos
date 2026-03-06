"""
V2/V3 Training Pipeline — Profile Definitions
===============================================
50 diverse profiles for multi-profile contrastive training data generation.
10 V1 profiles (from autopilot.py) + 20 V2 profiles + 20 V3 profiles.
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

# ══════════════════════════════════════════════════════════════
# V3 NEW PROFILES (20 new — common professions, likely-user markets)
# ══════════════════════════════════════════════════════════════

V3_NEW_PROFILES = [
    # ── Education ──
    {
        "id": "math_teacher_texas",
        "role": "High school math teacher",
        "location": "Austin, Texas, USA",
        "context": "Teaching AP Calculus and Statistics, tracking education policy, EdTech tools, and STEM curriculum developments",
        "interests": ["math education", "EdTech", "AP curriculum", "STEM outreach", "Texas education policy"],
        "tracked_companies": "Texas Instruments, Desmos, Khan Academy, College Board",
        "tracked_institutions": "TEA, NCTM, UT Austin",
        "tracked_industries": "education, EdTech",
        "diversity_tag": "mid-career/education",
    },

    # ── Software Engineering ──
    {
        "id": "backend_engineer_singapore",
        "role": "Backend software engineer at a cloud infrastructure company",
        "location": "Singapore",
        "context": "Building distributed systems and microservices for Southeast Asian markets, tracking cloud and DevOps trends",
        "interests": ["distributed systems", "Kubernetes", "Go", "cloud infrastructure", "DevOps"],
        "tracked_companies": "AWS, Google Cloud, Grab, Sea Group",
        "tracked_institutions": "IMDA, NUS Computing",
        "tracked_industries": "cloud computing, software engineering, fintech",
        "diversity_tag": "mid-career/tech",
    },

    # ── Real Estate ──
    {
        "id": "real_estate_dubai",
        "role": "Real estate developer and investment manager",
        "location": "Dubai, UAE",
        "context": "Managing luxury residential and commercial development projects, tracking Dubai property market regulations and foreign investment trends",
        "interests": ["Dubai real estate", "property development", "off-plan sales", "RERA regulations", "luxury market"],
        "tracked_companies": "Emaar, DAMAC, Nakheel, Aldar Properties",
        "tracked_institutions": "RERA, Dubai Land Department, DIFC",
        "tracked_industries": "real estate, construction, finance",
        "diversity_tag": "senior/GCC",
    },

    # ── Marketing ──
    {
        "id": "marketing_director_london",
        "role": "Marketing director at a consumer goods company",
        "location": "London, UK",
        "context": "Leading brand strategy for FMCG products across European markets, tracking consumer behavior trends and digital advertising",
        "interests": ["brand strategy", "digital advertising", "consumer insights", "e-commerce", "sustainability marketing"],
        "tracked_companies": "Unilever, P&G, Diageo, WPP",
        "tracked_institutions": "CIM, ASA, IPA",
        "tracked_industries": "FMCG, advertising, retail",
        "diversity_tag": "senior/marketing",
    },

    # ── Accounting ──
    {
        "id": "auditor_toronto",
        "role": "Senior auditor at a Big Four accounting firm",
        "location": "Toronto, Canada",
        "context": "Leading audit engagements for TSX-listed companies, tracking IFRS changes, Canadian tax reform, and ESG reporting standards",
        "interests": ["IFRS standards", "ESG reporting", "Canadian tax policy", "forensic accounting", "audit technology"],
        "tracked_companies": "Deloitte, KPMG, PwC, EY",
        "tracked_institutions": "CPA Canada, OSC, IASB",
        "tracked_industries": "accounting, finance, regulatory",
        "diversity_tag": "mid-career/finance",
    },

    # ── Automotive ──
    {
        "id": "auto_engineer_stuttgart",
        "role": "EV powertrain engineer at an automotive manufacturer",
        "location": "Stuttgart, Germany",
        "context": "Developing electric vehicle drivetrain systems, tracking battery technology, EU emissions regulation, and EV market competition",
        "interests": ["EV powertrains", "battery technology", "solid-state batteries", "EU emissions regulation", "autonomous driving"],
        "tracked_companies": "Mercedes-Benz, Porsche, BMW, CATL, BYD",
        "tracked_institutions": "VDA, Fraunhofer ISE, KIT",
        "tracked_industries": "automotive, electric vehicles, energy storage",
        "diversity_tag": "mid-career/engineering",
    },

    # ── Journalism ──
    {
        "id": "journalist_dc",
        "role": "Investigative journalist covering technology and policy",
        "location": "Washington DC, USA",
        "context": "Covering the intersection of tech regulation, AI policy, and corporate lobbying for a major publication",
        "interests": ["tech regulation", "AI policy", "antitrust", "data privacy", "congressional hearings"],
        "tracked_companies": "Google, Meta, Apple, Microsoft, OpenAI",
        "tracked_institutions": "FTC, FCC, Senate Commerce Committee, NIST",
        "tracked_industries": "technology, media, government",
        "diversity_tag": "mid-career/media",
    },

    # ── E-commerce ──
    {
        "id": "ecommerce_istanbul",
        "role": "E-commerce founder and CEO",
        "location": "Istanbul, Turkey",
        "context": "Running a cross-border e-commerce platform selling Turkish goods to EU and GCC markets, tracking logistics, payments, and Turkish trade policy",
        "interests": ["cross-border e-commerce", "payment gateways", "Turkish exports", "last-mile delivery", "marketplace platforms"],
        "tracked_companies": "Trendyol, Hepsiburada, Getir, iyzico",
        "tracked_institutions": "Turkish Trade Ministry, DEIK, Istanbul Chamber of Commerce",
        "tracked_industries": "e-commerce, logistics, retail",
        "diversity_tag": "executive/non-English",
    },

    # ── Environmental ──
    {
        "id": "env_consultant_amsterdam",
        "role": "Environmental consultant specializing in carbon markets",
        "location": "Amsterdam, Netherlands",
        "context": "Advising corporations on EU ETS compliance, carbon offset strategies, and sustainability reporting under CSRD",
        "interests": ["carbon markets", "EU ETS", "CSRD reporting", "climate risk", "circular economy"],
        "tracked_companies": "Shell, Arcadis, South Pole, Climeworks",
        "tracked_institutions": "EU Commission, GRI, CDP, Dutch Environment Ministry",
        "tracked_industries": "environmental consulting, energy, finance",
        "diversity_tag": "mid-career/sustainability",
    },

    # ── HR ──
    {
        "id": "hr_director_sydney",
        "role": "HR director at a tech company",
        "location": "Sydney, Australia",
        "context": "Managing talent acquisition and retention for a 500-person tech company, tracking Australian employment law and remote work trends",
        "interests": ["talent acquisition", "remote work policy", "DEI", "Australian employment law", "HR tech"],
        "tracked_companies": "Atlassian, Canva, Culture Amp, Employment Hero",
        "tracked_institutions": "Fair Work Commission, AHRI",
        "tracked_industries": "technology, HR, professional services",
        "diversity_tag": "senior/HR",
    },

    # ── Dentistry ──
    {
        "id": "dentist_riyadh",
        "role": "Dentist and private practice owner",
        "location": "Riyadh, Saudi Arabia",
        "context": "Running a multi-chair dental clinic, tracking Saudi healthcare licensing, dental technology, and Vision 2030 healthcare privatization",
        "interests": ["dental implants", "orthodontics", "dental imaging", "Saudi healthcare policy", "practice management"],
        "tracked_companies": "Align Technology, Dentsply Sirona, Straumann, Henry Schein",
        "tracked_institutions": "Saudi MOH, SCFHS, Saudi Dental Society",
        "tracked_industries": "healthcare, dental, medical devices",
        "diversity_tag": "mid-career/healthcare/GCC",
    },

    # ── Aviation ──
    {
        "id": "airline_pilot_dubai",
        "role": "Commercial airline pilot (Boeing 777 captain)",
        "location": "Dubai, UAE",
        "context": "Flying long-haul routes for a major Gulf carrier, tracking aviation safety, fleet orders, route expansions, and pilot labor market",
        "interests": ["aviation safety", "fleet management", "pilot training", "airline route strategy", "aviation regulation"],
        "tracked_companies": "Emirates, Boeing, Airbus, Rolls-Royce",
        "tracked_institutions": "GCAA, IATA, ICAO",
        "tracked_industries": "aviation, aerospace, tourism",
        "diversity_tag": "senior/GCC/aviation",
    },

    # ── Social Media ──
    {
        "id": "social_media_la",
        "role": "Social media strategist and content creator",
        "location": "Los Angeles, California, USA",
        "context": "Managing brand partnerships and content strategy across TikTok, Instagram, and YouTube for lifestyle and tech brands",
        "interests": ["content creation", "influencer marketing", "TikTok algorithm", "brand partnerships", "video production"],
        "tracked_companies": "TikTok, Instagram, YouTube, Spotify",
        "tracked_institutions": "None",
        "tracked_industries": "social media, entertainment, advertising",
        "diversity_tag": "mid-career/creative",
    },

    # ── Electrical Trades ──
    {
        "id": "electrician_manchester",
        "role": "Electrical contractor and business owner",
        "location": "Manchester, UK",
        "context": "Running a 10-person electrical contracting firm, tracking UK building regulations, EV charger installations, and smart home technology",
        "interests": ["EV charging installation", "smart home systems", "UK wiring regulations", "solar PV", "small business"],
        "tracked_companies": "Tesla (Powerwall), Pod Point, Schneider Electric, Hager",
        "tracked_institutions": "NICEIC, IET, NAPIT",
        "tracked_industries": "electrical, construction, renewable energy",
        "diversity_tag": "mid-career/blue-collar",
    },

    # ── Construction Management ──
    {
        "id": "construction_pm_doha",
        "role": "Senior construction project manager",
        "location": "Doha, Qatar",
        "context": "Managing large-scale infrastructure and commercial building projects, tracking Qatar National Vision 2030 developments and GCC construction markets",
        "interests": ["project management", "BIM", "construction safety", "Qatar infrastructure", "contract management"],
        "tracked_companies": "Ashghal, Qatar Rail, Consolidated Contractors, QDVC",
        "tracked_institutions": "Qatar Ministry of Municipality, PMI",
        "tracked_industries": "construction, infrastructure, real estate",
        "diversity_tag": "senior/GCC/construction",
    },

    # ── Investment Banking ──
    {
        "id": "investment_analyst_hk",
        "role": "Equity research analyst at an investment bank",
        "location": "Hong Kong",
        "context": "Covering Asia-Pacific technology and semiconductor sectors, tracking IPOs, earnings, and regulatory changes in Chinese tech",
        "interests": ["semiconductor industry", "Asia tech equities", "IPO analysis", "Chinese tech regulation", "HKEX listings"],
        "tracked_companies": "TSMC, Samsung, Alibaba, Tencent, SMIC",
        "tracked_institutions": "HKEX, SFC, CSRC",
        "tracked_industries": "investment banking, semiconductors, technology",
        "diversity_tag": "mid-career/finance",
    },

    # ── Public Health ──
    {
        "id": "public_health_geneva",
        "role": "Public health researcher and epidemiologist",
        "location": "Geneva, Switzerland",
        "context": "Working on global disease surveillance and pandemic preparedness, tracking WHO policy, vaccination programs, and infectious disease outbreaks",
        "interests": ["epidemiology", "pandemic preparedness", "vaccination policy", "global health governance", "antimicrobial resistance"],
        "tracked_companies": "Pfizer, Moderna, GSK, bioMerieux",
        "tracked_institutions": "WHO, GAVI, Wellcome Trust, Swiss TPH",
        "tracked_industries": "public health, pharmaceuticals, policy",
        "diversity_tag": "senior/healthcare",
    },

    # ── Logistics ──
    {
        "id": "trucking_owner_atlanta",
        "role": "Trucking and logistics company owner",
        "location": "Atlanta, Georgia, USA",
        "context": "Running a 30-truck fleet serving the Southeast US, tracking diesel prices, FMCSA regulations, driver shortages, and EV truck adoption",
        "interests": ["fleet management", "diesel markets", "FMCSA regulations", "EV trucks", "freight rates"],
        "tracked_companies": "Freightliner, Kenworth, Tesla Semi, Samsara",
        "tracked_institutions": "FMCSA, ATA, Georgia DOT",
        "tracked_industries": "trucking, logistics, transportation",
        "diversity_tag": "executive/blue-collar",
    },

    # ── Veterinary ──
    {
        "id": "veterinarian_stockholm",
        "role": "Veterinarian and animal clinic owner",
        "location": "Stockholm, Sweden",
        "context": "Running a companion animal clinic, tracking veterinary medicine advances, Swedish animal welfare regulations, and pet industry trends",
        "interests": ["veterinary medicine", "animal welfare", "pet nutrition", "veterinary imaging", "Swedish regulations"],
        "tracked_companies": "Zoetis, IDEXX, Royal Canin, Agria Pet Insurance",
        "tracked_institutions": "Swedish Board of Agriculture, SVA, SVF",
        "tracked_industries": "veterinary, pet care, healthcare",
        "diversity_tag": "mid-career/healthcare/non-English",
    },

    # ── IT Infrastructure ──
    {
        "id": "sysadmin_osaka",
        "role": "IT systems administrator at a manufacturing company",
        "location": "Osaka, Japan",
        "context": "Managing on-prem and hybrid cloud infrastructure for a mid-size manufacturer, tracking cybersecurity threats, cloud migration, and Japanese IT regulations",
        "interests": ["network security", "cloud migration", "Active Directory", "VMware", "Japanese data protection"],
        "tracked_companies": "Microsoft, VMware, Cisco, NTT Data",
        "tracked_institutions": "IPA Japan, NISC",
        "tracked_industries": "IT infrastructure, manufacturing, cybersecurity",
        "diversity_tag": "mid-career/tech/non-English",
    },
]

ALL_PROFILES = V1_PROFILES + V2_NEW_PROFILES + V3_NEW_PROFILES


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
                        'japan', 'osaka', 'brazil', 'france', 'south korea', 'korea',
                        'usa', 'texas', 'austin', 'chicago', 'illinois', 'houston', 'atlanta', 'georgia',
                        'washington', 'los angeles', 'california',
                        'norway', 'canada', 'alberta', 'toronto', 'germany', 'stuttgart', 'berlin',
                        'india', 'mumbai', 'australia', 'sydney', 'portugal', 'lisbon', 'uk', 'london',
                        'manchester', 'nigeria', 'lagos', 'chile', 'santiago', 'austria', 'vienna',
                        'mexico', 'singapore', 'turkey', 'istanbul', 'netherlands', 'amsterdam',
                        'hong kong', 'switzerland', 'geneva', 'sweden', 'stockholm']:
            if country in loc:
                # Normalize
                norm = {'texas': 'USA', 'chicago': 'USA', 'illinois': 'USA', 'houston': 'USA',
                        'austin': 'USA', 'atlanta': 'USA', 'georgia': 'USA',
                        'washington': 'USA', 'los angeles': 'USA', 'california': 'USA',
                        'dubai': 'UAE', 'berlin': 'Germany', 'stuttgart': 'Germany',
                        'mumbai': 'India', 'sydney': 'Australia', 'osaka': 'Japan',
                        'lisbon': 'Portugal', 'london': 'UK', 'manchester': 'UK',
                        'lagos': 'Nigeria', 'santiago': 'Chile', 'vienna': 'Austria',
                        'toronto': 'Canada', 'alberta': 'Canada',
                        'seoul': 'South Korea', 'korea': 'South Korea',
                        'saudi': 'Saudi Arabia', 'bahrain': 'Bahrain', 'oman': 'Oman',
                        'qatar': 'Qatar', 'kuwait': 'Kuwait', 'japan': 'Japan',
                        'brazil': 'Brazil', 'france': 'France', 'norway': 'Norway',
                        'germany': 'Germany', 'india': 'India', 'australia': 'Australia',
                        'portugal': 'Portugal', 'nigeria': 'Nigeria', 'chile': 'Chile',
                        'austria': 'Austria', 'mexico': 'Mexico', 'usa': 'USA',
                        'uae': 'UAE', 'canada': 'Canada', 'uk': 'UK',
                        'singapore': 'Singapore', 'turkey': 'Turkey', 'istanbul': 'Turkey',
                        'netherlands': 'Netherlands', 'amsterdam': 'Netherlands',
                        'hong kong': 'Hong Kong', 'switzerland': 'Switzerland',
                        'geneva': 'Switzerland', 'sweden': 'Sweden',
                        'stockholm': 'Sweden'}.get(country, country.title())
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
