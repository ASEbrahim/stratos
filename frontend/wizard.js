/**
 * wizard.js — StratOS Onboarding Wizard (dashboard-integrated)
 * Extracted from wizard_v4.html with theme-adaptive CSS custom properties.
 * Stages: 1-2 = CSS + modal shell, 3-7 = AI + settings (added incrementally)
 */
(function() {
'use strict';

/* ═══════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════ */

const CK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
const STEP_NAMES = ['Priorities','Details','Your Feed'];

/* ═══════════════════════════════════════
   DATA  (from wizard_v4.html)
   ═══════════════════════════════════════ */

const CATS = [
  {id:'career',icon:'🎯',name:'Career & Jobs',desc:'Jobs, internships, hiring, promotions',
   subs:[{id:'jobhunt',name:'Job Hunting'},{id:'intern',name:'Internships'},{id:'research_pos',name:'Research Positions'},{id:'govjobs',name:'Government Jobs'}]},
  {id:'industry',icon:'🏢',name:'Industry Intel',desc:'Company tracking, sector trends, competitors',
   subs:[{id:'ind_oilgas',name:'Oil & Gas'},{id:'ind_telecom',name:'Telecom'},{id:'ind_tech',name:'Tech / Software'},{id:'ind_bank',name:'Banking & Finance'},{id:'ind_construct',name:'Construction'},{id:'ind_health',name:'Healthcare'}]},
  {id:'learning',icon:'🎓',name:'Learning & Development',desc:'Certifications, courses, skill building, research',
   subs:[{id:'certs',name:'Professional Certifications'},{id:'academic',name:'Academic Research'},{id:'courses',name:'Online Courses'},{id:'hands_on',name:'Hands-on Skills'},{id:'confevents',name:'Conferences & Events'}]},
  {id:'markets',icon:'📈',name:'Markets & Investing',desc:'Stocks, crypto, commodities, analysis',
   subs:[{id:'stocks',name:'Stocks'},{id:'crypto',name:'Crypto'},{id:'commodities',name:'Commodities'},{id:'realestate',name:'Real Estate'},{id:'forex',name:'Forex'},{id:'mktreports',name:'Market Research & Reports'}]},
  {id:'deals',icon:'💎',name:'Deals & Offers',desc:'Bank offers, student discounts, promotions',
   subs:[{id:'bankdeals',name:'Bank Deals'},{id:'telepromo',name:'Telecom Promotions'},{id:'studisc',name:'Student Discounts'},{id:'ccrewards',name:'Credit Card Rewards'},{id:'empben',name:'Employee Benefits'}]},
  {id:'interests',icon:'🧭',name:'Interests & Trends',desc:'Hobbies, emerging tech, topics you follow',
   subs:[],dynamic:true},
];

const INTEREST_SUGGESTIONS = [
  'Quantum Computing','AI / ML Developments','Open Source Projects',
  'Gaming Tech','Startup Scene','Space Tech','IoT & Smart Devices',
  'Robotics','3D Printing','Cybersecurity Trends'
];

const DEF_CATS = new Set(['career','learning','deals']);
const DEF_SUBS = {
  career:new Set(['jobhunt','intern']),industry:new Set(),
  learning:new Set(['certs']),markets:new Set(),
  deals:new Set(['bankdeals','studisc']),interests:new Set(),
};

const PANELS = {
  career_opps:{icon:'🎯',name:'Career Opportunities',qs:[
    {id:'opptype',label:'What are you looking for?',type:'m',pills:['Full-time Jobs','Internships'],defs:['Full-time Jobs','Internships']},
    {id:'stage',label:'Your stage',type:'s',pills:['Student','Fresh Graduate','Mid-Career','Senior'],def:'Student'},
    {id:'roles',label:'Role types',type:'m',pills:['Engineering','Software Development','Data Science','IT / Networking','Management'],canAdd:1,defs:['Engineering','Software Development']},
  ]},
  jobhunt:{icon:'🔍',name:'Job Hunting',qs:[
    {id:'stage',label:'Your stage',type:'s',pills:['Student','Fresh Graduate','Mid-Career','Senior'],def:'Student'},
    {id:'roles',label:'Role types',type:'m',pills:['Engineering','Software Development','Data Science','IT / Networking','Management'],canAdd:1,defs:['Engineering','Software Development']},
  ]},
  intern:{icon:'🧑‍💻',name:'Internships',qs:[
    {id:'itype',label:'Preferred type',type:'m',pills:['Summer Internship','Co-op / Year-long','Part-time','Remote'],defs:['Summer Internship']},
    {id:'ifields',label:'Fields',type:'m',pills:['Engineering','Software Development','Data Science','IT / Networking'],canAdd:1,defs:['Engineering','Software Development']},
  ]},
  research_pos:{icon:'🔬',name:'Research Positions',qs:[
    {id:'rtype',label:'Type',type:'m',pills:['University Lab','Industry R&D','Government Research'],defs:['University Lab']},
    {id:'rfields',label:'Fields',type:'m',pills:['AI / ML','Embedded Systems','Networking','Energy'],canAdd:1,defs:['AI / ML']},
  ]},
  govjobs:{icon:'🏛️',name:'Government Jobs',qs:[
    {id:'gdept',label:'Departments',type:'m',pills:['Digital / IT Authority','Ministry of Communications','Energy Sector','Central Bank','Defense / Military'],canAdd:1,defs:[]},
  ]},
  ind_oilgas:{icon:'🛢️',name:'Oil & Gas',qs:[
    {id:'ogfocus',label:'Sub-focus',type:'m',pills:['Upstream','Downstream','Petrochemicals','OPEC Policy','Energy Transition'],defs:['Downstream','Petrochemicals']},
    {id:'ogco',label:'Companies to track',hint:'Use + Add or AI suggestions for local companies',type:'m',pills:['ExxonMobil','Shell','Chevron','BP','TotalEnergies'],canAdd:1,defs:[]},
  ]},
  ind_telecom:{icon:'📡',name:'Telecom',qs:[
    {id:'telfocus',label:'Sub-focus',type:'m',pills:['5G Rollout','Fiber / Infrastructure','Mobile Services','Enterprise Solutions'],defs:['5G Rollout']},
    {id:'telco',label:'Companies to track',type:'m',pills:['Verizon','AT&T','T-Mobile','Vodafone','Deutsche Telekom'],canAdd:1,defs:[]},
  ]},
  ind_tech:{icon:'💻',name:'Tech / Software',qs:[
    {id:'techfocus',label:'Sub-focus',type:'m',pills:['Cloud & SaaS','AI & Automation','Cybersecurity','Fintech','GovTech'],defs:['Cloud & SaaS','AI & Automation']},
    {id:'techco',label:'Companies to track',type:'m',pills:['Microsoft','Google','Apple','Amazon','Meta'],canAdd:1,defs:[]},
  ]},
  ind_bank:{icon:'🏦',name:'Banking & Finance',qs:[
    {id:'bfocus',label:'Sub-focus',type:'m',pills:['Retail Banking','Islamic Finance','Investment Banking','Fintech Disruption','Central Bank Policy'],defs:['Retail Banking']},
    {id:'bco',label:'Institutions to track',type:'m',pills:['JPMorgan Chase','Goldman Sachs','HSBC','Citigroup','Deutsche Bank'],canAdd:1,defs:[]},
  ]},
  ind_construct:{icon:'🏗️',name:'Construction',qs:[
    {id:'confocus',label:'Sub-focus',type:'m',pills:['Mega Projects','Smart Infrastructure','Residential','Government Tenders'],defs:['Mega Projects']},
  ]},
  ind_health:{icon:'🏥',name:'Healthcare',qs:[
    {id:'hfocus',label:'Sub-focus',type:'m',pills:['HealthTech','Pharmaceuticals','Digital Health','Medical Devices'],defs:['HealthTech']},
  ]},
  certs:{icon:'📜',name:'Professional Certifications',qs:[
    {id:'clevel',label:'Level',type:'s',pills:['Beginner / Entry','Intermediate','Advanced / Expert'],def:'Beginner / Entry'},
    {id:'cfocus',label:'Focus',type:'m',pills:['Cloud & Infrastructure','Networking','Security','Data & AI','Project Management'],canAdd:1,defs:['Cloud & Infrastructure','Networking']},
  ]},
  academic:{icon:'📚',name:'Academic Research',qs:[
    {id:'afields',label:'Fields',type:'m',pills:['Computer Engineering','AI / ML','Embedded Systems','Cybersecurity'],canAdd:1,defs:['Computer Engineering','AI / ML']},
    {id:'asrc',label:'Sources',type:'m',pills:['IEEE','ACM','arXiv','Google Scholar'],defs:['IEEE','arXiv']},
  ]},
  courses:{icon:'🖥️',name:'Online Courses',qs:[
    {id:'cplat',label:'Platforms',type:'m',pills:['Coursera','Udemy','edX','Pluralsight','LinkedIn Learning'],defs:['Coursera','Udemy']},
    {id:'ctop',label:'Topics',type:'m',pills:['Cloud Computing','Data Science','Web Development','DevOps'],canAdd:1,defs:[]},
  ]},
  hands_on:{icon:'🔧',name:'Hands-on Skills',qs:[
    {id:'hskills',label:'Skills',type:'m',pills:['Linux Administration','Networking Labs','Docker / K8s','Microcontrollers'],canAdd:1,defs:['Linux Administration']},
  ]},
  confevents:{icon:'🎤',name:'Conferences & Events',qs:[
    {id:'etypes',label:'Types',type:'m',pills:['Career Fairs','Tech Conferences','Hackathons','Meetups','Webinars'],defs:['Tech Conferences','Hackathons']},
    {id:'eregion',label:'Region',type:'s',pills:['Local / National','Regional','Global','Online Only'],def:'Global'},
  ]},
  stocks:{icon:'📊',name:'Stocks',qs:[
    {id:'smkt',label:'Markets',type:'m',pills:['US (NYSE/NASDAQ)','European Markets','Asian Markets','Global Indices'],canAdd:1,defs:['US (NYSE/NASDAQ)']},
    {id:'sdepth',label:'Tracking level',type:'s',pills:['Just notable news','Daily price tracking','Deep analysis'],def:'Just notable news'},
  ]},
  crypto:{icon:'₿',name:'Crypto',qs:[
    {id:'ccoins',label:'What to track',type:'m',pills:['Bitcoin','Ethereum','Altcoins','DeFi','Web3'],canAdd:1,defs:['Bitcoin','Ethereum']},
    {id:'cdepth',label:'Tracking level',type:'s',pills:['Just notable news','Daily price tracking'],def:'Just notable news'},
  ]},
  commodities:{icon:'🪙',name:'Commodities',qs:[
    {id:'comms',label:'Which',type:'m',pills:['Gold','Silver','Oil (Brent/WTI)','Natural Gas','Copper'],defs:['Gold','Oil (Brent/WTI)']},
  ]},
  realestate:{icon:'🏠',name:'Real Estate',qs:[
    {id:'refocus',label:'Focus',type:'m',pills:['Residential','Commercial','REITs','Industrial','International Markets'],canAdd:1,defs:['Residential']},
  ]},
  forex:{icon:'💱',name:'Forex',qs:[
    {id:'fpairs',label:'Pairs',type:'m',pills:['EUR/USD','GBP/USD','USD/JPY','USD/CHF'],canAdd:1,defs:['EUR/USD']},
  ]},
  mktreports:{icon:'📋',name:'Market Research & Reports',qs:[
    {id:'mrsrc',label:'Sources',type:'m',pills:['Bloomberg','Reuters','Morningstar','S&P Global'],canAdd:1,defs:['Bloomberg']},
  ]},
  bankdeals:{icon:'🏦',name:'Bank Deals',qs:[
    {id:'bdkinds',label:'What kind',type:'m',pills:['Salary / Allowance Transfer','New Account Bonuses','Credit Card Rewards','Loan Offers'],defs:['Salary / Allowance Transfer','New Account Bonuses']},
  ]},
  telepromo:{icon:'📱',name:'Telecom Promotions',qs:[
    {id:'tpkinds',label:'Types',type:'m',pills:['Data Plans','Device Deals','Roaming Offers','Bundle Packages'],defs:['Data Plans','Device Deals']},
  ]},
  studisc:{icon:'🎓',name:'Student Discounts',qs:[
    {id:'sdtypes',label:'Types',type:'m',pills:['Telecom Deals','Tech / Software','Food & Lifestyle','Travel'],canAdd:1,defs:['Telecom Deals','Tech / Software']},
  ]},
  ccrewards:{icon:'💳',name:'Credit Card Rewards',qs:[
    {id:'cctypes',label:'Reward types',type:'m',pills:['Cashback','Travel Miles','Points / Loyalty','Dining'],defs:['Cashback']},
  ]},
  empben:{icon:'🏢',name:'Employee Benefits',qs:[
    {id:'ebtypes',label:'Types',type:'m',pills:['Health Insurance','Education Allowance','Housing','Travel Perks'],defs:[]},
  ]},
};

/* GEN, ENTITY_TAGS, DISCOVER_MORE — REMOVED.
   Step 3 entities are now generated exclusively by /api/wizard-rv-items.
   See initRv(), initRvWithAI(), renderRail() for the AI-only flow. */

// Map wizard panel selections to Yahoo Finance ticker symbols
const WIZ_TICKER_MAP = {
  // Crypto panel (ccoins)
  'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD',
  // Commodities panel (comms)
  'Gold': 'GC=F', 'Silver': 'SI=F', 'Oil (Brent/WTI)': 'CL=F',
  'Natural Gas': 'NG=F', 'Copper': 'HG=F',
  // Forex panel (fpairs)
  'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X',
  'USD/JPY': 'JPY=X', 'USD/CHF': 'CHF=X',
  // Stock indices (smkt)
  'US (NYSE/NASDAQ)': '^GSPC', 'European Markets': '^STOXX50E',
  'Asian Markets': '^N225', 'Global Indices': '^GSPC',
};

/* ═══════════════════════════════════════
   ROLE CLASSIFICATION & ADAPTIVE PILLS
   ═══════════════════════════════════════ */

const ROLE_KEYWORDS = {
  tech:['software','developer','devops','sre','full.?stack','frontend','backend','web dev','mobile dev','ios dev','android dev','cloud arch','sysadmin','system admin','dba','qa ','test eng','automation eng','platform eng','infrastructure','tech lead','cto','vp eng','it manager','it director'],
  data:['data scien','machine learning','ml eng','\\bai\\b','artificial intelligence','deep learning','\\bnlp\\b','computer vision','data analy','business intelligence','\\bbi\\b','statistician','quantitative','data eng'],
  cybersec:['security eng','cybersec','infosec','penetration','ethical hack','soc analyst','threat','vulnerability','\\bciso\\b','security arch'],
  finance:['financ','\\bcfa\\b','accountant','accounting','audit','banker','banking','invest','portfolio','wealth','asset manag','fund manag','treasury','\\bcfo\\b','controller','actuari','risk manag','underwriter','loan','credit analy','tax ','bookkeep','financial plann'],
  marketing:['marketing','brand manag','advertis','\\bpr\\b','public relation','communications','social media','\\bseo\\b','\\bsem\\b','content strat','copywrite','creative director','\\bcmo\\b','growth','digital market','performance market'],
  sales:['sales','business develop','account exec','account manag','revenue','customer success','sales eng','presales'],
  healthcare:['doctor','physician','\\bnurse','medical','pharma','clinical','surgeon','dentist','therapist','psycholog','psychiatr','veterinar','optom','radiolog','patholog','biotech','biomedic','epidemiolog','public health','health admin','hospital'],
  education:['teacher','professor','lecturer','instructor','tutor','\\beducat','dean','principal','curriculum','librarian','school admin','training manag','instructional design'],
  legal:['lawyer','attorney','solicitor','barrister','legal','paralegal','compliance','regulatory','patent','trademark','contract manag','litigation','judge','magistrate','general counsel'],
  creative:['designer','\\bux\\b','\\bui ','graphic','illustrat','animat','video prod','film','photograph','art direc','creative','\\bwriter\\b','author','\\beditor\\b','journalist','reporter','producer','musician','composer','game design'],
  hr:['human resource','\\bhr ','recruit','talent','people operation','compensation','benefit','payroll','workforce','organizat.*develop','learning.*develop'],
  consulting:['consultant','advisor','strateg','management consult','business analyst'],
  operations:['operations','supply chain','logistics','procurement','warehouse','inventory','manufactur','production','quality manag','lean','six sigma','plant manag','\\bcoo\\b'],
  realestate:['real estate','realtor','broker','property manag','appraiser','mortgage'],
  hospitality:['hotel','restaurant','\\bchef\\b','culinary','pastry','bartend','hospitality','tourism','travel agent','concierge','event plan','catering','sommelier'],
  government:['government','public sector','civil serv','diplomat','policy analy','public admin','military','defense','intelligence analy','foreign affair'],
  trades:['electrician','plumber','carpenter','mechanic','welder','\\bhvac\\b','construction','mason','roofer','painter','landscap','technician','foreman'],
  science:['scientist','\\bresearch','biologist','chemist','physicist','geolog','marine bio','environment','ecolog','astrono','meteorolog','archaeolog','\\blab '],
  agriculture:['farmer','\\bagri','agrono','ranch','horticultur','forestry','fishery','viticultur','food scien'],
  nonprofit:['nonprofit','non.?profit','\\bngo\\b','social work','philanthrop','fundrais','community organ','humanitarian','advocacy'],
  retail:['retail','merchandis','store manag','buyer','visual merchan','e.?commerce'],
  engineering:['mechanical eng','civil eng','electrical eng','chemical eng','structural eng','aerospace','industrial eng','environmental eng','petroleum eng','materials eng','biomedical eng','robotics eng'],
};

function classifyRole(role) {
  if (!role) return 'default';
  const r = role.toLowerCase();
  let best = 'default', bestScore = 0;
  for (const [domain, keywords] of Object.entries(ROLE_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (new RegExp(kw, 'i').test(r)) score++;
    }
    if (score > bestScore) { bestScore = score; best = domain; }
  }
  return best;
}

function inferStage(role) {
  const r = role.toLowerCase();
  if (/student|undergrad|freshman|sophomore|junior(?! dev)|senior year|phd candidate|master.*student/.test(r)) return 'Student';
  if (/fresh.*grad|new grad|entry.?level|junior|associate|trainee|apprentice|graduate/.test(r)) return 'Fresh Graduate';
  if (/senior|lead|principal|staff|director|vp|vice.?president|chief|head of|c-level|executive|partner|managing/.test(r)) return 'Senior';
  return 'Mid-Career';
}

// Question-ID → domain → [pills]. Only questions where role matters.
const DOMAIN_PILLS = {
  roles: {
    tech:['Software Engineering','DevOps / SRE','Cloud Architecture','Full Stack Development','Backend / API'],
    data:['Data Science','ML Engineering','Data Analytics','Business Intelligence','AI Research'],
    cybersec:['Security Engineering','Penetration Testing','SOC Analysis','GRC / Compliance','Incident Response'],
    finance:['Financial Analysis','Risk Management','Investment Banking','Accounting / Audit','Treasury'],
    marketing:['Digital Marketing','Brand Management','Content Strategy','Growth / Analytics','PR / Communications'],
    sales:['Account Management','Business Development','Sales Engineering','Customer Success','Revenue Operations'],
    healthcare:['Clinical','Health Administration','Medical Research','Pharmaceuticals','Public Health'],
    education:['Teaching / Instruction','Curriculum Design','Education Administration','Instructional Design','Academic Research'],
    legal:['Corporate Law','Litigation','Compliance / Regulatory','Intellectual Property','Contract Management'],
    creative:['UX / UI Design','Graphic Design','Content Creation','Video / Film','Art Direction'],
    hr:['Talent Acquisition','HR Business Partner','Compensation & Benefits','Learning & Development','People Ops'],
    consulting:['Strategy Consulting','Management Consulting','IT Consulting','Operations Consulting','Advisory'],
    operations:['Supply Chain','Logistics','Procurement','Quality Management','Production / Manufacturing'],
    science:['Research Scientist','Lab Management','Field Research','Data Analysis','R&D'],
    hospitality:['Hotel Management','Food & Beverage','Event Planning','Tourism','Guest Services'],
    realestate:['Commercial Brokerage','Residential Sales','Property Management','Real Estate Development','Appraisal'],
    government:['Policy Analysis','Public Administration','Regulatory Affairs','Defense / Military','Diplomacy'],
    trades:['Project Supervision','Site Management','Technical Specialist','Maintenance','Safety / Inspection'],
    retail:['Store Management','Merchandising','E-commerce','Buying / Sourcing','Customer Experience'],
    agriculture:['Farm Management','Agronomy','Agricultural Research','Food Processing','Sustainability'],
    nonprofit:['Program Management','Fundraising','Advocacy / Policy','Community Development','Social Work'],
    engineering:['Mechanical Design','Civil / Structural','Electrical Systems','Process Engineering','Project Engineering'],
    default:['General Management','Operations','Analysis / Research','Coordination','Administration'],
  },
  ifields: {
    tech:['Software Engineering','Cloud / DevOps','Mobile Development','Data Engineering'],
    data:['Data Science','ML / AI','Business Analytics','Data Engineering'],
    cybersec:['Security Operations','Ethical Hacking','GRC','Digital Forensics'],
    finance:['Financial Analysis','Investment Research','Risk Management','Fintech'],
    marketing:['Digital Marketing','Content / Social','Market Research','Brand Strategy'],
    healthcare:['Clinical Research','Health Informatics','Public Health','Biotech'],
    legal:['Legal Research','Compliance','Intellectual Property','Regulatory'],
    creative:['UX / UI Design','Graphic / Visual','Content Production','Media'],
    engineering:['Mechanical Design','Electrical Systems','Civil Engineering','R&D'],
    science:['Lab Research','Field Research','Data Analysis','Environmental Science'],
    default:['Business Operations','Research / Analysis','Project Support','Communications'],
  },
  cfocus: {
    tech:['Cloud & Infrastructure','DevOps / CI-CD','Software Architecture','Data & AI','Security'],
    data:['Data & AI','Cloud Platforms','Python / R','Statistics & Analytics','MLOps'],
    cybersec:['Offensive Security','Defensive Security','Cloud Security','GRC','Digital Forensics'],
    finance:['CFA / Finance','Risk (FRM)','Accounting (CPA)','Financial Modeling','Compliance'],
    marketing:['Digital Marketing','Analytics','Content Strategy','Social Media','SEO / SEM'],
    healthcare:['Clinical Certifications','Health Informatics','Public Health','Healthcare Admin','Pharma'],
    legal:['Compliance','Privacy (CIPP)','Regulatory','Contract Management','Mediation'],
    creative:['UX Certification','Adobe Suite','Video / Motion','Accessibility','Design Thinking'],
    hr:['HR Certification (PHR/SPHR)','Talent Management','L&D','Compensation','DE&I'],
    operations:['Supply Chain (CSCP)','Lean / Six Sigma','PMP / Agile','Quality (ASQ)','Logistics'],
    consulting:['Strategy','PMP / Agile','Six Sigma','Change Management','Analytics'],
    engineering:['PE License','Project Management','Quality / Safety','Industry-Specific','Technical Skills'],
    default:['Project Management','Cloud & Infrastructure','Data & Analytics','Leadership','Industry-Specific'],
  },
  afields: {
    tech:['Computer Science','AI / ML','Systems Engineering','Cybersecurity'],
    data:['Machine Learning','Statistics','Data Mining','NLP / Computer Vision'],
    cybersec:['Information Security','Cryptography','Network Security','Digital Forensics'],
    finance:['Finance / Economics','Quantitative Methods','Fintech','Behavioral Economics'],
    marketing:['Consumer Behavior','Digital Media','Communications','Market Analytics'],
    healthcare:['Medical Research','Public Health','Biomedical','Health Informatics'],
    legal:['Law / Jurisprudence','Regulatory Policy','Criminology','International Law'],
    science:['Biology / Chemistry','Physics','Environmental Science','Materials Science'],
    engineering:['Mechanical / Civil','Electrical / Electronics','Chemical','Environmental'],
    education:['Pedagogy','Educational Technology','Curriculum Studies','Learning Sciences'],
    default:['Your Field','Interdisciplinary','Methods / Analytics','Industry Research'],
  },
  ctop: {
    tech:['Cloud Computing','System Design','DevOps','Algorithms & Data Structures'],
    data:['Machine Learning','Data Visualization','Deep Learning','Statistical Analysis'],
    cybersec:['Ethical Hacking','Network Security','Incident Response','Cloud Security'],
    finance:['Financial Modeling','Investment Analysis','Blockchain / DeFi','Python for Finance'],
    marketing:['Digital Marketing','Google Analytics','Content Marketing','Social Media Strategy'],
    healthcare:['Health Informatics','Clinical Research','Biostatistics','Public Health'],
    legal:['Legal Tech','Compliance Frameworks','Contract Drafting','Regulatory Analysis'],
    creative:['UX Design','Motion Graphics','Digital Illustration','Creative Writing'],
    operations:['Supply Chain Management','Lean / Six Sigma','Project Management','Business Analytics'],
    hr:['People Analytics','Organizational Design','Leadership Development','HR Technology'],
    default:['Project Management','Data Analysis','Business Strategy','Communication Skills'],
  },
  hskills: {
    tech:['Linux Administration','Docker / K8s','CI/CD Pipelines','Microservices'],
    data:['Python / Jupyter','SQL & Databases','TensorFlow / PyTorch','Data Pipelines'],
    cybersec:['CTF Challenges','Network Pentesting','SIEM Tools','Malware Analysis'],
    finance:['Excel / VBA','Bloomberg Terminal','Python for Finance','SQL for Analytics'],
    marketing:['Google Ads','A/B Testing','Analytics Dashboards','CRM Tools'],
    creative:['Figma / Sketch','Adobe Creative Suite','Video Editing','Prototyping'],
    engineering:['CAD / Simulation','MATLAB','Lab Equipment','Technical Writing'],
    science:['Lab Techniques','R / SPSS','Field Methods','Scientific Writing'],
    operations:['ERP Systems','Process Mapping','Automation Tools','Inventory Systems'],
    default:['Spreadsheets / Data','Presentation Skills','Project Tools','Industry Software'],
  },
  rfields: {
    tech:['AI / ML','Distributed Systems','Cloud Computing','Software Engineering'],
    data:['Deep Learning','NLP','Recommender Systems','Causal Inference'],
    cybersec:['Threat Detection','Cryptography','Network Security','Privacy'],
    finance:['Quantitative Finance','Behavioral Economics','Fintech','Risk Modeling'],
    healthcare:['Clinical Trials','Genomics','Public Health','Medical Devices'],
    science:['Your Discipline','Computational Methods','Environmental','Materials'],
    engineering:['Your Discipline','Simulation','Smart Systems','Sustainability'],
    default:['Your Field','Applied Research','Data-Driven Methods','Industry Applications'],
  },
};

let _currentDomain = 'default';

function getEffectivePills(qId, originalPills) {
  const override = DOMAIN_PILLS[qId]?.[_currentDomain];
  return override || DOMAIN_PILLS[qId]?.['default'] || originalPills;
}

function getAllPills(qId, originalPills) {
  // Merge domain pills + default pills + original pills (deduped, for "View all")
  const seen = new Set();
  const result = [];
  const domain = DOMAIN_PILLS[qId]?.[_currentDomain];
  const dflt = DOMAIN_PILLS[qId]?.['default'];
  for (const src of [domain, dflt, originalPills]) {
    if (!src) continue;
    for (const p of src) { if (!seen.has(p)) { seen.add(p); result.push(p); } }
  }
  return result;
}

function getEffectiveDefs(qId, originalDefs, type) {
  // For single-select, keep original default. For multi-select, clear defaults so user/AI chooses.
  if (type === 's') return originalDefs;
  const override = DOMAIN_PILLS[qId]?.[_currentDomain];
  if (override) return []; // When pills change, don't auto-select stale defaults
  return originalDefs;
}

// Deterministic question IDs — changing these should invalidate suggestion cache
const DETERMINISTIC_QS = new Set(['stage','opptype','itype','clevel','rtype','sdepth','cdepth','eregion']);
// Panels that share the 'stage' question — changing in one propagates to all
const STAGE_SHARED_PANELS = ['career_opps', 'jobhunt'];

/* ═══════════════════════════════════════
   STATE
   ═══════════════════════════════════════ */

let step = 0;
let selCats, selSubs, customSubs, interestTopics;
let panelSel, panelCustom, activeTab;
let rvItems, rvCollapsed, discoverAdded;
let _wizMode = null;       // 'suggest' | 'generate'
let _wizGenerateData = null; // response from /api/generate-profile (Stage 5)
let _tabSuggestCache = {};  // cache per tab: {tabId: {suggestions:[], loading:false}}
let _rvItemsCache = null;   // AI-generated role-aware items for Step 3: {sections:{}, discover:[]}
let _rvLoading = false;     // true while fetching rv items from backend
let _viewAllPills = new Set();      // Issue 4: qIds where "View all" is active
let _collapsedSections = new Set(); // Issue 6: collapsed section IDs in Step 2
let _s2CollapseAll = false;         // Issue 6: collapse-all toggle state
let _suggestDebounceTimer = null;   // Step 3: debounce timer for suggestion refresh
let _suggestAbortCtrl = null;       // Step 3: AbortController for in-flight suggest requests
let _s2BannerDismissed = false;     // Step 3: banner dismissed flag

const WIZ_STORAGE_KEY = 'stratos_wizard_state';

function _wizSaveState() {
  try {
    // Serialize Sets to arrays for JSON storage
    const serSubs = {};
    for (const [k, v] of Object.entries(selSubs)) serSubs[k] = [...v];
    const serPanelSel = {};
    for (const [sid, qs] of Object.entries(panelSel)) {
      serPanelSel[sid] = {};
      for (const [qid, set] of Object.entries(qs)) serPanelSel[sid][qid] = [...set];
    }
    localStorage.setItem(WIZ_STORAGE_KEY, JSON.stringify({
      selCats: [...selCats],
      selSubs: serSubs,
      customSubs,
      interestTopics,
      panelSel: serPanelSel,
      panelCustom,
      ts: Date.now()
    }));
  } catch(e) { /* storage full or unavailable */ }
}

function _wizLoadState() {
  try {
    const raw = localStorage.getItem(WIZ_STORAGE_KEY);
    if (!raw) return false;
    const saved = JSON.parse(raw);
    // Expire after 24 hours
    if (Date.now() - (saved.ts || 0) > 86400000) { localStorage.removeItem(WIZ_STORAGE_KEY); return false; }
    if (!saved.selCats?.length) return false;

    selCats = new Set(saved.selCats);
    for (const [k, v] of Object.entries(saved.selSubs || {})) {
      if (selSubs.hasOwnProperty(k)) selSubs[k] = new Set(v);
    }
    if (saved.customSubs) {
      for (const [k, v] of Object.entries(saved.customSubs)) {
        if (customSubs.hasOwnProperty(k)) customSubs[k] = v;
      }
    }
    if (saved.interestTopics) interestTopics = saved.interestTopics;
    if (saved.panelSel) {
      for (const [sid, qs] of Object.entries(saved.panelSel)) {
        if (!panelSel[sid]) continue;
        for (const [qid, arr] of Object.entries(qs)) {
          panelSel[sid][qid] = new Set(arr);
        }
      }
    }
    if (saved.panelCustom) {
      for (const [sid, qs] of Object.entries(saved.panelCustom)) {
        if (!panelCustom[sid]) continue;
        for (const [qid, arr] of Object.entries(qs)) {
          panelCustom[sid][qid] = arr;
        }
      }
    }
    return true;
  } catch(e) { return false; }
}

function _wizClearState() {
  try { localStorage.removeItem(WIZ_STORAGE_KEY); } catch(e) {}
}

/* ═══════════════════════════════════════
   PROFILE PRESETS
   ═══════════════════════════════════════ */

const PRESETS_STORAGE_KEY = 'stratos_wizard_presets';
let _presetMenuOpen = false;

function _getPresets() {
  try {
    const raw = localStorage.getItem(PRESETS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch(e) { return {}; }
}

function _setPresets(presets) {
  try {
    localStorage.setItem(PRESETS_STORAGE_KEY, JSON.stringify(presets));
  } catch(e) {
    if (typeof showToast === 'function') showToast('Storage full \u2014 delete some presets to save new ones', 'warning');
  }
}

function _serializeWizardState() {
  const serSubs = {};
  for (const [k, v] of Object.entries(selSubs)) serSubs[k] = [...v];
  const serPanelSel = {};
  for (const [sid, qs] of Object.entries(panelSel)) {
    serPanelSel[sid] = {};
    for (const [qid, set] of Object.entries(qs)) serPanelSel[sid][qid] = [...set];
  }
  return {
    selCats: [...selCats],
    selSubs: serSubs,
    customSubs: JSON.parse(JSON.stringify(customSubs)),
    interestTopics: [...interestTopics],
    panelSel: serPanelSel,
    panelCustom: JSON.parse(JSON.stringify(panelCustom)),
    rvItems: JSON.parse(JSON.stringify(rvItems || {})),
    rvItemsCache: _rvItemsCache ? JSON.parse(JSON.stringify(_rvItemsCache)) : null,
  };
}

function _loadPresetState(preset) {
  // Fully replace all wizard state from preset — no blending
  selCats = new Set(preset.selCats || []);
  for (const c of CATS) {
    selSubs[c.id] = new Set(preset.selSubs?.[c.id] || []);
    customSubs[c.id] = preset.customSubs?.[c.id] || [];
  }
  interestTopics = preset.interestTopics ? [...preset.interestTopics] : [];
  // Restore panelSel: re-init defaults first, then overlay preset
  for (const [sid, cfg] of Object.entries(PANELS)) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of cfg.qs) panelCustom[sid][q.id] = [];
  }
  if (preset.panelSel) {
    for (const [sid, qs] of Object.entries(preset.panelSel)) {
      if (!panelSel[sid]) panelSel[sid] = {};
      for (const [qid, arr] of Object.entries(qs)) panelSel[sid][qid] = new Set(arr);
    }
  }
  if (preset.panelCustom) {
    for (const [sid, qs] of Object.entries(preset.panelCustom)) {
      if (!panelCustom[sid]) panelCustom[sid] = {};
      for (const [qid, arr] of Object.entries(qs)) panelCustom[sid][qid] = arr;
    }
  }
  rvItems = preset.rvItems ? JSON.parse(JSON.stringify(preset.rvItems)) : {};
  _rvItemsCache = preset.rvItemsCache ? JSON.parse(JSON.stringify(preset.rvItemsCache)) : null;
  // Clear transient caches
  _tabSuggestCache = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  if (_suggestAbortCtrl) { _suggestAbortCtrl.abort(); _suggestAbortCtrl = null; }
  _s2BannerDismissed = false;
  _viewAllPills = new Set();
  _collapsedSections = new Set();
  _s2CollapseAll = false;
  rvCollapsed = new Set();
  discoverAdded = new Set();
  _rvLoading = false;
}

function savePreset() {
  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const suggested = role + (location ? ' \u00B7 ' + location : '');
  const name = prompt('Save profile as:', suggested);
  if (!name || !name.trim()) return;
  const key = name.trim();
  const presets = _getPresets();
  if (presets[key]) {
    if (!confirm(`"${key}" already exists. Overwrite?`)) return;
  }
  presets[key] = {
    name: key,
    role, location,
    savedAt: new Date().toISOString(),
    ..._serializeWizardState()
  };
  _setPresets(presets);
  renderPresetBar();
  if (typeof showToast === 'function') showToast(`Profile "${key}" saved`, 'success');
}

function loadPreset(name) {
  const presets = _getPresets();
  const preset = presets[name];
  if (!preset) return;
  _loadPresetState(preset);
  // Update role/location inputs
  if (preset.role) {
    const roleInput = document.getElementById('simple-role');
    if (roleInput) roleInput.value = preset.role;
    _currentDomain = classifyRole(preset.role);
  }
  if (preset.location) {
    const locInput = document.getElementById('simple-location');
    if (locInput) locInput.value = preset.location;
  }
  _presetMenuOpen = false;
  renderPresetBar();
  // Re-render with loaded state
  renderAll();
  _wizSaveState();
  if (typeof showToast === 'function') showToast(`Loaded "${name}"`, 'success');
}

function deletePreset(name, evt) {
  if (evt) { evt.stopPropagation(); evt.preventDefault(); }
  if (!confirm(`Delete "${name}"?`)) return;
  const presets = _getPresets();
  delete presets[name];
  _setPresets(presets);
  renderPresetBar();
}

function togglePresetMenu() {
  _presetMenuOpen = !_presetMenuOpen;
  renderPresetBar();
}

function renderPresetBar() {
  const el = document.getElementById('wiz-preset-dd');
  if (!el) return;
  const presets = _getPresets();
  const names = Object.keys(presets).sort((a, b) => {
    const ta = presets[a].savedAt || '';
    const tb = presets[b].savedAt || '';
    return tb.localeCompare(ta); // newest first
  });
  const hasPresets = names.length > 0;

  let menuHtml = '';
  if (hasPresets) {
    menuHtml = names.map(n => {
      const p = presets[n];
      const date = p.savedAt ? new Date(p.savedAt).toLocaleDateString(undefined, {month:'short', day:'numeric'}) : '';
      return `<div class="preset-item" onclick="_wiz.loadPreset('${escAttr(n)}')">
        <span class="preset-item-name">${esc(n)}</span>
        <span class="preset-item-date">${esc(date)}</span>
        <button class="preset-del" onclick="_wiz.deletePreset('${escAttr(n)}', event)" title="Delete">&times;</button>
      </div>`;
    }).join('');
  } else {
    menuHtml = '<div class="preset-empty">No saved profiles</div>';
  }

  el.innerHTML = `
    <button class="preset-save-btn-hdr" onclick="_wiz.savePreset()">Save Profile</button>
    <button class="preset-btn ${_presetMenuOpen ? 'open' : ''}" onclick="_wiz.togglePresetMenu()">
      <span class="preset-name">${hasPresets ? `${names.length} saved` : 'Profiles'}</span>
      <span class="preset-chev">\u25BC</span>
    </button>
    <div class="preset-menu ${_presetMenuOpen ? 'open' : ''}">${menuHtml}</div>`;
}

function initState() {
  // Classify role for adaptive pills
  const roleVal = document.getElementById('simple-role')?.value?.trim() || '';
  _currentDomain = classifyRole(roleVal);
  const inferredStage = inferStage(roleVal);

  selCats = new Set();
  selSubs = {}; customSubs = {};
  for (const c of CATS) { selSubs[c.id] = new Set(); customSubs[c.id] = []; }
  interestTopics = [];
  panelSel = {}; panelCustom = {};
  for (const [sid, cfg] of Object.entries(PANELS)) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of cfg.qs) {
      panelCustom[sid][q.id] = [];
      if (q.type === 's' && q.id === 'stage') {
        panelSel[sid][q.id] = new Set([inferredStage]);
      } else if (q.type === 's' && q.def) {
        panelSel[sid][q.id] = new Set([q.def]);
      } else {
        const defs = getEffectiveDefs(q.id, q.defs || [], q.type);
        panelSel[sid][q.id] = new Set(defs);
      }
    }
  }
  activeTab = null;
  rvItems = {}; rvCollapsed = new Set();
  discoverAdded = new Set();
  _tabSuggestCache = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  if (_suggestAbortCtrl) { _suggestAbortCtrl.abort(); _suggestAbortCtrl = null; }
  _s2BannerDismissed = false;
  _rvItemsCache = null;
  _rvLoading = false;
  _viewAllPills = new Set();
  _collapsedSections = new Set();
  _s2CollapseAll = false;
}

/* ═══════════════════════════════════════
   CSS INJECTION  (Stage 1: Theme Integration)
   Maps wizard colors → dashboard CSS custom properties.
   All selectors scoped under .wiz-scope.
   ═══════════════════════════════════════ */


const WIZ_CSS = `
/* === Variables: inherit from dashboard theme, fallback to defaults === */
.wiz-scope {
  --bg: var(--bg-primary, #080d18);
  --bg-primary: var(--bg-primary, #0c1222);
  --bg-card: var(--bg-panel-solid, #131b2e);
  --bg-card-hover: var(--bg-hover, #1a2640);
  --accent2: var(--accent-light, #38bdf8);
  --accent-dim: var(--accent-bg, rgba(52,211,153,0.1));
  --accent-glow: var(--accent-border, rgba(52,211,153,0.2));
  --text: var(--text-primary, #f8fafc);
  --text2: var(--text-secondary, #94a3b8);
  --text3: var(--text-muted, #64748b);
  --brd: var(--border-color, rgba(255,255,255,0.06));
  font-family: system-ui, -apple-system, sans-serif;
  color: var(--text);
}
.wiz-scope *, .wiz-scope *::before, .wiz-scope *::after { box-sizing:border-box;margin:0;padding:0; }
/* ── Atmosphere: Deep ── */
.wiz-scope[data-wiz-atmos="deep"] { --bg:#000000;--bg-primary:#020206;--bg-card:rgba(255,255,255,0.03);--bg-card-hover:rgba(255,255,255,0.06);--brd:rgba(255,255,255,0.04); }
.wiz-scope[data-wiz-atmos="deep"] .modal { background:#020206;box-shadow:0 32px 80px rgba(0,0,0,0.9),0 0 0 1px rgba(255,255,255,0.04); }
.wiz-scope[data-wiz-atmos="deep"] .grad-bar { height:2px;opacity:.8; }
.wiz-scope[data-wiz-atmos="deep"] .rail { background:rgba(0,0,0,0.5);backdrop-filter:blur(20px); }
.wiz-scope[data-wiz-atmos="deep"] .p-card { border-color:rgba(255,255,255,0.03); }
.wiz-scope[data-wiz-atmos="deep"] .p-card.sel { box-shadow:0 0 0 1px var(--accent-glow),0 8px 40px rgba(0,0,0,0.5); }
.wiz-scope[data-wiz-atmos="deep"] .accordion { border-color:rgba(255,255,255,0.03); }
.wiz-scope[data-wiz-atmos="deep"] .build-btn { box-shadow:0 8px 40px var(--accent-glow),0 0 0 1px var(--accent-glow); }

/* ── Atmosphere: Arcane ── */
.wiz-scope[data-wiz-atmos="arcane"] { --bg:#050510;--bg-primary:#08081a;--bg-card:rgba(255,255,255,0.035);--bg-card-hover:rgba(255,255,255,0.06);--brd:rgba(255,255,255,0.05); }
.wiz-scope[data-wiz-atmos="arcane"] .modal { background:#08081a;box-shadow:0 32px 80px rgba(0,0,0,0.8),0 0 0 1px var(--accent-dim);overflow:hidden; }
.wiz-scope[data-wiz-atmos="arcane"] .wiz-stars-canvas { display:block; }
.wiz-scope[data-wiz-atmos="arcane"] .grad-bar { height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent-light),var(--accent));background-size:300% 100%;animation:wizBarShift 6s ease infinite; }
.wiz-scope[data-wiz-atmos="arcane"] .rail { background:rgba(8,8,26,0.7);backdrop-filter:blur(16px); }
.wiz-scope[data-wiz-atmos="arcane"] .p-card { backdrop-filter:blur(8px);background:rgba(255,255,255,0.03);border-color:rgba(255,255,255,0.05); }
.wiz-scope[data-wiz-atmos="arcane"] .p-card:hover { box-shadow:0 20px 50px rgba(0,0,0,0.4),0 0 20px var(--accent-dim); }
.wiz-scope[data-wiz-atmos="arcane"] .p-card.sel { box-shadow:0 0 0 1px var(--accent-glow),0 0 30px var(--accent-dim),0 8px 40px rgba(0,0,0,0.4); }
.wiz-scope[data-wiz-atmos="arcane"] .accordion { backdrop-filter:blur(8px);background:rgba(255,255,255,0.025); }
.wiz-scope[data-wiz-atmos="arcane"] .build-btn { box-shadow:0 8px 40px var(--accent-glow),0 0 20px var(--accent-dim),0 0 0 1px var(--accent-glow); }
.wiz-scope[data-wiz-atmos="arcane"] .hdr-logo { filter:drop-shadow(0 0 8px var(--accent-glow)); }
.wiz-scope[data-wiz-atmos="arcane"] .ring-fg { filter:drop-shadow(0 0 4px var(--accent-glow)); }

/* ── Backdrop ── */
.wiz-scope .backdrop { position:fixed;inset:0;background:rgba(0,0,0,0.65);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);z-index:9998;opacity:0;transition:opacity .35s;pointer-events:none; }
.wiz-scope .backdrop.open { opacity:1;pointer-events:auto; }

/* ── Modal ── */
.wiz-scope .modal { position:fixed;inset:2vh 3vw;background:var(--bg-primary);border-radius:18px;z-index:9999;display:flex;flex-direction:column;overflow:clip;box-shadow:0 32px 80px rgba(0,0,0,0.6),0 0 0 1px var(--brd);opacity:0;transform:scale(.94) translateY(16px);transition:opacity .4s cubic-bezier(.22,1,.36,1),transform .4s cubic-bezier(.22,1,.36,1);pointer-events:none; }
.wiz-scope .modal.open { opacity:1;transform:none;pointer-events:auto; }
.wiz-scope .modal::after { content:'';position:absolute;inset:0;opacity:0.03;pointer-events:none;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='.8'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");z-index:0; }
.wiz-scope .modal > *:not(.wiz-stars-canvas) { position:relative;z-index:1; }

/* ── Top Gradient Bar ── */
.wiz-scope .grad-bar { height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent));background-size:200% 100%;animation:wizBarShift 4s ease infinite;flex-shrink:0; }
@keyframes wizBarShift { 0%,100% { background-position:0% center; } 50% { background-position:100% center; } }

/* ── Header ── */
.wiz-scope .hdr { display:flex;align-items:center;padding:16px 28px;border-bottom:1px solid var(--brd);flex-shrink:0;gap:14px; }
.wiz-scope .hdr-logo { font-size:22px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.5px; }
.wiz-scope .hdr-sub { font-size:13px;color:var(--text3);font-weight:500;margin-left:4px; }
.wiz-scope .hdr-badges { display:flex;gap:8px;margin-left:8px;align-items:center; }
.wiz-scope .badge { font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;padding:4px 10px;border-radius:6px;background:var(--accent-dim);color:var(--accent);border:1px solid var(--accent-glow); }
.wiz-scope .badge.blue { background:rgba(56,189,248,0.1);color:var(--accent2);border-color:rgba(56,189,248,0.15); }

/* ── Atmosphere Selector ── */
.wiz-scope .atmos-selector { display:flex;align-items:center;gap:2px;margin-left:auto;background:rgba(255,255,255,0.03);border:1px solid var(--brd);border-radius:10px;padding:3px; }
.wiz-scope .atmos-btn { padding:5px 12px;border-radius:8px;border:none;background:transparent;color:var(--text3);font-size:11px;font-weight:600;cursor:pointer;transition:all 0.2s;white-space:nowrap;letter-spacing:0.3px; }
.wiz-scope .atmos-btn:hover { color:var(--text);background:rgba(255,255,255,0.04); }
.wiz-scope .atmos-btn.active { color:var(--accent);background:var(--accent-dim);box-shadow:0 0 8px var(--accent-dim); }

/* Star canvas for Arcane atmosphere */
.wiz-scope .wiz-stars-canvas { position:absolute;inset:0;z-index:0;pointer-events:none;display:none; }

/* ── Progress Ring ── */
.wiz-scope .ring-wrap { position:relative;width:40px;height:40px;margin:0 4px;flex-shrink:0; }
.wiz-scope .ring-wrap svg { transform:rotate(-90deg); }
.wiz-scope .ring-bg { fill:none;stroke:var(--brd);stroke-width:3.5; }
.wiz-scope .ring-fg { fill:none;stroke:url(#wizRingGrad);stroke-width:3.5;stroke-linecap:round;transition:stroke-dashoffset .5s cubic-bezier(.4,0,.2,1); }
.wiz-scope .ring-label { position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }

/* ── Close button ── */
.wiz-scope .close-btn { width:36px;height:36px;border-radius:10px;border:1px solid var(--brd);background:transparent;color:var(--text3);font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s; }
.wiz-scope .close-btn:hover { background:rgba(255,255,255,0.05);color:var(--text); }

/* ── Body Layout ── */
.wiz-scope .body { display:flex;flex:1;min-height:0; }

/* ── Left Rail ── */
.wiz-scope .rail { width:290px;min-width:290px;background:color-mix(in srgb,var(--bg-card) 60%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-right:1px solid var(--brd);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0; }
.wiz-scope .rail::-webkit-scrollbar { width:5px; }
.wiz-scope .rail::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.06);border-radius:10px; }
.wiz-scope .rail-inner { padding:18px 16px;flex:1;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.06) transparent; }
.wiz-scope .rail-inner::-webkit-scrollbar { width:5px; }
.wiz-scope .rail-inner::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.06);border-radius:10px; }

/* ── Rail sections ── */
.wiz-scope .rail-section { margin-bottom:20px; }
.wiz-scope .rail-cat-hdr { display:flex;align-items:center;gap:9px;padding:10px 14px;margin-bottom:6px;font-size:13px;font-weight:700;color:var(--text);cursor:pointer;border-radius:10px;transition:background 0.15s; }
.wiz-scope .rail-cat-hdr:hover { background:rgba(255,255,255,0.03); }
.wiz-scope .rail-cat-hdr.cat-career { background:rgba(52,211,153,0.06); }
.wiz-scope .rail-cat-hdr.cat-industry { background:rgba(56,189,248,0.06); }
.wiz-scope .rail-cat-hdr.cat-learning { background:rgba(168,85,247,0.06); }
.wiz-scope .rail-cat-hdr.cat-markets { background:rgba(245,158,11,0.06); }
.wiz-scope .rail-cat-hdr.cat-deals { background:rgba(236,72,153,0.06); }
.wiz-scope .rail-cat-hdr.cat-interests { background:rgba(249,115,22,0.06); }
.wiz-scope .rail-cat-icon { width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:8px;font-size:14px; }
.wiz-scope .rail-cat-hdr.cat-career .rail-cat-icon { background:rgba(52,211,153,0.15); }
.wiz-scope .rail-cat-hdr.cat-industry .rail-cat-icon { background:rgba(56,189,248,0.15); }
.wiz-scope .rail-cat-hdr.cat-learning .rail-cat-icon { background:rgba(168,85,247,0.15); }
.wiz-scope .rail-cat-hdr.cat-markets .rail-cat-icon { background:rgba(245,158,11,0.15); }
.wiz-scope .rail-cat-hdr.cat-deals .rail-cat-icon { background:rgba(236,72,153,0.15); }
.wiz-scope .rail-cat-hdr.cat-interests .rail-cat-icon { background:rgba(249,115,22,0.15); }
.wiz-scope .rail-cat-count { margin-left:auto;font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0c1222; }
.wiz-scope .rail-sec-chev { margin-left:4px;font-size:10px;color:var(--text3);transition:transform .2s; }
.wiz-scope .rail-section.collapsed .rail-sec-chev { transform:rotate(-90deg); }
.wiz-scope .rail-section.collapsed .rail-sec-body { display:none; }
.wiz-scope .rail-sec-body { padding:0 0 4px; }

/* ── Rail items (tree style) ── */
.wiz-scope .rail-items { display:flex;flex-direction:column;gap:2px;margin-left:16px;padding-left:14px;border-left:1px solid var(--brd); }
.wiz-scope .rail-item { display:flex;align-items:center;gap:7px;padding:5px 10px;font-size:12px;color:var(--text2);border-radius:6px;cursor:default;transition:all 0.15s; }
.wiz-scope .rail-item:hover { background:rgba(255,255,255,0.03);color:var(--text); }
.wiz-scope .rail-item .bullet { width:4px;height:4px;border-radius:50%;background:var(--text3);flex-shrink:0; }
.wiz-scope .rail-item .rp-x { cursor:pointer;opacity:.5;font-size:13px;margin-left:auto; }
.wiz-scope .rail-item .rp-x:hover { opacity:1; }
.wiz-scope .rail-empty { font-size:12px;color:var(--text3);font-style:italic;padding:4px 0 4px 30px; }

/* ── Rail AI Suggestions ── */
.wiz-scope .rail-ai-hdr { display:flex;align-items:center;gap:6px;padding:8px 0 4px 16px;margin-left:16px;padding-left:14px;font-size:11px;font-weight:600;color:var(--accent);letter-spacing:0.3px; }
.wiz-scope .rail-suggestions { display:flex;flex-wrap:wrap;gap:5px;margin-left:16px;padding-left:14px;border-left:1px solid var(--brd);padding-bottom:4px; }
.wiz-scope .sug-pill { font-size:11px;padding:4px 10px;border-radius:20px;border:1px dashed var(--accent-glow);background:transparent;color:var(--accent-light);cursor:pointer;transition:all 0.2s;white-space:nowrap; }
.wiz-scope .sug-pill:hover { border-color:var(--accent);background:var(--accent-dim);box-shadow:0 0 12px var(--accent-dim); }
.wiz-scope .sug-pill.added { opacity:.55;border-style:solid;cursor:pointer; }
.wiz-scope .sug-pill.added:hover { opacity:.8;background:rgba(255,60,60,0.06); }

/* ── Discover pills ── */
.wiz-scope .rail-discover-hdr { display:flex;align-items:center;gap:6px;padding:6px 0 4px 16px;margin-left:16px;padding-left:14px;font-size:11px;font-weight:600;color:var(--accent2);letter-spacing:0.3px; }
.wiz-scope .disc-pill { font-size:11px;padding:4px 10px;border-radius:20px;border:1px dashed rgba(56,189,248,0.25);background:transparent;color:#7dd3fc;cursor:pointer;transition:all 0.2s;white-space:nowrap; }
.wiz-scope .disc-pill:hover { background:rgba(56,189,248,0.08);border-color:rgba(56,189,248,0.5); }
.wiz-scope .disc-pill.added { opacity:.35;pointer-events:none;border-style:solid; }
.wiz-scope .disc-pill .disc-tag { font-size:10px;opacity:.6;margin-left:2px; }

/* ── Rail divider ── */
.wiz-scope .rail-divider { height:1px;margin:6px 16px;background:var(--brd); }

/* ── Feed Summary ── */
.wiz-scope .feed-summary { padding:14px 16px;margin:0 12px;background:rgba(255,255,255,0.02);border:1px solid var(--brd);border-radius:12px; }
.wiz-scope .feed-summary-title { font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:var(--text3);margin-bottom:10px; }
.wiz-scope .feed-stat { display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.03); }
.wiz-scope .feed-stat:last-child { border-bottom:none; }
.wiz-scope .feed-stat-label { font-size:12px;color:var(--text2); }
.wiz-scope .feed-stat-val { font-size:12px;font-weight:700;color:var(--text); }
.wiz-scope .feed-stat-val.accent { background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }

/* ── Rail Bottom ── */
.wiz-scope .rail-bottom { margin-top:auto;padding:14px 16px 18px;border-top:1px solid var(--brd);display:flex;flex-direction:column;gap:12px;flex-shrink:0; }

/* ── Depth Toggle ── */
.wiz-scope .depth-toggle { display:flex;background:rgba(255,255,255,0.04);border-radius:12px;padding:4px;border:1px solid var(--brd);cursor:pointer; }
.wiz-scope .depth-opt { flex:1;text-align:center;padding:7px;font-size:12px;font-weight:700;color:var(--text3);border-radius:9px;transition:all 0.25s;z-index:1;display:flex;align-items:center;justify-content:center;gap:5px; }
.wiz-scope .depth-opt.active { color:var(--text);background:var(--accent-dim);box-shadow:0 2px 8px var(--accent-dim); }

/* ── Build Button ── */
.wiz-scope .build-btn { position:relative;width:100%;padding:14px 24px;border:none;border-radius:14px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0c1222;font-size:15px;font-weight:800;letter-spacing:0.3px;cursor:pointer;overflow:hidden;box-shadow:0 8px 32px var(--accent-glow),0 0 0 1px var(--accent-glow);transition:all 0.25s; }
.wiz-scope .build-btn:hover:not(:disabled) { transform:translateY(-2px);box-shadow:0 12px 40px var(--accent-glow),0 0 0 1px var(--accent-glow); }
.wiz-scope .build-btn:active:not(:disabled) { transform:translateY(0) scale(.98); }
.wiz-scope .build-btn:disabled { opacity:.3;cursor:not-allowed;transform:none;box-shadow:none; }
.wiz-scope .build-btn::after { content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);background-size:200% 100%;animation:wizShimmer2 2.5s infinite;border-radius:inherit; }
.wiz-scope .build-btn:disabled::after { animation:none; }
@keyframes wizShimmer2 { 0% { background-position:-200% center; } 100% { background-position:200% center; } }

/* ── Main Panel ── */
.wiz-scope .main { flex:1;overflow-y:auto;padding:28px 36px 40px;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.08) transparent; }
.wiz-scope .main::-webkit-scrollbar { width:5px; }
.wiz-scope .main::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.08);border-radius:4px; }

/* ── Step Progress Bar ── */
.wiz-scope .steps-bar { display:flex!important;align-items:center!important;margin-bottom:32px;height:auto!important;max-height:50px!important;line-height:1!important; }
.wiz-scope .step-num-circle { width:34px!important;height:34px!important;min-width:34px!important;max-width:34px!important;min-height:34px!important;max-height:34px!important;display:flex!important;align-items:center!important;justify-content:center!important;border-radius:50%!important;font-size:13px!important;font-weight:700;border:2px solid var(--brd)!important;color:var(--text3);transition:all 0.2s;flex-shrink:0;line-height:1!important;padding:0!important; }
.wiz-scope .step-num-circle.done { border-color:var(--accent)!important;background:var(--accent)!important;color:#0c1222!important; }
.wiz-scope .step-num-circle.active { border-color:var(--accent)!important;color:var(--accent)!important;background:var(--accent-dim)!important; }
.wiz-scope .step-bar-label { font-size:12px!important;font-weight:600;color:var(--text3);margin-left:8px;white-space:nowrap;line-height:1!important; }
.wiz-scope .step-bar-label.active { color:var(--text); }
.wiz-scope .step-line { flex:1;height:2px!important;max-height:2px!important;min-height:2px!important;background:var(--brd);margin:0 14px;border:none!important;padding:0!important; }
.wiz-scope .step-line.done { background:linear-gradient(90deg,var(--accent),var(--accent2))!important; }

/* ── Section Headers ── */
.wiz-scope .sec-title { font-size:22px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px; }
.wiz-scope .sec-sub { font-size:14px;color:var(--text3);margin-bottom:24px;font-weight:500; }
.wiz-scope .sec-title2 { font-size:20px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px; }
.wiz-scope .sec-sub2 { font-size:13px;color:var(--text3);margin-bottom:20px; }
.wiz-scope .section-divider { height:1px;background:linear-gradient(90deg,transparent,var(--brd),var(--brd),transparent);margin:8px 0 32px; }

/* ── Priority Cards Grid ── */
.wiz-scope .cards-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:40px; }
.wiz-scope .p-card { position:relative;background:var(--bg-card);border:2px solid transparent;border-radius:16px;padding:28px;cursor:pointer;transition:all 0.3s cubic-bezier(0.22,1,0.36,1);overflow:hidden;animation:wizCardEntry 0.5s cubic-bezier(0.22,1,0.36,1) both; }
.wiz-scope .p-card:nth-child(1) { animation-delay:0.05s; }
.wiz-scope .p-card:nth-child(2) { animation-delay:0.1s; }
.wiz-scope .p-card:nth-child(3) { animation-delay:0.15s; }
.wiz-scope .p-card:nth-child(4) { animation-delay:0.2s; }
.wiz-scope .p-card:nth-child(5) { animation-delay:0.25s; }
.wiz-scope .p-card:nth-child(6) { animation-delay:0.3s; }
@keyframes wizCardEntry { from { opacity:0;transform:translateY(20px) scale(0.96); } to { opacity:1;transform:translateY(0) scale(1); } }
.wiz-scope .p-card:hover { background:var(--bg-card-hover);transform:translateY(-6px);box-shadow:0 20px 40px rgba(0,0,0,0.3); }
.wiz-scope .p-card:active { transform:translateY(-1px) scale(.98);transition-duration:.1s; }
.wiz-scope .p-card.sel { border-color:var(--accent);box-shadow:0 0 0 1px var(--accent-glow),0 8px 32px var(--accent-dim); }
.wiz-scope .p-card.sel::before { content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--accent-dim),rgba(255,255,255,0.02));border-radius:inherit;pointer-events:none; }

/* ── Card icon ── */
.wiz-scope .card-icon { width:52px;height:52px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:16px;position:relative; }
.wiz-scope .card-icon.ci-career { background:rgba(52,211,153,0.15); }
.wiz-scope .card-icon.ci-industry { background:rgba(56,189,248,0.15); }
.wiz-scope .card-icon.ci-learning { background:rgba(168,85,247,0.15); }
.wiz-scope .card-icon.ci-markets { background:rgba(245,158,11,0.15); }
.wiz-scope .card-icon.ci-deals { background:rgba(236,72,153,0.15); }
.wiz-scope .card-icon.ci-interests { background:rgba(249,115,22,0.15); }
.wiz-scope .card-name { font-size:16px;font-weight:700;color:var(--text);margin-bottom:6px; }
.wiz-scope .card-desc { font-size:13px;color:var(--text3);line-height:1.5; }

/* ── Card checkmark ── */
.wiz-scope .card-check { position:absolute;top:14px;right:14px;width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;opacity:0;transform:scale(0);transition:all 0.35s cubic-bezier(0.34,1.56,0.64,1); }
.wiz-scope .p-card.sel .card-check { opacity:1;transform:scale(1); }
.wiz-scope .card-check svg { width:14px;height:14px;stroke:#0c1222;stroke-width:3;fill:none; }

/* ── Card sub-pills (expand on select) ── */
.wiz-scope .card-pills { display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;max-height:0;overflow:hidden;transition:max-height 0.4s ease; }
.wiz-scope .p-card.sel .card-pills { max-height:200px; }
.wiz-scope .card-pill { font-size:11px;padding:4px 10px;border-radius:20px;background:var(--accent-dim);color:var(--accent-light);border:1px solid var(--accent-glow);font-weight:600; }
.wiz-scope .card-pill-x { margin-left:4px;cursor:pointer;opacity:.6; }
.wiz-scope .card-pill-x:hover { opacity:1; }
.wiz-scope .card-pill.sp-add { border-style:dashed;background:transparent;cursor:pointer; }
.wiz-scope .card-pill-hint { font-size:13px;color:var(--text3);font-style:italic;padding:4px 0; }
.wiz-scope .add-inp-card { border:1px solid var(--accent-glow);background:var(--bg-card);color:var(--text);padding:5px 10px;border-radius:20px;font-size:13px;outline:none;width:120px; }
.wiz-scope .card-tap { position:absolute;bottom:10px;right:14px;font-size:11px;color:var(--accent-light);opacity:.5;pointer-events:none;transition:opacity .2s; }
.wiz-scope .p-card:hover .card-tap { opacity:.8; }

/* ── Accordions ── */
.wiz-scope .accordion { margin-bottom:12px;border-radius:14px;border:1px solid var(--brd);overflow:hidden;background:var(--bg-card);transition:border-color 0.2s; }
.wiz-scope .accordion.open { border-color:var(--accent-glow); }
.wiz-scope .acc-header { display:flex;align-items:center;gap:12px;padding:16px 24px;cursor:pointer;transition:background 0.15s;user-select:none; }
.wiz-scope .acc-header:hover { background:rgba(255,255,255,0.02); }
.wiz-scope .acc-icon { width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px; }
.wiz-scope .acc-icon.ai-career { background:rgba(52,211,153,0.12); }
.wiz-scope .acc-icon.ai-industry { background:rgba(56,189,248,0.12); }
.wiz-scope .acc-icon.ai-learning { background:rgba(168,85,247,0.12); }
.wiz-scope .acc-icon.ai-markets { background:rgba(245,158,11,0.12); }
.wiz-scope .acc-icon.ai-deals { background:rgba(236,72,153,0.12); }
.wiz-scope .acc-icon.ai-interests { background:rgba(249,115,22,0.12); }
.wiz-scope .acc-title { font-size:15px;font-weight:700;color:var(--text); }
.wiz-scope .acc-tag { font-size:10px;padding:2px 8px;border-radius:10px;background:var(--accent-dim);color:var(--accent);margin-left:auto; }
.wiz-scope .acc-chevron { color:var(--text3);transition:transform 0.3s;font-size:13px;margin-left:8px; }
.wiz-scope .accordion.open .acc-chevron { transform:rotate(180deg); }
.wiz-scope .acc-body { max-height:0;overflow:hidden;transition:max-height 0.4s cubic-bezier(0.22,1,0.36,1); }
.wiz-scope .accordion.open .acc-body { max-height:800px; }
.wiz-scope .acc-content { padding:0 24px 24px;display:flex;flex-direction:column;gap:18px; }

/* ── Accordion Pills ── */
.wiz-scope .acc-group-label { font-size:12px;font-weight:600;color:var(--text2);margin-bottom:8px; }
.wiz-scope .acc-pills { display:flex;flex-wrap:wrap;gap:8px; }
.wiz-scope .acc-pill { font-size:13px;font-weight:600;padding:7px 16px;border-radius:24px;background:rgba(255,255,255,0.04);border:1.5px solid var(--brd);color:var(--text2);cursor:pointer;transition:all 0.2s;position:relative;user-select:none; }
.wiz-scope .acc-pill:hover { background:rgba(255,255,255,0.07);border-color:rgba(255,255,255,0.12);color:var(--text); }
.wiz-scope .acc-pill:active { transform:scale(.95); }
.wiz-scope .acc-pill.on { background:var(--accent-dim);border-color:var(--accent);color:var(--accent-light);animation:wizPillPop 0.3s cubic-bezier(0.34,1.56,0.64,1); }
@keyframes wizPillPop { 0% { transform:scale(0.85); } 60% { transform:scale(1.05); } 100% { transform:scale(1); } }
.wiz-scope .acc-pill.on::after { content:'\\2713';margin-left:6px;font-size:11px; }
.wiz-scope .acc-pill.pill-decisive { border-width:2px;border-color:var(--accent-glow); }
.wiz-scope .acc-pill.pill-decisive.on { box-shadow:0 0 12px var(--accent-dim);border-color:var(--accent); }
.wiz-scope .acc-pill.pill-sug { border-style:dashed;color:var(--accent-light);border-color:var(--accent-glow); }
.wiz-scope .acc-pill .pill-x { margin-left:4px;cursor:pointer;opacity:.6; }
.wiz-scope .acc-pill .pill-x:hover { opacity:1; }
.wiz-scope .add-pill { font-size:13px;font-weight:600;padding:7px 16px;border-radius:24px;background:transparent;border:1.5px dashed var(--accent-glow);color:var(--accent);cursor:pointer;transition:all 0.2s; }
.wiz-scope .add-pill:hover { background:var(--accent-dim);border-color:var(--accent); }
.wiz-scope .add-inp { border:1px solid var(--accent-glow);background:var(--bg-card);color:var(--text);padding:6px 12px;border-radius:24px;font-size:13px;outline:none;width:140px; }
.wiz-scope .s2-hint { font-weight:400;color:var(--text3);font-size:12px; }

/* ── View all toggle ── */
.wiz-scope .va-tog { background:none;border:none;color:var(--accent-light);font-size:12px;cursor:pointer;padding:4px 0;margin-top:4px; }
.wiz-scope .va-tog:hover { text-decoration:underline; }

/* ── Interests section ── */
.wiz-scope .int-wrap { margin-bottom:16px; }
.wiz-scope .int-inp { width:100%;padding:10px 16px;border-radius:14px;border:1px solid var(--brd);background:var(--bg-card);color:var(--text);font-size:14px;outline:none;transition:border-color 0.2s; }
.wiz-scope .int-inp:focus { border-color:var(--accent); }

/* ── AI Suggestions (in detail panels) ── */
.wiz-scope .ai-sug { margin-top:20px;padding:16px;border-radius:16px;background:color-mix(in srgb,var(--accent-dim) 30%,var(--bg-card) 70%);border:1px solid var(--accent-glow);box-shadow:0 4px 16px color-mix(in srgb,var(--accent) 5%,transparent); }
.wiz-scope .ai-sug-hdr { font-size:13px;font-weight:700;color:var(--accent-light);margin-bottom:10px;display:flex;align-items:center;gap:8px; }
.wiz-scope .ai-sug-hdr.flash { animation:wizFlash .4s; }
.wiz-scope .ai-sug-spin { width:14px;height:14px;border:2px solid var(--accent-glow);border-top-color:var(--accent);border-radius:50%;animation:wizSpin .8s linear infinite; }
.wiz-scope .ai-sug-pill { display:inline-flex;align-items:center;padding:5px 12px;border-radius:16px;font-size:12px;cursor:pointer;border:1px dashed var(--accent-glow);color:var(--accent-light);margin:3px;transition:all 0.2s; }
.wiz-scope .ai-sug-pill:hover { background:var(--accent-dim);border-style:solid; }
.wiz-scope .ai-sug-pill.added { opacity:.55;border-style:solid;border-color:var(--accent-glow);cursor:pointer; }
.wiz-scope .ai-sug-pill.added:hover { opacity:.8;background:rgba(255,60,60,0.06); }
.wiz-scope .ai-sug-pill .sug-x { margin-left:5px;font-size:11px;opacity:.5;cursor:pointer; }
.wiz-scope .ai-sug-pill .sug-x:hover { opacity:1;color:#ff6b6b; }
.wiz-scope .ai-sug-pill.shimmer { width:80px;height:28px;background:linear-gradient(90deg,var(--bg-card) 25%,var(--accent-dim) 50%,var(--bg-card) 75%);background-size:200% 100%;animation:wizShimmer 1.5s infinite;border:none;border-radius:16px; }
.wiz-scope .ai-ref { background:none;border:none;color:var(--accent-light);cursor:pointer;font-size:16px;padding:2px 4px;border-radius:6px;transition:background 0.2s; }
.wiz-scope .ai-ref:hover { background:var(--accent-dim); }
.wiz-scope .ai-ref.spin { animation:wizSpin .8s linear infinite; }

/* ── S2 Banner ── */
.wiz-scope .s2-banner { display:flex;align-items:center;gap:10px;padding:10px 16px;border-radius:14px;background:var(--accent-dim);border:1px solid var(--accent-glow);margin-bottom:20px;font-size:13px;color:var(--text2);line-height:1.4; }
.wiz-scope .s2-banner-icon { font-size:18px;flex-shrink:0; }
.wiz-scope .s2-banner-x { cursor:pointer;margin-left:auto;opacity:.6;font-size:18px;flex-shrink:0; }
.wiz-scope .s2-banner-x:hover { opacity:1; }

/* ── Quick Setup ── */
.wiz-scope .quick-setup-wrap { display:flex;justify-content:center;margin-top:8px; }
.wiz-scope .quick-setup-btn { display:flex;flex-direction:column;align-items:center;gap:4px;padding:14px 32px;border-radius:14px;border:1.5px solid var(--accent-glow);background:linear-gradient(135deg,var(--accent-dim),color-mix(in srgb,var(--accent-dim) 40%,var(--bg-card) 60%));color:var(--accent-light);font-size:15px;font-weight:700;cursor:pointer;transition:all 0.25s;letter-spacing:0.3px; }
.wiz-scope .quick-setup-btn:hover { background:linear-gradient(135deg,var(--accent),var(--accent-light));color:#fff;border-color:var(--accent);transform:translateY(-2px);box-shadow:0 8px 28px var(--accent-dim); }
.wiz-scope .quick-setup-btn:active { transform:translateY(0) scale(.97); }
.wiz-scope .quick-setup-sub { font-size:11px;font-weight:400;opacity:.7; }

/* ── Presets ── */
.wiz-scope .preset-dd { position:relative;display:inline-flex;align-items:center;gap:6px; }
.wiz-scope .preset-btn { padding:5px 12px;border-radius:8px;border:1px solid var(--brd);background:rgba(255,255,255,0.03);color:var(--text2);font-size:12px;font-weight:600;cursor:pointer;transition:all 0.2s;display:flex;align-items:center;gap:6px;white-space:nowrap; }
.wiz-scope .preset-btn:hover { background:rgba(255,255,255,0.06);color:var(--text); }
.wiz-scope .preset-btn.open { background:rgba(255,255,255,0.06);color:var(--text); }
.wiz-scope .preset-chev { font-size:9px;transition:transform 0.2s; }
.wiz-scope .preset-btn.open .preset-chev { transform:rotate(180deg); }
.wiz-scope .preset-save-btn-hdr { padding:5px 12px;border-radius:8px;border:1px solid var(--accent-glow);background:var(--accent-dim);color:var(--accent-light);font-size:12px;font-weight:600;cursor:pointer;transition:all 0.2s;white-space:nowrap; }
.wiz-scope .preset-save-btn-hdr:hover { background:var(--accent-glow);color:var(--text); }
.wiz-scope .preset-item-name { flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
.wiz-scope .preset-item-date { font-size:10px;color:var(--text3); }
.wiz-scope .preset-del { background:none;border:none;color:var(--text3);font-size:14px;cursor:pointer;padding:0 2px;opacity:0;transition:opacity 0.15s; }
.wiz-scope .preset-item:hover .preset-del { opacity:1; }
.wiz-scope .preset-del:hover { color:#ff6b6b; }
.wiz-scope .preset-empty { padding:12px 16px;font-size:12px;color:var(--text3);text-align:center;font-style:italic; }
.wiz-scope .preset-dd .preset-menu { position:absolute;top:calc(100% + 6px);right:0;min-width:220px;background:var(--bg-card);border:1px solid var(--brd);border-radius:14px;box-shadow:0 8px 32px rgba(0,0,0,.25);z-index:10;padding:8px 0;display:none; }
.wiz-scope .preset-dd .preset-menu.open { display:block; }
.wiz-scope .preset-item { display:flex;align-items:center;gap:8px;padding:8px 16px;font-size:13px;color:var(--text);cursor:pointer;transition:background .15s; }
.wiz-scope .preset-item:hover { background:var(--bg-card-hover); }
.wiz-scope .preset-item .pdel { margin-left:auto;color:var(--text3);font-size:11px;opacity:0;transition:opacity .15s; }
.wiz-scope .preset-item:hover .pdel { opacity:1; }
.wiz-scope .preset-sep { height:1px;background:var(--brd);margin:4px 0; }
.wiz-scope .preset-save-row { padding:6px 12px;display:flex;gap:6px; }
.wiz-scope .preset-save-inp { flex:1;padding:6px 10px;border-radius:8px;border:1px solid var(--brd);background:var(--bg);color:var(--text);font-size:12px;outline:none; }
.wiz-scope .preset-save-inp:focus { border-color:var(--accent); }
.wiz-scope .preset-save-btn { padding:6px 12px;border-radius:8px;border:none;background:var(--accent);color:#fff;font-size:12px;font-weight:600;cursor:pointer; }

/* ── Loading / Building ── */
.wiz-scope .ld { display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;text-align:center;min-height:300px; }
.wiz-scope .ld-t { font-size:22px;font-weight:800;color:var(--text);margin:24px 0 8px; }
.wiz-scope .ld-s { font-size:14px;color:var(--text2);margin-bottom:28px; }
.wiz-scope .ld-bar { width:min(320px,80%);height:6px;border-radius:3px;background:var(--brd);overflow:hidden;margin-bottom:28px;box-shadow:inset 0 1px 3px rgba(0,0,0,.2); }
.wiz-scope .ld-bar-fill { height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent-light));border-radius:3px;transition:width .6s cubic-bezier(.4,0,.2,1);box-shadow:0 0 8px var(--accent-dim); }
.wiz-scope .ld-list { text-align:left;display:flex;flex-direction:column;gap:12px; }
.wiz-scope .ls { display:flex;align-items:center;gap:10px;font-size:14px;color:var(--text3);transition:color .3s; }
.wiz-scope .ls.on { color:var(--text); }
.wiz-scope .ls.ok { color:var(--accent-light); }
.wiz-scope .ls.ok .ls-d svg { display:block; }
.wiz-scope .ls.ok .ls-sp { display:none; }
.wiz-scope .ls:not(.ok) .ls-d svg { display:none; }
.wiz-scope .ls-d { width:20px;height:20px;display:flex;align-items:center;justify-content:center; }
.wiz-scope .ls-sp { width:16px;height:16px;border:2px solid var(--brd);border-top-color:var(--accent);border-radius:50%; }
.wiz-scope .ls.on .ls-sp { animation:wizSpin .8s linear infinite; }

/* ── Spinner ring ── */
.wiz-scope .wiz-ring { width:48px;height:48px;border:3px solid var(--brd);border-top-color:var(--accent);border-radius:50%;animation:wizSpin 1s linear infinite;filter:drop-shadow(0 0 6px var(--accent-dim)); }

/* ── Done state ── */
.wiz-scope .done { display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;text-align:center;animation:wizFadeUp .5s; }
.wiz-scope .done-c { width:64px;height:64px;border-radius:50%;background:linear-gradient(135deg,var(--accent-dim),color-mix(in srgb,var(--accent) 20%,var(--bg-card) 80%));border:2px solid var(--accent);display:flex;align-items:center;justify-content:center;animation:wizCheckPop .5s cubic-bezier(.3,1.5,.6,1);box-shadow:0 0 24px var(--accent-dim); }
.wiz-scope .done-c svg { color:var(--accent-light);width:30px;height:30px; }
.wiz-scope .done-c.err { background:rgba(255,60,60,.12);border-color:rgba(255,60,60,.5); }
.wiz-scope .done-c.err svg { color:#ff6b6b; }
.wiz-scope .done-t { font-size:24px;font-weight:700;color:var(--text);margin:20px 0 8px; }
.wiz-scope .done-s { font-size:14px;color:var(--text2);margin-bottom:28px;line-height:1.5;max-width:400px; }
.wiz-scope .btn { padding:10px 24px;border-radius:14px;font-size:14px;font-weight:600;cursor:pointer;transition:all 0.2s;border:none; }
.wiz-scope .btn.bp { background:linear-gradient(135deg,var(--accent),var(--accent-light));color:#fff; }
.wiz-scope .btn.bp:hover { opacity:.9;transform:translateY(-1px); }
.wiz-scope .btn.bo { background:transparent;border:1.5px solid var(--brd);color:var(--text2); }
.wiz-scope .btn.bo:hover { border-color:var(--accent-glow);color:var(--text); }

/* ── Tooltip System ── */
.wiz-scope [data-tip] { position:relative; }
.wiz-scope [data-tip]::after { content:attr(data-tip);position:absolute;bottom:calc(100% + 10px);left:50%;transform:translateX(-50%);background:var(--bg-card);color:var(--text2);font-size:11.5px;font-weight:500;line-height:1.5;padding:10px 14px;border-radius:10px;border:1px solid var(--accent-glow);border-left:3px solid var(--accent);width:max-content;max-width:240px;pointer-events:none;opacity:0;transition:opacity 0.15s;z-index:9999;box-shadow:0 8px 32px rgba(0,0,0,0.5);white-space:normal; }
.wiz-scope [data-tip]:hover::after { opacity:1; }
.wiz-scope [data-tip-pos="bottom"]::after { bottom:auto;top:calc(100% + 10px); }
.wiz-scope [data-tip-pos="right"]::after { bottom:auto;left:calc(100% + 10px);top:50%;transform:translateY(-50%); }

/* ── Animations ── */
@keyframes wizSpin { to { transform:rotate(360deg); } }
@keyframes wizFadeUp { from { opacity:0;transform:translateY(16px); } to { opacity:1;transform:none; } }
@keyframes wizCheckPop { 0% { transform:scale(0); } 60% { transform:scale(1.2); } 100% { transform:scale(1); } }
@keyframes wizPop { 0% { transform:scale(.7);opacity:0; } 60% { transform:scale(1.08); } 100% { transform:scale(1);opacity:1; } }
@keyframes wizFlash { 0%,100% { opacity:1; } 50% { opacity:.4; } }
@keyframes wizShimmer { 0% { background-position:200% 0; } 100% { background-position:-200% 0; } }
.wiz-scope .wiz-hidden { display:none!important; }

/* ── Mobile ── */
@media (max-width:1000px) {
  .wiz-scope .modal { inset:1vh 2vw; }
  .wiz-scope .rail { width:240px;min-width:240px; }
  .wiz-scope .cards-grid { grid-template-columns:repeat(2,1fr); }
  .wiz-scope .main { padding:20px 24px 32px; }
  .wiz-scope .atmos-selector { display:none; }
}
@media (max-width:768px) {
  .wiz-scope .modal { inset:2vh 2vw;border-radius:16px; }
  .wiz-scope .hdr { padding:10px 16px;gap:8px;flex-wrap:wrap; }
  .wiz-scope .badge { font-size:9px; }
  .wiz-scope .atmos-selector { display:none; }
  .wiz-scope .body { flex-direction:column; }
  .wiz-scope .rail { width:100%;min-width:unset;border-right:none;border-top:1px solid var(--brd);position:fixed;bottom:0;left:0;right:0;z-index:10000;max-height:70vh;transform:translateY(calc(100% - 52px));transition:transform .35s cubic-bezier(.4,0,.2,1);background:var(--bg);border-radius:16px 16px 0 0;box-shadow:0 -4px 24px rgba(0,0,0,.3); }
  .wiz-scope .rail.expanded { transform:translateY(0); }
  .wiz-scope .rail-handle { display:flex;align-items:center;justify-content:center;padding:10px 16px;cursor:pointer;gap:8px; }
  .wiz-scope .rail-handle-bar { width:36px;height:4px;border-radius:2px;background:var(--text3);opacity:.4; }
  .wiz-scope .rail-handle-text { font-size:13px;color:var(--text2);font-weight:500; }
  .wiz-scope .rail-inner { display:none; }
  .wiz-scope .rail.expanded .rail-inner { display:block; }
  .wiz-scope .main { padding:20px 16px 80px; }
  .wiz-scope .cards-grid { grid-template-columns:repeat(2,1fr);gap:12px; }
  .wiz-scope .sec-title { font-size:20px; }
  .wiz-scope .sec-sub { font-size:13px; }
  .wiz-scope .ld-t { font-size:18px; }
  .wiz-scope .done-t { font-size:20px; }
  .wiz-scope .done-s { font-size:13px; }
  .wiz-scope .steps-bar { display:none; }
  .wiz-scope .feed-summary { display:none; }
}
@media (max-width:540px) {
  .wiz-scope .cards-grid { grid-template-columns:1fr; }
}
`;

function injectCSS() {
  if (document.getElementById('wiz-styles')) return;
  const style = document.createElement('style');
  style.id = 'wiz-styles';
  style.textContent = WIZ_CSS;
  document.head.appendChild(style);
}

function removeCSS() {
  const el = document.getElementById('wiz-styles');
  if (el) el.remove();
}

/* ═══════════════════════════════════════
   DOM INJECTION
   ═══════════════════════════════════════ */

function injectDOM(role, location) {
  if (document.getElementById('wiz-root')) return;
  const wrapper = document.createElement('div');
  wrapper.id = 'wiz-root';
  wrapper.className = 'wiz-scope';
  const circumference = 2 * Math.PI * 16; // r=16 for ring
  const isDeep = document.getElementById('wiz-deep-mode')?.checked;
  wrapper.innerHTML = `
    <div class="backdrop" id="wiz-bk" onclick="_wiz.closeWizard()"></div>
    <div class="modal" id="wiz-modal">
      <canvas class="wiz-stars-canvas" id="wiz-stars-canvas"></canvas>
      <div class="grad-bar"></div>
      <div class="hdr">
        <div class="hdr-logo" data-tip="StratOS News Intelligence Platform">StratOS</div>
        <div class="hdr-sub">Intelligence Wizard</div>
        <div class="hdr-badges">
          ${role ? `<span class="badge" data-tip="Your professional role">${esc(role)}</span>` : ''}
          ${location ? `<span class="badge blue" data-tip="Your location">${esc(location)}</span>` : ''}
        </div>
        <div class="preset-dd" id="wiz-preset-dd"></div>
        <div class="atmos-selector">
          <button class="atmos-btn active" data-atmos="clean" onclick="_wiz.setAtmosphere(this)">Clean</button>
          <button class="atmos-btn" data-atmos="deep" onclick="_wiz.setAtmosphere(this)">Deep</button>
          <button class="atmos-btn" data-atmos="arcane" onclick="_wiz.setAtmosphere(this)">Arcane</button>
        </div>
        <div class="ring-wrap" id="wiz-ring-wrap" data-tip="Profile completion progress">
          <svg width="40" height="40" viewBox="0 0 42 42">
            <defs><linearGradient id="wizRingGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="var(--accent)"/><stop offset="100%" stop-color="var(--accent2)"/></linearGradient></defs>
            <circle class="ring-bg" cx="21" cy="21" r="16"/>
            <circle class="ring-fg" id="wiz-ring-fg" cx="21" cy="21" r="16" stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"/>
          </svg>
          <span class="ring-label" id="wiz-ring-pct">0%</span>
        </div>
        <button class="close-btn" onclick="_wiz.closeWizard()" data-tip="Close wizard (Esc)">&times;</button>
      </div>
      <div class="body">
        <div class="rail" id="wiz-rail">
          <div class="rail-handle" id="wiz-rail-handle" onclick="_wiz.toggleRail()">
            <div class="rail-handle-bar"></div>
            <span class="rail-handle-text" id="wiz-rail-handle-text">0 items \u00B7 Build</span>
          </div>
          <div class="rail-inner" id="wiz-rail-scroll"></div>
          <div id="wiz-feed-summary" style="padding:4px 0;"></div>
          <div class="rail-bottom" id="wiz-rail-bottom">
            <div class="depth-toggle" id="wiz-depth-toggle" onclick="_wiz.toggleDeepMode()" data-tip="Quick: fast headlines only. Deep: full article analysis">
              <div class="depth-opt ${isDeep ? '' : 'active'}" id="wiz-mode-quick">&#x26A1; Quick</div>
              <div class="depth-opt ${isDeep ? 'active' : ''}" id="wiz-mode-deep">&#x25CF; Deep</div>
            </div>
            <button class="build-btn" id="wiz-build-btn" onclick="_wiz.doBuild()" disabled data-tip="Generate your personalized intelligence feed">&#x2728; BUILD FEED &#x2728;</button>
          </div>
        </div>
        <div class="main" id="wiz-main">
          <div id="wiz-step-bar"></div>
          <div id="wiz-priorities"></div>
          <div id="wiz-details"></div>
          <div id="wiz-loading" class="wiz-hidden"></div>
        </div>
      </div>
    </div>`;
  document.body.appendChild(wrapper);

  // Create hidden checkbox for deep-mode state
  let deepCb = document.getElementById('wiz-deep-mode');
  if (!deepCb) {
    deepCb = document.createElement('input');
    deepCb.type = 'checkbox';
    deepCb.id = 'wiz-deep-mode';
    deepCb.style.display = 'none';
    wrapper.appendChild(deepCb);
  }
}

function removeDOM() {
  const el = document.getElementById('wiz-root');
  if (el) el.remove();
}


/* ═══════════════════════════════════════
   NAVIGATION HELPERS
   ═══════════════════════════════════════ */

function getActiveTabs() {
  const tabs = [];
  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    if (c.id === 'interests') { tabs.push({id:c.id,name:c.name,icon:c.icon,type:'interests'}); continue; }
    const allSubs = [...c.subs, ...customSubs[c.id]];
    const active = allSubs.filter(s => selSubs[c.id].has(s.id));
    if (active.length) tabs.push({id:c.id,name:c.name,icon:c.icon,type:'category',subs:active});
  }
  return tabs;
}

function getTabSections(tab) {
  const secs = [];
  if (tab.id === 'career') {
    const hasJ = tab.subs.some(s => s.id === 'jobhunt'), hasI = tab.subs.some(s => s.id === 'intern');
    if (hasJ && hasI) secs.push({id:'career_opps',name:'Career Opportunities',icon:'🎯'});
    else if (hasJ) secs.push({id:'jobhunt',name:'Job Hunting',icon:'🔍'});
    else if (hasI) secs.push({id:'intern',name:'Internships',icon:'🧑‍💻'});
    for (const s of tab.subs) if (s.id !== 'jobhunt' && s.id !== 'intern') secs.push({id:s.id,name:s.name,icon:getSubIcon(s.id)});
  } else {
    for (const s of tab.subs) secs.push({id:s.id,name:s.name,icon:getSubIcon(s.id)});
  }
  return secs;
}

function getS3Sections(cat) {
  const allSubs = [...cat.subs, ...customSubs[cat.id]];
  const active = allSubs.filter(s => selSubs[cat.id].has(s.id));
  if (cat.id === 'career') {
    const secs = [];
    const hasJ = active.some(s => s.id === 'jobhunt'), hasI = active.some(s => s.id === 'intern');
    if (hasJ && hasI) secs.push({id:'career_opps',name:'Career Opportunities',icon:'🎯'});
    else if (hasJ) secs.push({id:'jobhunt',name:'Job Hunting',icon:'🔍'});
    else if (hasI) secs.push({id:'intern',name:'Internships',icon:'🧑‍💻'});
    for (const s of active) if (s.id !== 'jobhunt' && s.id !== 'intern') secs.push({id:s.id,name:s.name,icon:getSubIcon(s.id)});
    return secs;
  }
  return active.map(s => ({id:s.id,name:s.name,icon:getSubIcon(s.id)}));
}

function getSubIcon(sid) { return PANELS[sid]?.icon || '\uD83D\uDCCC'; }

function tabHasSelections(tab) {
  if (tab.type === 'interests') return interestTopics.length > 0;
  if (!tab.subs) return false;
  const sections = getTabSections(tab);
  for (const sec of sections) {
    const panel = PANELS[sec.id]; if (!panel) continue;
    const sel = panelSel[sec.id]; if (!sel) continue;
    for (const q of panel.qs) { if (sel[q.id] && sel[q.id].size > 0) return true; }
  }
  return false;
}


/* ═══════════════════════════════════════
   PROGRESS RING
   ═══════════════════════════════════════ */

function updateRing() {
  const fg = document.getElementById('wiz-ring-fg');
  const pct = document.getElementById('wiz-ring-pct');
  if (!fg || !pct) return;
  // Calculate: how many selected categories have at least one detail configured?
  const total = selCats.size;
  if (total === 0) { pct.textContent = '0%'; fg.style.strokeDashoffset = 2 * Math.PI * 16; return; }
  let done = 0;
  for (const cid of selCats) {
    if (cid === 'interests') { if (interestTopics.length > 0) done++; continue; }
    const subs = selSubs[cid];
    if (!subs || subs.size === 0) continue;
    // Check if any sub has panel selections
    let hasSel = false;
    for (const sid of subs) {
      const ps = panelSel[sid];
      if (ps) { for (const qid of Object.keys(ps)) { if (ps[qid] && ps[qid].size > 0) { hasSel = true; break; } } }
      if (hasSel) break;
    }
    if (hasSel) done++;
  }
  const ratio = done / total;
  const circumference = 2 * Math.PI * 16;
  fg.style.strokeDashoffset = circumference * (1 - ratio);
  pct.textContent = Math.round(ratio * 100) + '%';
}

function updateBuildButton() {
  const btn = document.getElementById('wiz-build-btn');
  if (!btn) return;
  if (selCats.size === 0) { btn.disabled = true; return; }
  const hasActiveSubs = [...selCats].some(id => {
    if (id === 'interests') return interestTopics.length > 0;
    return selSubs[id] && selSubs[id].size > 0;
  });
  btn.disabled = !hasActiveSubs;
}

/* ═══════════════════════════════════════
   PRIORITIES — Card grid
   ═══════════════════════════════════════ */

function renderPriorities() {
  const el = document.getElementById('wiz-priorities');
  if (!el) return;
  el.innerHTML = `
    <h2 class="sec-title">Choose Your Focus Areas</h2>
    <p class="sec-sub">Select the categories that matter most to you</p>
    <div class="cards-grid">${CATS.map(c => renderCard(c)).join('')}</div>
    <div class="quick-setup-wrap">
      <button class="quick-setup-btn" onclick="_wiz.skipToQuick()">&#x26A1; Quick Setup <span class="quick-setup-sub">Auto-configure based on your role</span></button>
    </div>`;
}

function renderCard(c) {
  const sel = selCats.has(c.id);
  let pillsHTML = '';
  if (c.dynamic) {
    pillsHTML = `<div class="card-pills"><span class="card-pill-hint">You'll customize these in details \u2192</span></div>`;
  } else {
    const allSubs = [...c.subs, ...customSubs[c.id]];
    const subsHTML = allSubs.map(s => {
      const on = selSubs[c.id].has(s.id);
      const isC = customSubs[c.id].some(cs => cs.id === s.id);
      return `<span class="card-pill ${on ? '' : ''}" onclick="event.stopPropagation();_wiz.togSub('${c.id}','${s.id}',this)" style="cursor:pointer${on ? '' : ';opacity:.5'}">
        ${s.name}${isC ? `<span class="card-pill-x" onclick="event.stopPropagation();_wiz.rmCustomSub('${c.id}','${s.id}')">&times;</span>` : ''}
      </span>`;
    }).join('');
    pillsHTML = `<div class="card-pills">${subsHTML}
      <span id="wiz-saw-${c.id}" style="display:none" onclick="event.stopPropagation()"><input class="add-inp-card" id="wiz-sai-${c.id}" placeholder="Type & Enter" onkeydown="_wiz.addSubKey(event,'${c.id}')"></span>
      <span class="card-pill sp-add" id="wiz-sab-${c.id}" onclick="event.stopPropagation();_wiz.showAddSub('${c.id}')" style="cursor:pointer">+ Add</span>
    </div>`;
  }
  const tapHint = sel ? '' : `<div class="card-tap">Tap to add</div>`;
  return `<div class="p-card ${sel ? 'sel' : ''}" onclick="_wiz.togCat('${c.id}')" data-tip="${esc(c.desc)}">
    <div class="card-check">${CK}</div>
    <div class="card-icon ci-${c.id}">${c.icon}</div>
    <div class="card-name">${c.name}</div>
    <div class="card-desc">${c.desc}</div>
    ${pillsHTML}${tapHint}
  </div>`;
}

/* ═══════════════════════════════════════
   DETAILS — Accordion sections
   ═══════════════════════════════════════ */

function renderDetails() {
  const el = document.getElementById('wiz-details');
  if (!el) return;
  const tabs = getActiveTabs();
  if (!tabs.length) { el.innerHTML = ''; return; }

  const bannerH = _s2BannerDismissed ? '' : `<div class="s2-banner"><span class="s2-banner-icon">&#x2728;</span><span><strong>Bold selections steer the AI</strong> &mdash; pick your career stage and preferences to get personalized suggestions below.</span><span class="s2-banner-x" onclick="_wiz.dismissS2Banner()">&times;</span></div>`;

  let sectionsHTML = '';
  for (const tab of tabs) {
    if (tab.type === 'interests') {
      const isOpen = !_collapsedSections.has('interests_det');
      const itemsH = interestTopics.map(t => `<span class="acc-pill on">${esc(t)}<span class="pill-x" onclick="event.stopPropagation();_wiz.rmIntS2('${escAttr(t)}')">&times;</span></span>`).join('');
      const sugH = INTEREST_SUGGESTIONS.map(s => {
        const on = interestTopics.includes(s);
        return `<span class="acc-pill pill-sug ${on ? 'on' : ''}" onclick="_wiz.togIntS2('${escAttr(s)}')">${esc(s)}</span>`;
      }).join('');
      sectionsHTML += `<div class="accordion ${isOpen ? 'open' : ''}">
        <div class="acc-header" onclick="_wiz.togDetSection('interests_det')" data-tip="Customize your personal interests and hobbies">
          <span class="acc-icon ai-${tab.id}">${tab.icon}</span>
          <span class="acc-title">${tab.name}</span>
          ${interestTopics.length ? `<span class="acc-tag">${interestTopics.length} topics</span>` : ''}
          <span class="acc-chevron">&#9660;</span>
        </div>
        <div class="acc-body"><div class="acc-content">
          <div><div class="acc-group-label">What do you follow? <span class="s2-hint">\u00B7 Type a topic and press Enter</span></div>
          <div class="int-wrap"><input class="int-inp" id="wiz-int-inp" placeholder="e.g. Quantum Computing, Gaming..." onkeydown="_wiz.intKeyS2(event)"></div></div>
          ${interestTopics.length ? `<div><div class="acc-group-label">Your topics</div><div class="acc-pills">${itemsH}</div></div>` : ''}
          <div><div class="acc-group-label">Suggested for your role <span class="s2-hint">\u00B7 tap to add</span></div>
          <div class="acc-pills">${sugH}</div></div>
        </div></div>
      </div>`;
      continue;
    }

    const sections = getTabSections(tab);
    for (const sec of sections) {
      const panel = PANELS[sec.id];
      if (!panel) {
        sectionsHTML += renderGenericDetSection(sec);
        continue;
      }
      const sel = panelSel[sec.id] || {};
      const custom = panelCustom[sec.id] || {};
      const isOpen = !_collapsedSections.has(sec.id);
      let selCount = 0;
      for (const q of panel.qs) { selCount += (sel[q.id]?.size || 0); }

      let bodyH = panel.qs.map(q => {
        const isViewAll = _viewAllPills.has(q.id);
        const basePills = isViewAll ? getAllPills(q.id, q.pills) : getEffectivePills(q.id, q.pills);
        const fullCount = getAllPills(q.id, q.pills).length;
        const shortCount = getEffectivePills(q.id, q.pills).length;
        const hasMore = fullCount > shortCount;
        const picked = sel[q.id] || new Set();
        const all = [...basePills, ...(custom[q.id] || [])];
        const hint = q.hint ? ` <span class="s2-hint">\u00B7 ${q.hint}</span>` : '';
        const isDecisive = DETERMINISTIC_QS.has(q.id);

        let h = `<div><div class="acc-group-label">${q.label}${hint}</div><div class="acc-pills">`;
        for (const p of all) {
          const isC = (custom[q.id] || []).includes(p);
          h += `<span class="acc-pill ${isDecisive ? 'pill-decisive' : ''} ${picked.has(p) ? 'on' : ''}" onclick="_wiz.togPanel('${sec.id}','${q.id}','${escAttr(p)}','${q.type}')">
            ${esc(p)}${isC ? `<span class="pill-x" onclick="event.stopPropagation();_wiz.rmPanelCustom('${sec.id}','${q.id}','${escAttr(p)}')">&times;</span>` : ''}
          </span>`;
        }
        if (q.canAdd) {
          h += `<span id="wiz-aw-${sec.id}-${q.id}" style="display:none"><input class="add-inp" id="wiz-ai-${sec.id}-${q.id}" placeholder="Type & Enter" onkeydown="_wiz.addPanelKey(event,'${sec.id}','${q.id}')"></span>`;
          h += `<span class="add-pill" id="wiz-ab-${sec.id}-${q.id}" onclick="_wiz.showPanelAdd('${sec.id}','${q.id}')">+ Add</span>`;
        }
        h += `</div>`;
        if (hasMore && q.type !== 's') {
          h += `<button class="va-tog" onclick="_wiz.togViewAll('${q.id}')">${isViewAll ? 'Show less' : `View all (${fullCount})`}</button>`;
        }
        h += `</div>`;
        return h;
      }).join('');

      bodyH += renderTabSuggestions(tab.id);

      // Discover pills for this category
      const discoverItems = getRailDiscover(tab.id);
      if (discoverItems.length) {
        bodyH += `<div><div class="acc-group-label" style="display:flex;align-items:center;gap:6px">&#128269; Discover <span class="s2-hint">\u00B7 related topics to explore</span></div>
          <div class="acc-pills">${discoverItems.map(d =>
            `<span class="acc-pill pill-sug ${discoverAdded.has(d.name) ? 'on' : ''}" onclick="_wiz.discoverAdd('${escAttr(d.name)}','${escAttr(d.target || '')}')">${esc(d.name)}${discoverAdded.has(d.name) ? '<span class="pill-x" onclick="event.stopPropagation();_wiz.discoverRm(\''+escAttr(d.name)+'\',\''+escAttr(d.target || '')+'\')">&times;</span>' : ''}</span>`
          ).join('')}</div></div>`;
      }

      sectionsHTML += `<div class="accordion ${isOpen ? 'open' : ''}">
        <div class="acc-header" onclick="_wiz.togDetSection('${sec.id}')">
          <span class="acc-icon ai-${tab.id}">${sec.icon}</span>
          <span class="acc-title">${sec.name}</span>
          ${selCount ? `<span class="acc-tag">${selCount} selected</span>` : ''}
          <span class="acc-chevron">&#9660;</span>
        </div>
        <div class="acc-body"><div class="acc-content">${bodyH}</div></div>
      </div>`;

      if (tab.id !== 'interests' && !_tabSuggestCache[tab.id]) fetchTabSuggestion(tab.id);
    }
  }

  el.innerHTML = `
    <div class="section-divider"></div>
    <h2 class="sec-title2">Fine-tune Your Selections</h2>
    <p class="sec-sub2">Customize each category to get exactly the intelligence you need.</p>
    ${bannerH}${sectionsHTML}`;
}

function renderGenericDetSection(sec) {
  const custom = panelCustom[sec.id]?.kw || [];
  const isOpen = !_collapsedSections.has(sec.id);
  let html = `<div class="accordion ${isOpen ? 'open' : ''}">
    <div class="acc-header" onclick="_wiz.togDetSection('${sec.id}')">
      <span class="acc-icon">\uD83D\uDCCC</span>
      <span class="acc-title">${sec.name}</span>
      ${custom.length ? `<span class="acc-tag">${custom.length} keywords</span>` : ''}
      <span class="acc-chevron">&#9660;</span>
    </div>
    <div class="acc-body"><div class="acc-content">
      <div><div class="acc-group-label">Keywords to track <span class="s2-hint">\u00B7 Type and press Enter</span></div><div class="acc-pills">`;
  for (const p of custom) html += `<span class="acc-pill on">${esc(p)}<span class="pill-x" onclick="event.stopPropagation();_wiz.rmPanelCustom('${sec.id}','kw','${escAttr(p)}')">&times;</span></span>`;
  html += `<span id="wiz-aw-${sec.id}-kw" style="display:none"><input class="add-inp" id="wiz-ai-${sec.id}-kw" placeholder="Type & Enter" onkeydown="_wiz.addPanelKey(event,'${sec.id}','kw')"></span>`;
  html += `<span class="add-pill" id="wiz-ab-${sec.id}-kw" onclick="_wiz.showPanelAdd('${sec.id}','kw')">+ Add</span></div></div>
    </div></div>
  </div>`;
  return html;
}

function togDetSection(secId) {
  // Toggle: if in collapsed set, remove (= open). If not, add (= collapse).
  _collapsedSections.has(secId) ? _collapsedSections.delete(secId) : _collapsedSections.add(secId);
  renderDetails();
}

/* ═══════════════════════════════════════
   LEFT RAIL — Live preview + per-category discover
   ═══════════════════════════════════════ */

function renderRail() {
  const scroll = document.getElementById('wiz-rail-scroll');
  if (!scroll) return;

  let html = '';
  let totalItems = 0;
  let catCount = 0;

  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    catCount++;

    if (catCount > 1) html += `<div class="rail-divider"></div>`;

    if (c.id === 'interests') {
      if (!interestTopics.length) continue;
      totalItems += interestTopics.length;
      const col = rvCollapsed.has('interests');
      html += `<div class="rail-section ${col ? 'collapsed' : ''}">
        <div class="rail-cat-hdr cat-interests" onclick="_wiz.togRvCollapse('interests')" data-tip="Your personal interest topics">
          <span class="rail-cat-icon">${c.icon}</span>
          <span>${c.name}</span>
          <span class="rail-cat-count">${interestTopics.length}</span>
          <span class="rail-sec-chev">\u25BC</span>
        </div>
        <div class="rail-sec-body">
          <div class="rail-items">
            ${interestTopics.map(t => `<div class="rail-item"><span class="bullet"></span>${esc(t)}<span class="rp-x" onclick="event.stopPropagation();_wiz.rvRmInt('${escAttr(t)}')">&times;</span></div>`).join('')}
          </div>
        </div>
      </div>`;
      continue;
    }

    const sections = getS3Sections(c);
    if (!sections.length) continue;

    let catItems = [];
    for (const sec of sections) {
      const items = rvItems[sec.id] || [];
      catItems.push(...items.map(it => ({name: it, sec: sec.id})));
    }
    const subs = selSubs[c.id];
    let subCount = subs ? subs.size : 0;

    totalItems += catItems.length + subCount;
    const col = rvCollapsed.has(c.id);

    html += `<div class="rail-section ${col ? 'collapsed' : ''}">
      <div class="rail-cat-hdr cat-${c.id}" onclick="_wiz.togRvCollapse('${c.id}')" data-tip="${esc(c.desc)}">
        <span class="rail-cat-icon">${c.icon}</span>
        <span>${c.name}</span>
        <span class="rail-cat-count">${catItems.length || subCount}</span>
        <span class="rail-sec-chev">\u25BC</span>
      </div>
      <div class="rail-sec-body">`;

    if (catItems.length) {
      html += `<div class="rail-items">`;
      for (const it of catItems) {
        html += `<div class="rail-item"><span class="bullet"></span>${esc(it.name)}<span class="rp-x" onclick="event.stopPropagation();_wiz.rvRm('${escAttr(it.sec)}','${escAttr(it.name)}')">&times;</span></div>`;
      }
      html += `</div>`;
    } else if (subCount) {
      const subNames = [...subs].map(sid => {
        const sub = [...c.subs, ...customSubs[c.id]].find(s => s.id === sid);
        return sub ? sub.name : sid;
      });
      html += `<div class="rail-items">${subNames.map(n => `<div class="rail-item"><span class="bullet"></span>${esc(n)}</div>`).join('')}</div>`;
    } else {
      html += `<div class="rail-empty">No items yet</div>`;
    }

    // Suggestions and discover moved to main panel (renderDetails)

    html += `</div></div>`;
  }

  if (!html) {
    html = `<div class="rail-empty" style="padding:20px;text-align:center">Select categories to see your preview here</div>`;
  }

  scroll.innerHTML = html;

  const handleText = document.getElementById('wiz-rail-handle-text');
  if (handleText) handleText.textContent = `${totalItems} items \u00B7 Build`;

  renderFeedSummary(totalItems);
  updateBuildButton();
  updateRing();
}

function getRailDiscover(catId) {
  if (!_rvItemsCache?.discover?.length) return [];
  // Match discover items whose target belongs to a sub of this category
  const catSubs = selSubs[catId] || new Set();
  const catSubIds = new Set(catSubs);
  // Also include S3 section IDs (career_opps etc.)
  const cat = CATS.find(c => c.id === catId);
  if (cat) {
    const s3 = getS3Sections(cat);
    for (const sec of s3) catSubIds.add(sec.id);
  }
  return _rvItemsCache.discover.filter(d => catSubIds.has(d.target));
}

function renderFeedSummary(totalItems) {
  const el = document.getElementById('wiz-feed-summary');
  if (!el) return;
  const catNum = selCats.size;
  const isDeep = document.getElementById('wiz-deep-mode')?.checked;
  if (catNum === 0) { el.innerHTML = ''; return; }
  el.innerHTML = `<div class="feed-summary">
    <div class="feed-summary-title">Feed Summary</div>
    <div class="feed-stat"><span class="feed-stat-label">Categories</span><span class="feed-stat-val accent">${catNum} selected</span></div>
    <div class="feed-stat"><span class="feed-stat-label">Tracked topics</span><span class="feed-stat-val">${totalItems} items</span></div>
    <div class="feed-stat"><span class="feed-stat-label">Feed depth</span><span class="feed-stat-val accent">${isDeep ? 'Deep analysis' : 'Quick scan'}</span></div>
  </div>`;
}

function renderStepBar() {
  const el = document.getElementById('wiz-step-bar');
  if (!el) return;
  const hasCats = selCats.size > 0;
  const hasDetails = hasCats && [...selCats].some(id => {
    if (id === 'interests') return interestTopics.length > 0;
    return selSubs[id] && selSubs[id].size > 0;
  });
  el.innerHTML = `<div class="steps-bar">
    <div class="step-num-circle ${hasCats ? 'done' : 'active'}" data-tip="Step 1: Choose your priority categories">1</div>
    <span class="step-bar-label ${hasCats ? 'active' : ''}">Priorities</span>
    <div class="step-line ${hasCats ? 'done' : ''}"></div>
    <div class="step-num-circle ${hasDetails ? 'done' : hasCats ? 'active' : ''}" data-tip="Step 2: Fine-tune your selections">2</div>
    <span class="step-bar-label ${hasCats ? 'active' : ''}">Details</span>
    <div class="step-line ${hasDetails ? 'done' : ''}"></div>
    <div class="step-num-circle ${hasDetails ? 'active' : ''}" data-tip="Step 3: Preview and build your feed">3</div>
    <span class="step-bar-label ${hasDetails ? 'active' : ''}">Build</span>
  </div>`;
}

function toggleRail() {
  const rail = document.getElementById('wiz-rail');
  if (rail) rail.classList.toggle('expanded');
}

/* ── Atmosphere system ── */
let _wizStarEngine = null;

function setAtmosphere(btnEl) {
  const atmos = btnEl.getAttribute('data-atmos');
  const root = document.getElementById('wiz-root');
  if (!root) return;
  root.querySelectorAll('.atmos-btn').forEach(b => b.classList.remove('active'));
  btnEl.classList.add('active');
  if (atmos === 'clean') {
    root.removeAttribute('data-wiz-atmos');
  } else {
    root.setAttribute('data-wiz-atmos', atmos);
  }
  // Start/stop star canvas
  if (atmos === 'arcane') {
    _startWizStars();
  } else {
    _stopWizStars();
  }
  try { localStorage.setItem('wiz-atmosphere', atmos); } catch(e) {}
}

function restoreAtmosphere() {
  try {
    const a = localStorage.getItem('wiz-atmosphere');
    if (a && a !== 'clean') {
      const root = document.getElementById('wiz-root');
      if (root) {
        root.setAttribute('data-wiz-atmos', a);
        root.querySelectorAll('.atmos-btn').forEach(b => {
          b.classList.toggle('active', b.getAttribute('data-atmos') === a);
        });
        if (a === 'arcane') _startWizStars();
      }
    }
  } catch(e) {}
}

function _stopWizStars() {
  if (!_wizStarEngine) return;
  cancelAnimationFrame(_wizStarEngine.raf);
  _wizStarEngine = null;
  const c = document.getElementById('wiz-stars-canvas');
  if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); }
}

function _startWizStars() {
  _stopWizStars();
  const canvas = document.getElementById('wiz-stars-canvas');
  if (!canvas) return;
  const modal = canvas.parentElement;
  if (!modal) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = modal.offsetWidth;
    canvas.height = modal.offsetHeight;
  }
  resize();

  const isMobile = window.innerWidth <= 768;
  const COUNT = isMobile ? 30 : 120;
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#34d399';

  // Parse accent for glow
  const temp = document.createElement('div');
  temp.style.color = accent;
  document.body.appendChild(temp);
  const rgb = getComputedStyle(temp).color.match(/\d+/g) || [52, 211, 153];
  temp.remove();
  const ar = +rgb[0], ag = +rgb[1], ab = +rgb[2];

  const stars = [];
  for (let i = 0; i < COUNT; i++) {
    stars.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.4 + 0.3,
      a: Math.random() * 0.4 + 0.08,
      speed: Math.random() * 0.08 + 0.02,
      phase: Math.random() * Math.PI * 2,
      cr: Math.random() < 0.3 ? ar : 255,
      cg: Math.random() < 0.3 ? ag : 255,
      cb: Math.random() < 0.3 ? ab : 255
    });
  }

  // Shooting stars
  const shooters = [];
  let lastShoot = Date.now();

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const t = Date.now() * 0.001;

    // Stars
    for (const s of stars) {
      s.y -= s.speed;
      if (s.y < -2) { s.y = canvas.height + 2; s.x = Math.random() * canvas.width; }
      const flicker = 0.6 + 0.4 * Math.sin(t * 2 + s.phase);
      const alpha = s.a * flicker;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${alpha})`;
      ctx.fill();
      if (s.r > 1) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r * 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${alpha * 0.15})`;
        ctx.fill();
      }
    }

    // Constellation lines between close stars
    for (let i = 0; i < stars.length; i++) {
      for (let j = i + 1; j < stars.length; j++) {
        const dx = stars[i].x - stars[j].x;
        const dy = stars[i].y - stars[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(stars[i].x, stars[i].y);
          ctx.lineTo(stars[j].x, stars[j].y);
          ctx.strokeStyle = `rgba(${ar},${ag},${ab},${0.06 * (1 - dist / 100)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    // Shooting stars
    if (Date.now() - lastShoot > 5000 + Math.random() * 4000) {
      lastShoot = Date.now();
      const angle = Math.random() * 0.5 + 0.2;
      const spd = Math.random() * 5 + 3;
      shooters.push({
        x: Math.random() * canvas.width * 0.6,
        y: Math.random() * canvas.height * 0.3,
        vx: Math.cos(angle) * spd, vy: Math.sin(angle) * spd,
        life: 1, len: Math.random() * 35 + 20
      });
    }
    for (let i = shooters.length - 1; i >= 0; i--) {
      const sh = shooters[i];
      sh.x += sh.vx; sh.y += sh.vy; sh.life -= 0.015;
      if (sh.life <= 0) { shooters.splice(i, 1); continue; }
      const grad = ctx.createLinearGradient(sh.x, sh.y, sh.x - sh.vx * sh.len / 5, sh.y - sh.vy * sh.len / 5);
      grad.addColorStop(0, `rgba(255,255,255,${sh.life * 0.7})`);
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.beginPath();
      ctx.moveTo(sh.x, sh.y);
      ctx.lineTo(sh.x - sh.vx * sh.len / 5, sh.y - sh.vy * sh.len / 5);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    _wizStarEngine.raf = requestAnimationFrame(draw);
  }

  _wizStarEngine = { raf: requestAnimationFrame(draw) };

  // Resize on window change
  const onResize = () => { if (_wizStarEngine) resize(); };
  window.addEventListener('resize', onResize);
  _wizStarEngine.onResize = onResize;
}

function _cleanupWizStars() {
  if (_wizStarEngine) {
    cancelAnimationFrame(_wizStarEngine.raf);
    if (_wizStarEngine.onResize) window.removeEventListener('resize', _wizStarEngine.onResize);
    _wizStarEngine = null;
  }
}

function toggleDeepMode() {
  const quick = document.getElementById('wiz-mode-quick');
  const deep = document.getElementById('wiz-mode-deep');
  const cb = document.getElementById('wiz-deep-mode');
  if (!quick || !deep) return;
  const isOn = !cb?.checked;
  if (cb) cb.checked = isOn;
  quick.classList.toggle('active', !isOn);
  deep.classList.toggle('active', isOn);
  renderRail();
}


/* ═══════════════════════════════════════
   CATEGORY / SUB TOGGLES
   ═══════════════════════════════════════ */

function togCat(id) {
  if (selCats.has(id)) { selCats.delete(id); if (id !== 'interests') selSubs[id].clear(); else interestTopics = []; }
  else { selCats.add(id); selSubs[id] = selSubs[id] && selSubs[id].size ? selSubs[id] : new Set(); }
  _rvItemsCache = null;
  renderAll(); _wizSaveState();
  // Re-fetch discover items in background
  if (selCats.size > 0) initRvWithAI();
}

function togSub(cid, sid, el) {
  selSubs[cid].has(sid) ? selSubs[cid].delete(sid) : selSubs[cid].add(sid);
  if (selSubs[cid].has(sid) && PANELS[sid] && !panelSel[sid]) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of PANELS[sid].qs) { panelCustom[sid][q.id] = []; panelSel[sid][q.id] = q.type === 's' && q.def ? new Set([q.def]) : new Set(q.defs || []); }
  }
  _rvItemsCache = null;
  if (el) el.classList.toggle('on', selSubs[cid].has(sid));
  renderAll(); _wizSaveState();
  // Re-fetch discover items in background
  if (selCats.size > 0) initRvWithAI();
}

function showAddSub(cid) {
  const w = document.getElementById('wiz-saw-' + cid);
  if (w) w.style.display = 'inline';
  const b = document.getElementById('wiz-sab-' + cid);
  if (b) b.style.display = 'none';
  const i = document.getElementById('wiz-sai-' + cid);
  if (i) i.focus();
}

function addSubKey(e, cid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const name = e.target.value.trim();
    const id = 'custom_' + name.toLowerCase().replace(/\W+/g, '_');
    if (!customSubs[cid].some(s => s.id === id)) { customSubs[cid].push({id, name}); selSubs[cid].add(id); panelSel[id] = {kw: new Set()}; panelCustom[id] = {kw: []}; }
    e.target.value = '';
    renderPriorities(); _wizSaveState();
    setTimeout(() => {
      const w = document.getElementById('wiz-saw-' + cid), b = document.getElementById('wiz-sab-' + cid), i = document.getElementById('wiz-sai-' + cid);
      if (w) w.style.display = 'inline'; if (b) b.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const b = document.getElementById('wiz-sab-' + cid); if (b) b.style.display = '';
  }
}

function rmCustomSub(cid, sid) {
  customSubs[cid] = customSubs[cid].filter(s => s.id !== sid);
  selSubs[cid].delete(sid); delete panelSel[sid]; delete panelCustom[sid];
  renderAll(); _wizSaveState();
}

function clearAll() {
  initState();
  _wizClearState();
  _tabSuggestCache = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  if (_suggestAbortCtrl) { _suggestAbortCtrl.abort(); _suggestAbortCtrl = null; }
  _s2BannerDismissed = false;
  renderAll();
  if (typeof showToast === 'function') showToast('All selections cleared', 'info');
}

/** Render all sections — called after any toggle */
function renderAll() {
  renderStepBar();
  renderPriorities();
  renderDetails();
  renderRail();
  updateBuildButton();
  updateRing();
}

/* ═══════════════════════════════════════
   DETAIL PANEL INTERACTIONS
   ═══════════════════════════════════════ */

function dismissS2Banner() { _s2BannerDismissed = true; const b = document.querySelector('.s2-banner'); if (b) b.remove(); }

function togIntS2(val) { const i = interestTopics.indexOf(val); i >= 0 ? interestTopics.splice(i, 1) : interestTopics.push(val); renderDetails(); renderRail(); _wizSaveState(); }
function rmIntS2(val) { interestTopics = interestTopics.filter(v => v !== val); renderDetails(); renderRail(); _wizSaveState(); }
function intKeyS2(e) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim(); if (!interestTopics.includes(v)) interestTopics.push(v);
    e.target.value = ''; renderDetails(); renderRail(); _wizSaveState();
    setTimeout(() => { const i = document.getElementById('wiz-int-inp'); if (i) i.focus(); }, 60);
  }
}

function togPanel(sid, qid, val, type) {
  if (!panelSel[sid]) panelSel[sid] = {}; if (!panelSel[sid][qid]) panelSel[sid][qid] = new Set();
  const s = panelSel[sid][qid];
  const oldVal = type === 's' ? [...s][0] : null;
  if (type === 's') { s.clear(); s.add(val); } else { s.has(val) ? s.delete(val) : s.add(val); }
  // Find the parent category tab for this section
  const parentTab = getActiveTabs().find(t => {
    if (t.type === 'interests') return false;
    const secs = getTabSections(t);
    return secs.some(sec => sec.id === sid);
  });
  const tabId = parentTab?.id;
  const changed = DETERMINISTIC_QS.has(qid) && (type !== 's' || oldVal !== val);
  if (changed && tabId) {
    if (qid === 'stage') {
      for (const pid of STAGE_SHARED_PANELS) {
        if (pid !== sid) {
          if (!panelSel[pid]) panelSel[pid] = {};
          panelSel[pid]['stage'] = new Set([val]);
        }
      }
      for (const tid of Object.keys(_tabSuggestCache)) {
        if (tid !== tabId) delete _tabSuggestCache[tid];
      }
    }
    clearTimeout(_suggestDebounceTimer);
    _suggestDebounceTimer = setTimeout(() => refreshSuggestions(tabId), 800);
  }
  renderDetails(); renderRail(); updateRing(); _wizSaveState();
}

function showPanelAdd(sid, qid) {
  const w = document.getElementById('wiz-aw-' + sid + '-' + qid); if (w) w.style.display = 'inline';
  const b = document.getElementById('wiz-ab-' + sid + '-' + qid); if (b) b.style.display = 'none';
  const i = document.getElementById('wiz-ai-' + sid + '-' + qid); if (i) i.focus();
}

function addPanelKey(e, sid, qid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim();
    if (!panelCustom[sid]) panelCustom[sid] = {}; if (!panelCustom[sid][qid]) panelCustom[sid][qid] = [];
    if (!panelSel[sid]) panelSel[sid] = {}; if (!panelSel[sid][qid]) panelSel[sid][qid] = new Set();
    if (!panelCustom[sid][qid].includes(v)) { panelCustom[sid][qid].push(v); panelSel[sid][qid].add(v); }
    e.target.value = ''; renderDetails(); _wizSaveState();
    setTimeout(() => {
      const w = document.getElementById('wiz-aw-' + sid + '-' + qid), b = document.getElementById('wiz-ab-' + sid + '-' + qid), i = document.getElementById('wiz-ai-' + sid + '-' + qid);
      if (w) w.style.display = 'inline'; if (b) b.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const b = document.getElementById('wiz-ab-' + sid + '-' + qid); if (b) b.style.display = '';
  }
}

function rmPanelCustom(sid, qid, val) {
  panelCustom[sid][qid] = (panelCustom[sid][qid] || []).filter(v => v !== val);
  if (panelSel[sid]?.[qid]) panelSel[sid][qid].delete(val);
  renderDetails(); _wizSaveState();
}

function togViewAll(qId) {
  _viewAllPills.has(qId) ? _viewAllPills.delete(qId) : _viewAllPills.add(qId);
  renderDetails();
}

/* ═══════════════════════════════════════
   REVIEW ITEMS (AI entities)
   ═══════════════════════════════════════ */

async function fetchRvItems() {
  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  if (!role) return;
  // Build sections list from current selections
  const sections = [];
  for (const c of CATS) {
    if (!selCats.has(c.id) || c.id === 'interests') continue;
    for (const sec of getS3Sections(c)) {
      sections.push({id: sec.id, name: sec.name, category: c.name});
    }
  }
  if (!sections.length) return;
  _rvLoading = true;
  try {
    const resp = await fetch('/api/wizard-rv-items', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role, location, sections})
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data.sections && Object.keys(data.sections).length) {
        _rvItemsCache = data;
        console.log('[Wizard] rv-items: got AI entities for', Object.keys(data.sections).length, 'sections');
      }
    }
  } catch(e) { console.warn('[Wizard] rv-items fetch failed:', e); }
  _rvLoading = false;
}

function initRv() {
  rvItems = {};
  for (const c of CATS) {
    if (!selCats.has(c.id) || c.id === 'interests') continue;
    const sections = getS3Sections(c);
    for (const sec of sections) {
      // Use AI-generated items only — no hardcoded fallback
      const aiSec = _rvItemsCache?.sections?.[sec.id];
      if (aiSec && aiSec.items?.length) {
        rvItems[sec.id] = [...aiSec.items.slice(0, 8)];
      } else {
        // Empty — renderRail will show "No items yet" placeholder
        rvItems[sec.id] = [];
      }
    }
  }
}

async function initRvWithAI() {
  if (_rvItemsCache) {
    initRv();
    renderRail();
    return;
  }
  // Show loading in main area
  const loading = document.getElementById('wiz-loading');
  if (loading) {
    loading.classList.remove('wiz-hidden');
    const role = document.getElementById('simple-role')?.value?.trim() || '';
    loading.innerHTML = '<div class="ld"><div class="wiz-ring"></div><div class="ld-t">Personalizing your review...</div><div class="ld-s">Finding relevant entities for <strong>' + esc(role) + '</strong></div></div>';
  }
  await fetchRvItems();
  initRv();
  if (loading) { loading.classList.add('wiz-hidden'); loading.innerHTML = ''; }
  renderRail();
}

function togRvCollapse(id) { rvCollapsed.has(id) ? rvCollapsed.delete(id) : rvCollapsed.add(id); renderRail(); }
function collapseAllRv() { for (const c of CATS) if (selCats.has(c.id)) rvCollapsed.add(c.id); renderRail(); }
function expandAllRv() { rvCollapsed.clear(); renderRail(); }

function rvRm(sid, val) { rvItems[sid] = (rvItems[sid] || []).filter(v => v !== val); discoverAdded.delete(val); renderRail(); }
function rvRmInt(val) { interestTopics = interestTopics.filter(v => v !== val); renderRail(); renderDetails(); }
function rvShowAdd(sid) {
  const w = document.getElementById('wiz-rw-' + sid); if (w) w.style.display = 'inline';
  const a = document.getElementById('wiz-ra-' + sid); if (a) a.style.display = 'none';
  const i = document.getElementById('wiz-ri-' + sid); if (i) i.focus();
}
function rvAddKey(e, sid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim(); if (!rvItems[sid]) rvItems[sid] = []; if (!rvItems[sid].includes(v)) rvItems[sid].push(v);
    e.target.value = ''; renderRail();
    setTimeout(() => {
      const w = document.getElementById('wiz-rw-' + sid), a = document.getElementById('wiz-ra-' + sid), i = document.getElementById('wiz-ri-' + sid);
      if (w) w.style.display = 'inline'; if (a) a.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const a = document.getElementById('wiz-ra-' + sid); if (a) a.style.display = '';
  }
}

/* ═══════════════════════════════════════
   BUILD / DONE
   ═══════════════════════════════════════ */

function buildWizardContext() {
  // Build a CLEAN context string from wizard selections for the generate API.
  //
  // CRITICAL: Only include role-relevant signals. Default pill text like
  // "5G Rollout", "Cloud & SaaS", "AI & Automation" are generic tech defaults
  // that contaminate every profile regardless of role.
  //
  // INCLUDE: category/sub names, deterministic answers (stage, opptype),
  //          user-typed custom items, interest topics
  // EXCLUDE: default pill selections (non-deterministic, tech-biased defaults)
  const parts = [];
  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    if (c.id === 'interests' && interestTopics.length) {
      parts.push(`Interests: ${interestTopics.join(', ')}`);
      continue;
    }
    const allSubs = [...c.subs, ...(customSubs[c.id] || [])];
    const activeSubs = allSubs.filter(s => selSubs[c.id]?.has(s.id));
    if (!activeSubs.length) continue;
    const subDetails = [];
    for (const sub of activeSubs) {
      const panel = PANELS[sub.id];
      if (panel && panelSel[sub.id]) {
        const contextParts = [];
        for (const q of panel.qs) {
          const sel = panelSel[sub.id]?.[q.id];
          if (!sel || !sel.size) continue;
          if (DETERMINISTIC_QS.has(q.id)) {
            // Stage, opptype, clevel, etc. — always include (role-relevant context)
            contextParts.push([...sel].join(', '));
          } else {
            // Non-deterministic: ONLY include user-added custom items, not defaults
            const customs = panelCustom[sub.id]?.[q.id] || [];
            if (customs.length) contextParts.push(customs.join(', '));
          }
        }
        if (contextParts.length) subDetails.push(`${sub.name} (${contextParts.join('; ')})`);
        else subDetails.push(sub.name);
      } else {
        subDetails.push(sub.name);
      }
    }
    parts.push(`${c.name}: ${subDetails.join(', ')}`);
  }
  const result = parts.join('. ');
  console.debug('[Wizard] buildWizardContext:', result);
  return result;
}

function doBuild() {
  const role = document.getElementById('simple-role')?.value?.trim() || 'your profile';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const deep = document.getElementById('wiz-deep-mode')?.checked || false;
  const modeLabel = deep ? 'Deep analysis' : 'Quick generation';
  // Show loading in main area
  const main = document.getElementById('wiz-main');
  if (!main) return;
  main.innerHTML = `<div class="ld" id="wiz-build-ld">
    <div class="wiz-ring"></div>
    <div class="ld-t">Building your feed...</div>
    <div class="ld-s">${modeLabel} for <strong>${esc(role)}</strong></div>
    <div class="ld-bar"><div class="ld-bar-fill" id="wiz-bar"></div></div>
    <div class="ld-list">
      <div class="ls on" id="wiz-l0"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Generating context from your choices</span></div>
      <div class="ls" id="wiz-l1"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Building tracking categories${deep ? ' (deep thinking)' : ''}</span></div>
      <div class="ls" id="wiz-l2"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Selecting news sources & scoring model</span></div>
    </div>
  </div>`;
  main.scrollTo(0, 0);
  // Disable build button
  const btn = document.getElementById('wiz-build-btn');
  if (btn) btn.disabled = true;
  const wizContext = buildWizardContext();
  callGenerateProfile(role, location, wizContext, deep);
}

async function callGenerateProfile(role, location, context, deep = false) {
  console.debug('[Wizard] callGenerateProfile:', {role, location, context, deep});
  let stepIdx = 0;
  const bar = document.getElementById('wiz-bar');
  const setBar = (pct) => { if (bar) bar.style.width = pct + '%'; };
  const advanceStep = () => {
    if (stepIdx > 0) { const prev = document.getElementById('wiz-l' + (stepIdx - 1)); if (prev) { prev.classList.remove('on'); prev.classList.add('ok'); } }
    if (stepIdx < 3) { const cur = document.getElementById('wiz-l' + stepIdx); if (cur) cur.classList.add('on'); stepIdx++; }
  };
  advanceStep(); setBar(5);
  try {
    await new Promise(r => setTimeout(r, 600));
    advanceStep(); setBar(15);
    let progressPct = 15;
    const startTime = Date.now();
    const progressTimer = setInterval(() => {
      if (progressPct < 75) { progressPct += (75 - progressPct) * 0.06; setBar(Math.round(progressPct)); }
      const subtitle = document.querySelector('.ld-s');
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      if (subtitle && elapsed > 3) {
        const mode = deep ? 'Deep analysis' : 'Generating';
        subtitle.innerHTML = mode + ' for <strong>' + esc(role) + '</strong> (' + elapsed + 's)';
      }
    }, 400);
    const resp = await fetch('/api/generate-profile', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: AbortSignal.timeout(180000),
      body: JSON.stringify({role, location, context, deep})
    });
    clearInterval(progressTimer);
    if (!resp.ok) throw new Error('Server error: ' + resp.status);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    _wizGenerateData = data;
    advanceStep(); setBar(80);
    await new Promise(r => setTimeout(r, 500));
    advanceStep(); setBar(100);
    await new Promise(r => setTimeout(r, 400));
    showDone();
  } catch (e) {
    console.error('Wizard generate failed:', e);
    setBar(100);
    _wizGenerateData = null;
    showDone(e.message);
  }
}

function showDone(errorMsg) {
  const el = document.getElementById('wiz-main');
  if (!el) return;
  if (errorMsg) {
    el.innerHTML = '<div class="done">' +
      '<div class="done-c err">' + CK.replace('viewBox', 'width="30" height="30" fill="none" viewBox') + '</div>' +
      '<div class="done-t">Generation had issues</div>' +
      '<div class="done-s">We couldn\'t fully generate your profile: ' + esc(errorMsg) + '.<br>You can still use the wizard selections or try again.</div>' +
      '<div style="display:flex;gap:12px">' +
        '<button class="btn bo" onclick="_wiz.restoreMainView()">Try Again</button>' +
        '<button class="btn bp" onclick="_wiz.finishWizard()" style="padding:14px 36px;font-size:16px">Use Selections Anyway \u2192</button>' +
      '</div></div>';
  } else {
    const catCount = _wizGenerateData?.categories?.length || 0;
    const itemCount = (_wizGenerateData?.categories || []).reduce((a, c) => a + (c.items?.length || 0), 0);
    el.innerHTML = '<div class="done">' +
      '<div class="done-c">' + CK.replace('viewBox', 'width="30" height="30" fill="none" viewBox') + '</div>' +
      '<div class="done-t">Your feed is ready!</div>' +
      '<div class="done-s">Generated ' + catCount + ' categories with ' + itemCount + ' tracking items. Your dashboard will now show signals tailored to your profile.</div>' +
      '<button class="btn bp" onclick="_wiz.finishWizard()" style="padding:14px 36px;font-size:16px">Apply & Close \u2192</button>' +
      '</div>';
  }
}

function restoreMainView() {
  const main = document.getElementById('wiz-main');
  if (!main) return;
  main.innerHTML = '<div id="wiz-step-bar"></div><div id="wiz-priorities"></div><div id="wiz-details"></div><div id="wiz-loading" class="wiz-hidden"></div>';
  renderAll();
}

/* ═══════════════════════════════════════
   FINISH — Connect wizard output to settings
   (Stage 6: will be expanded with real settings integration)
   ═══════════════════════════════════════ */

function _wizExtractTickers() {
  const tickers = [];
  const seen = new Set();
  const panelQMap = { crypto: 'ccoins', commodities: 'comms', forex: 'fpairs', stocks: 'smkt' };
  for (const [panelId, qId] of Object.entries(panelQMap)) {
    const sel = panelSel[panelId]?.[qId];
    if (!sel) continue;
    for (const item of sel) {
      const ticker = WIZ_TICKER_MAP[item];
      if (ticker && !seen.has(ticker)) {
        seen.add(ticker);
        tickers.push(ticker);
      }
    }
  }
  return tickers;
}

async function finishWizard() {
  _wizClearState(); // Clear saved wizard state — user has committed to this config

  // Clear stale category library — wizard generates a complete new profile
  try {
    const libKey = typeof _categoryLibraryKey === 'function' ? _categoryLibraryKey() : 'categoryLibrary';
    localStorage.removeItem(libKey);
  } catch(e) {}
  // Clear previous simpleCategories to prevent load-priority contamination
  try { localStorage.removeItem('simpleCategories'); } catch(e) {}

  if (_wizGenerateData) {
    // Apply AI-generated categories, preserving any pinned (manually-added) ones
    if (_wizGenerateData.categories && Array.isArray(_wizGenerateData.categories)) {
      const pinned = (typeof simpleCategories !== 'undefined' ? simpleCategories : []).filter(c => c.pinned);
      const generated = _wizGenerateData.categories;
      const pinnedLabels = new Set(pinned.map(c => (c.label || c.name || '').toLowerCase()));
      const filtered = generated.filter(c => !pinnedLabels.has((c.label || c.name || '').toLowerCase()));
      if (typeof simpleCategories !== 'undefined') {
        simpleCategories.length = 0;
        [...filtered, ...pinned].forEach(c => simpleCategories.push(c));
      }
    }
    // Apply tickers — merge LLM-generated with wizard panel selections
    if (typeof simpleTickers !== 'undefined') {
      const llmTickers = (_wizGenerateData.tickers && Array.isArray(_wizGenerateData.tickers))
        ? _wizGenerateData.tickers : [];
      const wizTickers = _wizExtractTickers();
      const seen = new Set();
      simpleTickers.length = 0;
      for (const t of [...llmTickers, ...wizTickers]) {
        const upper = t.toUpperCase();
        if (!seen.has(upper)) { seen.add(upper); simpleTickers.push(t); }
      }
    }
    // Apply timelimit
    if (_wizGenerateData.timelimit && typeof simpleTimelimit !== 'undefined') {
      simpleTimelimit = _wizGenerateData.timelimit;
    }
    // Apply context
    if (_wizGenerateData.context && typeof simpleContext !== 'undefined') {
      simpleContext = _wizGenerateData.context;
      const ctxEl = document.getElementById('simple-context');
      if (ctxEl) ctxEl.value = simpleContext;
    }

    // Call settings.js render/sync functions
    if (typeof renderDynamicCategories === 'function') renderDynamicCategories();
    if (typeof renderSimpleTickers === 'function') renderSimpleTickers();
    if (typeof updateTimelimitButtons === 'function') updateTimelimitButtons();
    if (typeof syncToAdvanced === 'function') syncToAdvanced();
    if (typeof saveSimpleState === 'function') saveSimpleState();

    const catCount = _wizGenerateData.categories?.length || 0;

    // Auto-save to backend BEFORE closing wizard — closeWizard removes DOM
    // elements that syncToAdvanced reads, causing a race condition if done after
    if (typeof saveConfig === 'function') {
      window._pendingDynamicCategories = typeof simpleCategories !== 'undefined' ? simpleCategories : null;
      try {
        await saveConfig();
        window._pendingDynamicCategories = null;
        closeWizard();
        if (typeof showToast === 'function') showToast(`Profile saved! ${catCount} categories configured.`, 'success');
      } catch(e) {
        window._pendingDynamicCategories = null;
        closeWizard();
        if (typeof showToast === 'function') showToast(`Wizard applied ${catCount} categories but save failed. Click Save to retry.`, 'warning');
      }
    } else {
      closeWizard();
      if (typeof showToast === 'function') showToast(`Wizard applied ${catCount} categories. Click Save to persist.`, 'success');
    }
  } else {
    // No generate data — just close
    closeWizard();
    if (typeof showToast === 'function') showToast('Wizard closed. No categories were generated.', 'info');
  }
}

/* ═══════════════════════════════════════
   SKIP TO QUICK SETUP
   ═══════════════════════════════════════ */

/* Domain-specific fallback category/sub defaults when AI is unavailable */
const DOMAIN_CAT_MAP = {
  tech:['career','learning','industry'], data:['career','learning','industry'], cybersec:['career','learning','industry'],
  finance:['markets','career','industry'], marketing:['career','learning','interests'], sales:['career','deals','interests'],
  healthcare:['career','learning','industry'], education:['career','learning','interests'], legal:['career','industry','learning'],
  creative:['career','interests','learning'], hr:['career','learning','deals'], consulting:['career','learning','interests'],
  operations:['career','industry','learning'], realestate:['markets','career','industry'], hospitality:['career','deals','learning'],
  government:['career','learning','interests'], trades:['career','deals','learning'], science:['career','learning','interests'],
  agriculture:['career','markets','interests'], nonprofit:['career','interests','learning'], retail:['career','deals','learning'],
  engineering:['career','learning','industry'], default:['career','learning','deals'],
};
const DOMAIN_SUB_MAP = {
  tech:{career:['jobhunt'],industry:['ind_tech'],learning:['certs','courses']},
  data:{career:['jobhunt'],industry:['ind_tech'],learning:['courses','academic']},
  cybersec:{career:['jobhunt'],industry:['ind_tech'],learning:['certs','hands_on']},
  finance:{markets:['stocks','forex'],career:['jobhunt'],industry:['ind_bank']},
  marketing:{career:['jobhunt'],interests:[]},
  sales:{career:['jobhunt'],deals:['bankdeals']},
  healthcare:{career:['jobhunt'],industry:['ind_health'],learning:['certs','academic']},
  education:{career:['jobhunt','research_pos'],learning:['academic','courses']},
  legal:{career:['jobhunt'],industry:['ind_bank'],learning:['certs']},
  creative:{career:['jobhunt'],learning:['courses']},
  hr:{career:['jobhunt'],learning:['certs','courses']},
  consulting:{career:['jobhunt'],learning:['certs']},
  operations:{career:['jobhunt'],industry:['ind_construct'],learning:['certs']},
  realestate:{markets:['realestate','stocks'],career:['jobhunt'],industry:['ind_construct']},
  hospitality:{career:['jobhunt'],deals:['bankdeals','studisc']},
  government:{career:['govjobs'],learning:['certs']},
  trades:{career:['jobhunt'],deals:['bankdeals'],learning:['hands_on']},
  science:{career:['research_pos'],learning:['academic','courses']},
  agriculture:{career:['jobhunt'],markets:['commodities']},
  nonprofit:{career:['jobhunt'],industry:['ind_health']},
  retail:{career:['jobhunt'],deals:['bankdeals','ccrewards']},
  engineering:{career:['jobhunt'],learning:['certs','academic'],industry:['ind_construct']},
  default:{career:['jobhunt','intern'],learning:['certs'],deals:['bankdeals','studisc']},
};

function applyDomainDefaults() {
  const cats = DOMAIN_CAT_MAP[_currentDomain] || DOMAIN_CAT_MAP.default;
  const subs = DOMAIN_SUB_MAP[_currentDomain] || DOMAIN_SUB_MAP.default;
  selCats = new Set(cats);
  for (const c of CATS) {
    if (selCats.has(c.id) && subs[c.id]) {
      selSubs[c.id] = new Set(subs[c.id].filter(s => c.subs.some(cs => cs.id === s)));
    } else if (!selCats.has(c.id)) {
      selSubs[c.id] = new Set();
    }
  }
}

function applySmartPanelDefaults() {
  const inferredStage = inferStage(document.getElementById('simple-role')?.value?.trim() || '');
  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    const allSubs = [...c.subs, ...(customSubs[c.id] || [])];
    for (const sub of allSubs) {
      if (!selSubs[c.id].has(sub.id)) continue;
      const panel = PANELS[sub.id];
      if (!panel) continue;
      if (!panelSel[sub.id]) panelSel[sub.id] = {};
      if (!panelCustom[sub.id]) panelCustom[sub.id] = {};
      for (const q of panel.qs) {
        if (!panelCustom[sub.id][q.id]) panelCustom[sub.id][q.id] = [];
        if (q.id === 'stage') {
          panelSel[sub.id][q.id] = new Set([inferredStage]);
        } else if (q.id === 'opptype') {
          // Cascade: stage affects opportunity type
          if (inferredStage === 'Student') panelSel[sub.id][q.id] = new Set(['Internships']);
          else if (inferredStage === 'Fresh Graduate') panelSel[sub.id][q.id] = new Set(['Full-time Jobs','Internships']);
          else panelSel[sub.id][q.id] = new Set(['Full-time Jobs']);
        } else if (q.type === 's' && q.def) {
          panelSel[sub.id][q.id] = new Set([q.def]);
        } else {
          const pills = getEffectivePills(q.id, q.pills);
          panelSel[sub.id][q.id] = new Set(pills.slice(0, Math.min(3, pills.length)));
        }
      }
    }
  }
}

async function skipToQuick() {
  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';

  // Show loading in main area
  const main = document.getElementById('wiz-main');
  if (!main) return;
  main.innerHTML = '<div class="ld">' +
    '<div class="wiz-ring"></div>' +
    '<div class="ld-t">Analyzing your profile...</div>' +
    '<div class="ld-s">Finding the best setup for <strong>' + esc(role) + '</strong>' + (location ? ' in <strong>' + esc(location) + '</strong>' : '') + '</div>' +
    '<div class="ld-bar"><div class="ld-bar-fill" id="wiz-qbar"></div></div>' +
    '<div class="ld-list">' +
      '<div class="ls on" id="wiz-q0"><div class="ls-d"><div class="ls-sp"></div>' + CK + '</div><span>Analyzing role &amp; location</span></div>' +
      '<div class="ls" id="wiz-q1"><div class="ls-d"><div class="ls-sp"></div>' + CK + '</div><span>Selecting relevant categories</span></div>' +
      '<div class="ls" id="wiz-q2"><div class="ls-d"><div class="ls-sp"></div>' + CK + '</div><span>Preparing your review</span></div>' +
    '</div></div>';
  main.scrollTo(0, 0);

  let qIdx = 0;
  const qbar = document.getElementById('wiz-qbar');
  const setQBar = (pct) => { if (qbar) qbar.style.width = pct + '%'; };
  const advQ = () => {
    if (qIdx > 0) { const prev = document.getElementById('wiz-q' + (qIdx - 1)); if (prev) { prev.classList.remove('on'); prev.classList.add('ok'); } }
    if (qIdx < 3) { const cur = document.getElementById('wiz-q' + qIdx); if (cur) cur.classList.add('on'); qIdx++; }
  };
  advQ(); setQBar(10);

  try {
    await new Promise(r => setTimeout(r, 400));
    advQ(); setQBar(40);
    _currentDomain = classifyRole(role);
    let aiSuccess = false;
    try {
      const available = CATS.map(c => ({ id: c.id, label: c.name, subs: (c.subs || []).map(s => ({id: s.id, label: s.name})) }));
      const resp = await fetch('/api/wizard-preselect', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({role, location, available_categories: available})
      });
      if (resp.ok) {
        const data = await resp.json();
        if (!data.error && data.selected_categories?.length) {
          selCats = new Set(data.selected_categories.filter(id => CATS.some(c => c.id === id)));
          const aiSubs = data.selected_subs || {};
          for (const c of CATS) {
            if (selCats.has(c.id) && aiSubs[c.id]) {
              const validSubIds = new Set(c.subs.map(s => s.id));
              selSubs[c.id] = new Set(aiSubs[c.id].filter(s => validSubIds.has(s)));
            } else if (!selCats.has(c.id)) { selSubs[c.id] = new Set(); }
          }
          aiSuccess = true;
        }
      }
    } catch(e) { /* AI unavailable */ }
    if (!aiSuccess) applyDomainDefaults();
    applySmartPanelDefaults();

    await new Promise(r => setTimeout(r, 300));
    advQ(); setQBar(50);
    await fetchRvItems();
    setQBar(80);
    initRv();

    advQ(); setQBar(100);
    await new Promise(r => setTimeout(r, 400));

    // Restore main view with selections applied
    rvCollapsed.clear();
    restoreMainView();
  } catch (e) {
    console.error('Quick setup failed:', e);
    restoreMainView();
    if (typeof showToast === 'function') showToast('Quick setup had an issue. Please select manually.', 'warning');
  }
}

function discoverAdd(name, target) {
  if (discoverAdded.has(name)) {
    // Toggle off
    discoverAdded.delete(name);
    rvItems[target] = (rvItems[target] || []).filter(v => v !== name);
  } else {
    discoverAdded.add(name);
    if (!rvItems[target]) rvItems[target] = [];
    if (!rvItems[target].includes(name)) rvItems[target].push(name);
  }
  renderRail();
  renderDetails();
}
function discoverRm(name, target) {
  discoverAdded.delete(name);
  rvItems[target] = (rvItems[target] || []).filter(v => v !== name);
  renderRail();
  renderDetails();
}

/* ═══════════════════════════════════════
   UTILITY
   ═══════════════════════════════════════ */

function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function escAttr(s) { return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"'); }


/* ═══════════════════════════════════════
   KEYBOARD SHORTCUTS
   ═══════════════════════════════════════ */

function setupKeyboard() {
  document.addEventListener('keydown', function wizKeyHandler(e) {
    const root = document.getElementById('wiz-root');
    const modal = root?.querySelector('.modal');
    if (!modal || !modal.classList.contains('open')) return;
    const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
    if (inInput) {
      if (e.key === 'Escape') { e.preventDefault(); e.target.blur(); }
      return;
    }
    if (e.key === 'Escape') { e.preventDefault(); if (_presetMenuOpen) { _presetMenuOpen = false; renderPresetBar(); } else closeWizard(); }
    else if (e.key >= '1' && e.key <= '6') {
      e.preventDefault(); const idx = parseInt(e.key) - 1;
      if (idx < CATS.length) togCat(CATS[idx].id);
    }
  });
  document.addEventListener('click', function wizClickOutside(e) {
    if (!_presetMenuOpen) return;
    const dd = document.querySelector('.wiz-scope .preset-dd');
    if (dd && !dd.contains(e.target)) { _presetMenuOpen = false; renderPresetBar(); }
  });
}

/* ═══════════════════════════════════════
   OPEN / CLOSE
   ═══════════════════════════════════════ */

function openWizard(opts) {
  opts = opts || {};
  const role = document.getElementById('simple-role')?.value?.trim();
  const location = document.getElementById('simple-location')?.value?.trim();
  if (!role) {
    if (typeof showToast === 'function') showToast('Enter your role first', 'warning');
    return;
  }
  _wizMode = opts.mode || 'suggest';
  initState();
  const restored = _wizLoadState();
  if (restored) console.log('[Wizard] Restored saved state from localStorage');
  injectCSS();
  injectDOM(role, location);
  _presetMenuOpen = false;
  renderPresetBar();

  requestAnimationFrame(() => {
    const root = document.getElementById('wiz-root');
    if (root) {
      const bk = root.querySelector('.backdrop');
      const modal = root.querySelector('.modal');
      if (bk) bk.classList.add('open');
      if (modal) modal.classList.add('open');
    }
  });
  document.body.style.overflow = 'hidden';

  // Restore saved atmosphere
  restoreAtmosphere();

  // Render initial state
  renderAll();

  // Fetch AI pre-selection in background
  if (!restored && role) {
    _currentDomain = classifyRole(role);
    fetchPreselection(role, location);
  }

  // Kick off AI entity fetch in background for rail
  if (!_rvItemsCache) initRvWithAI();
}


/* ═══════════════════════════════════════
   AI PRE-SELECTION
   ═══════════════════════════════════════ */

async function fetchPreselection(role, location) {
  try {
    const available = CATS.map(c => ({
      id: c.id, label: c.name,
      subs: (c.subs || []).map(s => ({id: s.id, label: s.name}))
    }));
    const resp = await fetch('/api/wizard-preselect', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role, location, available_categories: available})
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.error) return;
    const aiCats = data.selected_categories || [];
    const aiSubs = data.selected_subs || {};
    selCats = new Set(aiCats.filter(id => CATS.some(c => c.id === id)));
    for (const c of CATS) {
      if (selCats.has(c.id) && aiSubs[c.id]) {
        const validSubIds = new Set([...c.subs.map(s => s.id)]);
        selSubs[c.id] = new Set(aiSubs[c.id].filter(s => validSubIds.has(s)));
      } else if (!selCats.has(c.id)) {
        selSubs[c.id] = new Set();
      }
    }
    renderAll();
  } catch (e) {
    console.log('Wizard: AI pre-selection unavailable, using defaults');
  }
}

/* ═══════════════════════════════════════
   AI TAB SUGGESTIONS
   ═══════════════════════════════════════ */

async function fetchTabSuggestion(tabId, extraExclude, isRefresh) {
  if (_tabSuggestCache[tabId]) return;
  _tabSuggestCache[tabId] = {suggestions: [], loading: true, added: new Set(), isRefresh: !!isRefresh};
  renderDetails();

  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const cat = CATS.find(c => c.id === tabId);
  if (!cat) { _tabSuggestCache[tabId].loading = false; return; }

  const existingItems = [];
  const deterministicParts = [];
  const tab = getActiveTabs().find(t => t.id === tabId);
  const sections = tab ? getTabSections(tab) : [];
  for (const sec of sections) {
    const panel = PANELS[sec.id];
    if (panel && panelSel[sec.id]) {
      for (const q of panel.qs) {
        const picked = panelSel[sec.id]?.[q.id];
        if (picked && picked.size) {
          if (DETERMINISTIC_QS.has(q.id)) { deterministicParts.push(q.label + ': ' + [...picked].join(', ')); }
          else { existingItems.push(...picked); }
        }
      }
    }
  }
  if (extraExclude && extraExclude.size) existingItems.push(...extraExclude);
  const selectionsContext = deterministicParts.join('; ');
  const selections = {};
  for (const sec of sections) {
    const panel = PANELS[sec.id];
    if (panel && panelSel[sec.id]) {
      for (const q of panel.qs) {
        const picked = panelSel[sec.id]?.[q.id];
        if (picked && picked.size) selections[q.label] = [...picked];
      }
    }
  }

  try {
    if (_suggestAbortCtrl) _suggestAbortCtrl.abort();
    _suggestAbortCtrl = new AbortController();
    const resp = await fetch('/api/wizard-tab-suggest', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      signal: AbortSignal.any([_suggestAbortCtrl.signal, AbortSignal.timeout(300000)]),
      body: JSON.stringify({ role, location, category_id: tabId, category_label: cat.name,
        existing_items: existingItems, selections_context: selectionsContext, selections,
        exclude_selected: extraExclude ? [...extraExclude] : [] })
    });
    if (!resp.ok) throw new Error('Request failed');
    const data = await resp.json();
    _tabSuggestCache[tabId] = { suggestions: data.suggestions || [], loading: false, added: new Set() };
  } catch (e) {
    if (e.name === 'AbortError') return;
    _tabSuggestCache[tabId] = {suggestions: [], loading: false, added: new Set()};
    console.log('Wizard: Tab suggestion unavailable for', tabId);
  }
  renderDetails();
}

function addTabSuggestion(tabId, keyword) {
  const cache = _tabSuggestCache[tabId];
  if (!cache) return;
  if (!cache.added) cache.added = new Set();
  // Toggle: if already added, remove it
  if (cache.added.has(keyword)) {
    cache.added.delete(keyword);
    if (tabId === 'interests') {
      interestTopics = interestTopics.filter(t => t !== keyword);
    } else {
      if (selSubs[tabId]) selSubs[tabId].delete(keyword);
    }
  } else {
    cache.added.add(keyword);
    if (tabId === 'interests') {
      if (!interestTopics.includes(keyword)) interestTopics.push(keyword);
    } else {
      if (!selSubs[tabId]) selSubs[tabId] = new Set();
      selSubs[tabId].add(keyword);
    }
  }
  renderDetails();
  renderRail();
}

function renderTabSuggestions(tabId) {
  const cache = _tabSuggestCache[tabId];
  if (!cache) return '';
  if (cache.loading) {
    // Shimmer placeholders instead of just a spinner
    const shimmerPills = '<div class="ai-sug-pill shimmer"></div>'.repeat(5);
    return `<div class="ai-sug"><div class="ai-sug-hdr ${cache.isRefresh ? 'flash' : ''}"><div class="ai-sug-spin"></div> Suggesting...<button class="ai-ref spin" disabled style="opacity:.3">&#x21bb;</button></div><div class="acc-pills">${shimmerPills}</div></div>`;
  }
  const added = cache.added || new Set();
  const kept = cache.keptFromPrev || new Set();
  const allSugs = cache.suggestions || [];
  // No suggestions at all and nothing kept? Hide section
  if (!allSugs.length && !kept.size) return '';

  // Build pills: first show kept (previously selected, always "added"), then current suggestions
  let pillsHtml = '';
  for (const s of kept) {
    pillsHtml += `<div class="ai-sug-pill added">\u2713 ${esc(s)}</div>`;
  }
  for (const s of allSugs) {
    const isAdded = added.has(s);
    pillsHtml += `<div class="ai-sug-pill ${isAdded ? 'added' : ''}" onclick="_wiz.addTabSuggestion('${escAttr(tabId)}','${escAttr(s)}')">${isAdded ? '\u2713 ' : '+ '}${esc(s)}${isAdded ? '<span class="sug-x">\u00D7</span>' : ''}</div>`;
  }
  return `<div class="ai-sug"><div class="ai-sug-hdr">\u2728 Suggestions<button class="ai-ref" onclick="_wiz.refreshSuggestions('${escAttr(tabId)}')" title="Get new suggestions">&#x21bb;</button></div><div class="acc-pills">${pillsHtml}</div></div>`;
}

function refreshSuggestions(tabId) {
  const cache = _tabSuggestCache[tabId];
  const allPreviouslyAdded = new Set();
  if (cache?.added) for (const item of cache.added) allPreviouslyAdded.add(item);
  if (cache?.keptFromPrev) for (const item of cache.keptFromPrev) allPreviouslyAdded.add(item);
  delete _tabSuggestCache[tabId];
  fetchTabSuggestion(tabId, allPreviouslyAdded, true).then(() => {
    const newCache = _tabSuggestCache[tabId];
    if (newCache && allPreviouslyAdded.size) {
      newCache.keptFromPrev = allPreviouslyAdded;
      renderDetails();
    }
  });
}

function closeWizard() {
  _wizSaveState();
  _cleanupWizStars();
  const root = document.getElementById('wiz-root');
  if (root) {
    const bk = root.querySelector('.backdrop');
    const modal = root.querySelector('.modal');
    if (bk) bk.classList.remove('open');
    if (modal) modal.classList.remove('open');
  }
  document.body.style.overflow = '';
  setTimeout(() => { removeDOM(); removeCSS(); }, 400);
}

/* ═══════════════════════════════════════
   GLOBAL EXPORTS (for inline onclick handlers)
   ═══════════════════════════════════════ */

window._wiz = {
  // Priorities
  togCat, togSub, showAddSub, addSubKey, rmCustomSub,
  // Details
  togPanel, showPanelAdd, addPanelKey, rmPanelCustom,
  togIntS2, rmIntS2, intKeyS2,
  togViewAll,
  togDetSection,
  refreshSuggestions,
  dismissS2Banner,
  // Rail / Review
  togRvCollapse, collapseAllRv, expandAllRv,
  rvRm, rvRmInt, rvShowAdd, rvAddKey,
  discoverAdd, discoverRm, addTabSuggestion,
  toggleRail, toggleDeepMode,
  // Build
  doBuild, finishWizard, restoreMainView,
  // Presets
  savePreset, loadPreset, deletePreset, togglePresetMenu,
  // Other
  skipToQuick, closeWizard, clearAll, setAtmosphere,
};

// Public API
window.openWizard = openWizard;
window.closeWizard = closeWizard;

// Set up keyboard once
setupKeyboard();

/* Button wiring removed — Generate/Suggest use their original
   settings.js implementations directly. Setup Wizard button
   calls openWizard() via onclick in HTML. */

})();
