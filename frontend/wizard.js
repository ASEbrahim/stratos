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

const CK = '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>';
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
   See initRv(), initRvWithAI(), renderS3() for the AI-only flow. */

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
  const role = document.getElementById('wiz-br')?.textContent?.trim() || '';
  const location = document.getElementById('wiz-bl')?.textContent?.trim() || '';
  const suggested = role + (location && location !== '\u2014' ? ' \u00B7 ' + location : '');
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
  // Update role/location display in header
  if (preset.role) {
    const br = document.getElementById('wiz-br');
    if (br) br.textContent = preset.role;
    const roleInput = document.getElementById('simple-role');
    if (roleInput) roleInput.value = preset.role;
    _currentDomain = classifyRole(preset.role);
  }
  if (preset.location) {
    const bl = document.getElementById('wiz-bl');
    if (bl) bl.textContent = preset.location;
    const locInput = document.getElementById('simple-location');
    if (locInput) locInput.value = preset.location;
  }
  _presetMenuOpen = false;
  renderPresetBar();
  // Re-render current step with loaded state
  if (step === 0) renderS1();
  else if (step === 1) { renderS2(); activeTab = null; renderS2(); }
  else { initRv(); renderS3(); }
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
  const el = document.getElementById('wiz-preset-bar');
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

  el.innerHTML = `<div class="preset-dd">
    <button class="preset-btn ${_presetMenuOpen ? 'open' : ''}" onclick="_wiz.togglePresetMenu()">
      <span class="preset-name">${hasPresets ? `\uD83D\uDCC2 ${names.length} saved profile${names.length > 1 ? 's' : ''}` : '\uD83D\uDCC2 No saved profiles'}</span>
      <span class="preset-chev">\u25BC</span>
    </button>
    <div class="preset-menu ${_presetMenuOpen ? 'open' : ''}">${menuHtml}</div>
  </div>`;
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
/* === Variable mapping: wizard → dashboard theme === */
.wiz-scope {
  --bg: var(--bg-primary);
  --card: var(--bg-panel-solid);
  --card-hover: var(--bg-hover);
  --accent: var(--accent);
  --accent-light: var(--accent-light);
  --accent-dim: var(--accent-bg);
  --accent-glow: var(--accent-border);
  --text: var(--text-primary);
  --text-dim: var(--text-secondary);
  --text-muted: var(--text-muted);
  --border: var(--border-strong);
  --border-hover: var(--border);
  --danger: #ef4444;
  --r: 14px; --rs: 10px; --pill: 999px;
  --ease: cubic-bezier(.4,0,.2,1);
  --spring: cubic-bezier(.34,1.56,.64,1);
  font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;
  color: var(--text);
}

/* === Backdrop + Modal === */
.wiz-scope .wiz-bk {
  position:fixed;inset:0;background:rgba(0,0,0,.55);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  z-index:9998;opacity:0;pointer-events:none;transition:opacity .35s ease;
}
.wiz-scope .wiz-bk.open { opacity:1;pointer-events:auto; }
.wiz-scope .wiz-modal {
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(.96);
  width:min(90vw,920px);max-height:85vh;height:85vh;
  background:var(--bg);border:1px solid rgba(255,255,255,.08);border-radius:20px;
  z-index:9999;display:flex;flex-direction:column;
  opacity:0;pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease),background-color .35s var(--ease),border-color .35s var(--ease),color .35s var(--ease),box-shadow .35s var(--ease);overflow:hidden;
  box-shadow:0 25px 80px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.04),0 0 60px var(--accent-border);
}
.wiz-scope .wiz-modal.open { opacity:1;pointer-events:auto;transform:translate(-50%,-50%) scale(1); }

/* === Header === */
.wiz-scope .hdr { padding:20px 36px 0;flex-shrink:0; }
.wiz-scope .hdr-top { display:flex;align-items:center;justify-content:space-between;margin-bottom:14px; }
.wiz-scope .brand { display:flex;align-items:center;gap:10px; }
.wiz-scope .brand svg { width:24px;height:24px;color:var(--accent-light); }
.wiz-scope .brand span { font-size:16px;font-weight:700;letter-spacing:-.02em;background:linear-gradient(135deg,var(--text-primary),var(--accent-light));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.wiz-scope .xbtn { width:34px;height:34px;border-radius:10px;border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.03);color:var(--text-muted);font-size:20px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .xbtn:hover { background:rgba(239,68,68,.1);color:#f87171;border-color:rgba(239,68,68,.2); }
.wiz-scope .badge { display:inline-flex;align-items:center;gap:8px;padding:7px 16px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:var(--pill);font-size:13px;color:var(--text-secondary);margin-bottom:14px; }
.wiz-scope .badge strong { color:var(--accent-light);font-weight:600; }
.wiz-scope .prog-t { width:100%;height:3px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden; }
.wiz-scope .prog-f { height:100%;background:linear-gradient(90deg,var(--accent),var(--accent-light));border-radius:3px;transition:width .5s var(--ease);box-shadow:0 0 8px var(--accent-border); }
.wiz-scope .dots { display:flex;align-items:center;justify-content:center;gap:8px;margin-top:14px;padding-bottom:4px; }
.wiz-scope .dot { display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-muted);transition:color .2s var(--ease); }
.wiz-scope .dot.a { color:var(--accent-light);font-weight:600; }
.wiz-scope .dot.d { color:var(--text-secondary); }
.wiz-scope .dot-c { width:24px;height:24px;border-radius:50%;border:2px solid rgba(255,255,255,.1);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;transition:opacity .25s var(--ease),transform .25s var(--ease),background-color .25s var(--ease),border-color .25s var(--ease),color .25s var(--ease),box-shadow .25s var(--ease);color:var(--text-muted); }
.wiz-scope .dot.a .dot-c { border-color:var(--accent);background:var(--accent);color:#fff;box-shadow:0 0 12px var(--accent-border); }
.wiz-scope .dot.d .dot-c { border-color:var(--accent);color:var(--accent);background:var(--accent-dim); }
.wiz-scope .dot-c svg { width:12px;height:12px;stroke:#fff;stroke-width:3;fill:none; }
.wiz-scope .conn { width:32px;height:2px;background:rgba(255,255,255,.06);border-radius:2px;transition:opacity .3s var(--ease),transform .3s var(--ease),background-color .3s var(--ease),border-color .3s var(--ease),color .3s var(--ease),box-shadow .3s var(--ease); }
.wiz-scope .conn.d { background:linear-gradient(90deg,var(--accent),var(--accent-light));box-shadow:0 0 6px var(--accent-border); }

/* === Body / Slides === */
.wiz-scope .bod { flex:1;overflow:hidden;position:relative; }
.wiz-scope .slides { display:flex;height:100%;transition:transform .45s var(--ease); }
.wiz-scope .sl { min-width:100%;height:100%;overflow-y:auto;padding:28px 40px 110px;scrollbar-width:thin;scrollbar-color:var(--border) transparent; }
.wiz-scope .sl::-webkit-scrollbar { width:5px; }
.wiz-scope .sl::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }
.wiz-scope .inn { max-width:820px;margin:0 auto; }

/* === Footer === */
.wiz-scope .ftr { flex-shrink:0;padding:16px 36px 20px;display:flex;justify-content:space-between;align-items:center;border-top:1px solid rgba(255,255,255,.06);background:var(--bg);gap:8px; }
.wiz-scope .wiz-build-row { display:flex;align-items:center;gap:16px; }
.wiz-scope .wiz-mode-toggle { display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;padding:6px 14px;border-radius:var(--pill);border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.02);transition:opacity .25s var(--ease),transform .25s var(--ease),background-color .25s var(--ease),border-color .25s var(--ease),color .25s var(--ease),box-shadow .25s var(--ease); }
.wiz-scope .wiz-mode-toggle:hover { border-color:var(--accent-border);background:rgba(255,255,255,.04); }
.wiz-scope .wiz-mode-toggle input { display:none; }
.wiz-scope .wiz-mode-slider { position:relative;width:38px;height:20px;background:rgba(255,255,255,.12);border-radius:10px;transition:background .25s var(--ease);flex-shrink:0; }
.wiz-scope .wiz-mode-slider::after { content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;background:#fff;border-radius:50%;transition:transform .25s var(--ease);box-shadow:0 1px 4px rgba(0,0,0,.4); }
.wiz-scope .wiz-mode-toggle input:checked + .wiz-mode-slider { background:var(--accent);box-shadow:0 0 12px var(--accent-border); }
.wiz-scope .wiz-mode-toggle input:checked + .wiz-mode-slider::after { transform:translateX(18px); }
.wiz-scope .wiz-mode-label { font-size:12px;color:var(--text-secondary);white-space:nowrap;min-width:100px;font-weight:500; }
.wiz-scope .wiz-clr { font-size:12px;padding:6px 12px;opacity:.6;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .wiz-clr:hover { opacity:1;color:var(--danger, #ef4444);border-color:var(--danger, #ef4444); }
.wiz-scope .btn { display:inline-flex;align-items:center;gap:8px;padding:12px 26px;border-radius:var(--pill);font-size:15px;font-weight:600;border:none;cursor:pointer;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease);font-family:inherit;outline:none; }
.wiz-scope .btn:active { transform:scale(.97); }
.wiz-scope .bp { background:linear-gradient(135deg,var(--accent),var(--accent-light));color:#fff;box-shadow:0 2px 12px var(--accent-border); }
.wiz-scope .bp:hover { background:linear-gradient(135deg,var(--accent-light),var(--accent));box-shadow:0 4px 24px var(--accent-border),0 0 0 1px var(--accent); }
.wiz-scope .bp:disabled { background:var(--card);color:var(--text-muted);cursor:not-allowed;box-shadow:none;border:1px solid var(--border); }
.wiz-scope .bp:disabled:active { transform:none; }
.wiz-scope .bg_ { background:none;color:var(--text-dim);padding:12px 16px;border:1px solid var(--border);border-radius:var(--pill); }
.wiz-scope .bg_:hover { color:var(--text);background:rgba(255,255,255,.05);border-color:var(--accent); }
.wiz-scope .bo { background:var(--card);color:var(--text);border:1px solid var(--border);padding:12px 26px; }
.wiz-scope .bo:hover { border-color:var(--border-hover);background:var(--card-hover); }

/* === Titles === */
.wiz-scope .title { font-size:26px;font-weight:700;letter-spacing:-.03em;margin-bottom:6px;background:linear-gradient(135deg,var(--text-primary) 40%,var(--accent-light));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.wiz-scope .sub { color:var(--text-dim);font-size:15px;margin-bottom:28px;line-height:1.5; }

/* === Step 1: Card Grid === */
.wiz-scope .card-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:16px;align-items:start; }
.wiz-scope .gcard {
  background:var(--card);border:1.5px solid var(--border);border-radius:16px;
  cursor:pointer;transition:opacity .3s var(--ease),transform .3s var(--ease),background-color .3s var(--ease),border-color .3s var(--ease),color .3s var(--ease),box-shadow .3s var(--ease);position:relative;
  user-select:none;-webkit-tap-highlight-color:transparent;overflow:hidden;
}
.wiz-scope .gcard:hover { border-color:var(--border-hover);background:var(--card-hover);transform:translateY(-2px); }
.wiz-scope .gcard.sel { border-color:var(--accent);background:var(--accent-dim);box-shadow:0 0 28px var(--accent-border),inset 0 0 24px var(--accent-bg);transform:translateY(0); }
.wiz-scope .gcard.sel:hover { background:var(--accent-dim);box-shadow:0 0 36px var(--accent-border);transform:translateY(0); }
.wiz-scope .gcard-head { padding:26px 24px 22px;display:flex;flex-direction:column; }
.wiz-scope .gcard-icon { font-size:38px;margin-bottom:14px;display:block;filter:drop-shadow(0 4px 12px rgba(0,0,0,.2)); }
.wiz-scope .gcard-name { font-size:17px;font-weight:700;letter-spacing:-.01em;margin-bottom:5px; }
.wiz-scope .gcard-desc { font-size:13px;color:var(--text-secondary);line-height:1.5; }
.wiz-scope .gcard-chk { position:absolute;top:14px;right:14px;width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent-light));display:flex;align-items:center;justify-content:center;opacity:0;transform:scale(.4);transition:opacity .3s var(--spring),transform .3s var(--spring),background-color .3s var(--spring),border-color .3s var(--spring),color .3s var(--spring),box-shadow .3s var(--spring);box-shadow:0 0 14px var(--accent-border); }
.wiz-scope .gcard.sel .gcard-chk { opacity:1;transform:scale(1); }
.wiz-scope .gcard-chk svg { width:13px;height:13px;stroke:#fff;stroke-width:3;fill:none; }
.wiz-scope .gcard-body { max-height:0;opacity:0;overflow:hidden;transition:max-height .45s var(--ease),opacity .3s var(--ease),padding .3s var(--ease);padding:0 24px; }
.wiz-scope .gcard.sel .gcard-body { max-height:320px;opacity:1;padding:0 24px 22px; }
.wiz-scope .gcard-body-sep { height:1px;background:rgba(255,255,255,.06);margin-bottom:12px; }
.wiz-scope .gcard-body-label { font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px; }
.wiz-scope .gcard-body-hint { font-size:12px;color:var(--text-muted);font-style:italic;padding-top:4px; }
.wiz-scope .gcard:not(.sel) { opacity:.65; }
.wiz-scope .gcard:not(.sel):hover { opacity:.9; }
.wiz-scope .gcard-tap { font-size:11px;color:var(--text-muted);text-align:center;padding:0 24px 16px;opacity:.8;letter-spacing:.01em; }

/* === Sub-pills (Step 1 cards) === */
.wiz-scope .spills { display:flex;flex-wrap:wrap;gap:8px;max-height:96px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.1) transparent;padding-right:4px; }
.wiz-scope .spills::-webkit-scrollbar { width:3px; }
.wiz-scope .spills::-webkit-scrollbar-thumb { background:rgba(255,255,255,.15);border-radius:3px; }
.wiz-scope .sp {
  display:inline-flex;align-items:center;gap:4px;padding:8px 16px;
  border-radius:var(--pill);font-size:13px;font-weight:500;
  background:rgba(255,255,255,.05);border:1.5px solid rgba(255,255,255,.1);
  cursor:pointer;transition:opacity .15s var(--ease),transform .15s var(--ease),background-color .15s var(--ease),border-color .15s var(--ease),color .15s var(--ease),box-shadow .15s var(--ease);white-space:nowrap;
}
.wiz-scope .sp:hover { border-color:rgba(255,255,255,.18);background:rgba(255,255,255,.08); }
.wiz-scope .sp:active { transform:scale(.94);transition-duration:.05s; }
.wiz-scope .sp.on { border-color:var(--accent);background:var(--accent-dim);color:var(--accent); }
.wiz-scope .sp.on:hover { background:var(--accent-glow); }
.wiz-scope .sp-add { border-style:dashed;color:var(--text-muted);border-color:rgba(255,255,255,.1); }
.wiz-scope .sp-add:hover { border-color:var(--accent);color:var(--accent); }
.wiz-scope .sp-x { font-size:14px;color:var(--text-muted);margin-left:2px;transition:color .15s; }
.wiz-scope .sp-x:hover { color:var(--danger); }
.wiz-scope .add-inp-card { background:rgba(255,255,255,.04);border:1.5px solid var(--accent);border-radius:var(--pill);padding:8px 14px;font-size:13px;color:var(--text);outline:none;font-family:inherit;width:130px;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .add-inp-card::placeholder { color:var(--text-muted); }
.wiz-scope .add-inp-card:focus { box-shadow:0 0 10px var(--accent-glow); }

/* === Step 2: Tab Bar === */
.wiz-scope .tab-wrap { position:relative;margin-bottom:28px; }
.wiz-scope .tab-bar { display:flex;flex-wrap:nowrap;gap:6px;padding:6px 0 10px;border-bottom:2px solid var(--border);overflow-x:auto;overflow-y:hidden;scrollbar-width:none;-ms-overflow-style:none;scroll-behavior:smooth;cursor:grab;user-select:none; }
.wiz-scope .tab-bar::-webkit-scrollbar { display:none; }
.wiz-scope .tab-bar.dragging { cursor:grabbing;scroll-behavior:auto; }
.wiz-scope .tab-bar.dragging .tab { pointer-events:none; }
.wiz-scope .tab-arr { position:absolute;top:0;bottom:2px;width:36px;z-index:3;display:flex;align-items:center;justify-content:center;border:none;color:var(--text-dim);font-size:20px;cursor:pointer;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease);opacity:0;pointer-events:none;font-family:inherit;line-height:1; }
.wiz-scope .tab-arr.show { opacity:1;pointer-events:auto; }
.wiz-scope .tab-arr:hover { color:var(--text); }
.wiz-scope .tab-arr-l { left:0;background:linear-gradient(to right,var(--bg) 50%,transparent);padding-right:6px; }
.wiz-scope .tab-arr-r { right:0;background:linear-gradient(to left,var(--bg) 50%,transparent);padding-left:6px; }
.wiz-scope .tab {
  display:flex;align-items:center;gap:8px;padding:10px 20px;
  font-size:14px;font-weight:500;color:var(--text-muted);cursor:pointer;
  transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease);border:1px solid var(--border);background:none;font-family:inherit;outline:none;
  position:relative;white-space:nowrap;flex-shrink:0;border-radius:var(--pill);
}
.wiz-scope .tab:hover { color:var(--text-dim);border-color:var(--accent);background:rgba(255,255,255,.03); }
.wiz-scope .tab.active { color:var(--accent);font-weight:600;border-color:var(--accent);background:var(--accent-dim);box-shadow:0 0 12px var(--accent-border); }
.wiz-scope .tab::after { display:none; }
.wiz-scope .tab.active::after { display:none; }
.wiz-scope .tab-icon { font-size:17px; }
.wiz-scope .tab-body { animation:wizFadeUp .3s ease; }
@keyframes wizFadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.wiz-scope .tab-dot { width:7px;height:7px;border-radius:50%;background:var(--accent);display:none;flex-shrink:0;box-shadow:0 0 6px var(--accent-glow); }
.wiz-scope .tab-dot.show { display:block; }

/* === Step 2 Sections === */
.wiz-scope .s2-section { margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid var(--border); }
.wiz-scope .s2-section:last-child { margin-bottom:0;padding-bottom:0;border-bottom:none; }
.wiz-scope .s2-sec-hdr { display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700;margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,.04); }
.wiz-scope .s2-sec-icon { font-size:20px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.3)); }
.wiz-scope .s2-label { font-size:13px;font-weight:600;color:var(--accent-light);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;margin-top:24px;opacity:.8; }
.wiz-scope .s2-label:first-of-type { margin-top:0; }
.wiz-scope .s2-hint { font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-muted);font-size:12px; }

/* === Pills === */
.wiz-scope .pills { display:flex;flex-wrap:wrap;gap:10px; }
.wiz-scope .pill {
  display:inline-flex;align-items:center;gap:6px;padding:11px 20px;
  border-radius:var(--pill);font-size:14px;font-weight:500;
  background:var(--card);border:1.5px solid var(--border);cursor:pointer;
  transition:opacity .15s var(--ease),transform .15s var(--ease),background-color .15s var(--ease),border-color .15s var(--ease),color .15s var(--ease),box-shadow .15s var(--ease);user-select:none;white-space:nowrap;
}
.wiz-scope .pill:hover { border-color:var(--border-hover);background:var(--card-hover); }
.wiz-scope .pill:active:not(.pill-add) { transform:scale(.94);transition-duration:.05s; }
.wiz-scope .pill.on { border-color:var(--accent);background:var(--accent-dim);color:var(--accent); }
.wiz-scope .pill.on:hover { background:var(--accent-glow); }
.wiz-scope .pill-x { font-size:16px;color:var(--text-muted);cursor:pointer;transition:color .15s;margin-left:2px; }
.wiz-scope .pill-x:hover { color:var(--danger); }
.wiz-scope .pill-add { border-style:dashed;color:var(--text-muted); }
.wiz-scope .pill-add:hover { border-color:var(--accent);color:var(--accent); }
.wiz-scope .pill-sug { border-style:dashed;opacity:.5;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .pill-sug:hover { opacity:.8;border-color:var(--border-hover); }
.wiz-scope .pill-sug.on { opacity:1;border-style:solid; }
.wiz-scope .add-inp { background:var(--card);border:1.5px solid var(--accent);border-radius:var(--pill);padding:11px 16px;font-size:14px;color:var(--text);outline:none;font-family:inherit;width:160px;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .add-inp::placeholder { color:var(--text-muted); }
.wiz-scope .add-inp:focus { box-shadow:0 0 12px var(--accent-glow); }

/* === Interest input (Step 2) === */
.wiz-scope .int-wrap { display:flex;gap:10px;margin-bottom:16px; }
.wiz-scope .int-inp { flex:1;padding:12px 18px;background:var(--card);border:1.5px solid var(--border);border-radius:var(--pill);font-size:14px;color:var(--text);outline:none;font-family:inherit;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .int-inp:focus { border-color:var(--accent);box-shadow:0 0 12px var(--accent-glow); }
.wiz-scope .int-inp::placeholder { color:var(--text-muted); }

/* === Step 3: Review === */
.wiz-scope .s3-top { display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px; }
.wiz-scope .s3-toggle { background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:var(--pill);padding:6px 14px;font-size:12px;font-weight:500;color:var(--text-muted);cursor:pointer;font-family:inherit;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease);white-space:nowrap;margin-top:6px; }
.wiz-scope .s3-toggle:hover { border-color:var(--accent-border);color:var(--accent-light);background:rgba(255,255,255,.03); }
.wiz-scope .rv { background:var(--card);border:1px solid var(--border);border-radius:var(--r);margin-bottom:14px;transition:opacity .25s var(--ease),transform .25s var(--ease),background-color .25s var(--ease),border-color .25s var(--ease),color .25s var(--ease),box-shadow .25s var(--ease);overflow:hidden; }
.wiz-scope .rv:hover { border-color:var(--accent-border);box-shadow:0 2px 12px rgba(0,0,0,.15); }
.wiz-scope .rv-h { display:flex;align-items:center;gap:12px;padding:20px 24px;cursor:pointer;user-select:none;transition:background .15s var(--ease); }
.wiz-scope .rv-h:hover { background:rgba(255,255,255,.02); }
.wiz-scope .rv-i { font-size:24px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.3)); }
.wiz-scope .rv-t { font-size:17px;font-weight:700; }
.wiz-scope .rv-chev { margin-left:auto;font-size:16px;color:var(--text-muted);transition:transform .3s var(--ease);width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06); }
.wiz-scope .rv.collapsed .rv-chev { transform:rotate(-90deg); }
.wiz-scope .rv-body { padding:0 24px 22px;transition:max-height .4s var(--ease),opacity .3s var(--ease),padding .3s var(--ease);max-height:2000px;opacity:1;overflow:hidden; }
.wiz-scope .rv.collapsed .rv-body { max-height:0;opacity:0;padding:0 24px; }
.wiz-scope .rv-sub { margin-top:16px; }
.wiz-scope .rv-sub:first-child { margin-top:0; }
.wiz-scope .rv-sub-hdr { display:flex;align-items:center;gap:7px;margin-bottom:10px; }
.wiz-scope .rv-sub-icon { font-size:16px; }
.wiz-scope .rv-sub-name { font-size:14px;font-weight:600;color:var(--text-dim); }
.wiz-scope .rv-sub-ctx { font-size:12px;color:var(--text-muted);margin-bottom:10px;line-height:1.4; }
.wiz-scope .rv-lbl { font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent-light);margin-bottom:8px;font-weight:600;opacity:.7; }
.wiz-scope .rv-pills { display:flex;flex-wrap:wrap;gap:8px; }
.wiz-scope .rv-p { display:inline-flex;align-items:center;gap:4px;padding:7px 14px;border-radius:var(--pill);font-size:13px;font-weight:500;background:var(--accent-dim);color:var(--accent);border:1px solid var(--accent-glow);cursor:pointer;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .rv-p:hover { border-color:var(--danger);color:var(--danger);background:rgba(239,68,68,.1); }
.wiz-scope .rv-p:hover .rv-x { opacity:1; }
.wiz-scope .rv-x { font-size:14px;opacity:0;transition:opacity .15s;margin-left:1px; }
.wiz-scope .rv-a { display:inline-flex;align-items:center;gap:4px;padding:7px 14px;border-radius:var(--pill);font-size:13px;color:var(--text-muted);border:1px dashed var(--border);cursor:pointer;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease);background:none;font-family:inherit; }
.wiz-scope .rv-a:hover { border-color:var(--accent);color:var(--accent); }
.wiz-scope .rv-tag { font-size:10px;color:var(--text-muted);background:rgba(255,255,255,.05);padding:2px 7px;border-radius:4px;margin-left:3px;white-space:nowrap;letter-spacing:.01em;pointer-events:none; }

/* === Discover More === */
.wiz-scope .disc { background:rgba(255,255,255,.02);border:1px dashed var(--accent-border);border-radius:var(--r);padding:22px 24px;margin-top:24px; }
.wiz-scope .disc-h { display:flex;align-items:center;gap:8px;font-size:15px;font-weight:600;color:var(--accent-light);margin-bottom:4px; }
.wiz-scope .disc-s { font-size:12px;color:var(--text-muted);margin-bottom:16px;line-height:1.5; }
.wiz-scope .disc-pills { display:flex;flex-wrap:wrap;gap:10px; }
.wiz-scope .disc-p { display:inline-flex;align-items:center;gap:6px;padding:9px 18px;border-radius:var(--pill);font-size:13px;font-weight:500;background:rgba(255,255,255,.03);border:1.5px dashed var(--border);color:var(--text-dim);cursor:pointer;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .disc-p:hover { border-color:var(--accent);color:var(--accent);background:var(--accent-dim); }
.wiz-scope .disc-p.added { border-style:solid;border-color:var(--accent);color:var(--accent);background:var(--accent-dim);opacity:.55;pointer-events:none; }
.wiz-scope .disc-tag { font-size:10px;color:var(--text-muted);background:rgba(255,255,255,.06);padding:2px 8px;border-radius:5px; }

/* === Skip Link === */
.wiz-scope .skip { display:block;text-align:center;margin-top:28px;font-size:13px;color:var(--text-muted);cursor:pointer;background:none;border:1px solid transparent;border-radius:var(--pill);font-family:inherit;transition:opacity .25s var(--ease),transform .25s var(--ease),background-color .25s var(--ease),border-color .25s var(--ease),color .25s var(--ease),box-shadow .25s var(--ease);padding:10px 20px; }
.wiz-scope .skip:hover { color:var(--accent);border-color:var(--accent-border);background:rgba(255,255,255,.02); }

/* === AI Suggestions (Stage 4) === */
.wiz-scope .ai-sug { margin-top:24px;padding:16px 20px;background:rgba(255,255,255,.02);border:1px dashed var(--accent-border);border-radius:var(--r); }
.wiz-scope .ai-sug-hdr { display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent-light);margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em;opacity:.8; }
.wiz-scope .ai-sug-spin { width:12px;height:12px;border:2px solid rgba(255,255,255,.1);border-top-color:var(--accent);border-right-color:var(--accent-light);border-radius:50%;animation:wizSpin .6s linear infinite; }
@keyframes wizSpin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
.wiz-scope .ai-sug-pill { display:inline-flex;align-items:center;gap:4px;padding:8px 16px;border-radius:var(--pill);font-size:13px;font-weight:500;background:rgba(255,255,255,.03);border:1.5px dashed var(--border);color:var(--text-dim);cursor:pointer;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .ai-sug-pill:hover { border-color:var(--accent);color:var(--accent);background:var(--accent-dim); }
.wiz-scope .ai-sug-pill.added { opacity:.4;pointer-events:none;border-style:solid;border-color:var(--accent);color:var(--accent); }
.wiz-scope .ai-sug-pill.shimmer { width:100px;height:34px;background:linear-gradient(90deg,rgba(255,255,255,.03) 25%,rgba(255,255,255,.08) 50%,rgba(255,255,255,.03) 75%);background-size:200% 100%;animation:wizShimmer 1.5s ease infinite;border-color:transparent;pointer-events:none; }
@keyframes wizShimmer { from{background-position:200% 0} to{background-position:-200% 0} }

/* Step 3: Decisive pills — predefined choices that steer AI suggestions */
.wiz-scope .pill.pill-decisive { border-color:var(--accent-border);font-weight:600; }
.wiz-scope .pill.pill-decisive.on { background:var(--accent);color:#fff;border-color:var(--accent); }
.wiz-scope .pill.pill-decisive.on:hover { background:var(--accent-light);border-color:var(--accent-light); }

/* Step 3: Banner */
.wiz-scope .s2-banner { display:flex;align-items:center;gap:10px;padding:12px 18px;background:rgba(255,255,255,.02);border:1px solid var(--accent-border);border-radius:var(--r);margin-bottom:20px;font-size:13px;color:var(--text-dim);line-height:1.4; }
.wiz-scope .s2-banner-icon { font-size:18px;flex-shrink:0; }
.wiz-scope .s2-banner-x { margin-left:auto;cursor:pointer;color:var(--text-muted);font-size:16px;padding:2px 6px;border-radius:4px;transition:color .15s; }
.wiz-scope .s2-banner-x:hover { color:var(--accent); }

/* Step 3: Flash indicator on suggestions header */
.wiz-scope .ai-sug-hdr.flash { animation:wizFlash .6s ease; }
@keyframes wizFlash { 0%{opacity:1} 30%{opacity:.4} 60%{opacity:1} 100%{opacity:1} }

/* === Loading / Done === */
.wiz-scope .ld { display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100%;text-align:center;background:var(--bg);padding:20px; }
.wiz-scope .wiz-ring {
  width:64px;height:64px;border-radius:50%;
  border:4px solid rgba(255,255,255,.08);
  border-top-color:var(--accent);border-right-color:var(--accent-light);
  margin-bottom:28px;animation:wizRingSpin 1s linear infinite;box-sizing:border-box;
  box-shadow:0 0 24px var(--accent-border),inset 0 0 12px rgba(255,255,255,.02);
}
@keyframes wizRingSpin { to{transform:rotate(360deg)} }
.wiz-scope .ld-t { font-size:22px;font-weight:700;margin-bottom:8px;background:linear-gradient(135deg,var(--text-primary),var(--accent-light));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.wiz-scope .ld-s { font-size:14px;color:var(--text-dim);margin-bottom:0; }
.wiz-scope .ld-bar { width:220px;height:4px;background:rgba(255,255,255,.06);border-radius:3px;margin-top:18px;overflow:hidden; }
.wiz-scope .ld-bar-fill { height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--accent-light));border-radius:3px;transition:width .8s ease;box-shadow:0 0 8px var(--accent-border); }
.wiz-scope .ld-list { margin-top:28px;display:flex;flex-direction:column;gap:16px;text-align:left;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:14px;padding:20px 24px; }
.wiz-scope .ls { display:flex;align-items:center;gap:14px;font-size:14px;color:var(--text-muted);transition:opacity .4s ease,transform .4s ease,background-color .4s ease,border-color .4s ease,color .4s ease,box-shadow .4s ease;opacity:0;transform:translateY(6px);animation:wizStepIn .4s ease forwards; }
.wiz-scope .ls:nth-child(1) { animation-delay:.1s; }
.wiz-scope .ls:nth-child(2) { animation-delay:.3s; }
.wiz-scope .ls:nth-child(3) { animation-delay:.5s; }
@keyframes wizStepIn { to{opacity:1;transform:translateY(0)} }
.wiz-scope .ls.on { color:var(--text-primary);font-weight:600; }
.wiz-scope .ls.ok { color:var(--accent-light);font-weight:500; }
.wiz-scope .ls-d {
  width:26px;height:26px;border-radius:50%;border:2.5px solid rgba(255,255,255,.12);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  transition:opacity .3s ease,transform .3s ease,background-color .3s ease,border-color .3s ease,color .3s ease,box-shadow .3s ease;box-sizing:border-box;
}
.wiz-scope .ls.on .ls-d {
  border-color:transparent;
  border-top-color:var(--accent);border-right-color:var(--accent-light);border-bottom-color:var(--accent);
  animation:wizRingSpin .8s linear infinite;
  box-shadow:0 0 10px var(--accent-border);
}
.wiz-scope .ls.ok .ls-d {
  border-color:var(--accent);background:var(--accent);
  animation:wizCheckPop .35s var(--spring);
  box-shadow:0 0 12px var(--accent-border);
}
@keyframes wizCheckPop { from{transform:scale(.5);opacity:.5} to{transform:scale(1);opacity:1} }
.wiz-scope .ls-d svg { width:13px;height:13px;stroke:#fff;stroke-width:3;fill:none;opacity:0;transition:opacity .3s; }
.wiz-scope .ls.ok .ls-d svg { opacity:1; }
.wiz-scope .ls-sp { display:none; }
.wiz-scope .done { display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100%;text-align:center;animation:wizFadeUp .5s ease;background:var(--bg);padding:20px; }
.wiz-scope .done-c {
  width:76px;height:76px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent-light));
  display:flex;align-items:center;justify-content:center;margin-bottom:24px;
  animation:wizPop .5s var(--spring);
  box-shadow:0 0 40px var(--accent-border),0 0 80px rgba(139,92,246,.08);
}
@keyframes wizPop { from{transform:scale(0);opacity:0} to{transform:scale(1);opacity:1} }
.wiz-scope .done-c svg { width:34px;height:34px;stroke:#fff;stroke-width:2.5;fill:none; }
.wiz-scope .done-t { font-size:26px;font-weight:700;margin-bottom:10px;background:linear-gradient(135deg,var(--text-primary),var(--accent-light));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.wiz-scope .done-s { font-size:15px;color:var(--text-dim);margin-bottom:32px;max-width:400px;line-height:1.6; }
.wiz-scope .done-c.err { background:linear-gradient(135deg,#ef4444,#f87171); box-shadow:0 0 40px rgba(239,68,68,.2); }
.wiz-scope .done-c.err ~ .done-t { background:linear-gradient(135deg,#fca5a5,#f87171);-webkit-background-clip:text;background-clip:text; }
.wiz-scope .done-c.err ~ .done-s { color:var(--text-secondary); }

/* === View All toggle (Issue 4) === */
.wiz-scope .va-tog { background:none;border:none;color:var(--text-muted);font-size:12px;cursor:pointer;padding:4px 8px;font-family:inherit;transition:color .2s var(--ease); }
.wiz-scope .va-tog:hover { color:var(--accent); }

/* === Refresh suggestions button (Issue 5) === */
.wiz-scope .ai-ref { background:none;border:1px solid var(--border);border-radius:6px;color:var(--text-muted);font-size:14px;cursor:pointer;padding:2px 8px;font-family:inherit;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease);margin-left:auto;line-height:1; }
.wiz-scope .ai-ref:hover { color:var(--accent);border-color:var(--accent); }
.wiz-scope .ai-ref.spin { animation:wizRefSpin .6s linear infinite; }
@keyframes wizRefSpin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

/* === Collapsible sections (Issue 6) === */
.wiz-scope .s2-sec-hdr.coll { cursor:pointer;user-select:none; }
.wiz-scope .s2-sec-hdr.coll:hover { color:var(--accent); }
.wiz-scope .s2-sec-chev { margin-left:auto;font-size:14px;color:var(--text-muted);transition:transform .2s var(--ease); }
.wiz-scope .s2-sec-body { transition:max-height .35s var(--ease),opacity .25s var(--ease);max-height:2000px;opacity:1;overflow:hidden; }
.wiz-scope .s2-sec-body.collapsed { max-height:0;opacity:0;overflow:hidden; }
.wiz-scope .s2-toolbar { display:flex;justify-content:flex-end;margin-bottom:12px; }
.wiz-scope .s2-coll-btn { background:none;border:1px solid var(--border);border-radius:var(--pill);padding:4px 12px;font-size:11px;font-weight:500;color:var(--text-muted);cursor:pointer;font-family:inherit;transition:opacity .2s var(--ease),transform .2s var(--ease),background-color .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease),box-shadow .2s var(--ease); }
.wiz-scope .s2-coll-btn:hover { border-color:var(--border-hover);color:var(--text); }

/* === Profile Presets === */
.wiz-scope .preset-bar { display:flex;align-items:center;gap:8px;margin-top:6px; }
.wiz-scope .preset-dd { position:relative;flex:1;min-width:0; }
.wiz-scope .preset-btn { display:flex;align-items:center;gap:6px;width:100%;padding:5px 10px;background:var(--card);border:1px solid var(--border);border-radius:var(--pill);color:var(--text-dim);font-size:12px;cursor:pointer;font-family:inherit;transition:border-color .2s; }
.wiz-scope .preset-btn:hover { border-color:var(--accent); }
.wiz-scope .preset-btn .preset-name { flex:1;text-align:left;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
.wiz-scope .preset-btn .preset-chev { font-size:10px;opacity:.6;transition:transform .2s; }
.wiz-scope .preset-btn.open .preset-chev { transform:rotate(180deg); }
.wiz-scope .preset-menu { display:none;position:absolute;top:calc(100% + 4px);left:0;right:0;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:4px;z-index:10;max-height:200px;overflow-y:auto;box-shadow:0 8px 24px rgba(0,0,0,.3); }
.wiz-scope .preset-menu.open { display:block; }
.wiz-scope .preset-item { display:flex;align-items:center;padding:7px 10px;border-radius:7px;cursor:pointer;font-size:12px;color:var(--text);transition:background .15s; }
.wiz-scope .preset-item:hover { background:var(--card-hover); }
.wiz-scope .preset-item .preset-item-name { flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
.wiz-scope .preset-item .preset-item-date { font-size:10px;color:var(--text-muted);margin-left:8px;white-space:nowrap; }
.wiz-scope .preset-item .preset-del { display:none;background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:14px;padding:0 2px;margin-left:4px;line-height:1;font-family:inherit; }
.wiz-scope .preset-item:hover .preset-del { display:inline; }
.wiz-scope .preset-item .preset-del:hover { color:#f43f5e; }
.wiz-scope .preset-empty { padding:10px;text-align:center;font-size:12px;color:var(--text-muted);font-style:italic; }
.wiz-scope .preset-save { display:flex;align-items:center;gap:4px;padding:5px 12px;background:none;border:1px solid var(--border);border-radius:var(--pill);color:var(--text-dim);font-size:12px;cursor:pointer;font-family:inherit;white-space:nowrap;transition:opacity .2s,transform .2s,background-color .2s,border-color .2s,color .2s,box-shadow .2s; }
.wiz-scope .preset-save:hover { border-color:var(--accent);color:var(--accent); }

/* === Utility === */
.wiz-scope .wiz-hidden { display:none!important; }
.wiz-scope .no-anim .gcard-body, .wiz-scope .no-anim .gcard, .wiz-scope .no-anim .gcard-chk { transition:none!important; }

/* === Responsive === */
@media(max-width:750px) {
  .wiz-scope .card-grid { grid-template-columns:repeat(2,1fr); }
  .wiz-scope .sl { padding:24px 24px 100px; }
  .wiz-scope .hdr { padding:18px 24px 0; }
  .wiz-scope .ftr { padding:14px 24px 18px;flex-wrap:wrap;gap:8px; }
  .wiz-scope .wiz-build-row { flex-wrap:wrap;gap:10px;justify-content:center;width:100%; }
  .wiz-scope .title { font-size:22px; }
  .wiz-scope .gcard-head { padding:20px 18px 16px; }
  .wiz-scope .gcard-icon { font-size:32px;margin-bottom:10px; }
}
@media(max-width:540px) {
  .wiz-scope .wiz-modal { width:100vw;height:100vh;max-height:100vh;border-radius:0; }
  .wiz-scope .card-grid { grid-template-columns:1fr; }
  .wiz-scope .sl { padding:20px 18px 100px; }
  .wiz-scope .hdr { padding:16px 18px 0; }
  .wiz-scope .ftr { padding:12px 18px 16px;flex-wrap:wrap;gap:6px;justify-content:center; }
  .wiz-scope .dot > span { display:none; }
  .wiz-scope .tab { padding:8px 14px;font-size:13px; }
  .wiz-scope .wiz-build-row { flex-direction:column;align-items:stretch;width:100%; }
  .wiz-scope .wiz-mode-toggle { justify-content:center; }
  .wiz-scope .wiz-build-row .btn { width:100%;justify-content:center; }
  .wiz-scope .title { font-size:20px; }
  .wiz-scope .sub { font-size:14px;margin-bottom:20px; }
  .wiz-scope .ld-t { font-size:18px; }
  .wiz-scope .done-t { font-size:22px; }
  .wiz-scope .done-s { font-size:14px; }
  .wiz-scope .badge { font-size:12px;padding:5px 12px; }
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
  wrapper.innerHTML = `
    <div class="wiz-bk" id="wiz-bk" onclick="_wiz.closeWizard()"></div>
    <div class="wiz-modal" id="wiz-modal">
      <div class="hdr">
        <div class="hdr-top">
          <div class="brand">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            <span>StratOS</span>
          </div>
          <button class="xbtn" onclick="_wiz.closeWizard()" title="Close">&times;</button>
        </div>
        <div class="badge"><span>&#128100;</span><strong id="wiz-br">${esc(role)}</strong><span>&middot;</span><span id="wiz-bl">${esc(location || '—')}</span></div>
        <div class="preset-bar" id="wiz-preset-bar"></div>
        <div class="prog-t"><div class="prog-f" id="wiz-prog"></div></div>
        <div class="dots" id="wiz-dots"></div>
      </div>
      <div class="bod">
        <div class="slides" id="wiz-slides">
          <div class="sl" id="wiz-s1"><div class="inn" id="wiz-s1i"></div></div>
          <div class="sl" id="wiz-s2"><div class="inn" id="wiz-s2i"></div></div>
          <div class="sl" id="wiz-s3"><div class="inn" id="wiz-s3i"></div></div>
        </div>
      </div>
      <div class="ftr" id="wiz-ftr">
        <button class="btn bg_" id="wiz-bbk" onclick="_wiz.goBack()">&#8592; Back</button>
        <button class="btn bg_ wiz-clr" id="wiz-clr" onclick="_wiz.clearAll()" title="Reset all selections">Clear all</button>
        <button class="preset-save wiz-hidden" id="wiz-psave" onclick="_wiz.savePreset()" title="Save this configuration as a named preset">&#x1F4BE; Save profile</button>
        <button class="btn bp" id="wiz-bnx" onclick="_wiz.goNext()">Next &#8594;</button>
        <div class="wiz-hidden wiz-build-row" id="wiz-build-row">
          <label class="wiz-mode-toggle" title="Deep thinking uses a larger model for higher quality — takes longer">
            <input type="checkbox" id="wiz-deep-mode">
            <span class="wiz-mode-slider"></span>
            <span class="wiz-mode-label" id="wiz-mode-label">Quick (~1 min)</span>
          </label>
          <button class="btn bp" id="wiz-bbuild" onclick="_wiz.doBuild()">&#x2728; Build my feed</button>
        </div>
      </div>
    </div>`;
  document.body.appendChild(wrapper);

  // Wire up deep-mode toggle label
  const deepCb = document.getElementById('wiz-deep-mode');
  if (deepCb) deepCb.addEventListener('change', () => {
    const lbl = document.getElementById('wiz-mode-label');
    if (lbl) lbl.textContent = deepCb.checked ? 'Deep (~5 min)' : 'Quick (~1 min)';
  });
}

function removeDOM() {
  const el = document.getElementById('wiz-root');
  if (el) el.remove();
}

/* ═══════════════════════════════════════
   NAVIGATION
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

function goTo(s) {
  step = s;
  const slides = document.getElementById('wiz-slides');
  const prog = document.getElementById('wiz-prog');
  if (slides) slides.style.transform = `translateX(-${s * 100}%)`;
  if (prog) prog.style.width = `${((s + 1) / 3) * 100}%`;
  renderDots();
  const ftr = document.getElementById('wiz-ftr');
  const bk = document.getElementById('wiz-bbk');
  const nx = document.getElementById('wiz-bnx');
  const clr = document.getElementById('wiz-clr');
  const psave = document.getElementById('wiz-psave');
  const buildRow = document.getElementById('wiz-build-row');
  if (ftr) ftr.classList.remove('wiz-hidden');
  // Back always visible (closes wizard on step 0)
  if (clr) clr.classList.toggle('wiz-hidden', s === 2);
  if (psave) psave.classList.toggle('wiz-hidden', s !== 2);
  if (nx) nx.classList.toggle('wiz-hidden', s === 2);
  if (buildRow) buildRow.classList.toggle('wiz-hidden', s !== 2);
  if (s === 0) { renderS1(); }
  else if (s === 1) { renderS2(); }
  else { rvCollapsed.clear(); initRvWithAI(); }
  updNx();
  const slide = document.getElementById(`wiz-s${s + 1}`);
  if (slide) slide.scrollTo(0, 0);
}

function goNext() {
  if (step === 0) { const t = getActiveTabs(); if (!t.length || (t.length === 1 && t[0].type === 'interests' && !interestTopics.length)) { goTo(2); return; } goTo(1); return; }
  if (step === 1) { goTo(2); return; }
  if (step === 2) { doBuild(); return; }
}

function goBack() {
  if (step === 2) { const t = getActiveTabs(); if (!t.length) { goTo(0); return; } goTo(1); return; }
  if (step > 0) goTo(step - 1);
  else closeWizard();
}

function updNx() {
  const nx = document.getElementById('wiz-bnx');
  if (nx) nx.disabled = (step === 0 && selCats.size === 0);
}

function renderDots() {
  const el = document.getElementById('wiz-dots');
  if (!el) return;
  el.innerHTML = STEP_NAMES.map((n, i) => {
    const c = i === step ? 'a' : i < step ? 'd' : '';
    const d = i < step ? CK : (i + 1);
    const cn = i < STEP_NAMES.length - 1 ? `<div class="conn ${i < step ? 'd' : ''}"></div>` : '';
    return `<div class="dot ${c}"><div class="dot-c">${d}</div><span>${n}</span></div>${cn}`;
  }).join('');
}

/* ═══════════════════════════════════════
   STEP 1 — PRIORITIES
   ═══════════════════════════════════════ */

function renderS1() {
  const el = document.getElementById('wiz-s1i');
  if (!el) return;
  el.innerHTML = `
    <h1 class="title">What matters to you?</h1>
    <p class="sub">Select your areas of interest, then pick focus areas within each.</p>
    <div class="card-grid">${CATS.map(c => renderCard(c)).join('')}</div>
    <button class="skip" onclick="_wiz.skipToQuick()">Not sure? Skip to quick setup \u2192</button>`;
}

function renderCard(c) {
  const sel = selCats.has(c.id);
  let bodyHTML = '';
  if (c.dynamic) {
    bodyHTML = `<div class="gcard-body"><div class="gcard-body-sep"></div><div class="gcard-body-hint">You'll customize these in the next step \u2192</div></div>`;
  } else {
    const allSubs = [...c.subs, ...customSubs[c.id]];
    const subsHTML = allSubs.map(s => {
      const on = selSubs[c.id].has(s.id);
      const isC = customSubs[c.id].some(cs => cs.id === s.id);
      return `<div class="sp ${on ? 'on' : ''}" onclick="event.stopPropagation();_wiz.togSub('${c.id}','${s.id}',this)">
        ${s.name}${isC ? `<span class="sp-x" onclick="event.stopPropagation();_wiz.rmCustomSub('${c.id}','${s.id}')">&times;</span>` : ''}
      </div>`;
    }).join('');
    bodyHTML = `<div class="gcard-body">
      <div class="gcard-body-sep"></div>
      <div class="gcard-body-label">Focus areas</div>
      <div class="spills">${subsHTML}
        <span id="wiz-saw-${c.id}" style="display:none" onclick="event.stopPropagation()"><input class="add-inp-card" id="wiz-sai-${c.id}" placeholder="Type & Enter" onkeydown="_wiz.addSubKey(event,'${c.id}')"></span>
        <div class="sp sp-add" id="wiz-sab-${c.id}" onclick="event.stopPropagation();_wiz.showAddSub('${c.id}')">+ Add</div>
      </div>
    </div>`;
  }
  const tapHint = sel ? '' : `<div class="gcard-tap">Tap to add</div>`;
  return `<div class="gcard ${sel ? 'sel' : ''}" onclick="_wiz.togCat('${c.id}')">
    <div class="gcard-chk">${CK}</div>
    <div class="gcard-head"><span class="gcard-icon">${c.icon}</span><div class="gcard-name">${c.name}</div><div class="gcard-desc">${c.desc}</div></div>
    ${bodyHTML}${tapHint}
  </div>`;
}

function togCat(id) {
  if (selCats.has(id)) { selCats.delete(id); if (id !== 'interests') selSubs[id].clear(); else interestTopics = []; }
  else { selCats.add(id); selSubs[id] = selSubs[id] && selSubs[id].size ? selSubs[id] : new Set(); }
  _rvItemsCache = null; // Invalidate Step 3 cache — sections changed
  renderS1(); updNx(); _wizSaveState();
}

function togSub(cid, sid, el) {
  selSubs[cid].has(sid) ? selSubs[cid].delete(sid) : selSubs[cid].add(sid);
  if (selSubs[cid].has(sid) && PANELS[sid] && !panelSel[sid]) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of PANELS[sid].qs) { panelCustom[sid][q.id] = []; panelSel[sid][q.id] = q.type === 's' && q.def ? new Set([q.def]) : new Set(q.defs || []); }
  }
  _rvItemsCache = null; // Invalidate Step 3 cache — sections changed
  if (el) el.classList.toggle('on', selSubs[cid].has(sid));
  _wizSaveState();
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
    renderS1(); _wizSaveState();
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
  renderS1(); _wizSaveState();
}

function clearAll() {
  initState(); // Reset all selections to defaults
  _wizClearState(); // Clear localStorage
  _tabSuggestCache = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  if (_suggestAbortCtrl) { _suggestAbortCtrl.abort(); _suggestAbortCtrl = null; }
  _s2BannerDismissed = false;
  if (step === 0) renderS1();
  else if (step === 1) { renderS1(); goTo(0); }
  else { goTo(0); renderS1(); }
  updNx();
  if (typeof showToast === 'function') showToast('All selections cleared', 'info');
}

/* ═══════════════════════════════════════
   STEP 2 — DETAILS
   ═══════════════════════════════════════ */

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

function renderS2() {
  const tabs = getActiveTabs();
  const el = document.getElementById('wiz-s2i');
  if (!el) return;
  if (!tabs.length) {
    el.innerHTML = `<h1 class="title">Let's get specific</h1><p class="sub">Tell us more so we can dial in your feed.</p>
      <div style="text-align:center;padding:60px 20px;color:var(--text-muted)"><p style="font-size:16px">No detail panels needed.</p><small style="font-size:13px">Click Next to see your generated feed.</small></div>`;
    return;
  }
  if (!activeTab || !tabs.some(t => t.id === activeTab)) activeTab = tabs[0].id;
  const bannerH = _s2BannerDismissed ? '' : `<div class="s2-banner"><span class="s2-banner-icon">&#x2728;</span><span><strong>Bold selections steer the AI</strong> &mdash; pick your career stage and preferences to get personalized suggestions below.</span><span class="s2-banner-x" onclick="_wiz.dismissS2Banner()">&times;</span></div>`;
  el.innerHTML = `
    <h1 class="title">Let's dial it in</h1>
    <p class="sub">Configure each area. Click the tabs to switch between them.</p>
    ${bannerH}
    <div class="tab-wrap">
      <button class="tab-arr tab-arr-l" id="wiz-tab-arr-l" onclick="_wiz.scrollTabDir(-1)" title="Scroll left">\u2039</button>
      <div class="tab-bar" id="wiz-tab-bar">${tabs.map(t => {
        const done = tabHasSelections(t);
        return `<button class="tab ${t.id === activeTab ? 'active' : ''}" onclick="_wiz.switchTab('${t.id}')" data-tid="${t.id}">
          <span class="tab-icon">${t.icon}</span><span>${t.name}</span><span class="tab-dot ${done ? 'show' : ''}"></span>
        </button>`;
      }).join('')}
      </div>
      <button class="tab-arr tab-arr-r" id="wiz-tab-arr-r" onclick="_wiz.scrollTabDir(1)" title="Scroll right">\u203A</button>
    </div>
    <div class="tab-body" id="wiz-tab-body"></div>`;
  renderTabBody();
  initTabScroll();
}

function getSubIcon(sid) { return PANELS[sid]?.icon || '\uD83D\uDCCC'; }

function switchTab(id) { activeTab = id; renderS2(); }

function renderTabBody() {
  const el = document.getElementById('wiz-tab-body');
  if (!el || !activeTab) { if (el) el.innerHTML = ''; return; }
  const tabs = getActiveTabs();
  const tab = tabs.find(t => t.id === activeTab);
  if (!tab) { el.innerHTML = ''; return; }

  if (tab.type === 'interests') {
    const itemsH = interestTopics.map(t => `<div class="pill on">${esc(t)}<span class="pill-x" onclick="event.stopPropagation();_wiz.rmIntS2('${escAttr(t)}')">&times;</span></div>`).join('');
    const sugH = INTEREST_SUGGESTIONS.map(s => {
      const on = interestTopics.includes(s);
      return `<div class="pill pill-sug ${on ? 'on' : ''}" onclick="_wiz.togIntS2('${escAttr(s)}')">${esc(s)}</div>`;
    }).join('');
    el.innerHTML = `
      <div class="s2-label">What do you follow? <span class="s2-hint">\u00B7 Type a topic and press Enter</span></div>
      <div class="int-wrap"><input class="int-inp" id="wiz-int-inp" placeholder="e.g. Quantum Computing, Gaming..." onkeydown="_wiz.intKeyS2(event)"></div>
      ${interestTopics.length ? `<div class="s2-label" style="margin-top:20px">Your topics</div><div class="pills" style="margin-bottom:20px">${itemsH}</div>` : ''}
      <div class="s2-label" style="margin-top:${interestTopics.length ? 8 : 20}px">Suggested for your role <span class="s2-hint">\u00B7 tap to add</span></div>
      <div class="pills">${sugH}</div>`;
    el.style.animation = 'none'; el.offsetHeight; el.style.animation = '';
    return;
  }

  const sections = getTabSections(tab);
  const multiSec = sections.length > 1;

  // Issue 6: Collapse all / Expand all toolbar
  let toolbarH = '';
  if (multiSec || sections.length === 1) {
    const totalSecs = sections.reduce((n, sec) => n + (PANELS[sec.id]?.qs?.length || 1), 0);
    if (totalSecs > 2) {
      const allCol = _s2CollapseAll;
      toolbarH = `<div class="s2-toolbar"><button class="s2-coll-btn" onclick="_wiz.togS2CollapseAll()">${allCol ? 'Expand all' : 'Collapse all'}</button></div>`;
    }
  }

  let bodyH = sections.map(sec => {
    const panel = PANELS[sec.id];
    if (!panel) return renderGenericSection(sec);
    const sel = panelSel[sec.id] || {}; const custom = panelCustom[sec.id] || {};
    let html = multiSec ? `<div class="s2-section">
      <div class="s2-sec-hdr coll" onclick="_wiz.togS2Section('${sec.id}')"><span class="s2-sec-icon">${sec.icon}</span> ${sec.name}<span class="s2-sec-chev">${_collapsedSections.has(sec.id) ? '\u25B6' : '\u25BC'}</span></div>
      <div class="s2-sec-body ${_collapsedSections.has(sec.id) ? 'collapsed' : ''}">` : '';

    html += panel.qs.map(q => {
      const isViewAll = _viewAllPills.has(q.id);
      const basePills = isViewAll ? getAllPills(q.id, q.pills) : getEffectivePills(q.id, q.pills);
      const fullCount = getAllPills(q.id, q.pills).length;
      const shortCount = getEffectivePills(q.id, q.pills).length;
      const hasMore = fullCount > shortCount;
      const picked = sel[q.id] || new Set(); const all = [...basePills, ...(custom[q.id] || [])];
      const hint = q.hint ? ` <span class="s2-hint">\u00B7 ${q.hint}</span>` : '';

      // Issue 6: Per-question collapsible (for non-multi-section views)
      const qKey = `${sec.id}_${q.id}`;
      const qCollapsed = !multiSec && _collapsedSections.has(qKey);
      let h = '';
      if (!multiSec) {
        h += `<div class="s2-label" style="cursor:pointer;display:flex;align-items:center" onclick="_wiz.togS2Section('${qKey}')">${q.label}${hint}<span class="s2-sec-chev">${qCollapsed ? '\u25B6' : '\u25BC'}</span></div>`;
        h += `<div class="s2-sec-body ${qCollapsed ? 'collapsed' : ''}">`;
      } else {
        h += `<div class="s2-label">${q.label}${hint}</div>`;
      }
      const isDecisive = DETERMINISTIC_QS.has(q.id);
      h += `<div class="pills">`;
      for (const p of all) {
        const isC = (custom[q.id] || []).includes(p);
        h += `<div class="pill ${isDecisive ? 'pill-decisive' : ''} ${picked.has(p) ? 'on' : ''}" onclick="_wiz.togPanel('${sec.id}','${q.id}','${escAttr(p)}','${q.type}')">
          ${esc(p)}${isC ? `<span class="pill-x" onclick="event.stopPropagation();_wiz.rmPanelCustom('${sec.id}','${q.id}','${escAttr(p)}')">&times;</span>` : ''}
        </div>`;
      }
      if (q.canAdd) {
        h += `<span id="wiz-aw-${sec.id}-${q.id}" style="display:none"><input class="add-inp" id="wiz-ai-${sec.id}-${q.id}" placeholder="Type & Enter" onkeydown="_wiz.addPanelKey(event,'${sec.id}','${q.id}')"></span>`;
        h += `<div class="pill pill-add" id="wiz-ab-${sec.id}-${q.id}" onclick="_wiz.showPanelAdd('${sec.id}','${q.id}')">+ Add</div>`;
      }
      h += `</div>`;
      // Issue 4: "View all" toggle for domain pills
      if (hasMore && q.type !== 's') {
        h += `<button class="va-tog" onclick="_wiz.togViewAll('${q.id}')">${isViewAll ? 'Show less' : `View all (${fullCount})`}</button>`;
      }
      if (!multiSec) h += `</div>`; // close s2-sec-body for per-question collapse
      return h;
    }).join('');
    if (multiSec) html += `</div></div>`; // close s2-sec-body + s2-section
    return html;
  }).join('');

  el.innerHTML = toolbarH + bodyH;
  // Append AI suggestions area
  el.innerHTML += renderTabSuggestions(activeTab);
  el.style.animation = 'none'; el.offsetHeight; el.style.animation = '';
  // Trigger AI suggestion fetch if not cached
  if (activeTab && activeTab !== 'interests') fetchTabSuggestion(activeTab);
}

function renderGenericSection(sec) {
  const custom = panelCustom[sec.id]?.kw || [];
  let html = `<div class="s2-section"><div class="s2-sec-hdr"><span class="s2-sec-icon">\uD83D\uDCCC</span> ${sec.name}</div>`;
  html += `<div class="s2-label">Keywords to track <span class="s2-hint">\u00B7 Type and press Enter</span></div><div class="pills">`;
  for (const p of custom) html += `<div class="pill on">${esc(p)}<span class="pill-x" onclick="event.stopPropagation();_wiz.rmPanelCustom('${sec.id}','kw','${escAttr(p)}')">&times;</span></div>`;
  html += `<span id="wiz-aw-${sec.id}-kw" style="display:none"><input class="add-inp" id="wiz-ai-${sec.id}-kw" placeholder="Type & Enter" onkeydown="_wiz.addPanelKey(event,'${sec.id}','kw')"></span>`;
  html += `<div class="pill pill-add" id="wiz-ab-${sec.id}-kw" onclick="_wiz.showPanelAdd('${sec.id}','kw')">+ Add</div></div></div>`;
  return html;
}

function dismissS2Banner() { _s2BannerDismissed = true; const b = document.querySelector('.s2-banner'); if (b) b.remove(); }

/* Step 2 Interactions */
function togIntS2(val) { const i = interestTopics.indexOf(val); i >= 0 ? interestTopics.splice(i, 1) : interestTopics.push(val); renderTabBody(); _wizSaveState(); }
function rmIntS2(val) { interestTopics = interestTopics.filter(v => v !== val); renderTabBody(); _wizSaveState(); }
function intKeyS2(e) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim(); if (!interestTopics.includes(v)) interestTopics.push(v);
    e.target.value = ''; renderTabBody(); _wizSaveState();
    setTimeout(() => { const i = document.getElementById('wiz-int-inp'); if (i) i.focus(); }, 60);
  }
}

function togPanel(sid, qid, val, type) {
  if (!panelSel[sid]) panelSel[sid] = {}; if (!panelSel[sid][qid]) panelSel[sid][qid] = new Set();
  const s = panelSel[sid][qid];
  const oldVal = type === 's' ? [...s][0] : null;
  if (type === 's') { s.clear(); s.add(val); } else { s.has(val) ? s.delete(val) : s.add(val); }
  const changed = DETERMINISTIC_QS.has(qid) && (type !== 's' || oldVal !== val);
  if (changed && activeTab) {
    // Cross-panel propagation: stage shared between career_opps and jobhunt
    if (qid === 'stage') {
      for (const pid of STAGE_SHARED_PANELS) {
        if (pid !== sid) {
          if (!panelSel[pid]) panelSel[pid] = {};
          panelSel[pid]['stage'] = new Set([val]);
        }
      }
      // Stage is cross-cutting — lazily invalidate other tabs (re-fetch on switch)
      for (const tid of Object.keys(_tabSuggestCache)) {
        if (tid !== activeTab) delete _tabSuggestCache[tid];
      }
    }
    // Debounced refresh — 800ms delay to batch rapid pill clicks
    clearTimeout(_suggestDebounceTimer);
    _suggestDebounceTimer = setTimeout(() => refreshSuggestions(activeTab), 800);
  }
  renderTabBody(); _wizSaveState();
}

function showPanelAdd(sid, qid) {
  const w = document.getElementById(`wiz-aw-${sid}-${qid}`); if (w) w.style.display = 'inline';
  const b = document.getElementById(`wiz-ab-${sid}-${qid}`); if (b) b.style.display = 'none';
  const i = document.getElementById(`wiz-ai-${sid}-${qid}`); if (i) i.focus();
}

function addPanelKey(e, sid, qid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim();
    if (!panelCustom[sid]) panelCustom[sid] = {}; if (!panelCustom[sid][qid]) panelCustom[sid][qid] = [];
    if (!panelSel[sid]) panelSel[sid] = {}; if (!panelSel[sid][qid]) panelSel[sid][qid] = new Set();
    if (!panelCustom[sid][qid].includes(v)) { panelCustom[sid][qid].push(v); panelSel[sid][qid].add(v); }
    e.target.value = ''; renderTabBody(); _wizSaveState();
    setTimeout(() => {
      const w = document.getElementById(`wiz-aw-${sid}-${qid}`), b = document.getElementById(`wiz-ab-${sid}-${qid}`), i = document.getElementById(`wiz-ai-${sid}-${qid}`);
      if (w) w.style.display = 'inline'; if (b) b.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const b = document.getElementById(`wiz-ab-${sid}-${qid}`); if (b) b.style.display = '';
  }
}

function rmPanelCustom(sid, qid, val) {
  panelCustom[sid][qid] = (panelCustom[sid][qid] || []).filter(v => v !== val);
  if (panelSel[sid]?.[qid]) panelSel[sid][qid].delete(val);
  renderTabBody(); _wizSaveState();
}

// Issue 4: Toggle "View all" pills
function togViewAll(qId) {
  _viewAllPills.has(qId) ? _viewAllPills.delete(qId) : _viewAllPills.add(qId);
  renderTabBody();
}

// Issue 6: Toggle section collapse
function togS2Section(secId) {
  _collapsedSections.has(secId) ? _collapsedSections.delete(secId) : _collapsedSections.add(secId);
  renderTabBody();
}
function togS2CollapseAll() {
  _s2CollapseAll = !_s2CollapseAll;
  if (_s2CollapseAll) {
    // Collapse all current sections
    const tabs = getActiveTabs();
    const tab = tabs.find(t => t.id === activeTab);
    if (tab && tab.type !== 'interests') {
      const sections = getTabSections(tab);
      for (const sec of sections) {
        _collapsedSections.add(sec.id);
        const panel = PANELS[sec.id];
        if (panel) { for (const q of panel.qs) _collapsedSections.add(`${sec.id}_${q.id}`); }
      }
    }
  } else {
    _collapsedSections.clear();
  }
  renderTabBody();
}

/* ═══════════════════════════════════════
   TAB SCROLL SYSTEM
   ═══════════════════════════════════════ */

function initTabScroll() {
  const bar = document.getElementById('wiz-tab-bar');
  if (!bar) return;
  bar.addEventListener('scroll', updateTabArrows, { passive: true });
  let down = false, startX = 0, scrollL = 0, moved = false;
  bar.addEventListener('mousedown', e => {
    if (e.target.closest('.tab-arr')) return;
    down = true; moved = false; startX = e.pageX; scrollL = bar.scrollLeft;
  });
  const stop = () => {
    if (!down) return; down = false; bar.classList.remove('dragging');
    if (moved) { const suppress = e => { e.stopPropagation(); e.preventDefault(); }; bar.addEventListener('click', suppress, {capture:true, once:true}); }
  };
  bar.addEventListener('mouseup', stop);
  bar.addEventListener('mouseleave', stop);
  bar.addEventListener('mousemove', e => {
    if (!down) return; e.preventDefault();
    const dx = e.pageX - startX;
    if (Math.abs(dx) > 3) { moved = true; bar.classList.add('dragging'); }
    if (moved) bar.scrollLeft = scrollL - dx;
  });
  setTimeout(() => { updateTabArrows(); scrollActiveTabIntoView(); }, 30);
}

function updateTabArrows() {
  const bar = document.getElementById('wiz-tab-bar');
  const al = document.getElementById('wiz-tab-arr-l'), ar = document.getElementById('wiz-tab-arr-r');
  if (!bar || !al || !ar) return;
  al.classList.toggle('show', bar.scrollLeft > 4);
  ar.classList.toggle('show', bar.scrollLeft < bar.scrollWidth - bar.clientWidth - 4);
}

function scrollTabDir(dir) {
  const bar = document.getElementById('wiz-tab-bar');
  if (!bar) return;
  bar.scrollBy({left: dir * 160, behavior: 'smooth'});
}

function scrollActiveTabIntoView() {
  const bar = document.getElementById('wiz-tab-bar');
  if (!bar) return;
  const active = bar.querySelector('.tab.active');
  if (!active) return;
  const bRect = bar.getBoundingClientRect(), tRect = active.getBoundingClientRect();
  const pad = 40;
  if (tRect.left < bRect.left + pad) bar.scrollBy({left: tRect.left - bRect.left - pad, behavior: 'smooth'});
  else if (tRect.right > bRect.right - pad) bar.scrollBy({left: tRect.right - bRect.right + pad, behavior: 'smooth'});
}

/* ═══════════════════════════════════════
   STEP 3 — REVIEW
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
        // Empty — renderS3 will show "Couldn't load suggestions" + Add button
        rvItems[sec.id] = [];
      }
    }
  }
}

async function initRvWithAI() {
  // If we already have AI data cached, use it directly
  if (_rvItemsCache) {
    initRv();
    renderS3();
    return;
  }
  // Show loading state in Step 3 while fetching
  const el = document.getElementById('wiz-s3i');
  if (el) {
    const role = document.getElementById('simple-role')?.value?.trim() || '';
    el.innerHTML = `<div class="ld">
      <div class="wiz-ring"></div>
      <div class="ld-t">Personalizing your review...</div>
      <div class="ld-s">Finding relevant entities for <strong>${esc(role)}</strong></div>
    </div>`;
  }
  // Fetch AI-generated entities
  await fetchRvItems();
  // Initialize items from AI response (empty sections if fetch failed)
  initRv();
  renderS3();
}

function renderS3() {
  const el = document.getElementById('wiz-s3i');
  if (!el) return;
  const allExpanded = rvCollapsed.size === 0;
  let sectionCount = 0;
  let sectionsHTML = '';

  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    if (c.id === 'interests') {
      if (interestTopics.length) {
        sectionCount++;
        const col = rvCollapsed.has('interests');
        sectionsHTML += `<div class="rv ${col ? 'collapsed' : ''}">
          <div class="rv-h" onclick="_wiz.togRvCollapse('interests')"><span class="rv-i">\uD83E\uDDED</span><span class="rv-t">Interests & Trends</span><span class="rv-chev">\u25BC</span></div>
          <div class="rv-body"><div class="rv-sub"><div class="rv-lbl">Topics we'll track</div>
          <div class="rv-pills">${interestTopics.map(t => `<span class="rv-p" onclick="_wiz.rvRmInt('${escAttr(t)}')">${esc(t)}<span class="rv-x">&times;</span></span>`).join('')}
          <button class="rv-a" onclick="_wiz.goTo(1);_wiz.switchTab('interests')">+ Add</button></div></div></div></div>`;
      }
      continue;
    }
    const sections = getS3Sections(c);
    if (!sections.length) continue;
    sectionCount++;
    const col = rvCollapsed.has(c.id);
    let inner = '';
    for (const sec of sections) {
      const items = rvItems[sec.id] || [];
      const aiSec = _rvItemsCache?.sections?.[sec.id];
      const genLabel = aiSec?.label || 'Tracking';
      const aiTags = aiSec?.tags || {};
      const panel = PANELS[sec.id];
      let ctx = '';
      // Only show deterministic answers (stage, opptype, etc.) — not default pill text
      if (panel && panelSel[sec.id]) { ctx = panel.qs.filter(q => DETERMINISTIC_QS.has(q.id)).map(q => [...(panelSel[sec.id][q.id] || [])].join(', ')).filter(Boolean).join(' \u00B7 '); }
      inner += `<div class="rv-sub">
        <div class="rv-sub-hdr"><span class="rv-sub-icon">${sec.icon}</span><span class="rv-sub-name">${sec.name}</span></div>
        ${ctx ? `<div class="rv-sub-ctx">${esc(ctx)}</div>` : ''}
        <div class="rv-lbl">${esc(genLabel)}</div>
        <div class="rv-pills">
          ${items.length ? items.map(it => { const tg = aiTags[it] || ''; return `<span class="rv-p" onclick="_wiz.rvRm('${sec.id}','${escAttr(it)}')">${esc(it)}${tg ? `<span class="rv-tag">${esc(tg)}</span>` : ''}<span class="rv-x">&times;</span></span>`; }).join('') : `<span class="rv-empty" style="color:var(--text-muted);font-size:13px;font-style:italic">Couldn\u2019t load suggestions \u2014 add your own below</span>`}
          <span id="wiz-rw-${sec.id}" style="display:none"><input class="add-inp" id="wiz-ri-${sec.id}" placeholder="Add..." onkeydown="_wiz.rvAddKey(event,'${sec.id}')"></span>
          <button class="rv-a" id="wiz-ra-${sec.id}" onclick="_wiz.rvShowAdd('${sec.id}')">+ Add</button>
        </div>
      </div>`;
    }
    sectionsHTML += `<div class="rv ${col ? 'collapsed' : ''}">
      <div class="rv-h" onclick="_wiz.togRvCollapse('${c.id}')"><span class="rv-i">${c.icon}</span><span class="rv-t">${c.name}</span><span class="rv-chev">\u25BC</span></div>
      <div class="rv-body">${inner}</div>
    </div>`;
  }

  const toggleBtn = sectionCount >= 2 ? `<button class="s3-toggle" onclick="_wiz.${allExpanded ? 'collapseAllRv' : 'expandAllRv'}()">${allExpanded ? 'Collapse all' : 'Expand all'}</button>` : '';
  // Only show discover section if AI returned discover items — no hardcoded fallback
  const discoverItems = (_rvItemsCache?.discover?.length) ? _rvItemsCache.discover : [];
  const discHTML = discoverItems.filter(d => !discoverAdded.has(d.name)).length ? `
    <div class="disc">
      <div class="disc-h">\uD83D\uDCA1 Discover more</div>
      <div class="disc-s">We also found these \u2014 want to track them?</div>
      <div class="disc-pills">${discoverItems.map(d => `<div class="disc-p ${discoverAdded.has(d.name) ? 'added' : ''}" onclick="_wiz.discoverAdd('${escAttr(d.name)}','${escAttr(d.target || Object.keys(rvItems)[0] || '')}')">${esc(d.name)}<span class="disc-tag">${esc(d.tag || '')}</span></div>`).join('')}</div>
    </div>` : '';

  el.innerHTML = `<div class="s3-top"><div><h1 class="title">Your intelligence profile</h1><p class="sub" style="margin-bottom:0">Generated from your choices. Tap to remove, or add your own.</p></div>${toggleBtn}</div>
    <div style="margin-top:24px">${sectionsHTML}</div>
    ${discHTML}
    <div style="display:flex;gap:12px;justify-content:center;margin-top:28px">
      <button class="btn bo" onclick="_wiz.goBack()">\u2190 Adjust</button>
    </div>`;
}

function togRvCollapse(id) { rvCollapsed.has(id) ? rvCollapsed.delete(id) : rvCollapsed.add(id); renderS3(); }
function collapseAllRv() { for (const c of CATS) if (selCats.has(c.id)) rvCollapsed.add(c.id); renderS3(); }
function expandAllRv() { rvCollapsed.clear(); renderS3(); }

function rvRm(sid, val) { rvItems[sid] = (rvItems[sid] || []).filter(v => v !== val); renderS3(); }
function rvRmInt(val) { interestTopics = interestTopics.filter(v => v !== val); renderS3(); }
function rvShowAdd(sid) {
  const w = document.getElementById('wiz-rw-' + sid); if (w) w.style.display = 'inline';
  const a = document.getElementById('wiz-ra-' + sid); if (a) a.style.display = 'none';
  const i = document.getElementById('wiz-ri-' + sid); if (i) i.focus();
}
function rvAddKey(e, sid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim(); if (!rvItems[sid]) rvItems[sid] = []; if (!rvItems[sid].includes(v)) rvItems[sid].push(v);
    e.target.value = ''; renderS3();
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
  console.log('[Wizard] buildWizardContext:', result);
  return result;
}

function doBuild() {
  const ftr = document.getElementById('wiz-ftr');
  if (ftr) ftr.classList.add('wiz-hidden');
  const role = document.getElementById('simple-role')?.value?.trim() || 'your profile';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const deep = document.getElementById('wiz-deep-mode')?.checked || false;
  const modeLabel = deep ? 'Deep analysis' : 'Quick generation';
  const el = document.getElementById('wiz-s3i');
  if (!el) return;
  el.innerHTML = `<div class="ld">
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
  const slide = document.getElementById('wiz-s3');
  if (slide) slide.scrollTo(0, 0);

  // Build enriched context from wizard selections ONLY
  // CRITICAL: Do NOT include #simple-context — it may contain old AI-generated text
  // with tech/telecom keywords from a previous run, which would contaminate the new profile.
  const wizContext = buildWizardContext();

  // Call the real generate-profile API with wizard-only context
  callGenerateProfile(role, location, wizContext, deep);
}

async function callGenerateProfile(role, location, context, deep = false) {
  console.log('[Wizard] callGenerateProfile:', {role, location, context, deep});
  let stepIdx = 0;
  const bar = document.getElementById('wiz-bar');
  const setBar = (pct) => { if (bar) bar.style.width = pct + '%'; };
  const advanceStep = () => {
    if (stepIdx > 0) { const prev = document.getElementById('wiz-l' + (stepIdx - 1)); if (prev) { prev.classList.remove('on'); prev.classList.add('ok'); } }
    if (stepIdx < 3) { const cur = document.getElementById('wiz-l' + stepIdx); if (cur) cur.classList.add('on'); stepIdx++; }
  };
  advanceStep(); setBar(10); // Step 0: generating context
  try {
    await new Promise(r => setTimeout(r, 600));
    advanceStep(); setBar(30); // Step 1: building categories

    // Animate progress bar — rate depends on mode (deep takes ~5x longer)
    let progressPct = 30;
    const tickMs = deep ? 8000 : 1500;
    const startTime = Date.now();
    const progressTimer = setInterval(() => {
      if (progressPct < 70) {
        progressPct += 1;
        setBar(progressPct);
      }
      // Show elapsed time
      const subtitle = document.querySelector('.ld-s');
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      if (subtitle && elapsed > 5) {
        const mode = deep ? 'Deep analysis' : 'Generating';
        subtitle.innerHTML = `${mode} for <strong>${esc(role)}</strong> (${elapsed}s)`;
      }
    }, tickMs);

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
    advanceStep(); setBar(80); // Step 2: selecting sources
    await new Promise(r => setTimeout(r, 500));
    advanceStep(); setBar(100); // Complete
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
  const el = document.getElementById('wiz-s3i');
  if (!el) return;
  if (errorMsg) {
    el.innerHTML = `<div class="done">
      <div class="done-c err">${CK.replace('viewBox', 'width="30" height="30" fill="none" viewBox')}</div>
      <div class="done-t">Generation had issues</div>
      <div class="done-s">We couldn't fully generate your profile: ${esc(errorMsg)}.<br>You can still use the wizard selections or try again.</div>
      <div style="display:flex;gap:12px">
        <button class="btn bo" onclick="_wiz.goTo(0)">Try Again</button>
        <button class="btn bp" onclick="_wiz.finishWizard()" style="padding:14px 36px;font-size:16px">Use Selections Anyway \u2192</button>
      </div>
    </div>`;
  } else {
    const catCount = _wizGenerateData?.categories?.length || 0;
    const itemCount = (_wizGenerateData?.categories || []).reduce((a, c) => a + (c.items?.length || 0), 0);
    el.innerHTML = `<div class="done">
      <div class="done-c">${CK.replace('viewBox', 'width="30" height="30" fill="none" viewBox')}</div>
      <div class="done-t">Your feed is ready!</div>
      <div class="done-s">Generated ${catCount} categories with ${itemCount} tracking items. Your dashboard will now show signals tailored to your profile.</div>
      <button class="btn bp" onclick="_wiz.finishWizard()" style="padding:14px 36px;font-size:16px">Apply & Close \u2192</button>
    </div>`;
  }
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

  // Show loading animation in Step 3 area while AI analyzes
  step = 2;
  const slides = document.getElementById('wiz-slides');
  const prog = document.getElementById('wiz-prog');
  if (slides) slides.style.transform = 'translateX(-200%)';
  if (prog) prog.style.width = '100%';
  renderDots();
  const bk = document.getElementById('wiz-bbk');
  if (bk) bk.classList.add('wiz-hidden');
  const ftr = document.getElementById('wiz-ftr');
  if (ftr) ftr.classList.add('wiz-hidden');

  const el = document.getElementById('wiz-s3i');
  if (!el) return;
  el.innerHTML = `<div class="ld">
    <div class="wiz-ring"></div>
    <div class="ld-t">Analyzing your profile...</div>
    <div class="ld-s">Finding the best setup for <strong>${esc(role)}</strong>${location ? ` in <strong>${esc(location)}</strong>` : ''}</div>
    <div class="ld-bar"><div class="ld-bar-fill" id="wiz-qbar"></div></div>
    <div class="ld-list">
      <div class="ls on" id="wiz-q0"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Analyzing role &amp; location</span></div>
      <div class="ls" id="wiz-q1"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Selecting relevant categories</span></div>
      <div class="ls" id="wiz-q2"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Preparing your review</span></div>
    </div>
  </div>`;
  const slide = document.getElementById('wiz-s3');
  if (slide) slide.scrollTo(0, 0);

  let qIdx = 0;
  const qbar = document.getElementById('wiz-qbar');
  const setQBar = (pct) => { if (qbar) qbar.style.width = pct + '%'; };
  const advQ = () => {
    if (qIdx > 0) { const prev = document.getElementById('wiz-q' + (qIdx - 1)); if (prev) { prev.classList.remove('on'); prev.classList.add('ok'); } }
    if (qIdx < 3) { const cur = document.getElementById('wiz-q' + qIdx); if (cur) cur.classList.add('on'); qIdx++; }
  };
  advQ(); setQBar(10); // Step 0: Analyzing role

  try {
    // Step 1: AI pre-selection + domain classification
    await new Promise(r => setTimeout(r, 400));
    advQ(); setQBar(40);

    _currentDomain = classifyRole(role);
    let aiSuccess = false;
    try {
      const available = CATS.map(c => ({
        id: c.id, label: c.name,
        subs: (c.subs || []).map(s => ({id: s.id, label: s.name}))
      }));
      const resp = await fetch('/api/wizard-preselect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
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
            } else if (!selCats.has(c.id)) {
              selSubs[c.id] = new Set();
            }
          }
          aiSuccess = true;
        }
      }
    } catch(e) { /* AI unavailable */ }

    // Fallback to domain-specific defaults if AI failed
    if (!aiSuccess) applyDomainDefaults();

    // Apply smart panel defaults (stage, opptype, domain pills) for all selected subs
    applySmartPanelDefaults();

    // Step 2: Fetch AI-generated role-aware entities for review
    await new Promise(r => setTimeout(r, 300));
    advQ(); setQBar(50);

    // Fetch role-aware items from backend
    await fetchRvItems();
    setQBar(80);

    // Initialize rvItems from AI response (empty sections if fetch failed)
    initRv();

    // Complete loading animation
    advQ(); setQBar(100);
    await new Promise(r => setTimeout(r, 400));

    // Land on Step 3 review — user can vet, adjust, then click "Build my feed"
    rvCollapsed.clear();
    const ftr2 = document.getElementById('wiz-ftr');
    if (ftr2) ftr2.classList.remove('wiz-hidden');
    const bk2 = document.getElementById('wiz-bbk');
    if (bk2) bk2.classList.remove('wiz-hidden');
    const clr2 = document.getElementById('wiz-clr');
    if (clr2) clr2.classList.add('wiz-hidden');
    const psave2 = document.getElementById('wiz-psave');
    if (psave2) psave2.classList.remove('wiz-hidden');
    const nx2 = document.getElementById('wiz-bnx');
    if (nx2) nx2.classList.add('wiz-hidden');
    const buildRow2 = document.getElementById('wiz-build-row');
    if (buildRow2) buildRow2.classList.remove('wiz-hidden');
    renderS3();
  } catch (e) {
    console.error('Quick setup failed:', e);
    // On failure, land on Step 1 so user can manually select
    goTo(0);
    if (typeof showToast === 'function') showToast('Quick setup had an issue. Please select manually.', 'warning');
  }
}

function discoverAdd(name, target) {
  if (discoverAdded.has(name)) return;
  discoverAdded.add(name);
  if (!rvItems[target]) rvItems[target] = [];
  if (!rvItems[target].includes(name)) rvItems[target].push(name);
  renderS3();
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
    const modal = document.getElementById('wiz-modal');
    if (!modal || !modal.classList.contains('open')) return;
    const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
    if (inInput) {
      if (e.key === 'Escape') { e.preventDefault(); e.target.blur(); }
      return;
    }
    if (e.key === 'Enter') { e.preventDefault(); goNext(); }
    else if (e.key === 'Escape') { e.preventDefault(); if (_presetMenuOpen) { _presetMenuOpen = false; renderPresetBar(); } else if (step > 0) goBack(); else closeWizard(); }
    else if (step === 0 && e.key >= '1' && e.key <= '6') {
      e.preventDefault(); const idx = parseInt(e.key) - 1;
      if (idx < CATS.length) togCat(CATS[idx].id);
    }
  });
  // Close preset dropdown when clicking outside
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
  // Try restoring previous wizard state from localStorage
  const restored = _wizLoadState();
  if (restored) console.log('[Wizard] Restored saved state from localStorage');
  injectCSS();
  injectDOM(role, location);
  _presetMenuOpen = false;
  renderPresetBar();

  // Open with animation
  requestAnimationFrame(() => {
    const bk = document.getElementById('wiz-bk');
    const modal = document.getElementById('wiz-modal');
    if (bk) bk.classList.add('open');
    if (modal) modal.classList.add('open');
  });
  document.body.style.overflow = 'hidden';
  goTo(0);
}

/* ═══════════════════════════════════════
   AI PRE-SELECTION (Stage 3)
   ═══════════════════════════════════════ */

async function fetchPreselection(role, location) {
  try {
    const available = CATS.map(c => ({
      id: c.id,
      label: c.name,
      subs: (c.subs || []).map(s => ({id: s.id, label: s.name}))
    }));
    const resp = await fetch('/api/wizard-preselect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role, location, available_categories: available})
    });
    if (!resp.ok) return; // Fallback: keep defaults
    const data = await resp.json();
    if (data.error) return;

    const aiCats = data.selected_categories || [];
    const aiSubs = data.selected_subs || {};

    // Apply AI selections (only if wizard is still on Step 1)
    if (step !== 0) return;

    // Replace default selections with AI selections
    selCats = new Set(aiCats.filter(id => CATS.some(c => c.id === id)));
    for (const c of CATS) {
      if (selCats.has(c.id) && aiSubs[c.id]) {
        const validSubIds = new Set([...c.subs.map(s => s.id)]);
        selSubs[c.id] = new Set(aiSubs[c.id].filter(s => validSubIds.has(s)));
      } else if (!selCats.has(c.id)) {
        selSubs[c.id] = new Set();
      }
    }
    renderS1();
    updNx();
  } catch (e) {
    // Silently fail — keep default selections
    console.log('Wizard: AI pre-selection unavailable, using defaults');
  }
}

/* ═══════════════════════════════════════
   AI TAB SUGGESTIONS (Stage 4)
   ═══════════════════════════════════════ */

async function fetchTabSuggestion(tabId, extraExclude, isRefresh) {
  if (_tabSuggestCache[tabId]) return; // Already fetched or loading
  _tabSuggestCache[tabId] = {suggestions: [], loading: true, added: new Set(), isRefresh: !!isRefresh};
  // Re-render to show shimmer placeholders
  renderTabBody();

  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const cat = CATS.find(c => c.id === tabId);
  if (!cat) { _tabSuggestCache[tabId].loading = false; return; }

  // Gather existing items AND structured deterministic context from the tab's panel selections.
  // Use getTabSections to resolve actual displayed sections — this handles the career_opps
  // synthetic panel correctly (shown when both jobhunt + intern are selected).
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
          if (DETERMINISTIC_QS.has(q.id)) {
            deterministicParts.push(`${q.label}: ${[...picked].join(', ')}`);
          } else {
            existingItems.push(...picked);
          }
        }
      }
    }
  }
  // Include previously-added suggestions so backend doesn't re-suggest them
  if (extraExclude && extraExclude.size) {
    existingItems.push(...extraExclude);
  }
  const selectionsContext = deterministicParts.join('; ');

  // Build structured selections dict for backend
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
    // Cancel any in-flight suggestion request
    if (_suggestAbortCtrl) _suggestAbortCtrl.abort();
    _suggestAbortCtrl = new AbortController();
    const resp = await fetch('/api/wizard-tab-suggest', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: _suggestAbortCtrl.signal,
      body: JSON.stringify({
        role, location,
        category_id: tabId,
        category_label: cat.name,
        existing_items: existingItems,
        selections_context: selectionsContext,
        selections,
        exclude_selected: extraExclude ? [...extraExclude] : []
      })
    });
    if (!resp.ok) throw new Error('Request failed');
    const data = await resp.json();
    _tabSuggestCache[tabId] = {
      suggestions: data.suggestions || [],
      loading: false,
      added: new Set()
    };
  } catch (e) {
    if (e.name === 'AbortError') return; // Request cancelled by newer one
    _tabSuggestCache[tabId] = {suggestions: [], loading: false, added: new Set()};
    console.log('Wizard: Tab suggestion unavailable for', tabId);
  }
  // Re-render if still on same tab
  if (activeTab === tabId && step === 1) renderTabBody();
}

function addTabSuggestion(tabId, keyword) {
  const cache = _tabSuggestCache[tabId];
  if (!cache) return;
  if (!cache.added) cache.added = new Set();
  cache.added.add(keyword);
  // Add to interest topics if interests tab
  if (tabId === 'interests') {
    if (!interestTopics.includes(keyword)) interestTopics.push(keyword);
  }
  renderTabBody();
}

function renderTabSuggestions(tabId) {
  const cache = _tabSuggestCache[tabId];
  if (!cache) return '';
  if (cache.loading) {
    // Shimmer placeholders instead of just a spinner
    const shimmerPills = '<div class="ai-sug-pill shimmer"></div>'.repeat(5);
    return `<div class="ai-sug"><div class="ai-sug-hdr ${cache.isRefresh ? 'flash' : ''}"><div class="ai-sug-spin"></div> Suggesting...<button class="ai-ref spin" disabled style="opacity:.3">&#x21bb;</button></div><div class="pills">${shimmerPills}</div></div>`;
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
    pillsHtml += `<div class="ai-sug-pill ${isAdded ? 'added' : ''}" onclick="_wiz.addTabSuggestion('${escAttr(tabId)}','${escAttr(s)}')">${isAdded ? '\u2713 ' : '+ '}${esc(s)}</div>`;
  }
  return `<div class="ai-sug"><div class="ai-sug-hdr">\u2728 Suggestions<button class="ai-ref" onclick="_wiz.refreshSuggestions('${escAttr(tabId)}')" title="Get new suggestions">&#x21bb;</button></div><div class="pills">${pillsHtml}</div></div>`;
}

function refreshSuggestions(tabId) {
  const cache = _tabSuggestCache[tabId];
  // Collect ALL previously added items (both from current batch and from prior refreshes)
  const allPreviouslyAdded = new Set();
  if (cache?.added) for (const item of cache.added) allPreviouslyAdded.add(item);
  if (cache?.keptFromPrev) for (const item of cache.keptFromPrev) allPreviouslyAdded.add(item);

  // Delete cache so fetchTabSuggestion can run fresh
  delete _tabSuggestCache[tabId];

  // Re-fetch with previously-added items excluded from AI suggestions
  fetchTabSuggestion(tabId, allPreviouslyAdded, true).then(() => {
    const newCache = _tabSuggestCache[tabId];
    if (newCache && allPreviouslyAdded.size) {
      // Store kept items separately — they render as permanent "added" pills
      newCache.keptFromPrev = allPreviouslyAdded;
      if (activeTab === tabId && step === 1) renderTabBody();
    }
  });
}

function closeWizard() {
  _wizSaveState(); // Preserve state so user can resume later
  const bk = document.getElementById('wiz-bk');
  const modal = document.getElementById('wiz-modal');
  if (bk) bk.classList.remove('open');
  if (modal) modal.classList.remove('open');
  document.body.style.overflow = '';
  // Clean teardown after transition
  setTimeout(() => { removeDOM(); removeCSS(); }, 400);
}

/* ═══════════════════════════════════════
   GLOBAL EXPORTS (for inline onclick handlers)
   ═══════════════════════════════════════ */

window._wiz = {
  // Navigation
  goTo, goNext, goBack,
  // Step 1
  togCat, togSub, showAddSub, addSubKey, rmCustomSub,
  // Step 2
  switchTab, togPanel, showPanelAdd, addPanelKey, rmPanelCustom,
  togIntS2, rmIntS2, intKeyS2,
  scrollTabDir,
  togViewAll,              // Issue 4: View all pills
  togS2Section, togS2CollapseAll, // Issue 6: Collapsible sections
  refreshSuggestions,      // Issue 5: Refresh AI suggestions
  dismissS2Banner,         // Step 3: Dismiss banner
  // Step 3
  togRvCollapse, collapseAllRv, expandAllRv,
  rvRm, rvRmInt, rvShowAdd, rvAddKey,
  discoverAdd, addTabSuggestion,
  // Build
  doBuild, finishWizard,
  // Presets
  savePreset, loadPreset, deletePreset, togglePresetMenu,
  // Other
  skipToQuick, closeWizard, clearAll,
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
