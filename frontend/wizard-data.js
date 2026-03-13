/**
 * wizard-data.js — StratOS Wizard data constants
 * Extracted from wizard.js to reduce main file size.
 * Loaded before wizard.js via <script defer> ordering.
 */
window._wizData = (function() {
'use strict';

const CATS = [
  {id:'career',icon:'🎯',name:'Career & Jobs',desc:'Jobs, internships, hiring, promotions',
   subs:[{id:'jobhunt',name:'Job Hunting'},{id:'intern',name:'Internships'},{id:'research_pos',name:'Research Positions'},{id:'govjobs',name:'Government Jobs'},{id:'freelance',name:'Freelancing & Contracts'},{id:'remote',name:'Remote Work'},{id:'startup_jobs',name:'Startup Jobs'},{id:'executive',name:'Executive & Leadership'},{id:'teaching_pos',name:'Teaching Positions'},{id:'creative_jobs',name:'Creative & Design Jobs'},{id:'trade_jobs',name:'Skilled Trades'}]},
  {id:'industry',icon:'🏢',name:'Industry Intel',desc:'Company tracking, sector trends, competitors',
   subs:[{id:'ind_oilgas',name:'Oil & Gas'},{id:'ind_telecom',name:'Telecom'},{id:'ind_tech',name:'Tech / Software'},{id:'ind_bank',name:'Banking & Finance'},{id:'ind_construct',name:'Construction'},{id:'ind_health',name:'Healthcare'},{id:'ind_auto',name:'Automotive'},{id:'ind_aero',name:'Aerospace & Defense'},{id:'ind_retail',name:'Retail & E-Commerce'},{id:'ind_media',name:'Media & Entertainment'},{id:'ind_food',name:'Food & Agriculture'},{id:'ind_edu',name:'Education & EdTech'},{id:'ind_legal',name:'Legal & Compliance'},{id:'ind_energy',name:'Energy & Renewables'},{id:'ind_logistics',name:'Logistics & Supply Chain'},{id:'ind_pharma',name:'Pharmaceuticals'},{id:'ind_realestate',name:'Real Estate & PropTech'},{id:'ind_gaming',name:'Gaming & Esports'}]},
  {id:'learning',icon:'🎓',name:'Learning & Development',desc:'Certifications, courses, skill building, research',
   subs:[{id:'certs',name:'Professional Certifications'},{id:'academic',name:'Academic Research'},{id:'courses',name:'Online Courses'},{id:'hands_on',name:'Hands-on Skills'},{id:'confevents',name:'Conferences & Events'},{id:'bootcamps',name:'Bootcamps & Intensives'},{id:'workshops',name:'Workshops & Seminars'},{id:'mentorship',name:'Mentorship & Coaching'},{id:'open_source',name:'Open Source & Community'},{id:'competitions',name:'Competitions & Hackathons'},{id:'fellowships',name:'Fellowships & Grants'}]},
  {id:'markets',icon:'📈',name:'Markets & Investing',desc:'Stocks, crypto, commodities, analysis',
   subs:[{id:'stocks',name:'Stocks'},{id:'crypto',name:'Crypto'},{id:'commodities',name:'Commodities'},{id:'realestate',name:'Real Estate'},{id:'forex',name:'Forex'},{id:'mktreports',name:'Market Research & Reports'},{id:'etfs',name:'ETFs & Index Funds'},{id:'bonds',name:'Bonds & Fixed Income'},{id:'startups_vc',name:'Startups & Venture Capital'},{id:'options',name:'Options & Derivatives'}]},
  {id:'deals',icon:'💎',name:'Deals & Offers',desc:'Bank offers, student discounts, promotions',
   subs:[{id:'bankdeals',name:'Bank Deals'},{id:'telepromo',name:'Telecom Promotions'},{id:'studisc',name:'Student Discounts'},{id:'ccrewards',name:'Credit Card Rewards'},{id:'empben',name:'Employee Benefits'},{id:'techdeals',name:'Tech & Software Deals'},{id:'traveldeals',name:'Travel & Hotel Deals'},{id:'insurance',name:'Insurance Offers'},{id:'cashback',name:'Cashback & Loyalty'},{id:'subscriptions',name:'Subscription Deals'}]},
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
    {id:'opptype',label:'What are you looking for?',type:'m',pills:['Full-time Jobs','Internships','Freelance / Contract','Remote Positions'],defs:['Full-time Jobs','Internships']},
    {id:'stage',label:'Your stage',type:'s',pills:['Student','Fresh Graduate','Mid-Career','Senior'],def:'Student'},
    {id:'roles',label:'Role types',type:'m',pills:['Engineering','Software Development','Data Science','IT / Networking','Management','Design','Marketing','Finance','Healthcare','Legal'],canAdd:1,defs:['Engineering','Software Development']},
  ]},
  jobhunt:{icon:'🔍',name:'Job Hunting',qs:[
    {id:'stage',label:'Your stage',type:'s',pills:['Student','Fresh Graduate','Mid-Career','Senior'],def:'Student'},
    {id:'roles',label:'Role types',type:'m',pills:['Engineering','Software Development','Data Science','IT / Networking','Management','Design','Marketing','Finance','Healthcare','Legal','Operations','Sales'],canAdd:1,defs:['Engineering','Software Development']},
    {id:'jplatforms',label:'Job platforms',type:'m',pills:['LinkedIn','Indeed','Glassdoor','AngelList','Hired','We Work Remotely','Dice','ZipRecruiter'],canAdd:1,defs:['LinkedIn','Indeed']},
    {id:'jformat',label:'Work format',type:'m',pills:['On-site','Hybrid','Fully Remote','Contract / Freelance'],defs:['On-site','Hybrid']},
  ]},
  intern:{icon:'🧑‍💻',name:'Internships',qs:[
    {id:'itype',label:'Preferred type',type:'m',pills:['Summer Internship','Co-op / Year-long','Part-time','Remote','Paid','Research Assistantship'],defs:['Summer Internship']},
    {id:'ifields',label:'Fields',type:'m',pills:['Software Engineering','Data Science','Mechanical Engineering','Electrical Engineering','Civil Engineering','Marketing','Finance','Design','Healthcare','Legal'],canAdd:1,defs:['Software Engineering']},
    {id:'ico',label:'Target companies',type:'m',pills:['FAANG / Big Tech','Fortune 500','Startups','Government','Research Labs','Consulting Firms'],canAdd:1,defs:[]},
  ]},
  research_pos:{icon:'🔬',name:'Research Positions',qs:[
    {id:'rtype',label:'Type',type:'m',pills:['University Lab','Industry R&D','Government Research','Postdoc','Research Fellowship','Think Tank'],defs:['University Lab']},
    {id:'rfields',label:'Fields',type:'m',pills:['AI / ML','Embedded Systems','Networking','Energy','Biomedical','Materials Science','Quantum Computing','Robotics','Environmental Science','Economics'],canAdd:1,defs:['AI / ML']},
    {id:'rfunding',label:'Funding sources',type:'m',pills:['NSF','NIH','DARPA','EU Horizon','Industry Sponsored','University Funded'],canAdd:1,defs:[]},
  ]},
  govjobs:{icon:'🏛️',name:'Government Jobs',qs:[
    {id:'gdept',label:'Departments',type:'m',pills:['Digital / IT Authority','Ministry of Communications','Energy Sector','Central Bank','Defense / Military','Health Ministry','Education','Justice / Legal','Transportation','Environment'],canAdd:1,defs:[]},
    {id:'glevel',label:'Level',type:'m',pills:['Entry-level','Mid-level','Senior / Director','Political Appointee'],defs:['Entry-level']},
  ]},
  freelance:{icon:'🔗',name:'Freelancing & Contracts',qs:[
    {id:'fplatforms',label:'Platforms',type:'m',pills:['Upwork','Toptal','Fiverr','Freelancer.com','99designs','Guru','PeoplePerHour','Independent'],canAdd:1,defs:['Upwork']},
    {id:'ftype',label:'Contract type',type:'m',pills:['Short-term Projects','Long-term Retainers','Hourly','Fixed-price','Agency Subcontract'],defs:['Short-term Projects']},
    {id:'fskills',label:'Skills to market',type:'m',pills:['Web Development','Mobile Apps','Data Analysis','Graphic Design','Writing / Content','Video Editing','Consulting','Translation'],canAdd:1,defs:[]},
  ]},
  remote:{icon:'🌐',name:'Remote Work',qs:[
    {id:'rprefs',label:'Preferences',type:'m',pills:['Fully Remote','Hybrid','Async Teams','Digital Nomad','Time-zone Flexible'],defs:['Fully Remote']},
    {id:'rplatforms',label:'Remote job boards',type:'m',pills:['We Work Remotely','Remote.co','FlexJobs','Remotive','Arc.dev','Working Nomads'],canAdd:1,defs:['We Work Remotely']},
    {id:'rtools',label:'Collaboration tools',type:'m',pills:['Slack','Notion','Linear','Figma','VS Code Live Share','Zoom'],canAdd:1,defs:[]},
  ]},
  startup_jobs:{icon:'🚀',name:'Startup Jobs',qs:[
    {id:'sstage',label:'Startup stage',type:'m',pills:['Pre-seed / Bootstrapped','Seed','Series A-B','Growth / Scale-up','Pre-IPO'],defs:['Seed','Series A-B']},
    {id:'ssector',label:'Sectors',type:'m',pills:['AI / ML','Fintech','HealthTech','EdTech','SaaS','Climate Tech','Web3','Cybersecurity','DevTools'],canAdd:1,defs:['AI / ML','SaaS']},
    {id:'splatforms',label:'Where to look',type:'m',pills:['AngelList','Y Combinator Jobs','Crunchbase','Wellfound','Hacker News: Who is Hiring'],canAdd:1,defs:['AngelList']},
  ]},
  executive:{icon:'👔',name:'Executive & Leadership',qs:[
    {id:'elevel',label:'Level',type:'m',pills:['VP / Director','C-Suite','Board / Advisory','Partner','Managing Director'],defs:['VP / Director']},
    {id:'eplatforms',label:'Executive networks',type:'m',pills:['LinkedIn Premium','ExecThread','Egon Zehnder','Spencer Stuart','Korn Ferry','Heidrick & Struggles'],canAdd:1,defs:[]},
    {id:'efocus',label:'Focus areas',type:'m',pills:['Strategy & Growth','Digital Transformation','M&A','Operational Excellence','Talent & Culture'],defs:['Strategy & Growth']},
  ]},
  teaching_pos:{icon:'🍎',name:'Teaching Positions',qs:[
    {id:'tlevel',label:'Level',type:'m',pills:['K-12','Community College','University / College','Vocational','Online / EdTech','Corporate Training'],defs:['University / College']},
    {id:'tsubject',label:'Subject area',type:'m',pills:['STEM','Humanities','Business','Arts','Languages','Special Education','Computer Science','Mathematics'],canAdd:1,defs:[]},
    {id:'tplatforms',label:'Job boards',type:'m',pills:['HigherEdJobs','SchoolSpring','Teach Away','Indeed Education','Chronicle Vitae'],canAdd:1,defs:[]},
  ]},
  creative_jobs:{icon:'🎨',name:'Creative & Design Jobs',qs:[
    {id:'cjtype',label:'Type',type:'m',pills:['UI/UX Design','Graphic Design','Motion / Video','Brand / Marketing','Game Art','3D / AR / VR','Photography','Illustration'],canAdd:1,defs:['UI/UX Design']},
    {id:'cjplatforms',label:'Portfolio / job sites',type:'m',pills:['Dribbble','Behance','Coroflot','Carbonmade','Working Not Working','Creativepool'],canAdd:1,defs:['Dribbble','Behance']},
    {id:'cjtools',label:'Tools',type:'m',pills:['Figma','Adobe Creative Suite','Blender','After Effects','Sketch','Cinema 4D'],canAdd:1,defs:[]},
  ]},
  trade_jobs:{icon:'🔨',name:'Skilled Trades',qs:[
    {id:'tjtype',label:'Trade',type:'m',pills:['Electrical','Plumbing','HVAC','Welding','Carpentry','Automotive','Heavy Equipment','CNC / Machining','Pipefitting','Masonry'],canAdd:1,defs:[]},
    {id:'tjcerts',label:'Certifications',type:'m',pills:['Journeyman License','Master License','OSHA Safety','EPA Certification','Welding Certification','CDL'],canAdd:1,defs:[]},
  ]},
  /* -- Industry subs -- */
  ind_oilgas:{icon:'🛢️',name:'Oil & Gas',qs:[
    {id:'ogfocus',label:'Sub-focus',type:'m',pills:['Upstream Exploration','Downstream Refining','Petrochemicals','OPEC Policy','Energy Transition','LNG','Offshore Drilling','Pipeline Infrastructure'],defs:['Downstream Refining','Petrochemicals']},
    {id:'ogco',label:'Companies to track',hint:'Use + Add or AI suggestions for local companies',type:'m',pills:['ExxonMobil','Shell','Chevron','BP','TotalEnergies','Saudi Aramco','ConocoPhillips','Schlumberger','Halliburton','Baker Hughes'],canAdd:1,defs:[]},
    {id:'ogregion',label:'Regions',type:'m',pills:['Middle East','North America','North Sea','West Africa','Latin America','Caspian'],canAdd:1,defs:[]},
  ]},
  ind_telecom:{icon:'📡',name:'Telecom',qs:[
    {id:'telfocus',label:'Sub-focus',type:'m',pills:['5G / 6G Rollout','Fiber / Infrastructure','Mobile Services','Enterprise Solutions','Satellite / LEO','IoT Connectivity','Network Security'],defs:['5G / 6G Rollout']},
    {id:'telco',label:'Companies to track',type:'m',pills:['Verizon','AT&T','T-Mobile','Vodafone','Deutsche Telekom','STC','Zain','Ericsson','Nokia Networks','Qualcomm'],canAdd:1,defs:[]},
  ]},
  ind_tech:{icon:'💻',name:'Tech / Software',qs:[
    {id:'techfocus',label:'Sub-focus',type:'m',pills:['Cloud & SaaS','AI & Automation','Cybersecurity','Fintech','GovTech','DevOps & Infrastructure','Mobile / Apps','Blockchain / Web3','Open Source','Quantum Computing'],defs:['Cloud & SaaS','AI & Automation']},
    {id:'techco',label:'Companies to track',type:'m',pills:['Microsoft','Google','Apple','Amazon','Meta','NVIDIA','Salesforce','Oracle','IBM','Palantir','OpenAI','Anthropic'],canAdd:1,defs:[]},
    {id:'techlang',label:'Technologies',type:'m',pills:['Python','JavaScript / TypeScript','Rust','Go','Java','Kubernetes','Terraform','React','PyTorch','LangChain'],canAdd:1,defs:[]},
  ]},
  ind_bank:{icon:'🏦',name:'Banking & Finance',qs:[
    {id:'bfocus',label:'Sub-focus',type:'m',pills:['Retail Banking','Islamic Finance','Investment Banking','Fintech Disruption','Central Bank Policy','Wealth Management','Insurance','Payments & Processing','Regulatory / Basel'],defs:['Retail Banking']},
    {id:'bco',label:'Institutions to track',type:'m',pills:['JPMorgan Chase','Goldman Sachs','HSBC','Citigroup','Deutsche Bank','Morgan Stanley','Barclays','UBS','BNP Paribas','Bank of America'],canAdd:1,defs:[]},
    {id:'bplatforms',label:'Platforms to track',type:'m',pills:['Stripe','Square','PayPal','Revolut','Wise','Robinhood','Plaid','Marqeta'],canAdd:1,defs:[]},
  ]},
  ind_construct:{icon:'🏗️',name:'Construction',qs:[
    {id:'confocus',label:'Sub-focus',type:'m',pills:['Mega Projects','Smart Infrastructure','Residential','Government Tenders','Green Building','BIM / Digital Twin','Modular / Prefab','Road & Bridge'],defs:['Mega Projects']},
    {id:'conco',label:'Companies to track',type:'m',pills:['Bechtel','Vinci','ACS Group','AECOM','Fluor','Skanska','Bouygues','Turner Construction'],canAdd:1,defs:[]},
    {id:'contools',label:'Standards / tools',type:'m',pills:['LEED Certification','AutoCAD','Revit','Procore','PlanGrid','Building Codes'],canAdd:1,defs:[]},
  ]},
  ind_health:{icon:'🏥',name:'Healthcare',qs:[
    {id:'hfocus',label:'Sub-focus',type:'m',pills:['HealthTech','Pharmaceuticals','Digital Health','Medical Devices','Telemedicine','Clinical Research','Hospital Systems','Health Insurance','Mental Health','Genomics'],defs:['HealthTech']},
    {id:'hco',label:'Companies / orgs to track',type:'m',pills:['UnitedHealth','Johnson & Johnson','Medtronic','Abbott Labs','Siemens Healthineers','Epic Systems','Cerner','WHO','Mayo Clinic'],canAdd:1,defs:[]},
    {id:'hjournals',label:'Journals & sources',type:'m',pills:['NEJM','The Lancet','JAMA','BMJ','PubMed','Cochrane Library'],canAdd:1,defs:[]},
  ]},
  ind_auto:{icon:'🚗',name:'Automotive',qs:[
    {id:'autofocus',label:'Sub-focus',type:'m',pills:['EVs & Battery Tech','Autonomous Driving','Manufacturing','Motorsport','Connected Cars','Aftermarket','Commercial Vehicles','Ride-sharing'],defs:['EVs & Battery Tech']},
    {id:'autoco',label:'Companies to track',type:'m',pills:['Tesla','Toyota','Volkswagen','BMW','BYD','Rivian','GM','Ford','Hyundai','Waymo','Lucid Motors'],canAdd:1,defs:[]},
  ]},
  ind_aero:{icon:'✈️',name:'Aerospace & Defense',qs:[
    {id:'aerofocus',label:'Sub-focus',type:'m',pills:['Commercial Aviation','Defense Contracts','Space / Launch','Drones / UAVs','Satellites','Air Traffic Systems','Military Tech'],defs:['Commercial Aviation']},
    {id:'aeroco',label:'Companies to track',type:'m',pills:['Boeing','Airbus','Lockheed Martin','SpaceX','Raytheon','Northrop Grumman','General Dynamics','L3Harris','Blue Origin','BAE Systems'],canAdd:1,defs:[]},
  ]},
  ind_retail:{icon:'🛒',name:'Retail & E-Commerce',qs:[
    {id:'retailfocus',label:'Sub-focus',type:'m',pills:['E-Commerce','Brick & Mortar','D2C Brands','Marketplace','Luxury / Fashion','Grocery / FMCG','Supply Chain','Retail Tech'],defs:['E-Commerce']},
    {id:'retailco',label:'Companies to track',type:'m',pills:['Amazon','Shopify','Walmart','Alibaba','Target','Costco','Etsy','Shein','Zara / Inditex','Nike'],canAdd:1,defs:[]},
  ]},
  ind_media:{icon:'🎬',name:'Media & Entertainment',qs:[
    {id:'mediafocus',label:'Sub-focus',type:'m',pills:['Streaming','News / Publishing','Advertising','Gaming','Social Media','Film / TV Production','Music Industry','Podcasting'],defs:['Streaming','Social Media']},
    {id:'mediaco',label:'Companies to track',type:'m',pills:['Netflix','Disney','Spotify','Warner Bros Discovery','TikTok / ByteDance','YouTube','Activision Blizzard','Electronic Arts','The New York Times','Substack'],canAdd:1,defs:[]},
  ]},
  ind_food:{icon:'🌾',name:'Food & Agriculture',qs:[
    {id:'foodfocus',label:'Sub-focus',type:'m',pills:['AgriTech','Food Processing','Supply Chain','Organic / Sustainability','Precision Farming','Vertical Farming','Plant-based / Alt Protein','Food Safety'],defs:['AgriTech']},
    {id:'foodco',label:'Companies to track',type:'m',pills:['Cargill','ADM','John Deere','Corteva','Bayer Crop Science','Beyond Meat','Impossible Foods','Syngenta'],canAdd:1,defs:[]},
  ]},
  ind_edu:{icon:'🏫',name:'Education & EdTech',qs:[
    {id:'edufocus',label:'Sub-focus',type:'m',pills:['K-12','Higher Education','EdTech Platforms','Corporate Training','AI in Education','Student Assessment','LMS Systems','Education Policy'],defs:['Higher Education','EdTech Platforms']},
    {id:'educo',label:'Companies to track',type:'m',pills:['Coursera','Duolingo','Khan Academy','Pearson','Chegg','Canvas / Instructure','Blackboard','2U','Byju\'s','Udemy'],canAdd:1,defs:[]},
  ]},
  ind_legal:{icon:'⚖️',name:'Legal & Compliance',qs:[
    {id:'legalfocus',label:'Sub-focus',type:'m',pills:['Corporate Law','IP / Patents','Regulatory Compliance','LegalTech','Employment Law','Data Privacy / GDPR','International Trade','Tax Law','M&A'],defs:['Corporate Law']},
    {id:'legalco',label:'Firms / platforms to track',type:'m',pills:['Clio','LegalZoom','Thomson Reuters Legal','LexisNexis','Harvey AI','Latham & Watkins','Baker McKenzie','Allen & Overy'],canAdd:1,defs:[]},
  ]},
  ind_energy:{icon:'⚡',name:'Energy & Renewables',qs:[
    {id:'enfocus',label:'Sub-focus',type:'m',pills:['Solar','Wind','Nuclear','Grid / Infrastructure','Energy Storage','Hydrogen','Carbon Capture','Smart Grid','Geothermal'],defs:['Solar','Wind']},
    {id:'enco',label:'Companies to track',type:'m',pills:['NextEra Energy','Enphase','First Solar','Vestas','Siemens Energy','Brookfield Renewable','Plug Power','ChargePoint','Sunrun'],canAdd:1,defs:[]},
  ]},
  ind_logistics:{icon:'📦',name:'Logistics & Supply Chain',qs:[
    {id:'logfocus',label:'Sub-focus',type:'m',pills:['Shipping / Freight','Warehousing','Last-mile Delivery','Supply Chain Tech','Cold Chain','Customs / Trade','Fleet Management','Reverse Logistics'],defs:['Supply Chain Tech']},
    {id:'logco',label:'Companies to track',type:'m',pills:['FedEx','UPS','Maersk','DHL','Flexport','C.H. Robinson','XPO Logistics','Amazon Logistics','Kuehne + Nagel'],canAdd:1,defs:[]},
  ]},
  ind_pharma:{icon:'💊',name:'Pharmaceuticals',qs:[
    {id:'pharmfocus',label:'Sub-focus',type:'m',pills:['Drug Discovery','Clinical Trials','Biotech','Generics','Regulatory / FDA','mRNA Technology','Gene Therapy','Biosimilars','CRO / CDMO'],defs:['Drug Discovery']},
    {id:'pharmco',label:'Companies to track',type:'m',pills:['Pfizer','Roche','Johnson & Johnson','Novartis','AstraZeneca','Eli Lilly','Merck','Moderna','Gilead','Amgen','BioNTech'],canAdd:1,defs:[]},
  ]},
  ind_realestate:{icon:'🏘️',name:'Real Estate & PropTech',qs:[
    {id:'refocus2',label:'Sub-focus',type:'m',pills:['Residential','Commercial','PropTech','REITs','Development','Property Management','Co-working','Real Estate Finance'],defs:['Residential']},
    {id:'reco',label:'Platforms / companies',type:'m',pills:['Zillow','Redfin','CoStar','WeWork','Compass','Opendoor','Matterport','Prologis'],canAdd:1,defs:[]},
  ]},
  ind_gaming:{icon:'🎮',name:'Gaming & Esports',qs:[
    {id:'gamefocus',label:'Sub-focus',type:'m',pills:['Game Development','Esports','Streaming','Mobile Gaming','Console / PC','VR / AR Gaming','Game Engines','Indie Games'],defs:['Game Development']},
    {id:'gameco',label:'Companies to track',type:'m',pills:['Nintendo','Sony PlayStation','Xbox / Microsoft','Valve / Steam','Epic Games','Riot Games','miHoYo','Unity','Unreal Engine','Roblox'],canAdd:1,defs:[]},
  ]},
  /* -- Learning subs -- */
  certs:{icon:'📜',name:'Professional Certifications',qs:[
    {id:'clevel',label:'Level',type:'s',pills:['Beginner / Entry','Intermediate','Advanced / Expert'],def:'Beginner / Entry'},
    {id:'cfocus',label:'Focus',type:'m',pills:['Cloud & Infrastructure','Networking','Security','Data & AI','Project Management','Agile / Scrum','DevOps','Database','Business Analysis','Financial'],canAdd:1,defs:['Cloud & Infrastructure','Networking']},
    {id:'ccerts',label:'Specific certs',type:'m',pills:['AWS Solutions Architect','Azure Administrator','CompTIA A+','CCNA','CISSP','PMP','Google Cloud','Terraform','CKA (Kubernetes)','ITIL','CFA','CPA','Six Sigma'],canAdd:1,defs:[]},
  ]},
  academic:{icon:'📚',name:'Academic Research',qs:[
    {id:'afields',label:'Fields',type:'m',pills:['Computer Science','AI / ML','Electrical Engineering','Mechanical Engineering','Biomedical','Physics','Chemistry','Economics','Psychology','Environmental Science','Mathematics','Materials Science'],canAdd:1,defs:['Computer Science','AI / ML']},
    {id:'asrc',label:'Sources',type:'m',pills:['IEEE','ACM','arXiv','Google Scholar','PubMed','Nature','Science','Springer','Elsevier','SSRN'],defs:['IEEE','arXiv']},
    {id:'atools',label:'Research tools',type:'m',pills:['LaTeX / Overleaf','Zotero','Mendeley','MATLAB','R / RStudio','Jupyter','Python / SciPy'],canAdd:1,defs:[]},
  ]},
  courses:{icon:'🖥️',name:'Online Courses',qs:[
    {id:'cplat',label:'Platforms',type:'m',pills:['Coursera','Udemy','edX','Pluralsight','LinkedIn Learning','Codecademy','freeCodeCamp','MIT OpenCourseWare','Brilliant','MasterClass','DataCamp','fast.ai'],defs:['Coursera','Udemy']},
    {id:'ctop',label:'Topics',type:'m',pills:['Cloud Computing','Data Science','Web Development','DevOps','Mobile Development','AI / Machine Learning','Cybersecurity','Blockchain','Design','Business / MBA','Finance','Marketing'],canAdd:1,defs:[]},
  ]},
  hands_on:{icon:'🔧',name:'Hands-on Skills',qs:[
    {id:'hskills',label:'Skills',type:'m',pills:['Linux Administration','Networking Labs','Docker / K8s','Microcontrollers','Raspberry Pi','Arduino','PCB Design','3D Printing','Soldering','PLC Programming','CNC Operation','Pen Testing Labs'],canAdd:1,defs:['Linux Administration']},
    {id:'hplatforms',label:'Lab platforms',type:'m',pills:['TryHackMe','HackTheBox','AWS Labs','Katacoda','LeetCode','HackerRank','LabEx','Cisco Packet Tracer'],canAdd:1,defs:[]},
  ]},
  confevents:{icon:'🎤',name:'Conferences & Events',qs:[
    {id:'etypes',label:'Types',type:'m',pills:['Career Fairs','Tech Conferences','Hackathons','Meetups','Webinars','Trade Shows','Academic Symposiums','Industry Summits','Networking Events'],defs:['Tech Conferences','Hackathons']},
    {id:'eregion',label:'Region',type:'s',pills:['Local / National','Regional','Global','Online Only'],def:'Global'},
    {id:'enamed',label:'Specific events',type:'m',pills:['CES','AWS re:Invent','Google I/O','WWDC','GTC (NVIDIA)','RSA Conference','Web Summit','SXSW','Gartner Symposium'],canAdd:1,defs:[]},
  ]},
  bootcamps:{icon:'🏕️',name:'Bootcamps & Intensives',qs:[
    {id:'bctype',label:'Focus',type:'m',pills:['Full-Stack Web','Data Science','UX / UI Design','Cybersecurity','Mobile Development','DevOps','AI / ML','Product Management'],canAdd:1,defs:['Full-Stack Web']},
    {id:'bcproviders',label:'Providers',type:'m',pills:['General Assembly','Flatiron School','Le Wagon','App Academy','Lambda School','Springboard','Ironhack','Thinkful','BrainStation'],canAdd:1,defs:[]},
    {id:'bcformat',label:'Format',type:'m',pills:['Full-time On-site','Full-time Remote','Part-time / Flex','Self-paced'],defs:['Full-time Remote']},
  ]},
  workshops:{icon:'🛠️',name:'Workshops & Seminars',qs:[
    {id:'wstype',label:'Format',type:'m',pills:['In-person','Virtual','Multi-day','One-day Intensive','Lunch & Learn','Weekend Workshop'],defs:['Virtual']},
    {id:'wstopic',label:'Topic areas',type:'m',pills:['Technical Skills','Leadership','Communication','Design Thinking','Agile / Scrum','Product Strategy','Data Literacy','Writing / Documentation'],canAdd:1,defs:[]},
  ]},
  mentorship:{icon:'🤝',name:'Mentorship & Coaching',qs:[
    {id:'mentype',label:'Type',type:'m',pills:['1-on-1 Mentorship','Group Coaching','Career Coaching','Technical Mentorship','Executive Coaching','Peer Mentoring'],defs:['1-on-1 Mentorship']},
    {id:'menplatforms',label:'Platforms',type:'m',pills:['ADPList','MentorCruise','GrowthMentor','Plato','Clarity.fm','Score.org'],canAdd:1,defs:[]},
  ]},
  open_source:{icon:'🐙',name:'Open Source & Community',qs:[
    {id:'osfocus',label:'Focus',type:'m',pills:['Contributing','Project Discovery','Community Events','Maintainership','Documentation','Issue Triage'],defs:['Contributing','Project Discovery']},
    {id:'osprojects',label:'Ecosystems',type:'m',pills:['Linux / Kernel','React / Next.js','Python / PyPI','Rust Ecosystem','Kubernetes','Homebrew','Apache Foundation','CNCF Projects'],canAdd:1,defs:[]},
    {id:'osplatforms',label:'Platforms',type:'m',pills:['GitHub','GitLab','SourceForge','Open Source Friday','Hacktoberfest','Google Summer of Code'],defs:['GitHub']},
  ]},
  competitions:{icon:'🏆',name:'Competitions & Hackathons',qs:[
    {id:'comptype',label:'Type',type:'m',pills:['Hackathons','Coding Competitions','CTFs','Data Challenges','Design Challenges','Robotics Competitions','Business Plan','Game Jams'],canAdd:1,defs:['Hackathons','Coding Competitions']},
    {id:'compplatforms',label:'Platforms',type:'m',pills:['Devpost','MLH','Kaggle','Codeforces','LeetCode','HackerRank','TopCoder','CTFtime','itch.io Game Jams'],canAdd:1,defs:['Devpost','Kaggle']},
  ]},
  fellowships:{icon:'🎖️',name:'Fellowships & Grants',qs:[
    {id:'feltype',label:'Type',type:'m',pills:['Research Fellowships','Industry Fellowships','Government Grants','Travel Grants','Startup Grants','Diversity Fellowships','Teaching Fellowships'],defs:['Research Fellowships']},
    {id:'felproviders',label:'Known programs',type:'m',pills:['Fulbright','Rhodes','Marshall','NSF Graduate','Google PhD Fellowship','Microsoft Research','Facebook Fellowship','Thiel Fellowship','Y Combinator'],canAdd:1,defs:[]},
  ]},
  /* -- Markets subs -- */
  stocks:{icon:'📊',name:'Stocks',qs:[
    {id:'smkt',label:'Markets',type:'m',pills:['US (NYSE/NASDAQ)','European Markets','Asian Markets','Middle East (Tadawul/ADX)','Global Indices','Emerging Markets'],canAdd:1,defs:['US (NYSE/NASDAQ)']},
    {id:'ssector',label:'Sectors',type:'m',pills:['Technology','Healthcare','Financial','Energy','Consumer','Industrial','Real Estate','Utilities'],canAdd:1,defs:[]},
    {id:'sdepth',label:'Tracking level',type:'s',pills:['Just notable news','Daily price tracking','Deep analysis'],def:'Just notable news'},
  ]},
  crypto:{icon:'₿',name:'Crypto',qs:[
    {id:'ccoins',label:'What to track',type:'m',pills:['Bitcoin','Ethereum','Solana','Altcoins','DeFi','NFTs','Web3','Stablecoins','Layer 2s','Memecoins'],canAdd:1,defs:['Bitcoin','Ethereum']},
    {id:'cexchanges',label:'Exchanges',type:'m',pills:['Coinbase','Binance','Kraken','Uniswap','dYdX','Gemini'],canAdd:1,defs:[]},
    {id:'cdepth',label:'Tracking level',type:'s',pills:['Just notable news','Daily price tracking'],def:'Just notable news'},
  ]},
  commodities:{icon:'🪙',name:'Commodities',qs:[
    {id:'comms',label:'Which',type:'m',pills:['Gold','Silver','Oil (Brent/WTI)','Natural Gas','Copper','Platinum','Palladium','Lithium','Uranium','Agricultural (Wheat/Corn/Soy)'],defs:['Gold','Oil (Brent/WTI)']},
  ]},
  realestate:{icon:'🏠',name:'Real Estate',qs:[
    {id:'refocus',label:'Focus',type:'m',pills:['Residential','Commercial','REITs','Industrial','International Markets','Rental Market','Mortgage Rates','Housing Data'],canAdd:1,defs:['Residential']},
    {id:'reregion',label:'Regions',type:'m',pills:['US','Europe','Middle East','Asia Pacific','UK','Canada','Australia'],canAdd:1,defs:[]},
  ]},
  forex:{icon:'💱',name:'Forex',qs:[
    {id:'fpairs',label:'Pairs',type:'m',pills:['EUR/USD','GBP/USD','USD/JPY','USD/CHF','AUD/USD','USD/CAD','EUR/GBP','USD/SAR','USD/KWD'],canAdd:1,defs:['EUR/USD']},
    {id:'ffocus',label:'Focus',type:'m',pills:['Major Pairs','Emerging Market','Central Bank Policy','Technical Analysis','Carry Trades'],defs:['Major Pairs']},
  ]},
  mktreports:{icon:'📋',name:'Market Research & Reports',qs:[
    {id:'mrsrc',label:'Sources',type:'m',pills:['Bloomberg','Reuters','Morningstar','S&P Global','McKinsey','Gartner','Forrester','CB Insights','PitchBook','Statista'],canAdd:1,defs:['Bloomberg']},
    {id:'mrfocus',label:'Report types',type:'m',pills:['Equity Research','Industry Reports','Economic Outlook','Earnings Analysis','IPO Reports','Macro Strategy'],defs:['Industry Reports']},
  ]},
  etfs:{icon:'📦',name:'ETFs & Index Funds',qs:[
    {id:'etffocus',label:'Focus',type:'m',pills:['S&P 500 (SPY/VOO)','Tech (QQQ)','Bond ETFs (BND)','Emerging Markets','Dividend ETFs','Sector ETFs','Thematic (ARK)','International','ESG / Sustainable'],defs:['S&P 500 (SPY/VOO)']},
    {id:'etfproviders',label:'Providers',type:'m',pills:['Vanguard','BlackRock (iShares)','State Street (SPDR)','Invesco','ARK Invest','Schwab'],canAdd:1,defs:[]},
  ]},
  bonds:{icon:'🏛️',name:'Bonds & Fixed Income',qs:[
    {id:'bondfocus',label:'Focus',type:'m',pills:['US Treasury','Corporate Bonds','Municipal Bonds','High Yield / Junk','International Sovereign','Sukuk (Islamic Bonds)','Inflation-Protected (TIPS)'],defs:['US Treasury']},
    {id:'bondmetrics',label:'Metrics to track',type:'m',pills:['Yield Curves','Credit Spreads','Fed Funds Rate','10-Year Treasury','Bond Auctions'],defs:['Yield Curves']},
  ]},
  startups_vc:{icon:'💰',name:'Startups & Venture Capital',qs:[
    {id:'vcfocus',label:'Focus',type:'m',pills:['Seed / Angel','Series A-C','IPO Pipeline','SPAC','Sector: AI','Sector: Fintech','Sector: Climate','Sector: Health','Sector: SaaS'],defs:['Seed / Angel']},
    {id:'vcfirms',label:'VC firms to track',type:'m',pills:['Sequoia','Andreessen Horowitz (a16z)','Y Combinator','Accel','Benchmark','Tiger Global','SoftBank Vision','Khosla Ventures','First Round','Founders Fund'],canAdd:1,defs:[]},
    {id:'vcsources',label:'Sources',type:'m',pills:['Crunchbase','PitchBook','CB Insights','TechCrunch','The Information','AngelList'],defs:['Crunchbase','TechCrunch']},
  ]},
  options:{icon:'📈',name:'Options & Derivatives',qs:[
    {id:'optfocus',label:'Focus',type:'m',pills:['Options Strategies','Futures','Swaps','Risk Management','Volatility (VIX)','Commodities Futures','Index Options'],defs:['Options Strategies']},
    {id:'optplatforms',label:'Platforms',type:'m',pills:['ThinkorSwim','Interactive Brokers','Tastytrade','Robinhood','CME Group','CBOE'],canAdd:1,defs:[]},
  ]},
  /* -- Deals subs -- */
  bankdeals:{icon:'🏦',name:'Bank Deals',qs:[
    {id:'bdkinds',label:'What kind',type:'m',pills:['Salary / Allowance Transfer','New Account Bonuses','Credit Card Rewards','Loan Offers','Savings / Deposit Rates','Mortgage Deals','Fee Waivers'],defs:['Salary / Allowance Transfer','New Account Bonuses']},
    {id:'bdbanks',label:'Banks to track',type:'m',pills:['NBK','KFH','Boubyan','Gulf Bank','Warba Bank','Burgan Bank','ABK','CBK'],canAdd:1,defs:[]},
  ]},
  telepromo:{icon:'📱',name:'Telecom Promotions',qs:[
    {id:'tpkinds',label:'Types',type:'m',pills:['Data Plans','Device Deals','Roaming Offers','Bundle Packages','5G Plans','Fiber / Home Internet','Family Plans','Prepaid Deals'],defs:['Data Plans','Device Deals']},
    {id:'tpproviders',label:'Providers',type:'m',pills:['Zain','Ooredoo','STC','Verizon','AT&T','T-Mobile','Vodafone','Etisalat'],canAdd:1,defs:[]},
  ]},
  studisc:{icon:'🎓',name:'Student Discounts',qs:[
    {id:'sdtypes',label:'Types',type:'m',pills:['Telecom Deals','Tech / Software','Food & Lifestyle','Travel','Entertainment','Education / Courses','Banking','Cloud Credits'],canAdd:1,defs:['Tech / Software','Education / Courses']},
    {id:'sdplatforms',label:'Discount platforms',type:'m',pills:['GitHub Student Pack','UNiDAYS','Student Beans','SheerID','Apple Education','Microsoft Imagine'],canAdd:1,defs:['GitHub Student Pack']},
  ]},
  ccrewards:{icon:'💳',name:'Credit Card Rewards',qs:[
    {id:'cctypes',label:'Reward types',type:'m',pills:['Cashback','Travel Miles','Points / Loyalty','Dining','Shopping','Airport Lounge','Hotel Status','No Annual Fee'],defs:['Cashback']},
    {id:'ccnetworks',label:'Networks',type:'m',pills:['Visa','Mastercard','Amex','Discover','Diners Club'],defs:[]},
  ]},
  empben:{icon:'🏢',name:'Employee Benefits',qs:[
    {id:'ebtypes',label:'Types',type:'m',pills:['Health Insurance','Education Allowance','Housing','Travel Perks','Stock Options / RSU','401k / Pension','Gym / Wellness','Parental Leave','Remote Work Stipend'],defs:[]},
  ]},
  techdeals:{icon:'🖥️',name:'Tech & Software Deals',qs:[
    {id:'tdfocus',label:'Type',type:'m',pills:['SaaS Discounts','Hardware Deals','Developer Tools','Cloud Credits','Black Friday / Cyber Monday','Lifetime Deals','Open Source Alternatives'],defs:['SaaS Discounts','Developer Tools']},
    {id:'tdplatforms',label:'Where to find',type:'m',pills:['AppSumo','Product Hunt','Hacker News Deals','Slickdeals','CamelCamelCamel','Honey'],canAdd:1,defs:[]},
  ]},
  traveldeals:{icon:'✈️',name:'Travel & Hotel Deals',qs:[
    {id:'travfocus',label:'Type',type:'m',pills:['Flight Deals','Hotel Deals','Loyalty Programs','Travel Hacks','Error Fares','Points Optimization','Vacation Packages','Cruise Deals'],defs:['Flight Deals']},
    {id:'travplatforms',label:'Platforms',type:'m',pills:['Google Flights','Skyscanner','Secret Flying','The Points Guy','Scott\'s Cheap Flights','Hopper','Award Wallet'],canAdd:1,defs:[]},
  ]},
  insurance:{icon:'🛡️',name:'Insurance Offers',qs:[
    {id:'insfocus',label:'Type',type:'m',pills:['Health Insurance','Car Insurance','Life Insurance','Travel Insurance','Home / Renters','Pet Insurance','Professional Liability','Disability'],defs:['Health Insurance']},
  ]},
  cashback:{icon:'💵',name:'Cashback & Loyalty',qs:[
    {id:'cbfocus',label:'Type',type:'m',pills:['Cashback Apps','Loyalty Programs','Shopping Portals','Browser Extensions','Receipt Scanning','Dining Rewards','Gas Rewards'],defs:['Cashback Apps']},
    {id:'cbplatforms',label:'Platforms',type:'m',pills:['Rakuten','Honey','TopCashback','Ibotta','Fetch Rewards','Dosh','RetailMeNot'],canAdd:1,defs:[]},
  ]},
  subscriptions:{icon:'📺',name:'Subscription Deals',qs:[
    {id:'subfocus',label:'Type',type:'m',pills:['Streaming (Video)','Music','Cloud Storage','News / Magazines','Productivity','Gaming','VPN','Meal Kits','Fitness'],defs:['Streaming (Video)']},
    {id:'subtracker',label:'Subscription tracking',type:'m',pills:['Truebill / Rocket Money','Bobby','TrackMySubs','Trim'],canAdd:1,defs:[]},
  ]},
};

const WIZ_TICKER_MAP = {
  'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD',
  'Gold': 'GC=F', 'Silver': 'SI=F', 'Oil (Brent/WTI)': 'CL=F',
  'Natural Gas': 'NG=F', 'Copper': 'HG=F',
  'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X',
  'USD/JPY': 'JPY=X', 'USD/CHF': 'CHF=X',
  'US (NYSE/NASDAQ)': '^GSPC', 'European Markets': '^STOXX50E',
  'Asian Markets': '^N225', 'Global Indices': '^GSPC',
};

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

const DOMAIN_PILLS = {
  roles: {
    tech:['Software Development','DevOps / SRE','Cloud Architecture','Full Stack Development','Backend / API'],
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
    tech:['Software Development','Cloud / DevOps','Mobile Development','Data Engineering'],
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
  tech:{career:['jobhunt','remote','startup_jobs'],industry:['ind_tech'],learning:['certs','courses','open_source','competitions']},
  data:{career:['jobhunt','remote'],industry:['ind_tech'],learning:['courses','academic','competitions']},
  cybersec:{career:['jobhunt','remote'],industry:['ind_tech'],learning:['certs','hands_on','competitions']},
  finance:{markets:['stocks','forex','bonds','etfs'],career:['jobhunt'],industry:['ind_bank']},
  marketing:{career:['jobhunt','freelance'],industry:['ind_media','ind_retail'],interests:[]},
  sales:{career:['jobhunt'],industry:['ind_retail'],deals:['bankdeals','ccrewards']},
  healthcare:{career:['jobhunt'],industry:['ind_health','ind_pharma'],learning:['certs','academic']},
  education:{career:['teaching_pos','research_pos'],industry:['ind_edu'],learning:['academic','courses','mentorship']},
  legal:{career:['jobhunt'],industry:['ind_legal'],learning:['certs','confevents']},
  creative:{career:['freelance','creative_jobs'],industry:['ind_media'],learning:['courses','workshops']},
  hr:{career:['jobhunt'],learning:['certs','courses','workshops']},
  consulting:{career:['jobhunt','freelance'],learning:['certs','confevents']},
  operations:{career:['jobhunt'],industry:['ind_logistics','ind_construct'],learning:['certs']},
  realestate:{markets:['realestate','stocks'],career:['jobhunt'],industry:['ind_realestate','ind_construct']},
  hospitality:{career:['jobhunt'],industry:['ind_food'],deals:['traveldeals','empben']},
  government:{career:['govjobs'],learning:['certs','confevents']},
  trades:{career:['trade_jobs'],deals:['techdeals'],learning:['hands_on','bootcamps']},
  science:{career:['research_pos'],learning:['academic','fellowships','confevents']},
  agriculture:{career:['jobhunt'],industry:['ind_food'],markets:['commodities']},
  nonprofit:{career:['jobhunt'],industry:['ind_health','ind_edu'],learning:['fellowships']},
  retail:{career:['jobhunt'],industry:['ind_retail'],deals:['bankdeals','ccrewards','cashback']},
  engineering:{career:['jobhunt'],learning:['certs','academic','confevents'],industry:['ind_construct','ind_energy']},
  default:{career:['jobhunt','intern'],learning:['certs'],deals:['bankdeals','studisc']},
};

return { CATS, INTEREST_SUGGESTIONS, DEF_CATS, DEF_SUBS, PANELS, WIZ_TICKER_MAP,
         ROLE_KEYWORDS, DOMAIN_PILLS, DOMAIN_CAT_MAP, DOMAIN_SUB_MAP };
})();
