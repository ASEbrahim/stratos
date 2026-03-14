// ═══════════════════════════════════════════════════════════
// settings-sources.js — News Source Catalog & Custom Feeds
// Extracted from settings.js for modularity.
// Depends on globals from settings.js: configData, esc(),
//   showToast(), getAuthToken(), rebuildNavFromConfig(), setActive()
// ═══════════════════════════════════════════════════════════
'use strict';

// ═══════════════════════════════════════════════════════════
// NEWS SOURCE MANAGEMENT (Finance & Politics feed toggles)
// ═══════════════════════════════════════════════════════════

let sourcesCatalog = { finance: [], politics: [], jobs: [] };
let customFeeds = []; // [{url, name, on}]  — only custom tab
let customTabName = 'Custom';
let sourcesActiveTab = 'finance';

// Fallback catalog — used when backend hasn't been updated yet
const FALLBACK_CATALOG = {
    finance: [
        {id:"cnbc_top",name:"CNBC Top News",region:"US",category:"markets",on:true},
        {id:"cnbc_finance",name:"CNBC Finance",region:"US",category:"markets",on:true},
        {id:"marketwatch",name:"MarketWatch",region:"US",category:"markets",on:true},
        {id:"mw_pulse",name:"MarketWatch Pulse",region:"US",category:"markets",on:false},
        {id:"yahoo_fin",name:"Yahoo Finance",region:"US",category:"markets",on:true},
        {id:"investing",name:"Investing.com",region:"Global",category:"markets",on:true},
        {id:"reuters_biz",name:"Reuters Business",region:"Global",category:"business",on:true},
        {id:"reuters_co",name:"Reuters Companies",region:"Global",category:"corporate",on:false},
        {id:"ft",name:"Financial Times",region:"UK",category:"business",on:true},
        {id:"bloomberg",name:"Bloomberg Markets",region:"Global",category:"markets",on:true},
        {id:"wsj_world",name:"WSJ / Dow Jones",region:"US",category:"business",on:false},
        {id:"economist",name:"The Economist",region:"Global",category:"business",on:false},
        {id:"barrons",name:"Barron's",region:"US",category:"markets",on:false},
        {id:"coindesk",name:"CoinDesk",region:"Global",category:"crypto",on:false},
        {id:"cointelegraph",name:"CoinTelegraph",region:"Global",category:"crypto",on:false},
        {id:"zawya",name:"Zawya",region:"GCC",category:"business",on:true},
        {id:"arabianbiz",name:"Arabian Business",region:"GCC",category:"business",on:false},
        {id:"gulfnews_biz",name:"Gulf News Business",region:"GCC",category:"business",on:false},
        {id:"oilprice",name:"OilPrice.com",region:"Global",category:"energy",on:false},
        {id:"rigzone",name:"Rigzone",region:"Global",category:"energy",on:false},
    ],
    politics: [
        {id:"bbc_world",name:"BBC World",region:"UK",category:"world",on:true},
        {id:"bbc_mideast",name:"BBC Middle East",region:"UK",category:"mideast",on:true},
        {id:"reuters_world",name:"Reuters World",region:"Global",category:"world",on:true},
        {id:"aljazeera",name:"Al Jazeera",region:"Qatar",category:"world",on:true},
        {id:"ap_top",name:"AP News",region:"US",category:"world",on:false},
        {id:"nyt_world",name:"NYT World",region:"US",category:"world",on:true},
        {id:"nyt_mideast",name:"NYT Middle East",region:"US",category:"mideast",on:true},
        {id:"wapo_world",name:"WaPo World",region:"US",category:"world",on:true},
        {id:"wapo_politics",name:"WaPo Politics",region:"US",category:"us_politics",on:false},
        {id:"cnn_world",name:"CNN World",region:"US",category:"world",on:false},
        {id:"fox_world",name:"Fox News World",region:"US",category:"world",on:false},
        {id:"guardian",name:"The Guardian",region:"UK",category:"world",on:false},
        {id:"dw",name:"DW News",region:"Germany",category:"world",on:false},
        {id:"france24",name:"France 24",region:"France",category:"world",on:false},
        {id:"kuwait_times",name:"Kuwait Times",region:"Kuwait",category:"local",on:true},
        {id:"arab_times",name:"Arab Times",region:"Kuwait",category:"local",on:true},
        {id:"kuna",name:"KUNA",region:"Kuwait",category:"local",on:false},
        {id:"gulfnews",name:"Gulf News",region:"UAE",category:"gcc",on:false},
        {id:"arabnews",name:"Arab News",region:"KSA",category:"gcc",on:false},
        {id:"middleeasteye",name:"Middle East Eye",region:"UK",category:"mideast",on:false},
        {id:"scmp",name:"SCMP",region:"HK",category:"asia",on:false},
        {id:"nikkei_asia",name:"Nikkei Asia",region:"Japan",category:"asia",on:false},
    ]
};

function showSourceTab(type) {
    sourcesActiveTab = type;
    document.getElementById('source-catalog-finance').classList.toggle('hidden', type !== 'finance');
    document.getElementById('source-catalog-politics').classList.toggle('hidden', type !== 'politics');
    document.getElementById('source-catalog-jobs')?.classList.toggle('hidden', type !== 'jobs');
    document.getElementById('source-catalog-custom').classList.toggle('hidden', type !== 'custom');

    const tabs = {
        finance:  { el: document.getElementById('src-tab-finance'),  active: 'px-5 py-2 text-xs font-semibold transition-all bg-emerald-500/20 text-emerald-400 border-r border-slate-600', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 border-r border-slate-600' },
        politics: { el: document.getElementById('src-tab-politics'), active: 'px-5 py-2 text-xs font-semibold transition-all bg-blue-500/20 text-blue-400 border-r border-slate-600', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 border-r border-slate-600' },
        jobs:     { el: document.getElementById('src-tab-jobs'),     active: 'px-5 py-2 text-xs font-semibold transition-all bg-amber-500/20 text-amber-400 border-r border-slate-600', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 border-r border-slate-600' },
        custom:   { el: document.getElementById('src-tab-custom'),   active: 'px-5 py-2 text-xs font-semibold transition-all bg-purple-500/20 text-purple-400 flex items-center gap-1', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 flex items-center gap-1' },
    };

    Object.entries(tabs).forEach(([key, cfg]) => {
        if (cfg.el) cfg.el.className = key === type ? cfg.active : cfg.inactive;
    });

    if (type === 'custom') renderCustomCatalog();
}

async function loadSourceCatalog() {
    // Show loading state
    ['finance', 'politics', 'jobs'].forEach(type => {
        const c = document.getElementById(`source-catalog-${type}`);
        if (c) c.innerHTML = '<p class="text-xs text-slate-500 italic py-2">Loading sources...</p>';
    });

    let usedFallback = false;

    try {
        const [finResp, polResp, jobResp] = await Promise.all([
            fetch('/api/feed-catalog/finance'),
            fetch('/api/feed-catalog/politics'),
            fetch('/api/feed-catalog/jobs')
        ]);

        if (!finResp.ok || !polResp.ok) throw new Error('Catalog endpoint returned error');

        const finData = await finResp.json();
        const polData = await polResp.json();
        const jobData = jobResp.ok ? await jobResp.json() : { catalog: [] };
        sourcesCatalog.finance = finData.catalog || [];
        sourcesCatalog.politics = polData.catalog || [];
        sourcesCatalog.jobs = jobData.catalog || [];
    } catch(e) {
        console.warn('Feed catalog API unavailable, using fallback:', e.message);
        usedFallback = true;
        // Use fallback — apply any saved toggles from configData
        sourcesCatalog.finance = FALLBACK_CATALOG.finance.map(f => ({...f}));
        sourcesCatalog.politics = FALLBACK_CATALOG.politics.map(f => ({...f}));

        if (configData) {
            const finToggles = configData.extra_feeds_finance || {};
            const polToggles = configData.extra_feeds_politics || {};
            // If profile has saved toggles, apply them; otherwise default all off
            if (Object.keys(finToggles).length > 0) {
                sourcesCatalog.finance.forEach(f => { if (f.id in finToggles) f.on = finToggles[f.id]; });
            } else {
                sourcesCatalog.finance.forEach(f => { f.on = false; });
            }
            if (Object.keys(polToggles).length > 0) {
                sourcesCatalog.politics.forEach(f => { if (f.id in polToggles) f.on = polToggles[f.id]; });
            } else {
                sourcesCatalog.politics.forEach(f => { f.on = false; });
            }
        }
    }

    // Load custom feeds from config
    if (configData) {
        customFeeds = configData.custom_feeds || [];
        // Migrate from old format if needed
        if (!customFeeds.length && (configData.custom_feeds_finance?.length || configData.custom_feeds_politics?.length)) {
            customFeeds = [
                ...(configData.custom_feeds_finance || []).map(f => ({...f, on: true})),
                ...(configData.custom_feeds_politics || []).map(f => ({...f, on: true})),
            ];
        }
        customTabName = configData.custom_tab_name || 'Custom';
        const label = document.getElementById('custom-tab-label');
        if (label) label.textContent = customTabName;
    }

    renderSourceCatalog('finance');
    renderSourceCatalog('politics');
    renderSourceCatalog('jobs');
    renderCustomCatalog();
}

function renderSourceCatalog(type) {
    const container = document.getElementById(`source-catalog-${type}`);
    if (!container) return;

    const items = sourcesCatalog[type] || [];
    if (!items.length) {
        container.innerHTML = '<p class="text-xs text-slate-500 italic py-2">No sources available.</p>';
        return;
    }

    // Group by category instead of region for cleaner UX
    const categoryLabels = {
        markets: '📈 Markets', business: '💼 Business', corporate: '🏢 Corporate',
        crypto: '₿ Crypto', energy: '⛽ Energy', world: '🌍 World News',
        mideast: '🕌 Middle East', local: '📍 Local (Kuwait)', us_politics: '🇺🇸 US Politics',
        gcc: '🏜 GCC', asia: '🌏 Asia',
    };

    const groups = {};
    items.forEach(item => {
        const cat = item.category || 'other';
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push(item);
    });

    // Sort: local first, then by group size
    const catOrder = ['local', 'gcc', 'mideast', 'markets', 'business', 'corporate', 'energy', 'crypto', 'world', 'us_politics', 'asia'];
    const sortedCats = Object.keys(groups).sort((a, b) => {
        const ia = catOrder.indexOf(a);
        const ib = catOrder.indexOf(b);
        if (ia !== -1 && ib !== -1) return ia - ib;
        if (ia !== -1) return -1;
        if (ib !== -1) return 1;
        return 0;
    });

    const accent = type === 'finance' ? '#34d399' : '#60a5fa';
    const accentDim = type === 'finance' ? 'rgba(16,185,129,0.15)' : 'rgba(59,130,246,0.15)';

    let html = '<div class="space-y-4">';

    sortedCats.forEach(cat => {
        const catItems = groups[cat];
        const label = categoryLabels[cat] || cat;
        const enabledInCat = catItems.filter(i => i.on).length;

        html += `<div class="mb-5">
            <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-semibold text-slate-400 tracking-wide">${label}</span>
                <span class="text-[10px] text-slate-600 font-mono">${enabledInCat}/${catItems.length}</span>
            </div>
            <div class="flex flex-wrap gap-2">`;

        catItems.forEach(item => {
            const isOn = item.on;
            const regionTag = item.region || '';
            html += `<button onclick="toggleSource('${type}', '${item.id}')"
                class="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer select-none hover:scale-[1.03] active:scale-95"
                style="${isOn
                    ? `background:${accentDim}; color:${accent}; border:1px solid ${accent}50; box-shadow:0 0 12px ${accent}20`
                    : 'background:rgba(30,41,59,0.4); color:rgba(148,163,184,0.45); border:1px solid rgba(51,65,85,0.3)'
                }"
                title="${item.name} — ${regionTag}">
                <span class="w-2 h-2 rounded-full flex-shrink-0 transition-all" style="background:${isOn ? accent : 'rgba(71,85,105,0.5)'}; ${isOn ? `box-shadow:0 0 6px ${accent}60` : ''}"></span>
                ${esc(item.name)}
                <span class="text-[9px] opacity-40 font-normal">${esc(regionTag)}</span>
            </button>`;
        });

        html += `</div></div>`;
    });

    // Summary
    const enabledCount = items.filter(i => i.on).length;
    html += `</div>
        <div class="flex items-center justify-between mt-5 pt-4" style="border-top:1px solid rgba(51,65,85,0.3)">
            <span class="text-xs text-slate-500">${enabledCount} of ${items.length} sources enabled</span>
            <div class="flex gap-3">
                <button onclick="toggleAllSources('${type}', true)" class="text-xs text-slate-500 hover:text-emerald-400 transition-colors font-medium">Enable all</button>
                <span class="text-slate-700">·</span>
                <button onclick="toggleAllSources('${type}', false)" class="text-xs text-slate-500 hover:text-red-400 transition-colors font-medium">Disable all</button>
            </div>
        </div>`;

    container.innerHTML = html;
}

function toggleSource(type, id) {
    const items = sourcesCatalog[type];
    const item = items.find(i => i.id === id);
    if (!item) return;

    item.on = !item.on;
    window._settingsDirty = true;
    renderSourceCatalog(type);
}

function toggleAllSources(type, state) {
    const items = sourcesCatalog[type];
    items.forEach(i => i.on = state);
    window._settingsDirty = true;
    renderSourceCatalog(type);
}

// ═══════════════════════════════════════════════════════════
// CUSTOM FEEDS TAB
// ═══════════════════════════════════════════════════════════

function renderCustomCatalog() {
    const container = document.getElementById('source-catalog-custom');
    if (!container) return;

    const accent = '#a855f7';
    const accentDim = 'rgba(168,85,247,0.15)';

    let html = '<div class="space-y-4">';

    // Add feed form at top
    html += `<div class="flex gap-2 flex-wrap">
        <input type="text" id="custom-feed-url" class="flex-1 min-w-[200px] bg-slate-900 border border-slate-700 rounded-lg px-3 py-2.5 text-xs text-slate-200 focus:border-purple-500 focus:outline-none" placeholder="Any URL or RSS feed (auto-detects RSS)" onkeydown="if(event.key==='Enter') addCustomFeed()">
        <input type="text" id="custom-feed-name" class="w-36 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2.5 text-xs text-slate-200 focus:border-purple-500 focus:outline-none" placeholder="Label" onkeydown="if(event.key==='Enter') addCustomFeed()">
        <button onclick="addCustomFeed()" class="px-4 py-2.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium rounded-lg transition-colors flex items-center gap-1.5">
            <i data-lucide="plus" class="w-3.5 h-3.5"></i> Add Feed
        </button>
    </div>`;

    if (!customFeeds.length) {
        html += `<div class="text-center py-8">
            <i data-lucide="rss" class="w-8 h-8 text-slate-700 mx-auto mb-2"></i>
            <p class="text-xs text-slate-500">No custom feeds yet. Add RSS feeds above to create your own tab.</p>
        </div>`;
    } else {
        // Render feeds with toggles
        html += '<div class="flex flex-wrap gap-2">';
        customFeeds.forEach((feed, idx) => {
            const isOn = feed.on !== false;
            html += `<button onclick="toggleCustomFeed(${idx})"
                class="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer select-none hover:scale-[1.03] active:scale-95"
                style="${isOn
                    ? `background:${accentDim}; color:${accent}; border:1px solid ${accent}50; box-shadow:0 0 12px ${accent}20`
                    : 'background:rgba(30,41,59,0.4); color:rgba(148,163,184,0.45); border:1px solid rgba(51,65,85,0.3)'
                }"
                title="${esc(feed.url)}">
                <span class="w-2 h-2 rounded-full flex-shrink-0 transition-all" style="background:${isOn ? accent : 'rgba(71,85,105,0.5)'}; ${isOn ? `box-shadow:0 0 6px ${accent}60` : ''}"></span>
                ${esc(feed.name)}
                <span onclick="event.stopPropagation(); removeCustomFeed(${idx})" class="ml-0.5 p-0.5 rounded hover:bg-red-900/30 transition-colors" title="Remove feed">
                    <i data-lucide="x" class="w-3 h-3 opacity-40 hover:opacity-100"></i>
                </span>
            </button>`;
        });
        html += '</div>';

        // Summary
        const enabledCount = customFeeds.filter(f => f.on !== false).length;
        html += `<div class="flex items-center justify-between mt-3 pt-3" style="border-top:1px solid rgba(51,65,85,0.3)">
            <span class="text-xs text-slate-500">${enabledCount} of ${customFeeds.length} feeds enabled</span>
            <div class="flex gap-3">
                <button onclick="toggleAllCustomFeeds(true)" class="text-xs text-slate-500 hover:text-purple-400 transition-colors font-medium">Enable all</button>
                <span class="text-slate-700">·</span>
                <button onclick="toggleAllCustomFeeds(false)" class="text-xs text-slate-500 hover:text-red-400 transition-colors font-medium">Disable all</button>
            </div>
        </div>`;
    }

    // RSS Suggestions section
    html += `<div class="mt-4 pt-4" style="border-top:1px solid rgba(51,65,85,0.3)">
        <div class="flex items-center justify-between mb-3">
            <span class="text-xs font-medium text-slate-400 flex items-center gap-1.5">
                <i data-lucide="lightbulb" class="w-3.5 h-3.5 text-amber-500"></i> Suggested Feeds
            </span>
            <div class="flex gap-1">
                <button onclick="_showRssSuggestions('finance')" id="rss-sug-finance" class="text-[10px] px-2 py-1 rounded transition-colors bg-slate-800 text-slate-400 hover:text-purple-400">Finance</button>
                <button onclick="_showRssSuggestions('politics')" id="rss-sug-politics" class="text-[10px] px-2 py-1 rounded transition-colors bg-slate-800 text-slate-400 hover:text-purple-400">Politics</button>
                <button onclick="_showRssSuggestions('general')" id="rss-sug-general" class="text-[10px] px-2 py-1 rounded transition-colors bg-slate-800 text-slate-400 hover:text-purple-400">General</button>
                <button onclick="_showRssSuggestions('jobs')" id="rss-sug-jobs" class="text-[10px] px-2 py-1 rounded transition-colors bg-slate-800 text-slate-400 hover:text-purple-400">Jobs</button>
                <button onclick="_showRssSuggestions('media')" id="rss-sug-media" class="text-[10px] px-2 py-1 rounded transition-colors bg-slate-800 text-slate-400 hover:text-purple-400">Media</button>
            </div>
        </div>
        <div id="rss-suggestions-list" class="flex flex-wrap gap-2"></div>
    </div>`;

    html += '</div>';
    container.innerHTML = html;
    lucide.createIcons();

    // Auto-show finance suggestions
    _showRssSuggestions('finance');
}

// General RSS suggestions (tech, science, world news)
var _GENERAL_RSS_SUGGESTIONS = [
    { url: 'https://feeds.arstechnica.com/arstechnica/index', name: 'Ars Technica' },
    { url: 'https://www.theverge.com/rss/index.xml', name: 'The Verge' },
    { url: 'https://feeds.bbci.co.uk/news/world/rss.xml', name: 'BBC World' },
    { url: 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml', name: 'NYT World' },
    { url: 'https://www.aljazeera.com/xml/rss/all.xml', name: 'Al Jazeera' },
    { url: 'https://feeds.reuters.com/reuters/topNews', name: 'Reuters' },
    { url: 'https://techcrunch.com/feed/', name: 'TechCrunch' },
    { url: 'https://www.wired.com/feed/rss', name: 'Wired' },
    { url: 'https://www.nature.com/nature.rss', name: 'Nature' },
    { url: 'https://www.sciencedaily.com/rss/all.xml', name: 'ScienceDaily' },
];

var _JOBS_RSS_SUGGESTIONS = [
    { url: 'https://news.google.com/rss/search?q=Kuwait+jobs+hiring&hl=en&gl=KW&ceid=KW:en', name: 'Kuwait Jobs (Google)' },
    { url: 'https://news.google.com/rss/search?q=GCC+jobs+hiring+careers&hl=en&gl=AE&ceid=AE:en', name: 'GCC Jobs (Google)' },
    { url: 'https://www.linkedin.com/jobs/search/feed?keywords=&location=Kuwait&sortBy=DD', name: 'LinkedIn Kuwait' },
    { url: 'https://news.google.com/rss/search?q=indeed+jobs+Kuwait&hl=en', name: 'Indeed Kuwait (Google)' },
    { url: 'https://www.gulftalent.com/resources/rss.xml', name: 'GulfTalent' },
    { url: 'https://news.google.com/rss/search?q=remote+jobs+hiring+2026&hl=en', name: 'Remote Jobs (Google)' },
    { url: 'https://weworkremotely.com/categories/remote-programming-jobs.rss', name: 'WeWorkRemotely Dev' },
    { url: 'https://remoteok.com/remote-jobs.rss', name: 'RemoteOK' },
];

var _MEDIA_RSS_SUGGESTIONS = [
    // YouTube channels (via native Atom feeds — no API key needed)
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCBcRF18a7Qf58cCRy5xuWwQ', name: 'MKBHD' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UC2C_jShtL725hvbm1arSV9w', name: 'CGP Grey' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCVHFbqXqoYvEWM1Ddxl0QDg', name: 'Android Authority' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCXuqSBlHAE6Xw-yeJA0Tunw', name: 'Linus Tech Tips' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCWX3yGbODrc0VOfIhaCjYQw', name: 'Last Week Tonight' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCsooa4yRKGN_zEE8iknghZA', name: 'TED-Ed' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UC9-y-6csu5WGm29I7JiwpnA', name: 'Computerphile' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCZYTClx2T1of7BRZ86-8fow', name: 'Sci Show' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UC6nSFpj9HTCZ5t-N3Rm3-HA', name: 'Vsauce' },
    { url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCHnyfMqiRRG1u-2MsSQLbXA', name: 'Veritasium' },
    // Image boards (require CF Worker proxy for Kuwait ISP bypass)
    { url: 'https://yande.re/post/atom?tags=scenery', name: 'Yande.re Scenery ⚡' },
    { url: 'https://yande.re/post/atom?tags=landscape', name: 'Yande.re Landscape ⚡' },
    { url: 'https://danbooru.donmai.us/posts.atom?tags=scenery', name: 'Danbooru Scenery ⚡' },
    { url: 'https://safebooru.org/index.php?page=atom&s=post&tags=landscape', name: 'Safebooru Landscape' },
    // Manga
    { url: 'https://mangadex.org/rss/latest', name: 'MangaDex Latest' },
    // Twitch VODs (via RSS bridge)
    { url: 'https://twitchrss.appspot.com/vod/shroud', name: 'Twitch: shroud' },
    { url: 'https://twitchrss.appspot.com/vod/loltyler1', name: 'Twitch: tyler1' },
];

function _showRssSuggestions(type) {
    const list = document.getElementById('rss-suggestions-list');
    if (!list) return;
    // Highlight active tab
    ['finance', 'politics', 'general', 'jobs', 'media'].forEach(t => {
        const btn = document.getElementById('rss-sug-' + t);
        if (btn) {
            if (t === type) { btn.className = 'text-[10px] px-2 py-1 rounded transition-colors bg-purple-900/40 text-purple-400'; }
            else { btn.className = 'text-[10px] px-2 py-1 rounded transition-colors bg-slate-800 text-slate-400 hover:text-purple-400'; }
        }
    });

    if (type === 'general') {
        _renderRssSuggestionItems(list, _GENERAL_RSS_SUGGESTIONS);
        return;
    }
    if (type === 'jobs') {
        _renderRssSuggestionItems(list, _JOBS_RSS_SUGGESTIONS);
        return;
    }
    if (type === 'media') {
        _renderRssSuggestionItems(list, _MEDIA_RSS_SUGGESTIONS);
        return;
    }

    // Fetch from catalog API
    fetch('/api/feed-catalog/' + type, {
        headers: { 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' }
    })
    .then(r => r.json())
    .then(data => {
        const items = (data.catalog || []).map(f => ({ url: f.url, name: f.name }));
        _renderRssSuggestionItems(list, items);
    })
    .catch(() => { list.innerHTML = '<span class="text-xs text-slate-600">Failed to load suggestions</span>'; });
}

function _renderRssSuggestionItems(container, items) {
    const existingUrls = new Set(customFeeds.map(f => f.url));
    const available = items.filter(f => !existingUrls.has(f.url));
    if (!available.length) {
        container.innerHTML = '<span class="text-xs text-slate-600">All feeds from this category are already added</span>';
        return;
    }
    container.innerHTML = available.slice(0, 12).map(f =>
        `<button onclick="_addSuggestedFeed('${esc(f.url)}', '${esc(f.name)}')"
            class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium bg-slate-800/60 text-slate-400 border border-slate-700/50 hover:border-purple-500/50 hover:text-purple-400 hover:bg-purple-900/20 transition-all cursor-pointer"
            title="${esc(f.url)}">
            <i data-lucide="plus" class="w-3 h-3"></i> ${esc(f.name)}
        </button>`
    ).join('');
    lucide.createIcons();
}

function _addSuggestedFeed(url, name) {
    if (customFeeds.some(f => f.url === url)) {
        if (typeof showToast === 'function') showToast('Already added', 'warning');
        return;
    }
    customFeeds.push({ url, name, on: true });
    window._settingsDirty = true;
    renderCustomCatalog();
    if (configData) configData.custom_feeds = JSON.parse(JSON.stringify(customFeeds));
    if (typeof rebuildNavFromConfig === 'function') rebuildNavFromConfig();
    if (typeof showToast === 'function') showToast(`Added "${name}"`, 'success');
}

function addCustomFeed() {
    const urlEl = document.getElementById('custom-feed-url');
    const nameEl = document.getElementById('custom-feed-name');

    const url = urlEl.value.trim();
    const name = nameEl.value.trim() || (() => { try { return new URL(url).hostname.replace('www.', ''); } catch(e) { return url; } })();

    if (!url) {
        if (typeof showToast === 'function') showToast('Enter a feed URL', 'warning');
        urlEl.focus();
        return;
    }
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        urlEl.style.borderColor = '#ef4444';
        setTimeout(() => { urlEl.style.borderColor = ''; }, 2000);
        if (typeof showToast === 'function') showToast('URL must start with http:// or https://', 'error');
        return;
    }

    if (customFeeds.some(f => f.url === url)) {
        if (typeof showToast === 'function') showToast('This feed is already added', 'warning');
        return;
    }

    // Check if URL looks like an RSS feed already
    const looksLikeRss = /\.(xml|rss|atom)$/i.test(url) || /\/feed\/?$/i.test(url) || /\/rss\/?$/i.test(url);

    if (looksLikeRss) {
        _commitCustomFeed(url, name, urlEl, nameEl);
    } else {
        // Try auto-discovery: fetch the page and look for RSS links
        if (typeof showToast === 'function') showToast('Detecting RSS feed...', 'info');
        const btn = urlEl.parentElement?.querySelector('button');
        if (btn) { btn.disabled = true; btn.style.opacity = '0.5'; }

        fetch('/api/discover-rss', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            body: JSON.stringify({ url })
        })
        .then(r => r.json())
        .then(data => {
            if (btn) { btn.disabled = false; btn.style.opacity = ''; }
            if (data.feeds && data.feeds.length > 0) {
                const feed = data.feeds[0];
                const feedUrl = feed.url;
                const feedName = name || feed.title || (() => { try { return new URL(feedUrl).hostname.replace('www.', ''); } catch(e) { return feedUrl; } })();
                if (customFeeds.some(f => f.url === feedUrl)) {
                    if (typeof showToast === 'function') showToast('This feed is already added', 'warning');
                    return;
                }
                urlEl.value = feedUrl;
                if (typeof showToast === 'function') showToast(`Found RSS: ${feedName}`, 'success');
                _commitCustomFeed(feedUrl, feedName, urlEl, nameEl);
            } else {
                // No RSS found — add as-is (feedparser may still handle it)
                _commitCustomFeed(url, name, urlEl, nameEl);
            }
        })
        .catch(() => {
            if (btn) { btn.disabled = false; btn.style.opacity = ''; }
            _commitCustomFeed(url, name, urlEl, nameEl);
        });
    }
}

function _commitCustomFeed(url, name, urlEl, nameEl) {
    customFeeds.push({ url, name, on: true });
    window._settingsDirty = true;

    if (urlEl) urlEl.value = '';
    if (nameEl) nameEl.value = '';
    renderCustomCatalog();
    if (configData) configData.custom_feeds = JSON.parse(JSON.stringify(customFeeds));
    if (typeof rebuildNavFromConfig === 'function') rebuildNavFromConfig();
    if (typeof showToast === 'function') showToast(`Added "${name}"`, 'success');
}

function removeCustomFeed(index) {
    const removed = customFeeds[index];
    customFeeds.splice(index, 1);
    window._settingsDirty = true;
    renderCustomCatalog();
    if (typeof showToast === 'function') showToast(`Removed "${removed?.name || 'feed'}"`, 'info');
}

function toggleCustomFeed(index) {
    if (customFeeds[index]) {
        customFeeds[index].on = customFeeds[index].on === false ? true : false;
        window._settingsDirty = true;
        renderCustomCatalog();
    }
}

function toggleAllCustomFeeds(state) {
    customFeeds.forEach(f => f.on = state);
    window._settingsDirty = true;
    renderCustomCatalog();
}

async function editCustomTabName() {
    const current = customTabName || 'Custom';
    const newName = await stratosPrompt({ title: 'Rename Tab', label: 'Tab name', defaultValue: current });
    if (newName && newName.trim()) {
        customTabName = newName.trim();
        const label = document.getElementById('custom-tab-label');
        if (label) label.textContent = customTabName;
        window._settingsDirty = true;
        // Update sidebar nav to reflect new name
        if (typeof rebuildNavFromConfig === 'function') rebuildNavFromConfig();
    }
}

// ═══════════════════════════════════════════════════════════
// REFRESH FEEDS (re-fetch from backend for active source tab)
// ═══════════════════════════════════════════════════════════

async function refreshSourceFeeds(btn) {
    const typeMap = { finance: 'finance', politics: 'politics', jobs: 'jobs', custom: 'custom' };
    const feedType = typeMap[sourcesActiveTab] || 'finance';

    const icon = btn ? (btn.querySelector('svg') || btn.querySelector('i')) : null;
    if (icon) icon.style.animation = 'spin 0.8s linear infinite';
    if (btn) { btn.disabled = true; btn.style.opacity = '0.6'; }

    const status = document.getElementById('save-sources-status');
    try {
        await loadExtraFeeds(feedType);
        if (status) status.innerHTML = '<span class="text-emerald-400">✓ ' + feedType.charAt(0).toUpperCase() + feedType.slice(1) + ' feeds refreshed</span>';
        if (typeof showToast === 'function') showToast(feedType + ' feeds refreshed', 'success');
    } catch(e) {
        if (status) status.innerHTML = '<span class="text-red-400">✗ Refresh failed</span>';
    }

    if (icon) icon.style.animation = '';
    if (btn) { btn.disabled = false; btn.style.opacity = ''; }
    if (status) setTimeout(() => { status.innerHTML = ''; }, 4000);
}

// ═══════════════════════════════════════════════════════════
// SAVE SOURCE TOGGLES
// ═══════════════════════════════════════════════════════════

async function saveSourceToggles(silent) {
    const btn = document.getElementById('save-sources-btn');
    const status = document.getElementById('save-sources-status');
    if (!silent) {
        btn.disabled = true;
        btn.innerHTML = '<div class="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Saving...';
    }

    // Build toggle maps
    const financeToggles = {};
    const politicsToggles = {};
    const jobsToggles = {};

    sourcesCatalog.finance.forEach(item => {
        financeToggles[item.id] = item.on;
    });
    sourcesCatalog.politics.forEach(item => {
        politicsToggles[item.id] = item.on;
    });
    (sourcesCatalog.jobs || []).forEach(item => {
        jobsToggles[item.id] = item.on;
    });

    try {
        const resp = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({
                extra_feeds_finance: financeToggles,
                extra_feeds_politics: politicsToggles,
                extra_feeds_jobs: jobsToggles,
                custom_feeds: customFeeds,
                custom_tab_name: customTabName,
            })
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const result = await resp.json();
        if (!silent) {
            if (result.status === 'saved') {
                status.innerHTML = '<span class="text-emerald-400">✓ Saved!</span> <a onclick="setActive(\'finance_news\')" class="text-emerald-500/60 hover:text-emerald-400 cursor-pointer text-[10px] underline">Finance</a> <span class="text-slate-600">|</span> <a onclick="setActive(\'politics\')" class="text-blue-500/60 hover:text-blue-400 cursor-pointer text-[10px] underline">Politics</a> <span class="text-slate-600">|</span> <a onclick="setActive(\'custom_feeds\')" class="text-purple-500/60 hover:text-purple-400 cursor-pointer text-[10px] underline">Custom</a>';
            } else {
                status.innerHTML = '<span class="text-red-400">✗ ' + (result.error || 'Failed') + '</span>';
            }
        }
    } catch(e) {
        if (!silent) status.innerHTML = '<span class="text-red-400">✗ Network error</span>';
    }

    if (!silent) {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="save" class="w-3.5 h-3.5"></i> Save Sources';
        setTimeout(() => { status.innerHTML = ''; }, 4000);
        lucide.createIcons();
    }
}
