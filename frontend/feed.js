// === NEWS FEED ===

// Debounced renderFeed for search input — 200ms delay
let _feedSearchTimer = 0;
function _debouncedRenderFeed() {
    clearTimeout(_feedSearchTimer);
    _feedSearchTimer = setTimeout(renderFeed, 200);
}

// Base filtering keywords (always applied)
const KUWAIT_BASE = ['kuwait', 'kuwaiti', 'koc ', 'knpc', 'kipic', 'kuwait petroleum', 'kuwait oil', 'mina abdullah', 'ahmadi', 'shuaiba', 'paaet', 'kuwait university'];
const REGIONAL_KW = ['gcc', 'gulf cooperation', 'saudi', 'aramco', 'uae', 'emirates', 'dubai', 'abu dhabi', 'adnoc', 'qatar', 'qatari', 'bahrain', 'bahraini', 'oman', 'omani', 'middle east', 'mena', 'gulf state', 'gulf region', 'riyadh', 'jeddah', 'doha', 'muscat', 'manama', 'neom', 'vision 2030'];
const MARKET_KW = ['nvidia', 'nvda', ' amd ', 'semiconductor', 'chip stock', 'chip market', 'fabless', 'tsmc', 'intel', 'qualcomm', 'broadcom', 'stock', 'trading', 'investor'];

// Dynamic keywords from config (populated after config loads)
function getKuwaitKeywords() {
    // Combine base keywords with user-configured banks and companies
    const banks = configData?.news?.finance?.keywords || [];
    const companies = configData?.news?.career?.keywords || [];
    return [...KUWAIT_BASE, ...banks.map(b => b.toLowerCase()), ...companies.map(c => c.toLowerCase())];
}

// Remove duplicates by URL
function deduplicateNews(items) {
    const seen = new Set();
    return items.filter(item => {
        const key = item.url || item.title;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function matchesKuwait(item) {
    const text = ((item.title || '') + ' ' + (item.summary || '')).toLowerCase();
    // Check root field first (backend already categorized as kuwait)
    if (item.root === 'kuwait' && item.category !== 'finance' && item.category !== 'banks') return true;
    // If it's finance/banks category, it goes to Banks tab instead
    if (item.category === 'finance' || item.category === 'banks') return false;
    // Check dynamic keywords from config + base keywords
    const kuwaitKeywords = getKuwaitKeywords();
    return kuwaitKeywords.some(k => text.includes(k));
}

function matchesBanks(item) {
    // Explicit bank/finance categorization from backend
    if (item.category === 'finance' || item.category === 'banks') return true;
    // Check text for bank keywords
    const text = ((item.title || '') + ' ' + (item.summary || '')).toLowerCase();
    const bankKW = ['nbk', 'boubyan', 'kfh', 'cbk', 'kib', 'warba', 'burgan', 'gulf bank',
                    'student allowance', 'allowance transfer', 'cash gift', 'bank offer',
                    'bank promotion', 'bank deal', 'banking offer'];
    return bankKW.some(k => text.includes(k));
}

function matchesRegional(item) {
    const text = ((item.title || '') + ' ' + (item.summary || '')).toLowerCase();
    // Banks items stay in Banks tab, not Regional
    if (matchesBanks(item)) return false;
    // Check if item is explicitly marked as regional
    if (item.root === 'regional') return true;
    // Exclude Kuwait items
    if (matchesKuwait(item)) return false;
    // Check for regional keywords
    return REGIONAL_KW.some(k => text.includes(k));
}

function matchesMarket(item) {
    const text = ((item.title || '') + ' ' + (item.summary || '')).toLowerCase();
    // Banks items stay in Banks tab, not Global Markets
    if (matchesBanks(item)) return false;
    return MARKET_KW.some(k => text.includes(k));
}

// Score filter state: which levels are visible
let scoreFilters = { critical: true, high: true, medium: true, noise: false };

function toggleScoreFilter(level) {
    if (level === 'all') {
        // Toggle all on (but keep noise off by default)
        const allOn = scoreFilters.critical && scoreFilters.high && scoreFilters.medium;
        scoreFilters = { critical: !allOn, high: !allOn, medium: !allOn, noise: scoreFilters.noise };
    } else {
        scoreFilters[level] = !scoreFilters[level];
    }
    // Update pill visuals
    document.querySelectorAll('.score-pill').forEach(pill => {
        const f = pill.dataset.filter;
        if (f === 'all') {
            pill.classList.toggle('active', scoreFilters.critical && scoreFilters.high && scoreFilters.medium);
        } else {
            pill.classList.toggle('active', scoreFilters[f]);
        }
    });
    renderFeed();
}

function passesScoreFilter(score) {
    if (score >= 9.0) return scoreFilters.critical;
    if (score >= 7.0) return scoreFilters.high;
    if (score >= 5.0) return scoreFilters.medium;
    return scoreFilters.noise;
}

function timeAgo(timestamp) {
    if (!timestamp) return '';
    try {
        const d = new Date(timestamp);
        if (isNaN(d.getTime())) return '';
        const now = new Date();
        const diffMs = now - d;
        const mins = Math.floor(diffMs / 60000);
        if (mins < 1) return 'just now';
        if (mins < 60) return `${mins}m ago`;
        const hrs = Math.floor(mins / 60);
        if (hrs < 24) return `${hrs}h ago`;
        const days = Math.floor(hrs / 24);
        if (days < 7) return `${days}d ago`;
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch(e) { return ''; }
}

function renderBriefing() {
    const panel = document.getElementById('briefing-panel');
    if (activeRoot !== 'dashboard' || !data?.briefing) {
        panel.classList.add('hidden');
        return;
    }
    
    const b = data.briefing;
    const stats = b.summary_stats || {};
    
    // Client-side fallback: if briefing is missing critical data, derive from newsData
    if (newsData && newsData.length > 0) {
        const criticalNews = newsData.filter(n => (n.score || 0) >= 9.0);
        const highNews = newsData.filter(n => (n.score || 0) >= 7.0 && (n.score || 0) < 9.0);
        const mediumNews = newsData.filter(n => (n.score || 0) >= 5.0 && (n.score || 0) < 7.0);
        
        // Fix counts if briefing counts seem stale
        if (!b.critical_count && criticalNews.length > 0) b.critical_count = criticalNews.length;
        if (!b.high_count && highNews.length > 0) b.high_count = highNews.length;
        if (!b.medium_count && mediumNews.length > 0) b.medium_count = mediumNews.length;
        
        // Inject critical alerts if missing
        if ((!b.critical_alerts || b.critical_alerts.length === 0) && criticalNews.length > 0) {
            b.critical_alerts = criticalNews.slice(0, 3).map(n => ({
                headline: n.title,
                analysis: n.score_reason || n.summary || '',
                url: n.url,
                score: n.score,
                action: ''
            }));
        }
    }
    
    let html = '';
    
    // ── Header card with stats ──
    html += `<div class="glass-panel rounded-xl overflow-hidden">`;
    
    // Top bar: title + stat badges
    html += `<div class="px-5 pt-5 pb-3">
        <div class="flex items-center justify-between">
            <div class="flex items-center gap-2.5">
                <div class="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                    <i data-lucide="activity" class="w-4.5 h-4.5 text-emerald-400"></i>
                </div>
                <div>
                    <h3 class="text-sm font-bold text-slate-100">Daily Overview</h3>
                    <p class="text-[10px] text-slate-500 mt-0.5">${esc(b.greeting || '')}${stats.total_items_analyzed ? ' · ' + stats.total_items_analyzed + ' items analyzed' : ''}</p>
                </div>
            </div>
            <div class="flex gap-2">
                ${b.critical_count ? `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> ${b.critical_count} Critical</span>` : ''}
                ${b.high_count ? `<span class="px-2.5 py-1 rounded-full text-[10px] font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20">${b.high_count} High</span>` : ''}
                ${b.medium_count ? `<span class="px-2.5 py-1 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">${b.medium_count} Notable</span>` : ''}
            </div>
        </div>
    </div>`;
    
    // ── Critical alerts ──
    if (b.critical_alerts && b.critical_alerts.length > 0) {
        html += `<div class="px-5 pb-1">`;
        b.critical_alerts.forEach(alert => {
            html += `<div class="rounded-lg p-3.5 mb-2.5" style="background: linear-gradient(135deg, rgba(16,185,129,0.06) 0%, rgba(16,185,129,0.02) 100%); border: 1px solid rgba(16,185,129,0.12);">
                <div class="flex items-start gap-3">
                    <div class="mt-0.5 flex-shrink-0">
                        <span class="relative flex h-2.5 w-2.5">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-50"></span>
                            <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                        </span>
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-start justify-between gap-3">
                            <a href="${esc(alert.url || '#')}" target="_blank" class="text-[13px] font-semibold text-slate-100 hover:text-emerald-300 transition-colors leading-snug">${esc(alert.headline)}</a>
                            <span class="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 flex-shrink-0">${(alert.score || 9).toFixed(1)}</span>
                        </div>
                        <p class="text-[11px] text-slate-400 mt-1.5 leading-relaxed">${esc(alert.analysis || '')}</p>
                        ${alert.action ? `<p class="text-[11px] text-emerald-400/70 mt-2 flex items-center gap-1.5 font-medium"><i data-lucide="arrow-right" class="w-3 h-3"></i> ${esc(alert.action)}</p>` : ''}
                    </div>
                </div>
            </div>`;
        });
        html += `</div>`;
    }
    
    // ── Top picks ──
    if (b.high_priority && b.high_priority.length > 0) {
        html += `<div class="px-5 pb-4">
            <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                <i data-lucide="star" class="w-3 h-3"></i> Top Picks
            </div>
            <div class="space-y-1">`;
        b.high_priority.slice(0, 5).forEach((hp, i) => {
            html += `<a href="${esc(hp.url || '#')}" target="_blank" class="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-800/40 transition-colors group">
                <span class="text-[10px] font-mono text-slate-600 w-4">${i + 1}.</span>
                <span class="text-xs text-slate-300 group-hover:text-emerald-300 transition-colors truncate flex-1">${esc(hp.title)}</span>
                <span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20 flex-shrink-0">${(hp.score || 0).toFixed(1)}</span>
            </a>`;
        });
        html += `</div></div>`;
    }
    
    html += `</div>`;
    
    panel.innerHTML = html;
    panel.classList.remove('hidden');
    lucide.createIcons();
}

function renderFeed() {
    const container = document.getElementById('news-feed');
    const searchQuery = (document.getElementById('feed-search')?.value || '').toLowerCase().trim();
    let filtered = [];

    // Check if profile is empty — show onboarding welcome
    const _cfg = typeof configData !== 'undefined' ? configData : null;
    const _profileEmpty = !_cfg?.profile?.role?.trim() && 
        !(_cfg?.dynamic_categories?.length > 0) &&
        !_cfg?.news?.career?.keywords?.length;
    
    if (_profileEmpty && activeRoot === 'dashboard' && newsData.length === 0) {
        // Hide feed controls, briefing, etc.
        const controls = document.getElementById('feed-controls');
        if (controls) controls.classList.add('hidden');
        const feedCount = document.getElementById('feed-count');
        if (feedCount) feedCount.textContent = '';
        const briefing = document.getElementById('briefing-panel');
        if (briefing) briefing.classList.add('hidden');
        
        const profileName = typeof getActiveProfile === 'function' ? getActiveProfile() : '';
        const greeting = profileName ? `Welcome, ${profileName}` : 'Welcome to STRAT_OS';
        
        container.classList.add('onboarding-active');
        // Single HTML — CSS media queries handle mobile vs desktop layout
        container.innerHTML = `
        <div class="ob-wrap">
            <div class="ob-icon">
                <i data-lucide="radar" class="w-7 h-7 text-emerald-400"></i>
            </div>
            <h2 class="ob-greeting">${greeting}</h2>
            <p class="ob-subtitle">Your intelligence dashboard is ready to be configured.</p>

            <div class="ob-steps">
                <div onclick="setActive('settings')" class="ob-step ob-step-link group" style="background: rgba(16,185,129,0.04); border: 1px solid rgba(16,185,129,0.12);">
                    <div class="ob-badge" style="background:rgba(16,185,129,0.1);"><span class="text-emerald-400">1</span></div>
                    <div class="ob-text">
                        <div class="ob-title-row">
                            <p class="ob-title text-slate-200 group-hover:text-emerald-300">Set up your profile</p>
                            <i data-lucide="arrow-right" class="ob-arrow text-slate-600 group-hover:text-emerald-400"></i>
                        </div>
                        <p class="ob-desc">Enter role & location, then hit <span class="text-purple-400">Suggest</span> and <span class="text-emerald-400">Generate</span></p>
                    </div>
                </div>

                <div class="ob-step" style="background: rgba(96,165,250,0.03); border: 1px solid rgba(96,165,250,0.08);">
                    <div class="ob-badge" style="background:rgba(59,130,246,0.1);"><span class="text-blue-400">2</span></div>
                    <div class="ob-text">
                        <p class="ob-title text-slate-300">Pick news sources & save</p>
                        <p class="ob-desc">Toggle feeds, then hit <span class="text-emerald-400">Save All</span></p>
                    </div>
                </div>

                <div class="ob-step" style="background: rgba(168,85,247,0.03); border: 1px solid rgba(168,85,247,0.08);">
                    <div class="ob-badge" style="background:rgba(168,85,247,0.1);"><span class="text-purple-400">3</span></div>
                    <div class="ob-text">
                        <p class="ob-title text-slate-300">Refresh & explore</p>
                        <p class="ob-desc">Tap <span class="text-emerald-400">News</span> to start scanning</p>
                    </div>
                </div>

                <button onclick="setActive('settings')" class="ob-cta">
                    <i data-lucide="settings" class="w-4 h-4"></i> Go to Settings
                </button>
            </div>
        </div>`;
        lucide.createIcons();
        return;
    }

    container.classList.remove('onboarding-active');

    // First deduplicate all news
    const uniqueNews = deduplicateNews(newsData);

    if (activeRoot === 'dashboard') {
        filtered = uniqueNews.filter(d => (d.score || 0) >= 5.0);
    } else if (activeRoot === 'saved') {
        filtered = getSavedSignals();
    } else if (activeRoot === 'kuwait') {
        filtered = uniqueNews.filter(d => matchesKuwait(d));
    } else if (activeRoot === 'banks') {
        filtered = uniqueNews.filter(d => matchesBanks(d));
    } else if (activeRoot === 'regional') {
        filtered = uniqueNews.filter(d => matchesRegional(d));
    } else if (activeRoot === 'global') {
        filtered = uniqueNews.filter(d => matchesMarket(d));
    } else if (activeRoot === 'ai') {
        filtered = uniqueNews.filter(d => (d.root === 'ai' || d.root === 'global' || d.category === 'tech') && !matchesKuwait(d) && !matchesBanks(d) && !matchesRegional(d) && !matchesMarket(d) && !matchesAnyDynamicCategory(d));
    } else if (activeRoot === 'finance_news') {
        filtered = financeNewsData.slice();
    } else if (activeRoot === 'politics') {
        filtered = politicsNewsData.slice();
    } else if (activeRoot === 'custom_feeds') {
        filtered = customNewsData.slice();
    } else {
        // Dynamic category tab — filter by keyword matching (word-boundary safe)
        const navItem = (window.navItems || []).find(n => n.id === activeRoot);
        if (navItem && navItem.dynamic && navItem.keywords && navItem.keywords.length > 0) {
            const kwLower = navItem.keywords.map(k => k.toLowerCase());
            filtered = uniqueNews.filter(d => {
                const text = ((d.title || '') + ' ' + (d.summary || '') + ' ' + (d.category || '')).toLowerCase();
                // Match if any keyword matches (word-boundary), or if the item's category matches
                return kwLower.some(kw => matchesKeyword(text, kw)) || d.category === activeRoot;
            });
        } else {
            // Unknown tab — show everything from all roots
            filtered = uniqueNews.filter(d => (d.score || 0) >= 5.0);
        }
    }

    // Apply score filter (skip for headline tabs — they're unscored RSS)
    if (activeRoot !== 'finance_news' && activeRoot !== 'politics' && activeRoot !== 'custom_feeds') {
        filtered = filtered.filter(d => passesScoreFilter(d.score || 0));
    }
    
    // Apply search filter
    if (searchQuery) {
        filtered = filtered.filter(d => {
            const text = ((d.title || '') + ' ' + (d.summary || '') + ' ' + (d.source || '') + ' ' + (d.category || '') + ' ' + (d.score_reason || '')).toLowerCase();
            return searchQuery.split(/\s+/).every(word => text.includes(word));
        });
    }

    // Filter out dismissed items (Tier 1 feedback loop)
    filtered = filtered.filter(d => !d._dismissed);

    // Sort: extra tabs by time, scored tabs by score
    if (activeRoot === 'finance_news' || activeRoot === 'politics' || activeRoot === 'custom_feeds') {
        filtered.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));
    } else {
        filtered.sort((a, b) => (b.score || 0) - (a.score || 0));
    }
    
    // Render briefing (dashboard only)
    renderBriefing();
    
    // Render top movers (dashboard only)
    if (typeof renderAlerts === 'function') renderAlerts();
    
    // Show/hide AI agent (dashboard only)
    if (typeof showAgentPanel === 'function') {
        showAgentPanel(activeRoot === 'dashboard');
    }
    
    // Show/hide controls appropriately
    const controls = document.getElementById('feed-controls');
    if (activeRoot === 'settings') {
        controls.classList.add('hidden');
    } else {
        controls.classList.remove('hidden');
    }
    
    // Show/hide refresh icon in search bar + update placeholder
    const refreshBtn = document.getElementById('feed-refresh-btn');
    const searchInput = document.getElementById('feed-search');
    if (activeRoot === 'finance_news' || activeRoot === 'politics' || activeRoot === 'custom_feeds') {
        refreshBtn?.classList.remove('hidden');
        if (searchInput) searchInput.placeholder = 'Search headlines...';
        window._refreshCurrentFeed = async () => {
            const icon = refreshBtn?.querySelector('svg');
            if (icon) icon.classList.add('animate-spin');
            await loadExtraFeeds(activeRoot === 'finance_news' ? 'finance' : activeRoot === 'politics' ? 'politics' : 'custom');
            if (icon) icon.classList.remove('animate-spin');
            renderFeed();
            if (typeof showToast === 'function') showToast('Feed refreshed', 'success');
        };
    } else {
        refreshBtn?.classList.add('hidden');
        if (searchInput) searchInput.placeholder = 'Filter signals...';
        window._refreshCurrentFeed = null;
    }
    
    // Update feed count + grid/list toggle for Saved view
    const isExtraFeedTab = (activeRoot === 'finance_news' || activeRoot === 'politics' || activeRoot === 'custom_feeds');
    const countEl = document.getElementById('feed-count');
    const countText = `${filtered.length} ${isExtraFeedTab ? 'headlines' : 'signals'}`;
    if (activeRoot === 'saved' && filtered.length > 0) {
        const isGrid = localStorage.getItem('savedViewGrid') === '1';
        countEl.innerHTML = `<span>${countText}</span>
            <button onclick="toggleSavedViewMode()" class="ml-3 text-[10px] px-2 py-0.5 rounded transition-all" style="border:1px solid var(--border-strong);color:var(--text-muted);" title="Toggle grid/list view">
                <span style="display:inline-flex;align-items:center;gap:3px;">
                    ${isGrid ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg> List' : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg> Grid'}
                </span>
            </button>`;
        if (isGrid) container.classList.add('saved-grid-mode');
        else container.classList.remove('saved-grid-mode');
    } else {
        countEl.textContent = countText;
        container.classList.remove('saved-grid-mode');
    }

    if (!filtered.length) {
        const navItem = (window.navItems || []).find(n => n.id === activeRoot);
        const displayName = navItem ? navItem.label : activeRoot;
        const isExtraFeed = activeRoot === 'finance_news' || activeRoot === 'politics' || activeRoot === 'custom_feeds';
        const refreshAction = isExtraFeed 
            ? (activeRoot === 'finance_news' ? "loadExtraFeeds('finance').then(renderFeed)" : activeRoot === 'politics' ? "loadExtraFeeds('politics').then(renderFeed)" : "loadExtraFeeds('custom').then(renderFeed)")
            : '';
        const emptyMsg = searchQuery 
            ? 'No matches for "' + esc(searchQuery) + '"' 
            : activeRoot === 'saved' ? 'No saved signals yet. Bookmark signals from any tab to see them here.'
            : (isExtraFeed ? 'No headlines yet.' : 'No signals detected.');
        container.innerHTML = `<div class="p-8 text-center text-slate-500 border border-dashed border-slate-700 rounded-xl">
            <p>${emptyMsg}</p>
            ${isExtraFeed ? `<button onclick="${refreshAction}" class="mt-3 text-[10px] text-slate-400 hover:text-emerald-400 inline-flex items-center gap-1 transition-colors"><i data-lucide="refresh-cw" class="w-3 h-3"></i> Refresh</button>` : ''}
        </div>`;
        if (isExtraFeed) lucide.createIcons();
        return;
    }

    // Store for AI ask reference
    window.currentItems = filtered;

    // === HEADLINE TABS (Finance / Politics / Custom) — interactive newspaper style ===
    if (activeRoot === 'finance_news' || activeRoot === 'politics' || activeRoot === 'custom_feeds') {
        const isFinance = activeRoot === 'finance_news';
        const isCustom = activeRoot === 'custom_feeds';
        const icon = isFinance ? 'bar-chart-3' : isCustom ? 'rss' : 'globe-2';
        const customLabel = (configData?.custom_tab_name || (typeof customTabName !== 'undefined' ? customTabName : 'Custom')) + ' Headlines';
        const label = isFinance ? 'Finance Headlines' : isCustom ? customLabel : 'World Headlines';
        const accentColor = isFinance ? 'emerald' : isCustom ? 'purple' : 'blue';
        const accentHex = isFinance ? '#34d399' : isCustom ? '#a855f7' : '#60a5fa';
        
        // Group by source
        const bySource = {};
        filtered.forEach(item => {
            const src = item.source || 'Unknown';
            if (!bySource[src]) bySource[src] = [];
            bySource[src].push(item);
        });

        // Get saved source order from localStorage
        const orderKey = `stratos_source_order_${activeRoot}`;
        let savedOrder = [];
        try { savedOrder = JSON.parse(localStorage.getItem(orderKey) || '[]'); } catch(e) {}
        
        // Sort sources: saved order first, then by item count descending
        const sourceKeys = Object.keys(bySource);
        sourceKeys.sort((a, b) => {
            const idxA = savedOrder.indexOf(a);
            const idxB = savedOrder.indexOf(b);
            if (idxA !== -1 && idxB !== -1) return idxA - idxB;
            if (idxA !== -1) return -1;
            if (idxB !== -1) return 1;
            return bySource[b].length - bySource[a].length;
        });

        // Get collapsed state per tab
        if (!window._collapsedSources) window._collapsedSources = {};
        if (!window._collapsedSources[activeRoot]) window._collapsedSources[activeRoot] = {};

        // --- Header with source count + refresh ---
        let html = `<div class="mb-3 flex items-center gap-2">
            <i data-lucide="${icon}" class="w-4 h-4 text-${accentColor}-400"></i>
            <span class="text-xs font-bold text-slate-400 uppercase tracking-widest">${label}</span>
            <span class="text-[10px] text-slate-600 ml-auto">${filtered.length} headlines · ${sourceKeys.length} sources</span>
            <button onclick="(async(btn){const i=btn.querySelector('svg');if(i)i.classList.add('animate-spin');await loadExtraFeeds('${isFinance ? 'finance' : isCustom ? 'custom' : 'politics'}');if(i)i.classList.remove('animate-spin');renderFeed();if(typeof showToast==='function')showToast('Feed refreshed','success');})(this)" class="text-[10px] text-slate-500 hover:text-${accentColor}-400 flex items-center gap-1 transition-colors ml-2" id="headline-refresh-btn">
                <i data-lucide="refresh-cw" class="w-3 h-3"></i> Refresh
            </button>
        </div>`;

        // --- Source quick-jump pills ---
        html += `<div class="flex flex-wrap gap-1.5 mb-4 pb-3" style="border-bottom:1px solid rgba(51,65,85,0.3)">`;
        sourceKeys.forEach(source => {
            const count = bySource[source].length;
            const isCollapsed = window._collapsedSources[activeRoot][source];
            html += `<button onclick="document.getElementById('src-${esc(source.replace(/[^a-zA-Z0-9]/g,'_'))}')?.scrollIntoView({behavior:'smooth',block:'start'})" 
                class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-medium transition-all hover:scale-105"
                style="background:${isCollapsed ? 'rgba(30,41,59,0.4)' : `rgba(${isFinance ? '16,185,129' : '59,130,246'},0.1)`};
                       border:1px solid ${isCollapsed ? 'rgba(51,65,85,0.3)' : `rgba(${isFinance ? '16,185,129' : '59,130,246'},0.25)`};
                       color:${isCollapsed ? 'rgb(100,116,139)' : accentHex}">
                ${esc(source)} <span style="opacity:0.6">${count}</span>
            </button>`;
        });
        html += `<button onclick="toggleAllHeadlineSources()" class="px-2 py-1 rounded-full text-[10px] font-medium transition-all text-slate-500 hover:text-slate-300" style="background:rgba(30,41,59,0.3);border:1px solid rgba(51,65,85,0.3)">
            ${Object.keys(window._collapsedSources[activeRoot] || {}).length > 0 ? '⊕ Expand All' : '⊖ Collapse All'}
        </button>`;
        html += `</div>`;

        // --- Source groups (collapsible, draggable) ---
        html += `<div id="source-groups-container" data-order-key="${orderKey}" class="space-y-1">`;
        sourceKeys.forEach((source) => {
            const items = bySource[source];
            const safeId = source.replace(/[^a-zA-Z0-9]/g, '_');
            const isCollapsed = window._collapsedSources[activeRoot][source];
            
            html += `<div class="source-group" draggable="true" data-source="${esc(source)}" id="src-${safeId}">
                <div class="flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-all hover:bg-slate-800/30 source-drag-handle"
                     onclick="toggleHeadlineSource('${esc(source)}')" style="${isCollapsed ? '' : `border-left:2px solid ${accentHex}20`}">
                    <i data-lucide="grip-vertical" class="w-3 h-3 text-slate-700 hover:text-slate-500 flex-shrink-0 cursor-grab"></i>
                    <i data-lucide="${isCollapsed ? 'chevron-right' : 'chevron-down'}" class="w-3 h-3 text-slate-500 flex-shrink-0 transition-transform"></i>
                    <span class="text-[11px] font-bold uppercase tracking-wider" style="color:${accentHex}">${esc(source)}</span>
                    <div class="flex-1 border-t border-slate-800/30"></div>
                    <span class="text-[10px] font-mono px-1.5 py-0.5 rounded" style="background:rgba(${isFinance ? '16,185,129' : '59,130,246'},0.08);color:${accentHex}">${items.length}</span>
                </div>`;

            if (!isCollapsed) {
                html += `<div class="pl-2 space-y-0.5 mb-3" style="border-left:2px solid ${accentHex}10;margin-left:9px">`;
                items.forEach(item => {
                    const age = timeAgo(item.timestamp);
                    // Check if article is less than 1 hour old
                    let isNew = false;
                    if (item.timestamp) {
                        try { isNew = (Date.now() - new Date(item.timestamp).getTime()) < 3600000; } catch(e) {}
                    }
                    
                    const thumb = item.thumbnail || '';
                    html += `
                    <a href="${esc(item.url)}" target="_blank" rel="noopener"
                       class="group flex items-start gap-3 px-3 py-2.5 rounded-lg transition-all hover:bg-slate-800/50"
                       style="border-left:2px solid transparent"
                       onmouseenter="this.style.borderLeftColor='${accentHex}'"
                       onmouseleave="this.style.borderLeftColor='transparent'">
                        <div class="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 transition-all group-hover:scale-150" style="background:${accentHex}40"></div>
                        <div class="flex-1 min-w-0">
                            <div class="flex items-center gap-2">
                                <h4 class="text-[13px] text-slate-200 group-hover:text-white transition-colors leading-snug font-medium">${esc(item.title)}</h4>
                                ${isNew ? `<span class="flex-shrink-0 text-[8px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider" style="background:rgba(${isFinance ? '16,185,129' : '59,130,246'},0.15);color:${accentHex}">new</span>` : ''}
                            </div>
                            ${item.summary ? `<p class="text-[11px] text-slate-500 mt-0.5 line-clamp-1 group-hover:text-slate-400 transition-colors">${esc(item.summary)}</p>` : ''}
                        </div>
                        ${thumb ? `<img src="${esc(thumb)}" alt="" class="w-14 h-10 rounded object-cover flex-shrink-0 mt-0.5 opacity-70 group-hover:opacity-100 transition-opacity" onerror="this.style.display='none'" loading="lazy">` : ''}
                        ${age ? `<span class="text-[10px] text-slate-600 flex-shrink-0 mt-0.5 tabular-nums">${age}</span>` : ''}
                    </a>`;
                });
                html += `</div>`;
            }
            html += `</div>`;
        });
        html += `</div>`;

        if (!filtered.length) {
            html += `<div class="p-8 text-center text-slate-500 border border-dashed border-slate-700 rounded-xl">
                <i data-lucide="loader" class="w-5 h-5 mx-auto mb-2 animate-spin text-slate-600"></i>
                Loading headlines...
            </div>`;
        }

        container.innerHTML = html;
        initSourceDragDrop(); // Initialize drag-to-reorder
        if (!searchQuery) renderNav();
        lucide.createIcons();
        return;
    }

    // === NORMAL SIGNAL CARDS ===
    container.innerHTML = filtered.map((item, idx) => {
        const s = item.score || 0;
        let scoreClass = 'score-noise';
        let cardClass = 'card-noise';
        let ping = '';

        if (s >= 9.0) {
            scoreClass = 'score-critical';
            cardClass = 'card-critical';
            ping = '<div class="absolute top-2 right-2"><div class="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></div></div>';
        } else if (s >= 7.0) {
            scoreClass = 'score-high';
            cardClass = 'card-high';
        } else if (s >= 5.0) {
            scoreClass = 'score-medium';
            cardClass = 'card-medium';
        }
        
        const age = timeAgo(item.timestamp);
        const reason = item.score_reason || '';
        // Clean up score reason for display
        const reasonDisplay = reason.replace(/^(Rule|LLM|Pre-filtered): ?/i, '').substring(0, 80);

        const hasContent = item.content && item.content.length > 10;
        const contentDiffers = hasContent && item.content.trim() !== (item.summary || '').trim() && item.content.length > (item.summary || '').length + 20;
        const cardId = 'card-' + idx;

        return `
        <div class="group block glass-panel ${cardClass} transition-all p-4 rounded-xl relative overflow-hidden" data-card-idx="${idx}" style="--card-hover:var(--bg-hover)" onmouseenter="this.style.background=getComputedStyle(document.documentElement).getPropertyValue('--bg-hover')" onmouseleave="this.style.background=''">
            ${ping}
            <div class="flex justify-between items-start mb-2">
                <div class="flex gap-2 items-center text-[10px] tracking-wider font-mono flex-wrap">
                    <span class="drill-pill text-slate-400 font-bold uppercase hover:text-emerald-400" onclick="event.stopPropagation(); drillFilter('${esc(item.root || 'global')}')">${esc(item.root || 'global')}</span>
                    <span class="text-slate-600">|</span>
                    <span class="drill-pill text-slate-500 uppercase hover:text-emerald-400" onclick="event.stopPropagation(); drillFilter('${esc(item.category || 'general')}')">${esc(item.category || 'general')}</span>
                    ${age ? `<span class="text-slate-600">|</span><span class="text-slate-600 time-ago">${age}</span>` : ''}
                </div>
                ${item.retained ? '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border border-amber-700/40 text-amber-500 bg-amber-500/10" title="Kept from previous scan">kept</span>' : ''}
                <span class="px-2 py-0.5 rounded text-xs font-mono font-bold border ${scoreClass}">${s.toFixed(1)}</span>
            </div>
            <a href="${esc(item.url)}" target="_blank" class="block" onclick="trackSignalClick(${idx})">
                <h3 class="text-base font-semibold text-slate-200 mb-1 group-hover:text-emerald-400 transition-colors leading-snug">${esc(item.title)}</h3>
            </a>
            <p class="drill-pill text-slate-500 text-xs mb-2 inline hover:text-emerald-400" onclick="drillFilter('${esc(item.source)}')">${esc(item.source)}</p>
            <p class="text-slate-400 text-sm line-clamp-2">${esc(item.summary || '')}</p>
            ${reasonDisplay ? `<p class="drill-pill text-[10px] text-slate-600 mt-2 font-mono truncate hover:text-emerald-400" onclick="drillFilter('${esc(reasonDisplay.split(':')[0].trim())}')">${esc(reasonDisplay)}</p>` : ''}
            <div class="flex items-center gap-3 mt-2">
                <button onclick="thumbsUpSignal(${idx})" class="text-[10px] ${item._thumbs === 'up' ? 'text-emerald-400 font-bold' : 'text-slate-500 hover:text-emerald-400'} flex items-center gap-1 transition-colors" title="Relevant — train scorer to score higher">
                    <i data-lucide="thumbs-up" class="w-3 h-3"></i>
                </button>
                <button onclick="thumbsDownSignal(${idx})" class="text-[10px] ${item._thumbs === 'down' ? 'text-red-400 font-bold' : 'text-slate-500 hover:text-red-400'} flex items-center gap-1 transition-colors" title="Noise — train scorer to score lower">
                    <i data-lucide="thumbs-down" class="w-3 h-3"></i>
                </button>
                <span class="text-slate-700">|</span>
                <button onclick="toggleSaveSignal(${idx})" class="text-[10px] ${isSignalSaved(item) ? 'text-emerald-400' : 'text-slate-500 hover:text-emerald-400'} flex items-center gap-1 transition-colors">
                    <i data-lucide="${isSignalSaved(item) ? 'bookmark-check' : 'bookmark'}" class="w-3 h-3"></i> ${isSignalSaved(item) ? 'Saved' : 'Save'}
                </button>
                ${contentDiffers ? `
                    <button onclick="toggleCardContent('${cardId}')" class="text-[10px] text-slate-500 hover:text-emerald-400 flex items-center gap-1 transition-colors">
                        <i data-lucide="chevron-down" class="w-3 h-3" id="${cardId}-icon"></i> Read More
                    </button>
                ` : ''}
                <button onclick="askAI(${idx})" class="text-[10px] text-slate-500 hover:text-purple-400 flex items-center gap-1 transition-colors">
                    <i data-lucide="sparkles" class="w-3 h-3"></i> Ask AI
                </button>
                <button onclick="toggleRating(${idx})" class="text-[10px] text-slate-500 hover:text-amber-400 flex items-center gap-1 transition-colors">
                    <i data-lucide="star" class="w-3 h-3"></i> Rate
                </button>
                <button onclick="dismissSignal(${idx})" class="text-[10px] text-slate-500 hover:text-red-400 flex items-center gap-1 transition-colors" title="Dismiss — teaches the scorer this is noise">
                    <i data-lucide="x-circle" class="w-3 h-3"></i> Dismiss
                </button>
            </div>
            ${contentDiffers ? `
                <div id="${cardId}" class="card-content mt-2 text-xs text-slate-400 leading-relaxed border-t border-slate-800/50 pt-2">${esc(item.content).substring(0, 800)}${item.content.length > 800 ? '...' : ''}</div>
            ` : ''}
            <div id="ai-response-${idx}" class="hidden mt-2 border-t border-purple-900/30 pt-2">
                <div class="flex items-center gap-2 mb-2">
                    <input id="ai-input-${idx}" type="text" placeholder="Ask about this signal..." class="flex-1 bg-slate-900/50 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300 placeholder-slate-600 focus:border-purple-500 focus:outline-none" onkeydown="if(event.key==='Enter')submitAI(${idx})">
                    <button onclick="submitAI(${idx})" class="text-[10px] px-2 py-1 bg-purple-900/40 text-purple-300 rounded hover:bg-purple-800/50 transition-colors">Send</button>
                </div>
                <div id="ai-answer-${idx}" class="text-xs text-slate-400 leading-relaxed"></div>
            </div>
            <div id="rating-panel-${idx}" class="hidden mt-2 border-t border-amber-900/30 pt-2">
                <div class="flex items-center gap-2 mb-2">
                    <span class="text-[10px] text-amber-400 font-medium">Your score:</span>
                    <select id="rating-score-${idx}" class="bg-slate-900/50 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300 focus:border-amber-500 focus:outline-none">
                        <option value="">--</option>
                        ${[10,9,8,7,6,5,4,3,2,1].map(n => `<option value="${n}" ${Math.round(s) === n ? 'selected' : ''}>${n}.0</option>`).join('')}
                    </select>
                    <input id="rating-note-${idx}" type="text" placeholder="Why? (optional)" class="flex-1 bg-slate-900/50 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300 placeholder-slate-600 focus:border-amber-500 focus:outline-none" onkeydown="if(event.key==='Enter')submitRating(${idx})">
                    <button onclick="submitRating(${idx})" class="text-[10px] px-2 py-1 bg-amber-900/40 text-amber-300 rounded hover:bg-amber-800/50 transition-colors">Save</button>
                </div>
                <div id="rating-status-${idx}" class="text-[10px] text-slate-500"></div>
            </div>
        </div>
        `;
    }).join('');
    
    // Update nav badges after render (but not during search - too frequent)
    if (!searchQuery) renderNav();
    lucide.createIcons();
}

// === SAVED VIEW: GRID / LIST TOGGLE ===
function toggleSavedViewMode() {
    const isGrid = localStorage.getItem('savedViewGrid') === '1';
    localStorage.setItem('savedViewGrid', isGrid ? '0' : '1');
    renderFeed();
}

// === DRAG-TO-REORDER SOURCE GROUPS (Finance / Politics) ===
function initSourceDragDrop() {
    const container = document.getElementById('source-groups-container');
    if (!container) return;
    
    const orderKey = container.dataset.orderKey;
    let draggedEl = null;
    
    container.querySelectorAll('.source-group').forEach(group => {
        group.addEventListener('dragstart', (e) => {
            draggedEl = group;
            group.style.opacity = '0.4';
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', group.dataset.source);
        });
        
        group.addEventListener('dragend', () => {
            group.style.opacity = '1';
            draggedEl = null;
            // Remove all drop indicators
            container.querySelectorAll('.source-group').forEach(g => {
                g.style.borderTop = '';
                g.style.borderBottom = '';
            });
        });
        
        group.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            if (!draggedEl || draggedEl === group) return;
            
            const rect = group.getBoundingClientRect();
            const midY = rect.top + rect.height / 2;
            
            // Clear all indicators
            container.querySelectorAll('.source-group').forEach(g => {
                g.style.borderTop = '';
                g.style.borderBottom = '';
            });
            
            // Show drop indicator
            if (e.clientY < midY) {
                group.style.borderTop = '2px solid rgba(16, 185, 129, 0.5)';
            } else {
                group.style.borderBottom = '2px solid rgba(16, 185, 129, 0.5)';
            }
        });
        
        group.addEventListener('dragleave', () => {
            group.style.borderTop = '';
            group.style.borderBottom = '';
        });
        
        group.addEventListener('drop', (e) => {
            e.preventDefault();
            if (!draggedEl || draggedEl === group) return;
            
            const rect = group.getBoundingClientRect();
            const midY = rect.top + rect.height / 2;
            
            if (e.clientY < midY) {
                container.insertBefore(draggedEl, group);
            } else {
                container.insertBefore(draggedEl, group.nextSibling);
            }
            
            // Clear indicators
            container.querySelectorAll('.source-group').forEach(g => {
                g.style.borderTop = '';
                g.style.borderBottom = '';
            });
            
            // Save new order to localStorage
            const newOrder = [...container.querySelectorAll('.source-group')].map(g => g.dataset.source);
            try { localStorage.setItem(orderKey, JSON.stringify(newOrder)); } catch(e) {}
        });
    });
}


// === HEADLINE SOURCE COLLAPSE/EXPAND ===
function toggleHeadlineSource(source) {
    if (!window._collapsedSources) window._collapsedSources = {};
    if (!window._collapsedSources[activeRoot]) window._collapsedSources[activeRoot] = {};
    if (window._collapsedSources[activeRoot][source]) {
        delete window._collapsedSources[activeRoot][source];
    } else {
        window._collapsedSources[activeRoot][source] = true;
    }
    renderFeed();
}

function toggleAllHeadlineSources() {
    if (!window._collapsedSources) window._collapsedSources = {};
    if (!window._collapsedSources[activeRoot]) window._collapsedSources[activeRoot] = {};
    // If any are collapsed, expand all; otherwise collapse all
    if (Object.keys(window._collapsedSources[activeRoot]).length > 0) {
        window._collapsedSources[activeRoot] = {};
    } else {
        const container = document.getElementById('source-groups-container');
        if (container) {
            container.querySelectorAll('.source-group').forEach(g => {
                if (g.dataset.source) window._collapsedSources[activeRoot][g.dataset.source] = true;
            });
        }
    }
    renderFeed();
}
