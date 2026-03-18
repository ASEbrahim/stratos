// === STATE ===
let data = null;
let marketData = {};
let newsData = [];
let financeNewsData = [];
let politicsNewsData = [];
let customNewsData = [];
let jobsNewsData = [];
let activeRoot = 'dashboard';
let marketChart = null;
let currentSymbol = 'NVDA';
let currentTimeframe = '1m';
let currentDataVersion = null;
let statusPollInterval = null;
let isAutoLoading = false; // Prevent multiple auto-reloads
let _scanPollingInterval = null;
var _isScanRunning = false;
let candleData = { opens: [], highs: [], lows: [], volumes: [], timestamps: [] }; // Per-candle data for tooltip

// F11: Collapsible widget sections
function toggleWidgetSection(bodyId, headerEl) {
    const body = document.getElementById(bodyId);
    if (!body) return;
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? '' : 'none';
    const chevron = headerEl?.querySelector('[data-lucide="chevron-down"], [data-lucide="chevron-right"]');
    if (chevron) {
        chevron.setAttribute('data-lucide', isHidden ? 'chevron-down' : 'chevron-right');
        lucide.createIcons();
    }
}

let navSections = buildNavSections([]);

function buildNavSections(dynamicCats) {
    // Build intelligence tabs from dynamic categories or use defaults
    let intelItems;
    // Filter to only enabled categories
    const enabledCats = (dynamicCats || []).filter(cat => cat.enabled !== false);
    const hasDynamic = enabledCats.length > 0;
    
    if (hasDynamic) {
        // Map dynamic categories to nav items using their root for data filtering
        intelItems = enabledCats.map(cat => ({
            id: cat.id,
            label: cat.label || cat.name || 'Untitled',
            icon: cat.icon || 'tag',
            subtitle: `${(cat.items || cat.keywords || []).length} tracking keywords`,
            dynamic: true,
            root: cat.root || 'kuwait',
            scorer_type: cat.scorer_type || 'auto',
            keywords: cat.items || cat.keywords || []
        }));
    } else {
        // No fallback tabs — user must run the wizard to configure categories
        intelItems = [];
    }
    
    // Overview section: AI & Tech only shown when no dynamic categories
    // (dynamic categories already cover tech topics — showing both causes duplicates)
    const overviewItems = [
        { id: 'dashboard', label: 'Summary', icon: 'layout-dashboard', subtitle: 'Strategic Intelligence Stream' },
        { id: 'markets_view', label: 'Markets', icon: 'trending-up', subtitle: 'Expanded Charts & Analysis' },
        { id: 'saved', label: 'Saved', icon: 'bookmark', subtitle: 'Your Saved Signals' },
    ];
    // AI & Tech tab only useful with dynamic categories covering tech topics
    // Removed: no longer added as fallback

    // Check if profile is empty (no role, no categories configured)
    const _cfgCheck = typeof configData !== 'undefined' ? configData : null;
    const profileEmpty = !_cfgCheck?.profile?.role?.trim() && !hasDynamic && 
        !_cfgCheck?.news?.career?.keywords?.length;

    // If profile is empty, only show Overview + Settings (no categories/feeds)
    if (profileEmpty) {
        return [
            {
                label: 'Overview',
                collapsible: true,
                storageKey: 'navOverviewCollapsed',
                items: [
                    { id: 'dashboard', label: 'Summary', icon: 'layout-dashboard', subtitle: 'Get Started' },
                    { id: 'markets_view', label: 'Markets', icon: 'trending-up', subtitle: 'Expanded Charts & Analysis' },
                    { id: 'saved', label: 'Saved', icon: 'bookmark', subtitle: 'Your Saved Signals' },
                ]
            },
            {
                label: null,
                collapsible: false,
                items: [
                    { id: 'settings', label: 'Settings', icon: 'settings', subtitle: 'Set Up Your Profile' }
                ]
            }
        ];
    }
    
    // Feeds section — Finance, Politics, Jobs, Custom
    const feedItems = [
        { id: 'finance_news', label: 'Finance', icon: 'bar-chart-3', subtitle: 'Market News & Analysis' },
        { id: 'politics', label: 'Politics', icon: 'landmark', subtitle: 'Global Headlines' },
        { id: 'jobs_feeds', label: 'Jobs', icon: 'briefcase', subtitle: 'Career & Job Listings' },
    ];
    // Add custom tab if feeds exist (guard with typeof — configData is in settings.js which may load later)
    const _cfg = typeof configData !== 'undefined' ? configData : null;
    const hasCustomFeeds = (_cfg?.custom_feeds?.length > 0) || (typeof customFeeds !== 'undefined' && customFeeds.length > 0);
    if (hasCustomFeeds) {
        const tabName = _cfg?.custom_tab_name || (typeof customTabName !== 'undefined' ? customTabName : 'Custom');
        feedItems.push({ id: 'custom_feeds', label: tabName, icon: 'rss', subtitle: tabName + ' Feeds' });
    }
    
    return [
        {
            label: 'Overview',
            collapsible: true,
            storageKey: 'navOverviewCollapsed',
            items: overviewItems
        },
        {
            label: 'Categories',
            collapsible: true,
            storageKey: 'navIntelCollapsed',
            items: intelItems
        },
        {
            label: 'Feeds',
            collapsible: true,
            storageKey: 'navFeedsCollapsed',
            items: feedItems
        },
        {
            label: null,
            collapsible: false,
            items: [
                { id: 'youtube_kb', label: 'YouTube', icon: 'play-circle', subtitle: 'Knowledge Base' }
            ]
        },
        {
            label: null,
            collapsible: false,
            items: [
                { id: 'settings', label: 'Settings', icon: 'settings', subtitle: 'Customize Your Intelligence Feed' }
            ]
        }
    ];
}

// ═══════════════════════════════════════════════════════════
// MARKETS EXPANDED VIEW
// ═══════════════════════════════════════════════════════════
let _marketsExpanded = false;

function toggleMarketsExpanded(expand) {
    const mainGrid = document.getElementById('main-content');
    const feedCol = document.getElementById('feed-column');
    const sidebarCol = document.getElementById('sidebar-column');
    const marketsWidget = document.getElementById('markets-widget');
    const chartWrapper = document.getElementById('chart-wrapper');
    const agentPanel = document.getElementById('agent-panel');
    const briefingPanel = document.getElementById('briefing-panel');
    const feedControls = document.getElementById('feed-controls');
    const feedCount = document.getElementById('feed-count');
    const newsFeed = document.getElementById('news-feed');
    const assetAnalysis = document.getElementById('asset-analysis');
    
    if (!mainGrid || !feedCol || !sidebarCol || !marketsWidget) return;
    
    if (expand && !_marketsExpanded) {
        _marketsExpanded = true;
        
        // Swap layout: markets becomes full-width primary, agent below
        mainGrid.className = 'hidden grid grid-cols-1 gap-6';
        mainGrid.classList.remove('hidden');
        
        // Move markets widget to feed column (top)
        feedCol.insertBefore(marketsWidget, feedCol.firstChild);
        
        // Enlarge chart
        if (chartWrapper) chartWrapper.style.height = '400px';
        
        // Move asset analysis below markets
        if (assetAnalysis) feedCol.insertBefore(assetAnalysis, marketsWidget.nextSibling);
        
        // Show agent panel below asset analysis (for market discussion)
        if (agentPanel) {
            agentPanel.classList.remove('hidden');
            feedCol.insertBefore(agentPanel, assetAnalysis ? assetAnalysis.nextSibling : marketsWidget.nextSibling);
            // Auto-open the chat body in markets view
            const body = document.getElementById('agent-body');
            const chevron = document.getElementById('agent-chevron');
            if (body) body.classList.remove('hidden');
            if (chevron) chevron.style.transform = 'rotate(180deg)';
            agentOpen = true;
        }
        
        // Hide ALL regular feed content (search bar, filters, signals, briefing)
        if (briefingPanel) briefingPanel.classList.add('hidden');
        if (feedControls) feedControls.classList.add('hidden');
        if (feedCount) feedCount.classList.add('hidden');
        if (newsFeed) newsFeed.classList.add('hidden');
        const statusBanner = document.getElementById('status-banner');
        if (statusBanner) statusBanner.classList.add('hidden');
        
        // Hide sidebar (content moved to main)
        sidebarCol.classList.add('hidden');
        
        // Resize chart (ResizeObserver handles it, but nudge just in case)
        if (typeof _tvChart !== 'undefined' && _tvChart) {
            const el = document.getElementById('tv-chart-container');
            if (el) setTimeout(() => _tvChart.applyOptions({width:el.clientWidth,height:el.clientHeight}), 100);
        }
        
    } else if (!expand && _marketsExpanded) {
        _marketsExpanded = false;
        
        // Restore layout
        mainGrid.className = 'hidden grid grid-cols-1 xl:grid-cols-3 gap-8';
        mainGrid.classList.remove('hidden');
        
        // Move markets widget back to sidebar
        sidebarCol.insertBefore(marketsWidget, sidebarCol.firstChild);
        
        // Move asset analysis back to sidebar
        if (assetAnalysis) sidebarCol.insertBefore(assetAnalysis, marketsWidget.nextSibling);
        
        // Move agent back to feed column top
        if (agentPanel) feedCol.insertBefore(agentPanel, feedCol.firstChild);
        
        // Restore chart size
        if (chartWrapper) chartWrapper.style.height = '220px';
        
        // Show regular feed content
        if (briefingPanel) briefingPanel.classList.remove('hidden');
        if (feedControls) feedControls.classList.remove('hidden');
        if (feedCount) feedCount.classList.remove('hidden');
        if (newsFeed) newsFeed.classList.remove('hidden');
        const statusBanner = document.getElementById('status-banner');
        if (statusBanner) statusBanner.classList.remove('hidden');
        
        // Show sidebar
        sidebarCol.classList.remove('hidden');
        
        // Resize chart back
        if (typeof _tvChart !== 'undefined' && _tvChart) {
            const el = document.getElementById('tv-chart-container');
            if (el) setTimeout(() => _tvChart.applyOptions({width:el.clientWidth,height:el.clientHeight}), 100);
        }
    }
}

function rebuildNavFromConfig() {
    const dynamicCats = configData?.dynamic_categories || [];
    navSections = buildNavSections(dynamicCats);
    window.navItems = navSections.flatMap(s => s.items);
    renderNav();
}

// Flat list for lookups
let navItems = navSections.flatMap(s => s.items);

/* ═══════════════════════════════════════════════════════════
   SAVED SIGNALS
   ═══════════════════════════════════════════════════════════ */
function _savedKey() {
    const profile = getActiveProfile() || 'default';
    return `stratos_saved_${profile}`;
}

let _savedSignalsCache = null;

function getSavedSignals() {
    // Return localStorage cache synchronously; backend sync happens async
    try { return JSON.parse(localStorage.getItem(_savedKey()) || '[]'); }
    catch(e) { return []; }
}

function _saveSavedSignals(arr) {
    localStorage.setItem(_savedKey(), JSON.stringify(arr));
}

// Sync saved signals from backend → localStorage (call on login/page load)
async function syncSavedSignals() {
    try {
        const r = await fetch('/api/saved-signals', {
            headers: { 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' }
        });
        if (!r.ok) return;
        const data = await r.json();
        const serverSignals = data.signals || [];
        if (serverSignals.length > 0) {
            // Merge: server is source of truth, but keep local extras with matching URLs
            const local = getSavedSignals();
            const urlMap = {};
            local.forEach(s => { if (s.url) urlMap[s.url] = s; });
            // Server signals take priority, but enrich with local data (summary, content)
            const merged = serverSignals.map(s => {
                const localMatch = urlMap[s.url];
                return {
                    ...s,
                    summary: s.summary || (localMatch ? localMatch.summary : ''),
                    content: localMatch ? localMatch.content : '',
                    score_reason: localMatch ? localMatch.score_reason : '',
                    timestamp: localMatch ? localMatch.timestamp : s.saved_at,
                };
            });
            _saveSavedSignals(merged);
        }
        if (typeof renderFeed === 'function') renderFeed();
        if (typeof renderNav === 'function') renderNav();
    } catch(e) { /* silent — localStorage fallback works */ }
}

function isSignalSaved(item) {
    const saved = getSavedSignals();
    return saved.some(s => s.url === item.url && s.title === item.title);
}

function toggleSaveSignal(idx) {
    // Use the currently rendered feed items
    const item = (window.currentItems || [])[idx];
    if (!item) return;

    const saved = getSavedSignals();
    const existIdx = saved.findIndex(s => s.url === item.url && s.title === item.title);

    if (existIdx >= 0) {
        saved.splice(existIdx, 1);
        if (typeof showToast === 'function') showToast('Removed from saved');
        // Remove from backend
        fetch('/api/unsave-signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            body: JSON.stringify({ url: item.url })
        }).catch(function() {});
    } else {
        saved.unshift({
            title: item.title,
            url: item.url,
            summary: item.summary || '',
            source: item.source || '',
            score: item.score || 0,
            root: item.root || 'global',
            category: item.category || '',
            timestamp: item.timestamp || new Date().toISOString(),
            saved_at: new Date().toISOString(),
            score_reason: item.score_reason || '',
            content: item.content || '',
        });
        if (typeof showToast === 'function') showToast('Signal saved');
        // Tier 1 feedback loop: report save to backend
        _sendFeedback(item, 'save');
    }

    _saveSavedSignals(saved);
    renderFeed();
    renderNav();
}

// ═══════════════════════════════════════════════════════════════
// TIER 1: Feedback loop — implicit signals to backend
// ═══════════════════════════════════════════════════════════════

function _sendFeedback(item, action, userScore) {
    if (!item) return;
    fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Auth-Token': authToken || '' },
        signal: AbortSignal.timeout(15000),
        body: JSON.stringify({
            news_id: item.id || '',
            title: item.title || '',
            url: item.url || '',
            root: item.root || '',
            category: item.category || '',
            ai_score: item.score || 0,
            user_score: userScore !== undefined ? userScore : null,
            action: action
        })
    }).catch(function() {}); // Fire-and-forget, don't block UI
}

function dismissSignal(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    item._dismissed = true;
    _sendFeedback(item, 'dismiss');
    if (typeof showToast === 'function') showToast('Dismissed — scorer will learn from this', 'info', 2000);
    renderFeed();
}

function thumbsUpSignal(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    item._thumbs = 'up';
    _sendFeedback(item, 'thumbs_up', 9.0);
    if (typeof showToast === 'function') showToast('\u{1F44D} Marked as relevant — will train scorer', 'success', 2000);
    renderFeed();
}

function thumbsDownSignal(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    item._thumbs = 'down';
    _sendFeedback(item, 'thumbs_down', 1.0);
    if (typeof showToast === 'function') showToast('\u{1F44E} Marked as noise — will train scorer', 'info', 2000);
    renderFeed();
}

function trackSignalClick(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    _sendFeedback(item, 'click');
}

function removeSavedSignal(savedIdx) {
    const saved = getSavedSignals();
    if (savedIdx >= 0 && savedIdx < saved.length) {
        saved.splice(savedIdx, 1);
        _saveSavedSignals(saved);
        if (typeof showToast === 'function') showToast('Removed from saved');
        renderFeed();
        renderNav();
    }
}

/**
 * Word-boundary keyword matching for dynamic category filtering.
 * Short keywords (≤4 chars) use word-boundary regex to prevent false positives
 * (e.g., 'SLB' won't match 'possible', 'KOC' won't match 'knock').
 * Longer keywords use substring match (safe — 'Halliburton' won't false-positive).
 */
function matchesKeyword(text, kw) {
    if (kw.length <= 4) {
        // Word-boundary match for short terms (SLB, CGG, IOI, KOC, etc.)
        try {
            return new RegExp('\\b' + kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i').test(text);
        } catch(e) {
            return text.includes(kw);
        }
    }
    return text.includes(kw);
}

/**
 * Check if a news item belongs to any dynamic category.
 * Used by AI & Tech to exclude items already claimed by dynamic tabs.
 */
function matchesAnyDynamicCategory(item) {
    const dynItems = (window.navItems || []).filter(n => n.dynamic && n.keywords && n.keywords.length > 0);
    if (dynItems.length === 0) return false;
    
    // Direct category match (backend tagged this item with a dynamic category ID)
    if (dynItems.some(n => item.category === n.id)) return true;
    
    // Keyword match
    const text = ((item.title || '') + ' ' + (item.summary || '') + ' ' + (item.category || '')).toLowerCase();
    return dynItems.some(nav => 
        nav.keywords.some(kw => matchesKeyword(text, kw.toLowerCase()))
    );
}

// === HELPERS ===
function esc(text) {
    if (!text) return "";
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateClock() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function setOnline() {
    document.getElementById('system-status').innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-emerald-500 status-dot"></span> Online';
    document.getElementById('system-status').className = 'text-[11px] text-emerald-500 flex items-center gap-2 mt-1 online';
}

function setOffline() {
    document.getElementById('system-status').innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-red-500"></span> Offline';
    document.getElementById('system-status').className = 'text-[11px] text-red-500 flex items-center gap-2 mt-1';
}

// === INIT ===
async function init() {
    renderNav();  // Render default nav immediately (fast first paint)
    updateClock();
    setInterval(updateClock, 1000);

    // Fetch config early to rebuild nav from dynamic categories
    // This is lightweight — just fetches JSON, doesn't populate settings forms
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            configData = await response.json();
            rebuildNavFromConfig();  // Replace default nav with dynamic categories
        }
    } catch(e) {
        console.warn('Could not load config for nav:', e);
    }

    // Prevent status polling from triggering auto-reload during initial load
    isAutoLoading = true;

    await loadData();

    // Sync saved signals from backend (non-blocking)
    syncSavedSignals();

    // Now that initial load is complete, start status polling
    startStatusPolling();

    // Allow future auto-reloads
    isAutoLoading = false;

    // Show active profile in sidebar
    _updateSidebarProfile();

    // Restore display settings (density, font size, auto-refresh)
    _restoreDisplaySettings();

    // Show notification bell if permission not yet granted
    _updateNotifBellState();

    // Auto-start guided tour for new users (after a delay so page settles)
    setTimeout(function() {
        if (typeof maybeStartTour === 'function') maybeStartTour();
    }, 1200);

    // Pulse help button for empty profiles
    _pulseHelpIfNew();

    // Deferred agent warmup — pre-load model into VRAM
    setTimeout(function() {
        fetch('/api/agent-warmup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
            .then(function() { console.log('[Agent] Model pre-loaded'); })
            .catch(function() { /* silent — warmup is best-effort */ });
    }, 5000);
}

// === PAGE VISIBILITY — pause polling when tab is hidden ===
let _savedStatusInterval = null;
let _savedAutoRefreshTimer = null;
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Tab hidden: pause status polling and auto-refresh
        if (typeof statusPollInterval !== 'undefined' && statusPollInterval) {
            _savedStatusInterval = true;
            clearInterval(statusPollInterval);
            statusPollInterval = null;
        }
        if (typeof _autoRefreshTimer !== 'undefined' && _autoRefreshTimer) {
            _savedAutoRefreshTimer = true;
            clearInterval(_autoRefreshTimer);
            _autoRefreshTimer = null;
        }
    } else {
        // Tab visible: resume polling
        if (_savedStatusInterval) {
            _savedStatusInterval = null;
            if (typeof checkStatus === 'function' && typeof _fallbackToPolling === 'function') {
                _fallbackToPolling();
            }
        }
        if (_savedAutoRefreshTimer) {
            _savedAutoRefreshTimer = null;
            // Re-read the auto-refresh interval from the select element
            var secs = parseInt(document.getElementById('cfg-auto-refresh')?.value || '0');
            if (secs > 0 && typeof _setAutoRefresh === 'function') _setAutoRefresh(secs);
        }
        // Immediate status check on return — but only if authenticated
        if (typeof checkStatus === 'function' && typeof getAuthToken === 'function' && getAuthToken()) checkStatus();
    }
});

function _updateSidebarProfile() {
    const name = typeof getActiveProfile === 'function' ? getActiveProfile() : '';
    const profileEl = document.getElementById('sidebar-profile');
    const loginEl = document.getElementById('sidebar-login');
    if (!profileEl) return;
    if (!name) {
        profileEl.style.display = 'none';
        if (loginEl) loginEl.style.display = '';
        return;
    }
    profileEl.style.display = '';
    if (loginEl) loginEl.style.display = 'none';
    const nameEl = document.getElementById('sidebar-profile-name');
    const avatarEl = document.getElementById('sidebar-profile-avatar');
    if (nameEl) nameEl.textContent = name;
    if (avatarEl) {
        _applyAvatarToElement(avatarEl, name);
    }
}

function _getAvatarKey() {
    return 'stratos_avatar_' + (typeof getActiveProfile === 'function' ? getActiveProfile() || 'default' : 'default');
}

function _applyAvatarToElement(el, name) {
    const avatarUrl = localStorage.getItem(_getAvatarKey());
    if (avatarUrl && avatarUrl.startsWith('data:')) {
        el.textContent = '';
        el.style.backgroundImage = 'url(' + avatarUrl + ')';
        el.style.backgroundSize = 'cover';
        el.style.backgroundPosition = 'center';
    } else {
        const saved = window._savedAvatarInitials || '';
        const ini = saved || (name || '').split(/[\s_-]+/).map(w => w[0] || '').join('').toUpperCase().slice(0,2) || '?';
        el.textContent = ini;
        el.style.backgroundImage = '';
    }
}

function _applyAvatarEverywhere(dataUrl) {
    // Sidebar avatar
    var sa = document.getElementById('sidebar-profile-avatar');
    if (sa) { sa.textContent = ''; sa.style.backgroundImage = 'url(' + dataUrl + ')'; sa.style.backgroundSize = 'cover'; sa.style.backgroundPosition = 'center'; }
    // Settings preview
    var sp = document.getElementById('profile-avatar-preview');
    if (sp) { sp.textContent = ''; sp.style.backgroundImage = 'url(' + dataUrl + ')'; sp.style.backgroundSize = 'cover'; sp.style.backgroundPosition = 'center'; }
    if (typeof showToast === 'function') showToast('Avatar updated', 'success');
}

function _handleAvatarUpload(e) {
    var file = e.target.files[0];
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) { if (typeof showToast === 'function') showToast('Image too large (max 5MB)', 'error'); return; }
    var reader = new FileReader();
    reader.onload = function(ev) {
        var img = new Image();
        img.onload = function() {
            var canvas = document.createElement('canvas');
            canvas.width = canvas.height = 96;
            var ctx = canvas.getContext('2d');
            var size = Math.min(img.width, img.height);
            var sx = (img.width - size) / 2, sy = (img.height - size) / 2;
            ctx.drawImage(img, sx, sy, size, size, 0, 0, 96, 96);
            var dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            localStorage.setItem(_getAvatarKey(), dataUrl);
            _applyAvatarEverywhere(dataUrl);
            // Sync to backend for cross-device access
            fetch('/api/update-profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ avatar_image: dataUrl })
            }).catch(function() {});
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
}

/* ═══ Profile Settings ═══ */
function _initProfileSettings() {
    const name = typeof getActiveProfile === 'function' ? getActiveProfile() : '';
    const nameInput = document.getElementById('profile-display-name');
    const avatarInput = document.getElementById('profile-avatar-input');
    const avatarPreview = document.getElementById('profile-avatar-preview');
    if (nameInput && name) nameInput.value = name;
    if (avatarPreview && name) {
        _applyAvatarToElement(avatarPreview, name);
    }
    if (avatarInput) {
        // Load saved avatar initials from backend, fall back to auto-generated
        const savedAvatar = window._savedAvatarInitials || '';
        if (savedAvatar) {
            avatarInput.value = savedAvatar;
        } else if (name) {
            const ini = name.split(/[\s_-]+/).map(w => w[0] || '').join('').toUpperCase().slice(0,2);
            avatarInput.value = ini;
        }
    }
    /* Load email from profile */
    const emailInput = document.getElementById('profile-email');
    if (emailInput && typeof configData !== 'undefined' && configData?.profile?.email) {
        emailInput.value = configData.profile.email;
    }
}

function _togglePinChange() {
    const fields = document.getElementById('pin-change-fields');
    const chev = document.getElementById('pin-change-chevron');
    if (!fields) return;
    fields.classList.toggle('hidden');
    if (chev) chev.style.transform = fields.classList.contains('hidden') ? '' : 'rotate(180deg)';
}

async function _updatePin() {
    const currentPin = document.getElementById('profile-current-pin')?.value.trim();
    const newPin = document.getElementById('profile-new-pin')?.value.trim();
    const confirmPin = document.getElementById('profile-confirm-pin')?.value.trim();
    const errEl = document.getElementById('profile-error-msg');
    if (errEl) { errEl.classList.add('hidden'); errEl.textContent = ''; }
    if (!currentPin || !newPin) {
        if (errEl) { errEl.textContent = 'Enter current and new PIN'; errEl.classList.remove('hidden'); }
        return;
    }
    if (newPin !== confirmPin) {
        if (errEl) { errEl.textContent = 'New PINs do not match'; errEl.classList.remove('hidden'); }
        return;
    }
    try {
        const r = await fetch('/api/update-profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ current_pin: currentPin, new_pin: newPin })
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'PIN update failed');
        document.getElementById('profile-current-pin').value = '';
        document.getElementById('profile-new-pin').value = '';
        document.getElementById('profile-confirm-pin').value = '';
        const statusEl = document.getElementById('profile-save-status');
        if (statusEl) { statusEl.textContent = 'PIN updated'; statusEl.style.opacity = '1'; setTimeout(() => statusEl.style.opacity = '0', 2000); }
    } catch (e) {
        if (errEl) { errEl.textContent = e.message; errEl.classList.remove('hidden'); }
    }
}

function _forgotPin() {
    const emailInput = document.getElementById('profile-email');
    const email = emailInput ? emailInput.value.trim() : '';
    if (!email) {
        if (typeof showToast === 'function') showToast('Enter your account email first', 'error');
        if (emailInput) { emailInput.focus(); emailInput.style.borderColor = '#f87171'; setTimeout(() => emailInput.style.borderColor = '', 2000); }
        return;
    }
    if (typeof showToast === 'function') showToast('PIN reset is not available yet — contact your administrator', 'info');
}

async function _deleteAccount() {
    const confirmed = await stratosConfirm('This will permanently remove all your data and profiles. This action cannot be undone.', { title: 'Delete Account', okText: 'Delete', cancelText: 'Keep Account' });
    if (!confirmed) return;
    const password = await stratosPrompt({ title: 'Confirm Deletion', label: 'Enter your password to confirm' });
    if (!password) return;
    try {
        const token = getAuthToken();
        const r = await fetch('/api/auth/delete-account', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
            body: JSON.stringify({ password })
        });
        const d = await r.json();
        if (r.ok) {
            clearAuthToken();
            if (typeof _clearProfileLocalStorage === 'function') _clearProfileLocalStorage();
            location.reload();
        } else {
            if (typeof showToast === 'function') showToast(d.error || 'Failed to delete account', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Connection error', 'error');
    }
}

async function _saveProfileSettings() {
    const errEl = document.getElementById('profile-error-msg');
    const statusEl = document.getElementById('profile-save-status');
    if (errEl) { errEl.classList.add('hidden'); errEl.textContent = ''; }

    const body = {};
    const nameVal = document.getElementById('profile-display-name')?.value.trim();
    const avatarVal = document.getElementById('profile-avatar-input')?.value.trim();
    const emailVal = document.getElementById('profile-email')?.value.trim();
    const currentPin = document.getElementById('profile-current-pin')?.value.trim();
    const newPin = document.getElementById('profile-new-pin')?.value.trim();

    if (nameVal) body.display_name = nameVal;
    if (avatarVal) body.avatar = avatarVal;
    if (emailVal !== undefined) body.email = emailVal;
    if (newPin) {
        body.new_pin = newPin;
        body.current_pin = currentPin;
    }

    if (Object.keys(body).length === 0) {
        if (errEl) { errEl.textContent = 'No changes to save'; errEl.classList.remove('hidden'); }
        return;
    }

    try {
        const r = await fetch('/api/update-profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Update failed');

        // Update sidebar profile display + avatar everywhere
        if (data.profile?.name) {
            // Persist custom initials so _applyAvatarToElement uses them
            if (data.profile.avatar) window._savedAvatarInitials = data.profile.avatar;
            _updateSidebarProfile();
            const sidebarName = document.getElementById('sidebar-profile-name');
            const sidebarAvatar = document.getElementById('sidebar-profile-avatar');
            const avatarPreview = document.getElementById('profile-avatar-preview');
            if (sidebarName) sidebarName.textContent = data.profile.name;
            if (sidebarAvatar) _applyAvatarToElement(sidebarAvatar, data.profile.name);
            if (avatarPreview) _applyAvatarToElement(avatarPreview, data.profile.name);
        }

        // Clear PIN fields
        const cpEl = document.getElementById('profile-current-pin');
        const npEl = document.getElementById('profile-new-pin');
        if (cpEl) cpEl.value = '';
        if (npEl) npEl.value = '';

        // Show success
        if (statusEl) {
            statusEl.textContent = 'Profile updated';
            statusEl.style.opacity = '1';
            setTimeout(() => statusEl.style.opacity = '0', 3000);
        }
        if (typeof showToast === 'function') showToast('Profile updated', 'success');
    } catch (e) {
        if (errEl) { errEl.textContent = e.message; errEl.classList.remove('hidden'); }
    }
}

/* ═══ Display Settings ═══ */
function _applyDensity(val, skipPersist) {
    document.body.className = document.body.className.replace(/density-\w+/g, '').trim();
    if (val && val !== 'normal') document.body.classList.add('density-' + val);
    localStorage.setItem('stratos_density', val);
    if (!skipPersist) _persistUIPref('density', val);
}
function _applyFontSize(val, skipPersist) {
    var sizes = { small: '14px', medium: '16px', large: '18px', xlarge: '20px' };
    var scales = { small: '0.875', medium: '1', large: '1.125', xlarge: '1.25' };
    var fs = sizes[val] || '16px';
    var sc = scales[val] || '1';
    document.documentElement.style.fontSize = fs;
    document.documentElement.style.setProperty('--ui-scale', sc);
    // Scale all Lucide icons via CSS custom property
    document.documentElement.setAttribute('data-fontsize', val || 'medium');
    localStorage.setItem('stratos_fontsize', val);
    if (!skipPersist) _persistUIPref('font_size', val);
}
function _setDefaultChartType(val) {
    localStorage.setItem('stratos_chart_type', val);
    if (typeof setChartType === 'function' && val) setChartType(val);
    _persistUIPref('chart_type', val);
}
function _persistUIPref(key, val) {
    var body = { ui_preferences: {} };
    body.ui_preferences[key] = val;
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
        body: JSON.stringify(body)
    }).catch(function() {});
}

var _autoRefreshTimer = null;
function _setAutoRefresh(secs, skipPersist) {
    localStorage.setItem('stratos_auto_refresh', secs);
    if (_autoRefreshTimer) { clearInterval(_autoRefreshTimer); _autoRefreshTimer = null; }
    var s = parseInt(secs);
    if (s > 0) {
        // Trigger immediate refresh on interval change, then start timer
        if (!skipPersist && typeof refreshMarket === 'function') refreshMarket();
        _autoRefreshTimer = setInterval(function() {
            if (typeof refreshMarket === 'function') refreshMarket();
        }, s * 1000);
    }
    // Persist to backend unless loading from server
    if (!skipPersist) {
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            body: JSON.stringify({ ui_preferences: { auto_refresh: s } })
        }).catch(function() {});
    }
}

function _toggleRetention(enabled) {
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
        body: JSON.stringify({ scoring: { retain_high_scores: enabled } })
    }).then(function(r) {
        if (!r.ok) throw new Error(r.status);
        if (typeof showToast === 'function') showToast(enabled ? 'High-score retention enabled' : 'High-score retention disabled', 'success');
    }).catch(function() {
        if (typeof showToast === 'function') showToast('Failed to update retention setting', 'error');
    });
}

// ── TTS Settings — extracted to tts-settings.js ──

function _exportSignals(fmt) {
    var data = typeof newsData !== 'undefined' ? newsData : [];
    if (!data.length) { if (typeof showToast === 'function') showToast('No signals to export', 'warning'); return; }
    var content, mime, ext;
    if (fmt === 'csv') {
        var rows = [['Title','Score','Category','Source','URL','Timestamp','Reason']];
        data.forEach(function(d) { rows.push(['"' + (d.title||'').replace(/"/g,'""') + '"', d.score||0, d.category||'', d.source||'', d.url||'', d.timestamp||'', '"' + (d.score_reason||'').replace(/"/g,'""') + '"']); });
        content = rows.map(function(r){return r.join(',');}).join('\n');
        mime = 'text/csv'; ext = 'csv';
    } else {
        content = JSON.stringify(data, null, 2);
        mime = 'application/json'; ext = 'json';
    }
    var blob = new Blob([content], { type: mime });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = 'stratos_signals_' + new Date().toISOString().slice(0,10) + '.' + ext;
    a.click(); URL.revokeObjectURL(url);
    if (typeof showToast === 'function') showToast('Exported ' + data.length + ' signals as ' + ext.toUpperCase(), 'success');
}

function _clearCache() {
    var keep = ['stratos-session', 'stratos-profile', 'stratos_avatar_'];
    var keys = [];
    for (var i = 0; i < localStorage.length; i++) {
        var k = localStorage.key(i);
        if (!keep.some(function(p) { return k.startsWith(p); })) keys.push(k);
    }
    keys.forEach(function(k) { localStorage.removeItem(k); });
    if (typeof showToast === 'function') showToast('Cache cleared (' + keys.length + ' items)', 'success');
    _updateStorageUsage();
}

function _updateStorageUsage() {
    var total = 0;
    for (var i = 0; i < localStorage.length; i++) {
        var k = localStorage.key(i);
        total += (k.length + (localStorage.getItem(k) || '').length) * 2;
    }
    var el = document.getElementById('storage-usage');
    if (el) el.textContent = Math.round(total / 1024);
}

function _restoreDisplaySettings() {
    var density = localStorage.getItem('stratos_density');
    if (density) { _applyDensity(density); var el = document.getElementById('cfg-density'); if (el) el.value = density; }
    var fontSize = localStorage.getItem('stratos_fontsize');
    if (fontSize) { _applyFontSize(fontSize); var el2 = document.getElementById('cfg-font-size'); if (el2) el2.value = fontSize; }
    var chartType = localStorage.getItem('stratos_chart_type');
    if (chartType) { var el3 = document.getElementById('cfg-chart-type'); if (el3) el3.value = chartType; if (typeof setChartType === 'function') setChartType(chartType); }
    var autoRefresh = localStorage.getItem('stratos_auto_refresh');
    if (autoRefresh) { _setAutoRefresh(autoRefresh); var el4 = document.getElementById('cfg-auto-refresh'); if (el4) el4.value = autoRefresh; }
    else { _setAutoRefresh('0'); } // Default off — DB sync restores user's preference
    // Retention toggle — read from server config
    var retainEl = document.getElementById('cfg-retain-high');
    if (retainEl && typeof configData !== 'undefined' && configData) {
        var retainOn = configData.scoring?.retain_high_scores !== false; // default true
        retainEl.checked = retainOn;
    }
    // TTS toggle — read from localStorage
    var ttsEl = document.getElementById('cfg-tts');
    if (ttsEl) {
        ttsEl.checked = localStorage.getItem('stratos_tts_enabled') !== '0';
    }
    // TTS voice picker — lazy-load voices
    if (typeof _loadTTSVoices === 'function') _loadTTSVoices();
    // STT toggle — read from localStorage
    var sttEl = document.getElementById('cfg-stt');
    if (sttEl) {
        sttEl.checked = localStorage.getItem('stratos_stt_enabled') !== '0';
    }
    _updateStorageUsage();
}

/* ═══ Profile Popup Menu ═══ */
var _profilePopupCloseHandler = null;

function _toggleProfilePopup(e) {
    e.stopPropagation();
    if (e.preventDefault) e.preventDefault();
    var popup = document.getElementById('profile-popup');
    if (!popup) return;
    var isHidden = popup.classList.contains('hidden');
    /* If closing, clean up */
    if (!isHidden) {
        popup.classList.add('hidden');
        _removeProfilePopupClose();
        return;
    }
    /* Opening */
    popup.classList.remove('hidden');
    _renderProfilePopup();
    /* Delay registering close handler so the current touch/click doesn't trigger it */
    setTimeout(function() {
        _profilePopupCloseHandler = function _close(ev) {
            if (popup.contains(ev.target)) return;
            popup.classList.add('hidden');
            _removeProfilePopupClose();
        };
        document.addEventListener('click', _profilePopupCloseHandler, true);
        document.addEventListener('touchend', _profilePopupCloseHandler, true);
    }, 300);
}

function _removeProfilePopupClose() {
    if (_profilePopupCloseHandler) {
        document.removeEventListener('click', _profilePopupCloseHandler, true);
        document.removeEventListener('touchend', _profilePopupCloseHandler, true);
        _profilePopupCloseHandler = null;
    }
}

function _renderProfilePopup() {
    var el = document.getElementById('profile-popup-content');
    if (!el) return;
    var name = typeof getActiveProfile === 'function' ? getActiveProfile() : 'User';
    var role = configData?.profile?.role || '';
    var avatarUrl = localStorage.getItem(_getAvatarKey());
    var ini = (name || '').split(/[\s_-]+/).map(function(w){return w[0]||'';}).join('').toUpperCase().slice(0,2) || '?';

    var avatarStyle = avatarUrl && avatarUrl.startsWith('data:')
        ? 'background-image:url(' + avatarUrl + ');background-size:cover;background-position:center;'
        : 'background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);color:#10b981;';

    var themes = ['midnight','coffee','rose','noir','aurora','cosmos','sakura','nebula'];
    var currentTheme = localStorage.getItem('stratos-theme') || 'midnight';
    var themeColors = {midnight:'#34d399',coffee:'#fbbf24',rose:'#fb7185',noir:'#a78bfa',aurora:'#34d399',cosmos:'#e8b931',sakura:'#f0a0b8',nebula:'#38bdf8'};
    var themeBtns = themes.map(function(t) {
        var active = t === currentTheme ? 'ring-1 ring-white/40 scale-110' : '';
        return '<button onclick="event.stopPropagation();setTheme(\'' + t + '\');_renderProfilePopup()" class="w-5 h-5 rounded-full ' + active + '" style="background:' + themeColors[t] + ';-webkit-tap-highlight-color:transparent;" title="' + t + '"></button>';
    }).join('');

    el.innerHTML =
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">' +
            '<div style="width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0;' + avatarStyle + '">' + (avatarUrl ? '' : ini) + '</div>' +
            '<div style="min-width:0;">' +
                '<div style="font-size:13px;font-weight:600;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(name) + '</div>' +
                (role ? '<div style="font-size:10px;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(role) + '</div>' : '') +
            '</div>' +
        '</div>' +
        '<div style="display:flex;gap:4px;margin-bottom:10px;align-items:center;">' +
            '<span style="font-size:9px;color:var(--text-muted);margin-right:4px;">Theme</span>' + themeBtns +
        '</div>' +
        '<div style="border-top:1px solid var(--border-strong);padding-top:8px;display:flex;flex-direction:column;gap:2px;">' +
            '<button onclick="event.stopPropagation();setActive(\'settings\');document.getElementById(\'profile-popup\').classList.add(\'hidden\');_removeProfilePopupClose()" class="text-left text-xs px-2 py-2.5 rounded hover:bg-white/5 active:bg-white/10 transition-colors" style="color:var(--text-secondary);-webkit-tap-highlight-color:transparent;"><i data-lucide="settings" style="width:12px;height:12px;display:inline;vertical-align:-2px;margin-right:6px;"></i>Account Settings</button>' +
            '<button onclick="event.stopPropagation();_exportSignals(\'json\');document.getElementById(\'profile-popup\').classList.add(\'hidden\');_removeProfilePopupClose()" class="text-left text-xs px-2 py-2.5 rounded hover:bg-white/5 active:bg-white/10 transition-colors" style="color:var(--text-secondary);-webkit-tap-highlight-color:transparent;"><i data-lucide="download" style="width:12px;height:12px;display:inline;vertical-align:-2px;margin-right:6px;"></i>Export Signals</button>' +
            '<button onclick="event.stopPropagation();logout()" class="text-left text-xs px-2 py-2.5 rounded hover:bg-red-500/10 active:bg-red-500/20 transition-colors" style="color:#f87171;-webkit-tap-highlight-color:transparent;"><i data-lucide="log-out" style="width:12px;height:12px;display:inline;vertical-align:-2px;margin-right:6px;"></i>Logout</button>' +
        '</div>' +
        '<div style="border-top:1px solid var(--border-strong);margin-top:6px;padding-top:6px;text-align:center;">' +
            '<span style="font-size:9px;color:var(--text-muted);">StratOS v1.0</span>' +
        '</div>';
    if (typeof lucide !== 'undefined') lucide.createIcons();
}

async function loadData() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('main-content').classList.add('hidden');
    
    try {
        const response = await fetch('/api/data', {signal: AbortSignal.timeout(10000)});
        if (!response.ok) { setOffline(); showError('Server error: ' + response.status); return; }
        data = await response.json();

        if (data.error) {
            setOffline();
            showError(data.error);
            return;
        }
        
        setOnline();
        marketData = data.market || {};
        newsData = data.news || [];
        
        // If this is a fresh/empty profile, don't show stale data from another user's scan
        // BUT only do this if the config confirms empty profile AND server has no real data
        const hasRole = configData?.profile?.role?.trim();
        const hasDynCats = configData?.dynamic_categories?.length > 0;
        const hasKeywords = configData?.news?.career?.keywords?.length > 0 || 
                           configData?.news?.finance?.keywords?.length > 0;
        const configLoaded = typeof configData !== 'undefined' && configData !== null;
        const serverHasNews = newsData.length > 0 && newsData.some(n => n.score >= 5);
        // Only clear data if we're SURE this is an empty profile with no real data
        if (configLoaded && !hasRole && !hasDynCats && !hasKeywords && !serverHasNews) {
            // Fresh profile — clear stale data
            newsData = [];
            marketData = {};
            data.news = [];
            data.market = {};
            data.briefing = null;
            // Clear extra feed data too
            if (typeof financeNewsData !== 'undefined') financeNewsData.length = 0;
            if (typeof politicsNewsData !== 'undefined') politicsNewsData.length = 0;
            if (typeof customNewsData !== 'undefined') customNewsData.length = 0;
        }
        
        // Get version and also fetch current status to sync
        const statusResp = await fetch('/api/status', {signal: AbortSignal.timeout(8000)});
        if (!statusResp.ok) { console.error('/api/status failed:', statusResp.status); return; }
        const status = await statusResp.json();
        currentDataVersion = status.data_version || data.meta?.generated_at || Date.now().toString();

        // Sync avatar from backend (cross-device)
        if (status.avatar_image && status.avatar_image.startsWith('data:')) {
            localStorage.setItem(_getAvatarKey(), status.avatar_image);
            _applyAvatarEverywhere(status.avatar_image);
        }
        if (status.avatar) {
            window._savedAvatarInitials = status.avatar;
        }
        // Sync email from backend
        if (status.email) {
            const emailEl = document.getElementById('profile-email');
            if (emailEl && !emailEl.value) emailEl.value = status.email;
        }
        // Sync UI state (theme/mode/stars) from backend
        if (status.ui_state && typeof _uiStateDirty === 'function' && _uiStateDirty(status.ui_state)) {
            if (typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(status.ui_state);
        }

        document.getElementById('last-updated').textContent = data.last_updated || '--:--';
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('main-content').classList.remove('hidden');
        
        initChart();
        renderTickerButtons();
        renderMarketOverview();
        
        if (Object.keys(marketData).length > 0) {
            currentSymbol = Object.keys(marketData)[0];
            updateChart(currentSymbol);
        }
        
        renderFeed();
        renderAlerts();
        updateSyncIndicators();
        lucide.createIcons();
        
    } catch (err) {
        setOffline();
        showError(err.message);
    }
}

let _sseSource = null;
let _sseConnected = false;

// Close SSE on page unload to free server resources
window.addEventListener('beforeunload', () => {
    if (_sseSource) { try { _sseSource.close(); } catch(e){} _sseSource = null; }
});

function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);
    // Try SSE first — falls back to polling automatically
    _connectSSE();
}

function _connectSSE() {
    if (_sseSource) { try { _sseSource.close(); } catch(e){} }
    
    try {
        const _sseUrl = typeof getAuthToken === 'function' && getAuthToken()
            ? '/api/events?token=' + encodeURIComponent(getAuthToken())
            : '/api/events';
        _sseSource = new EventSource(_sseUrl);
        
        _sseSource.onopen = () => {
            console.log('[SSE] Connected — polling disabled');
            _sseConnected = true;
            _statusFailCount = 0;
            // Stop polling — SSE will push updates
            if (statusPollInterval) { clearInterval(statusPollInterval); statusPollInterval = null; }
        };
        
        // Named events from backend
        _sseSource.addEventListener('scan', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleSSEScan(d);
            } catch(err) { console.warn('[SSE] parse error:', err); }
        });
        
        _sseSource.addEventListener('complete', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleSSEComplete(d);
            } catch(err) { console.warn('[SSE] parse error:', err); }
        });
        
        _sseSource.addEventListener('scan_cancelled', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleSSECancelled(d);
            } catch(err) { console.warn('[SSE] parse error:', err); }
        });

        _sseSource.addEventListener('scan_error', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleSSEError(d);
            } catch(err) {}
        });
        
        _sseSource.addEventListener('status', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleSSEStatus(d);
            } catch(err) {}
        });

        _sseSource.addEventListener('pass1_complete', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleSSEPass1Complete(d);
            } catch(err) { console.warn('[SSE] pass1_complete parse error:', err); }
        });

        _sseSource.addEventListener('briefing_ready', async (e) => {
            try {
                const res = await fetch('/api/briefing', {headers: {'X-Auth-Token': authToken}});
                if (res.ok) {
                    const briefing = await res.json();
                    if (data) { data.briefing = briefing; }
                    if (typeof renderBriefing === 'function') renderBriefing();
                }
            } catch(err) { console.debug('[SSE] briefing_ready fetch error:', err); }
        });

        _sseSource.addEventListener('critical_signal', (e) => {
            try {
                const d = JSON.parse(e.data);
                _handleCriticalSignal(d);
            } catch(err) { console.warn('[SSE] critical_signal parse error:', err); }
        });

        _sseSource.addEventListener('youtube_processing', (e) => {
            try {
                const d = JSON.parse(e.data);
                if (typeof _handleYouTubeSSE === 'function') _handleYouTubeSSE(d);
            } catch(err) { console.warn('[SSE] youtube_processing parse error:', err); }
        });

        _sseSource.addEventListener('lens_extracted', (e) => {
            try {
                const d = JSON.parse(e.data);
                if (typeof _handleLensExtracted === 'function') _handleLensExtracted(d);
            } catch(err) { console.warn('[SSE] lens_extracted parse error:', err); }
        });

        _sseSource.addEventListener('narration_resolved', (e) => {
            try {
                const d = JSON.parse(e.data);
                if (typeof _handleNarrationResolved === 'function') _handleNarrationResolved(d);
            } catch(err) { console.warn('[SSE] narration_resolved parse error:', err); }
        });

        _sseSource.onerror = () => {
            // SSE disconnected — fall back to polling
            if (_sseConnected) {
                console.log('[SSE] Disconnected — falling back to polling');
                _sseConnected = false;
            }
            if (_sseSource) { try { _sseSource.close(); } catch(e){} _sseSource = null; }
            _fallbackToPolling();
            // Retry SSE after 30s
            setTimeout(_connectSSE, 30000);
        };
        
    } catch (e) {
        console.warn('[SSE] Not supported, using polling');
        _fallbackToPolling();
    }
}

function _fallbackToPolling() {
    if (!statusPollInterval) {
        checkStatus();
        statusPollInterval = setInterval(checkStatus, 5000);
    }
}

var _scanPhase = 'fetch'; // 'fetch' or 'score'

function _updateScanBars(pct) {
    // Update both main and settings progress bars
    var bar = document.getElementById('scan-bar');
    var sBar = document.getElementById('settings-scan-bar');
    var pctEl = document.getElementById('scan-percent');
    var sPctEl = document.getElementById('settings-scan-percent');
    var phaseEl = document.getElementById('scan-phase');
    var label = pct > 0 ? Math.round(pct) + '%' : '';
    var width = pct > 0 ? pct + '%' : '0%';
    if (bar) bar.style.width = width;
    if (sBar) sBar.style.width = width;
    if (pctEl) pctEl.textContent = label;
    if (sPctEl) sPctEl.textContent = label;
    if (phaseEl) phaseEl.textContent = _scanPhase === 'fetch' ? 'Fetching' : 'Scoring';
}

function _resetBarForPhase(phase) {
    // Instantly reset bar to 0 (no transition) then re-enable transition
    _scanPhase = phase;
    var bar = document.getElementById('scan-bar');
    var sBar = document.getElementById('settings-scan-bar');
    if (bar) { bar.style.transition = 'none'; bar.style.width = '0%'; bar.offsetHeight; bar.style.transition = ''; }
    if (sBar) { sBar.style.transition = 'none'; sBar.style.width = '0%'; sBar.offsetHeight; sBar.style.transition = ''; }
    _updateScanBars(0);
}

function _handleSSEScan(d) {
    // Ignore scan events for other profiles
    if (d.scan_profile_id && window._myProfileId && d.scan_profile_id !== window._myProfileId) return;

    // Scan in progress — update banners
    const notifContainer = document.getElementById('status-notification');
    const scanningBanner = document.getElementById('scanning-banner');
    const newdataBanner = document.getElementById('newdata-banner');
    const systemStatus = document.getElementById('system-status');
    const settingsScanBanner = document.getElementById('settings-scan-banner');
    const settingsScanProgress = document.getElementById('settings-scan-progress');
    const isOnSettings = activeRoot === 'settings';

    const progress = d.progress || d.status || 'Processing...';

    // Two-phase progress bar:
    // Phase 1 (Fetching): starting → market → news → news_done = 0→100%
    // Phase 2 (Scoring):  scoring → scoring_pass2 → discovery → briefing = 0→100%
    var pct = 0;
    var isFetchPhase = ['starting', 'market', 'news', 'news_done'].includes(d.status);
    var isScorePhase = ['scoring', 'scoring_pass2', 'discovery', 'briefing'].includes(d.status);

    // Transition from fetch to score: reset bar
    if (isScorePhase && _scanPhase === 'fetch') {
        _resetBarForPhase('score');
    } else if (isFetchPhase && _scanPhase !== 'fetch') {
        _resetBarForPhase('fetch');
    }

    if (isFetchPhase) {
        if (d.status === 'starting') pct = 10;
        else if (d.status === 'market') pct = 45;
        else if (d.status === 'news') pct = 75;
        else if (d.status === 'news_done') pct = 100;
    } else if (isScorePhase) {
        if (d.status === 'scoring' && d.scored >= 0 && d.total > 0) pct = (d.scored / d.total) * 80;
        else if (d.status === 'scoring') pct = 2;
        else if (d.status === 'scoring_pass2' && d.scored >= 0 && d.total > 0) pct = 82 + (d.scored / d.total) * 8;
        else if (d.status === 'scoring_pass2') pct = 82;
        else if (d.status === 'discovery') pct = 92;
        else if (d.status === 'briefing') pct = 96;
    }

    // Update scan title based on stage
    var title = 'Scanning in background...';
    if (_marketRefreshPending) title = 'Refreshing market data...';
    else if (d.status === 'starting') title = 'Initializing scan...';
    else if (d.status === 'market') title = 'Fetching market data...';
    else if (d.status === 'news') title = 'Fetching news articles...';
    else if (d.status === 'news_done') title = 'Articles fetched!';
    else if (d.status === 'scoring') title = 'Scoring with AI...';
    else if (d.status === 'scoring_pass2') title = 'Re-scoring deferred articles...';
    else if (d.status === 'discovery') title = 'Running discovery...';
    else if (d.status === 'briefing') title = 'Generating briefing...';
    var scanTitle = document.getElementById('scan-title');
    if (scanTitle) scanTitle.textContent = title;

    if (!isOnSettings) {
        notifContainer.classList.remove('hidden');
        scanningBanner.classList.remove('hidden');
        newdataBanner.classList.add('hidden');
        document.getElementById('scan-progress').textContent = progress;
    } else {
        notifContainer.classList.add('hidden');
    }

    if (settingsScanBanner) {
        settingsScanBanner.classList.remove('hidden');
        if (settingsScanProgress) settingsScanProgress.textContent = progress;
    }

    _updateScanBars(pct);

    systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full animate-pulse" style="background:var(--accent);"></span> Scanning...';
    systemStatus.className = 'text-[10px] flex items-center gap-2 mt-1';
    systemStatus.style.color = 'var(--accent)';

    // Keep scan button in scanning state (handles page reload during scan)
    if (!_isScanRunning) {
        _isScanRunning = true;
        _startScanPolling();
    }
    // Update button progress from SSE data
    if (isScorePhase && d.scored > 0 && d.total > 0) {
        _setScanButtonScanning(`Stop (${d.scored}/${d.total})`);
    } else {
        _setScanButtonScanning();
    }

    // Suppress data version changes during scan
    if (window._firstScanPending && d.data_version) {
        currentDataVersion = d.data_version;
    }
}

async function _handleSSEComplete(d) {
    console.log('[SSE] Scan complete, reloading data...');
    _stopScanPolling();
    _setScanButtonIdle();

    // Fill bar to 100%, then hide after a brief moment
    _scanPhase = 'score'; // ensure we're in score phase for the final 100%
    _updateScanBars(100);
    var scanTitle = document.getElementById('scan-title');
    if (scanTitle) scanTitle.textContent = 'Complete!';
    var scanProg = document.getElementById('scan-progress');
    if (scanProg) scanProg.textContent = 'Loading results...';
    // Reset phase for next scan
    setTimeout(function() { _scanPhase = 'fetch'; }, 3000);

    const systemStatus = document.getElementById('system-status');

    // First scan for fresh profile → soft refresh (no hard reload)
    if (window._firstScanPending) {
        window._firstScanPending = false;
        if (typeof showToast === 'function') showToast('First scan complete! Loading results...', 'success', 2000);
        // Re-fetch config so new dynamic categories appear in nav
        try {
            const cfgResp = await fetch('/api/config');
            if (cfgResp.ok) { configData = await cfgResp.json(); rebuildNavFromConfig(); }
        } catch(e) {}
        isAutoLoading = true;
        await loadNewData(true);
        isAutoLoading = false;
        return;
    }

    systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Loading...';
    systemStatus.className = 'text-[10px] text-emerald-400 flex items-center gap-2 mt-1';

    isAutoLoading = true;
    // Skip the generic "Data updated" toast — we'll show a specific one below
    await loadNewData(true);
    // Update version from the event
    if (d.data_version) currentDataVersion = d.data_version;
    await new Promise(r => setTimeout(r, 400));
    isAutoLoading = false;

    // Now hide banners and reset bars
    document.getElementById('scanning-banner')?.classList.add('hidden');
    document.getElementById('status-notification')?.classList.add('hidden');
    var settingsScanBanner = document.getElementById('settings-scan-banner');
    if (settingsScanBanner) settingsScanBanner.classList.add('hidden');
    _updateScanBars(0);

    // Market-only refresh: show specific toast and reset button
    if (_marketRefreshPending) {
        _marketRefreshPending = false;
        if (typeof showToast === 'function') showToast('Market data refreshed', 'success');
        var mktBtn = document.getElementById('refresh-market-btn');
        var mktIcon = mktBtn?.querySelector('svg');
        if (mktIcon) mktIcon.classList.remove('animate-spin');
        if (mktBtn) mktBtn.disabled = false;
    } else {
        if (typeof showToast === 'function') showToast('Data updated', 'success', 2000);
    }

    // Show notification bell button (user must click to grant permission)
    _updateNotifBellState();
}

async function _handleSSECancelled(d) {
    console.log('[SSE] Scan cancelled:', d);
    _stopScanPolling();
    _setScanButtonIdle();
    _marketRefreshPending = false;
    _updateScanBars(0);

    const settingsScanBanner = document.getElementById('settings-scan-banner');
    if (settingsScanBanner) settingsScanBanner.classList.add('hidden');

    document.getElementById('status-notification')?.classList.add('hidden');
    document.getElementById('scanning-banner')?.classList.add('hidden');

    const systemStatus = document.getElementById('system-status');
    systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-yellow-500"></span> Cancelled';
    systemStatus.className = 'text-[10px] text-yellow-500 flex items-center gap-2 mt-1';

    const msg = `Scan stopped: ${d.scored || 0}/${d.total || 0} scored, ${d.high || 0} high, ${d.medium || 0} medium`;
    if (typeof showToast === 'function') showToast(msg, 'warning', 5000);

    if (d.scored > 0) {
        isAutoLoading = true;
        await loadNewData();
        if (d.data_version) currentDataVersion = d.data_version;
        await new Promise(r => setTimeout(r, 500));
        isAutoLoading = false;
    }
}

async function _handleSSEPass1Complete(d) {
    console.log('[SSE] Pass 1 complete — loading partial results:', d);
    if (d.data_version) currentDataVersion = d.data_version;
    isAutoLoading = true;
    try {
        await loadNewData(true);
    } catch(e) { console.warn('[SSE] Pass 1 data load failed:', e); }
    isAutoLoading = false;
    if (d.deferred > 0 && typeof showToast === 'function') {
        showToast(`Results loaded (${d.deferred} still scoring...)`, 'info', 3000);
    }
}

function _handleSSEError(d) {
    console.warn('[SSE] Scan error:', d.message);
    _stopScanPolling();
    _setScanButtonIdle();
    const settingsScanBanner = document.getElementById('settings-scan-banner');
    if (settingsScanBanner) settingsScanBanner.classList.add('hidden');
    
    document.getElementById('status-notification')?.classList.add('hidden');
    document.getElementById('scanning-banner')?.classList.add('hidden');
    
    if (typeof showToast === 'function') showToast('Scan failed: ' + (d.message || 'Unknown error'), 'error', 5000);
}

function _handleSSEStatus(d) {
    // Initial status on connect — sync state
    const systemStatus = document.getElementById('system-status');
    if (d.is_scanning) {
        _handleSSEScan({progress: d.progress, status: d.stage, scored: d.scored, total: d.total});
    } else {
        _setScanButtonIdle();
        systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Online';
        systemStatus.className = 'text-[10px] text-emerald-500 flex items-center gap-2 mt-1';
        if (d.data_version) currentDataVersion = d.data_version;
    }
}

function _handleCriticalSignal(d) {
    // In-app toast
    const label = d.category ? `[${d.category}] ` : '';
    const scoreStr = d.score ? ` (${d.score.toFixed(1)})` : '';
    if (typeof showToast === 'function') {
        showToast(`Critical: ${label}${d.title}${scoreStr}`, 'warning', 6000);
    }

    // Browser notification (if permission granted and not muted)
    if (typeof Notification !== 'undefined' && Notification.permission === 'granted' && localStorage.getItem('notif_muted') !== '1') {
        try {
            const n = new Notification('StratOS Critical Signal', {
                body: `${label}${d.title}${scoreStr}\n${d.reason || ''}`,
                icon: '/favicon.ico',
                tag: 'stratos-critical-' + (d.url || '').slice(-20),
            });
            if (d.url) {
                n.onclick = () => { window.open(d.url, '_blank'); n.close(); };
            }
        } catch(err) { console.warn('Notification error:', err); }
    }
}

function _updateNotifBellState() {
    const btn = document.getElementById('notif-bell-btn');
    if (!btn) return;
    if (typeof Notification === 'undefined') { btn.classList.add('hidden'); return; }
    if (Notification.permission === 'default') {
        // Not yet asked — show bell so user can click to grant
        btn.classList.remove('hidden');
        btn.title = 'Enable browser notifications';
        btn.classList.remove('text-amber-400');
        btn.classList.add('text-slate-600');
    } else if (Notification.permission === 'granted') {
        const muted = localStorage.getItem('notif_muted') === '1';
        btn.classList.remove('hidden');
        if (muted) {
            btn.title = 'Notifications muted — click to enable';
            btn.classList.remove('text-amber-400');
            btn.classList.add('text-slate-600');
        } else {
            btn.title = 'Notifications enabled — click to mute';
            btn.classList.remove('text-slate-600');
            btn.classList.add('text-amber-400');
        }
    } else {
        // Denied — hide bell
        btn.classList.add('hidden');
    }
}

function toggleNotifications() {
    if (typeof Notification === 'undefined') return;
    if (Notification.permission === 'default') {
        Notification.requestPermission().then(p => {
            console.log('[Notifications] Permission:', p);
            _updateNotifBellState();
            if (p === 'granted' && typeof showToast === 'function') {
                showToast('Notifications enabled — you\'ll be alerted for critical signals', 'success', 3000);
            }
        });
    } else if (Notification.permission === 'granted') {
        // Toggle mute state
        const muted = localStorage.getItem('notif_muted') === '1';
        localStorage.setItem('notif_muted', muted ? '0' : '1');
        _updateNotifBellState();
        if (typeof showToast === 'function') {
            showToast(muted ? 'Notifications enabled' : 'Notifications muted', muted ? 'success' : 'info', 2000);
        }
    }
}

let _statusFailCount = 0;

async function checkStatus() {
    // If SSE is connected, skip polling entirely
    if (_sseConnected) return;
    try {
        const response = await fetch('/api/status', { signal: AbortSignal.timeout(8000) });
        if (!response.ok) throw new Error('Status check failed');
        const status = await response.json();
        _statusFailCount = 0; // Reset on success
        
        const notifContainer = document.getElementById('status-notification');
        const scanningBanner = document.getElementById('scanning-banner');
        const newdataBanner = document.getElementById('newdata-banner');
        const systemStatus = document.getElementById('system-status');
        const settingsScanBanner = document.getElementById('settings-scan-banner');
        const settingsScanProgress = document.getElementById('settings-scan-progress');
        
        const isOnSettings = activeRoot === 'settings';
        
        // Track our profile_id for SSE and polling scoping
        if (status.my_profile_id) window._myProfileId = status.my_profile_id;

        // Only show scan banner if it's our profile's scan (or scan_profile_id unknown)
        const isOurScan = !status.scan_profile_id || !window._myProfileId || status.scan_profile_id === window._myProfileId;

        if (status.is_scanning && isOurScan) {
            // Show scanning indicator (but not main banner if on settings)
            if (!isOnSettings) {
                notifContainer.classList.remove('hidden');
                scanningBanner.classList.remove('hidden');
                newdataBanner.classList.add('hidden');
                document.getElementById('scan-progress').textContent = status.progress || 'Processing...';
            } else {
                notifContainer.classList.add('hidden');
            }

            // Update settings banner
            if (settingsScanBanner) {
                settingsScanBanner.classList.remove('hidden');
                if (settingsScanProgress) settingsScanProgress.textContent = status.progress || 'Processing...';
            }

            // Update progress bar from polling data (mirrors SSE _handleSSEScan two-phase logic)
            var pollPct = 0;
            var st = status.stage;
            var pollIsFetch = ['starting', 'market', 'news', 'news_done'].includes(st);
            var pollIsScore = ['scoring', 'scoring_pass2', 'discovery', 'briefing'].includes(st);

            if (pollIsScore && _scanPhase === 'fetch') _resetBarForPhase('score');
            else if (pollIsFetch && _scanPhase !== 'fetch') _resetBarForPhase('fetch');

            if (pollIsFetch) {
                if (st === 'starting') pollPct = 10;
                else if (st === 'market') pollPct = 45;
                else if (st === 'news') pollPct = 75;
                else if (st === 'news_done') pollPct = 100;
            } else if (pollIsScore) {
                if (st === 'scoring' && status.scored >= 0 && status.total > 0) pollPct = (status.scored / status.total) * 80;
                else if (st === 'scoring') pollPct = 2;
                else if (st === 'scoring_pass2' && status.scored >= 0 && status.total > 0) pollPct = 82 + (status.scored / status.total) * 8;
                else if (st === 'scoring_pass2') pollPct = 82;
                else if (st === 'discovery') pollPct = 92;
                else if (st === 'briefing') pollPct = 96;
            }
            _updateScanBars(pollPct);

            // Update scan title
            var pollTitle = 'Scanning in background...';
            if (st === 'starting') pollTitle = 'Initializing scan...';
            else if (st === 'market') pollTitle = 'Fetching market data...';
            else if (st === 'news') pollTitle = 'Fetching news articles...';
            else if (st === 'news_done') pollTitle = 'Articles fetched!';
            else if (st === 'scoring') pollTitle = 'Scoring with AI...';
            else if (st === 'scoring_pass2') pollTitle = 'Re-scoring deferred articles...';
            else if (st === 'discovery') pollTitle = 'Running discovery...';
            else if (st === 'briefing') pollTitle = 'Generating briefing...';
            var scanTitleEl = document.getElementById('scan-title');
            if (scanTitleEl) scanTitleEl.textContent = pollTitle;

            // Update system status to show scanning
            systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full animate-pulse" style="background:var(--accent);"></span> Scanning...';
            systemStatus.className = 'text-[10px] flex items-center gap-2 mt-1';
            systemStatus.style.color = 'var(--accent)';

            // Slow down polling while scanning to reduce tunnel load
            if (statusPollInterval) {
                clearInterval(statusPollInterval);
                statusPollInterval = setInterval(checkStatus, 8000);
            }

            // During first scan, suppress any data version changes from market-only writes
            if (window._firstScanPending) {
                currentDataVersion = status.data_version;
            }
            
        } else if (currentDataVersion && status.data_version && status.data_version !== currentDataVersion && !isAutoLoading) {
            // New data available! 
            console.log('New data detected, auto-reloading...', {current: currentDataVersion, server: status.data_version});
            isAutoLoading = true;
            
            // Hide settings banner
            if (settingsScanBanner) settingsScanBanner.classList.add('hidden');
            
            // First scan for fresh profile → soft refresh (no hard reload)
            if (window._firstScanPending) {
                window._firstScanPending = false;
                if (typeof showToast === 'function') showToast('First scan complete! Loading results...', 'success', 2000);
                // Re-fetch config so new dynamic categories appear in nav
                try {
                    const cfgResp = await fetch('/api/config');
                    if (cfgResp.ok) { configData = await cfgResp.json(); rebuildNavFromConfig(); }
                } catch(e) {}
                await loadNewData(true);
                isAutoLoading = false;
                return;
            }

            // Update system status
            systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Loading...';
            systemStatus.className = 'text-[10px] text-emerald-400 flex items-center gap-2 mt-1';

            // Auto-reload the data
            await loadNewData();
            
            // Restore normal polling speed
            if (statusPollInterval) {
                clearInterval(statusPollInterval);
                statusPollInterval = setInterval(checkStatus, 5000);
            }
            
            await new Promise(r => setTimeout(r, 500));
            isAutoLoading = false;
            return;
            
        } else if (!currentDataVersion && status.data_version && !status.is_scanning && !isAutoLoading) {
            // First load complete - data is ready, load it automatically
            console.log('First scan complete, auto-loading data...');
            isAutoLoading = true;
            await loadNewData();
            await new Promise(r => setTimeout(r, 500));
            isAutoLoading = false;
            return;
            
        } else {
            // Hide notifications, show online
            notifContainer.classList.add('hidden');
            scanningBanner.classList.add('hidden');
            newdataBanner.classList.add('hidden');

            // Hide settings banner
            if (settingsScanBanner) settingsScanBanner.classList.add('hidden');

            systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Online';
            systemStatus.className = 'text-[10px] text-emerald-500 flex items-center gap-2 mt-1';
        }

        // Sync UI state from backend (cross-device theme/mode/stars)
        if (status.ui_state && typeof _uiStateDirty === 'function' && _uiStateDirty(status.ui_state)) {
            if (typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(status.ui_state);
        }

        // Check search API status and show limit warning if needed
        try {
            const searchResponse = await fetch('/api/search-status');
            const searchStatus = await searchResponse.json();
            const searchLimitBanner = document.getElementById('search-limit-banner');

            if (searchStatus.provider === 'google' && searchStatus.limit_reached) {
                searchLimitBanner.classList.remove('hidden');
                document.getElementById('status-notification').classList.remove('hidden');
            } else {
                searchLimitBanner.classList.add('hidden');
            }
        } catch (e) {
            // Ignore search status errors
        }
        
        lucide.createIcons();
    } catch (err) {
        _statusFailCount++;
        // Exponential backoff: slow down polling on consecutive failures
        if (_statusFailCount >= 3 && statusPollInterval) {
            clearInterval(statusPollInterval);
            const backoff = Math.min(30000, 5000 * Math.pow(1.5, _statusFailCount - 3));
            console.warn(`Status check failed ${_statusFailCount}x, backing off to ${Math.round(backoff/1000)}s`);
            statusPollInterval = setInterval(checkStatus, backoff);
        }
        // Server not responding - show offline
        const systemStatus = document.getElementById('system-status');
        systemStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span> Offline';
        systemStatus.className = 'text-[10px] text-red-500 flex items-center gap-2 mt-1';
    }
}

async function loadNewData(skipToast) {
    // Silent background update — no loading spinner, no hiding content
    // This prevents disruption on other devices when one device triggers a scan
    try {
        const response = await fetch('/api/data');
        if (!response.ok) { console.error('/api/data failed:', response.status); return; }
        const newData = await response.json();

        if (newData.error) {
            console.warn('Data load error:', newData.error);
            return;
        }
        
        setOnline();
        
        // Update the global data object (includes briefing, timestamps, etc.)
        data = newData;
        
        marketData = newData.market || {};
        newsData = newData.news || [];
        
        // Sync version + avatar
        try {
            const statusResp = await fetch('/api/status');
            if (!statusResp.ok) throw new Error(statusResp.status);
            const statusData = await statusResp.json();
            currentDataVersion = statusData.data_version;
            if (statusData.avatar_image && statusData.avatar_image.startsWith('data:')) {
                localStorage.setItem(_getAvatarKey(), statusData.avatar_image);
            }
            if (statusData.avatar) {
                window._savedAvatarInitials = statusData.avatar;
            }
            if (statusData.email) {
                const emailEl = document.getElementById('profile-email');
                if (emailEl && !emailEl.value) emailEl.value = statusData.email;
            }
            // Sync UI state (theme/mode/stars) from backend
            if (statusData.ui_state && typeof _uiStateDirty === 'function' && _uiStateDirty(statusData.ui_state)) {
                if (typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(statusData.ui_state);
            }
        } catch (e) {
            currentDataVersion = Date.now().toString();
        }

        document.getElementById('last-updated').textContent = newData.last_updated || '--:--';
        
        // If still on loading screen (first load), transition to content
        const loadingEl = document.getElementById('loading');
        if (loadingEl && !loadingEl.classList.contains('hidden')) {
            loadingEl.classList.add('hidden');
            document.getElementById('main-content').classList.remove('hidden');
        }
        
        // Silently refresh visible components
        // Don't recreate chart — just update data (preserves zoom, drawings, minBarSpacing)
        if (!_tvChart) initChart();
        renderTickerButtons();
        renderMarketOverview();
        _refreshAllCompareCharts();
        
        if (Object.keys(marketData).length > 0 && !window._fs) {
            const sym = currentSymbol || Object.keys(marketData)[0];
            updateChart(sym);
        }
        
        renderFeed();
        
        // Hide stale notifications — but don't hide scanning banner if a scan is still running
        document.getElementById('newdata-banner')?.classList.add('hidden');
        if (!_isScanRunning && !_marketRefreshPending) {
            document.getElementById('status-notification')?.classList.add('hidden');
            document.getElementById('scanning-banner')?.classList.add('hidden');
        }
        
        lucide.createIcons();
        
        if (!skipToast && typeof showToast === 'function') showToast('Data updated', 'success', 2000);

    } catch (err) {
        console.warn('Background data refresh failed:', err.message);
    }
}

var _marketRefreshPending = false;

async function refreshMarket() {
    const btn = document.getElementById('refresh-market-btn');
    const icon = btn?.querySelector('svg');
    if (icon) icon.classList.add('animate-spin');
    if (btn) btn.disabled = true;
    _marketRefreshPending = true;

    // Show scanning banner for market refresh
    var notif = document.getElementById('status-notification');
    var banner = document.getElementById('scanning-banner');
    var scanTitle = document.getElementById('scan-title');
    var scanProg = document.getElementById('scan-progress');
    if (notif) notif.classList.remove('hidden');
    if (banner) banner.classList.remove('hidden');
    if (scanTitle) scanTitle.textContent = 'Refreshing market data...';
    if (scanProg) scanProg.textContent = 'Fetching quotes for all tickers...';
    // Gradually fill bar during market refresh (no intermediate SSE events)
    var _mktPct = 10;
    _updateScanBars(_mktPct);
    var _mktBarAnim = setInterval(function() {
        if (!_marketRefreshPending) { clearInterval(_mktBarAnim); return; }
        // Ease toward 90% — faster ramp so user sees movement immediately
        _mktPct += (90 - _mktPct) * 0.12;
        _updateScanBars(Math.round(_mktPct));
    }, 400);
    // Safety timeout: if SSE complete event is lost, clean up after 90s
    var _mktTimeout = setTimeout(function() {
        if (!_marketRefreshPending) return;
        clearInterval(_mktBarAnim);
        _marketRefreshPending = false;
        _updateScanBars(0);
        if (notif) notif.classList.add('hidden');
        if (banner) banner.classList.add('hidden');
        if (btn) { btn.disabled = false; var ic = btn.querySelector('svg'); if (ic) ic.classList.remove('animate-spin'); }
        if (typeof showToast === 'function') showToast('Market refresh timed out', 'warning');
    }, 90000);

    try {
        const r = await fetch('/api/refresh-market');
        if (!r.ok) throw new Error('Server error');
        // Don't load data or show toast here — SSE 'complete' event will
        // handle the actual data reload once the background fetch finishes
    } catch (err) {
        clearTimeout(_mktTimeout);
        clearInterval(_mktBarAnim);
        _marketRefreshPending = false;
        _updateScanBars(0);
        if (notif) notif.classList.add('hidden');
        if (banner) banner.classList.add('hidden');
        console.error('Market refresh failed:', err);
        if (typeof showToast === 'function') showToast('Market refresh failed', 'error');
        const iconAfter = btn?.querySelector('svg');
        if (iconAfter) iconAfter.classList.remove('animate-spin');
        if (btn) btn.disabled = false;
    }
}

// Keep refreshNews as alias for backward compatibility
function refreshNews() { toggleScan(); }

async function toggleScan() {
    if (_isScanRunning) {
        // Cancel the running scan
        _setScanButtonStopping();
        // Show warning if cancel takes too long (model may be mid-generation)
        window._cancelTimeoutId = setTimeout(() => {
            if (_isScanRunning) {
                if (typeof showToast === 'function')
                    showToast('Waiting for current article to finish scoring...', 'warning', 5000);
            }
        }, 15000);
        try {
            const cancelResp = await fetch('/api/scan/cancel', { method: 'POST', signal: AbortSignal.timeout(15000) });
            if (!cancelResp.ok) throw new Error('HTTP ' + cancelResp.status);
        } catch (err) {
            console.error('Cancel request failed:', err);
            if (typeof showToast === 'function') showToast('Cancel request failed', 'error');
            _setScanButtonScanning(); // revert to scanning state
            if (window._cancelTimeoutId) { clearTimeout(window._cancelTimeoutId); window._cancelTimeoutId = null; }
        }
        return;
    }

    // Start a new scan
    _isScanRunning = true;
    _setScanButtonScanning();

    // Detect if this is the first scan for a fresh profile
    const _cfg = typeof configData !== 'undefined' ? configData : null;
    const isFirstScan = newsData.length === 0 && !!_cfg?.profile?.role?.trim();
    if (isFirstScan) {
        window._firstScanPending = true;
    }

    try {
        const r = await fetch('/api/refresh-news');
        if (!r.ok) throw new Error('Server error');
        await new Promise(r => setTimeout(r, 1000));
        if (typeof showToast === 'function') showToast('News scan started', 'success');
        _startScanPolling();
    } catch (err) {
        console.error('News refresh failed:', err);
        if (typeof showToast === 'function') showToast('News refresh failed', 'error');
        window._firstScanPending = false;
        _isScanRunning = false;
        _setScanButtonIdle();
    }
}

function _setScanButtonIdle() {
    _isScanRunning = false;
    if (window._cancelTimeoutId) { clearTimeout(window._cancelTimeoutId); window._cancelTimeoutId = null; }
    const btn = document.getElementById('scan-btn');
    if (!btn) return;
    btn.className = 'p-1.5 rounded-lg transition-all group';
    btn.style.cssText = 'background:rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399;';
    btn.disabled = false;
    btn.title = 'Refresh news & intelligence (uses API)';
    // Replace icon with radar
    const iconEl = btn.querySelector('[data-lucide], svg');
    if (iconEl) {
        const newIcon = document.createElement('i');
        newIcon.setAttribute('data-lucide', 'radar');
        newIcon.className = 'w-3.5 h-3.5 group-hover:rotate-180 transition-all duration-300';
        newIcon.id = 'scan-btn-icon';
        iconEl.replaceWith(newIcon);
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }
}

function _setScanButtonScanning(progressText) {
    const btn = document.getElementById('scan-btn');
    if (!btn) return;
    btn.className = 'p-1.5 rounded-lg transition-all group';
    btn.style.cssText = 'background:rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #f87171;';
    btn.disabled = false;
    btn.title = progressText ? 'Stop scan (' + progressText + ')' : 'Stop the running scan';
    // Replace icon with square (stop icon)
    const iconEl = btn.querySelector('[data-lucide], svg');
    if (iconEl) {
        const newIcon = document.createElement('i');
        newIcon.setAttribute('data-lucide', 'square');
        newIcon.className = 'w-3.5 h-3.5';
        newIcon.id = 'scan-btn-icon';
        iconEl.replaceWith(newIcon);
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }
}

function _setScanButtonStopping() {
    const btn = document.getElementById('scan-btn');
    if (!btn) return;
    btn.className = 'p-1.5 rounded-lg transition-all cursor-not-allowed opacity-70';
    btn.style.cssText = 'background:rgba(100,116,139,0.15); border: 1px solid rgba(100,116,139,0.3); color: #94a3b8;';
    btn.disabled = true;
    btn.title = 'Cancelling scan...';
}

function _startScanPolling() {
    _stopScanPolling();
    _scanPollingInterval = setInterval(async () => {
        try {
            const r = await fetch('/api/scan/status', { signal: AbortSignal.timeout(5000) });
            if (!r.ok) return;
            const s = await r.json();
            if (!s.running) {
                _stopScanPolling();
                if (!_isScanRunning) return;
                _setScanButtonIdle();
                // SSE may have missed the complete event — load data as fallback
                console.log('[Polling] Scan finished, loading data as SSE fallback...');
                document.getElementById('scanning-banner')?.classList.add('hidden');
                document.getElementById('status-notification')?.classList.add('hidden');
                var _settingsBanner = document.getElementById('settings-scan-banner');
                if (_settingsBanner) _settingsBanner.classList.add('hidden');
                try { await loadNewData(); } catch(e) { console.warn('Polling data load failed:', e); }
                return;
            }
            // Update button progress text
            if (s.scored > 0 && s.total > 0) {
                _setScanButtonScanning(`Stop (${s.scored}/${s.total})`);
            }
        } catch (err) {
            // Ignore transient errors
        }
    }, 2000);
}

function _stopScanPolling() {
    if (_scanPollingInterval) {
        clearInterval(_scanPollingInterval);
        _scanPollingInterval = null;
    }
}


// === EXTRA FEEDS (Finance, Politics & Custom) ===
async function loadExtraFeeds(type) {
    try {
        const resp = await fetch(`/api/${type}-news?_=${Date.now()}`, {
            headers: { 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' }
        });
        if (!resp.ok) { console.warn(`Extra feeds (${type}): HTTP ${resp.status}`); return; }
        const result = await resp.json();
        if (type === 'finance') {
            financeNewsData = result.items || [];
        } else if (type === 'politics') {
            politicsNewsData = result.items || [];
        } else if (type === 'jobs') {
            jobsNewsData = result.items || [];
        } else if (type === 'custom') {
            customNewsData = result.items || [];
        }
    } catch (e) {
        console.error(`Failed to load ${type} feeds:`, e);
    }
}

async function loadExtraFeedsIfNeeded() {
    // Always (re)load when navigating to these tabs so source changes apply
    if (activeRoot === 'finance_news') {
        await loadExtraFeeds('finance');
        renderFeed();
    } else if (activeRoot === 'politics') {
        await loadExtraFeeds('politics');
        renderFeed();
    } else if (activeRoot === 'jobs_feeds') {
        await loadExtraFeeds('jobs');
        renderFeed();
    } else if (activeRoot === 'custom_feeds') {
        await loadExtraFeeds('custom');
        renderFeed();
    }
}

function showError(msg) {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('main-content').classList.remove('hidden');
    
    // Check if this is first run (no data yet)
    const isFirstRun = msg.includes('No data yet') || msg.includes('Run a scan');
    
    if (isFirstRun) {
        document.getElementById('news-feed').innerHTML = `
            <div class="glass-panel rounded-xl p-10 text-center">
                <div class="w-16 h-16 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin mx-auto mb-6"></div>
                <h3 class="text-lg font-semibold text-slate-200 mb-2">First scan in progress...</h3>
                <p class="text-slate-400 text-sm mb-4">Your intelligence dashboard is being prepared.</p>
                <div class="bg-slate-800/50 rounded-lg p-4 max-w-md mx-auto mb-4">
                    <div class="flex items-center gap-3 text-left text-xs text-slate-500">
                        <i data-lucide="cpu" class="w-4 h-4 text-blue-400 flex-shrink-0"></i>
                        <span>AI is analyzing and scoring 80+ news items for relevance to your profile</span>
                    </div>
                </div>
                <p class="text-[11px] text-slate-600">This takes 3-5 minutes on first run. Future loads will be instant.</p>
            </div>
        `;
        lucide.createIcons();
    } else {
        document.getElementById('news-feed').innerHTML = `
            <div class="p-8 text-center border border-dashed border-slate-700 rounded-xl">
                <p class="text-slate-400 mb-2">Unable to load data</p>
                <p class="text-slate-500 text-sm mb-4">${esc(msg)}</p>
                <button onclick="loadData()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm">Try Again</button>
            </div>
        `;
    }
}


// ═══════════════════════════════════════════════════════════
// ONBOARDING WIZARD (3-step interactive)
// ═══════════════════════════════════════════════════════════


/* ── First-time tab tooltips ── */
function _showTabTooltip(tabId) {
    var tips = {
        profile: 'This is where you define what StratOS tracks for you.',
        sources: 'Toggle RSS feeds and set your search provider here.',
        market: 'Manage your watchlist and save ticker presets.',
        system: 'Customize the look and manage your data.'
    };
    var tip = tips[tabId];
    if (!tip) return;
    var key = 'stratos_tab_tip_' + tabId;
    try { if (localStorage.getItem(key)) return; } catch(e) { return; }
    var page = document.getElementById('stab-' + tabId);
    if (!page) return;
    /* Remove any existing tooltip */
    var old = page.querySelector('.stab-tooltip');
    if (old) old.remove();
    var div = document.createElement('div');
    div.className = 'stab-tooltip';
    div.innerHTML = '<span>' + tip + '</span><button class="stab-tip-close" onclick="this.parentElement.remove()">&times;</button>';
    page.insertBefore(div, page.firstChild);
    try { localStorage.setItem(key, '1'); } catch(e) {}
}

function _pulseHelpIfNew() {
    const cfg = typeof configData !== 'undefined' ? configData : null;
    const btn = document.getElementById('help-btn');
    if (!cfg || !btn) return;

    const hasRole = cfg.profile?.role?.trim();
    const hasCats = cfg.dynamic_categories?.length > 0;
    
    if (!hasRole && !hasCats) {
        // Add pulsing ring effect
        btn.style.position = 'relative';
        btn.style.borderColor = 'var(--accent, #34d399)';
        btn.style.color = 'var(--accent, #34d399)';
        btn.style.animation = 'help-pulse 2s ease-in-out infinite';
        
        // Inject keyframes if not already present
        if (!document.getElementById('help-pulse-style')) {
            const style = document.createElement('style');
            style.id = 'help-pulse-style';
            style.textContent = `
                @keyframes help-pulse {
                    0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
                    50% { box-shadow: 0 0 0 6px rgba(52,211,153,0); }
                }
            `;
            document.head.appendChild(style);
        }
    }
}
