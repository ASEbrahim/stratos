// === NAVIGATION ===
let sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

// === SIDEBAR RESIZE ===
(function initSidebarResize() {
    document.addEventListener('DOMContentLoaded', () => {
        const handle = document.getElementById('sidebar-resize-handle');
        const sidebar = document.getElementById('sidebar');
        if (!handle || !sidebar) return;
        
        // Restore saved width
        const savedWidth = localStorage.getItem('sidebarWidth');
        if (savedWidth && !sidebarCollapsed) {
            sidebar.style.width = savedWidth + 'px';
            sidebar.classList.toggle('sidebar-narrow', parseInt(savedWidth) < 200);
        }
        
        let isResizing = false;
        let startX, startWidth;
        
        handle.addEventListener('mousedown', (e) => {
            if (sidebarCollapsed) return;
            isResizing = true;
            startX = e.clientX;
            startWidth = sidebar.getBoundingClientRect().width;
            sidebar.classList.add('resizing');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            handle.style.background = 'rgba(16,185,129,0.4)';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const dx = e.clientX - startX;
            const newWidth = Math.min(400, Math.max(160, startWidth + dx));
            sidebar.style.width = newWidth + 'px';
            // Hide bottom section (themes/profile) when sidebar is narrow
            sidebar.classList.toggle('sidebar-narrow', newWidth < 200);
        });
        
        document.addEventListener('mouseup', () => {
            if (!isResizing) return;
            isResizing = false;
            sidebar.classList.remove('resizing');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            handle.style.background = '';
            // Save width and narrow state
            const width = sidebar.getBoundingClientRect().width;
            localStorage.setItem('sidebarWidth', Math.round(width));
            sidebar.classList.toggle('sidebar-narrow', width < 200);
        });
    });
})();

function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed;
    localStorage.setItem('sidebarCollapsed', sidebarCollapsed);
    applySidebarState();
}

function applySidebarState() {
    const sidebar = document.getElementById('sidebar');
    const floatBtn = document.getElementById('sidebar-float-toggle');
    const handle = document.getElementById('sidebar-resize-handle');
    if (sidebarCollapsed) {
        sidebar.classList.add('sidebar-collapsed');
        sidebar.style.width = '';
        sidebar.style.minWidth = '';
        sidebar.style.overflow = '';
        sidebar.style.borderRight = '';
        sidebar.style.padding = '';
        if (handle) handle.style.display = 'none';
        if (floatBtn) {
            floatBtn.classList.remove('hidden', 'sf-hidden');
            floatBtn.classList.add('sf-visible');
            floatBtn.style.left = '0px';
        }
    } else {
        sidebar.classList.remove('sidebar-collapsed');
        const savedWidth = localStorage.getItem('sidebarWidth');
        sidebar.style.width = savedWidth ? savedWidth + 'px' : '';
        sidebar.style.minWidth = '160px';
        sidebar.style.overflow = '';
        sidebar.style.borderRight = '';
        sidebar.style.padding = '';
        sidebar.classList.toggle('sidebar-narrow', savedWidth && parseInt(savedWidth) < 200);
        if (handle) handle.style.display = '';
        if (floatBtn) {
            floatBtn.classList.remove('sf-visible');
            floatBtn.classList.add('sf-hidden');
        }
    }
    lucide.createIcons();
}

function getTabCount(id) {
    if (id === 'saved') return typeof getSavedSignals === 'function' ? getSavedSignals().length : 0;
    if (id === 'finance_news') return financeNewsData.length;
    if (id === 'politics') return politicsNewsData.length;
    if (id === 'custom_feeds') return customNewsData.length;
    if (!newsData || !newsData.length) return 0;
    const uniqueNews = deduplicateNews(newsData);
    if (id === 'dashboard') return uniqueNews.filter(d => (d.score || 0) >= 5.0).length;
    if (id === 'kuwait') return uniqueNews.filter(d => matchesKuwait(d)).length;
    if (id === 'banks') return uniqueNews.filter(d => matchesBanks(d)).length;
    if (id === 'regional') return uniqueNews.filter(d => matchesRegional(d)).length;
    if (id === 'global') return uniqueNews.filter(d => matchesMarket(d)).length;
    if (id === 'ai') return uniqueNews.filter(d => (d.root === 'ai' || d.root === 'global' || d.category === 'tech') && !matchesKuwait(d) && !matchesBanks(d) && !matchesRegional(d) && !matchesMarket(d) && !matchesAnyDynamicCategory(d)).length;
    // Dynamic category tabs (word-boundary safe matching)
    const navItem = (window.navItems || []).find(n => n.id === id);
    if (navItem && navItem.dynamic && navItem.keywords && navItem.keywords.length > 0) {
        const kwLower = navItem.keywords.map(k => k.toLowerCase());
        return uniqueNews.filter(d => {
            const text = ((d.title || '') + ' ' + (d.summary || '') + ' ' + (d.category || '')).toLowerCase();
            return kwLower.some(kw => matchesKeyword(text, kw)) || d.category === id;
        }).length;
    }
    return 0;
}

function renderNav() {
    let html = '';
    navSections.forEach((section, si) => {
        // Categories (navIntelCollapsed) default to collapsed for new users
        const collapseDefault = section.storageKey === 'navIntelCollapsed';
        const stored = localStorage.getItem(section.storageKey);
        const isCollapsed = section.collapsible && (stored === 'true' || (stored === null && collapseDefault));
        const hasActiveItem = section.items.some(item => item.id === activeRoot);
        
        if (section.label) {
            html += `<div class="nav-section-header flex items-center justify-between px-2 py-1.5 mb-1 rounded hover:bg-slate-800/30 ${isCollapsed ? 'collapsed' : ''}" 
                onclick="toggleNavSection('${section.storageKey}')">
                <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider sidebar-label">${section.label}</span>
                <i data-lucide="chevron-down" class="w-3 h-3 text-slate-600 chevron-icon sidebar-label"></i>
            </div>`;
        }
        
        // Calculate max-height for animation (enough for items)
        const maxH = section.items.length * 48;
        const collapsedClass = isCollapsed ? 'collapsed' : '';
        
        // Intelligence section gets a scrollable container when it has many items
        const isIntel = section.storageKey === 'navIntelCollapsed';
        const scrollStyle = isIntel ? 'max-height: min(' + maxH + 'px, calc(100vh - 380px)); overflow-y: auto;' : 'max-height: ' + maxH + 'px;';
        
        html += `<div class="nav-section-items ${section.collapsible ? collapsedClass : ''} ${isIntel ? 'intel-scroll' : ''}" style="${scrollStyle}" data-section="${section.storageKey || ''}">`;
        
        section.items.forEach(item => {
            const count = getTabCount(item.id);
            const badge = (count > 0 && item.id !== 'settings') 
                ? `<span class="nav-badge ml-auto ${activeRoot === item.id ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'}">${count}</span>` 
                : '';
            html += `<button onclick="setActive('${item.id}')" title="${item.label}"
                class="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium rounded-lg transition-all ${activeRoot === item.id ? 'active-nav' : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'}">
                <i data-lucide="${item.icon}" class="w-4 h-4 flex-shrink-0"></i> <span class="truncate">${item.label}</span>${badge}
            </button>`;
        });
        
        html += '</div>';
        
        // Separator between sections (not after last)
        if (si < navSections.length - 1) {
            html += '<div class="border-t border-slate-800/50 my-2"></div>';
        }
    });
    
    document.getElementById('nav-menu').innerHTML = html;
    lucide.createIcons();
    applySidebarState();
}

function toggleNavSection(storageKey) {
    if (!storageKey) return;
    const isCollapsed = localStorage.getItem(storageKey) === 'true';
    const willCollapse = !isCollapsed;
    localStorage.setItem(storageKey, willCollapse);
    
    // If collapsing a section that contains the active tab, switch to Summary
    if (willCollapse) {
        const section = navSections.find(s => s.storageKey === storageKey);
        if (section && section.items.some(item => item.id === activeRoot)) {
            activeRoot = 'dashboard';
            const nav = (window.navItems || navItems).find(n => n.id === 'dashboard');
            if (nav) {
                document.getElementById('page-title').textContent = nav.label;
                document.getElementById('page-subtitle').textContent = nav.subtitle;
            }
            document.getElementById('main-content').classList.remove('hidden');
            document.getElementById('settings-panel').classList.add('hidden');
            renderFeed();
        }
    }
    
    renderNav();
}

function setActive(id) {
    activeRoot = id;
    const nav = (window.navItems || navItems).find(n => n.id === id);
    if (nav) {
        document.getElementById('page-title').textContent = nav.label;
        document.getElementById('page-subtitle').textContent = nav.subtitle;
    }
    
    // Clear search when switching tabs
    const searchBox = document.getElementById('feed-search');
    if (searchBox) searchBox.value = '';
    
    // Hide score filters on Finance/Politics/Custom (RSS-only, no scoring)
    const scoreFilters = document.getElementById('score-filters');
    const isExtraFeed = (id === 'finance_news' || id === 'politics' || id === 'jobs_feeds' || id === 'custom_feeds');
    if (scoreFilters) scoreFilters.classList.toggle('hidden', isExtraFeed);
    if (searchBox) {
        searchBox.placeholder = isExtraFeed
            ? 'Search headlines...'
            : 'Filter signals... (e.g. KOC, Warba, NVDA)';
    }
    
    renderNav();
    
    // Restore from old expanded markets if needed
    if (_marketsExpanded) toggleMarketsExpanded(false);
    
    // Show/hide panels based on active tab
    const mainContent = document.getElementById('main-content');
    const settingsPanel = document.getElementById('settings-panel');
    const marketsPanel = document.getElementById('markets-panel');
    const sibylPanel = document.getElementById('sibyl-panel');

    // Hide sibyl panel by default (shown only on sibyl_hue tab)
    if (sibylPanel && id !== 'sibyl_hue') {
        sibylPanel.classList.add('hidden');
        if (typeof hideSibylPanel === 'function') hideSibylPanel();
    }

    if (id === 'sibyl_hue') {
        // Sibyl Intelligence Hue panel
        mainContent.classList.add('hidden');
        settingsPanel.classList.add('hidden');
        if (marketsPanel) marketsPanel.classList.add('hidden');
        const ytPanelSb = document.getElementById('youtube-kb-panel');
        if (ytPanelSb) ytPanelSb.classList.add('hidden');
        if (sibylPanel) {
            sibylPanel.classList.remove('hidden');
            if (typeof initSibylPanel === 'function') initSibylPanel();
        }
    } else if (id === 'markets_view') {
        // Dedicated markets panel
        mainContent.classList.add('hidden');
        settingsPanel.classList.add('hidden');
        if (marketsPanel) { marketsPanel.classList.remove('hidden'); initMarketsPanel(); }
        const ytPanelM = document.getElementById('youtube-kb-panel');
        if (ytPanelM) ytPanelM.classList.add('hidden');
    } else if (id === 'youtube_kb') {
        mainContent.classList.add('hidden');
        settingsPanel.classList.add('hidden');
        if (marketsPanel) marketsPanel.classList.add('hidden');
        const ytPanel = document.getElementById('youtube-kb-panel');
        if (ytPanel) { ytPanel.classList.remove('hidden'); if (typeof initYouTubeKB === 'function') initYouTubeKB(); }
    } else if (id === 'image_gen') {
        // Open image gen as modal overlay (doesn't replace main content)
        if (typeof toggleImageGenPanel === 'function') toggleImageGenPanel();
        // Reset active to previous tab so nav doesn't stay on image_gen
        return;
    } else if (id === 'settings') {
        mainContent.classList.add('hidden');
        if (marketsPanel) marketsPanel.classList.add('hidden');
        const ytPanelS = document.getElementById('youtube-kb-panel');
        if (ytPanelS) ytPanelS.classList.add('hidden');
        settingsPanel.classList.remove('hidden');
        // Only hide the main notification if no scan/refresh is running
        if (!window._isScanRunning && !window._marketRefreshPending) {
            document.getElementById('status-notification').classList.add('hidden');
        }
        if (!window._settingsDirty) { loadConfig(); }
        loadPresets();
        if (typeof _initProfileSettings === 'function') _initProfileSettings();
    } else {
        mainContent.classList.remove('hidden');
        settingsPanel.classList.add('hidden');
        if (marketsPanel) marketsPanel.classList.add('hidden');
        const ytPanel = document.getElementById('youtube-kb-panel');
        if (ytPanel) ytPanel.classList.add('hidden');
        renderFeed();
        if (id === 'finance_news' || id === 'politics' || id === 'jobs_feeds' || id === 'custom_feeds') {
            loadExtraFeedsIfNeeded();
        }
    }
    if (typeof renderAlerts === 'function') renderAlerts();
    lucide.createIcons();
}

