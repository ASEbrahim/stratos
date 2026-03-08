// === CONFIG MANAGEMENT ===
let configData = null;
window._settingsDirty = false;

// Track unsaved changes — any input modification marks settings as dirty
function initSettingsDirtyTracking() {
    const panel = document.getElementById('settings-panel');
    if (!panel) return;
    
    // Listen for any input/change events in the settings panel
    panel.addEventListener('input', () => { window._settingsDirty = true; });
    panel.addEventListener('change', () => { window._settingsDirty = true; });
}

// Call once on page load (deferred to ensure DOM exists)
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initSettingsDirtyTracking, 500);
});

async function loadConfig() {
    try {
        const response = await fetch('/api/config', {signal: AbortSignal.timeout(10000)});
        if (!response.ok) { console.error('/api/config GET failed:', response.status); return; }
        configData = await response.json();

        // Populate form fields
        // Search provider
        const provider = configData.search?.provider || 'duckduckgo';
        document.getElementById('cfg-search-provider').value = provider;
        document.getElementById('cfg-serper-api-key').value = configData.search?.serper_api_key || '';
        // Re-lock API key field
        const apiKeyInput = document.getElementById('cfg-serper-api-key');
        if (apiKeyInput) { apiKeyInput.readOnly = true; apiKeyInput.type = 'password'; }
        _serperKeyUnlocked = false;
        toggleSearchProviderSettings();

        // Load search status (will also populate credits field)
        loadSearchStatus();

        // Timelimit
        document.getElementById('cfg-timelimit').value = configData.news?.timelimit || 'w';

        // Tickers — populate shared bubble system
        const tickerSymbols = (configData.market?.tickers || []).map(t => t.symbol);
        if (typeof simpleTickers !== 'undefined') {
            simpleTickers.length = 0;
            tickerSymbols.forEach(t => simpleTickers.push(t));
            if (typeof renderSimpleTickers === 'function') renderSimpleTickers();
        }

        // Career keywords/queries are now managed via dynamic categories in Simple mode

        // Finance, Tech, RSS — now handled by Simple mode dynamic categories & News Sources UI
        // (fields removed from Advanced; data preserved in backend config)

        // Profile
        document.getElementById('cfg-profile-role').value = configData.profile?.role || '';
        document.getElementById('cfg-profile-location').value = configData.profile?.location || '';
        
        // Build context string from rich profile if no simple context field exists
        let context = configData.profile?.context || '';
        if (!context && configData.profile?.goals) {
            const parts = [];
            const p = configData.profile;
            if (p.nationality) parts.push(`${p.nationality} national`);
            if (p.graduation) parts.push(`graduating ${p.graduation}`);
            if (p.experience_level) parts.push(`experience: ${p.experience_level}`);
            if (p.scholarship) parts.push(`scholarship: ${p.scholarship}`);
            if (p.goals?.primary) parts.push(`Goal: ${p.goals.primary}`);
            if (p.goals?.financial) parts.push(`Financial: ${p.goals.financial}`);
            if (p.goals?.investment) parts.push(`Investment: ${p.goals.investment}`);
            context = parts.join('\n');
        }
        document.getElementById('cfg-profile-context').value = context;

        lucide.createIcons();
        
        // Initialize settings mode (simple or advanced)
        setSettingsMode(simpleSettingsMode);
        
        // Rebuild nav tabs from dynamic categories if present
        if (typeof rebuildNavFromConfig === 'function') {
            rebuildNavFromConfig();
        }
    } catch (err) {
        console.error('Failed to load config:', err);
    }
    // Fresh data from server = not dirty
    window._settingsDirty = false;
    // Load feed source catalog
    loadSourceCatalog();
}

function toggleSearchProviderSettings() {
    const provider = document.getElementById('cfg-search-provider').value;
    const serperSettings = document.getElementById('serper-api-settings');

    // Hide all first
    serperSettings.classList.add('hidden');

    // Show relevant settings
    if (provider === 'serper') {
        serperSettings.classList.remove('hidden');
    }
}

// Add event listener for provider change
document.getElementById('cfg-search-provider').addEventListener('change', toggleSearchProviderSettings);

// === SERPER API KEY LOCK ===
let _serperKeyUnlocked = false;

function toggleSearchProvider() {
    const body = document.getElementById('search-provider-body');
    const chevron = document.getElementById('search-provider-chevron');
    if (!body) return;
    const hidden = body.classList.toggle('hidden');
    if (chevron) chevron.style.transform = hidden ? '' : 'rotate(180deg)';
}

function toggleSerperKeyLock() {
    const input = document.getElementById('cfg-serper-api-key');
    const lockBtn = document.getElementById('serper-key-lock-btn');
    if (!input || !lockBtn) return;

    var saveBtn = document.getElementById('serper-key-save-btn');
    if (_serperKeyUnlocked) {
        // Re-lock
        input.readOnly = true;
        input.type = 'password';
        _serperKeyUnlocked = false;
        lockBtn.title = 'Click to unlock';
        lockBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>';
        lockBtn.classList.remove('text-emerald-400/60');
        lockBtn.classList.add('text-amber-400/60');
        if (saveBtn) saveBtn.classList.add('hidden');
        return;
    }

    // Direct unlock — no PIN gate (key is already masked server-side)
    input.readOnly = false;
    input.type = 'text';
    _serperKeyUnlocked = true;
    lockBtn.title = 'Click to re-lock';
    lockBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 017 0"/></svg>';
    lockBtn.classList.remove('text-amber-400/60');
    lockBtn.classList.add('text-emerald-400/60');
    if (saveBtn) saveBtn.classList.remove('hidden');
    // Clear the masked value so the user starts fresh
    input.value = '';
    input.focus();
}

async function saveSerperKey() {
    const input = document.getElementById('cfg-serper-api-key');
    if (!input) return;
    const key = input.value.trim();
    if (!key || key.includes('\u2022')) {
        if (typeof showToast === 'function') showToast('Enter a valid API key first', 'error');
        return;
    }
    try {
        const headers = { 'Content-Type': 'application/json' };
        if (typeof getAuthToken === 'function') headers['X-Auth-Token'] = getAuthToken();

        const resp = await fetch('/api/save-serper-key', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ serper_api_key: key })
        });
        const result = await resp.json();
        if (result.status === 'saved') {
            if (typeof showToast === 'function') showToast('API key saved and activated', 'success');
            // Re-lock the field
            toggleSerperKeyLock();
        } else {
            if (typeof showToast === 'function') showToast(result.error || 'Save failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Save failed: ' + e.message, 'error');
    }
}

function clearAllInputs() {
    // Clear all text inputs and textareas in Settings (preserve API key)
    // cfg-serper-api-key is intentionally NOT cleared — it's PIN-protected
    document.getElementById('cfg-serper-credits').value = '';
    document.getElementById('cfg-profile-role').value = '';
    document.getElementById('cfg-profile-location').value = '';
    document.getElementById('cfg-profile-context').value = '';

    // Clear Simple mode fields
    const simpleRole = document.getElementById('simple-role');
    const simpleLoc = document.getElementById('simple-location');
    const simpleCtx = document.getElementById('simple-context');
    if (simpleRole) simpleRole.value = '';
    if (simpleLoc) simpleLoc.value = '';
    if (simpleCtx) simpleCtx.value = '';

    // Clear dynamic categories
    if (typeof simpleCategories !== 'undefined') {
        simpleCategories.length = 0;
        if (typeof renderDynamicCategories === 'function') renderDynamicCategories();
    }

    // Clear shared tickers
    if (typeof simpleTickers !== 'undefined') {
        simpleTickers.length = 0;
        if (typeof renderSimpleTickers === 'function') renderSimpleTickers();
    }

    // Reset dropdowns to defaults
    document.getElementById('cfg-search-provider').value = 'serper';
    document.getElementById('cfg-timelimit').value = 'w';
    toggleSearchProviderSettings();

    // Re-lock API key if it was unlocked
    if (_serperKeyUnlocked) toggleSerperKeyLock();

    // Show status
    const status = document.getElementById('save-status');
    status.textContent = 'All fields cleared. Click Save to apply.';
    status.className = 'text-sm text-center text-yellow-400';
    setTimeout(() => { status.textContent = ''; }, 3000);
}

async function syncSerperCredits() {
    const creditsInput = document.getElementById('cfg-serper-credits');
    const credits = parseInt(creditsInput.value);

    if (isNaN(credits) || credits < 0) {
        if (typeof showToast === 'function') showToast('Please enter a valid credit number', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/serper-credits', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ credits: credits })
        });

        const result = await response.json();
        if (result.status === 'updated') {
            loadSearchStatus();
            if (typeof showToast === 'function') showToast(`Credits synced! Remaining: ${credits}`, 'success');
        }
    } catch (err) {
        console.error('Failed to sync credits:', err);
        if (typeof showToast === 'function') showToast('Failed to sync credits', 'error');
    }
}

async function loadSearchStatus() {
    try {
        const response = await fetch('/api/search-status');
        const status = await response.json();
        const banner = document.getElementById('search-status-banner');

        if (status.provider === 'serper') {
            banner.classList.remove('hidden');

            // Update credits input field with current remaining
            const creditsInput = document.getElementById('cfg-serper-credits');
            if (creditsInput && status.remaining !== 'unknown') {
                creditsInput.value = status.remaining;
            }

            if (status.limit_reached) {
                banner.className = 'mb-4 p-3 rounded-lg text-sm bg-red-500/20 border border-red-500/50 text-red-300';
                banner.innerHTML = `<strong>Serper limit reached!</strong> <a href="https://serper.dev/api-key" target="_blank" class="underline hover:text-red-200">Get more credits</a>`;
            } else if (status.warning) {
                banner.className = 'mb-4 p-3 rounded-lg text-sm bg-yellow-500/20 border border-yellow-500/50 text-yellow-300';
                banner.innerHTML = `<strong>Serper.dev:</strong> ${status.remaining} queries remaining (${status.used} used)`;
            } else if (status.used === 0) {
                // Not synced yet - prompt user to sync
                banner.className = 'mb-4 p-3 rounded-lg text-sm bg-blue-500/20 border border-blue-500/50 text-blue-300';
                banner.innerHTML = `<strong>Serper.dev:</strong> Active — Enter your current credits below and click Sync`;
            } else {
                banner.className = 'mb-4 p-3 rounded-lg text-sm bg-emerald-500/20 border border-emerald-500/50 text-emerald-300';
                banner.innerHTML = `<strong>Serper.dev:</strong> ${status.remaining} queries remaining (${status.used} used)`;
            }
        } else {
            banner.classList.add('hidden');
        }
    } catch (err) {
        console.error('Failed to load search status:', err);
    }
}

async function saveConfig() {
    const btn = document.getElementById('save-config-btn');
    const status = document.getElementById('save-status');
    btn.disabled = true;
    btn.innerHTML = '<div class="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Saving...';
    
    try {
        // Helper to split by both comma and newline
        const splitValues = (str) => str.split(/[,\n]/).map(s => s.trim()).filter(s => s);

        // Helper to parse RSS feeds - supports both formats:
        // 1. url | name | root | category  (pipe-delimited)
        // 2. Just a URL (auto-detect name/root/category)
        const parseRssFeeds = (str) => {
            return str.split('\n')
                .map(line => line.trim())
                .filter(line => line && line.startsWith('http'))
                .map(line => {
                    if (line.includes('|')) {
                        // Pipe-delimited format
                        const parts = line.split('|').map(p => p.trim());
                        return {
                            url: parts[0] || '',
                            name: parts[1] || 'Unknown',
                            root: parts[2] || 'global',
                            category: parts[3] || 'general'
                        };
                    } else {
                        // Plain URL — auto-detect from domain
                        const url = line.trim();
                        let name = 'Unknown', root = 'global', category = 'general';
                        try {
                            const domain = new URL(url).hostname.replace('www.', '').replace('feeds.', '');
                            name = domain.split('.')[0].charAt(0).toUpperCase() + domain.split('.')[0].slice(1);
                            // Auto-detect root and category
                            const lc = url.toLowerCase();
                            if (lc.includes('kuwait') || lc.includes('kuna')) { root = 'kuwait'; }
                            else if (lc.includes('arab') || lc.includes('zawya') || lc.includes('national')) { root = 'regional'; category = 'business'; }
                            else if (lc.includes('verge') && lc.includes('ai')) { root = 'ai'; category = 'tech'; }
                            if (lc.includes('tech') || lc.includes('ieee') || lc.includes('ars') || lc.includes('verge') || lc.includes('hacker') || lc.includes('hnrss') || lc.includes('hardware')) { category = 'tech'; }
                        } catch(e) {}
                        return { url, name, root, category };
                    }
                })
                .filter(feed => feed.url.startsWith('http'));
        };
        
        // Build config object (tickers are NOT included here — they are saved
        // separately via saveTickers() to prevent accidental overwrite with
        // an empty array when saving unrelated settings like profile/categories)
        const newConfig = {
            profile: {
                role: document.getElementById('cfg-profile-role')?.value || '',
                location: document.getElementById('cfg-profile-location')?.value || '',
                context: document.getElementById('cfg-profile-context')?.value || '',
                interests: []
            },
            news: {
                timelimit: document.getElementById('cfg-timelimit')?.value || 'w',
                career: {
                    root: (document.getElementById('cfg-profile-location')?.value || 'global').toLowerCase().trim(),
                    keywords: [],
                    queries: []
                },
                finance: { root: 'kuwait', keywords: [], queries: [] },
                tech_trends: { root: 'global', keywords: [], queries: [] },
                regional: { root: 'regional', keywords: [], queries: [] },
            },
            search: {
                provider: document.getElementById('cfg-search-provider')?.value || 'duckduckgo',
                serper_credits: parseInt(document.getElementById('cfg-serper-credits')?.value) || null
            }
        };

        // Include dynamic categories — always sync from simpleCategories
        if (window._pendingDynamicCategories) {
            newConfig.dynamic_categories = window._pendingDynamicCategories;
        } else if (typeof simpleCategories !== 'undefined' && simpleCategories.length > 0) {
            newConfig.dynamic_categories = simpleCategories;
        } else if (configData?.dynamic_categories?.length > 0) {
            // Preserve existing dynamic categories when no local state
            newConfig.dynamic_categories = configData.dynamic_categories;
        }
        
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify(newConfig)
        });
        if (!response.ok) throw new Error('Server error: ' + response.status);
        const result = await response.json();

        if (result.status === 'saved') {
            status.textContent = '✓ Saved!';
            status.className = 'text-sm text-emerald-400 self-center ml-2';
            window._settingsDirty = false; // Clear dirty state
            if (typeof showToast === 'function') showToast('Settings saved', 'success');
            
            // Rebuild nav immediately from saved config
            try {
                const cfgResp = await fetch('/api/config');
                if (cfgResp.ok) {
                    configData = await cfgResp.json();
                    if (typeof rebuildNavFromConfig === 'function') {
                        rebuildNavFromConfig();
                    }
                }
            } catch(e) { console.warn('Nav rebuild after save failed:', e); }
        } else {
            status.textContent = '✗ Error: ' + (result.error || 'Unknown');
            status.className = 'text-sm text-red-400 self-center ml-2';
            if (typeof showToast === 'function') showToast('Save failed: ' + (result.error || 'Unknown'), 'error');
        }
    } catch (err) {
        status.textContent = '✗ Failed to save';
        status.className = 'text-sm text-red-400 self-center ml-2';
        if (typeof showToast === 'function') showToast('Save failed — check connection', 'error');
    }
    
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="save" class="w-4 h-4"></i> Save & Apply';
    lucide.createIcons();
    
    setTimeout(() => { status.textContent = ''; }, 3000);
}

// === PROFILE PRESETS ===
async function loadPresets() {
    try {
        const response = await fetch('/api/profiles', {signal: AbortSignal.timeout(10000)});
        if (!response.ok) { console.error('/api/profiles GET failed:', response.status); return; }
        const data = await response.json();
        const container = document.getElementById('presets-list');
        
        if (!data.presets || data.presets.length === 0) {
            container.innerHTML = '<p class="text-xs text-slate-500 italic">No saved presets yet. Save your current settings above.</p>';
            return;
        }
        
        container.innerHTML = data.presets.map(p => `
            <div class="flex items-center gap-2 p-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-slate-200 truncate">${p.has_pin ? '<svg class="inline w-3 h-3 text-slate-500 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>' : ''}${_escHtml(p.name)}</p>
                    <p class="text-xs text-slate-500 truncate">${_escHtml(p.role || 'No role')} · ${_escHtml(p.location || 'No location')}</p>
                </div>
                <button onclick="loadPreset(decodeURIComponent('${encodeURIComponent(p.name)}'))" class="px-2.5 py-1.5 bg-blue-700/50 hover:bg-blue-600/50 border border-blue-600/30 rounded text-xs text-blue-300 transition-all" title="Load this preset">
                    Load
                </button>
                <button onclick="deletePreset(decodeURIComponent('${encodeURIComponent(p.name)}'))" class="px-2 py-1.5 bg-slate-700/50 hover:bg-red-900/50 border border-slate-600/30 hover:border-red-500/30 rounded text-xs text-slate-400 hover:text-red-400 transition-all" title="Delete this preset">
                    ✕
                </button>
            </div>
        `).join('');
    } catch(e) {
        console.error('Failed to load presets:', e);
    }
}

async function savePreset() {
    const name = document.getElementById('preset-name-input').value.trim();
    if (!name) { if (typeof showToast === 'function') showToast('Enter a preset name', 'warning'); return; }
    
    // Save current settings first
    await saveConfig();
    
    try {
        const response = await fetch('/api/profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({ action: 'save', name })
        });
        if (!response.ok) throw new Error('HTTP ' + response.status);
        const result = await response.json();
        if (result.status === 'saved') {
            document.getElementById('preset-name-input').value = '';
            loadPresets();
            const status = document.getElementById('save-status');
            status.textContent = `✓ Profile "${name}" saved!`;
            status.className = 'text-sm text-center text-emerald-400';
            setTimeout(() => { status.textContent = ''; }, 3000);
            if (typeof showToast === 'function') showToast(`Preset "${name}" saved`, 'success');
        }
    } catch(e) { if (typeof showToast === 'function') showToast('Failed to save preset', 'error'); }
}

async function loadPreset(name) {
    if (!confirm(`Load preset "${name}"? This will replace your current settings.`)) return;
    try {
        const response = await fetch('/api/profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({ action: 'load', name })
        });
        if (!response.ok) throw new Error('HTTP ' + response.status);
        const result = await response.json();
        if (result.status === 'loaded') {
            await loadConfig();
            const status = document.getElementById('save-status');
            status.textContent = `✓ Preset "${name}" loaded! Click Refresh to apply.`;
            status.className = 'text-sm text-center text-blue-400';
            setTimeout(() => { status.textContent = ''; }, 5000);
            if (typeof showToast === 'function') showToast(`Loaded "${name}"`, 'success');
        }
    } catch(e) { if (typeof showToast === 'function') showToast('Failed to load preset', 'error'); }
}

async function deletePreset(name) {
    if (!confirm(`Delete preset "${name}"?`)) return;
    try {
        const delResp = await fetch('/api/profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({ action: 'delete', name })
        });
        if (!delResp.ok) throw new Error('HTTP ' + delResp.status);
        loadPresets();
        if (typeof showToast === 'function') showToast(`Deleted "${name}"`, 'info');
    } catch(e) { if (typeof showToast === 'function') showToast('Failed to delete preset', 'error'); }
}

// === SIMPLE SETTINGS MODE ===
let simpleSettingsMode = localStorage.getItem('settingsMode') || 'simple';
let simpleCategories = []; // AI-generated or loaded from config

// Normalize category format: ensure {id, label, items} (fix old {name, keywords} entries)
function normalizeCategories(cats) {
    if (!Array.isArray(cats)) return [];
    return cats.map(cat => ({
        ...cat,
        id: cat.id || (cat.label || cat.name || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'cat_' + Math.random().toString(36).slice(2,6),
        label: cat.label || cat.name || 'Untitled',
        items: cat.items || cat.keywords || [],
        icon: cat.icon || 'tag',
    }));
}
let simpleTickers = [];
let simpleTimelimit = 'w';
let simpleContext = '';
let isGenerating = false;

// Icons map for category rendering
const CATEGORY_ICONS = {
    'briefcase': 'briefcase', 'landmark': 'landmark', 'cpu': 'cpu',
    'bar-chart-3': 'bar-chart-3', 'globe': 'globe', 'heart': 'heart',
    'book-open': 'book-open', 'shield': 'shield', 'zap': 'zap',
    'building-2': 'building-2', 'flask-conical': 'flask-conical',
    'truck': 'truck', 'stethoscope': 'stethoscope', 'gavel': 'gavel',
    'palette': 'palette', 'target': 'target',
};

function setSettingsMode(mode) {
    simpleSettingsMode = mode;
    localStorage.setItem('settingsMode', mode);
    const simplePanel = document.getElementById('settings-simple');
    const advPanel = document.getElementById('settings-advanced');
    const simpleBtn = document.getElementById('settings-mode-simple');
    const advBtn = document.getElementById('settings-mode-advanced');
    
    if (mode === 'simple') {
        simplePanel.classList.remove('hidden');
        advPanel.classList.add('hidden');
        simpleBtn.style.background = 'var(--accent)';
        simpleBtn.style.color = 'white';
        advBtn.style.background = 'transparent';
        advBtn.style.color = '';
        advBtn.className = advBtn.className.replace('text-white', 'text-slate-400');
        // Show quick-save and clear buttons (they work in Simple mode)
        const qsBtn = document.getElementById('quick-save-btn');
        const qsStatus = document.getElementById('quick-save-status');
        const clrBtn = document.getElementById('clear-simple-btn');
        if (qsBtn) qsBtn.classList.remove('hidden');
        if (qsStatus) qsStatus.classList.remove('hidden');
        if (clrBtn) clrBtn.classList.remove('hidden');
        populateSimpleSettings();
        // Restore tab-based visibility for shared sections
        switchSettingsTab(_currentSettingsTab);
    } else {
        simplePanel.classList.add('hidden');
        advPanel.classList.remove('hidden');
        advBtn.style.background = 'var(--accent)';
        advBtn.style.color = 'white';
        simpleBtn.style.background = 'transparent';
        simpleBtn.style.color = '';
        // Hide quick-save and clear buttons (Advanced has its own Save card)
        const qsBtn = document.getElementById('quick-save-btn');
        const qsStatus = document.getElementById('quick-save-status');
        const clrBtn = document.getElementById('clear-simple-btn');
        if (qsBtn) qsBtn.classList.remove('hidden');
        if (qsStatus) qsStatus.classList.remove('hidden');
        if (clrBtn) clrBtn.classList.remove('hidden');

        // In advanced mode, show shared sections (search provider, news sources, market tickers)
        var sp = document.getElementById('search-provider-panel');
        if (sp) sp.style.display = '';
        var ss = document.getElementById('settings-sources');
        if (ss) {
            ss.style.display = '';
            var kids = ss.children;
            for (var k = 0; k < kids.length; k++) kids[k].style.display = '';
        }

        // Show/hide managed-by-Simple banner on legacy keyword fields
        updateAdvancedManagedState();
    }
    lucide.createIcons();
}

function updateAdvancedManagedState() {
    // When Simple mode has categories, show a banner on Advanced tab
    // and disable the legacy keyword fields that would conflict
    const hasDynamic = simpleCategories.length > 0 || (configData?.dynamic_categories?.length > 0);
    
    // Remove old banner if present
    const oldBanner = document.getElementById('adv-managed-banner');
    if (oldBanner) oldBanner.remove();
    
    const legacyFields = [];
    
    if (hasDynamic) {
        // Insert a useful config summary at top of Advanced panel
        const advPanel = document.getElementById('settings-advanced');
        if (advPanel) {
            const cats = configData?.dynamic_categories || simpleCategories || [];
            const enabledCount = cats.filter(c => c.enabled !== false).length;
            const totalItems = cats.reduce((s,c) => s + (c.items||[]).length, 0);
            const tickers = configData?.market?.tickers || [];
            const role = configData?.profile?.role || 'Not set';
            const loc = configData?.profile?.location || '';
            const timeRange = configData?.timelimit === 'd' ? 'Daily' : configData?.timelimit === 'm' ? 'Monthly' : 'Weekly';
            
            const banner = document.createElement('div');
            banner.id = 'adv-managed-banner';
            banner.className = 'mb-5 p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5';
            banner.innerHTML = `
                <div class="flex items-start gap-3">
                    <i data-lucide="shield-check" class="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5"></i>
                    <div class="flex-1 min-w-0">
                        <p class="text-xs font-semibold text-emerald-300 mb-2">Configuration Overview</p>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <div>
                                <div class="text-lg font-bold text-white">${enabledCount}<span class="text-xs text-slate-500 font-normal">/${cats.length}</span></div>
                                <div class="text-[9px] text-slate-500 uppercase tracking-wider">Categories</div>
                            </div>
                            <div>
                                <div class="text-lg font-bold text-white">${totalItems}</div>
                                <div class="text-[9px] text-slate-500 uppercase tracking-wider">Keywords</div>
                            </div>
                            <div>
                                <div class="text-lg font-bold text-white">${tickers.length}</div>
                                <div class="text-[9px] text-slate-500 uppercase tracking-wider">Tickers</div>
                            </div>
                            <div>
                                <div class="text-lg font-bold text-white">${timeRange}</div>
                                <div class="text-[9px] text-slate-500 uppercase tracking-wider">Time Range</div>
                            </div>
                        </div>
                        <div class="mt-2 text-[10px] text-slate-500">${role}${loc ? ' · ' + loc : ''} · Categories sync with <button onclick="setSettingsMode('simple')" class="text-emerald-400 underline hover:text-emerald-300">Simple tab</button></div>
                    </div>
                </div>`;
            advPanel.insertBefore(banner, advPanel.firstChild);
            lucide.createIcons();
        }
        
        // Disable legacy keyword fields
        legacyFields.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.disabled = true;
                el.style.opacity = '0.4';
                el.style.cursor = 'not-allowed';
            }
        });
    } else {
        // Enable legacy fields
        legacyFields.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.disabled = false;
                el.style.opacity = '';
                el.style.cursor = '';
            }
        });
    }
}

function populateSimpleSettings() {
    if (!configData) return;
    
    // Check if there's a saved working state in localStorage (includes role/location)
    const savedRaw = localStorage.getItem('simpleCategories');
    let savedState = null;
    if (savedRaw) {
        try { savedState = JSON.parse(savedRaw); } catch(e) {}
    }
    
    // Profile fields: localStorage state takes priority over server config
    // This ensures clearing persists across tab switches
    if (savedState && 'role' in savedState) {
        document.getElementById('simple-role').value = savedState.role || '';
        document.getElementById('simple-location').value = savedState.location || '';
    } else {
        document.getElementById('simple-role').value = configData.profile?.role || '';
        document.getElementById('simple-location').value = configData.profile?.location || '';
    }
    
    // Context field
    const ctxEl = document.getElementById('simple-context');
    
    // Priority: 1) localStorage (most recent working draft), 2) server config, 3) legacy build
    if (savedState && savedState.categories && savedState.categories.length > 0) {
        // localStorage has categories — use as truth (user may have unsaved changes)
        simpleCategories = normalizeCategories(savedState.categories);
        // Tickers: use localStorage array (even if empty — user may have cleared them).
        // Only fall back to server config if localStorage has no tickers key at all.
        simpleTickers = Array.isArray(savedState.tickers)
            ? savedState.tickers
            : (configData.market?.tickers || []).map(t => typeof t === 'object' ? (t.symbol || String(t)) : String(t));
        simpleTimelimit = savedState.timelimit || configData.news?.timelimit || 'w';
        simpleContext = ('context' in savedState) ? savedState.context : (configData.profile?.context || '');
    } else if (configData.dynamic_categories && configData.dynamic_categories.length > 0) {
        // Server has dynamic categories — use those
        simpleCategories = normalizeCategories(configData.dynamic_categories);
        simpleTickers = (configData.market?.tickers || []).map(t => t.symbol);
        simpleTimelimit = configData.news?.timelimit || 'w';
        simpleContext = configData.profile?.context || '';
        saveSimpleState();
    } else {
        // Nothing anywhere — build from legacy config
        buildCategoriesFromConfig();
    }
    
    if (ctxEl) ctxEl.value = simpleContext;
    
    renderDynamicCategories();
    renderSimpleTickers();
    updateTimelimitButtons();
    loadSimpleProfilesList();
    // Initialize sub-tabs and ticker presets
    switchSettingsTab(_currentSettingsTab);
    loadTickerPresets();
}

// Auto-update context when role or location changes
(function initRoleLocationSync() {
    // Wait for DOM
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            const roleEl = document.getElementById('simple-role');
            const locEl = document.getElementById('simple-location');
            if (!roleEl || !locEl) return;
            
            function rebuildContext() {
                const role = roleEl.value.trim();
                const location = locEl.value.trim();
                const ctxEl = document.getElementById('simple-context');
                if (!ctxEl || !role) return;
                
                const currentCtx = ctxEl.value.trim();
                
                // Extract interests from current context
                // Patterns: "with interests in X, Y, Z", "focusing on X, Y, Z", "with a focus on X, Y, Z"
                let interests = '';
                const patterns = [
                    /(?:with interests? in|focusing on|with a focus on|interested in)\s+(.+)/i,
                    /[,.]\s*([\w\s,]+(?:,\s*(?:and\s+)?[\w\s]+)+)\.?\s*$/i,
                ];
                for (const pat of patterns) {
                    const match = currentCtx.match(pat);
                    if (match) {
                        interests = match[1].replace(/\.\s*$/, '').trim();
                        break;
                    }
                }
                
                // Also check: if categories exist, build interests from category items
                if (!interests && simpleCategories.length > 0) {
                    const allItems = [];
                    simpleCategories.forEach(cat => {
                        (cat.items || []).slice(0, 3).forEach(item => {
                            if (!allItems.includes(item)) allItems.push(item);
                        });
                    });
                    interests = allItems.slice(0, 8).join(', ');
                }
                
                // Rebuild context
                let newCtx;
                if (interests) {
                    newCtx = `${role} in ${location || 'Kuwait'} with interests in ${interests}.`;
                } else {
                    newCtx = `${role} in ${location || 'Kuwait'}.`;
                }
                
                ctxEl.value = newCtx;
                simpleContext = newCtx;
                
                // Also sync to Advanced
                const advCtx = document.getElementById('cfg-profile-context');
                if (advCtx) advCtx.value = newCtx;
                const advRole = document.getElementById('cfg-profile-role');
                if (advRole) advRole.value = role;
                const advLoc = document.getElementById('cfg-profile-location');
                if (advLoc) advLoc.value = location;
            }
            
            // Listen for changes — use 'change' (fires on blur) to avoid rebuilding on every keystroke
            roleEl.addEventListener('change', () => { rebuildContext(); saveSimpleState(); });
            locEl.addEventListener('change', () => { rebuildContext(); saveSimpleState(); });
            
            // Save context when manually edited
            const ctxSync = document.getElementById('simple-context');
            if (ctxSync) {
                ctxSync.addEventListener('change', () => {
                    simpleContext = ctxSync.value.trim();
                    saveSimpleState();
                });
                // Also sync on input (live) so navigating away mid-typing preserves changes
                ctxSync.addEventListener('input', () => {
                    simpleContext = ctxSync.value.trim();
                    saveSimpleState();
                });
            }
        }, 500);
    });
})();

// Build default categories from existing config (backward compat)
function buildCategoriesFromConfig() {
    simpleCategories = [];

    const companies = configData.news?.career?.keywords || [];
    const banks = configData.news?.finance?.keywords || [];
    const tech = configData.news?.tech_trends?.keywords || [];

    // If all legacy keyword arrays are empty, don't create any categories —
    // let the "Generate" prompt show instead of resurrecting phantom categories
    if (!companies.length && !banks.length && !tech.length) {
        simpleTickers = (configData.market?.tickers || []).map(t => t.symbol);
        simpleTimelimit = configData.news?.timelimit || 'w';
        simpleContext = configData.profile?.context || '';
        return;
    }

    if (companies.length) {
        simpleCategories.push({ id: 'companies', label: 'Companies', icon: 'briefcase', items: [...companies] });
    }
    if (banks.length) {
        simpleCategories.push({ id: 'banks', label: 'Banks', icon: 'landmark', items: [...banks] });
    }
    if (tech.length) {
        simpleCategories.push({ id: 'tech', label: 'Tech Topics', icon: 'cpu', items: [...tech] });
    }

    simpleTickers = (configData.market?.tickers || []).map(t => t.symbol);
    simpleTimelimit = configData.news?.timelimit || 'w';
    simpleContext = configData.profile?.context || '';
}

// === DYNAMIC CATEGORY RENDERING ===
function renderDynamicCategories() {
    const container = document.getElementById('simple-categories-container');
    if (!container) return;
    
    if (simpleCategories.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-slate-500 border border-dashed border-slate-700 rounded-xl">
                <i data-lucide="sparkles" class="w-8 h-8 mx-auto mb-2 opacity-50"></i>
                <p class="text-sm">Enter your role above and click <strong>Generate</strong></p>
                <p class="text-xs mt-1">AI will create tracking categories tailored to you</p>
            </div>`;
        lucide.createIcons();
        renderAdvancedCategories();
        if (typeof renderCategoryPresets === 'function') renderCategoryPresets();
        return;
    }
    
    container.innerHTML = simpleCategories.map((cat, ci) => {
        const isEnabled = cat.enabled !== false;  // default true
        // Normalize: handle old {name,keywords} and new {label,items} formats
        const catLabel = cat.label || cat.name || 'Untitled';
        const catItems = cat.items || cat.keywords || [];
        return `
        <div class="glass-panel rounded-xl p-4 relative group/cat ${isEnabled ? '' : 'opacity-40'}" style="transition: opacity 0.2s">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                    <i data-lucide="${CATEGORY_ICONS[cat.icon] || 'tag'}" class="w-4 h-4" style="color:var(--accent)"></i>
                    <span class="text-sm font-bold text-slate-200">${esc(catLabel)}</span>
                    <span class="text-[10px] text-slate-600">${catItems.length}</span>
                    <button onclick="saveSingleCategory(${ci})" class="opacity-0 group-hover/cat:opacity-100 text-slate-500 hover:text-amber-400 transition-all p-1 rounded hover:bg-amber-900/20" title="Save to library">
                        <i data-lucide="bookmark" class="w-3.5 h-3.5"></i>
                    </button>
                    <button onclick="removeCategory(${ci})" class="opacity-0 group-hover/cat:opacity-100 text-slate-500 hover:text-red-400 transition-all p-1 rounded hover:bg-red-900/20" title="Delete category">
                        <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                    </button>
                </div>
                <div class="flex items-center gap-2">
                    <label class="relative inline-flex items-center cursor-pointer" title="${isEnabled ? 'Disable category (won\'t be scraped)' : 'Enable category'}">
                        <input type="checkbox" class="sr-only peer" ${isEnabled ? 'checked' : ''} 
                               onchange="toggleCategory(${ci}, this.checked)">
                        <div class="w-8 h-4 bg-slate-700 rounded-full peer peer-checked:bg-emerald-600 
                             after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
                             after:bg-slate-300 after:rounded-full after:h-3 after:w-3 
                             after:transition-all peer-checked:after:translate-x-4 peer-checked:after:bg-white"></div>
                    </label>
                </div>
            </div>
            <div class="flex flex-wrap gap-1.5" id="cat-bubbles-${ci}">
                ${catItems.map(item => `
                    <span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium cursor-pointer transition-all hover:opacity-75"
                         style="background:var(--accent-bg); border:1px solid var(--accent-border); color:var(--accent-light)"
                         onclick="removeCategoryItem(${ci}, '${esc(item).replace(/'/g, "\\'")}')">${esc(item)} <i data-lucide="x" class="w-2.5 h-2.5 opacity-40 hover:opacity-100"></i></span>
                `).join('')}
            </div>
            <input type="text" class="mt-2 w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-[11px] text-slate-300 focus:border-emerald-500/50 focus:outline-none placeholder-slate-600" 
                   placeholder="+ Add to ${esc(catLabel)}..." 
                   onkeydown="if(event.key==='Enter'){addCategoryItem(${ci}, this.value); this.value='';}">
        </div>`;
    }).join('');
    
    lucide.createIcons();
    renderAdvancedCategories();
    if (typeof renderCategoryPresets === 'function') renderCategoryPresets();
}

// Also render in Advanced panel
function renderAdvancedCategories() {
    const container = document.getElementById('adv-categories-list');
    const countEl = document.getElementById('adv-cat-count');
    if (!container) return;
    
    if (countEl) countEl.textContent = simpleCategories.length ? `${simpleCategories.length}` : '';
    
    if (simpleCategories.length === 0) {
        container.innerHTML = `<p class="text-xs text-slate-500 italic">No categories yet — use Generate in Simple mode or add manually below.</p>`;
        return;
    }
    
    // Human-readable labels for scorer types
    const SCORER_LABELS = { career: 'Job postings', banks: 'Bank offers', tech: 'Tech & science', regional: 'Regional GCC', auto: 'Auto-detect' };
    const ROOT_LABELS = { kuwait: 'Kuwait Radar', regional: 'Regional News', global: 'Global News', ai: 'AI & Tech' };
    
    container.innerHTML = simpleCategories.map((cat, ci) => {
        const isEnabled = cat.enabled !== false;
        const catLabel = cat.label || cat.name || 'Untitled';
        const catItems = cat.items || cat.keywords || [];
        const scorerType = cat.scorer_type || 'auto';
        const root = cat.root || 'kuwait';
        const iconName = CATEGORY_ICONS[cat.icon] || cat.icon || 'tag';
        
        return `
        <div class="rounded-lg ${isEnabled ? '' : 'opacity-40'}" style="background: var(--bg-panel-solid); border: 1px solid var(--border-strong); transition: opacity 0.2s">
            <div class="flex items-center justify-between px-3 py-2" style="border-bottom: 1px solid var(--border-strong)">
                <div class="flex items-center gap-2 min-w-0">
                    <i data-lucide="${iconName}" class="w-3.5 h-3.5 flex-shrink-0" style="color:var(--accent)"></i>
                    <span class="text-xs font-bold text-slate-200 truncate">${esc(catLabel)}</span>
                    <span class="text-[9px] text-slate-600">${catItems.length} items</span>
                </div>
                <div class="flex items-center gap-1.5 flex-shrink-0">
                    <label class="relative inline-flex items-center cursor-pointer" title="${isEnabled ? 'Disable' : 'Enable'}">
                        <input type="checkbox" class="sr-only peer" ${isEnabled ? 'checked' : ''} 
                               onchange="toggleCategory(${ci}, this.checked)">
                        <div class="w-7 h-3.5 bg-slate-700 rounded-full peer peer-checked:bg-emerald-600 
                             after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
                             after:bg-slate-300 after:rounded-full after:h-2.5 after:w-2.5 
                             after:transition-all peer-checked:after:translate-x-3.5 peer-checked:after:bg-white"></div>
                    </label>
                    <button onclick="removeCategory(${ci})" class="text-red-400/40 hover:text-red-400 transition-all p-0.5 rounded hover:bg-red-900/20" title="Delete">
                        <i data-lucide="trash-2" class="w-3 h-3"></i>
                    </button>
                </div>
            </div>
            <div class="px-3 py-2">
                <div class="flex flex-wrap gap-1 mb-1.5">
                    ${catItems.map(item => `
                        <span class="inline-flex items-center gap-0.5 px-2 py-0.5 rounded-full text-[10px] font-medium cursor-pointer transition-all hover:opacity-70"
                             style="background:var(--accent-bg); border:1px solid var(--accent-border); color:var(--accent-light)"
                             onclick="removeCategoryItem(${ci}, '${esc(item).replace(/'/g, "\\\\'")}')" title="Click to remove">${esc(item)} ×</span>
                    `).join('')}
                </div>
                <input type="text" class="w-full bg-transparent border-0 border-b border-slate-700/30 px-0 py-1 text-[10px] text-slate-400 focus:border-emerald-500/50 focus:outline-none placeholder-slate-600" 
                       placeholder="+ Add item..." 
                       onkeydown="if(event.key==='Enter'){addCategoryItem(${ci}, this.value); this.value='';}">
            </div>
            <div class="px-3 py-1.5 text-[9px] text-slate-600" style="border-top: 1px solid var(--border-strong)">
                <span>Scored as: <strong class="text-slate-500">${SCORER_LABELS[scorerType] || scorerType}</strong></span>
            </div>
        </div>`;
    }).join('');
    
    lucide.createIcons();
}

function toggleAdvAddCategory() {
    const form = document.getElementById('adv-add-cat-form');
    const toggle = document.getElementById('adv-add-cat-toggle');
    if (!form) return;
    const showing = !form.classList.contains('hidden');
    form.classList.toggle('hidden');
    if (toggle) toggle.classList.toggle('hidden');
    if (!showing) {
        document.getElementById('adv-new-cat-label')?.focus();
    }
}

function addAdvancedCategory() {
    const labelEl = document.getElementById('adv-new-cat-label');
    const scorerEl = document.getElementById('adv-new-cat-scorer');
    const rootEl = document.getElementById('adv-new-cat-root');
    const iconEl = document.getElementById('adv-new-cat-icon');
    const itemsEl = document.getElementById('adv-new-cat-items');
    
    const label = labelEl?.value.trim();
    if (!label) {
        if (typeof showToast === 'function') showToast('Enter a category label', 'warning');
        return;
    }
    
    const items = (itemsEl?.value || '')
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0);
    
    if (items.length === 0) {
        if (typeof showToast === 'function') showToast('Add at least one item', 'warning');
        return;
    }
    
    // Generate a short ID from label
    const id = label.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 12);
    
    // Check for duplicate ID
    if (simpleCategories.some(c => (c.id || '').toLowerCase() === id)) {
        if (typeof showToast === 'function') showToast('A category with a similar name already exists', 'warning');
        return;
    }
    
    const newCat = {
        id: id,
        label: label,
        icon: iconEl?.value || 'globe',
        items: items,
        enabled: true,
        scorer_type: scorerEl?.value || 'auto',
        root: rootEl?.value || 'kuwait',
        pinned: true,  // Manual categories survive Generate
    };
    
    simpleCategories.push(newCat);
    _syncCategoriesState();
    renderAdvancedCategories();
    
    // Clear inputs and collapse form
    if (labelEl) labelEl.value = '';
    if (itemsEl) itemsEl.value = '';
    toggleAdvAddCategory();
    
    window._settingsDirty = true;
    if (typeof showToast === 'function') showToast(`Added "${label}"`, 'success');
}

function renderSimpleTickers() {
    const container = document.getElementById('simple-tickers');
    if (!container) return;
    
    container.innerHTML = simpleTickers.map((t, idx) => {
        return `
        <span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium transition-all hover:opacity-75 select-none"
             style="background:var(--accent-bg); border:1px solid var(--accent-border); color:var(--accent-light); cursor:grab;"
             draggable="true"
             data-ticker-idx="${idx}"
             ondragstart="onTickerDragStart(event, ${idx})"
             ondragover="onTickerDragOver(event)"
             ondrop="onTickerDrop(event, ${idx})"
             ondragend="onTickerDragEnd(event)">${esc(t)} <span class="opacity-40 hover:opacity-100 cursor-pointer" onclick="event.stopPropagation(); removeSimpleTicker('${esc(t)}')">✕</span></span>`;
    }).join('');
    
    // Update count
    const countEl = document.getElementById('simple-tickers-count');
    if (countEl) countEl.textContent = simpleTickers.length ? `(${simpleTickers.length})` : '';
}

// Ticker drag-and-drop reordering
let _tickerDragIdx = -1;

function onTickerDragStart(e, idx) {
    _tickerDragIdx = idx;
    e.target.style.opacity = '0.4';
    e.dataTransfer.effectAllowed = 'move';
}

function onTickerDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function onTickerDrop(e, dropIdx) {
    e.preventDefault();
    if (_tickerDragIdx < 0 || _tickerDragIdx === dropIdx) return;
    const item = simpleTickers.splice(_tickerDragIdx, 1)[0];
    simpleTickers.splice(dropIdx, 0, item);
    _tickerDragIdx = -1;
    saveSimpleState();
    renderSimpleTickers();
}

function onTickerDragEnd(e) {
    e.target.style.opacity = '1';
    _tickerDragIdx = -1;
}

function clearAllTickers() {
    if (!simpleTickers.length) return;
    simpleTickers = [];
    saveSimpleState();
    renderSimpleTickers();
    if (typeof showToast === 'function') showToast('All tickers cleared', 'info');
}

function updateTimelimitButtons() {
    document.querySelectorAll('.simple-time-btn').forEach(btn => {
        if (btn.dataset.tl === simpleTimelimit) {
            btn.style.background = 'var(--accent-bg)';
            btn.style.borderColor = 'var(--accent-border)';
            btn.style.color = 'var(--accent-light)';
        } else {
            btn.style.background = 'var(--bg-panel-solid)';
            btn.style.borderColor = 'var(--border-strong)';
            btn.style.color = 'var(--text-secondary)';
        }
    });
}

// === CATEGORY MANIPULATION ===
function _syncCategoriesState() {
    // Keep configData and nav in sync with simpleCategories
    saveSimpleState();
    if (typeof configData !== 'undefined') {
        configData.dynamic_categories = JSON.parse(JSON.stringify(simpleCategories));
    }
    renderDynamicCategories();
    if (typeof rebuildNavFromConfig === 'function') {
        rebuildNavFromConfig();
    }
    window._settingsDirty = true;
}

function toggleCategory(catIdx, enabled) {
    if (catIdx >= simpleCategories.length) return;
    simpleCategories[catIdx].enabled = enabled;
    _syncCategoriesState();
}

function addCategoryItem(catIdx, value) {
    if (!value || !value.trim()) return;
    value = value.trim();
    if (catIdx >= simpleCategories.length) return;
    if (simpleCategories[catIdx].items.includes(value)) return;
    simpleCategories[catIdx].items.push(value);
    _syncCategoriesState();
}

function removeCategoryItem(catIdx, value) {
    if (catIdx >= simpleCategories.length) return;
    simpleCategories[catIdx].items = simpleCategories[catIdx].items.filter(v => v !== value);
    _syncCategoriesState();
}

function removeCategory(catIdx) {
    simpleCategories.splice(catIdx, 1);
    _syncCategoriesState();
}

function addSimpleTicker(value) {
    if (!value || !value.trim()) return;
    value = value.trim().toUpperCase();
    if (simpleTickers.includes(value)) {
        if (typeof showToast === 'function') showToast(`${value} already added`, 'warning');
        return;
    }
    simpleTickers.push(value);
    saveSimpleState();
    renderSimpleTickers();
    if (typeof showToast === 'function') showToast(`Added ${value}`, 'success');
}

// Top 10 most-tracked tickers across equities, crypto, and commodities
const TOP_10_TICKERS = [
    'NVDA',      // Nvidia — AI/GPU leader
    'AAPL',      // Apple — largest market cap
    'MSFT',      // Microsoft — enterprise + AI
    'AMZN',      // Amazon — cloud + e-commerce
    'GOOGL',     // Google — search + AI
    'BTC-USD',   // Bitcoin
    'ETH-USD',   // Ethereum
    'GC=F',      // Gold futures
    'CL=F',      // Crude oil futures
    'SPY',       // S&P 500 ETF — overall market
];

function addTop10Tickers() {
    let added = 0;
    for (const t of TOP_10_TICKERS) {
        if (!simpleTickers.includes(t)) {
            simpleTickers.push(t);
            added++;
        }
    }
    if (added > 0) {
        saveSimpleState();
        renderSimpleTickers();
        if (typeof showToast === 'function') showToast(`Added ${added} ticker${added > 1 ? 's' : ''}`, 'success');
    } else {
        if (typeof showToast === 'function') showToast('All Top 10 already added', 'info');
    }
}

// Top 10 highest-volatility tickers — fetched live from Yahoo Finance
async function addTopMoversTickers() {
    const btn = document.querySelector('[onclick*="addTopMoversTickers"]');
    const icon = btn?.querySelector('svg');
    if (icon) icon.classList.add('animate-spin');
    if (btn) btn.disabled = true;
    
    try {
        const r = await fetch('/api/top-movers');
        if (!r.ok) throw new Error('Server error');
        const data = await r.json();
        const movers = data.movers || [];
        
        if (!movers.length) {
            if (typeof showToast === 'function') showToast('No movers data available', 'warning');
            return;
        }
        
        let added = 0;
        const names = [];
        for (const m of movers) {
            if (!simpleTickers.includes(m.symbol)) {
                simpleTickers.push(m.symbol);
                added++;
            }
            names.push(`${m.symbol} (${m.change_pct > 0 ? '+' : ''}${m.change_pct}%)`);
        }
        
        if (added > 0) {
            saveSimpleState();
            renderSimpleTickers();
            if (typeof showToast === 'function') showToast(`Added ${added} mover${added > 1 ? 's' : ''}: ${names.slice(0, 3).join(', ')}...`, 'success', 4000);
        } else {
            if (typeof showToast === 'function') showToast('All top movers already added', 'info');
        }
    } catch (e) {
        console.error('Top movers fetch failed:', e);
        if (typeof showToast === 'function') showToast('Failed to fetch movers — is the backend running?', 'error');
    } finally {
        if (icon) icon.classList.remove('animate-spin');
        if (btn) btn.disabled = false;
    }
}

function removeSimpleTicker(value) {
    simpleTickers = simpleTickers.filter(v => v !== value);
    saveSimpleState();
    renderSimpleTickers();
}

function setSimpleTimelimit(tl) {
    simpleTimelimit = tl;
    saveSimpleState();
    updateTimelimitButtons();
}

// Persist simple state to localStorage so it survives page refresh
function saveSimpleState() {
    localStorage.setItem('simpleCategories', JSON.stringify({
        categories: simpleCategories,
        tickers: simpleTickers,
        timelimit: simpleTimelimit,
        context: simpleContext,
        role: document.getElementById('simple-role')?.value || '',
        location: document.getElementById('simple-location')?.value || '',
    }));
}

// === SAVE CONTEXT ONLY ===
async function saveContextOnly() {
    const ctxEl = document.getElementById('simple-context');
    const btn = document.getElementById('save-context-btn');
    if (!ctxEl || !btn) return;
    
    const ctx = ctxEl.value.trim();
    if (!ctx) {
        if (typeof showToast === 'function') showToast('Context is empty', 'warning');
        return;
    }
    
    // Update in-memory state
    simpleContext = ctx;
    saveSimpleState();
    
    // Save to backend via quickSave
    const origHTML = btn.innerHTML;
    btn.innerHTML = '<i data-lucide="loader-2" class="w-3 h-3 animate-spin"></i> Saving...';
    lucide.createIcons({nodes: [btn]});
    
    await quickSave();
    
    btn.innerHTML = '<i data-lucide="check" class="w-3 h-3"></i> Saved';
    btn.style.borderColor = 'rgba(16,185,129,0.5)';
    lucide.createIcons({nodes: [btn]});
    
    setTimeout(() => {
        btn.innerHTML = origHTML;
        btn.style.borderColor = '';
        lucide.createIcons({nodes: [btn]});
    }, 1500);
}

// === CONTEXT SUGGESTION ===
async function suggestContext() {
    const role = document.getElementById('simple-role').value.trim();
    const location = document.getElementById('simple-location').value.trim();
    const btn = document.getElementById('suggest-context-btn');
    const textarea = document.getElementById('simple-context');
    
    if (!role) {
        textarea.placeholder = '← Enter your role first, then click Suggest';
        setTimeout(() => { textarea.placeholder = 'What are you interested in? e.g., Investing, cutting edge technology, ticker trends...'; }, 3000);
        return;
    }
    
    btn.innerHTML = '<div class="w-4 h-4 border-2 border-slate-500/30 border-t-slate-300 rounded-full animate-spin"></div> Suggest';
    btn.disabled = true;
    
    try {
        const resp = await fetch('/api/suggest-context', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(30000),
            body: JSON.stringify({ role, location })
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        
        if (data.suggestion) {
            // Always replace — Suggest generates a complete context from scratch
            textarea.value = data.suggestion;
            simpleContext = data.suggestion;
            // Brief highlight effect
            textarea.style.borderColor = 'var(--accent)';
            setTimeout(() => { textarea.style.borderColor = ''; }, 1500);
        }
        // Tickers intentionally NOT applied here — user clicks Generate for that
    } catch (err) {
        console.error('Context suggestion failed:', err);
        if (typeof showToast === 'function') showToast('Suggestion failed — is Ollama running?', 'error');
    }
    
    btn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i> Suggest';
    btn.disabled = false;
    lucide.createIcons();
}

// === AI PROFILE GENERATION ===
let previousSimpleState = null; // For undo
let redoSimpleState = null;     // For redo

async function generateFromRole() {
    const role = document.getElementById('simple-role').value.trim();
    const location = document.getElementById('simple-location').value.trim();
    const btn = document.getElementById('simple-generate-btn');
    const status = document.getElementById('simple-generate-status');
    
    if (!role) {
        status.innerHTML = '<span class="text-amber-400">Enter a role first</span>';
        setTimeout(() => { status.innerHTML = ''; }, 2000);
        return;
    }
    
    if (isGenerating) return;
    isGenerating = true;
    
    // Snapshot current state for rollback
    previousSimpleState = {
        categories: JSON.parse(JSON.stringify(simpleCategories)),
        tickers: [...simpleTickers],
        timelimit: simpleTimelimit,
        context: simpleContext,
    };
    
    btn.disabled = true;
    btn.innerHTML = '<div class="w-3 h-3 border-2 border-purple-300/30 border-t-purple-300 rounded-full animate-spin"></div> Generating...';
    status.innerHTML = '<span class="text-purple-400 animate-pulse text-[11px]">AI is analyzing your role and creating tracking categories...</span>';
    
    try {
        const resp = await fetch('/api/generate-profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(180000),
            body: JSON.stringify({ role, location, context: document.getElementById('simple-context')?.value?.trim() || '' })
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        
        if (data.error) throw new Error(data.error);
        
        // Apply AI-generated config, but preserve pinned (manually-added) categories
        if (data.categories && Array.isArray(data.categories)) {
            const pinned = simpleCategories.filter(c => c.pinned);
            const generated = data.categories;
            // Avoid duplicates: remove generated categories that share a label with pinned ones
            const pinnedLabels = new Set(pinned.map(c => (c.label || c.name || '').toLowerCase()));
            const filtered = generated.filter(c => !pinnedLabels.has((c.label || c.name || '').toLowerCase()));
            simpleCategories = [...filtered, ...pinned];
        }
        if (data.tickers && Array.isArray(data.tickers)) {
            simpleTickers = data.tickers;
        }
        if (data.timelimit) {
            simpleTimelimit = data.timelimit;
        }
        if (data.context) {
            simpleContext = data.context;
        }
        
        saveSimpleState();
        renderDynamicCategories();
        renderSimpleTickers();
        updateTimelimitButtons();
        
        // Auto-sync generated categories to Advanced fields
        syncToAdvanced();
        window._settingsDirty = true; // Mark as unsaved so navigating away won't wipe changes
        
        // Show rollback button, hide redo (new generation replaces redo history)
        const rollbackBtn = document.getElementById('simple-rollback-btn');
        const redoBtn = document.getElementById('simple-redo-btn');
        if (rollbackBtn) rollbackBtn.classList.remove('hidden');
        if (redoBtn) redoBtn.classList.add('hidden');
        redoSimpleState = null;
        
        status.innerHTML = `<span class="text-emerald-400 text-[11px]">✓ Generated ${simpleCategories.length} categories with ${simpleCategories.reduce((a,c) => a + c.items.length, 0)} items — Advanced tab updated</span>`;
        setTimeout(() => { status.innerHTML = ''; }, 4000);
        if (typeof showToast === 'function') showToast(`Generated ${simpleCategories.length} categories`, 'success');
        
    } catch(err) {
        console.error('Profile generation failed:', err);
        status.innerHTML = `<span class="text-red-400 text-[11px]">Failed: ${err.message}</span>`;
        if (typeof showToast === 'function') showToast('Generation failed: ' + err.message, 'error');
        // Restore on failure
        if (previousSimpleState) {
            simpleCategories = previousSimpleState.categories;
            simpleTickers = previousSimpleState.tickers;
            simpleTimelimit = previousSimpleState.timelimit;
            simpleContext = previousSimpleState.context;
            renderDynamicCategories();
            renderSimpleTickers();
            updateTimelimitButtons();
        }
    }
    
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="sparkles" class="w-3.5 h-3.5"></i> Generate';
    isGenerating = false;
    lucide.createIcons();
}

/**
 * Sync Simple mode categories/tickers/context → Advanced tab fields.
 * Called after Generate so Advanced tab is immediately up-to-date.
 * Does NOT save to server — just populates the form fields.
 */
function syncToAdvanced() {
    // Sync ONLY profile fields + tickers from Simple → Advanced
    // DO NOT map category items into legacy keyword fields — that causes conflicts
    // (e.g. "internships, job" going into "Target Companies" which expects "Equate, KNPC")
    
    const roleField = document.getElementById('cfg-profile-role');
    const locationField = document.getElementById('cfg-profile-location');
    const contextField = document.getElementById('cfg-profile-context');
    const timelimitField = document.getElementById('cfg-timelimit');
    
    if (timelimitField) timelimitField.value = simpleTimelimit;
    
    // Sync profile fields
    const role = document.getElementById('simple-role')?.value || '';
    const location = document.getElementById('simple-location')?.value || '';
    const context = document.getElementById('simple-context')?.value || simpleContext || '';
    if (roleField) roleField.value = role;
    if (locationField) locationField.value = location;
    if (contextField) contextField.value = context;
    
    // Career keywords are now managed via dynamic categories in Simple mode
    
    console.log(`[syncToAdvanced] Profile synced. Categories: ${simpleCategories.length} (legacy fields ${simpleCategories.length > 0 ? 'cleared' : 'preserved'})`);
}

function rollbackGeneration() {
    if (!previousSimpleState) return;
    
    // Save current state for redo before overwriting
    redoSimpleState = {
        categories: JSON.parse(JSON.stringify(simpleCategories)),
        tickers: [...simpleTickers],
        timelimit: simpleTimelimit,
        context: simpleContext,
    };
    
    simpleCategories = previousSimpleState.categories;
    simpleTickers = previousSimpleState.tickers;
    simpleTimelimit = previousSimpleState.timelimit;
    simpleContext = previousSimpleState.context;
    previousSimpleState = null;
    
    const ctxEl = document.getElementById('simple-context');
    if (ctxEl) ctxEl.value = simpleContext;
    
    saveSimpleState();
    renderDynamicCategories();
    renderSimpleTickers();
    updateTimelimitButtons();
    
    // Hide undo, show redo
    const rollbackBtn = document.getElementById('simple-rollback-btn');
    const redoBtn = document.getElementById('simple-redo-btn');
    if (rollbackBtn) rollbackBtn.classList.add('hidden');
    if (redoBtn) redoBtn.classList.remove('hidden');
    
    const status = document.getElementById('simple-generate-status');
    status.innerHTML = '<span class="text-amber-400 text-[11px]">↩ Rolled back to previous settings</span>';
    setTimeout(() => { status.innerHTML = ''; }, 3000);
}

function redoGeneration() {
    if (!redoSimpleState) return;
    
    // Save current state back to previousSimpleState so undo works again
    previousSimpleState = {
        categories: JSON.parse(JSON.stringify(simpleCategories)),
        tickers: [...simpleTickers],
        timelimit: simpleTimelimit,
        context: simpleContext,
    };
    
    simpleCategories = redoSimpleState.categories;
    simpleTickers = redoSimpleState.tickers;
    simpleTimelimit = redoSimpleState.timelimit;
    simpleContext = redoSimpleState.context;
    redoSimpleState = null;
    
    const ctxEl = document.getElementById('simple-context');
    if (ctxEl) ctxEl.value = simpleContext;
    
    saveSimpleState();
    renderDynamicCategories();
    renderSimpleTickers();
    updateTimelimitButtons();
    
    // Hide redo, show undo
    const rollbackBtn = document.getElementById('simple-rollback-btn');
    const redoBtn = document.getElementById('simple-redo-btn');
    if (redoBtn) redoBtn.classList.add('hidden');
    if (rollbackBtn) rollbackBtn.classList.remove('hidden');
    
    const status = document.getElementById('simple-generate-status');
    status.innerHTML = '<span class="text-blue-400 text-[11px]">↪ Restored generated settings</span>';
    setTimeout(() => { status.innerHTML = ''; }, 3000);
}

// === SIMPLE PROFILE SAVE/LOAD ===
// Namespace localStorage key per logged-in user so profiles aren't shared
function _simpleProfilesKey() {
    const user = (typeof getActiveProfile === 'function') ? getActiveProfile() : '';
    return user ? `simpleProfiles_${user}` : 'simpleProfiles';
}

async function saveSimpleProfile() {
    const nameInput = document.getElementById('simple-profile-name');
    const name = nameInput.value.trim();
    const statusEl = document.getElementById('simple-profile-status');
    
    if (!name) {
        statusEl.innerHTML = '<span class="text-amber-400">Enter a profile name</span>';
        setTimeout(() => { statusEl.innerHTML = ''; }, 2000);
        return;
    }
    
    const profileData = {
        simple: true,
        role: document.getElementById('simple-role')?.value || '',
        location: document.getElementById('simple-location')?.value || '',
        context_text: document.getElementById('simple-context')?.value || '',
        categories: JSON.parse(JSON.stringify(simpleCategories)),
        tickers: [...simpleTickers],
        timelimit: simpleTimelimit,
        ai_context: simpleContext,
    };
    
    // Store in localStorage (keyed by name)
    const profiles = JSON.parse(localStorage.getItem(_simpleProfilesKey()) || '{}');
    profiles[name] = { data: profileData, saved_at: new Date().toISOString() };
    localStorage.setItem(_simpleProfilesKey(), JSON.stringify(profiles));
    
    nameInput.value = '';
    loadSimpleProfilesList();
    
    statusEl.innerHTML = `<span style="color:var(--accent)">✓ Saved "${name}"</span>`;
    setTimeout(() => { statusEl.innerHTML = ''; }, 3000);
}

function loadSimpleProfile(name) {
    const profiles = JSON.parse(localStorage.getItem(_simpleProfilesKey()) || '{}');
    const profile = profiles[name];
    if (!profile || !profile.data) return;
    
    const d = profile.data;
    
    // Snapshot current for rollback
    previousSimpleState = {
        categories: JSON.parse(JSON.stringify(simpleCategories)),
        tickers: [...simpleTickers],
        timelimit: simpleTimelimit,
        context: simpleContext,
    };
    
    // Apply profile
    if (d.role) document.getElementById('simple-role').value = d.role;
    if (d.location) document.getElementById('simple-location').value = d.location;
    const ctxEl = document.getElementById('simple-context');
    if (ctxEl && d.context_text) ctxEl.value = d.context_text;
    
    if (d.categories) simpleCategories = d.categories;
    if (d.tickers) simpleTickers = d.tickers;
    if (d.timelimit) simpleTimelimit = d.timelimit;
    if (d.ai_context) simpleContext = d.ai_context;
    
    saveSimpleState();
    renderDynamicCategories();
    renderSimpleTickers();
    updateTimelimitButtons();
    
    // Show rollback button, hide redo
    const rollbackBtn = document.getElementById('simple-rollback-btn');
    const redoBtn = document.getElementById('simple-redo-btn');
    if (rollbackBtn) rollbackBtn.classList.remove('hidden');
    if (redoBtn) redoBtn.classList.add('hidden');
    redoSimpleState = null;
    
    const statusEl = document.getElementById('simple-profile-status');
    statusEl.innerHTML = `<span style="color:var(--accent)">✓ Loaded "${name}"</span>`;
    setTimeout(() => { statusEl.innerHTML = ''; }, 3000);
    lucide.createIcons();
}

function deleteSimpleProfile(name) {
    const profiles = JSON.parse(localStorage.getItem(_simpleProfilesKey()) || '{}');
    delete profiles[name];
    localStorage.setItem(_simpleProfilesKey(), JSON.stringify(profiles));
    loadSimpleProfilesList();
    
    const statusEl = document.getElementById('simple-profile-status');
    statusEl.innerHTML = `<span class="text-slate-500 text-[11px]">Deleted "${name}"</span>`;
    setTimeout(() => { statusEl.innerHTML = ''; }, 2000);
}

function loadSimpleProfilesList() {
    const container = document.getElementById('simple-profiles-list');
    if (!container) return;
    
    const profiles = JSON.parse(localStorage.getItem(_simpleProfilesKey()) || '{}');
    const names = Object.keys(profiles);
    
    if (names.length === 0) {
        container.innerHTML = '<p class="text-[11px] text-slate-600 italic">No saved profiles yet.</p>';
        return;
    }
    
    container.innerHTML = names.map(name => {
        const p = profiles[name];
        const d = p.data || {};
        const catCount = (d.categories || []).length;
        const itemCount = (d.categories || []).reduce((a, c) => a + (c.items || []).length, 0);
        const role = d.role || '';
        const savedDate = p.saved_at ? new Date(p.saved_at).toLocaleDateString() : '';
        
        return `
        <div class="flex items-center gap-2 p-2.5 rounded-lg border border-slate-800 hover:border-slate-700 transition-all group">
            <div class="flex-1 min-w-0">
                <span class="text-xs font-medium text-slate-300">${esc(name)}</span>
                <span class="text-[10px] text-slate-600 ml-2">${role ? esc(role) + ' · ' : ''}${catCount} categories · ${itemCount} items${savedDate ? ' · ' + savedDate : ''}</span>
            </div>
            <button onclick="loadSimpleProfile('${esc(name).replace(/'/g, "\\'")}')" class="px-2 py-1 text-[10px] rounded border border-blue-700/40 text-blue-400 hover:bg-blue-900/30 hover:border-blue-500/50 transition-all">
                Load
            </button>
            <button onclick="deleteSimpleProfile('${esc(name).replace(/'/g, "\\'")}')" class="px-2 py-1 text-[10px] rounded border border-slate-700/40 text-slate-500 hover:text-red-400 hover:border-red-500/40 hover:bg-red-900/20 transition-all">
                <i data-lucide="trash-2" class="w-3 h-3"></i>
            </button>
        </div>`;
    }).join('');
    
    lucide.createIcons();
}

// === CATEGORY LIBRARY (per-category save/load) ===
function _categoryLibraryKey() {
    const user = window._currentProfile || '';
    return user ? `categoryLibrary_${user}` : 'categoryLibrary';
}

function saveSingleCategory(catIdx) {
    if (catIdx >= simpleCategories.length) return;
    const cat = simpleCategories[catIdx];
    const label = cat.label || cat.name || 'Untitled';
    
    const library = JSON.parse(localStorage.getItem(_categoryLibraryKey()) || '{}');
    
    // Use label as key (overwrite if same name)
    library[label] = {
        category: JSON.parse(JSON.stringify(cat)),
        saved_at: new Date().toISOString(),
    };
    localStorage.setItem(_categoryLibraryKey(), JSON.stringify(library));
    
    renderCategoryPresets();
    if (typeof showToast === 'function') showToast(`Saved "${label}" to library`, 'success');
}

function loadLibraryCategory(name) {
    const library = JSON.parse(localStorage.getItem(_categoryLibraryKey()) || '{}');
    const entry = library[name];
    if (!entry || !entry.category) return;
    
    const cat = JSON.parse(JSON.stringify(entry.category));
    cat.pinned = true;  // Library categories survive Generate
    
    // Check if already active (by label)
    const existing = simpleCategories.findIndex(c => (c.label || c.name) === (cat.label || cat.name));
    if (existing >= 0) {
        if (typeof showToast === 'function') showToast(`"${cat.label}" is already active`, 'warning');
        return;
    }
    
    simpleCategories.push(cat);
    _syncCategoriesState();
    renderAdvancedCategories();
    
    if (typeof showToast === 'function') showToast(`Added "${cat.label || name}"`, 'success');
}

function deleteLibraryCategory(name) {
    const library = JSON.parse(localStorage.getItem(_categoryLibraryKey()) || '{}');
    delete library[name];
    localStorage.setItem(_categoryLibraryKey(), JSON.stringify(library));
    renderCategoryPresets();
    if (typeof showToast === 'function') showToast(`Removed "${name}" from library`, 'info');
}

function renderCategoryPresets() {
    const bar = document.getElementById('category-presets-bar');
    const list = document.getElementById('category-library-list');
    const countEl = document.getElementById('category-library-count');
    if (!list) return;
    
    const library = JSON.parse(localStorage.getItem(_categoryLibraryKey()) || '{}');
    const names = Object.keys(library);
    
    // Show bar only if library has items
    if (bar) {
        bar.classList.toggle('hidden', names.length === 0);
    }
    if (countEl) countEl.textContent = names.length ? `${names.length} saved` : '';
    
    if (names.length === 0) {
        list.innerHTML = '';
        return;
    }
    
    // Check which are already active
    const activeLabels = new Set(simpleCategories.map(c => c.label || c.name));
    
    list.innerHTML = names.map(name => {
        const entry = library[name];
        const cat = entry.category || {};
        const itemCount = (cat.items || []).length;
        const isActive = activeLabels.has(cat.label || cat.name || name);
        const iconName = cat.icon || 'tag';
        
        if (isActive) {
            return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-medium select-none opacity-40"
                 style="background:var(--accent-bg); border:1px solid var(--accent-border); color:var(--accent-light)"
                 title="Already active"><i data-lucide="${iconName}" class="w-3 h-3"></i>${esc(name)} ✓</span>`;
        }
        
        return `<span class="inline-flex items-center gap-1 group/lib">
            <button onclick="loadLibraryCategory('${esc(name).replace(/'/g, "\\'")}')" 
                 class="inline-flex items-center gap-1 px-2.5 py-1 rounded-l-full text-[10px] font-medium cursor-pointer transition-all hover:opacity-80"
                 style="background:var(--bg-panel-solid); border:1px solid var(--border-strong); border-right:0; color:var(--text-secondary)"
                 title="${itemCount} items — click to add"><i data-lucide="${iconName}" class="w-3 h-3"></i>${esc(name)}</button>
            <button onclick="deleteLibraryCategory('${esc(name).replace(/'/g, "\\'")}')"
                 class="px-1.5 py-1 rounded-r-full text-[10px] text-slate-600 hover:text-red-400 transition-all"
                 style="background:var(--bg-panel-solid); border:1px solid var(--border-strong); border-left:0;"
                 title="Remove from library">×</button>
        </span>`;
    }).join('');
    
    lucide.createIcons();
}

// === SAVE FROM SIMPLE MODE ===
async function saveFromSimple() {
    const btn = document.getElementById('simple-save-btn');
    const status = document.getElementById('simple-save-status');
    btn.disabled = true;
    btn.innerHTML = '<div class="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Saving...';
    
    // Sync all Simple fields → Advanced fields
    syncToAdvanced();
    
    // Save via existing saveConfig (which builds the full payload)
    // But first, inject dynamic_categories into the config payload
    window._pendingDynamicCategories = simpleCategories;
    await saveConfig();
    window._pendingDynamicCategories = null;

    // Save tickers separately (saveConfig no longer includes them to
    // prevent accidental overwrite with empty array)
    await saveTickers();
    
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="save" class="w-4 h-4"></i> Save Settings';
    status.innerHTML = '✓ Settings saved!';
    setTimeout(() => { status.innerHTML = ''; }, 3000);
    lucide.createIcons();
}

// ═══════════════════════════════════════════════════════════
// QUICK SAVE (icon in header + Ctrl+S)
// ═══════════════════════════════════════════════════════════

function clearSimpleSettings() {
    if (!confirm('Clear all Simple settings? This will remove your role, context, and all generated categories.')) return;
    
    // Clear fields
    const roleEl = document.getElementById('simple-role');
    const locEl = document.getElementById('simple-location');
    const ctxEl = document.getElementById('simple-context');
    if (roleEl) roleEl.value = '';
    if (locEl) locEl.value = '';
    if (ctxEl) ctxEl.value = '';
    
    // Clear state
    simpleCategories = [];
    simpleContext = '';
    simpleTickers = [];
    simpleTimelimit = 'w';
    previousSimpleState = null;
    redoSimpleState = null;
    
    // Hide undo/redo buttons
    const rollbackBtn = document.getElementById('simple-rollback-btn');
    const redoBtn = document.getElementById('simple-redo-btn');
    if (rollbackBtn) rollbackBtn.classList.add('hidden');
    if (redoBtn) redoBtn.classList.add('hidden');
    
    // Persist cleared state (so tab switching doesn't restore old values)
    saveSimpleState();
    
    // Re-render
    renderDynamicCategories();
    renderSimpleTickers();
    updateTimelimitButtons();
    
    window._settingsDirty = true;
    
    // CRITICAL: Save cleared state to backend so it persists across refreshes
    quickSave();
    
    const status = document.getElementById('quick-save-status');
    if (status) {
        status.textContent = 'Cleared';
        status.style.opacity = '1';
        status.style.color = '#f87171';
        setTimeout(() => { status.style.opacity = '0'; status.style.color = ''; }, 2000);
    }
    if (typeof showToast === 'function') showToast('Settings cleared', 'info');
}

async function quickSave() {
    const btn = document.getElementById('quick-save-btn');
    const status = document.getElementById('quick-save-status');
    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i> Saving...';
    lucide.createIcons({nodes: [btn]});
    
    // Save both Simple and Advanced (Simple syncs to Advanced anyway)
    if (simpleSettingsMode === 'simple') {
        await saveFromSimple();
    } else {
        await saveConfig();
    }
    // Also save source toggles
    await saveSourceToggles(true); // silent mode
    
    btn.innerHTML = '<i data-lucide="check" class="w-3.5 h-3.5"></i> Saved';
    btn.style.borderColor = 'rgba(16,185,129,0.5)';
    lucide.createIcons({nodes: [btn]});
    
    setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.style.borderColor = '';
        btn.disabled = false;
        lucide.createIcons({nodes: [btn]});
    }, 1500);
}

// Ctrl+S shortcut when on settings page
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        const settingsPanel = document.getElementById('settings-panel');
        if (settingsPanel && !settingsPanel.classList.contains('hidden')) {
            e.preventDefault();
            quickSave();
        }
    }
});

// ═══════════════════════════════════════════════════════════
// NEWS SOURCE MANAGEMENT (Finance & Politics feed toggles)
// ═══════════════════════════════════════════════════════════

let sourcesCatalog = { finance: [], politics: [] };
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
    document.getElementById('source-catalog-custom').classList.toggle('hidden', type !== 'custom');
    
    const tabs = {
        finance:  { el: document.getElementById('src-tab-finance'),  active: 'px-5 py-2 text-xs font-semibold transition-all bg-emerald-500/20 text-emerald-400 border-r border-slate-600', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 border-r border-slate-600' },
        politics: { el: document.getElementById('src-tab-politics'), active: 'px-5 py-2 text-xs font-semibold transition-all bg-blue-500/20 text-blue-400 border-r border-slate-600', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 border-r border-slate-600' },
        custom:   { el: document.getElementById('src-tab-custom'),   active: 'px-5 py-2 text-xs font-semibold transition-all bg-purple-500/20 text-purple-400 flex items-center gap-1', inactive: 'px-5 py-2 text-xs font-semibold transition-all text-slate-500 hover:text-slate-300 flex items-center gap-1' },
    };
    
    Object.entries(tabs).forEach(([key, cfg]) => {
        if (cfg.el) cfg.el.className = key === type ? cfg.active : cfg.inactive;
    });
    
    if (type === 'custom') renderCustomCatalog();
}

async function loadSourceCatalog() {
    // Show loading state
    ['finance', 'politics'].forEach(type => {
        const c = document.getElementById(`source-catalog-${type}`);
        if (c) c.innerHTML = '<p class="text-xs text-slate-500 italic py-2">Loading sources...</p>';
    });
    
    let usedFallback = false;
    
    try {
        const [finResp, polResp] = await Promise.all([
            fetch('/api/feed-catalog/finance'),
            fetch('/api/feed-catalog/politics')
        ]);
        
        if (!finResp.ok || !polResp.ok) throw new Error('Catalog endpoint returned error');
        
        const finData = await finResp.json();
        const polData = await polResp.json();
        sourcesCatalog.finance = finData.catalog || [];
        sourcesCatalog.politics = polData.catalog || [];
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

function _showRssSuggestions(type) {
    const list = document.getElementById('rss-suggestions-list');
    if (!list) return;
    // Highlight active tab
    ['finance', 'politics', 'general'].forEach(t => {
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

    // Fetch from catalog API
    fetch('/api/feed-catalog/' + type, {
        headers: { 'X-Auth-Token': localStorage.getItem('auth_token') || '' }
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
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': localStorage.getItem('auth_token') || '' },
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

function editCustomTabName() {
    const current = customTabName || 'Custom';
    const newName = prompt('Rename custom tab:', current);
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
// MANUAL CATEGORY (Simple mode)
// ═══════════════════════════════════════════════════════════

async function saveTickers() {
    const btn = document.getElementById('save-tickers-btn');
    const status = document.getElementById('save-tickers-status');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<div class="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Saving...';
    }

    try {
        const resp = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({ tickers: [...simpleTickers] })
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const result = await resp.json();
        if (result.status === 'saved') {
            if (status) status.innerHTML = '<span class="text-emerald-400">✓ Saved!</span>';
            if (typeof showToast === 'function') showToast(`${simpleTickers.length} tickers saved — refresh market to apply`, 'success');
        } else {
            if (status) status.innerHTML = '<span class="text-red-400">✗ ' + (result.error || 'Failed') + '</span>';
            if (typeof showToast === 'function') showToast('Failed to save tickers', 'error');
        }
    } catch(e) {
        if (status) status.innerHTML = '<span class="text-red-400">✗ Network error</span>';
        if (typeof showToast === 'function') showToast('Network error saving tickers', 'error');
    }

    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="save" class="w-4 h-4"></i> Save Tickers';
        lucide.createIcons();
    }
    if (status) setTimeout(() => { status.innerHTML = ''; }, 4000);
}

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
    
    sourcesCatalog.finance.forEach(item => {
        financeToggles[item.id] = item.on;
    });
    sourcesCatalog.politics.forEach(item => {
        politicsToggles[item.id] = item.on;
    });
    
    try {
        const resp = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' },
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({
                extra_feeds_finance: financeToggles,
                extra_feeds_politics: politicsToggles,
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
// ═══════════════════════════════════════════════════════════
// ADVANCED SETTINGS — DRAG & DROP PANEL REORDERING
// ═══════════════════════════════════════════════════════════

(function() {
    let draggedPanel = null;
    let placeholder = null;

    function initAdvDragDrop() {
        document.querySelectorAll('.adv-draggable').forEach(panel => {
            panel.addEventListener('dragstart', onDragStart);
            panel.addEventListener('dragend', onDragEnd);
        });
        document.querySelectorAll('.adv-drop-col').forEach(col => {
            col.addEventListener('dragover', onDragOver);
            col.addEventListener('dragleave', onDragLeave);
            col.addEventListener('drop', onDrop);
        });
        restoreAdvLayout();
    }

    function onDragStart(e) {
        draggedPanel = e.currentTarget;
        draggedPanel.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', draggedPanel.dataset.panelId);
        // Create placeholder
        placeholder = document.createElement('div');
        placeholder.className = 'adv-drop-placeholder';
    }

    function onDragEnd(e) {
        if (draggedPanel) draggedPanel.classList.remove('dragging');
        draggedPanel = null;
        document.querySelectorAll('.adv-drop-col').forEach(c => c.classList.remove('drag-over'));
        if (placeholder && placeholder.parentNode) placeholder.remove();
        placeholder = null;
    }

    function getDropIndex(col, y) {
        const panels = [...col.querySelectorAll('.adv-draggable:not(.dragging)')];
        for (let i = 0; i < panels.length; i++) {
            const rect = panels[i].getBoundingClientRect();
            if (y < rect.top + rect.height / 2) return i;
        }
        return panels.length;
    }

    function onDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const col = e.currentTarget;
        col.classList.add('drag-over');

        // Position placeholder
        const panels = [...col.querySelectorAll('.adv-draggable:not(.dragging)')];
        const idx = getDropIndex(col, e.clientY);

        if (placeholder.parentNode) placeholder.remove();
        if (idx >= panels.length) {
            col.appendChild(placeholder);
        } else {
            col.insertBefore(placeholder, panels[idx]);
        }
    }

    function onDragLeave(e) {
        // Only remove if leaving the column entirely
        if (!e.currentTarget.contains(e.relatedTarget)) {
            e.currentTarget.classList.remove('drag-over');
            if (placeholder && placeholder.parentNode === e.currentTarget) placeholder.remove();
        }
    }

    function onDrop(e) {
        e.preventDefault();
        const col = e.currentTarget;
        col.classList.remove('drag-over');
        if (!draggedPanel) return;

        const idx = getDropIndex(col, e.clientY);
        const panels = [...col.querySelectorAll('.adv-draggable:not(.dragging)')];

        if (placeholder && placeholder.parentNode) placeholder.remove();

        // Insert the panel
        if (idx >= panels.length) {
            col.appendChild(draggedPanel);
        } else {
            col.insertBefore(draggedPanel, panels[idx]);
        }

        draggedPanel.classList.remove('dragging');
        saveAdvLayout();
    }

    function saveAdvLayout() {
        const layout = {};
        document.querySelectorAll('.adv-drop-col').forEach(col => {
            const colId = col.dataset.col;
            layout[colId] = [...col.querySelectorAll('.adv-draggable')].map(p => p.dataset.panelId);
        });
        localStorage.setItem('advPanelLayout', JSON.stringify(layout));
    }

    function restoreAdvLayout() {
        const saved = localStorage.getItem('advPanelLayout');
        if (!saved) return;
        try {
            const layout = JSON.parse(saved);
            Object.entries(layout).forEach(([colId, panelIds]) => {
                const col = document.querySelector(`.adv-drop-col[data-col="${colId}"]`);
                if (!col) return;
                panelIds.forEach(pid => {
                    const panel = document.querySelector(`.adv-draggable[data-panel-id="${pid}"]`);
                    if (panel) col.appendChild(panel);
                });
            });
        } catch(e) {}
    }

    // Init when DOM ready or when settings become visible
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAdvDragDrop);
    } else {
        initAdvDragDrop();
    }
})();

// ═══════════════════════════════════════════════════════════
// PROFILE QUESTIONS — guided context builder
// ═══════════════════════════════════════════════════════════

const _PROFILE_QUESTIONS = [
    { id:'pq-grad', q:'Are you a student or working professional?', placeholder:'e.g., Final-year CPEG student, Junior engineer at KOC...' },
    { id:'pq-companies', q:'Which companies are you interested in?', placeholder:'e.g., Equate, SLB, Halliburton, KOC, KNPC...' },
    { id:'pq-invest', q:'Do you invest or trade? What assets?', placeholder:'e.g., Stocks (NVDA, AMD), Crypto (BTC, ETH), Gold...' },
    { id:'pq-skills', q:'What skills or certifications are you pursuing?', placeholder:'e.g., Python, AWS Cloud, PMP, CCNA...' },
    { id:'pq-goals', q:'What are your main goals right now?', placeholder:'e.g., Land a fresh grad job, build portfolio, save money...' },
    { id:'pq-topics', q:'Any specific topics you want to track?', placeholder:'e.g., AI breakthroughs, oil prices, Kuwait banking offers...' },
];

function _toggleProfileQs() {
    const panel = document.getElementById('profile-qs-panel');
    const chevron = document.getElementById('profile-qs-chevron');
    if (!panel) return;
    const show = panel.classList.contains('hidden');
    panel.classList.toggle('hidden', !show);
    if (chevron) chevron.style.transform = show ? 'rotate(180deg)' : '';
    if (show) _renderProfileQs();
}

function _renderProfileQs() {
    const list = document.getElementById('profile-qs-list');
    if (!list) return;
    list.innerHTML = _PROFILE_QUESTIONS.map(q => `
        <div>
            <label class="text-[11px] font-medium block mb-1" style="color:var(--text-heading);">${q.q}</label>
            <input type="text" id="${q.id}" class="w-full bg-slate-900/80 border rounded-lg px-3 py-1.5 text-xs focus:outline-none transition-all" style="border-color:var(--border-strong); color:var(--text-heading);" placeholder="${q.placeholder}" onfocus="this.style.borderColor='rgb(168,85,247)'" onblur="this.style.borderColor='var(--border-strong)'">
        </div>
    `).join('');
}

function _applyProfileQs() {
    const answers = [];
    _PROFILE_QUESTIONS.forEach(q => {
        const el = document.getElementById(q.id);
        const val = el?.value?.trim();
        if (val) answers.push(val);
    });
    if (!answers.length) {
        if (typeof showToast === 'function') showToast('Answer at least one question first', 'warning');
        return;
    }
    const ctx = document.getElementById('simple-context');
    if (!ctx) return;
    const existing = ctx.value.trim();
    const newText = answers.join('. ') + '.';
    ctx.value = existing ? existing + '\n' + newText : newText;
    simpleContext = ctx.value;
    saveSimpleState();
    // Brief highlight
    ctx.style.borderColor = 'rgb(168,85,247)';
    setTimeout(() => { ctx.style.borderColor = ''; }, 1500);
    // Collapse panel
    _toggleProfileQs();
    if (typeof showToast === 'function') showToast(`Added ${answers.length} answer${answers.length>1?'s':''} to context`, 'success');
}

/* ═══════════════════════════════════════════════════
   SETTINGS SUB-TABS
   ═══════════════════════════════════════════════════ */
var _currentSettingsTab = 'profile';

function switchSettingsTab(tabId) {
    _currentSettingsTab = tabId;
    // Update tab button state
    document.querySelectorAll('.stab').forEach(function(b) {
        b.classList.toggle('active', b.dataset.stab === tabId);
    });
    // Show/hide panels inside settings-simple that have data-stab
    var simplePanel = document.getElementById('settings-simple');
    if (simplePanel) {
        var children = simplePanel.children;
        for (var i = 0; i < children.length; i++) {
            var el = children[i];
            if (el.dataset && el.dataset.stab) {
                el.style.display = el.dataset.stab === tabId ? '' : 'none';
            }
        }
    }
    // Handle shared sections outside settings-simple
    var sp = document.getElementById('search-provider-panel');
    if (sp) sp.style.display = tabId === 'sources' ? '' : 'none';
    var ss = document.getElementById('settings-sources');
    if (ss) {
        var anyVisible = false;
        var kids = ss.children;
        for (var j = 0; j < kids.length; j++) {
            if (kids[j].dataset && kids[j].dataset.stab) {
                var show = kids[j].dataset.stab === tabId;
                kids[j].style.display = show ? '' : 'none';
                if (show) anyVisible = true;
            }
        }
        ss.style.display = anyVisible ? '' : 'none';
    }
    /* Show first-time tab tooltip */
    if (typeof _showTabTooltip === 'function') _showTabTooltip(tabId);
}

/* ═══════════════════════════════════════════════════
   TICKER PRESETS
   ═══════════════════════════════════════════════════ */
var _defaultTickerPresets = [
    { name: "All Markets", tickers: ["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","BTC-USD","ETH-USD","SOL-USD","CL=F","GC=F","SPY","QQQ"] },
    { name: "Crypto", tickers: ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","DOGE-USD"] },
    { name: "Tech", tickers: ["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA"] },
    { name: "Oil & Energy", tickers: ["CL=F","GC=F","XOM","CVX","SLB"] },
    { name: "Index ETFs", tickers: ["SPY","QQQ","DIA","IWM"] }
];

var _tickerPresets = [];

function loadTickerPresets() {
    var token = localStorage.getItem('stratos_token');
    if (!token) {
        _tickerPresets = _defaultTickerPresets.slice();
        renderTickerPresets();
        return;
    }
    fetch('/api/ticker-presets', {
        headers: { 'Authorization': 'Bearer ' + token }
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.presets && data.presets.length > 0) {
            // Normalize: ensure all tickers are strings
            _tickerPresets = data.presets.map(function(p) {
                return {
                    name: p.name,
                    tickers: (p.tickers || []).map(function(t) {
                        return typeof t === 'object' ? (t.symbol || String(t)) : String(t);
                    })
                };
            });
        } else {
            _tickerPresets = _defaultTickerPresets.slice();
            // Seed defaults on server
            _defaultTickerPresets.forEach(function(p) {
                _savePresetToServer(p.name, p.tickers);
            });
        }
        renderTickerPresets();
    })
    .catch(function() {
        _tickerPresets = _defaultTickerPresets.slice();
        renderTickerPresets();
    });
}

function renderTickerPresets() {
    var container = document.getElementById('ticker-presets-list');
    if (!container) return;
    container.innerHTML = '';
    var countEl = document.getElementById('ticker-presets-count');
    if (countEl) countEl.textContent = _tickerPresets.length + ' preset' + (_tickerPresets.length !== 1 ? 's' : '');

    _tickerPresets.forEach(function(preset) {
        var btn = document.createElement('button');
        btn.className = 'ticker-preset-btn';
        btn.innerHTML = '<span>' + _escHtml(preset.name) + '</span>' +
            '<span class="text-[10px] text-slate-600">(' + (preset.tickers ? preset.tickers.length : 0) + ')</span>' +
            '<span class="preset-delete" onclick="event.stopPropagation(); deleteTickerPreset(\'' + _escHtml(preset.name).replace(/'/g, "\\'") + '\')" title="Delete preset">&times;</span>';
        btn.onclick = function() { applyTickerPreset(preset); };
        container.appendChild(btn);
    });
}

function _escHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

async function applyTickerPreset(preset) {
    if (!preset || !preset.tickers) return;
    simpleTickers = preset.tickers.map(function(t) {
        return typeof t === 'object' ? (t.symbol || String(t)) : String(t);
    });
    renderSimpleTickers();
    saveSimpleState();
    window._settingsDirty = true;
    // Auto-save to server so market refresh uses new tickers immediately
    await saveTickers();
}

function saveTickerPreset() {
    var nameInput = document.getElementById('ticker-preset-name');
    var name = nameInput ? nameInput.value.trim() : '';
    if (!name) {
        if (typeof showToast === 'function') showToast('Enter a preset name', 'warning');
        return;
    }
    if (!simpleTickers || simpleTickers.length === 0) {
        if (typeof showToast === 'function') showToast('Add some tickers first', 'warning');
        return;
    }
    var tickers = simpleTickers.map(function(t) {
        return typeof t === 'object' ? (t.symbol || String(t)) : String(t);
    });
    // Check if name already exists — overwrite
    var existing = _tickerPresets.findIndex(function(p) { return p.name === name; });
    if (existing >= 0) {
        _tickerPresets[existing].tickers = tickers;
    } else {
        _tickerPresets.push({ name: name, tickers: tickers });
    }
    renderTickerPresets();
    _savePresetToServer(name, tickers);
    if (nameInput) nameInput.value = '';
    if (typeof showToast === 'function') showToast('Saved preset: ' + name, 'success');
}

function deleteTickerPreset(name) {
    if (!confirm('Delete preset "' + name + '"?')) return;
    _tickerPresets = _tickerPresets.filter(function(p) { return p.name !== name; });
    renderTickerPresets();
    var token = localStorage.getItem('stratos_token');
    if (token) {
        fetch('/api/ticker-presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
            body: JSON.stringify({ action: 'delete', name: name })
        }).catch(function() {});
    }
    if (typeof showToast === 'function') showToast('Deleted preset: ' + name, 'info');
}

function _savePresetToServer(name, tickers) {
    var token = localStorage.getItem('stratos_token');
    if (!token) return;
    fetch('/api/ticker-presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
        body: JSON.stringify({ action: 'save', name: name, tickers: tickers })
    }).catch(function() {});
}
