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
    const saveBtn = document.getElementById('cfg-save-btn');
    if (saveBtn) { saveBtn.disabled = true; saveBtn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i> Loading...'; }
    try {
        const response = await fetch('/api/config', {signal: AbortSignal.timeout(10000)});
        if (!response.ok) { console.error('/api/config GET failed:', response.status); return; }
        configData = await response.json();

        // Populate form fields (defensive — elements may not exist yet)
        // Search provider
        const provider = configData.search?.provider || 'duckduckgo';
        const provEl = document.getElementById('cfg-search-provider');
        if (provEl) provEl.value = provider;
        const serperEl = document.getElementById('cfg-serper-api-key');
        if (serperEl) serperEl.value = configData.search?.serper_api_key || '';
        // Re-lock API key field
        const apiKeyInput = document.getElementById('cfg-serper-api-key');
        if (apiKeyInput) { apiKeyInput.readOnly = true; apiKeyInput.type = 'password'; }
        _serperKeyUnlocked = false;
        if (typeof toggleSearchProviderSettings === 'function') toggleSearchProviderSettings();

        // Load search status (will also populate credits field)
        if (typeof loadSearchStatus === 'function') loadSearchStatus();

        // Timelimit
        const tlEl = document.getElementById('cfg-timelimit');
        if (tlEl) tlEl.value = configData.news?.timelimit || 'w';

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
        const roleEl = document.getElementById('cfg-profile-role');
        const locEl = document.getElementById('cfg-profile-location');
        if (roleEl) roleEl.value = configData.profile?.role || '';
        if (locEl) locEl.value = configData.profile?.location || '';
        
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
        const ctxEl = document.getElementById('cfg-profile-context');
        if (ctxEl) ctxEl.value = context;

        // UI preferences from server (auto_refresh, density, font_size, chart_type)
        const uiPrefs = configData.ui_preferences || {};
        if (uiPrefs.auto_refresh !== undefined) {
            var arEl = document.getElementById('cfg-auto-refresh');
            if (arEl) arEl.value = String(uiPrefs.auto_refresh);
            if (typeof _setAutoRefresh === 'function') _setAutoRefresh(String(uiPrefs.auto_refresh), true);
        }
        if (uiPrefs.density) {
            var densEl = document.getElementById('cfg-density');
            if (densEl) densEl.value = uiPrefs.density;
            if (typeof _applyDensity === 'function') _applyDensity(uiPrefs.density);
        }
        if (uiPrefs.font_size) {
            var fsEl = document.getElementById('cfg-font-size');
            if (fsEl) fsEl.value = uiPrefs.font_size;
            if (typeof _applyFontSize === 'function') _applyFontSize(uiPrefs.font_size);
        }
        if (uiPrefs.chart_type) {
            var ctEl = document.getElementById('cfg-chart-type');
            if (ctEl) ctEl.value = uiPrefs.chart_type;
            if (typeof setChartType === 'function') setChartType(uiPrefs.chart_type);
        }

        lucide.createIcons();

        // Initialize settings mode (simple or advanced)
        setSettingsMode(simpleSettingsMode);
        
        // Rebuild nav tabs from dynamic categories if present
        if (typeof rebuildNavFromConfig === 'function') {
            rebuildNavFromConfig();
        }
    } catch (err) {
        console.error('Failed to load config:', err);
        if (typeof showToast === 'function') showToast('Failed to load settings', 'error');
    } finally {
        if (saveBtn) { saveBtn.disabled = false; saveBtn.innerHTML = '<i data-lucide="save" class="w-4 h-4"></i> Save & Apply'; lucide.createIcons(); }
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
    const container = document.getElementById('presets-list');
    if (container) container.innerHTML = '<div class="flex items-center gap-2 py-3"><i data-lucide="loader-2" class="w-4 h-4 animate-spin text-slate-500"></i><span class="text-xs text-slate-500">Loading presets...</span></div>';
    try {
        const response = await fetch('/api/profiles', {signal: AbortSignal.timeout(10000)});
        if (!response.ok) { console.error('/api/profiles GET failed:', response.status); return; }
        const data = await response.json();

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
    if (!(await stratosConfirm(`Load preset "${name}"? This will replace your current settings.`, { title: 'Load Preset', okText: 'Load', cancelText: 'Cancel' }))) return;
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
    if (!(await stratosConfirm(`Delete preset "${name}"?`, { title: 'Delete Preset', okText: 'Delete', cancelText: 'Cancel' }))) return;
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
    if (!simplePanel || !advPanel) return; // Settings panel not in DOM yet

    if (mode === 'simple') {
        simplePanel.classList.remove('hidden');
        advPanel.classList.add('hidden');
        if (simpleBtn) { simpleBtn.style.background = 'var(--accent)'; simpleBtn.style.color = 'white'; }
        if (advBtn) { advBtn.style.background = 'transparent'; advBtn.style.color = ''; advBtn.className = advBtn.className.replace('text-white', 'text-slate-400'); }
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
        if (advBtn) { advBtn.style.background = 'var(--accent)'; advBtn.style.color = 'white'; }
        if (simpleBtn) { simpleBtn.style.background = 'transparent'; simpleBtn.style.color = ''; }
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


// === DYNAMIC CATEGORY RENDERING + MANIPULATION — moved to settings-categories.js ===


// addSimpleTicker … setSimpleTimelimit — moved to settings-tickers.js

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

async function clearSimpleSettings() {
    if (!(await stratosConfirm('This will remove your role, context, and all generated categories.', { title: 'Clear Settings', okText: 'Clear All', cancelText: 'Cancel' }))) return;
    
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
// NEWS SOURCE MANAGEMENT — moved to settings-sources.js
// (FALLBACK_CATALOG, source catalog, custom feeds, RSS suggestions)
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// MANUAL CATEGORY (Simple mode)
// ═══════════════════════════════════════════════════════════

// saveTickers() — moved to settings-tickers.js

// saveSourceToggles() — moved to settings-sources.js

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
    /* Load YouTube channels when YouTube tab shown */
    if (tabId === 'youtube' && typeof initYouTubePanel === 'function') initYouTubePanel();
    /* Load workspace stats when system tab shown */
    if (tabId === 'system' && typeof initWorkspacePanel === 'function') initWorkspacePanel();
    /* Show first-time tab tooltip */
    if (typeof _showTabTooltip === 'function') _showTabTooltip(tabId);
}

// === TICKER PRESETS — moved to settings-tickers.js ===
