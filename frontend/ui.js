// === CHART RESIZE (Lightweight Charts — ResizeObserver handles it) ===
// The chart container uses ResizeObserver internally. No manual resize handle needed.

// === THEME SYSTEM ===
const validThemes = ['midnight', 'noir', 'coffee', 'rose', 'cosmos', 'nebula', 'aurora', 'sakura'];

// UI state sync variables (declared early to avoid TDZ in init code)
var _lastLocalUiChange = 0;
let _isApplyingFromServer = false;
let _syncTimer = null;

function setTheme(theme) {
    // Clear any custom overrides from previous theme before switching
    if (window._themeEditor) {
        const style = document.documentElement.style;
        [...style].forEach(p => { if (p.startsWith('--')) style.removeProperty(p); });
    }

    _lastLocalUiChange = Date.now();
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('stratos-theme', theme);

    // Update theme labels
    document.querySelectorAll('.theme-label').forEach(d => {
        d.classList.toggle('active', d.dataset.theme === theme);
    });

    // Update brand color
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent-light').trim();
    document.querySelectorAll('#sidebar-brand i, #sidebar-brand span').forEach(el => {
        el.style.color = accent;
    });

    // Update chart colors
    updateChartTheme();

    // Handle stars for starry themes
    renderStars();

    // Re-apply mode attribute (preserve mode across theme switches)
    const mode = localStorage.getItem('stratos-theme-mode') ||
        (localStorage.getItem('stratos-dark') === 'true' ? 'dark' : 'normal');
    applyThemeMode(mode);
    updateModeToggleUI(mode);

    // Preserve stars toggle UI across theme switches
    const starsOn = localStorage.getItem('stratos-stars') === 'true';
    updateStarsToggleUI(starsOn);

    if (window._themeEditor) window._themeEditor.onThemeChange();
    if (!_isApplyingFromServer) _syncUiStateToServer();
}

function applyThemeMode(mode) {
    const html = document.documentElement;
    html.removeAttribute('data-bright');
    html.removeAttribute('data-dark');
    if (mode === 'bright') html.setAttribute('data-bright', 'true');
    else if (mode === 'dark') html.setAttribute('data-dark', 'true');
    localStorage.setItem('stratos-theme-mode', mode);
    if (!_isApplyingFromServer) _syncUiStateToServer();
}

function cycleThemeMode() {
    const current = localStorage.getItem('stratos-theme-mode') || 'normal';
    const next = current === 'normal' ? 'dark' : current === 'dark' ? 'bright' : 'normal';
    _lastLocalUiChange = Date.now();
    applyThemeMode(next);
    updateModeToggleUI(next);

    // Re-render stars (mode changes star opacity)
    renderStars();

    // Update chart colors for new variant
    updateChartTheme();
}

function updateModeToggleUI(mode) {
    const btn = document.getElementById('dark-mode-toggle');
    const icon = document.getElementById('dark-mode-icon');
    const label = document.getElementById('dark-mode-label');
    if (!btn) return;

    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();

    if (mode === 'bright') {
        btn.style.color = accent;
        btn.style.borderColor = accent + '40';
        btn.style.background = accent + '10';
        icon.textContent = '☀';
        label.textContent = 'Brighter';
    } else if (mode === 'dark') {
        btn.style.color = accent;
        btn.style.borderColor = accent + '40';
        btn.style.background = accent + '10';
        icon.textContent = '🌑';
        label.textContent = 'Deeper';
    } else {
        btn.style.color = 'var(--text-muted)';
        btn.style.borderColor = 'var(--border-strong)';
        btn.style.background = 'transparent';
        icon.textContent = '✦';
        label.textContent = 'Normal';
    }
}

function updateChartTheme() {
    const chartLine = getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim();
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#64748b';
    const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-strong').trim() || 'rgba(51,65,85,0.5)';

    if (typeof _tvChart !== 'undefined' && _tvChart && typeof _tvSeries !== 'undefined' && _tvSeries) {
        _tvChart.applyOptions({
            layout: { textColor },
            rightPriceScale: { borderColor },
            timeScale: { borderColor },
        });
        if (typeof _chartType !== 'undefined' && _chartType === 'line') {
            _tvSeries.applyOptions({ color: chartLine || '#10b981' });
        }
    }
}

function toggleStars() {
    _lastLocalUiChange = Date.now();
    const starsOn = localStorage.getItem('stratos-stars') !== 'true';
    localStorage.setItem('stratos-stars', starsOn ? 'true' : 'false');
    updateStarsToggleUI(starsOn);
    renderStars();
    if (!_isApplyingFromServer) _syncUiStateToServer();
}

function updateStarsToggleUI(starsOn) {
    const btn = document.getElementById('stars-toggle');
    const icon = document.getElementById('stars-toggle-icon');
    const label = document.getElementById('stars-toggle-label');
    if (!btn) return;

    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();

    if (starsOn) {
        btn.style.color = accent;
        btn.style.borderColor = accent + '40';
        btn.style.background = accent + '10';
        icon.textContent = '✦';
        label.textContent = 'Stars';
    } else {
        btn.style.color = 'var(--text-muted)';
        btn.style.borderColor = 'var(--border-strong)';
        btn.style.background = 'transparent';
        icon.textContent = '✧';
        label.textContent = 'Stars';
    }
}

function renderStars() {
    const canvas = document.getElementById('star-canvas');
    if (!canvas) return;

    // Clear existing stars
    canvas.innerHTML = '';

    // Stars are entirely controlled by the toggle button
    const starsOn = localStorage.getItem('stratos-stars') === 'true';

    if (!starsOn) {
        canvas.style.display = 'none';
        return;
    }

    canvas.style.display = 'block';

    // For themes with custom star colors, use them; otherwise generic white/silver
    const hasStarVars = getComputedStyle(document.documentElement).getPropertyValue('--starry-theme').trim();
    let colors;
    if (hasStarVars) {
        const color1 = getComputedStyle(document.documentElement).getPropertyValue('--star-color-1').trim();
        const color2 = getComputedStyle(document.documentElement).getPropertyValue('--star-color-2').trim();
        const color3 = getComputedStyle(document.documentElement).getPropertyValue('--star-color-3').trim();
        colors = [color1, color2, color3];
    } else {
        // Generic stars for non-starry themes: white/silver tones
        const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
        colors = [
            'rgba(255, 255, 255, 0.3)',
            'rgba(200, 210, 230, 0.25)',
            accent ? accent.replace(')', ', 0.2)').replace('rgb(', 'rgba(') : 'rgba(180, 200, 220, 0.2)'
        ];
    }

    const currentTheme = document.documentElement.getAttribute('data-theme');
    const isSakura = currentTheme === 'sakura';
    const isMobile = window.innerWidth <= 768;

    // Reduce star count on mobile for performance (80 → 20)
    const starCount = isMobile ? 20 : 80;

    for (let i = 0; i < starCount; i++) {
        const star = document.createElement('div');
        star.className = 'stratos-star';
        const size = Math.random() * 2 + 0.5;
        const color = colors[Math.floor(Math.random() * 3)];
        const x = Math.random() * 100;
        const y = Math.random() * 100;

        // Sakura: ~40% of particles are petal-shaped
        const isPetal = isSakura && Math.random() < 0.4;
        const petalStyle = isPetal
            ? `border-radius: 50% 0 50% 0; transform: rotate(${Math.floor(Math.random() * 360)}deg); width: ${size * 2.5}px; height: ${size * 1.5}px;`
            : '';

        // Skip box-shadow glow and animations on mobile
        const glow = !isMobile && size > 1.5 ? `box-shadow: 0 0 ${size * 2}px ${color};` : '';
        const anim = !isMobile && size > 1.2 ? `animation: twinkle ${2 + Math.random() * 3}s ease-in-out infinite ${Math.random() * 3}s;` : '';

        star.style.cssText = `
            width: ${size}px; height: ${size}px;
            background: ${color};
            top: ${y}%; left: ${x}%;
            ${isPetal ? petalStyle : ''}
            ${glow}
            ${anim}
        `;
        canvas.appendChild(star);
    }
}

// Star parallax on scroll — disabled on touch devices to avoid scroll jank
(function() {
    const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (isTouch) return; // Skip scroll parallax entirely on mobile/tablet

    let ticking = false;
    function onScroll() {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(function() {
            const canvas = document.getElementById('star-canvas');
            if (canvas && canvas.style.display !== 'none') {
                const el = document.getElementById('main-content') || document.querySelector('[class*="overflow-y"]');
                const scrollY = el ? el.scrollTop : window.scrollY;
                canvas.style.transform = 'translateY(' + (scrollY * -0.08) + 'px)';
            }
            ticking = false;
        });
    }
    document.addEventListener('scroll', onScroll, { capture: true, passive: true });
})();

// Load saved theme + mode + stars
const savedTheme = validThemes.includes(localStorage.getItem('stratos-theme'))
    ? localStorage.getItem('stratos-theme') : 'midnight';
const savedMode = localStorage.getItem('stratos-theme-mode') ||
    (localStorage.getItem('stratos-dark') === 'true' ? 'dark' : 'normal');
applyThemeMode(savedMode);
setTheme(savedTheme);
// Restore stars toggle
const savedStars = localStorage.getItem('stratos-stars') === 'true';
updateStarsToggleUI(savedStars);

// === UI STATE SYNC (cross-device theme/stars persistence) ===

function _syncUiStateToServer() {
    if (_syncTimer) clearTimeout(_syncTimer);
    _syncTimer = setTimeout(() => {
        const payload = {};
        const t = localStorage.getItem('stratos-theme');
        const m = localStorage.getItem('stratos-theme-mode');
        const s = localStorage.getItem('stratos-stars');
        if (t) payload.theme = t;
        if (m) payload.theme_mode = m;
        if (s) payload.stars = s;
        if (Object.keys(payload).length === 0) return;
        const token = typeof getAuthToken === 'function' ? getAuthToken() : localStorage.getItem('stratos_auth_token');
        if (!token) return;
        fetch('/api/ui-state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
            body: JSON.stringify(payload)
        }).catch(() => {});
    }, 1500);
}

function _applyUiStateFromServer(uiState) {
    if (!uiState || typeof uiState !== 'object') return;
    if (Object.keys(uiState).length === 0) return;
    _isApplyingFromServer = true;
    try {
        if (uiState.theme && validThemes.includes(uiState.theme)) {
            setTheme(uiState.theme);
        }
        if (uiState.theme_mode) {
            applyThemeMode(uiState.theme_mode);
            if (typeof updateModeToggleUI === 'function') updateModeToggleUI(uiState.theme_mode);
        }
        if (uiState.stars != null) {
            localStorage.setItem('stratos-stars', uiState.stars);
            updateStarsToggleUI(uiState.stars === 'true' || uiState.stars === true);
            renderStars();
        }
        if (uiState.avatar_image) {
            var ak = typeof _getAvatarKey === 'function' ? _getAvatarKey() : null;
            if (ak) localStorage.setItem(ak, uiState.avatar_image);
            var sa = document.getElementById('sidebar-profile-avatar');
            if (sa) { sa.textContent = ''; sa.style.backgroundImage = 'url(' + uiState.avatar_image + ')'; sa.style.backgroundSize = 'cover'; sa.style.backgroundPosition = 'center'; }
            var sp = document.getElementById('profile-avatar-preview');
            if (sp) { sp.textContent = ''; sp.style.backgroundImage = 'url(' + uiState.avatar_image + ')'; sp.style.backgroundSize = 'cover'; sp.style.backgroundPosition = 'center'; }
        }
        if (uiState.avatar) window._savedAvatarInitials = uiState.avatar;
    } finally {
        _isApplyingFromServer = false;
    }
}

// Cooldown: suppress server overrides for 3s after a local theme/mode change
// (prevents race condition where server's stale state reverts the user's click)

function _uiStateDirty(serverState) {
    if (!serverState) return false;
    // Don't override local changes that haven't synced to the server yet
    if (Date.now() - _lastLocalUiChange < 3000) return false;
    return serverState.theme !== localStorage.getItem('stratos-theme')
        || serverState.theme_mode !== localStorage.getItem('stratos-theme-mode')
        || serverState.stars !== localStorage.getItem('stratos-stars');
}

// === ALERTS BANNER ===
function renderAlerts() {
    const banner = document.getElementById('alerts-banner');
    if (!banner) return;
    
    // Only show on dashboard/summary view, hide on markets/settings/etc
    const marketsVisible = !document.getElementById('markets-panel')?.classList.contains('hidden');
    const settingsVisible = !document.getElementById('settings-panel')?.classList.contains('hidden');
    if (activeRoot !== 'dashboard' || marketsVisible || settingsVisible) {
        banner.classList.add('hidden');
        return;
    }
    
    banner.classList.remove('hidden');
    if (typeof renderMarketOverview === 'function') renderMarketOverview();
}

// === SYNC AGE INDICATORS ===
function updateSyncIndicators() {
    const timestamps = data?.timestamps || {};
    const now = new Date();
    
    // Fallback: if no separate timestamps, use meta.generated_at
    const fallback = data?.meta?.generated_at || null;
    const marketTs = timestamps.market || fallback;
    
    function formatAge(isoStr) {
        if (!isoStr) return { text: '', cls: 'text-slate-600' };
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return { text: '', cls: 'text-slate-600' };
        const mins = Math.floor((now - d) / 60000);
        if (mins < 1) return { text: 'just now', cls: 'text-emerald-500' };
        if (mins < 60) return { text: `${mins}m ago`, cls: 'text-emerald-500' };
        const hrs = Math.floor(mins / 60);
        if (hrs < 3) return { text: `${hrs}h ${mins % 60}m ago`, cls: 'text-slate-400' };
        if (hrs < 6) return { text: `${hrs}h ago`, cls: 'stale-warning' };
        return { text: `${hrs}h ago`, cls: 'stale-critical' };
    }
    
    const marketAge = formatAge(marketTs);
    const marketEl = document.getElementById('market-sync-age');
    if (marketEl) marketEl.innerHTML = `<span class="${marketAge.cls}">${marketAge.text}</span>`;
}

// === DRILL-DOWN FILTER ===
function drillFilter(text) {
    const searchInput = document.getElementById('feed-search');
    if (searchInput) {
        searchInput.value = text;
        searchInput.dispatchEvent(new Event('input'));
    }
}

// === EXPANDABLE CARD ===
function toggleCardContent(cardId) {
    const el = document.getElementById(cardId);
    const icon = document.getElementById(cardId + '-icon');
    if (el) {
        el.classList.toggle('expanded');
        if (icon) icon.style.transform = el.classList.contains('expanded') ? 'rotate(180deg)' : '';
    }
}

// === ASK AI ===
function askAI(idx) {
    const panel = document.getElementById('ai-response-' + idx);
    if (!panel) return;
    panel.classList.toggle('hidden');
    if (!panel.classList.contains('hidden')) {
        const input = document.getElementById('ai-input-' + idx);
        if (input) input.focus();
    }
}

function submitAI(idx) {
    const input = document.getElementById('ai-input-' + idx);
    const answerEl = document.getElementById('ai-answer-' + idx);
    if (!input || !answerEl) return;

    const question = input.value.trim();
    if (!question) return;

    // Get the item data from current render
    const items = window.currentItems || [];
    const item = items[idx];
    if (!item) { answerEl.textContent = 'Item not found'; return; }

    answerEl.innerHTML = '<span class="text-purple-400 animate-pulse">Thinking...</span>';

    fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question,
            title: item.title,
            content: item.content || '',
            summary: item.summary || '',
            score: item.score,
            score_reason: item.score_reason || '',
            url: item.url || '',
            category: item.category || ''
        })
    })
    .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(data => {
        if (data.error) {
            answerEl.innerHTML = `<span class="text-red-400">Error: ${data.error}</span>`;
        } else {
            answerEl.innerHTML = `<div class="text-slate-300 whitespace-pre-wrap">${esc(data.answer)}</div>`;
        }
    })
    .catch(err => {
        answerEl.innerHTML = `<span class="text-red-400">Failed: ${err.message}</span>`;
    });
}

// init() is called from index.html after all scripts are loaded

// === USER RATING ===
function toggleRating(idx) {
    const panel = document.getElementById('rating-panel-' + idx);
    if (!panel) return;
    panel.classList.toggle('hidden');
    if (!panel.classList.contains('hidden')) {
        const scoreEl = document.getElementById('rating-score-' + idx);
        if (scoreEl) scoreEl.focus();
    }
}

function submitRating(idx) {
    const scoreEl = document.getElementById('rating-score-' + idx);
    const noteEl = document.getElementById('rating-note-' + idx);
    const statusEl = document.getElementById('rating-status-' + idx);
    if (!scoreEl || !statusEl) return;

    const userScore = scoreEl.value ? parseFloat(scoreEl.value) : null;
    const note = noteEl ? noteEl.value.trim() : '';
    
    if (userScore === null && !note) {
        statusEl.innerHTML = '<span class="text-amber-400">Pick a score or write a note</span>';
        return;
    }

    const items = window.currentItems || [];
    const item = items[idx];
    if (!item) { statusEl.textContent = 'Item not found'; return; }

    statusEl.innerHTML = '<span class="text-amber-400 animate-pulse">Saving...</span>';

    fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            news_id: item.id || '',
            title: item.title || '',
            url: item.url || '',
            root: item.root || '',
            category: item.category || '',
            ai_score: item.score || 0,
            user_score: userScore,
            note: note,
            action: 'rate'
        })
    })
    .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(data => {
        if (data.status === 'ok') {
            const diff = userScore !== null ? ` (AI: ${(item.score||0).toFixed(1)} → You: ${userScore.toFixed(1)})` : '';
            statusEl.innerHTML = `<span class="text-emerald-400">✓ Saved${diff}</span>`;
        } else {
            statusEl.innerHTML = `<span class="text-red-400">Error: ${data.error || 'unknown'}</span>`;
        }
    })
    .catch(err => {
        statusEl.innerHTML = `<span class="text-red-400">Failed: ${err.message}</span>`;
    });
}

// ═══════════════════════════════════════════════════════════
// TOAST NOTIFICATION SYSTEM
// ═══════════════════════════════════════════════════════════

function showToast(message, type = 'info', duration = 3000) {
    // type: 'success' | 'error' | 'warning' | 'info'
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed bottom-6 right-6 z-[999] flex flex-col gap-2 pointer-events-none';
        document.body.appendChild(container);
    }
    
    const colors = {
        success: { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)', text: '#34d399', icon: 'check-circle' },
        error:   { bg: 'rgba(239,68,68,0.12)',   border: 'rgba(239,68,68,0.3)',   text: '#f87171', icon: 'x-circle' },
        warning: { bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.3)',  text: '#fbbf24', icon: 'alert-triangle' },
        info:    { bg: 'rgba(96,165,250,0.12)',   border: 'rgba(96,165,250,0.3)',  text: '#60a5fa', icon: 'info' }
    };
    const c = colors[type] || colors.info;
    
    const toast = document.createElement('div');
    toast.className = 'pointer-events-auto flex items-center gap-2.5 px-4 py-2.5 rounded-lg text-xs font-medium backdrop-blur-md shadow-lg transition-all duration-300';
    toast.style.cssText = `background:${c.bg}; border:1px solid ${c.border}; color:${c.text}; transform:translateX(120%); opacity:0;`;
    toast.innerHTML = `<i data-lucide="${c.icon}" class="w-3.5 h-3.5 flex-shrink-0"></i> <span>${message}</span>`;
    
    container.appendChild(toast);
    if (typeof lucide !== 'undefined') lucide.createIcons();
    
    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
    });
    
    setTimeout(() => {
        toast.style.transform = 'translateX(120%)';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}
