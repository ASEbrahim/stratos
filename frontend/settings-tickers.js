'use strict';
// settings-tickers.js — Ticker rendering, drag-drop, presets
// Depends on: simpleTickers, simpleTimelimit, saveSimpleState(), esc(), showToast(), getAuthToken(), lucide

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

// ═══════════════════════════════════════════════════════════
// saveTickers
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
    var token = typeof getAuthToken === 'function' ? getAuthToken() : '';
    if (!token) {
        _tickerPresets = _defaultTickerPresets.slice();
        renderTickerPresets();
        return;
    }
    fetch('/api/ticker-presets', {
        headers: { 'X-Auth-Token': token }
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

async function deleteTickerPreset(name) {
    if (!(await stratosConfirm('Delete preset "' + name + '"?', { title: 'Delete Preset', okText: 'Delete', cancelText: 'Cancel' }))) return;
    _tickerPresets = _tickerPresets.filter(function(p) { return p.name !== name; });
    renderTickerPresets();
    var token = typeof getAuthToken === 'function' ? getAuthToken() : '';
    if (token) {
        fetch('/api/ticker-presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
            body: JSON.stringify({ action: 'delete', name: name })
        }).catch(function() {});
    }
    if (typeof showToast === 'function') showToast('Deleted preset: ' + name, 'info');
}

function _savePresetToServer(name, tickers) {
    var token = typeof getAuthToken === 'function' ? getAuthToken() : '';
    if (!token) return;
    fetch('/api/ticker-presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
        body: JSON.stringify({ action: 'save', name: name, tickers: tickers })
    }).catch(function() {});
}
