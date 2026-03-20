// === INTELLIGENCE HUE WIDGET ===
// Displays a behavioral profile score (0-100) derived from user engagement patterns.
// Fetches /api/hue once on init, renders collapsed/expanded states in the sidebar.

let _hueData = null;
let _hueExpanded = localStorage.getItem('stratos-hue-expanded') === 'true';
let _hueFreshnessTimer = null;

const HUE_COLORS = {
    'Clear':   '#00e5ff',
    'Stable':  '#4caf50',
    'Clouded': '#ffc107',
    'Turbid':  '#ff9800',
    'Dark':    '#f44336',
    'No data': '#666',
};

async function initHue() {
    const container = document.getElementById('hue-widget-container');
    if (!container) return;

    // Show placeholder
    container.innerHTML = _huePlaceholder();

    try {
        const token = typeof getAuthToken === 'function' ? getAuthToken() : '';
        const res = await fetch('/api/hue', {
            headers: { 'X-Auth-Token': token }
        });
        if (!res.ok) {
            container.innerHTML = '';
            return;
        }
        _hueData = await res.json();
        _renderHue();
        _startFreshnessDecay();
    } catch (e) {
        container.innerHTML = '';
    }
}

function _huePlaceholder() {
    return `<div style="padding:8px 12px; color:var(--text-muted); font-size:11px; font-family:var(--font-mono, monospace);">...</div>`;
}

function _renderHue() {
    const container = document.getElementById('hue-widget-container');
    if (!container || !_hueData) return;

    const hue = _hueData.hue || {};
    const overall = hue.overall ?? -1;
    const label = hue.label || '—';
    const color = hue.color || HUE_COLORS[label] || '#666';
    const provisional = hue.provisional;
    const dims = hue.dimensions || {};
    const nudges = hue.nudges || [];

    // Mobile: collapsed only
    const isMobile = window.innerWidth <= 768;
    const expanded = _hueExpanded && !isMobile;

    let displayVal = overall >= 0 ? overall : '—';
    let displayLabel = provisional ? 'building...' : label;

    let html = `<div class="hue-widget" style="
        padding: 8px 12px;
        border-radius: 8px;
        background: var(--bg-card, rgba(15,21,37,0.6));
        border: 1px solid var(--border-strong, rgba(30,42,69,0.5));
        cursor: pointer;
        transition: border-color 0.3s;
        font-family: var(--font-mono, monospace);
    " onclick="_toggleHueExpand()" title="Intelligence Hue — click to ${expanded ? 'collapse' : 'expand'}">`;

    // Collapsed row
    html += `<div style="display:flex; align-items:center; gap:8px;">
        <span style="width:8px; height:8px; border-radius:50%; background:${color}; flex-shrink:0; box-shadow:0 0 6px ${color}40;"></span>
        <span style="font-size:16px; font-weight:700; color:${color}; line-height:1;">${displayVal}</span>
        <span style="font-size:11px; color:var(--text-secondary, #8896b0); letter-spacing:0.5px;">${displayLabel}</span>
    </div>`;

    // Expanded dimensions
    if (expanded) {
        html += `<div style="margin-top:10px; display:flex; flex-direction:column; gap:6px;">`;
        const dimLabels = {
            freshness: 'Freshness',
            diversity: 'Diversity',
            coverage: 'Coverage',
            signal_strength: 'Signal',
            engagement: 'Engagement',
        };
        for (const [key, lbl] of Object.entries(dimLabels)) {
            const val = dims[key] ?? 0;
            const barColor = val >= 60 ? 'var(--accent, #10b981)' : val >= 40 ? '#ffc107' : '#f44336';
            html += `<div style="display:flex; align-items:center; gap:8px;" aria-label="${lbl}: ${val}">
                <span style="font-size:10px; color:var(--text-muted, #4a5677); width:68px; text-align:right;">${lbl}</span>
                <div style="flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;">
                    <div style="width:${val}%; height:100%; background:${barColor}; border-radius:3px; transition:width 0.5s;"></div>
                </div>
                <span style="font-size:10px; color:var(--text-secondary, #8896b0); width:24px; text-align:right;">${val}</span>
            </div>`;
        }
        html += `</div>`;

        // Nudges
        if (nudges.length > 0) {
            html += `<div style="margin-top:10px; display:flex; flex-direction:column; gap:4px;">`;
            for (const nudge of nudges) {
                const actionable = nudge.type === 'freshness' || nudge.type === 'diversity' || nudge.type === 'coverage';
                html += `<div style="font-size:10px; color:var(--text-secondary, #8896b0); padding:4px 0; line-height:1.4;${actionable ? ' cursor:pointer;' : ''}"
                    ${actionable ? `onclick="event.stopPropagation(); _handleHueNudge('${nudge.type}')"` : ''}
                    ${actionable ? `onmouseenter="this.style.color='var(--accent, #10b981)'" onmouseleave="this.style.color='var(--text-secondary, #8896b0)'"` : ''}>
                    ${actionable ? '→ ' : '• '}${_escHue(nudge.message)}
                </div>`;
            }
            html += `</div>`;
        }
    }

    html += `</div>`;
    container.innerHTML = html;
}

function _escHue(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function _toggleHueExpand() {
    if (window.innerWidth <= 768) return;
    _hueExpanded = !_hueExpanded;
    localStorage.setItem('stratos-hue-expanded', _hueExpanded ? 'true' : 'false');
    _renderHue();
}

function _startFreshnessDecay() {
    if (_hueFreshnessTimer) clearInterval(_hueFreshnessTimer);
    _hueFreshnessTimer = setInterval(() => {
        if (!_hueData || !_hueData.hue || !_hueData.hue.dimensions) return;
        const dims = _hueData.hue.dimensions;
        if (dims.freshness > 0) {
            dims.freshness = Math.max(0, dims.freshness - 1);
            // Recompute overall
            _hueData.hue.overall = Math.round(
                dims.freshness * 0.30 +
                dims.diversity * 0.20 +
                dims.coverage * 0.20 +
                dims.signal_strength * 0.15 +
                dims.engagement * 0.15
            );
            // Update label
            const o = _hueData.hue.overall;
            if (o >= 80) { _hueData.hue.label = 'Clear'; _hueData.hue.color = '#00e5ff'; }
            else if (o >= 60) { _hueData.hue.label = 'Stable'; _hueData.hue.color = '#4caf50'; }
            else if (o >= 40) { _hueData.hue.label = 'Clouded'; _hueData.hue.color = '#ffc107'; }
            else if (o >= 20) { _hueData.hue.label = 'Turbid'; _hueData.hue.color = '#ff9800'; }
            else { _hueData.hue.label = 'Dark'; _hueData.hue.color = '#f44336'; }
            _renderHue();
        }
    }, 60000);
}

function _handleHueNudge(type) {
    if (type === 'freshness') {
        // Trigger scan
        const token = typeof getAuthToken === 'function' ? getAuthToken() : '';
        fetch('/api/refresh-news', { headers: { 'X-Auth-Token': token } });
        if (typeof showToast === 'function') showToast('Scan triggered', 'success');
    } else if (type === 'diversity' || type === 'coverage') {
        // Navigate to settings
        if (typeof setActive === 'function') setActive('settings');
    }
}
