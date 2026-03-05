// === THEME EDITOR — Real-time CSS Variable Tweaker ===
// Floating panel for customizing theme colors with live preview.
// Saves custom overrides to localStorage per base theme.

(function() {
    'use strict';

    // ── Color groups for the editor UI ──
    const EDITOR_GROUPS = [
        {
            label: 'Backgrounds',
            icon: 'layers',
            vars: [
                { key: '--bg-primary',     label: 'Base',    type: 'hex' },
                { key: '--bg-secondary',   label: 'Secondary', type: 'hex' },
                { key: '--bg-panel-solid', label: 'Panel',   type: 'hex' },
                { key: '--sidebar-bg',     label: 'Sidebar', type: 'hex' },
            ]
        },
        {
            label: 'Accent',
            icon: 'palette',
            vars: [
                { key: '--accent',       label: 'Primary',  type: 'hex' },
                { key: '--accent-light', label: 'Light',    type: 'hex' },
                { key: '--accent-dim',   label: 'Dim',      type: 'hex' },
            ]
        },
        {
            label: 'Text',
            icon: 'type',
            vars: [
                { key: '--text-primary',   label: 'Primary',   type: 'hex' },
                { key: '--text-secondary', label: 'Secondary', type: 'hex' },
                { key: '--text-muted',     label: 'Muted',     type: 'hex' },
            ]
        },
        {
            label: 'Chart & Borders',
            icon: 'trending-up',
            vars: [
                { key: '--chart-line',    label: 'Chart Line', type: 'hex' },
                { key: '--border-strong', label: 'Border',     type: 'hex' },
            ]
        },
        {
            label: 'Effects',
            icon: 'sliders',
            vars: [
                { key: '--te-card-opacity',   label: 'Card Opacity',   type: 'range', min: 0.1, max: 1, step: 0.05, default: 0.82 },
                { key: '--te-panel-opacity',  label: 'Panel Opacity',  type: 'range', min: 0.1, max: 1, step: 0.05, default: 0.96 },
                { key: '--te-border-radius',  label: 'Border Radius',  type: 'range', min: 0, max: 24, step: 1, default: 12, unit: 'px' },
                { key: '--te-glow-intensity', label: 'Glow Intensity', type: 'range', min: 0, max: 1, step: 0.05, default: 0.15 },
                { key: '--te-blur',           label: 'Glass Blur',     type: 'range', min: 0, max: 32, step: 1, default: 12, unit: 'px' },
            ]
        },
    ];

    // ── Derived variables (auto-computed from user picks) ──
    function hexToRgb(hex) {
        hex = hex.replace('#', '');
        if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
        const n = parseInt(hex, 16);
        return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
    }

    function computeDerived(overrides) {
        const derived = {};
        // From --accent → accent-bg, accent-border
        if (overrides['--accent']) {
            const { r, g, b } = hexToRgb(overrides['--accent']);
            derived['--accent-bg'] = `rgba(${r},${g},${b},0.1)`;
            derived['--accent-border'] = `rgba(${r},${g},${b},0.2)`;
        }
        // From --chart-line → chart-fill-top
        if (overrides['--chart-line']) {
            const { r, g, b } = hexToRgb(overrides['--chart-line']);
            derived['--chart-fill-top'] = `rgba(${r},${g},${b},0.14)`;
            derived['--chart-fill-bottom'] = `rgba(${r},${g},${b},0.0)`;
        }
        // From --bg-panel-solid → bg-panel (with alpha)
        if (overrides['--bg-panel-solid']) {
            const { r, g, b } = hexToRgb(overrides['--bg-panel-solid']);
            derived['--bg-panel'] = `rgba(${r},${g},${b},0.82)`;
        }
        // From --bg-primary → bg-input, bg-hover
        if (overrides['--bg-primary']) {
            const { r, g, b } = hexToRgb(overrides['--bg-primary']);
            derived['--bg-input'] = `rgba(${r},${g},${b},0.7)`;
            derived['--bg-hover'] = `rgba(${Math.min(255,r+15)},${Math.min(255,g+15)},${Math.min(255,b+15)},0.35)`;
        }
        // From --border-strong → border (softer version)
        if (overrides['--border-strong']) {
            const { r, g, b } = hexToRgb(overrides['--border-strong']);
            derived['--border'] = `rgba(${r},${g},${b},0.12)`;
        }
        // sidebar-bg as rgba
        if (overrides['--sidebar-bg']) {
            const { r, g, b } = hexToRgb(overrides['--sidebar-bg']);
            const sidebarOp = overrides['--te-panel-opacity'] !== undefined ? parseFloat(overrides['--te-panel-opacity']) : 0.96;
            derived['--sidebar-bg'] = `rgba(${r},${g},${b},${sidebarOp})`;
        }
        // Effects sliders → CSS variables
        if (overrides['--te-card-opacity'] !== undefined) {
            const op = parseFloat(overrides['--te-card-opacity']);
            // Always compute bg-panel from current panel-solid color
            const panelHex = overrides['--bg-panel-solid'] || getCurrentHex('--bg-panel-solid');
            if (panelHex) {
                const { r, g, b } = hexToRgb(panelHex);
                derived['--bg-panel'] = `rgba(${r},${g},${b},${op})`;
            }
            derived['--card-opacity'] = String(op);
        }
        if (overrides['--te-panel-opacity'] !== undefined && !overrides['--sidebar-bg']) {
            // If sidebar-bg wasn't explicitly set, apply panel opacity to computed sidebar
            const sidebarHex = getCurrentHex('--sidebar-bg');
            if (sidebarHex) {
                const { r, g, b } = hexToRgb(sidebarHex);
                derived['--sidebar-bg'] = `rgba(${r},${g},${b},${parseFloat(overrides['--te-panel-opacity'])})`;
            }
        }
        if (overrides['--te-border-radius'] !== undefined) {
            const px = overrides['--te-border-radius'] + 'px';
            derived['--radius-lg'] = px;
            derived['--radius-xl'] = (parseFloat(overrides['--te-border-radius']) + 4) + 'px';
        }
        if (overrides['--te-glow-intensity'] !== undefined) {
            const gi = parseFloat(overrides['--te-glow-intensity']);
            derived['--glow-intensity'] = String(gi);
            // Compute accent glow shadow for glass-panel hover
            const accentHex = overrides['--accent'] || getCurrentHex('--accent');
            if (accentHex) {
                const { r, g, b } = hexToRgb(accentHex);
                derived['--panel-glow'] = `0 0 ${Math.round(gi * 40)}px rgba(${r},${g},${b},${(gi * 0.4).toFixed(2)}), 0 2px 8px rgba(0,0,0,0.15)`;
                derived['--panel-glow-border'] = `rgba(${r},${g},${b},${(gi * 0.35).toFixed(2)})`;
            }
        }
        if (overrides['--te-blur'] !== undefined) {
            derived['--glass-blur'] = overrides['--te-blur'] + 'px';
        }
        return derived;
    }

    // ── Storage helpers ──
    function getBaseName() {
        return document.documentElement.getAttribute('data-theme') || 'midnight';
    }

    function getStorageKey() {
        return `stratos-theme-custom-${getBaseName()}`;
    }

    function getPresetsKey() {
        return `stratos-theme-presets-${getBaseName()}`;
    }

    function loadOverrides() {
        try {
            const raw = localStorage.getItem(getStorageKey());
            return raw ? JSON.parse(raw) : {};
        } catch { return {}; }
    }

    function saveOverrides(overrides) {
        localStorage.setItem(getStorageKey(), JSON.stringify(overrides));
    }

    function clearOverrides() {
        localStorage.removeItem(getStorageKey());
    }

    // ── Preset storage helpers ──
    function _getPresets() {
        try {
            return JSON.parse(localStorage.getItem(getPresetsKey()) || '[]');
        } catch { return []; }
    }

    function _savePresets(presets) {
        localStorage.setItem(getPresetsKey(), JSON.stringify(presets));
    }

    function _refreshPresetList() {
        const sel = document.getElementById('te-preset-select');
        if (!sel) return;
        const presets = _getPresets();
        sel.innerHTML = '<option value="">Presets\u2026</option>' +
            presets.map((p, i) => `<option value="${i}">${p.name}</option>`).join('');
    }

    // ── Apply overrides to the document ──
    function applyOverrides(overrides) {
        // Apply user-picked values
        Object.entries(overrides).forEach(([key, val]) => {
            document.documentElement.style.setProperty(key, val);
        });
        // Apply auto-computed derived values
        const derived = computeDerived(overrides);
        Object.entries(derived).forEach(([key, val]) => {
            document.documentElement.style.setProperty(key, val);
        });
        // Update chart if available
        updateChart();
    }

    function clearAllOverrides() {
        // Remove all inline style properties
        const style = document.documentElement.style;
        const allVars = EDITOR_GROUPS.flatMap(g => g.vars.map(v => v.key));
        allVars.forEach(key => style.removeProperty(key));
        // Also clear derived
        ['--accent-bg','--accent-border','--chart-fill-top','--chart-fill-bottom',
         '--bg-panel','--bg-input','--bg-hover','--border',
         '--card-opacity','--radius-lg','--radius-xl','--glow-intensity','--glass-blur'].forEach(k => style.removeProperty(k));
        updateChart();
    }

    function updateChart() {
        const chartLine = getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim();
        if (typeof _tvChart !== 'undefined' && _tvChart && typeof _tvSeries !== 'undefined' && _tvSeries) {
            const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#64748b';
            const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-strong').trim() || 'rgba(51,65,85,0.5)';
            _tvChart.applyOptions({
                layout: { textColor: textColor },
                rightPriceScale: { borderColor: borderColor },
                timeScale: { borderColor: borderColor },
            });
            if (typeof _chartType !== 'undefined' && _chartType === 'line') {
                _tvSeries.applyOptions({ color: chartLine || '#10b981' });
            }
        }
    }

    // ── Read current computed color as hex ──
    function getCurrentHex(varName) {
        const val = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
        // If already hex, return it
        if (val.startsWith('#')) return val.length === 4 ? expandShortHex(val) : val;
        // If rgb/rgba, convert
        const m = val.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (m) {
            return '#' + [m[1],m[2],m[3]].map(x => parseInt(x).toString(16).padStart(2,'0')).join('');
        }
        // Fallback — try to use a canvas to resolve
        try {
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = 1;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = val;
            ctx.fillRect(0, 0, 1, 1);
            const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
            return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
        } catch { return '#888888'; }
    }

    function expandShortHex(hex) {
        return '#' + hex.slice(1).split('').map(c => c + c).join('');
    }

    // ── Build the editor panel DOM ──
    function buildPanel() {
        if (document.getElementById('theme-editor-panel')) return;

        const panel = document.createElement('div');
        panel.id = 'theme-editor-panel';
        panel.innerHTML = `
            <div class="te-backdrop" onclick="window._themeEditor.close()"></div>
            <div class="te-container">
                <div class="te-header">
                    <div class="te-header-left">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r="2.5"/><circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="12" r="2.5"/><circle cx="13.5" cy="17.5" r="2.5"/><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20"/></svg>
                        <span>Theme Editor</span>
                    </div>
                    <div class="te-header-right">
                        <button class="te-btn te-btn-reset" onclick="window._themeEditor.reset()" title="Reset to preset">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                            Reset
                        </button>
                        <button class="te-close" onclick="window._themeEditor.close()">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                        </button>
                    </div>
                </div>
                <div class="te-base-label">
                    Base: <span id="te-base-theme-name"></span>
                </div>
                <div class="te-body" id="te-body"></div>
                <div class="te-footer">
                    <div class="te-footer-hint">Changes save automatically per theme</div>
                    <div class="te-preset-row">
                        <select class="te-preset-select" id="te-preset-select" onchange="window._themeEditor.loadPreset(this.value)">
                            <option value="">Presets\u2026</option>
                        </select>
                        <button class="te-btn te-btn-save" onclick="window._themeEditor.savePreset()" title="Save current as preset">Save</button>
                        <button class="te-btn te-btn-del" onclick="window._themeEditor.deletePreset()" title="Delete selected preset">\u2715</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(panel);

        // Build color groups
        const body = panel.querySelector('#te-body');
        EDITOR_GROUPS.forEach(group => {
            const section = document.createElement('div');
            section.className = 'te-group';
            section.innerHTML = `<div class="te-group-label">${group.label}</div>`;
            const grid = document.createElement('div');
            grid.className = 'te-grid';

            group.vars.forEach(v => {
                const item = document.createElement('div');
                if (v.type === 'range') {
                    item.className = 'te-range-wrap';
                    item.innerHTML = `
                        <label class="te-color-label">${v.label}</label>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <input type="range" class="te-range-slider" data-var="${v.key}" min="${v.min}" max="${v.max}" step="${v.step}" value="${v.default}" />
                            <span class="te-range-val">${v.default}${v.unit||''}</span>
                        </div>
                    `;
                } else {
                    item.className = 'te-color-item';
                    item.innerHTML = `
                        <label class="te-color-label">${v.label}</label>
                        <div class="te-color-input-wrap">
                            <input type="color" class="te-color-picker" data-var="${v.key}" />
                            <input type="text" class="te-color-hex" data-var="${v.key}" maxlength="7" spellcheck="false" />
                        </div>
                    `;
                }
                grid.appendChild(item);
            });

            section.appendChild(grid);
            body.appendChild(section);
        });

        // Wire up events
        panel.querySelectorAll('.te-color-picker').forEach(picker => {
            picker.addEventListener('input', (e) => {
                const varName = e.target.dataset.var;
                const hex = e.target.value;
                // Update paired hex input
                const hexInput = panel.querySelector(`.te-color-hex[data-var="${varName}"]`);
                if (hexInput) hexInput.value = hex;
                // Apply
                applyAndSave(varName, hex);
            });
        });

        panel.querySelectorAll('.te-color-hex').forEach(input => {
            input.addEventListener('input', (e) => {
                let hex = e.target.value.trim();
                if (!hex.startsWith('#')) hex = '#' + hex;
                if (/^#[0-9a-fA-F]{6}$/.test(hex)) {
                    const varName = e.target.dataset.var;
                    // Update paired color picker
                    const picker = panel.querySelector(`.te-color-picker[data-var="${varName}"]`);
                    if (picker) picker.value = hex;
                    applyAndSave(varName, hex);
                }
            });
        });

        // Wire up range sliders
        panel.querySelectorAll('.te-range-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const varName = e.target.dataset.var;
                const val = e.target.value;
                // Update value display
                const valSpan = e.target.parentElement.querySelector('.te-range-val');
                const varDef = EDITOR_GROUPS.flatMap(g => g.vars).find(v => v.key === varName);
                if (valSpan) valSpan.textContent = val + (varDef?.unit || '');
                applyAndSave(varName, val);
            });
        });

        // ── Cosmos Solar System Controls ──
        _buildCosmosControls(body);
    }

    function _buildCosmosControls(body) {
        // Remove old if exists
        const old = document.getElementById('te-cosmos-section');
        if (old) old.remove();

        const theme = document.documentElement.getAttribute('data-theme');
        if (theme !== 'cosmos') return;

        const section = document.createElement('div');
        section.id = 'te-cosmos-section';
        section.className = 'te-group';

        const cx = parseFloat(localStorage.getItem('stratos-cosmos-cx') || '0.5');
        const cy = parseFloat(localStorage.getItem('stratos-cosmos-cy') || '0.35');
        const scale = parseFloat(localStorage.getItem('stratos-cosmos-scale') || '1');
        const blur = parseFloat(localStorage.getItem('stratos-cosmos-blur') || '0');
        const opacity = parseFloat(localStorage.getItem('stratos-cosmos-opacity') || '1');
        const density = parseFloat(localStorage.getItem('stratos-cosmos-density') || '1');

        section.innerHTML = `
            <div class="te-group-label">Solar System</div>
            <div style="display:flex;gap:12px;align-items:flex-start;">
                <div style="flex:0 0 auto;">
                    <label class="te-color-label" style="margin-bottom:4px;display:block;">Position</label>
                    <div id="te-cosmos-grid" style="
                        width:110px;height:80px;border-radius:6px;position:relative;cursor:crosshair;
                        background:var(--bg-primary);border:1px solid var(--border-strong);overflow:hidden;
                    ">
                        <div style="position:absolute;inset:0;opacity:0.06;
                            background:repeating-linear-gradient(0deg,transparent,transparent 19px,var(--text-muted) 19px,var(--text-muted) 20px),
                            repeating-linear-gradient(90deg,transparent,transparent 21px,var(--text-muted) 21px,var(--text-muted) 22px);
                        "></div>
                        <div id="te-cosmos-dot" style="
                            width:10px;height:10px;border-radius:50%;position:absolute;
                            background:var(--accent);box-shadow:0 0 6px var(--accent);
                            transform:translate(-50%,-50%);pointer-events:none;
                            left:${cx * 100}%;top:${cy * 100}%;
                        "></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:3px;">
                        <span class="te-color-label" style="font-size:9px;opacity:0.5;" id="te-cosmos-pos-label">${Math.round(cx*100)}%, ${Math.round(cy*100)}%</span>
                        <button onclick="window._themeEditor._resetCosmosPos()" style="font-size:9px;color:var(--accent);background:none;border:none;cursor:pointer;padding:0;">reset</button>
                    </div>
                </div>
                <div style="flex:1;min-width:0;">
                    <label class="te-color-label" style="margin-bottom:4px;display:block;">Scale</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-cosmos-scale" min="0.3" max="2.5" step="0.05" value="${scale}" style="flex:1;" />
                        <span class="te-range-val" id="te-cosmos-scale-val">${scale.toFixed(2)}x</span>
                    </div>
                    <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Blur</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-cosmos-blur" min="0" max="8" step="0.5" value="${blur}" style="flex:1;" />
                        <span class="te-range-val" id="te-cosmos-blur-val">${blur.toFixed(1)}px</span>
                    </div>
                    <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Opacity</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-cosmos-opacity" min="0.05" max="1" step="0.05" value="${opacity}" style="flex:1;" />
                        <span class="te-range-val" id="te-cosmos-opacity-val">${Math.round(opacity*100)}%</span>
                    </div>
                    <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Density</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-cosmos-density" min="0.2" max="2" step="0.1" value="${density}" style="flex:1;" />
                        <span class="te-range-val" id="te-cosmos-density-val">${density.toFixed(1)}x</span>
                    </div>
                </div>
            </div>
        `;
        body.appendChild(section);

        // Wire up grid drag
        const grid = section.querySelector('#te-cosmos-grid');
        const dot = section.querySelector('#te-cosmos-dot');
        const posLabel = section.querySelector('#te-cosmos-pos-label');
        let dragging = false;

        function updatePos(e) {
            const rect = grid.getBoundingClientRect();
            const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            dot.style.left = (x * 100) + '%';
            dot.style.top = (y * 100) + '%';
            localStorage.setItem('stratos-cosmos-cx', x.toFixed(3));
            localStorage.setItem('stratos-cosmos-cy', y.toFixed(3));
            posLabel.textContent = Math.round(x * 100) + '%, ' + Math.round(y * 100) + '%';
        }

        grid.addEventListener('mousedown', (e) => { dragging = true; updatePos(e); e.preventDefault(); });
        document.addEventListener('mousemove', (e) => { if (dragging) updatePos(e); });
        document.addEventListener('mouseup', () => { dragging = false; });
        // Touch support
        grid.addEventListener('touchstart', (e) => { dragging = true; updatePos(e.touches[0]); e.preventDefault(); }, { passive: false });
        document.addEventListener('touchmove', (e) => { if (dragging) updatePos(e.touches[0]); }, { passive: false });
        document.addEventListener('touchend', () => { dragging = false; });

        // Wire up sliders
        function _wireSlider(id, valId, key, fmt) {
            const sl = section.querySelector(id);
            const vl = section.querySelector(valId);
            if (!sl) return;
            sl.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                vl.textContent = fmt(v);
                localStorage.setItem(key, v.toString());
                // Density triggers star re-init
                if (key === 'stratos-cosmos-density' && typeof renderStars === 'function') renderStars();
            });
        }
        _wireSlider('#te-cosmos-scale', '#te-cosmos-scale-val', 'stratos-cosmos-scale', v => v.toFixed(2) + 'x');
        _wireSlider('#te-cosmos-blur', '#te-cosmos-blur-val', 'stratos-cosmos-blur', v => v.toFixed(1) + 'px');
        _wireSlider('#te-cosmos-opacity', '#te-cosmos-opacity-val', 'stratos-cosmos-opacity', v => Math.round(v * 100) + '%');
        _wireSlider('#te-cosmos-density', '#te-cosmos-density-val', 'stratos-cosmos-density', v => v.toFixed(1) + 'x');
    }

    function applyAndSave(varName, hex) {
        const overrides = loadOverrides();
        overrides[varName] = hex;
        saveOverrides(overrides);
        applyOverrides(overrides);
    }

    function syncPickersToCurrentTheme() {
        const panel = document.getElementById('theme-editor-panel');
        if (!panel) return;

        // Update base theme name
        const baseName = document.documentElement.getAttribute('data-theme') || 'midnight';
        const nameEl = panel.querySelector('#te-base-theme-name');
        if (nameEl) nameEl.textContent = baseName.charAt(0).toUpperCase() + baseName.slice(1);

        // Sync all pickers to current computed values
        panel.querySelectorAll('.te-color-picker').forEach(picker => {
            const hex = getCurrentHex(picker.dataset.var);
            picker.value = hex;
            // Also sync hex text input
            const hexInput = panel.querySelector(`.te-color-hex[data-var="${picker.dataset.var}"]`);
            if (hexInput) hexInput.value = hex;
        });

        // Sync range sliders from saved overrides
        const overrides = loadOverrides();
        panel.querySelectorAll('.te-range-slider').forEach(slider => {
            const varName = slider.dataset.var;
            if (overrides[varName] !== undefined) {
                slider.value = overrides[varName];
                const valSpan = slider.parentElement.querySelector('.te-range-val');
                const varDef = EDITOR_GROUPS.flatMap(g => g.vars).find(v => v.key === varName);
                if (valSpan) valSpan.textContent = overrides[varName] + (varDef?.unit || '');
            }
        });
    }

    // ── Public API ──
    window._themeEditor = {
        open() {
            buildPanel();
            // Apply any saved overrides for current theme
            const overrides = loadOverrides();
            if (Object.keys(overrides).length > 0) {
                applyOverrides(overrides);
            }
            syncPickersToCurrentTheme();
            _refreshPresetList();
            _buildCosmosControls(document.getElementById('te-body'));
            document.getElementById('theme-editor-panel').classList.add('te-open');
        },

        close() {
            const panel = document.getElementById('theme-editor-panel');
            if (panel) panel.classList.remove('te-open');
        },

        toggle() {
            const panel = document.getElementById('theme-editor-panel');
            if (panel && panel.classList.contains('te-open')) {
                this.close();
            } else {
                this.open();
            }
        },

        reset() {
            clearOverrides();
            clearAllOverrides();
            // Re-trigger base theme to restore CSS values
            const base = document.documentElement.getAttribute('data-theme') || 'midnight';
            setTheme(base);
            syncPickersToCurrentTheme();
        },

        savePreset() {
            const overrides = loadOverrides();
            if (!Object.keys(overrides).length) return;
            const name = prompt('Preset name:');
            if (!name || !name.trim()) return;
            const presets = _getPresets();
            // Overwrite if same name exists
            const idx = presets.findIndex(p => p.name === name.trim());
            if (idx >= 0) {
                presets[idx].overrides = { ...overrides };
            } else {
                presets.push({ name: name.trim(), overrides: { ...overrides } });
            }
            _savePresets(presets);
            _refreshPresetList();
            // Select the newly saved preset
            const sel = document.getElementById('te-preset-select');
            if (sel) sel.value = String(idx >= 0 ? idx : presets.length - 1);
        },

        loadPreset(indexStr) {
            if (indexStr === '' || indexStr == null) return;
            const presets = _getPresets();
            const preset = presets[parseInt(indexStr)];
            if (!preset) return;
            // Apply preset overrides as current
            saveOverrides(preset.overrides);
            clearAllOverrides();
            applyOverrides(preset.overrides);
            syncPickersToCurrentTheme();
        },

        deletePreset() {
            const sel = document.getElementById('te-preset-select');
            if (!sel || sel.value === '') return;
            const presets = _getPresets();
            const idx = parseInt(sel.value);
            const preset = presets[idx];
            if (!preset) return;
            if (!confirm(`Delete preset "${preset.name}"?`)) return;
            presets.splice(idx, 1);
            _savePresets(presets);
            _refreshPresetList();
        },

        // Called by setTheme() to re-apply overrides on theme switch
        onThemeChange() {
            setTimeout(() => {
                const overrides = loadOverrides();
                if (Object.keys(overrides).length > 0) {
                    applyOverrides(overrides);
                }
                if (document.getElementById('theme-editor-panel')?.classList.contains('te-open')) {
                    syncPickersToCurrentTheme();
                    _refreshPresetList();
                    _buildCosmosControls(document.getElementById('te-body'));
                }
            }, 50);
        },

        _resetCosmosPos() {
            localStorage.removeItem('stratos-cosmos-cx');
            localStorage.removeItem('stratos-cosmos-cy');
            const dot = document.getElementById('te-cosmos-dot');
            const lbl = document.getElementById('te-cosmos-pos-label');
            if (dot) { dot.style.left = '50%'; dot.style.top = '35%'; }
            if (lbl) lbl.textContent = '50%, 35%';
        }
    };

    // Apply saved overrides on page load
    setTimeout(() => {
        const overrides = loadOverrides();
        if (Object.keys(overrides).length > 0) {
            applyOverrides(overrides);
        }
    }, 100);

})();
