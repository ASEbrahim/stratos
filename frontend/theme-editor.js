// === THEME EDITOR — Real-time CSS Variable Tweaker ===
// Floating panel for customizing theme colors with live preview.
// Saves custom overrides to localStorage per base theme.

(function() {
    'use strict';

    // ── Undo history stack (max 20 entries) ──
    const UNDO_MAX = 20;
    const _undoStack = [];

    function _pushUndo(overrides) {
        _undoStack.push(JSON.stringify(overrides));
        if (_undoStack.length > UNDO_MAX) _undoStack.shift();
        _updateUndoBtn();
    }

    function _popUndo() {
        if (_undoStack.length === 0) return null;
        const state = JSON.parse(_undoStack.pop());
        _updateUndoBtn();
        return state;
    }

    function _updateUndoBtn() {
        const btn = document.getElementById('te-undo-btn');
        if (btn) {
            btn.disabled = _undoStack.length === 0;
            btn.title = _undoStack.length > 0
                ? `Undo (${_undoStack.length}) — Ctrl+Z`
                : 'Nothing to undo';
        }
    }

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
                { key: '--te-card-opacity',   label: 'Card Opacity',   type: 'range', min: 0.1, max: 1, step: 0.05, default: 0.45 },
                { key: '--te-panel-opacity',  label: 'Panel Opacity',  type: 'range', min: 0.1, max: 1, step: 0.05, default: 0.2 },
                { key: '--te-border-radius',  label: 'Border Radius',  type: 'range', min: 0, max: 30, step: 1, default: 25, unit: 'px' },
                { key: '--te-glow-intensity', label: 'Glow Intensity', type: 'range', min: 0, max: 1, step: 0.05, default: 0.15 },
                { key: '--te-blur',           label: 'Glass Blur ⚡',  type: 'range', min: 0, max: 20, step: 1, default: 2, unit: 'px' },
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

    // ── Layout data helpers (canvas element positions/scale/blur) ──
    const _LAYOUT_THEMES = ['cosmos','sakura','noir','rose','coffee','midnight','nebula','aurora'];
    const _LAYOUT_SUFFIXES = ['-cx','-cy','-scale','-blur','-opacity','-density','-visible'];
    const _LAYOUT_EXTRA_KEYS = [
        'stratos-sakura-tree',
        'stratos-sakura-tree-cx','stratos-sakura-tree-cy','stratos-sakura-tree-scale',
        'stratos-sakura-tree-blur','stratos-sakura-tree-opacity',
        'stratos-sakura-size','stratos-sakura-fall','stratos-sakura-wind',
        'stratos-stars-density','stratos-stars-drift','stratos-stars-brightness',
        'stratos-cosmos-preset',
    ];

    function _collectLayoutData() {
        const layout = {};
        for (const th of _LAYOUT_THEMES) {
            for (const sf of _LAYOUT_SUFFIXES) {
                const key = 'stratos-' + th + sf;
                const val = localStorage.getItem(key);
                if (val !== null) layout[key] = val;
            }
        }
        for (const key of _LAYOUT_EXTRA_KEYS) {
            const val = localStorage.getItem(key);
            if (val !== null) layout[key] = val;
        }
        return layout;
    }

    function _restoreLayoutData(layout) {
        if (!layout || typeof layout !== 'object') return;
        Object.entries(layout).forEach(([key, val]) => {
            localStorage.setItem(key, val);
        });
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
                        <button class="te-btn te-btn-reset" id="te-undo-btn" onclick="window._themeEditor.undo()" title="Nothing to undo" disabled>
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>
                            Undo
                        </button>
                        <button class="te-btn te-btn-reset" onclick="window._themeEditor.resetColors()" title="Reset colors to theme defaults">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                            Colors
                        </button>
                        <button class="te-btn te-btn-reset" onclick="window._themeEditor.resetLayout()" title="Reset element positions and layout">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/></svg>
                            Layout
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

        // ── Theme-specific element controls ──
        _buildThemeElementControls(body);
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

        const curPreset = localStorage.getItem('stratos-cosmos-preset') || 'P1';
        section.innerHTML = `
            <div class="te-group-label" style="display:flex;align-items:center;justify-content:space-between;">
                Solar System
                <div style="display:flex;gap:4px;">
                    <button class="te-cosmos-preset-btn" data-preset="P1" onclick="setCosmosPreset('P1');window._themeEditor._buildCosmosControls(document.getElementById('te-body'))" style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;cursor:pointer;border:1px solid ${curPreset==='P1'?'var(--accent)':'var(--border-strong)'};color:${curPreset==='P1'?'var(--accent)':'var(--text-muted)'};background:${curPreset==='P1'?'rgba(var(--accent-rgb,52,211,153),0.12)':'transparent'};" title="Classic flat solar system">P1</button>
                    <button class="te-cosmos-preset-btn" data-preset="P2" onclick="setCosmosPreset('P2');window._themeEditor._buildCosmosControls(document.getElementById('te-body'))" style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;cursor:pointer;border:1px solid ${curPreset==='P2'?'var(--accent)':'var(--border-strong)'};color:${curPreset==='P2'?'var(--accent)':'var(--text-muted)'};background:${curPreset==='P2'?'rgba(var(--accent-rgb,52,211,153),0.12)':'transparent'};" title="Tilted perspective solar system">P2</button>
                </div>
            </div>
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

    function _buildSakuraControls(body) {
        const old = document.getElementById('te-sakura-section');
        if (old) old.remove();
        // Always clean up tree section too (it's appended separately)
        const oldTree = document.getElementById('te-sakura-tree-section');
        if (oldTree) oldTree.remove();

        const theme = document.documentElement.getAttribute('data-theme');
        if (theme !== 'sakura') return;

        const section = document.createElement('div');
        section.id = 'te-sakura-section';
        section.className = 'te-group';

        const size = parseFloat(localStorage.getItem('stratos-sakura-size') || '1');
        const density = parseFloat(localStorage.getItem('stratos-sakura-density') || '1');
        const fall = parseFloat(localStorage.getItem('stratos-sakura-fall') || '1');
        const wind = parseFloat(localStorage.getItem('stratos-sakura-wind') || '1');
        const opacity = parseFloat(localStorage.getItem('stratos-sakura-opacity') || '1');

        section.innerHTML = `
            <div class="te-group-label">Petals</div>
            <div style="flex:1;min-width:0;">
                <label class="te-color-label" style="margin-bottom:4px;display:block;">Size</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-sakura-size" min="0.3" max="2.5" step="0.05" value="${size}" style="flex:1;" />
                    <span class="te-range-val" id="te-sakura-size-val">${size.toFixed(2)}x</span>
                </div>
                <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Density</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-sakura-density" min="0.2" max="3" step="0.1" value="${density}" style="flex:1;" />
                    <span class="te-range-val" id="te-sakura-density-val">${density.toFixed(1)}x</span>
                </div>
                <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Fall Speed</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-sakura-fall" min="0.1" max="3" step="0.1" value="${fall}" style="flex:1;" />
                    <span class="te-range-val" id="te-sakura-fall-val">${fall.toFixed(1)}x</span>
                </div>
                <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Wind</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-sakura-wind" min="0" max="3" step="0.1" value="${wind}" style="flex:1;" />
                    <span class="te-range-val" id="te-sakura-wind-val">${wind.toFixed(1)}x</span>
                </div>
                <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Opacity</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-sakura-opacity" min="0.05" max="1" step="0.05" value="${opacity}" style="flex:1;" />
                    <span class="te-range-val" id="te-sakura-opacity-val">${Math.round(opacity*100)}%</span>
                </div>
            </div>
        `;
        body.appendChild(section);

        function _wireSlider(id, valId, key, fmt) {
            const sl = section.querySelector(id);
            const vl = section.querySelector(valId);
            if (!sl) return;
            sl.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                vl.textContent = fmt(v);
                localStorage.setItem(key, v.toString());
                if (key === 'stratos-sakura-density' && typeof renderStars === 'function') renderStars();
            });
        }
        _wireSlider('#te-sakura-size', '#te-sakura-size-val', 'stratos-sakura-size', v => v.toFixed(2) + 'x');
        _wireSlider('#te-sakura-density', '#te-sakura-density-val', 'stratos-sakura-density', v => v.toFixed(1) + 'x');
        _wireSlider('#te-sakura-fall', '#te-sakura-fall-val', 'stratos-sakura-fall', v => v.toFixed(1) + 'x');
        _wireSlider('#te-sakura-wind', '#te-sakura-wind-val', 'stratos-sakura-wind', v => v.toFixed(1) + 'x');
        _wireSlider('#te-sakura-opacity', '#te-sakura-opacity-val', 'stratos-sakura-opacity', v => Math.round(v * 100) + '%');

        // ── Sakura Tree controls ──
        _buildSakuraTreeControls(body);
    }

    function _buildSakuraTreeControls(body) {
        const old = document.getElementById('te-sakura-tree-section');
        if (old) old.remove();

        const theme = document.documentElement.getAttribute('data-theme');
        if (theme !== 'sakura') return;

        const section = document.createElement('div');
        section.id = 'te-sakura-tree-section';
        section.className = 'te-group';

        const treeOn = localStorage.getItem('stratos-sakura-tree') !== 'false';
        const cx = parseFloat(localStorage.getItem('stratos-sakura-tree-cx') || '0.5');
        const cy = parseFloat(localStorage.getItem('stratos-sakura-tree-cy') || '0.55');
        const scale = parseFloat(localStorage.getItem('stratos-sakura-tree-scale') || '0.75');
        const blur = parseFloat(localStorage.getItem('stratos-sakura-tree-blur') || '0');
        const treeOpacity = parseFloat(localStorage.getItem('stratos-sakura-tree-opacity') || '1');

        section.innerHTML = `
            <div class="te-group-label" style="display:flex;align-items:center;justify-content:space-between;">
                Sakura Tree
                <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;color:var(--text-muted);">
                    <input type="checkbox" id="te-sakura-tree-toggle" ${treeOn ? 'checked' : ''} style="accent-color:var(--accent);" />
                    Show
                </label>
            </div>
            <div id="te-sakura-tree-body" style="${treeOn ? '' : 'opacity:0.4;pointer-events:none;'}">
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="flex:0 0 auto;">
                        <label class="te-color-label" style="margin-bottom:4px;display:block;">Position</label>
                        <div id="te-sktree-grid" style="
                            width:110px;height:80px;border-radius:6px;position:relative;cursor:crosshair;
                            background:var(--bg-primary);border:1px solid var(--border-strong);overflow:hidden;
                        ">
                            <div style="position:absolute;inset:0;opacity:0.06;
                                background:repeating-linear-gradient(0deg,transparent,transparent 19px,var(--text-muted) 19px,var(--text-muted) 20px),
                                repeating-linear-gradient(90deg,transparent,transparent 21px,var(--text-muted) 21px,var(--text-muted) 22px);
                            "></div>
                            <div id="te-sktree-dot" style="
                                width:10px;height:10px;border-radius:50%;position:absolute;
                                background:var(--accent);box-shadow:0 0 6px var(--accent);
                                transform:translate(-50%,-50%);pointer-events:none;
                                left:${cx * 100}%;top:${cy * 100}%;
                            "></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-top:3px;">
                            <span class="te-color-label" style="font-size:9px;opacity:0.5;" id="te-sktree-pos-label">${Math.round(cx*100)}%, ${Math.round(cy*100)}%</span>
                            <button id="te-sktree-reset" style="font-size:9px;color:var(--accent);background:none;border:none;cursor:pointer;padding:0;">reset</button>
                        </div>
                    </div>
                    <div style="flex:1;min-width:0;">
                        <label class="te-color-label" style="margin-bottom:4px;display:block;">Scale</label>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <input type="range" class="te-range-slider" id="te-sktree-scale" min="0.2" max="2" step="0.05" value="${scale}" style="flex:1;" />
                            <span class="te-range-val" id="te-sktree-scale-val">${scale.toFixed(2)}x</span>
                        </div>
                        <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Blur</label>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <input type="range" class="te-range-slider" id="te-sktree-blur" min="0" max="8" step="0.5" value="${blur}" style="flex:1;" />
                            <span class="te-range-val" id="te-sktree-blur-val">${blur.toFixed(1)}px</span>
                        </div>
                        <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Opacity</label>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <input type="range" class="te-range-slider" id="te-sktree-opacity" min="0.05" max="1" step="0.05" value="${treeOpacity}" style="flex:1;" />
                            <span class="te-range-val" id="te-sktree-opacity-val">${Math.round(treeOpacity*100)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        body.appendChild(section);

        // Wire toggle
        const toggle = section.querySelector('#te-sakura-tree-toggle');
        const treeBody = section.querySelector('#te-sakura-tree-body');
        toggle.addEventListener('change', () => {
            localStorage.setItem('stratos-sakura-tree', toggle.checked ? 'true' : 'false');
            treeBody.style.opacity = toggle.checked ? '' : '0.4';
            treeBody.style.pointerEvents = toggle.checked ? '' : 'none';
        });

        // Wire position grid drag
        const grid = section.querySelector('#te-sktree-grid');
        const dot = section.querySelector('#te-sktree-dot');
        const posLabel = section.querySelector('#te-sktree-pos-label');
        let dragging = false;
        function updatePos(e) {
            const rect = grid.getBoundingClientRect();
            const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            dot.style.left = (x * 100) + '%'; dot.style.top = (y * 100) + '%';
            localStorage.setItem('stratos-sakura-tree-cx', x.toFixed(3));
            localStorage.setItem('stratos-sakura-tree-cy', y.toFixed(3));
            posLabel.textContent = Math.round(x * 100) + '%, ' + Math.round(y * 100) + '%';
        }
        grid.addEventListener('mousedown', (e) => { dragging = true; updatePos(e); e.preventDefault(); });
        document.addEventListener('mousemove', (e) => { if (dragging) updatePos(e); });
        document.addEventListener('mouseup', () => { dragging = false; });
        grid.addEventListener('touchstart', (e) => { dragging = true; updatePos(e.touches[0]); e.preventDefault(); }, { passive: false });
        document.addEventListener('touchmove', (e) => { if (dragging) updatePos(e.touches[0]); }, { passive: false });
        document.addEventListener('touchend', () => { dragging = false; });

        // Reset button
        section.querySelector('#te-sktree-reset').addEventListener('click', () => {
            localStorage.setItem('stratos-sakura-tree-cx', '0.5');
            localStorage.setItem('stratos-sakura-tree-cy', '0.55');
            dot.style.left = '50%'; dot.style.top = '55%';
            posLabel.textContent = '50%, 55%';
        });

        // Wire sliders
        function _wireSlider(id, valId, key, fmt) {
            const sl = section.querySelector(id);
            const vl = section.querySelector(valId);
            if (!sl) return;
            sl.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                vl.textContent = fmt(v);
                localStorage.setItem(key, v.toString());
            });
        }
        _wireSlider('#te-sktree-scale', '#te-sktree-scale-val', 'stratos-sakura-tree-scale', v => v.toFixed(2) + 'x');
        _wireSlider('#te-sktree-blur', '#te-sktree-blur-val', 'stratos-sakura-tree-blur', v => v.toFixed(1) + 'px');
        _wireSlider('#te-sktree-opacity', '#te-sktree-opacity-val', 'stratos-sakura-tree-opacity', v => Math.round(v * 100) + '%');
    }

    function _buildStarFieldControls(body) {
        const old = document.getElementById('te-stars-section');
        if (old) old.remove();

        const section = document.createElement('div');
        section.id = 'te-stars-section';
        section.className = 'te-group';

        const density = parseFloat(localStorage.getItem('stratos-stars-density') || '1');
        const drift = parseFloat(localStorage.getItem('stratos-stars-drift') || '1');
        const brightness = parseFloat(localStorage.getItem('stratos-stars-brightness') || '1');

        section.innerHTML = `
            <div class="te-group-label">Star Field</div>
            <div style="flex:1;min-width:0;">
                <label class="te-color-label" style="margin-bottom:4px;display:block;">Density</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-stars-density" min="0.2" max="3" step="0.1" value="${density}" style="flex:1;" />
                    <span class="te-range-val" id="te-stars-density-val">${density.toFixed(1)}x</span>
                </div>
                <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Drift Speed</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-stars-drift" min="0" max="3" step="0.1" value="${drift}" style="flex:1;" />
                    <span class="te-range-val" id="te-stars-drift-val">${drift.toFixed(1)}x</span>
                </div>
                <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Brightness</label>
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" class="te-range-slider" id="te-stars-brightness" min="0.1" max="2" step="0.05" value="${brightness}" style="flex:1;" />
                    <span class="te-range-val" id="te-stars-brightness-val">${brightness.toFixed(2)}x</span>
                </div>
            </div>
        `;
        body.appendChild(section);

        function _wireSlider(id, valId, key, fmt) {
            const sl = section.querySelector(id);
            const vl = section.querySelector(valId);
            if (!sl) return;
            sl.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                vl.textContent = fmt(v);
                localStorage.setItem(key, v.toString());
                if (key === 'stratos-stars-density' && typeof renderStars === 'function') renderStars();
            });
        }
        _wireSlider('#te-stars-density', '#te-stars-density-val', 'stratos-stars-density', v => v.toFixed(1) + 'x');
        _wireSlider('#te-stars-drift', '#te-stars-drift-val', 'stratos-stars-drift', v => v.toFixed(1) + 'x');
        _wireSlider('#te-stars-brightness', '#te-stars-brightness-val', 'stratos-stars-brightness', v => v.toFixed(2) + 'x');
    }

    // Generic element controls: position grid + scale/blur/opacity sliders
    const _themeElementDefs = {
        noir:     { label: 'Pendulum' },
        rose:     { label: 'Rose Bloom' },
        coffee:   { label: 'Coffee Cup' },
        midnight: { label: 'Moon' },
        nebula:   { label: 'Black Hole' },
        aurora:   { label: 'Binary Stars' },
        sibyl:    { label: 'Neural Brain' },
    };

    function _buildGenericElementControls(body) {
        const old = document.getElementById('te-element-section');
        if (old) old.remove();

        const theme = document.documentElement.getAttribute('data-theme');
        const def = _themeElementDefs[theme];
        if (!def) return;

        const prefix = 'stratos-' + theme;
        const section = document.createElement('div');
        section.id = 'te-element-section';
        section.className = 'te-group';

        const elOn = localStorage.getItem(prefix + '-visible') !== 'false';
        const cx = parseFloat(localStorage.getItem(prefix + '-cx') || '0.5');
        const cy = parseFloat(localStorage.getItem(prefix + '-cy') || '0.35');
        const scale = parseFloat(localStorage.getItem(prefix + '-scale') || '1');
        const blur = parseFloat(localStorage.getItem(prefix + '-blur') || '0');
        const isSibylTheme = theme === 'sibyl';
        const defaultOpacity = isSibylTheme ? '0.5' : '1';
        const opacity = parseFloat(localStorage.getItem(prefix + '-opacity') || defaultOpacity);
        const opacityMax = 1.0; // globalAlpha caps at 1.0 — use Glow slider for extra brightness
        const glow = parseFloat(localStorage.getItem(prefix + '-glow') || '0');

        section.innerHTML = `
            <div class="te-group-label" style="display:flex;align-items:center;justify-content:space-between;">
                ${def.label}
                <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;color:var(--text-muted);">
                    <input type="checkbox" id="te-el-toggle" ${elOn ? 'checked' : ''} style="accent-color:var(--accent);" />
                    Show
                </label>
            </div>
            <div style="display:flex;gap:12px;align-items:flex-start;">
                <div style="flex:0 0 auto;">
                    <label class="te-color-label" style="margin-bottom:4px;display:block;">Position</label>
                    <div id="te-el-grid" style="
                        width:110px;height:80px;border-radius:6px;position:relative;cursor:crosshair;
                        background:var(--bg-primary);border:1px solid var(--border-strong);overflow:hidden;
                    ">
                        <div style="position:absolute;inset:0;opacity:0.06;
                            background:repeating-linear-gradient(0deg,transparent,transparent 19px,var(--text-muted) 19px,var(--text-muted) 20px),
                            repeating-linear-gradient(90deg,transparent,transparent 21px,var(--text-muted) 21px,var(--text-muted) 22px);
                        "></div>
                        <div id="te-el-dot" style="
                            width:10px;height:10px;border-radius:50%;position:absolute;
                            background:var(--accent);box-shadow:0 0 6px var(--accent);
                            transform:translate(-50%,-50%);pointer-events:none;
                            left:${cx * 100}%;top:${cy * 100}%;
                        "></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:3px;">
                        <span class="te-color-label" style="font-size:9px;opacity:0.5;" id="te-el-pos-label">${Math.round(cx*100)}%, ${Math.round(cy*100)}%</span>
                        <button onclick="window._themeEditor._resetElementPos()" style="font-size:9px;color:var(--accent);background:none;border:none;cursor:pointer;padding:0;">reset</button>
                    </div>
                </div>
                <div style="flex:1;min-width:0;">
                    <label class="te-color-label" style="margin-bottom:4px;display:block;">Scale</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-el-scale" min="0.3" max="2.5" step="0.05" value="${scale}" style="flex:1;" />
                        <span class="te-range-val" id="te-el-scale-val">${scale.toFixed(2)}x</span>
                    </div>
                    <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Blur</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-el-blur" min="0" max="8" step="0.5" value="${blur}" style="flex:1;" />
                        <span class="te-range-val" id="te-el-blur-val">${blur.toFixed(1)}px</span>
                    </div>
                    <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Opacity</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-el-opacity" min="0.05" max="${opacityMax}" step="0.05" value="${opacity}" style="flex:1;" />
                        <span class="te-range-val" id="te-el-opacity-val">${Math.round(opacity*100)}%</span>
                    </div>
                    ${isSibylTheme ? `
                    <label class="te-color-label" style="margin-bottom:2px;margin-top:6px;display:block;">Glow</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="range" class="te-range-slider" id="te-el-glow" min="0" max="3" step="0.1" value="${glow}" style="flex:1;" />
                        <span class="te-range-val" id="te-el-glow-val">${glow.toFixed(1)}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
        body.appendChild(section);

        // Wire position grid drag
        const grid = section.querySelector('#te-el-grid');
        const dot = section.querySelector('#te-el-dot');
        const posLabel = section.querySelector('#te-el-pos-label');
        let dragging = false;
        function updatePos(e) {
            const rect = grid.getBoundingClientRect();
            const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            dot.style.left = (x * 100) + '%'; dot.style.top = (y * 100) + '%';
            localStorage.setItem(prefix + '-cx', x.toFixed(3));
            localStorage.setItem(prefix + '-cy', y.toFixed(3));
            posLabel.textContent = Math.round(x * 100) + '%, ' + Math.round(y * 100) + '%';
        }
        grid.addEventListener('mousedown', (e) => { dragging = true; updatePos(e); e.preventDefault(); });
        document.addEventListener('mousemove', (e) => { if (dragging) updatePos(e); });
        document.addEventListener('mouseup', () => { dragging = false; });
        grid.addEventListener('touchstart', (e) => { dragging = true; updatePos(e.touches[0]); e.preventDefault(); }, { passive: false });
        document.addEventListener('touchmove', (e) => { if (dragging) updatePos(e.touches[0]); }, { passive: false });
        document.addEventListener('touchend', () => { dragging = false; });

        function _wireSlider(id, valId, key, fmt) {
            const sl = section.querySelector(id);
            const vl = section.querySelector(valId);
            if (!sl) return;
            sl.addEventListener('input', (e) => { const v = parseFloat(e.target.value); vl.textContent = fmt(v); localStorage.setItem(key, v.toString()); });
        }
        _wireSlider('#te-el-scale', '#te-el-scale-val', prefix + '-scale', v => v.toFixed(2) + 'x');
        _wireSlider('#te-el-blur', '#te-el-blur-val', prefix + '-blur', v => v.toFixed(1) + 'px');
        _wireSlider('#te-el-opacity', '#te-el-opacity-val', prefix + '-opacity', v => Math.round(v * 100) + '%');
        // Glow slider (Sibyl only)
        if (isSibylTheme) {
            _wireSlider('#te-el-glow', '#te-el-glow-val', prefix + '-glow', v => v.toFixed(1));
        }

        // Element visibility toggle
        const elToggle = section.querySelector('#te-el-toggle');
        const elBody = section.querySelector('#te-el-grid')?.closest('div[style*="display:flex;gap"]');
        if (elToggle) {
            elToggle.addEventListener('change', () => {
                const on = elToggle.checked;
                localStorage.setItem(prefix + '-visible', on ? 'true' : 'false');
                if (elBody) {
                    elBody.style.opacity = on ? '' : '0.4';
                    elBody.style.pointerEvents = on ? '' : 'none';
                }
            });
        }
    }

    function _buildThemeElementControls(body) {
        _buildCosmosControls(body);
        _buildSakuraControls(body);
        _buildGenericElementControls(body);
        _buildStarFieldControls(body);
    }

    function applyAndSave(varName, hex) {
        const overrides = loadOverrides();
        _pushUndo(overrides);
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

        // Sync range sliders from saved overrides (or reset to defaults)
        const overrides = loadOverrides();
        panel.querySelectorAll('.te-range-slider').forEach(slider => {
            const varName = slider.dataset.var;
            const varDef = EDITOR_GROUPS.flatMap(g => g.vars).find(v => v.key === varName);
            if (overrides[varName] !== undefined) {
                slider.value = overrides[varName];
                const valSpan = slider.parentElement.querySelector('.te-range-val');
                if (valSpan) valSpan.textContent = overrides[varName] + (varDef?.unit || '');
            } else if (varDef && varDef.default !== undefined) {
                slider.value = varDef.default;
                const valSpan = slider.parentElement.querySelector('.te-range-val');
                if (valSpan) valSpan.textContent = varDef.default + (varDef.unit || '');
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
            _buildThemeElementControls(document.getElementById('te-body'));
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

        undo() {
            const prev = _popUndo();
            if (!prev) return;
            saveOverrides(prev);
            clearAllOverrides();
            if (Object.keys(prev).length > 0) {
                applyOverrides(prev);
            }
            syncPickersToCurrentTheme();
        },

        resetColors() {
            clearOverrides();
            clearAllOverrides();
            // Re-trigger base theme to restore CSS values
            const base = document.documentElement.getAttribute('data-theme') || 'midnight';
            if (typeof setTheme === 'function') setTheme(base);
            syncPickersToCurrentTheme();
        },

        resetLayout() {
            // Clear all theme element localStorage values
            for (const th of _LAYOUT_THEMES) for (const sf of _LAYOUT_SUFFIXES) localStorage.removeItem('stratos-' + th + sf);
            for (const key of _LAYOUT_EXTRA_KEYS) localStorage.removeItem(key);
            _buildThemeElementControls(document.getElementById('te-body'));
            if (typeof renderStars === 'function') renderStars();
        },

        reset() {
            this.resetColors();
            this.resetLayout();
        },

        async savePreset() {
            const overrides = loadOverrides();
            const layout = _collectLayoutData();
            if (!Object.keys(overrides).length && !Object.keys(layout).length) return;
            const name = await stratosPrompt({ title: 'Save Theme Preset', label: 'Preset name', placeholder: 'My custom theme' });
            if (!name || !name.trim()) return;
            const presets = _getPresets();
            // Overwrite if same name exists
            const idx = presets.findIndex(p => p.name === name.trim());
            const presetData = { name: name.trim(), overrides: { ...overrides }, layout: { ...layout } };
            if (idx >= 0) {
                presets[idx] = presetData;
            } else {
                presets.push(presetData);
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
            // Restore layout data (canvas element positions/scale/blur)
            if (preset.layout) {
                _restoreLayoutData(preset.layout);
                _buildThemeElementControls(document.getElementById('te-body'));
            }
            syncPickersToCurrentTheme();
        },

        deletePreset() {
            const sel = document.getElementById('te-preset-select');
            if (!sel || sel.value === '') return;
            const presets = _getPresets();
            const idx = parseInt(sel.value);
            const preset = presets[idx];
            if (!preset) return;
            stratosConfirm(`Delete preset "${preset.name}"?`, { title: 'Delete Preset', okText: 'Delete', cancelText: 'Cancel' }).then(ok => {
                if (!ok) return;
                const presets2 = _getPresets();
                presets2.splice(idx, 1);
                _savePresets(presets2);
                _refreshPresetList();
            });
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
                    _buildThemeElementControls(document.getElementById('te-body'));
                }
            }, 50);
        },

        _buildCosmosControls(body) {
            _buildCosmosControls(body);
        },

        _resetCosmosPos() {
            localStorage.removeItem('stratos-cosmos-cx');
            localStorage.removeItem('stratos-cosmos-cy');
            const dot = document.getElementById('te-cosmos-dot');
            const lbl = document.getElementById('te-cosmos-pos-label');
            if (dot) { dot.style.left = '50%'; dot.style.top = '35%'; }
            if (lbl) lbl.textContent = '50%, 35%';
        },

        _resetElementPos() {
            const theme = document.documentElement.getAttribute('data-theme');
            const prefix = 'stratos-' + theme;
            localStorage.removeItem(prefix + '-cx');
            localStorage.removeItem(prefix + '-cy');
            const dot = document.getElementById('te-el-dot');
            const lbl = document.getElementById('te-el-pos-label');
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

    // Ctrl+Z keyboard shortcut for undo (only when theme editor is open)
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            const panel = document.getElementById('theme-editor-panel');
            if (panel && panel.classList.contains('te-open') && _undoStack.length > 0) {
                e.preventDefault();
                window._themeEditor.undo();
            }
        }
    });

})();
