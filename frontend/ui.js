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
    updateCosmosPresetUI();

    if (typeof _initSakuraQuickControls === 'function') _initSakuraQuickControls();
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

function togglePerfMode() {
    const on = localStorage.getItem('stratos-perf-mode') !== 'true';
    localStorage.setItem('stratos-perf-mode', on ? 'true' : 'false');
    document.body.classList.toggle('perf-mode', on);
    updatePerfToggleUI(on);
    renderStars();
}

function updatePerfToggleUI(on) {
    const btn = document.getElementById('perf-toggle');
    if (!btn) return;
    const icon = document.getElementById('perf-toggle-icon');
    const label = document.getElementById('perf-toggle-label');
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    if (on) {
        btn.style.color = accent; btn.style.borderColor = accent + '40'; btn.style.background = accent + '10';
        if (icon) icon.textContent = '⚡'; if (label) label.textContent = 'Perf';
    } else {
        btn.style.color = 'var(--text-muted)'; btn.style.borderColor = 'var(--border-strong)'; btn.style.background = 'transparent';
        if (icon) icon.textContent = '⚡'; if (label) label.textContent = 'Perf';
    }
}

function toggleStars() {
    _lastLocalUiChange = Date.now();
    const starsOn = localStorage.getItem('stratos-stars') !== 'true';
    localStorage.setItem('stratos-stars', starsOn ? 'true' : 'false');
    updateStarsToggleUI(starsOn);
    updateCosmosPresetUI();
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

// ── Zen Mode — hide all panels, show only background + sidebar ──
var _zenMode = false;
function toggleZenMode() {
    _zenMode = !_zenMode;
    const mc = document.getElementById('main-content');
    const btn = document.getElementById('zen-toggle');
    const icon = document.getElementById('zen-toggle-icon');
    const label = document.getElementById('zen-toggle-label');
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();

    if (_zenMode) {
        // Hide all main-content children
        if (mc) Array.from(mc.children).forEach(c => { c.dataset.zenPrevDisplay = c.style.display; c.style.display = 'none'; });
        if (btn) { btn.style.color = accent; btn.style.borderColor = accent + '40'; btn.style.background = accent + '10'; }
        if (icon) icon.textContent = '\u25CF';
        if (label) label.textContent = 'Zen';
    } else {
        // Restore all main-content children
        if (mc) Array.from(mc.children).forEach(c => { c.style.display = c.dataset.zenPrevDisplay || ''; delete c.dataset.zenPrevDisplay; });
        if (btn) { btn.style.color = 'var(--text-muted)'; btn.style.borderColor = 'var(--border-strong)'; btn.style.background = 'transparent'; }
        if (icon) icon.textContent = '\u25CB';
        if (label) label.textContent = 'Zen';
    }
}

// Cosmos solar system preset toggle (P1 = classic, P2 = tilted)
function setCosmosPreset(preset) {
    localStorage.setItem('stratos-cosmos-preset', preset);
    updateCosmosPresetUI();
    renderStars(); // re-init with new preset
}

function updateCosmosPresetUI() {
    const wrap = document.getElementById('cosmos-preset-wrap');
    if (!wrap) return;
    const theme = document.documentElement.getAttribute('data-theme');
    const starsOn = localStorage.getItem('stratos-stars') === 'true';
    wrap.style.display = (theme === 'cosmos' && starsOn) ? 'flex' : 'none';
    const cur = localStorage.getItem('stratos-cosmos-preset') || 'P1';
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    document.querySelectorAll('.cosmos-preset-btn').forEach(btn => {
        if (btn.dataset.preset === cur) {
            btn.style.color = accent;
            btn.style.borderColor = accent + '40';
            btn.style.background = accent + '10';
        } else {
            btn.style.color = 'var(--text-muted)';
            btn.style.borderColor = 'var(--border-strong)';
            btn.style.background = 'transparent';
        }
    });
}

/* ═══ INTERACTIVE STAR CANVAS ENGINE ═══ */
var _starEngine = null; // holds running engine state for cleanup
var _starGeneration = 0; // incremented on stop — old draw() loops check this to self-terminate

function _stopStarEngine() {
    _starGeneration++; // signal any running draw() loop to exit
    if (!_starEngine) return;
    cancelAnimationFrame(_starEngine.raf);
    if (_starEngine.onMove) document.removeEventListener('mousemove', _starEngine.onMove);
    if (_starEngine.onLeave) document.removeEventListener('mouseleave', _starEngine.onLeave);
    if (_starEngine.onScroll) document.removeEventListener('scroll', _starEngine.onScroll, true);
    _starEngine = null;
}

function _getStarColors() {
    const cs = getComputedStyle(document.documentElement);
    const hasStarVars = cs.getPropertyValue('--starry-theme').trim();

    function parseRgba(str) {
        const m = str.match(/rgba?\(([^)]+)\)/);
        if (!m) return { r: 255, g: 255, b: 255, a: 0.3 };
        const p = m[1].split(',').map(s => parseFloat(s.trim()));
        return { r: p[0], g: p[1], b: p[2], a: p[3] !== undefined ? p[3] : 1 };
    }

    if (hasStarVars) {
        return {
            c1: parseRgba(cs.getPropertyValue('--star-color-1').trim()),
            c2: parseRgba(cs.getPropertyValue('--star-color-2').trim()),
            c3: parseRgba(cs.getPropertyValue('--star-color-3').trim()),
            accent: cs.getPropertyValue('--accent').trim()
        };
    }
    // Fallback for non-starry themes
    const accent = cs.getPropertyValue('--accent').trim() || '#38bdf8';
    return {
        c1: { r: 255, g: 255, b: 255, a: 0.3 },
        c2: { r: 200, g: 210, b: 230, a: 0.25 },
        c3: { r: 180, g: 200, b: 220, a: 0.2 },
        accent: accent
    };
}

function renderStars() {
    const container = document.getElementById('star-canvas');
    if (!container) return;

    // Stop any running engine
    _stopStarEngine();

    const starsOn = localStorage.getItem('stratos-stars') === 'true';
    if (!starsOn) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    const canvas = document.getElementById('star-canvas-el');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    const isMobile = window.innerWidth <= 768;
    const _perfMode = localStorage.getItem('stratos-perf-mode') === 'true';

    const theme = document.documentElement.getAttribute('data-theme');
    const isSakura = theme === 'sakura';
    const isCosmos = theme === 'cosmos';
    const isNoir = theme === 'noir';
    const isRose = theme === 'rose';
    const isCoffee = theme === 'coffee';
    const isMidnight = theme === 'midnight';
    const isNebula = theme === 'nebula';
    const isAurora = theme === 'aurora';

    const _myGen = _starGeneration; // capture generation — draw() exits if stale
    const _perfMul = _perfMode ? 0.5 : 1;
    const _cosmosDensityInit = isCosmos ? parseFloat(localStorage.getItem('stratos-cosmos-density') || '1') : 1;
    const _sakuraDensityInit = isSakura ? parseFloat(localStorage.getItem('stratos-sakura-density') || '1') : 1;
    const _starsDensityInit = parseFloat(localStorage.getItem('stratos-stars-density') || '1');
    const COUNT = Math.round((isMobile ? 30 : 200) * _cosmosDensityInit * _sakuraDensityInit * _starsDensityInit * _perfMul);
    const MOUSE_RADIUS = 150;
    const LINE_RADIUS = 120;
    const LINE_MOUSE_RANGE = _perfMode ? 0 : 240;
    const DRIFT_SPEED_BASE = 0.06;

    // Solar system data (cosmos theme only - supports P1 classic & P2 tilted)
    const _ssPreset = isCosmos ? (localStorage.getItem('stratos-cosmos-preset') || 'P1') : '';
    const _SS_TILT = 0.38, _SS_ROT = -0.15;
    // P1: hex colors for classic flat view; P2: rgb arrays for tilted perspective
    const _ssP1Planets = [
        { dist: 82,  r: 3.8, color: '#b0a090', speed: 4.15,  phase: Math.random() * Math.PI * 2 },
        { dist: 120, r: 6,   color: '#e8c77a', speed: 1.62,  phase: Math.random() * Math.PI * 2 },
        { dist: 165, r: 6.3, color: '#5b9bd5', speed: 1.0,   phase: Math.random() * Math.PI * 2 },
        { dist: 218, r: 5,   color: '#d4714a', speed: 0.53,  phase: Math.random() * Math.PI * 2 },
        { dist: 300, r: 12,  color: '#d4a55a', speed: 0.084, phase: Math.random() * Math.PI * 2 },
        { dist: 390, r: 10.5,color: '#c9b77a', speed: 0.034, phase: Math.random() * Math.PI * 2, rings: true },
        { dist: 480, r: 7.5, color: '#7ec8c8', speed: 0.012, phase: Math.random() * Math.PI * 2 },
        { dist: 570, r: 7.2, color: '#4a6ad4', speed: 0.006, phase: Math.random() * Math.PI * 2 },
    ];
    const _ssP2Planets = [
        { dist: 72,  r: 3,   color: [176,160,144], speed: 4.2,  phase: Math.random() * Math.PI * 2 },
        { dist: 108, r: 5.2, color: [232,199,122], speed: 1.65, phase: Math.random() * Math.PI * 2 },
        { dist: 150, r: 5.2, color: [70,140,210],  speed: 1.0,  phase: Math.random() * Math.PI * 2, moon: { dist: 15, r: 1.5, speed: 5 } },
        { dist: 198, r: 4.2, color: [210,100,60],  speed: 0.53, phase: Math.random() * Math.PI * 2 },
        { dist: 285, r: 11,  color: [210,165,90],  speed: 0.084,phase: Math.random() * Math.PI * 2 },
        { dist: 372, r: 9,   color: [195,178,115], speed: 0.034,phase: Math.random() * Math.PI * 2, rings: true },
        { dist: 465, r: 6.3, color: [120,195,195], speed: 0.012,phase: Math.random() * Math.PI * 2 },
        { dist: 555, r: 6,   color: [65,100,210],  speed: 0.006,phase: Math.random() * Math.PI * 2 },
    ];
    const _ssPlanets = isCosmos ? (_ssPreset === 'P2' ? _ssP2Planets : _ssP1Planets) : [];
    const _ssDensity = parseFloat(localStorage.getItem('stratos-cosmos-density') || '1');
    const _ssAsteroids = [];
    if (isCosmos) {
        const aCount = Math.round((_ssPreset === 'P2' ? 100 : 80) * _ssDensity);
        for (let ai = 0; ai < aCount; ai++) {
            _ssAsteroids.push({
                dist: (_ssPreset === 'P2' ? 232 : 252) + Math.random() * (_ssPreset === 'P2' ? 38 : 30),
                angle: Math.random() * Math.PI * 2,
                speed: 0.12 + Math.random() * 0.1,
                r: Math.random() * 0.7 + 0.2,
                a: Math.random() * (_ssPreset === 'P2' ? 0.3 : 0.4) + (_ssPreset === 'P2' ? 0.05 : 0.15),
                yOff: _ssPreset === 'P2' ? (Math.random() - 0.5) * 4 : 0
            });
        }
    }

    // Sizing
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();

    // Get theme colors
    let colors = _getStarColors();

    function pickStar() {
        const r = Math.random();
        if (r < 0.15) return { c: colors.c1, bright: true };
        if (r < 0.35) return { c: colors.c2, bright: true };
        return { c: colors.c3, bright: false };
    }

    // Initialize stars
    const stars = [];
    for (let i = 0; i < COUNT; i++) {
        const pick = pickStar();
        const petal = isSakura && Math.random() < 0.35;
        stars.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            baseX: 0, baseY: 0,
            r: pick.bright ? Math.random() * 1.8 + 0.7 : Math.random() * 1.1 + 0.3,
            a: pick.bright ? Math.random() * 0.30 + 0.35 : Math.random() * 0.25 + 0.05,
            speed: Math.random() * 0.12 + 0.03,
            phase: Math.random() * Math.PI * 2,
            cr: pick.c.r, cg: pick.c.g, cb: pick.c.b,
            isBright: pick.bright,
            petal: petal,
            petalAngle: Math.random() * Math.PI * 2,
            petalSpin: (Math.random() - 0.5) * 0.008,
            petalSize: petal ? Math.random() * 3 + 2.5 : 0,
            petalSway: petal ? Math.random() * 0.4 + 0.2 : 0,
            petalFall: petal ? Math.random() * 0.25 + 0.1 : 0
        });
        stars[i].baseX = stars[i].x;
        stars[i].baseY = stars[i].y;
    }

    // ── Sakura tree (ported from auth.js) ──
    // Tree is generated at a fixed reference center (0.5, 0.55) with fixed scale 0.75.
    // Live position/scale/blur/opacity are applied via canvas transforms in draw().
    const _SK_REF_CX = 0.5, _SK_REF_CY = 0.55, _SK_REF_SC = 0.75;
    const _skTree = { branches: [], blossoms: [], tips: [], lastW: 0, lastH: 0 };
    if (isSakura) {
        let _tSeed = 12345;
        function _tRand() { _tSeed = (_tSeed * 16807 + 0) % 2147483647; return _tSeed / 2147483647; }
        function _tRandR(a, b) { return a + _tRand() * (b - a); }

        _skTree.genFn = function(w, h) {
            _tSeed = 12345;
            const br = [], bl = [], tips = [];
            const sc = _SK_REF_SC;
            const cx = w * _SK_REF_CX;
            const baseY = h * _SK_REF_CY;
            const trunkTop = baseY - (h * 0.15) * sc;
            const lean = w * 0.004 * sc;
            br.push({ x0:cx-w*0.003*sc, y0:baseY, x1:cx+lean, y1:trunkTop, cpx:cx-w*0.001*sc, cpy:baseY-(baseY-trunkTop)*0.53, w0:w*0.005*sc, d:0 });
            br.push({ x0:cx+w*0.003*sc, y0:baseY, x1:cx+lean, y1:trunkTop, cpx:cx+w*0.002*sc, cpy:baseY-(baseY-trunkTop)*0.53, w0:w*0.005*sc, d:0 });

            function addBranch(sx, sy, angle, length, width, depth) {
                if (depth > 3 || length < h * 0.012) return;
                const curve = _tRandR(-0.3, 0.3);
                const ex = sx + Math.cos(angle) * length, ey = sy + Math.sin(angle) * length;
                const mid = 0.5 + _tRandR(-0.15, 0.15);
                const cpx = sx + Math.cos(angle + curve) * length * mid;
                const cpy = sy + Math.sin(angle + curve) * length * mid;
                br.push({ x0:sx, y0:sy, cpx, cpy, x1:ex, y1:ey, w0:width, d:depth });
                if (depth >= 1) { tips.push({ x:ex, y:ey }); bl.push({ x:ex, y:ey, r:_tRandR(6, 14-depth*2), phase:_tRand()*Math.PI*2 }); }
                const forks = depth < 2 ? Math.floor(_tRandR(2, 4)) : Math.floor(_tRandR(1, 3));
                for (let i = 0; i < forks; i++) {
                    const spread = depth < 2 ? _tRandR(0.3, 0.9) : _tRandR(0.2, 0.7);
                    const side = (i % 2 === 0 ? 1 : -1);
                    addBranch(ex, ey, angle + side * spread + _tRandR(-0.15, 0.15), length * _tRandR(0.55, 0.78), width * _tRandR(0.45, 0.65), depth + 1);
                }
            }
            const mainAngles = [-Math.PI*0.7+_tRandR(-0.1,0.1), -Math.PI*0.45+_tRandR(-0.08,0.08), -Math.PI*0.3+_tRandR(-0.05,0.05), -Math.PI*0.12+_tRandR(-0.1,0.1)];
            for (let i = 0; i < mainAngles.length; i++) addBranch(cx+lean, trunkTop, mainAngles[i], _tRandR(h*0.06, h*0.11)*sc, w*_tRandR(0.0015, 0.003)*sc, 1);
            addBranch(cx, baseY-(baseY-trunkTop)*0.53, -Math.PI*0.6+_tRandR(-0.1,0.1), h*0.04*sc, w*0.001*sc, 2);
            for (const c of bl) {
                c.dots = [];
                const count = Math.floor(_tRandR(14, 28));
                for (let i = 0; i < count; i++) {
                    const ba = _tRand()*Math.PI*2, bd = _tRand()*c.r*1.2;
                    c.dots.push({ ox:Math.cos(ba)*bd, oy:Math.sin(ba)*bd, r:_tRandR(0.6, 2.2), bright:_tRand(), ph:_tRand()*Math.PI*2 });
                }
            }
            return { branches:br, blossoms:bl, tips };
        };
        const _stData = _skTree.genFn(canvas.width, canvas.height);
        Object.assign(_skTree, _stData, { lastW: canvas.width, lastH: canvas.height });
    }

    function _drawSakuraTree(cw, ch, t) {
        if (!isSakura || localStorage.getItem('stratos-sakura-tree') === 'false') return;
        if (Math.abs(cw - _skTree.lastW) > 50 || Math.abs(ch - _skTree.lastH) > 50) {
            const d = _skTree.genFn(cw, ch);
            Object.assign(_skTree, d, { lastW: cw, lastH: ch });
        }
        ctx.save();
        // Read live settings
        const treeOpacity = parseFloat(localStorage.getItem('stratos-sakura-tree-opacity') || '1');
        const treeScale = parseFloat(localStorage.getItem('stratos-sakura-tree-scale') || '1');
        const treeCx = parseFloat(localStorage.getItem('stratos-sakura-tree-cx') || '0.5');
        const treeCy = parseFloat(localStorage.getItem('stratos-sakura-tree-cy') || '0.55');
        const treeBlur = _perfMode ? 0 : parseFloat(localStorage.getItem('stratos-sakura-tree-blur') || '0');
        // Apply transforms (same pattern as _drawThemeElement / cosmos)
        const refPx = cw * _SK_REF_CX, refPy = ch * _SK_REF_CY;
        const livePx = cw * treeCx, livePy = ch * treeCy;
        const relScale = treeScale / _SK_REF_SC;
        ctx.globalAlpha = treeOpacity;
        if (treeBlur > 0) ctx.filter = `blur(${treeBlur}px)`;
        ctx.translate(livePx, livePy);
        ctx.scale(relScale, relScale);
        ctx.translate(-refPx, -refPy);

        const windBase = Math.sin(t * 0.4) * 0.8 + Math.sin(t * 0.7) * 0.4;
        const pulse = 0.85 + 0.15 * Math.sin(t * 0.6);

        // Ambient glow (at ref center, transforms will move it)
        const ambR = Math.min(cw, ch) * 0.18;
        const amb = ctx.createRadialGradient(refPx, refPy - ch * 0.15, 0, refPx, refPy - ch * 0.15, ambR);
        amb.addColorStop(0, `rgba(240,180,200,${0.035 * pulse})`);
        amb.addColorStop(0.6, `rgba(220,160,185,${0.015 * pulse})`);
        amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(refPx, refPy - ch * 0.15, ambR, 0, Math.PI * 2); ctx.fill();

        // Branches
        const sorted = [..._skTree.branches].sort((a,b) => b.w0 - a.w0);
        for (const b of sorted) {
            const sway = windBase * b.d * 0.8;
            ctx.beginPath();
            ctx.moveTo(b.x0 + (b.d > 0 ? sway * 0.3 : 0), b.y0);
            if (b.cpx !== undefined) ctx.quadraticCurveTo(b.cpx + sway * 0.5, b.cpy, b.x1 + sway, b.y1);
            else ctx.lineTo(b.x1 + sway, b.y1);
            const alpha = b.d === 0 ? 0.22 : b.d === 1 ? 0.16 : b.d === 2 ? 0.11 : 0.06;
            ctx.strokeStyle = `rgba(60,30,45,${alpha})`;
            ctx.lineWidth = b.w0; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.stroke();
        }

        // Blossom clusters
        for (const cl of _skTree.blossoms) {
            const clX = cl.x + windBase * 1.5, clY = cl.y;
            const cg = ctx.createRadialGradient(clX, clY, 0, clX, clY, cl.r * 2.5);
            cg.addColorStop(0, `rgba(255,190,215,${0.12 * pulse})`);
            cg.addColorStop(0.5, `rgba(245,170,200,${0.05 * pulse})`);
            cg.addColorStop(1, 'transparent');
            ctx.fillStyle = cg; ctx.fillRect(clX - cl.r*2.5, clY - cl.r*2.5, cl.r*5, cl.r*5);
            for (const dot of cl.dots) {
                const dp = 0.7 + 0.3 * Math.sin(t * 1.5 + dot.ph);
                const x = clX + dot.ox, y = clY + dot.oy, r = dot.r * (0.85 + 0.15 * dp);
                const c = dot.bright > 0.6 ? [255,225,240] : dot.bright > 0.3 ? [255,200,220] : [245,175,200];
                const dg = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
                dg.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${0.15 * dp})`); dg.addColorStop(1, 'transparent');
                ctx.fillStyle = dg; ctx.fillRect(x-r*2, y-r*2, r*4, r*4);
                ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.3 + 0.3*dp})`; ctx.fill();
            }
        }
        ctx.restore();
    }

    // Shooting stars
    const shooters = [];
    let lastShooter = Date.now();
    const SHOOT_INTERVAL = 6000;

    function spawnShooter() {
        const angle = Math.random() * 0.5 + 0.25;
        const speed = Math.random() * 6 + 4;
        shooters.push({
            x: Math.random() * canvas.width * 0.7,
            y: Math.random() * canvas.height * 0.35,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 1.0,
            len: Math.random() * 40 + 25,
            cr: colors.c1.r, cg: colors.c1.g, cb: colors.c1.b
        });
    }

    // Mouse + scroll tracking
    let mouseX = -1000, mouseY = -1000;
    let scrollOffset = 0;

    function onMove(e) { mouseX = e.clientX; mouseY = e.clientY; }
    function onLeave() { mouseX = -1000; mouseY = -1000; }
    function onScroll() {
        const el = document.getElementById('main-content') || document.querySelector('[class*="overflow-y"]');
        scrollOffset = el ? el.scrollTop : window.scrollY;
    }

    if (!isTouch) {
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseleave', onLeave);
    }
    document.addEventListener('scroll', onScroll, { capture: true, passive: true });

    let raf = 0;

    // Solar system helpers (cosmos theme - P1 classic + P2 tilted)
    function _ssLighten(hex, pct) {
        const n = parseInt(hex.slice(1), 16);
        let r = (n >> 16) + pct, g = ((n >> 8) & 0xff) + pct, b = (n & 0xff) + pct;
        return `rgb(${Math.min(255,r)},${Math.min(255,g)},${Math.min(255,b)})`;
    }
    // P2 tilted helpers
    function _ssProject(cx, cy, dist, angle) {
        const x3 = Math.cos(angle) * dist, y3 = Math.sin(angle) * dist;
        const cr = Math.cos(_SS_ROT), sr = Math.sin(_SS_ROT);
        return { x: cx + x3 * cr - y3 * sr, y: cy + (x3 * sr + y3 * cr) * _SS_TILT, depth: Math.sin(angle) };
    }
    function _ssTiltedOrbit(cx, cy, dist, alpha) {
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(_SS_ROT); ctx.scale(1, _SS_TILT);
        ctx.beginPath(); ctx.arc(0, 0, dist, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(120,150,210,${alpha})`; ctx.lineWidth = 0.5; ctx.stroke(); ctx.restore();
    }
    // Sun — shared by both presets (P2 has extra detail)
    function _ssDrawSun(cx, cy, t) {
        const pulse = 1 + Math.sin(t * 0.5) * (_ssPreset === 'P2' ? 0.05 : 0.08);
        const r = (_ssPreset === 'P2' ? 30 : 33) * pulse;
        if (_ssPreset === 'P2') {
            const g4 = ctx.createRadialGradient(cx, cy, r, cx, cy, r * 5);
            g4.addColorStop(0, 'rgba(232,185,49,0.06)'); g4.addColorStop(0.5, 'rgba(232,185,49,0.015)'); g4.addColorStop(1, 'rgba(232,185,49,0)');
            ctx.fillStyle = g4; ctx.beginPath(); ctx.arc(cx, cy, r * 5, 0, Math.PI * 2); ctx.fill();
            const g3 = ctx.createRadialGradient(cx, cy, r * 0.8, cx, cy, r * 2.5);
            g3.addColorStop(0, 'rgba(255,220,100,0.3)'); g3.addColorStop(0.5, 'rgba(232,185,49,0.08)'); g3.addColorStop(1, 'rgba(232,185,49,0)');
            ctx.fillStyle = g3; ctx.beginPath(); ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2); ctx.fill();
            const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.3);
            g2.addColorStop(0, 'rgba(255,245,220,0.5)'); g2.addColorStop(1, 'rgba(232,185,49,0)');
            ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(cx, cy, r * 1.3, 0, Math.PI * 2); ctx.fill();
        } else {
            const g3 = ctx.createRadialGradient(cx, cy, r, cx, cy, r * 6);
            g3.addColorStop(0, 'rgba(232,185,49,0.12)'); g3.addColorStop(1, 'rgba(232,185,49,0)');
            ctx.fillStyle = g3; ctx.beginPath(); ctx.arc(cx, cy, r * 6, 0, Math.PI * 2); ctx.fill();
            const g2 = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r * 2.5);
            g2.addColorStop(0, 'rgba(255,210,80,0.4)'); g2.addColorStop(1, 'rgba(232,185,49,0)');
            ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2); ctx.fill();
        }
        const g1 = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g1.addColorStop(0, '#fff8e0'); g1.addColorStop(0.2, _ssPreset === 'P2' ? '#ffe566' : '#ffd54f');
        g1.addColorStop(0.5, _ssPreset === 'P2' ? '#f0c030' : '#ffd54f');
        g1.addColorStop(0.8, '#e8b931'); g1.addColorStop(1, _ssPreset === 'P2' ? '#c08520' : '#c98a1a');
        ctx.fillStyle = g1; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
        if (_ssPreset === 'P2') {
            for (let i = 0; i < 3; i++) {
                const a = t * 0.3 + i * 2.1, hx = cx + Math.cos(a) * r * 0.4, hy = cy + Math.sin(a) * r * 0.4;
                const hg = ctx.createRadialGradient(hx, hy, 0, hx, hy, r * 0.3);
                hg.addColorStop(0, 'rgba(255,255,230,0.3)'); hg.addColorStop(1, 'rgba(255,255,230,0)');
                ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(hx, hy, r * 0.3, 0, Math.PI * 2); ctx.fill();
            }
        }
    }
    // P1 planet (hex colors, flat orbits)
    function _ssDrawPlanetP1(cx, cy, p, angle) {
        const px = cx + Math.cos(angle) * p.dist, py = cy + Math.sin(angle) * p.dist;
        const glow = ctx.createRadialGradient(px, py, 0, px, py, p.r * 3);
        glow.addColorStop(0, p.color + '40'); glow.addColorStop(1, p.color + '00');
        ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(px, py, p.r * 3, 0, Math.PI * 2); ctx.fill();
        const g = ctx.createRadialGradient(px - p.r * 0.3, py - p.r * 0.3, 0, px, py, p.r);
        g.addColorStop(0, _ssLighten(p.color, 30)); g.addColorStop(1, p.color);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, p.r, 0, Math.PI * 2); ctx.fill();
        if (p.rings) {
            ctx.save(); ctx.translate(px, py); ctx.scale(1, 0.35);
            ctx.beginPath(); ctx.arc(0, 0, p.r * 2, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(201,183,122,0.5)'; ctx.lineWidth = 2; ctx.stroke();
            ctx.beginPath(); ctx.arc(0, 0, p.r * 2.5, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(201,183,122,0.25)'; ctx.lineWidth = 1.5; ctx.stroke();
            ctx.restore();
        }
    }
    // P2 planet (rgb array colors, depth-based sizing)
    function _ssDrawPlanetP2(px, py, p, depth) {
        const c = p.color, ds = 1 + depth * 0.08, r = p.r * ds;
        const glow = ctx.createRadialGradient(px, py, 0, px, py, r * 4);
        glow.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},0.18)`); glow.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0)`);
        ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(px, py, r * 4, 0, Math.PI * 2); ctx.fill();
        const bg = ctx.createRadialGradient(px - r * 0.35, py - r * 0.35, 0, px + r * 0.2, py + r * 0.2, r * 1.2);
        bg.addColorStop(0, `rgba(${Math.min(255,c[0]+50)},${Math.min(255,c[1]+50)},${Math.min(255,c[2]+50)},1)`);
        bg.addColorStop(0.6, `rgba(${c[0]},${c[1]},${c[2]},1)`);
        bg.addColorStop(1, `rgba(${Math.max(0,c[0]-40)},${Math.max(0,c[1]-40)},${Math.max(0,c[2]-40)},1)`);
        ctx.fillStyle = bg; ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
        if (p.rings) {
            ctx.save(); ctx.translate(px, py); ctx.rotate(_SS_ROT * 0.5); ctx.scale(1, 0.28);
            [{ m: 1.8, a: 0.4, w: 2.5 }, { m: 2.15, a: 0.25, w: 2 }, { m: 2.5, a: 0.12, w: 1.2 }].forEach(ri => {
                ctx.beginPath(); ctx.arc(0, 0, r * ri.m, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${ri.a})`; ctx.lineWidth = ri.w; ctx.stroke();
            });
            ctx.restore();
        }
    }

    // ── Noir: Pendulum ──
    const _noirTrail = [];
    const _noirRipples = [];
    function _noirDrawGear(gcx, gcy, innerR, outerR, teeth, rotation, alpha) {
        ctx.save(); ctx.translate(gcx, gcy); ctx.rotate(rotation);
        ctx.beginPath();
        const step = Math.PI * 2 / teeth;
        for (let i = 0; i < teeth; i++) {
            const a = i * step;
            ctx.lineTo(Math.cos(a) * innerR, Math.sin(a) * innerR);
            ctx.lineTo(Math.cos(a + step * 0.1) * outerR, Math.sin(a + step * 0.1) * outerR);
            ctx.lineTo(Math.cos(a + step * 0.4) * outerR, Math.sin(a + step * 0.4) * outerR);
            ctx.lineTo(Math.cos(a + step * 0.5) * innerR, Math.sin(a + step * 0.5) * innerR);
        }
        ctx.closePath();
        ctx.strokeStyle = `rgba(167,139,250,${alpha})`; ctx.lineWidth = 1; ctx.stroke();
        ctx.beginPath(); ctx.arc(0, 0, innerR * 0.3, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(196,181,253,${alpha * 0.7})`; ctx.lineWidth = 0.8; ctx.stroke();
        ctx.restore();
    }
    function _noirDrawPendulum(cx, cy, t) {
        const pendulumLen = 90, maxAngle = 0.4;
        const angle = maxAngle * Math.sin(t * 1.2);
        const angVel = maxAngle * 1.2 * Math.cos(t * 1.2);
        const pivotX = cx, pivotY = cy - 45;
        const bobX = pivotX + Math.sin(angle) * pendulumLen;
        const bobY = pivotY + Math.cos(angle) * pendulumLen;
        const mdx = mouseX - pivotX, mdy = mouseY - pivotY;
        const mDist = Math.sqrt(mdx * mdx + mdy * mdy);
        const mBoost = mDist < 200 ? (1 - mDist / 200) * 0.4 : 0;
        const clockR = 60;
        for (let i = 0; i < 12; i++) {
            const a = (i / 12) * Math.PI * 2 - Math.PI / 2;
            const iR = clockR - 7, oR = clockR;
            const ta = 0.18 + 0.1 * Math.sin(t * 2 + i * 0.5) + mBoost;
            ctx.beginPath(); ctx.moveTo(pivotX + Math.cos(a) * iR, pivotY + Math.sin(a) * iR);
            ctx.lineTo(pivotX + Math.cos(a) * oR, pivotY + Math.sin(a) * oR);
            ctx.strokeStyle = `rgba(196,181,253,${ta})`; ctx.lineWidth = i % 3 === 0 ? 2 : 1; ctx.stroke();
            ctx.beginPath(); ctx.arc(pivotX + Math.cos(a) * oR, pivotY + Math.sin(a) * oR, 1.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(167,139,250,${ta + 0.05})`; ctx.fill();
        }
        for (let i = 0; i < 60; i++) {
            if (i % 5 === 0) continue;
            const a = (i / 60) * Math.PI * 2 - Math.PI / 2;
            const sta = 0.06 + 0.04 * Math.sin(t * 1.5 + i * 0.3) + mBoost * 0.5;
            ctx.beginPath(); ctx.moveTo(pivotX + Math.cos(a) * (clockR - 3), pivotY + Math.sin(a) * (clockR - 3));
            ctx.lineTo(pivotX + Math.cos(a) * clockR, pivotY + Math.sin(a) * clockR);
            ctx.strokeStyle = `rgba(139,92,246,${sta})`; ctx.lineWidth = 0.5; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(pivotX, pivotY, clockR + 2, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(167,139,250,${0.08 + mBoost * 0.5})`; ctx.lineWidth = 0.6; ctx.stroke();
        const gs = t * 0.8;
        _noirDrawGear(pivotX - 16, pivotY - 8, 11, 15, 8, gs, 0.3 + mBoost);
        _noirDrawGear(pivotX + 14, pivotY - 6, 8, 11, 6, -gs * 1.33, 0.25 + mBoost);
        _noirDrawGear(pivotX + 4, pivotY - 20, 6, 9, 7, gs * 1.14, 0.2 + mBoost);
        _noirDrawGear(pivotX - 8, pivotY + 12, 5, 8, 5, -gs * 1.6, 0.18 + mBoost);
        if (Math.abs(angVel) < 0.08 && _noirRipples.length < 6) {
            if (!_noirRipples.length || _noirRipples[_noirRipples.length - 1].radius > 12)
                _noirRipples.push({ x: bobX, y: bobY, radius: 4, alpha: 0.2 });
        }
        for (let i = _noirRipples.length - 1; i >= 0; i--) {
            const r = _noirRipples[i]; r.radius += 0.6; r.alpha -= 0.002;
            if (r.alpha <= 0) { _noirRipples.splice(i, 1); continue; }
            ctx.beginPath(); ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(167,139,250,${r.alpha})`; ctx.lineWidth = 0.8; ctx.stroke();
        }
        _noirTrail.push({ x: bobX, y: bobY });
        if (_noirTrail.length > 16) _noirTrail.shift();
        for (let i = 0; i < _noirTrail.length - 1; i++) {
            const p = _noirTrail[i];
            const a = (i / _noirTrail.length) * 0.2;
            const sz = 4 + (i / _noirTrail.length) * 5;
            const gl = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, sz);
            gl.addColorStop(0, `rgba(139,92,246,${a})`); gl.addColorStop(1, 'rgba(139,92,246,0)');
            ctx.beginPath(); ctx.arc(p.x, p.y, sz, 0, Math.PI * 2); ctx.fillStyle = gl; ctx.fill();
        }
        ctx.beginPath(); ctx.moveTo(pivotX, pivotY); ctx.lineTo(bobX, bobY);
        ctx.strokeStyle = `rgba(196,181,253,${0.4 + mBoost})`; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pivotX, pivotY); ctx.lineTo(bobX, bobY);
        ctx.strokeStyle = `rgba(167,139,250,${0.1 + mBoost * 0.3})`; ctx.lineWidth = 5; ctx.stroke();
        ctx.beginPath(); ctx.arc(pivotX, pivotY, 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(196,181,253,${0.6 + mBoost})`; ctx.fill();
        const bp = 0.7 + 0.3 * Math.sin(t * 3);
        const bg2 = ctx.createRadialGradient(bobX, bobY, 0, bobX, bobY, 35);
        bg2.addColorStop(0, `rgba(167,139,250,${0.2})`); bg2.addColorStop(0.5, `rgba(139,92,246,0.08)`); bg2.addColorStop(1, 'transparent');
        ctx.beginPath(); ctx.arc(bobX, bobY, 35, 0, Math.PI * 2); ctx.fillStyle = bg2; ctx.fill();
        const bg1 = ctx.createRadialGradient(bobX, bobY, 0, bobX, bobY, 14);
        bg1.addColorStop(0, `rgba(232,224,255,${0.55 * bp})`); bg1.addColorStop(0.4, `rgba(167,139,250,${0.35 * bp})`); bg1.addColorStop(1, 'rgba(139,92,246,0)');
        ctx.beginPath(); ctx.arc(bobX, bobY, 14, 0, Math.PI * 2); ctx.fillStyle = bg1; ctx.fill();
    }

    // ── Rose: Bloom ──
    function _roseDrawBloom(cx, cy, t) {
        const breathe = 1 + 0.06 * Math.sin(t * 0.8);
        const rotation = t * 0.05;
        const amb = ctx.createRadialGradient(cx, cy, 0, cx, cy, 180);
        amb.addColorStop(0, 'rgba(244,63,94,0.08)'); amb.addColorStop(0.3, 'rgba(251,113,133,0.04)');
        amb.addColorStop(0.6, 'rgba(225,29,72,0.015)'); amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(cx, cy, 180, 0, Math.PI * 2); ctx.fill();
        for (let i = 0; i < 20; i++) {
            const age = (t * 0.3 + i * 0.5) % 5;
            const px2 = cx + Math.sin(t * 0.2 + i * 1.7) * (10 + age * 8);
            const py2 = cy - age * 25;
            const pa = Math.max(0, 0.3 * (1 - age / 5));
            if (pa < 0.01) continue;
            ctx.globalAlpha = pa; ctx.fillStyle = 'rgba(253,164,175,0.8)';
            ctx.beginPath(); ctx.arc(px2, py2, 1 + (1 - age / 5), 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
        for (let layer = 4; layer >= 0; layer--) {
            const layerScale = (0.5 + layer * 0.12) * breathe;
            const petalCount = 5 + layer;
            const layerRot = rotation + layer * 0.15;
            const openFactor = 0.7 + layer * 0.08;
            for (let i = 0; i < petalCount; i++) {
                const ang = layerRot + (i / petalCount) * Math.PI * 2;
                const pr = 30 * layerScale * openFactor;
                const tipX = cx + Math.cos(ang) * pr, tipY = cy + Math.sin(ang) * pr;
                const cp1A = ang - 0.3, cp2A = ang + 0.3, cpD = pr * 0.7;
                ctx.beginPath(); ctx.moveTo(cx, cy);
                ctx.quadraticCurveTo(cx + Math.cos(cp1A) * cpD, cy + Math.sin(cp1A) * cpD, tipX, tipY);
                ctx.quadraticCurveTo(cx + Math.cos(cp2A) * cpD, cy + Math.sin(cp2A) * cpD, cx, cy);
                const bright = 0.5 + 0.5 * Math.sin(t * 0.5 + i + layer);
                const r = layer < 2 ? 244 : 251, g = layer < 2 ? 63 + bright * 30 : 113, b = layer < 2 ? 94 : 133;
                const alpha = (0.12 + layer * 0.04) * bright;
                ctx.fillStyle = `rgba(${r},${g|0},${b},${alpha})`; ctx.fill();
                ctx.strokeStyle = `rgba(253,164,175,${alpha * 0.5})`; ctx.lineWidth = 0.5; ctx.stroke();
            }
        }
        const coreG = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12 * breathe);
        coreG.addColorStop(0, 'rgba(255,240,243,0.5)'); coreG.addColorStop(0.5, 'rgba(244,63,94,0.3)'); coreG.addColorStop(1, 'rgba(225,29,72,0)');
        ctx.fillStyle = coreG; ctx.beginPath(); ctx.arc(cx, cy, 12 * breathe, 0, Math.PI * 2); ctx.fill();
    }

    // ── Coffee: Cup with steam ──
    const _coffeeSteam = [];
    const _coffeeBeans = [];
    if (isCoffee) {
        for (let s = 0; s < 4; s++) {
            for (let i = 0; i < 25; i++) {
                _coffeeSteam.push({ stream: s, streams: 4, life: Math.floor(Math.random() * 180), maxLife: Math.random() * 180 + 120, speed: Math.random() * 0.25 + 0.12, amplitude: Math.random() * 20 + 10, frequency: Math.random() * 0.008 + 0.004, phaseOff: Math.random() * Math.PI * 2, size: Math.random() * 1.8 + 0.8, baseX: 0, x: 0, y: 0, bright: Math.random() });
            }
        }
        for (let i = 0; i < 12; i++) {
            _coffeeBeans.push({ x: Math.random(), y: Math.random(), size: Math.random() * 4 + 2, rotation: Math.random() * Math.PI * 2, rotSpeed: (Math.random() - 0.5) * 0.003, vx: (Math.random() - 0.5) * 0.0001, vy: (Math.random() - 0.5) * 0.0001, alpha: Math.random() * 0.12 + 0.03 });
        }
    }
    function _coffeeDrawCup(cx, cy, t, cw, ch) {
        const warmG = ctx.createRadialGradient(cx, cy - 20, 0, cx, cy - 20, 120);
        warmG.addColorStop(0, 'rgba(212,148,60,0.04)'); warmG.addColorStop(0.5, 'rgba(184,122,46,0.02)'); warmG.addColorStop(1, 'transparent');
        ctx.fillStyle = warmG; ctx.beginPath(); ctx.arc(cx, cy - 20, 120, 0, Math.PI * 2); ctx.fill();
        for (const b of _coffeeBeans) {
            b.x += b.vx; b.y += b.vy; b.rotation += b.rotSpeed;
            if (b.x < -0.05 || b.x > 1.05) b.vx *= -1;
            if (b.y < -0.05 || b.y > 1.05) b.vy *= -1;
            ctx.save(); ctx.translate(b.x * cw, b.y * ch); ctx.rotate(b.rotation);
            ctx.fillStyle = `rgba(184,122,46,${b.alpha})`;
            ctx.beginPath(); ctx.ellipse(-b.size * 0.15, 0, b.size * 0.4, b.size * 0.7, 0, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.ellipse(b.size * 0.15, 0, b.size * 0.4, b.size * 0.7, 0, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = `rgba(12,8,6,${b.alpha * 1.2})`; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(0, -b.size * 0.5); ctx.bezierCurveTo(-b.size * 0.1, -b.size * 0.15, b.size * 0.1, b.size * 0.15, 0, b.size * 0.5); ctx.stroke();
            ctx.restore();
        }
        const cupCx = cx, cupTop = cy - 30, cupBot = cupTop + 40, cupW = 26, cupWB = 21;
        ctx.beginPath(); ctx.ellipse(cupCx, cupBot + 3, cupW * 1.3, 5, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(18,14,10,0.7)'; ctx.fill(); ctx.strokeStyle = 'rgba(184,122,46,0.15)'; ctx.lineWidth = 0.8; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cupCx - cupW, cupTop); ctx.lineTo(cupCx - cupWB, cupBot);
        ctx.quadraticCurveTo(cupCx, cupBot + 7, cupCx + cupWB, cupBot); ctx.lineTo(cupCx + cupW, cupTop); ctx.closePath();
        ctx.fillStyle = 'rgba(18,14,10,0.8)'; ctx.fill(); ctx.strokeStyle = 'rgba(184,122,46,0.25)'; ctx.lineWidth = 1; ctx.stroke();
        ctx.beginPath(); ctx.ellipse(cupCx, cupTop, cupW, 4, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(22,18,12,0.8)'; ctx.fill(); ctx.strokeStyle = 'rgba(212,148,60,0.3)'; ctx.lineWidth = 0.8; ctx.stroke();
        ctx.beginPath(); ctx.ellipse(cupCx + cupW + 8, cupTop + 14, 8, 13, 0, -Math.PI * 0.4, Math.PI * 0.5);
        ctx.strokeStyle = 'rgba(184,122,46,0.2)'; ctx.lineWidth = 1.5; ctx.stroke();
        const steamOriginY = cupTop - 3, steamSpread = cupW * 0.6;
        for (const p of _coffeeSteam) {
            p.life++;
            if (p.life > p.maxLife) { p.life = 0; p.baseX = cupCx + ((p.stream / p.streams) - 0.5) * steamSpread * 2 + (Math.random() - 0.5) * 8; p.x = p.baseX; p.y = steamOriginY; }
            p.y -= p.speed;
            const prog = p.life / p.maxLife;
            p.x = p.baseX + Math.sin(p.life * p.frequency + p.phaseOff) * p.amplitude * (1 + prog);
            p.baseX += (Math.random() - 0.5) * 0.2;
            let alpha = 1;
            if (prog < 0.1) alpha = prog / 0.1; else if (prog > 0.5) alpha = 1 - (prog - 0.5) / 0.5;
            alpha *= 0.35;
            const sz = p.size * (1 + prog * 1.2);
            const col = p.bright > 0.7 ? '255,240,212' : p.bright > 0.4 ? '232,170,84' : '212,148,60';
            const sg = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, sz);
            sg.addColorStop(0, `rgba(${col},${alpha * 0.7})`); sg.addColorStop(0.5, `rgba(${col},${alpha * 0.25})`); sg.addColorStop(1, `rgba(${col},0)`);
            ctx.beginPath(); ctx.arc(p.x, p.y, sz, 0, Math.PI * 2); ctx.fillStyle = sg; ctx.fill();
        }
        const pulse = 0.7 + 0.3 * Math.sin(t * 1.5);
        const tg = ctx.createRadialGradient(cupCx, cupTop, 0, cupCx, cupTop, 35);
        tg.addColorStop(0, `rgba(255,240,212,${0.03 * pulse})`); tg.addColorStop(1, 'transparent');
        ctx.fillStyle = tg; ctx.beginPath(); ctx.arc(cupCx, cupTop, 35, 0, Math.PI * 2); ctx.fill();
    }

    // ── Midnight: Moon + fireflies ──
    const _midnightFireflies = [];
    const _midnightClouds = [];
    if (isMidnight) {
        for (let i = 0; i < 30; i++) {
            _midnightFireflies.push({ x: 0.3 + Math.random() * 0.4, y: 0.2 + Math.random() * 0.3, phase: Math.random() * Math.PI * 2, glowSpeed: 0.3 + Math.random() * 0.7, wanderRx: 0.01 + Math.random() * 0.02, wanderRy: 0.005 + Math.random() * 0.015, wanderSpeed: 0.06 + Math.random() * 0.1, wanderPhase: Math.random() * Math.PI * 2, r: 0.8 + Math.random() * 1.2 });
        }
        for (let i = 0; i < 5; i++) {
            _midnightClouds.push({ x: Math.random(), y: 0.25 + Math.random() * 0.15, w: 0.06 + Math.random() * 0.1, h: 0.015 + Math.random() * 0.02, speed: 0.001 + Math.random() * 0.002, alpha: 0.02 + Math.random() * 0.02, phase: Math.random() * Math.PI * 2 });
        }
    }
    function _midnightDrawMoon(cx, cy, t, cw, ch) {
        const moonR = 35;
        const halo = ctx.createRadialGradient(cx, cy, moonR * 0.8, cx, cy, moonR * 6);
        halo.addColorStop(0, 'rgba(16,185,129,0.06)'); halo.addColorStop(0.3, 'rgba(52,211,153,0.03)');
        halo.addColorStop(0.6, 'rgba(5,150,105,0.01)'); halo.addColorStop(1, 'transparent');
        ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(cx, cy, moonR * 6, 0, Math.PI * 2); ctx.fill();
        const moonBg = ctx.createRadialGradient(cx - moonR * 0.2, cy - moonR * 0.2, 0, cx, cy, moonR);
        moonBg.addColorStop(0, 'rgba(200,210,230,0.12)'); moonBg.addColorStop(0.5, 'rgba(150,160,180,0.08)'); moonBg.addColorStop(1, 'rgba(100,110,130,0.04)');
        ctx.fillStyle = moonBg; ctx.beginPath(); ctx.arc(cx, cy, moonR, 0, Math.PI * 2); ctx.fill();
        ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, moonR + 1, 0, Math.PI * 2); ctx.clip();
        ctx.fillStyle = 'rgba(5,8,16,0.95)'; ctx.beginPath(); ctx.arc(cx + moonR * 0.6, cy, moonR * 0.95, 0, Math.PI * 2); ctx.fill(); ctx.restore();
        const crescentG = ctx.createRadialGradient(cx - moonR * 0.3, cy, moonR * 0.5, cx, cy, moonR * 1.2);
        crescentG.addColorStop(0, 'rgba(209,250,229,0.25)'); crescentG.addColorStop(0.4, 'rgba(52,211,153,0.08)'); crescentG.addColorStop(1, 'transparent');
        ctx.fillStyle = crescentG; ctx.beginPath(); ctx.arc(cx, cy, moonR * 1.2, 0, Math.PI * 2); ctx.fill();
        const craters = [[0.25,-0.3,0.12],[0.15,0.2,0.08],[-0.1,-0.15,0.06]];
        for (const [ox, oy, cr] of craters) {
            ctx.globalAlpha = 0.04; ctx.fillStyle = 'rgba(100,130,160,1)';
            ctx.beginPath(); ctx.arc(cx + moonR * ox - moonR * 0.15, cy + moonR * oy, moonR * cr, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
        for (const cl of _midnightClouds) {
            cl.x += cl.speed * 0.008; if (cl.x > 1.4) cl.x = -0.4;
            const breathe = 0.5 + 0.5 * Math.sin(t * 0.15 + cl.phase);
            const clx = cl.x * cw, cly = cl.y * ch, clw = cl.w * cw;
            const cg = ctx.createRadialGradient(clx, cly, 0, clx, cly, clw);
            cg.addColorStop(0, `rgba(16,185,129,${cl.alpha * breathe})`); cg.addColorStop(0.5, `rgba(52,211,153,${cl.alpha * breathe * 0.3})`); cg.addColorStop(1, 'transparent');
            ctx.fillStyle = cg; ctx.fillRect(clx - clw, cly - clw * cl.h / cl.w, clw * 2, clw * cl.h / cl.w * 2);
        }
        for (const f of _midnightFireflies) {
            const wp = t * f.wanderSpeed + f.wanderPhase;
            const fx = (f.x + Math.sin(wp) * f.wanderRx) * cw, fy = (f.y + Math.sin(wp * 2) * f.wanderRy) * ch;
            const glow = Math.pow(Math.max(0, Math.sin(t * f.glowSpeed + f.phase)), 1.5);
            if (glow < 0.03) continue;
            ctx.globalAlpha = glow;
            const fHalo = ctx.createRadialGradient(fx, fy, 0, fx, fy, f.r * 12);
            fHalo.addColorStop(0, 'rgba(52,211,153,0.10)'); fHalo.addColorStop(0.3, 'rgba(16,185,129,0.03)'); fHalo.addColorStop(1, 'transparent');
            ctx.fillStyle = fHalo; ctx.beginPath(); ctx.arc(fx, fy, f.r * 12, 0, Math.PI * 2); ctx.fill();
            const fCore = ctx.createRadialGradient(fx, fy, 0, fx, fy, f.r * 2);
            fCore.addColorStop(0, `rgba(209,250,229,${glow * 0.8})`); fCore.addColorStop(0.4, `rgba(52,211,153,${glow * 0.5})`); fCore.addColorStop(1, 'transparent');
            ctx.fillStyle = fCore; ctx.beginPath(); ctx.arc(fx, fy, f.r * 2, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    // ── Nebula: Black hole (offscreen-cached, redrawn every 3rd frame) ──
    const _BH_TILT = 0.30, _BH_ROT = -0.12;
    const _bhTierColors = [[200,190,255],[167,139,250],[56,189,248]];
    const _bhParticles = [], _bhBack = [], _bhFront = [];
    let _bhOffscreen = null, _bhOctx = null, _bhFrameCount = 0;
    const _BH_SKIP = _perfMode ? 4 : 3; // redraw every Nth frame
    if (isNebula) {
        _bhOffscreen = document.createElement('canvas');
        _bhOctx = _bhOffscreen.getContext('2d');
        const _bhCount = _perfMode ? 100 : 200;
        for (let i = 0; i < _bhCount; i++) {
            const band = Math.random(), dist = 50 + band * 260;
            const tier = band < 0.2 ? 0 : band < 0.55 ? 1 : 2;
            const p = { angle: Math.random() * Math.PI * 2, dist, speed: (0.06 + Math.random() * 0.14) * (180 / (dist + 30)), r: tier === 0 ? Math.random() * 2.2 + 0.8 : Math.random() * 1.6 + 0.4, a: tier === 0 ? Math.random() * 0.7 + 0.4 : Math.random() * 0.55 + 0.15, yOff: (Math.random() - 0.5) * 5, tier };
            _bhParticles.push(p);
            // Pre-split into back/front by initial depth to avoid sorting every frame
            if (Math.sin(p.angle) <= 0.1) _bhBack.push(p); else _bhFront.push(p);
        }
    }
    function _bhProject(cx, cy, dist, angle) {
        const x3 = Math.cos(angle) * dist, y3 = Math.sin(angle) * dist;
        const cr = Math.cos(_BH_ROT), sr = Math.sin(_BH_ROT);
        return { x: cx + x3 * cr - y3 * sr, y: cy + (x3 * sr + y3 * cr) * _BH_TILT, depth: Math.sin(angle) };
    }
    function _bhDrawDisk(c, cx, cy) {
        c.save(); c.translate(cx, cy); c.rotate(_BH_ROT); c.scale(1, _BH_TILT);
        for (let ring = 0; ring < 3; ring++) {
            const rd = 80 + ring * 70, alpha = [0.10, 0.06, 0.035][ring];
            c.beginPath(); c.arc(0, 0, rd, 0, Math.PI * 2);
            c.strokeStyle = `rgba(167,139,250,${alpha})`; c.lineWidth = 18 + ring * 8; c.stroke();
        }
        const haze = c.createRadialGradient(0, 0, 80, 0, 0, 320);
        haze.addColorStop(0, 'rgba(100,80,200,0.0)'); haze.addColorStop(0.4, 'rgba(80,60,180,0.04)');
        haze.addColorStop(0.7, 'rgba(56,130,220,0.025)'); haze.addColorStop(1, 'rgba(56,189,248,0)');
        c.fillStyle = haze; c.beginPath(); c.arc(0, 0, 320, 0, Math.PI * 2); c.fill();
        c.restore();
    }
    function _bhDrawVoid(c, cx, cy, t) {
        const pulse = 1 + Math.sin(t * 0.4) * 0.03, eventR = 30 * pulse;
        const lens = c.createRadialGradient(cx, cy, eventR, cx, cy, eventR * 5);
        lens.addColorStop(0, 'rgba(167,139,250,0.22)'); lens.addColorStop(0.25, 'rgba(120,100,220,0.10)');
        lens.addColorStop(0.5, 'rgba(56,189,248,0.04)'); lens.addColorStop(1, 'rgba(56,189,248,0)');
        c.fillStyle = lens; c.beginPath(); c.arc(cx, cy, eventR * 5, 0, Math.PI * 2); c.fill();
        const photon = c.createRadialGradient(cx, cy, eventR - 3, cx, cy, eventR + 14);
        photon.addColorStop(0, 'rgba(167,139,250,0)'); photon.addColorStop(0.25, 'rgba(167,139,250,0.55)');
        photon.addColorStop(0.45, 'rgba(220,210,255,0.7)'); photon.addColorStop(0.65, 'rgba(125,211,252,0.45)'); photon.addColorStop(1, 'rgba(56,189,248,0)');
        c.fillStyle = photon; c.beginPath(); c.arc(cx, cy, eventR + 14, 0, Math.PI * 2); c.fill();
        const voidG = c.createRadialGradient(cx, cy, 0, cx, cy, eventR);
        voidG.addColorStop(0, 'rgba(0,0,0,1)'); voidG.addColorStop(0.8, 'rgba(0,0,0,1)'); voidG.addColorStop(1, 'rgba(0,0,0,0.6)');
        c.fillStyle = voidG; c.beginPath(); c.arc(cx, cy, eventR, 0, Math.PI * 2); c.fill();
    }
    function _nebulaDrawBlackHole(cx, cy, t) {
        // Resize offscreen canvas if needed
        if (_bhOffscreen.width !== canvas.width || _bhOffscreen.height !== canvas.height) {
            _bhOffscreen.width = canvas.width; _bhOffscreen.height = canvas.height;
            _bhFrameCount = 0; // force redraw on resize
        }
        // Only redraw every Nth frame — blit cached image otherwise
        if (_bhFrameCount % _BH_SKIP === 0) {
            const c = _bhOctx;
            c.clearRect(0, 0, _bhOffscreen.width, _bhOffscreen.height);
            _bhDrawDisk(c, cx, cy);
            // Draw back particles (depth <= 0.1)
            for (const p of _bhParticles) {
                const ang = p.angle + t * p.speed;
                const pr = _bhProject(cx, cy, p.dist, ang);
                const d = pr.depth;
                if (d > 0.1) continue;
                const col = _bhTierColors[p.tier];
                c.globalAlpha = p.a * (0.65 + d * 0.35);
                c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
                c.beginPath(); c.arc(pr.x, pr.y + p.yOff, p.r, 0, Math.PI * 2); c.fill();
            }
            _bhDrawVoid(c, cx, cy, t);
            // Draw front particles (depth > 0.1)
            for (const p of _bhParticles) {
                const ang = p.angle + t * p.speed;
                const pr = _bhProject(cx, cy, p.dist, ang);
                const d = pr.depth;
                if (d <= 0.1) continue;
                const col = _bhTierColors[p.tier];
                c.globalAlpha = p.a * (0.6 + d * 0.4);
                if (p.tier === 0) {
                    c.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${p.a * 0.15})`;
                    c.beginPath(); c.arc(pr.x, pr.y + p.yOff, p.r * 4, 0, Math.PI * 2); c.fill();
                }
                c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
                c.beginPath(); c.arc(pr.x, pr.y + p.yOff, p.r, 0, Math.PI * 2); c.fill();
            }
            c.globalAlpha = 1;
        }
        _bhFrameCount++;
        // Blit cached offscreen canvas (single GPU texture copy)
        ctx.drawImage(_bhOffscreen, 0, 0);
    }

    // ── Aurora: Binary star system ──
    const _BIN_ORBIT_R = 45, _BIN_SPEED = 0.12, _BIN_TILT = 0.38;
    const _binStarA = { r: 22, color: '52,211,153', bright: '200,255,225', glow: '110,231,183' };
    const _binStarB = { r: 16, color: '56,189,248', bright: '200,235,255', glow: '125,211,252' };
    const _binParticles = [];
    if (isAurora) {
        const _binCount = _perfMode ? 80 : 180;
        for (let i = 0; i < _binCount; i++) {
            const band = Math.random(), dist = 55 + band * 180;
            _binParticles.push({ angle: Math.random() * Math.PI * 2, dist, speed: (0.04 + Math.random() * 0.06) * (120 / (dist + 30)), r: Math.random() * 1.2 + 0.3, a: Math.random() * 0.4 + 0.1, yOff: (Math.random() - 0.5) * 4, colorType: band < 0.4 ? 0 : band < 0.7 ? 1 : 2 });
        }
    }
    const _binColors = [[110,231,183],[56,189,248],[200,220,240]];
    function _binProject(cx, cy, dist, angle) {
        return { x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist * _BIN_TILT, depth: Math.sin(angle) };
    }
    function _auroraDrawBinary(cx, cy, t) {
        const ang = t * _BIN_SPEED;
        const s1x = cx + Math.cos(ang) * _BIN_ORBIT_R, s1y = cy + Math.sin(ang) * _BIN_ORBIT_R * _BIN_TILT, s1d = Math.sin(ang);
        const s2x = cx + Math.cos(ang + Math.PI) * _BIN_ORBIT_R, s2y = cy + Math.sin(ang + Math.PI) * _BIN_ORBIT_R * _BIN_TILT, s2d = Math.sin(ang + Math.PI);
        const amb = ctx.createRadialGradient(cx, cy, 0, cx, cy, 300);
        amb.addColorStop(0, 'rgba(52,211,153,0.06)'); amb.addColorStop(0.15, 'rgba(56,189,248,0.03)');
        amb.addColorStop(0.35, 'rgba(110,231,183,0.012)'); amb.addColorStop(0.6, 'rgba(16,185,129,0.004)'); amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(cx, cy, 300, 0, Math.PI * 2); ctx.fill();
        ctx.save(); ctx.translate(cx, cy); ctx.scale(1, _BIN_TILT);
        const ringDefs = [[60,'110,231,183',0.04],[100,'56,189,248',0.025],[140,'52,211,153',0.015]];
        for (const [rd, rc, ra] of ringDefs) { ctx.beginPath(); ctx.arc(0, 0, rd, 0, Math.PI * 2); ctx.strokeStyle = `rgba(${rc},${ra})`; ctx.lineWidth = 12; ctx.stroke(); }
        ctx.beginPath(); ctx.arc(0, 0, _BIN_ORBIT_R, 0, Math.PI * 2); ctx.strokeStyle = 'rgba(110,231,183,0.035)'; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
        const proj = [];
        for (const p of _binParticles) { const pr = _binProject(cx, cy, p.dist, p.angle + t * p.speed); proj.push({ x: pr.x, y: pr.y + p.yOff, d: pr.depth, r: p.r, a: p.a, ct: p.colorType }); }
        proj.sort((a, b) => a.d - b.d);
        for (const p of proj) { if (p.d > 0.1) continue; const c = _binColors[p.ct]; ctx.globalAlpha = p.a * (0.5 + p.d * 0.5); ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill(); }
        const bridgePulse = 0.7 + 0.3 * Math.sin(t * 0.8);
        const bridgeDefs = [['200,255,230',0.06,4],['110,231,183',0.035,8],['56,189,248',0.02,14]];
        for (const [bc, ba, bw] of bridgeDefs) { const sOff = Math.sin(t * 0.5 + bw) * 4; ctx.beginPath(); ctx.moveTo(s1x, s1y); ctx.quadraticCurveTo((s1x+s2x)/2 + sOff, (s1y+s2y)/2 - 5, s2x, s2y); ctx.strokeStyle = `rgba(${bc},${ba * bridgePulse})`; ctx.lineWidth = bw; ctx.lineCap = 'round'; ctx.stroke(); }
        const pair = [{ x: s1x, y: s1y, d: s1d, ..._binStarA, id: 'A' },{ x: s2x, y: s2y, d: s2d, ..._binStarB, id: 'B' }];
        pair.sort((a, b) => a.d - b.d);
        for (const star of pair) {
            const db = 0.75 + 0.25 * star.d, pulse = 1 + Math.sin(t * 0.4 + (star.id === 'A' ? 0 : Math.PI)) * 0.08, sr = star.r * pulse;
            const oh = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, sr * 8);
            oh.addColorStop(0, `rgba(${star.glow},${0.08 * db})`); oh.addColorStop(0.2, `rgba(${star.color},${0.04 * db})`); oh.addColorStop(0.5, `rgba(${star.color},${0.012 * db})`); oh.addColorStop(1, 'transparent');
            ctx.fillStyle = oh; ctx.beginPath(); ctx.arc(star.x, star.y, sr * 8, 0, Math.PI * 2); ctx.fill();
            const ig = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, sr * 3);
            ig.addColorStop(0, `rgba(${star.bright},${0.25 * db})`); ig.addColorStop(0.3, `rgba(${star.glow},${0.15 * db})`); ig.addColorStop(0.7, `rgba(${star.color},${0.04 * db})`); ig.addColorStop(1, 'transparent');
            ctx.fillStyle = ig; ctx.beginPath(); ctx.arc(star.x, star.y, sr * 3, 0, Math.PI * 2); ctx.fill();
            const bd = ctx.createRadialGradient(star.x - sr * 0.1, star.y - sr * 0.1, 0, star.x, star.y, sr);
            bd.addColorStop(0, `rgba(${star.bright},${0.5 * db})`); bd.addColorStop(0.3, `rgba(${star.bright},${0.35 * db})`); bd.addColorStop(0.7, `rgba(${star.color},${0.2 * db})`); bd.addColorStop(1, `rgba(${star.color},${0.05 * db})`);
            ctx.fillStyle = bd; ctx.beginPath(); ctx.arc(star.x, star.y, sr, 0, Math.PI * 2); ctx.fill();
            ctx.globalAlpha = 0.08 * db * pulse; ctx.strokeStyle = `rgba(${star.bright},0.4)`; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(star.x - sr * 5, star.y); ctx.lineTo(star.x + sr * 5, star.y); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(star.x, star.y - sr * 3.5); ctx.lineTo(star.x, star.y + sr * 3.5); ctx.stroke();
            ctx.globalAlpha = 1;
        }
        for (const p of proj) { if (p.d <= 0.1) continue; const c = _binColors[p.ct]; const da = p.a * (0.6 + p.d * 0.4); ctx.globalAlpha = da;
            if (p.ct === 0 && p.r > 0.8) { const pg = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 5); pg.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${da * 0.25})`); pg.addColorStop(1, 'transparent'); ctx.fillStyle = pg; ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 5, 0, Math.PI * 2); ctx.fill(); }
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill(); }
        ctx.globalAlpha = 1;
    }

    // ── Generic theme element draw helper ──
    function _drawThemeElement(themeName, drawFn, t) {
        const prefix = 'stratos-' + themeName;
        const cx = parseFloat(localStorage.getItem(prefix + '-cx') || '0.5');
        const cy = parseFloat(localStorage.getItem(prefix + '-cy') || '0.35');
        const scale = parseFloat(localStorage.getItem(prefix + '-scale') || '1');
        const opacity = parseFloat(localStorage.getItem(prefix + '-opacity') || '1');
        const blur = _perfMode ? 0 : parseFloat(localStorage.getItem(prefix + '-blur') || '0');
        const px = canvas.width * cx, py = canvas.height * cy;
        ctx.save();
        ctx.globalAlpha = opacity;
        if (blur > 0) ctx.filter = `blur(${blur}px)`;
        ctx.translate(px, py); ctx.scale(scale, scale); ctx.translate(-px, -py);
        drawFn(px, py, t, canvas.width, canvas.height);
        ctx.restore();
    }

    function draw() {
        if (_myGen !== _starGeneration) return; // stale engine — stop loop
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const t = Date.now() * 0.001;
        const now = Date.now();
        const parallax = scrollOffset * 0.06;

        // Solar system (cosmos theme - drawn first, behind stars)
        if (isCosmos) {
            const _cxPct = parseFloat(localStorage.getItem('stratos-cosmos-cx') || '0.5');
            const _cyPct = parseFloat(localStorage.getItem('stratos-cosmos-cy') || '0.35');
            const _sScale = parseFloat(localStorage.getItem('stratos-cosmos-scale') || '1');
            const _ssOpacity = parseFloat(localStorage.getItem('stratos-cosmos-opacity') || '1');
            const _ssBlur = _perfMode ? 0 : parseFloat(localStorage.getItem('stratos-cosmos-blur') || '0');
            const scx = canvas.width * _cxPct, scy = canvas.height * _cyPct;
            ctx.save();
            ctx.globalAlpha = _ssOpacity;
            if (_ssBlur > 0) ctx.filter = `blur(${_ssBlur}px)`;
            ctx.translate(scx, scy);
            ctx.scale(_sScale, _sScale);
            ctx.translate(-scx, -scy);
            if (_ssPreset === 'P2') {
                for (const p of _ssPlanets) _ssTiltedOrbit(scx, scy, p.dist, 0.07);
                _ssTiltedOrbit(scx, scy, 232, 0.03); _ssTiltedOrbit(scx, scy, 270, 0.03);
                const _rr = [];
                for (const a of _ssAsteroids) {
                    const pr = _ssProject(scx, scy, a.dist, a.angle + t * a.speed);
                    _rr.push({ t: 'a', x: pr.x, y: pr.y + a.yOff, d: pr.depth, r: a.r, a: a.a });
                }
                for (const p of _ssPlanets) {
                    const ang = p.phase + t * p.speed * 0.15;
                    const pr = _ssProject(scx, scy, p.dist, ang);
                    _rr.push({ t: 'p', x: pr.x, y: pr.y, d: pr.depth, p, ang });
                    if (p.moon) {
                        const mAng = ang + t * p.moon.speed * 0.3;
                        const mp = _ssProject(pr.x, pr.y, p.moon.dist, mAng);
                        _rr.push({ t: 'm', x: mp.x, y: mp.y, d: pr.depth + mp.depth * 0.01, r: p.moon.r });
                    }
                }
                _rr.push({ t: 's', x: scx, y: scy, d: 0 });
                _rr.sort((a, b) => a.d - b.d);
                for (const o of _rr) {
                    if (o.t === 'a') { ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2); ctx.fillStyle = `rgba(150,140,120,${o.a})`; ctx.fill(); }
                    else if (o.t === 's') _ssDrawSun(scx, scy, t);
                    else if (o.t === 'm') { ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2); ctx.fillStyle = 'rgba(200,200,210,0.7)'; ctx.fill(); }
                    else if (o.t === 'p') _ssDrawPlanetP2(o.x, o.y, o.p, o.d);
                }
            } else {
                for (const p of _ssPlanets) {
                    ctx.beginPath(); ctx.arc(scx, scy, p.dist, 0, Math.PI * 2);
                    ctx.strokeStyle = 'rgba(150,170,220,0.08)'; ctx.lineWidth = 0.5; ctx.stroke();
                }
                for (const a of _ssAsteroids) {
                    const ang = a.angle + t * a.speed;
                    ctx.beginPath(); ctx.arc(scx + Math.cos(ang) * a.dist, scy + Math.sin(ang) * a.dist, a.r, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(160,150,130,${a.a})`; ctx.fill();
                }
                for (const p of _ssPlanets) _ssDrawPlanetP1(scx, scy, p, p.phase + t * p.speed * 0.15);
                _ssDrawSun(scx, scy, t);
            }
            ctx.restore();
        }

        // Theme unique elements (drawn behind stars, like cosmos)
        // Each element respects its visibility toggle (stratos-{theme}-visible)
        if (isNoir && localStorage.getItem('stratos-noir-visible') !== 'false') _drawThemeElement('noir', _noirDrawPendulum, t);
        if (isRose && localStorage.getItem('stratos-rose-visible') !== 'false') _drawThemeElement('rose', _roseDrawBloom, t);
        if (isCoffee && localStorage.getItem('stratos-coffee-visible') !== 'false') _drawThemeElement('coffee', _coffeeDrawCup, t);
        if (isMidnight && localStorage.getItem('stratos-midnight-visible') !== 'false') _drawThemeElement('midnight', _midnightDrawMoon, t);
        if (isNebula && localStorage.getItem('stratos-nebula-visible') !== 'false') _drawThemeElement('nebula', _nebulaDrawBlackHole, t);
        if (isAurora && localStorage.getItem('stratos-aurora-visible') !== 'false') _drawThemeElement('aurora', _auroraDrawBinary, t);
        if (isSakura) _drawSakuraTree(canvas.width, canvas.height, t);

        // Shooting stars (skip in perf mode)
        if (!_perfMode && now - lastShooter > SHOOT_INTERVAL + Math.random() * 3000) {
            spawnShooter();
            lastShooter = now;
        }

        const nearby = [];

        for (let i = 0; i < stars.length; i++) {
            const s = stars[i];

            // Gentle drift — petals fall down, stars drift up
            if (isSakura && s.petal) {
                const _skFall = parseFloat(localStorage.getItem('stratos-sakura-fall') || '1');
                const _skWind = parseFloat(localStorage.getItem('stratos-sakura-wind') || '1');
                s.baseY += s.petalFall * _skFall;
                s.baseX += Math.sin(t * s.petalSway + s.phase) * 0.25 * _skWind;
                s.petalAngle += s.petalSpin;
                if (s.baseY > canvas.height + 20) {
                    s.baseY = -20;
                    s.baseX = Math.random() * canvas.width;
                    s.y = s.baseY; s.x = s.baseX;
                }
            } else {
                s.baseY -= DRIFT_SPEED_BASE * parseFloat(localStorage.getItem('stratos-stars-drift') || '1');
                if (s.baseY < -10) {
                    s.baseY = canvas.height + 10;
                    s.baseX = Math.random() * canvas.width;
                    s.y = s.baseY; s.x = s.baseX;
                }
            }

            let drawX = s.baseX, drawY = s.baseY - parallax;
            const dx = drawX - mouseX, dy = drawY - mouseY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // Mouse repulsion
            if (!isTouch && dist < MOUSE_RADIUS && dist > 0) {
                const force = (1 - dist / MOUSE_RADIUS) * 24;
                drawX += (dx / dist) * force;
                drawY += (dy / dist) * force;
            }

            s.x += (drawX - s.x) * 0.07;
            s.y += (drawY - s.y) * 0.07;

            if (s.y < -15 || s.y > canvas.height + 15) continue;

            const _starsBright = parseFloat(localStorage.getItem('stratos-stars-brightness') || '1');
            const flicker = 0.65 + 0.35 * Math.sin(t * s.speed * 4 + s.phase);
            const proxBoost = (!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.5 : 1;
            const alpha = Math.min(1, s.a * flicker * proxBoost * _starsBright);
            const radius = s.r * ((!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.35 : 1);

            ctx.globalAlpha = alpha;

            // Sakura petals: bezier petal shape with notch
            if (isSakura && s.petal) {
                const _skSize = parseFloat(localStorage.getItem('stratos-sakura-size') || '1');
                const _skOpacity = parseFloat(localStorage.getItem('stratos-sakura-opacity') || '1');
                const sz = s.petalSize * _skSize * ((!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.3 : 1);
                ctx.save();
                ctx.translate(s.x, s.y);
                ctx.rotate(s.petalAngle);
                ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${alpha * _skOpacity})`;
                ctx.beginPath();
                // Petal: stem tip at bottom, two lobes at top with center notch
                ctx.moveTo(0, sz * 1.6);                           // stem tip
                ctx.bezierCurveTo(sz * 1.2, sz * 0.8, sz * 1.4, -sz * 0.6, sz * 0.3, -sz * 1.2);  // right lobe
                ctx.quadraticCurveTo(0, -sz * 0.7, -sz * 0.3, -sz * 1.2);                          // notch
                ctx.bezierCurveTo(-sz * 1.4, -sz * 0.6, -sz * 1.2, sz * 0.8, 0, sz * 1.6);        // left lobe
                ctx.fill();
                // Subtle center vein
                ctx.globalAlpha = alpha * 0.15;
                ctx.strokeStyle = `rgb(${s.cr},${s.cg},${s.cb})`;
                ctx.lineWidth = 0.4;
                ctx.beginPath();
                ctx.moveTo(0, sz * 1.4);
                ctx.quadraticCurveTo(0, 0, 0, -sz * 0.8);
                ctx.stroke();
                ctx.restore();
            } else {
                ctx.fillStyle = `rgb(${s.cr},${s.cg},${s.cb})`;
                ctx.beginPath();
                ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
                ctx.fill();
            }

            // Glow on bright stars near cursor
            if (!isTouch && s.isBright && dist < MOUSE_RADIUS * 0.7) {
                ctx.globalAlpha = alpha * 0.12;
                ctx.beginPath();
                ctx.arc(s.x, s.y, radius * 3.5, 0, Math.PI * 2);
                ctx.fill();
            }

            // Track for constellation lines
            if (!isTouch) {
                const mDist = Math.sqrt((s.x - mouseX) ** 2 + (s.y - mouseY) ** 2);
                if (mDist < LINE_MOUSE_RANGE) {
                    nearby.push({ x: s.x, y: s.y, a: alpha, mDist: mDist });
                }
            }
        }

        // Constellation lines
        if (!isTouch && nearby.length > 1) {
            for (let a = 0; a < nearby.length; a++) {
                for (let b = a + 1; b < nearby.length; b++) {
                    const ddx = nearby[a].x - nearby[b].x;
                    const ddy = nearby[a].y - nearby[b].y;
                    const d = Math.sqrt(ddx * ddx + ddy * ddy);
                    if (d < LINE_RADIUS) {
                        let la = (1 - d / LINE_RADIUS) * 0.14;
                        const avgMD = (nearby[a].mDist + nearby[b].mDist) / 2;
                        la *= Math.max(0, (1 - avgMD / LINE_MOUSE_RANGE) * 1.3);
                        la = Math.min(la, 0.16);
                        ctx.globalAlpha = la;
                        ctx.strokeStyle = colors.accent;
                        ctx.lineWidth = 0.5;
                        ctx.beginPath();
                        ctx.moveTo(nearby[a].x, nearby[a].y);
                        ctx.lineTo(nearby[b].x, nearby[b].y);
                        ctx.stroke();
                    }
                }
            }
        }

        // Shooting stars
        for (let si = shooters.length - 1; si >= 0; si--) {
            const sh = shooters[si];
            sh.x += sh.vx; sh.y += sh.vy;
            sh.life -= 0.014;
            if (sh.life <= 0 || sh.x > canvas.width + 60 || sh.y > canvas.height + 60) {
                shooters.splice(si, 1); continue;
            }
            const spd = Math.sqrt(sh.vx * sh.vx + sh.vy * sh.vy);
            const tailX = sh.x - sh.vx * (sh.len / spd);
            const tailY = sh.y - sh.vy * (sh.len / spd);
            const grad = ctx.createLinearGradient(tailX, tailY, sh.x, sh.y);
            grad.addColorStop(0, 'transparent');
            grad.addColorStop(1, `rgba(${sh.cr},${sh.cg},${sh.cb},${sh.life * 0.5})`);
            ctx.globalAlpha = sh.life * 0.6;
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.3;
            ctx.beginPath();
            ctx.moveTo(tailX, tailY);
            ctx.lineTo(sh.x, sh.y);
            ctx.stroke();
            ctx.globalAlpha = sh.life * 0.8;
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(sh.x, sh.y, 1.2, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.globalAlpha = 1;
        raf = requestAnimationFrame(draw);
    }

    raf = requestAnimationFrame(draw);

    // Store engine state for cleanup
    _starEngine = { raf, onMove: isTouch ? null : onMove, onLeave: isTouch ? null : onLeave, onScroll };

    // Resize handler
    window._starResize = function() {
        resize();
        for (let i = 0; i < stars.length; i++) {
            stars[i].baseX = Math.random() * canvas.width;
            stars[i].baseY = Math.random() * canvas.height;
            stars[i].x = stars[i].baseX; stars[i].y = stars[i].baseY;
        }
    };
}

// Resize watcher for star canvas
window.addEventListener('resize', function() {
    if (typeof window._starResize === 'function') window._starResize();
});

// Load saved theme + mode + stars
const savedTheme = validThemes.includes(localStorage.getItem('stratos-theme'))
    ? localStorage.getItem('stratos-theme') : 'midnight';
const savedMode = localStorage.getItem('stratos-theme-mode') ||
    (localStorage.getItem('stratos-dark') === 'true' ? 'dark' : 'normal');
applyThemeMode(savedMode);
setTheme(savedTheme);
// Restore stars toggle + perf mode
const savedStars = localStorage.getItem('stratos-stars') === 'true';
updateStarsToggleUI(savedStars);
const savedPerfMode = localStorage.getItem('stratos-perf-mode') === 'true';
updatePerfToggleUI(savedPerfMode);
if (savedPerfMode) document.body.classList.add('perf-mode');
updateCosmosPresetUI();

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
            answerEl.innerHTML = `<span class="text-red-400">Error: ${esc(data.error)}</span>`;
        } else {
            answerEl.innerHTML = `<div class="text-slate-300 whitespace-pre-wrap">${esc(data.answer)}</div>
                <button onclick="continueInAgent(${idx}, this)" class="mt-2 text-[10px] px-2 py-1 bg-purple-900/30 text-purple-400 rounded hover:bg-purple-800/40 transition-colors flex items-center gap-1">
                    <i data-lucide="message-circle" class="w-3 h-3"></i> Continue in Agent
                </button>`;
            if (typeof lucide !== 'undefined') lucide.createIcons({nodes: [answerEl]});
        }
    })
    .catch(err => {
        answerEl.innerHTML = `<span class="text-red-400">Failed: ${esc(err.message)}</span>`;
    });
}

function continueInAgent(idx) {
    const items = window.currentItems || [];
    const item = items[idx];
    if (!item) return;
    // Build context message for the agent
    const ctx = `Regarding "${item.title}" (score: ${item.score}):\n${item.summary || ''}`;
    // Open agent panel and pre-fill
    if (typeof _openAgentPanel === 'function') _openAgentPanel();
    const input = document.getElementById('agent-input');
    if (input) {
        input.value = ctx;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
    }
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
            statusEl.innerHTML = `<span class="text-red-400">Error: ${esc(data.error || 'unknown')}</span>`;
        }
    })
    .catch(err => {
        statusEl.innerHTML = `<span class="text-red-400">Failed: ${esc(err.message)}</span>`;
    });
}
