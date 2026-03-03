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

/* ═══ INTERACTIVE STAR CANVAS ENGINE ═══ */
var _starEngine = null; // holds running engine state for cleanup

function _stopStarEngine() {
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
    const COUNT = isMobile ? 30 : 200;
    const MOUSE_RADIUS = 150;
    const LINE_RADIUS = 120;
    const LINE_MOUSE_RANGE = 240;
    const DRIFT_SPEED = 0.06;

    const theme = document.documentElement.getAttribute('data-theme');
    const isSakura = theme === 'sakura';

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

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const t = Date.now() * 0.001;
        const now = Date.now();
        const parallax = scrollOffset * 0.06;

        // Shooting stars
        if (now - lastShooter > SHOOT_INTERVAL + Math.random() * 3000) {
            spawnShooter();
            lastShooter = now;
        }

        const nearby = [];

        for (let i = 0; i < stars.length; i++) {
            const s = stars[i];

            // Gentle drift — petals fall down, stars drift up
            if (isSakura && s.petal) {
                s.baseY += s.petalFall;
                s.baseX += Math.sin(t * s.petalSway + s.phase) * 0.25;
                s.petalAngle += s.petalSpin;
                if (s.baseY > canvas.height + 20) {
                    s.baseY = -20;
                    s.baseX = Math.random() * canvas.width;
                    s.y = s.baseY; s.x = s.baseX;
                }
            } else {
                s.baseY -= DRIFT_SPEED;
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

            const flicker = 0.65 + 0.35 * Math.sin(t * s.speed * 4 + s.phase);
            const proxBoost = (!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.5 : 1;
            const alpha = Math.min(1, s.a * flicker * proxBoost);
            const radius = s.r * ((!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.35 : 1);

            ctx.globalAlpha = alpha;

            // Sakura petals: bezier petal shape with notch
            if (isSakura && s.petal) {
                const sz = s.petalSize * ((!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.3 : 1);
                ctx.save();
                ctx.translate(s.x, s.y);
                ctx.rotate(s.petalAngle);
                ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${alpha})`;
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
            answerEl.innerHTML = `<span class="text-red-400">Error: ${esc(data.error)}</span>`;
        } else {
            answerEl.innerHTML = `<div class="text-slate-300 whitespace-pre-wrap">${esc(data.answer)}</div>`;
        }
    })
    .catch(err => {
        answerEl.innerHTML = `<span class="text-red-400">Failed: ${esc(err.message)}</span>`;
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
            statusEl.innerHTML = `<span class="text-red-400">Error: ${esc(data.error || 'unknown')}</span>`;
        }
    })
    .catch(err => {
        statusEl.innerHTML = `<span class="text-red-400">Failed: ${esc(err.message)}</span>`;
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
