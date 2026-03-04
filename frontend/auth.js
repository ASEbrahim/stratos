/**
 * STRAT_OS Auth — Landing / Login / Register
 * Color theme rotates on every page load.
 */

const AUTH_TOKEN_KEY = 'stratos_auth_token';
const AUTH_PROFILE_KEY = 'stratos_active_profile';
const DEVICE_ID_KEY = 'stratos_device_id';
const AUTH_THEME_KEY = 'stratos_auth_theme_idx';

function getAuthToken()    { return localStorage.getItem(AUTH_TOKEN_KEY) || ''; }
function setAuthToken(t)   { localStorage.setItem(AUTH_TOKEN_KEY, t); }
function clearAuthToken()  { localStorage.removeItem(AUTH_TOKEN_KEY); localStorage.removeItem(AUTH_PROFILE_KEY); }
function getActiveProfile()  { return localStorage.getItem(AUTH_PROFILE_KEY) || ''; }
function setActiveProfile(n) { localStorage.setItem(AUTH_PROFILE_KEY, n); }

function getDeviceId() {
    let id = localStorage.getItem(DEVICE_ID_KEY);
    if (!id) {
        const arr = new Uint8Array(8);
        crypto.getRandomValues(arr);
        id = Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
        localStorage.setItem(DEVICE_ID_KEY, id);
    }
    return id;
}

const _originalFetch = window.fetch;
let _authRedirecting = false;
window.fetch = function(url, opts = {}) {
    if (typeof url === 'string' && url.startsWith('/api/')) {
        const t = getAuthToken(), d = getDeviceId();
        opts.headers = { ...(opts.headers || {}) };
        if (t) opts.headers['X-Auth-Token'] = t;
        if (d) opts.headers['X-Device-Id'] = d;
    }
    return _originalFetch.call(this, url, opts).then(r => {
        if (r.status === 401 && typeof url === 'string' && url.startsWith('/api/')
            && !_authRedirecting
            && !url.startsWith('/api/auth/')
            && !document.getElementById('auth-overlay')) {
            // Verify the token is truly invalid before logging out
            _authRedirecting = true;
            _originalFetch('/api/auth/check', { headers: { 'X-Auth-Token': getAuthToken() } })
                .then(cr => cr.ok ? cr.json() : null)
                .then(d => {
                    if (!d || !d.authenticated) {
                        clearAuthToken(); checkAuthAndInit();
                    }
                    _authRedirecting = false;
                })
                .catch(() => { _authRedirecting = false; });
        }
        return r;
    });
};

/* ═══ ROTATING COLOR THEMES (synced with app theme palette) ═══ */
const AUTH_THEMES = [
    { name:'Nebula', accent:'#7dd3fc', rgb:'125,211,252',
      orb1:'rgba(167,139,250,.14)', orb2:'rgba(56,189,248,.10)', orb3:'rgba(99,102,241,.08)',
      grid:'rgba(56,189,248,.04)', bg:'linear-gradient(160deg,#08060f 0%,#140e25 50%,#08060f 100%)',
      star1:'rgba(167,139,250,.5)', star2:'rgba(56,189,248,.4)', star3:'rgba(255,255,255,.25)' },
    { name:'Aurora', accent:'#6ee7b7', rgb:'110,231,183',
      orb1:'rgba(52,211,153,.14)', orb2:'rgba(56,189,248,.10)', orb3:'rgba(16,185,129,.07)',
      grid:'rgba(52,211,153,.04)', bg:'linear-gradient(160deg,#060b10 0%,#0a1520 50%,#060b10 100%)',
      star1:'rgba(52,211,153,.4)', star2:'rgba(56,189,248,.3)', star3:'rgba(255,255,255,.3)' },
    { name:'Noir', accent:'#a78bfa', rgb:'167,139,250',
      orb1:'rgba(139,92,246,.14)', orb2:'rgba(99,102,241,.10)', orb3:'rgba(192,132,252,.08)',
      grid:'rgba(139,92,246,.045)', bg:'linear-gradient(160deg,#050506 0%,#0f0a24 50%,#050506 100%)',
      star1:'rgba(167,139,250,.5)', star2:'rgba(192,132,252,.3)', star3:'rgba(255,255,255,.2)' },
    { name:'Cosmos', accent:'#f0cc55', rgb:'240,204,85',
      orb1:'rgba(232,185,49,.14)', orb2:'rgba(150,180,255,.10)', orb3:'rgba(100,130,220,.07)',
      grid:'rgba(232,185,49,.04)', bg:'linear-gradient(160deg,#07080f 0%,#0d1225 50%,#07080f 100%)',
      star1:'rgba(232,185,49,.6)', star2:'rgba(150,180,255,.4)', star3:'rgba(255,255,255,.3)' },
];

function _getAuthTheme() {
    let idx = parseInt(localStorage.getItem(AUTH_THEME_KEY) || '0', 10);
    if (isNaN(idx) || idx < 0) idx = 0;
    localStorage.setItem(AUTH_THEME_KEY, String((idx + 1) % AUTH_THEMES.length));
    return AUTH_THEMES[idx % AUTH_THEMES.length];
}
const _t = _getAuthTheme();

/* ═══ STATE ═══ */
let _deviceProfiles = [], _allProfiles = [], _showingAll = false;
let _authMode = 'legacy'; // 'email' or 'legacy'
let _hasDbUsers = false, _openReg = true, _smtpConfigured = false;
let _pendingVerifyEmail = '';
let _pendingVerifyName = '';

/* ═══ ENTRY POINT ═══ */
async function checkAuthAndInit() {
    try {
        // 1. Try DB-backed session first (new email auth)
        const token = getAuthToken();
        if (token) {
            const r = await _originalFetch('/api/auth/check', {
                headers: { 'X-Auth-Token': token }
            });
            if (r.ok) {
                const d = await r.json();
                if (d.authenticated) {
                    setActiveProfile(d.display_name || '');
                    if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state);
                    init(); return;
                }
            }
        }

        // 2+3. Fetch registration status and legacy auth in parallel
        const [regResult, authResult] = await Promise.allSettled([
            _originalFetch('/api/auth/registration-status'),
            _originalFetch('/api/auth-check', {
                headers: { 'X-Auth-Token': token, 'X-Device-Id': getDeviceId() }
            })
        ]);
        if (regResult.status === 'fulfilled' && regResult.value.ok) {
            const reg = await regResult.value.json();
            _hasDbUsers = reg.has_users;
            _openReg = reg.open_registration;
            _smtpConfigured = reg.smtp_configured;
        }
        if (authResult.status === 'fulfilled' && authResult.value.ok) {
            const d = await authResult.value.json();
            if (d.authenticated && d.active_profile) { setActiveProfile(d.active_profile); init(); return; }
            _deviceProfiles = (d.profiles || []).filter(p => p.has_pin);
            _allProfiles = (d.all_profiles || d.profiles || []).filter(p => p.has_pin);
        }

        // 4. Determine auth mode
        _authMode = _hasDbUsers ? 'email' : (_allProfiles.length > 0 ? 'legacy' : 'email');
        _showingAll = false;
        _showLanding();
    } catch (e) { console.error('Auth check failed:', e); init(); }
}
function showLoginScreen() { checkAuthAndInit(); }

/* ═══ BACKDROP HTML (shared) ═══ */
function _backdrop(inner) {
    return `<div class="auth-backdrop" style="background:${_t.bg};">
        <div class="auth-grid-bg" style="--grid-c:${_t.grid};"></div>
        <div class="auth-stars" id="auth-star-field">
            <canvas id="auth-star-canvas"></canvas>
        </div>
        <div class="auth-orb auth-orb-1" style="background:radial-gradient(circle,${_t.orb1} 0%,transparent 70%);"></div>
        <div class="auth-orb auth-orb-2" style="background:radial-gradient(circle,${_t.orb2} 0%,transparent 70%);"></div>
        <div class="auth-orb auth-orb-3" style="background:radial-gradient(circle,${_t.orb3} 0%,transparent 70%);"></div>
        ${inner}
    </div>`;
}

/* ═══ INTERACTIVE STAR CANVAS ENGINE ═══ */
function _initStarParallax() {
    const canvas = document.getElementById('auth-star-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    const isMobile = window.innerWidth <= 768;
    const COUNT = isMobile ? 40 : 300;
    const MOUSE_RADIUS = 160;
    const LINE_RADIUS = 130;
    const LINE_MOUSE_RANGE = 250;
    const DRIFT_SPEED = 0.12;

    // Parse theme star colors into hex for canvas
    function parseColor(rgba) {
        const m = rgba.match(/rgba?\(([^)]+)\)/);
        if (!m) return { r:255, g:255, b:255, a:0.3 };
        const p = m[1].split(',').map(s => parseFloat(s.trim()));
        return { r:p[0], g:p[1], b:p[2], a:p[3] !== undefined ? p[3] : 1 };
    }
    const c1 = parseColor(_t.star1), c2 = parseColor(_t.star2), c3 = parseColor(_t.star3);

    function resize() {
        canvas.width = canvas.parentElement.offsetWidth;
        canvas.height = canvas.parentElement.offsetHeight;
    }
    resize();

    // Pick a star color based on tier distribution
    function pickStar() {
        const r = Math.random();
        if (r < 0.15) return { c: c1, bright: true };   // 15% accent (brightest)
        if (r < 0.35) return { c: c2, bright: true };    // 20% secondary
        return { c: c3, bright: false };                   // 65% dim white
    }

    // Initialize stars
    const stars = [];
    for (let i = 0; i < COUNT; i++) {
        const pick = pickStar();
        const isBright = pick.bright;
        stars.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            baseX: 0, baseY: 0,
            r: isBright ? Math.random() * 2.0 + 0.8 : Math.random() * 1.2 + 0.3,
            a: isBright ? Math.random() * 0.35 + 0.40 : Math.random() * 0.30 + 0.06,
            speed: Math.random() * 0.15 + 0.03,
            phase: Math.random() * Math.PI * 2,
            cr: pick.c.r, cg: pick.c.g, cb: pick.c.b,
            isBright: isBright
        });
        stars[i].baseX = stars[i].x;
        stars[i].baseY = stars[i].y;
    }

    // Shooting stars
    const shooters = [];
    let lastShooter = Date.now();
    const SHOOT_INTERVAL = 5000;

    function spawnShooter() {
        const angle = Math.random() * 0.5 + 0.25;
        const speed = Math.random() * 7 + 5;
        shooters.push({
            x: Math.random() * canvas.width * 0.7,
            y: Math.random() * canvas.height * 0.35,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 1.0,
            len: Math.random() * 45 + 25,
            cr: c1.r, cg: c1.g, cb: c1.b
        });
    }

    // Mouse tracking
    let mouseX = -1000, mouseY = -1000;
    let _raf = 0;

    function onMouseMove(e) {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
    }
    function onMouseLeave() { mouseX = -1000; mouseY = -1000; }

    if (!isTouch) {
        canvas.parentElement.addEventListener('mousemove', onMouseMove);
        canvas.parentElement.addEventListener('mouseleave', onMouseLeave);
    }

    // Main draw loop
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const t = Date.now() * 0.001;
        const now = Date.now();

        // Spawn shooting stars
        if (now - lastShooter > SHOOT_INTERVAL + Math.random() * 3000) {
            spawnShooter();
            lastShooter = now;
        }

        // Collect visible stars near mouse for connection lines
        const nearby = [];

        for (let i = 0; i < stars.length; i++) {
            const s = stars[i];

            // Gentle upward drift
            s.baseY -= DRIFT_SPEED;
            if (s.baseY < -10) {
                s.baseY = canvas.height + 10;
                s.baseX = Math.random() * canvas.width;
                s.y = s.baseY;
                s.x = s.baseX;
            }

            let drawX = s.baseX, drawY = s.baseY;
            const dx = drawX - mouseX, dy = drawY - mouseY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // Mouse repulsion (desktop only)
            if (!isTouch && dist < MOUSE_RADIUS && dist > 0) {
                const force = (1 - dist / MOUSE_RADIUS) * 28;
                drawX += (dx / dist) * force;
                drawY += (dy / dist) * force;
            }

            // Smooth interpolation
            s.x += (drawX - s.x) * 0.07;
            s.y += (drawY - s.y) * 0.07;

            if (s.y < -15 || s.y > canvas.height + 15) continue;

            // Twinkle
            const flicker = 0.65 + 0.35 * Math.sin(t * s.speed * 4 + s.phase);
            const proxBoost = (!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.5 : 1;
            const alpha = Math.min(1, s.a * flicker * proxBoost);
            const radius = s.r * ((!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.4 : 1);

            ctx.globalAlpha = alpha;
            ctx.fillStyle = `rgb(${s.cr},${s.cg},${s.cb})`;
            ctx.beginPath();
            ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
            ctx.fill();

            // Soft glow on bright stars near cursor
            if (!isTouch && s.isBright && dist < MOUSE_RADIUS * 0.7) {
                ctx.globalAlpha = alpha * 0.15;
                ctx.beginPath();
                ctx.arc(s.x, s.y, radius * 4, 0, Math.PI * 2);
                ctx.fill();
            }

            // Track nearby stars for connection lines
            if (!isTouch) {
                const mDist = Math.sqrt((s.x - mouseX) ** 2 + (s.y - mouseY) ** 2);
                if (mDist < LINE_MOUSE_RANGE) {
                    nearby.push({ x: s.x, y: s.y, a: alpha, mDist: mDist });
                }
            }
        }

        // Connection lines between nearby stars (constellation effect)
        if (!isTouch && nearby.length > 1) {
            const accent = _t.accent;
            for (let a = 0; a < nearby.length; a++) {
                for (let b = a + 1; b < nearby.length; b++) {
                    const ddx = nearby[a].x - nearby[b].x;
                    const ddy = nearby[a].y - nearby[b].y;
                    const d = Math.sqrt(ddx * ddx + ddy * ddy);
                    if (d < LINE_RADIUS) {
                        let lineAlpha = (1 - d / LINE_RADIUS) * 0.16;
                        const avgMD = (nearby[a].mDist + nearby[b].mDist) / 2;
                        lineAlpha *= Math.max(0, (1 - avgMD / LINE_MOUSE_RANGE) * 1.4);
                        lineAlpha = Math.min(lineAlpha, 0.18);
                        ctx.globalAlpha = lineAlpha;
                        ctx.strokeStyle = accent;
                        ctx.lineWidth = 0.6;
                        ctx.beginPath();
                        ctx.moveTo(nearby[a].x, nearby[a].y);
                        ctx.lineTo(nearby[b].x, nearby[b].y);
                        ctx.stroke();
                    }
                }
            }
        }

        // Draw shooting stars
        for (let si = shooters.length - 1; si >= 0; si--) {
            const sh = shooters[si];
            sh.x += sh.vx;
            sh.y += sh.vy;
            sh.life -= 0.014;
            if (sh.life <= 0 || sh.x > canvas.width + 60 || sh.y > canvas.height + 60) {
                shooters.splice(si, 1);
                continue;
            }
            const speed = Math.sqrt(sh.vx * sh.vx + sh.vy * sh.vy);
            const tailX = sh.x - sh.vx * (sh.len / speed);
            const tailY = sh.y - sh.vy * (sh.len / speed);
            const grad = ctx.createLinearGradient(tailX, tailY, sh.x, sh.y);
            grad.addColorStop(0, 'transparent');
            grad.addColorStop(1, `rgba(${sh.cr},${sh.cg},${sh.cb},${sh.life * 0.6})`);
            ctx.globalAlpha = sh.life * 0.7;
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(tailX, tailY);
            ctx.lineTo(sh.x, sh.y);
            ctx.stroke();
            // Bright head
            ctx.globalAlpha = sh.life * 0.85;
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(sh.x, sh.y, 1.4, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.globalAlpha = 1;
        _raf = requestAnimationFrame(draw);
    }

    _raf = requestAnimationFrame(draw);

    // Resize handling
    window.addEventListener('resize', () => {
        resize();
        for (let i = 0; i < stars.length; i++) {
            stars[i].baseX = Math.random() * canvas.width;
            stars[i].baseY = Math.random() * canvas.height;
            stars[i].x = stars[i].baseX;
            stars[i].y = stars[i].baseY;
        }
    });

    /* Cleanup on overlay removal */
    const obs = new MutationObserver(() => {
        if (!document.getElementById('auth-star-field')) {
            cancelAnimationFrame(_raf);
            canvas.parentElement?.removeEventListener('mousemove', onMouseMove);
            canvas.parentElement?.removeEventListener('mouseleave', onMouseLeave);
            obs.disconnect();
        }
    });
    obs.observe(document.body, { childList: true, subtree: true });
}

/* ═══ SCREEN: LANDING ═══ */
function _showLanding() {
    _removeOverlay();
    const app = document.querySelector('.flex.h-screen');
    if (app) app.style.display = 'none';
    const ov = document.createElement('div');
    ov.id = 'auth-overlay';
    const hasProfiles = _hasDbUsers || _allProfiles.length > 0;
    const showLegacy = _authMode === 'legacy' && _allProfiles.length > 0;
    ov.innerHTML = _backdrop(`
        <div class="auth-landing">
            <div class="auth-logo-block">
                <div class="auth-logo-icon" style="color:${_t.accent};">
                    <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/>
                        <line x1="12" y1="22" x2="12" y2="15.5"/>
                        <polyline points="22 8.5 12 15.5 2 8.5"/>
                        <polyline points="2 15.5 12 8.5 22 15.5"/>
                        <line x1="12" y1="2" x2="12" y2="8.5"/>
                    </svg>
                </div>
                <div class="auth-brand">STRAT<span style="color:${_t.accent};">_</span>OS</div>
            </div>
            <div class="auth-tagline-sub">
                Careers &middot; Markets &middot; Tech &middot; Deals<br>
                <span class="auth-tagline-dim">Filtered. Scored. Ready.</span>
            </div>
            <div class="auth-landing-actions">
                ${hasProfiles ? `<button class="auth-btn-hero" style="--h-accent:${_t.accent};--h-rgb:${_t.rgb};" onclick="_showScreen('login')">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg>
                    Log In</button>` : ''}
                <button class="auth-btn-secondary" onclick="_authMode='email';_showScreen('register')">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4-4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/></svg>
                    Create Account</button>
                ${showLegacy ? `<button class="auth-btn-link" style="color:${_t.accent};" onclick="_authMode='legacy';_showScreen('login')">Use PIN login</button>` : ''}
            </div>
            <div class="auth-landing-footer"><span class="auth-version">v3.0</span></div>
        </div>`);
    document.body.appendChild(ov);
    _initStarParallax();
}

/* ═══ SCREEN ROUTER ═══ */
function _showScreen(mode) {
    _removeOverlay();
    const app = document.querySelector('.flex.h-screen');
    if (app) app.style.display = 'none';
    const ov = document.createElement('div');
    ov.id = 'auth-overlay';
    ov.innerHTML = _backdrop(`
        <div class="auth-box">
            <button class="auth-back-btn" onclick="_authMode=_hasDbUsers?'email':'legacy';_showLanding()" title="Back">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
            </button>
            <div class="auth-brand-sm">STRAT<span style="color:${_t.accent};">_</span>OS</div>
            <div id="auth-body"></div>
        </div>`);
    document.body.appendChild(ov);
    _initStarParallax();
    if (mode === 'register') {
        _authMode === 'legacy' ? _renderRegister() : _renderEmailRegister();
    } else if (mode === 'verify') {
        _renderVerify();
    } else if (mode === 'otp-request') {
        _renderOtpRequest();
    } else if (mode === 'otp-verify') {
        _renderOtpVerify();
    } else {
        _authMode === 'legacy' ? _renderLogin() : _renderEmailLogin();
    }
}

/* ═══ REGISTER ═══ */
function _renderRegister() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Create your profile</div>
        <div class="auth-screen-hint">Set up your intelligence dashboard</div>
        <div class="auth-form">
            <div class="auth-field-group">
                <label class="auth-label">Username</label>
                <input id="reg-name" class="auth-input" type="text" placeholder="e.g. Ahmad" maxlength="30" autocomplete="off" style="--fc:${_t.accent};">
            </div>
            <div class="auth-field-group">
                <label class="auth-label">PIN <span class="auth-label-hint">(min 4 characters)</span></label>
                <div class="pw-input-wrap"><input id="reg-pin" class="auth-input auth-pin-input" type="password" placeholder="••••" maxlength="20" autocomplete="off" style="--fc:${_t.accent};">${_pwEye('reg-pin')}</div>
            </div>
            <div class="auth-field-group">
                <label class="auth-label">Confirm PIN</label>
                <div class="pw-input-wrap"><input id="reg-pin2" class="auth-input auth-pin-input" type="password" placeholder="••••" maxlength="20" autocomplete="off"
                    onkeydown="if(event.key==='Enter')_doRegister()" style="--fc:${_t.accent};">${_pwEye('reg-pin2')}</div>
            </div>
            <div class="auth-error" id="reg-error"></div>
            <button class="auth-btn-primary" id="reg-btn" onclick="_doRegister()" style="--ba:${_t.accent};--br:${_t.rgb};">Create Profile</button>
        </div>`;
    setTimeout(() => document.getElementById('reg-name')?.focus(), 120);
}

async function _doRegister() {
    const name = document.getElementById('reg-name').value.trim();
    const pin  = document.getElementById('reg-pin').value;
    const pin2 = document.getElementById('reg-pin2').value;
    const err  = document.getElementById('reg-error');
    const btn  = document.getElementById('reg-btn');
    if (!name)                { err.textContent = 'Enter a display name'; return; }
    if (name.length < 2)     { err.textContent = 'Name must be at least 2 characters'; return; }
    if (!pin || pin.length<4) { err.textContent = 'PIN must be at least 4 characters'; return; }
    if (pin !== pin2)         { err.textContent = 'PINs do not match'; return; }
    btn.disabled = true; btn.textContent = 'Creating...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/register', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, pin, device_id: getDeviceId() })
        });
        const d = await r.json();
        if (r.ok && d.token) { setAuthToken(d.token); setActiveProfile(d.profile); _clearProfileLocalStorage(); if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state); _dismiss(); }
        else { err.textContent = d.error || 'Registration failed'; btn.disabled = false; btn.textContent = 'Create Profile'; }
    } catch (e) { err.textContent = 'Connection error'; btn.disabled = false; btn.textContent = 'Create Profile'; }
}

/* ═══ LOGIN ═══ */
// Profile card colors now derive from the active auth theme
const _CARD_AC = _t.accent;
const _CARD_BG = `rgba(${_t.rgb},0.10)`;
function _ini(n) { return n.split(/[\s_-]+/).map(w=>(w[0]||'')).join('').toUpperCase().slice(0,2); }

function _renderLogin() {
    const profiles = _showingAll ? _allProfiles : _deviceProfiles;
    const hasOthers = _allProfiles.length > _deviceProfiles.length;
    let content = '';
    if (profiles.length === 0 && !_showingAll) {
        content = `<div class="auth-screen-title">Welcome back</div>
            <div class="auth-screen-hint">No profiles found on this device</div>
            <div class="auth-empty-actions">
                ${_allProfiles.length > 0 ? `<button class="auth-btn-primary" onclick="_showingAll=true;_renderLogin()" style="--ba:${_t.accent};--br:${_t.rgb};">Show all profiles</button>
                    <div class="auth-empty-hint">Your profile may be registered on another device</div>` :
                    `<div class="auth-empty-hint">No accounts exist yet</div>`}
                <button class="auth-btn-link" style="color:${_t.accent};" onclick="_showScreen('register')">+ Create new profile</button>
            </div>`;
    } else {
        const cards = profiles.map((p, i) => {
            const ac = _CARD_AC, bg = _CARD_BG;
            const n = p.name.replace(/'/g, "\\'");
            return `<button class="auth-card" style="--ac:${ac};--bg:${bg};" onclick="_selectProfile('${n}',${i})">
                <div class="auth-card-lock"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></div>
                <div class="auth-avatar" style="background:${ac}18;border-color:${ac}50;color:${ac};">${_ini(p.name)}</div>
                <div class="auth-card-name">${p.name}</div>
                <div class="auth-card-role">${p.role || p.location || ''}</div>
            </button>`;
        }).join('');
        content = `<div class="auth-screen-title">${_showingAll ? 'All profiles' : 'Welcome back'}</div>
            <div class="auth-screen-hint">Select your profile to continue</div>
            <div class="auth-grid" id="auth-grid">${cards}</div>
            <div id="login-pin-box" style="display:none;" class="auth-form">
                <div class="auth-pin-header">
                    <div class="auth-pin-av" id="login-av"></div>
                    <span class="auth-pin-name" id="login-name"></span>
                    <button class="auth-pin-back" onclick="_backToGrid()" title="Back">&times;</button>
                </div>
                <div class="pw-input-wrap"><input type="password" id="login-pin" class="auth-input auth-pin-input" placeholder="Enter PIN" maxlength="20" autocomplete="off"
                    onkeydown="if(event.key==='Enter')_doLogin()" style="--fc:${_t.accent};">${_pwEye('login-pin')}</div>
                <div class="auth-error" id="login-error"></div>
                <button class="auth-btn-primary" id="login-btn" onclick="_doLogin()" style="--ba:${_t.accent};--br:${_t.rgb};">Login</button>
            </div>
            <div class="auth-login-footer" id="auth-login-footer">
                ${hasOthers && !_showingAll ? `<button class="auth-btn-link" onclick="_showingAll=true;_renderLogin()">Show all profiles</button>` : ''}
                ${_showingAll && _deviceProfiles.length > 0 ? `<button class="auth-btn-link" onclick="_showingAll=false;_renderLogin()">&larr; Show only this device</button>` : ''}
            </div>`;
    }
    document.getElementById('auth-body').innerHTML = content;
}

let _loginProfile = '', _loginIdx = 0;

function _selectProfile(name, idx) {
    _loginProfile = name; _loginIdx = idx;
    const ac = _CARD_AC;
    document.getElementById('auth-grid').style.display = 'none';
    document.getElementById('login-pin-box').style.display = '';
    document.querySelector('.auth-screen-title').textContent = 'Enter your PIN';
    document.querySelector('.auth-screen-hint').textContent = '';
    document.getElementById('login-name').textContent = name;
    const av = document.getElementById('login-av');
    av.textContent = _ini(name);
    av.style.cssText = `background:${ac}18;border:2px solid ${ac}60;color:${ac};`;
    document.getElementById('login-pin').value = '';
    document.getElementById('login-error').textContent = '';
    const footer = document.getElementById('auth-login-footer');
    if (footer) footer.style.display = 'none';
    setTimeout(() => document.getElementById('login-pin')?.focus(), 100);
}

function _backToGrid() {
    document.getElementById('auth-grid').style.display = '';
    document.getElementById('login-pin-box').style.display = 'none';
    document.querySelector('.auth-screen-title').textContent = _showingAll ? 'All profiles' : 'Welcome back';
    document.querySelector('.auth-screen-hint').textContent = 'Select your profile to continue';
    const footer = document.getElementById('auth-login-footer');
    if (footer) footer.style.display = '';
    _loginProfile = '';
}

async function _doLogin() {
    const pin = document.getElementById('login-pin').value;
    const err = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');
    if (!pin) { err.textContent = 'Enter your PIN'; return; }
    btn.disabled = true; btn.textContent = '...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile: _loginProfile, pin })
        });
        const d = await r.json();
        if (r.ok && d.token) { setAuthToken(d.token); setActiveProfile(d.profile); _clearProfileLocalStorage(); if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state); _dismiss(); }
        else { err.textContent = d.error || 'Login failed'; const inp = document.getElementById('login-pin'); inp.value = ''; _shake(inp); inp.focus(); }
    } catch (e) { err.textContent = 'Connection error'; }
    btn.disabled = false; btn.textContent = 'Login';
}

/* ═══ EMAIL REGISTER ═══ */
function _renderEmailRegister() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Create your account</div>
        <div class="auth-screen-hint">Set up your intelligence dashboard</div>
        <div class="auth-form">
            <div class="auth-field-group">
                <label class="auth-label">Username</label>
                <input id="reg-name" class="auth-input" type="text" placeholder="e.g. Ahmad" maxlength="30" autocomplete="name" style="--fc:${_t.accent};">
            </div>
            <div class="auth-field-group">
                <label class="auth-label">Email</label>
                <input id="reg-email" class="auth-input" type="email" placeholder="you@example.com" autocomplete="email" style="--fc:${_t.accent};">
            </div>
            <div class="auth-field-group">
                <label class="auth-label">Password <span class="auth-label-hint">(min 8 characters)</span></label>
                <div class="pw-input-wrap"><input id="reg-pass" class="auth-input" type="password" placeholder="••••••••" autocomplete="new-password" style="--fc:${_t.accent};" oninput="_updatePasswordStrength()">${_pwEye('reg-pass')}</div>
                <div id="pw-strength-bar" class="pw-strength-bar" style="display:none;">
                    <div id="pw-strength-fill" class="pw-strength-fill"></div>
                </div>
                <div id="pw-strength-label" class="pw-strength-label"></div>
            </div>
            <div class="auth-field-group">
                <label class="auth-label">Confirm Password</label>
                <div class="pw-input-wrap"><input id="reg-pass2" class="auth-input" type="password" placeholder="••••••••" autocomplete="new-password"
                    onkeydown="if(event.key==='Enter')_doEmailRegister()" oninput="_updatePasswordMatch()" style="--fc:${_t.accent};">${_pwEye('reg-pass2')}</div>
                <div id="pw-match-label" class="pw-match-label"></div>
            </div>
            <div class="auth-error" id="reg-error"></div>
            <button class="auth-btn-primary" id="reg-btn" onclick="_doEmailRegister()" style="--ba:${_t.accent};--br:${_t.rgb};">Create Account</button>
        </div>`;
    setTimeout(() => document.getElementById('reg-name')?.focus(), 120);
}

function _togglePw(id) {
    const el = document.getElementById(id);
    if (!el) return;
    const isHidden = el.type === 'password';
    el.type = isHidden ? 'text' : 'password';
    const btn = el.parentElement.querySelector('.pw-eye-btn');
    if (btn) btn.innerHTML = isHidden
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
}

function _pwEye(id) {
    return `<button type="button" class="pw-eye-btn" onclick="_togglePw('${id}')" tabindex="-1"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></button>`;
}

function _getPasswordStrength(pw) {
    if (!pw) return { score: 0, label: '', color: '' };
    let score = 0;
    if (pw.length >= 8) score++;
    if (pw.length >= 12) score++;
    if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) score++;
    if (/\d/.test(pw)) score++;
    if (/[^a-zA-Z0-9]/.test(pw)) score++;
    if (score <= 1) return { score: 1, label: 'Weak', color: '#f87171' };
    if (score <= 2) return { score: 2, label: 'Fair', color: '#fb923c' };
    if (score <= 3) return { score: 3, label: 'Good', color: '#facc15' };
    return { score: 4, label: 'Strong', color: '#34d399' };
}

function _updatePasswordStrength() {
    const pw = document.getElementById('reg-pass')?.value || '';
    const bar = document.getElementById('pw-strength-bar');
    const fill = document.getElementById('pw-strength-fill');
    const label = document.getElementById('pw-strength-label');
    if (!bar || !fill || !label) return;
    if (!pw) { bar.style.display = 'none'; label.textContent = ''; return; }
    const s = _getPasswordStrength(pw);
    bar.style.display = '';
    fill.style.width = `${s.score * 25}%`;
    fill.style.background = s.color;
    label.textContent = s.label;
    label.style.color = s.color;
    _updatePasswordMatch();
}

function _updatePasswordMatch() {
    const pw = document.getElementById('reg-pass')?.value || '';
    const pw2 = document.getElementById('reg-pass2')?.value || '';
    const label = document.getElementById('pw-match-label');
    if (!label) return;
    if (!pw2) { label.textContent = ''; return; }
    if (pw === pw2) { label.textContent = 'Passwords match'; label.style.color = '#34d399'; }
    else { label.textContent = 'Passwords do not match'; label.style.color = '#f87171'; }
}

async function _doEmailRegister() {
    const name = document.getElementById('reg-name').value.trim();
    const email = document.getElementById('reg-email').value.trim();
    const pass = document.getElementById('reg-pass').value;
    const pass2 = document.getElementById('reg-pass2').value;
    const err = document.getElementById('reg-error');
    const btn = document.getElementById('reg-btn');
    if (!name || name.length < 2) { err.textContent = 'Username must be at least 2 characters'; return; }
    if (!email || !email.includes('@')) { err.textContent = 'Enter a valid email'; return; }
    if (!pass || pass.length < 8) { err.textContent = 'Password must be at least 8 characters'; return; }
    if (pass !== pass2) { err.textContent = 'Passwords do not match'; return; }
    btn.disabled = true; btn.textContent = 'Creating...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/register', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password: pass, display_name: name })
        });
        const d = await r.json();
        if (r.ok) {
            // Registration always requires email verification now
            _pendingVerifyEmail = email;
            _pendingVerifyName = name;
            _showScreen('verify');
        } else {
            err.textContent = d.error || 'Registration failed';
            btn.disabled = false; btn.textContent = 'Create Account';
        }
    } catch (e) { err.textContent = 'Connection error'; btn.disabled = false; btn.textContent = 'Create Account'; }
}

/* ═══ EMAIL LOGIN ═══ */
function _renderEmailLogin() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Welcome back</div>
        <div class="auth-screen-hint">Sign in to your dashboard</div>
        <div class="auth-form">
            <div class="auth-field-group">
                <label class="auth-label">Email or Username</label>
                <input id="login-email" class="auth-input" type="text" placeholder="you@example.com" autocomplete="email" style="--fc:${_t.accent};">
            </div>
            <div class="auth-field-group">
                <label class="auth-label">Password</label>
                <div class="pw-input-wrap"><input id="login-pass" class="auth-input" type="password" placeholder="••••••••" autocomplete="current-password"
                    onkeydown="if(event.key==='Enter')_doEmailLogin()" style="--fc:${_t.accent};">${_pwEye('login-pass')}</div>
            </div>
            <div class="auth-error" id="login-error"></div>
            <button class="auth-btn-primary" id="login-btn" onclick="_doEmailLogin()" style="--ba:${_t.accent};--br:${_t.rgb};">Sign In</button>
            <button class="auth-btn-link" style="color:${_t.accent};margin-top:12px;" onclick="_showForgotPassword()">Forgot password?</button>
            ${_smtpConfigured ? `<button class="auth-btn-link" style="color:${_t.accent};margin-top:4px;" onclick="_showScreen('otp-request')">Sign in with email code</button>` : ''}
        </div>
        ${_allProfiles.length > 0 ? `<div style="margin-top:16px;"><button class="auth-btn-link" onclick="_authMode='legacy';_renderLogin()">Use PIN login instead</button></div>` : ''}`;
    setTimeout(() => document.getElementById('login-email')?.focus(), 120);
}

async function _doEmailLogin() {
    const identifier = document.getElementById('login-email').value.trim();
    const pass = document.getElementById('login-pass').value;
    const err = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');
    if (!identifier) { err.textContent = 'Enter your email or username'; return; }
    if (!pass) { err.textContent = 'Enter your password'; return; }
    btn.disabled = true; btn.textContent = 'Signing in...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/login', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: identifier, password: pass })
        });
        const d = await r.json();
        if (r.ok && d.token) {
            setAuthToken(d.token); setActiveProfile(d.display_name || '');
            _clearProfileLocalStorage();
            if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state);
            _dismiss();
        } else {
            err.textContent = d.error || 'Login failed';
            const inp = document.getElementById('login-pass'); inp.value = ''; _shake(inp); inp.focus();
        }
    } catch (e) { err.textContent = 'Connection error'; }
    btn.disabled = false; btn.textContent = 'Sign In';
}

/* ═══ EMAIL VERIFICATION ═══ */
function _renderVerify() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Verify your email</div>
        <div class="auth-screen-hint">Enter the 5-digit code sent to<br><strong style="color:${_t.accent};">${_pendingVerifyEmail || 'your email'}</strong></div>
        <div class="auth-form">
            <div class="auth-field-group">
                <input id="verify-code" class="auth-input auth-pin-input" type="text" placeholder="00000" maxlength="5" autocomplete="one-time-code"
                    oninput="if(this.value.length===5)_doVerify()" onkeydown="if(event.key==='Enter')_doVerify()" style="--fc:${_t.accent};font-size:24px;letter-spacing:8px;">
            </div>
            <div class="auth-error" id="verify-error"></div>
            <button class="auth-btn-primary" id="verify-btn" onclick="_doVerify()" style="--ba:${_t.accent};--br:${_t.rgb};">Verify & Sign In</button>
            <button class="auth-btn-link" id="resend-btn" style="color:${_t.accent};margin-top:12px;" onclick="_doResend()">Resend code</button>
        </div>`;
    setTimeout(() => document.getElementById('verify-code')?.focus(), 120);
}

async function _doVerify() {
    const code = document.getElementById('verify-code').value.trim();
    const err = document.getElementById('verify-error');
    const btn = document.getElementById('verify-btn');
    if (!code || code.length !== 5) { err.textContent = 'Enter the 5-digit code'; return; }
    btn.disabled = true; btn.textContent = 'Verifying...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/verify', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: _pendingVerifyEmail, code })
        });
        const d = await r.json();
        if (r.ok && d.token) {
            setAuthToken(d.token);
            setActiveProfile(d.display_name || _pendingVerifyName || '');
            _clearProfileLocalStorage();
            if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state);
            _dismiss();
        } else {
            err.textContent = d.error || 'Verification failed';
            document.getElementById('verify-code').value = '';
            _shake(document.getElementById('verify-code'));
            btn.disabled = false; btn.textContent = 'Verify & Sign In';
        }
    } catch (e) { err.textContent = 'Connection error'; btn.disabled = false; btn.textContent = 'Verify & Sign In'; }
}

async function _doResend() {
    const btn = document.getElementById('resend-btn');
    btn.disabled = true; btn.textContent = 'Sending...';
    try {
        await _originalFetch('/api/auth/resend-verification', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: _pendingVerifyEmail })
        });
        btn.textContent = 'Code sent!';
        setTimeout(() => { btn.disabled = false; btn.textContent = 'Resend code'; }, 60000);
    } catch (e) { btn.disabled = false; btn.textContent = 'Resend code'; }
}

/* ═══ FORGOT PASSWORD ═══ */
function _showForgotPassword() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Reset password</div>
        <div class="auth-screen-hint">Enter your email to receive a reset code</div>
        <div class="auth-form">
            <div class="auth-field-group">
                <label class="auth-label">Email</label>
                <input id="reset-email" class="auth-input" type="email" placeholder="you@example.com" autocomplete="email"
                    onkeydown="if(event.key==='Enter')_doForgotPassword()" style="--fc:${_t.accent};">
            </div>
            <div class="auth-error" id="reset-error"></div>
            <div class="auth-success" id="reset-success" style="display:none;"></div>
            <button class="auth-btn-primary" id="reset-btn" onclick="_doForgotPassword()" style="--ba:${_t.accent};--br:${_t.rgb};">Send Reset Code</button>
        </div>`;
    setTimeout(() => document.getElementById('reset-email')?.focus(), 120);
}

async function _doForgotPassword() {
    const email = document.getElementById('reset-email').value.trim();
    const err = document.getElementById('reset-error');
    const btn = document.getElementById('reset-btn');
    if (!email) { err.textContent = 'Enter your email'; return; }
    btn.disabled = true; btn.textContent = 'Sending...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/forgot-password', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        if (r.ok) { _showResetPassword(email); }
        else { const d = await r.json(); err.textContent = d.error || 'Failed'; btn.disabled = false; btn.textContent = 'Send Reset Code'; }
    } catch (e) { err.textContent = 'Connection error'; btn.disabled = false; btn.textContent = 'Send Reset Code'; }
}

function _showResetPassword(email) {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Enter reset code</div>
        <div class="auth-screen-hint">Check your email for the 6-digit code</div>
        <div class="auth-form">
            <div class="auth-field-group">
                <label class="auth-label">Reset Code</label>
                <input id="rp-code" class="auth-input auth-pin-input" type="text" placeholder="000000" maxlength="6" autocomplete="one-time-code" style="--fc:${_t.accent};">
            </div>
            <div class="auth-field-group">
                <label class="auth-label">New Password <span class="auth-label-hint">(min 8 characters)</span></label>
                <div class="pw-input-wrap"><input id="rp-pass" class="auth-input" type="password" placeholder="••••••••" autocomplete="new-password"
                    onkeydown="if(event.key==='Enter')_doResetPassword('${email}')" style="--fc:${_t.accent};">${_pwEye('rp-pass')}</div>
            </div>
            <div class="auth-error" id="rp-error"></div>
            <button class="auth-btn-primary" id="rp-btn" onclick="_doResetPassword('${email}')" style="--ba:${_t.accent};--br:${_t.rgb};">Reset Password</button>
        </div>`;
    setTimeout(() => document.getElementById('rp-code')?.focus(), 120);
}

async function _doResetPassword(email) {
    const code = document.getElementById('rp-code').value.trim();
    const pass = document.getElementById('rp-pass').value;
    const err = document.getElementById('rp-error');
    const btn = document.getElementById('rp-btn');
    if (!code || code.length !== 6) { err.textContent = 'Enter the 6-digit code'; return; }
    if (!pass || pass.length < 8) { err.textContent = 'Password must be at least 8 characters'; return; }
    btn.disabled = true; btn.textContent = 'Resetting...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/reset-password', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, code, new_password: pass })
        });
        const d = await r.json();
        if (r.ok) {
            _authMode = 'email';
            _showScreen('login');
        } else { err.textContent = d.error || 'Reset failed'; btn.disabled = false; btn.textContent = 'Reset Password'; }
    } catch (e) { err.textContent = 'Connection error'; btn.disabled = false; btn.textContent = 'Reset Password'; }
}

/* ═══ OTP (EMAIL CODE) LOGIN ═══ */
let _otpEmail = '';

function _renderOtpRequest() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Sign in with email code</div>
        <div class="auth-screen-hint">We'll send a one-time code to your email</div>
        <div class="auth-form">
            <div class="auth-field-group">
                <label class="auth-label">Email</label>
                <input id="otp-email" class="auth-input" type="email" placeholder="you@example.com" autocomplete="email"
                    onkeydown="if(event.key==='Enter')_doOtpRequest()" style="--fc:${_t.accent};">
            </div>
            <div class="auth-error" id="otp-req-error"></div>
            <button class="auth-btn-primary" id="otp-req-btn" onclick="_doOtpRequest()" style="--ba:${_t.accent};--br:${_t.rgb};">Send Code</button>
            <button class="auth-btn-link" style="color:${_t.accent};margin-top:12px;" onclick="_renderEmailLogin()">Sign in with password instead</button>
        </div>`;
    setTimeout(() => document.getElementById('otp-email')?.focus(), 120);
}

async function _doOtpRequest() {
    const email = document.getElementById('otp-email').value.trim();
    const err = document.getElementById('otp-req-error');
    const btn = document.getElementById('otp-req-btn');
    if (!email || !email.includes('@')) { err.textContent = 'Enter a valid email'; return; }
    btn.disabled = true; btn.textContent = 'Sending...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/otp-request', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        const d = await r.json();
        if (r.ok) {
            _otpEmail = email;
            _renderOtpVerify();
        } else {
            err.textContent = d.error || 'Failed to send code';
            btn.disabled = false; btn.textContent = 'Send Code';
        }
    } catch (e) { err.textContent = 'Connection error'; btn.disabled = false; btn.textContent = 'Send Code'; }
}

function _renderOtpVerify() {
    document.getElementById('auth-body').innerHTML = `
        <div class="auth-screen-title">Enter your login code</div>
        <div class="auth-screen-hint">Enter the 5-digit code sent to<br><strong style="color:${_t.accent};">${_otpEmail || 'your email'}</strong></div>
        <div class="auth-form">
            <div class="auth-field-group">
                <input id="otp-code" class="auth-input auth-pin-input" type="text" placeholder="00000" maxlength="5" autocomplete="one-time-code"
                    oninput="if(this.value.length===5)_doOtpVerify()" onkeydown="if(event.key==='Enter')_doOtpVerify()" style="--fc:${_t.accent};font-size:24px;letter-spacing:8px;">
            </div>
            <div class="auth-error" id="otp-verify-error"></div>
            <button class="auth-btn-primary" id="otp-verify-btn" onclick="_doOtpVerify()" style="--ba:${_t.accent};--br:${_t.rgb};">Sign In</button>
            <button class="auth-btn-link" id="otp-resend-btn" style="color:${_t.accent};margin-top:12px;" onclick="_doOtpResend()">Resend code</button>
        </div>`;
    setTimeout(() => document.getElementById('otp-code')?.focus(), 120);
}

async function _doOtpVerify() {
    const code = document.getElementById('otp-code').value.trim();
    const err = document.getElementById('otp-verify-error');
    const btn = document.getElementById('otp-verify-btn');
    if (!code || code.length !== 5) { err.textContent = 'Enter the 5-digit code'; return; }
    btn.disabled = true; btn.textContent = 'Verifying...'; err.textContent = '';
    try {
        const r = await _originalFetch('/api/auth/otp-verify', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: _otpEmail, code })
        });
        const d = await r.json();
        if (r.ok && d.token) {
            setAuthToken(d.token); setActiveProfile(d.display_name || '');
            _clearProfileLocalStorage();
            if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state);
            _dismiss();
        } else {
            err.textContent = d.error || 'Invalid code';
            const inp = document.getElementById('otp-code'); inp.value = ''; _shake(inp); inp.focus();
        }
    } catch (e) { err.textContent = 'Connection error'; }
    btn.disabled = false; btn.textContent = 'Sign In';
}

async function _doOtpResend() {
    const btn = document.getElementById('otp-resend-btn');
    const err = document.getElementById('otp-verify-error');
    btn.disabled = true; btn.textContent = 'Sending...';
    try {
        const r = await _originalFetch('/api/auth/otp-request', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: _otpEmail })
        });
        if (r.ok) {
            btn.textContent = 'Code sent!';
            err.textContent = '';
            setTimeout(() => { btn.disabled = false; btn.textContent = 'Resend code'; }, 60000);
        } else {
            btn.disabled = false; btn.textContent = 'Resend code';
        }
    } catch (e) { btn.disabled = false; btn.textContent = 'Resend code'; }
}

/* ═══ LOGOUT & HELPERS ═══ */
/** Clear per-profile localStorage keys on login/register/switch to prevent bleed */
function _clearProfileLocalStorage() {
    // Remove all profile-scoped localStorage keys
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (!k) continue;
        // Preserved — auth tokens, device ID, theme, and tour preference persist across sessions
        if (k === AUTH_TOKEN_KEY || k === AUTH_PROFILE_KEY || k === DEVICE_ID_KEY || k === AUTH_THEME_KEY || k === 'stratos_tour_never') continue;
        keysToRemove.push(k);
    }
    keysToRemove.forEach(k => localStorage.removeItem(k));
    // Clear in-memory data to prevent cross-profile bleed
    if (typeof newsData !== 'undefined') { newsData = []; }
    if (typeof marketData !== 'undefined') { marketData = {}; }
    if (typeof financeNewsData !== 'undefined') financeNewsData.length = 0;
    if (typeof politicsNewsData !== 'undefined') politicsNewsData.length = 0;
    if (typeof customNewsData !== 'undefined') customNewsData.length = 0;
    if (typeof configData !== 'undefined') { configData = null; }
}
async function logout() {
    const token = getAuthToken();
    try { await _originalFetch('/api/auth/logout', { method:'POST', headers:{'X-Auth-Token':token} }); } catch(e){}
    try { await _originalFetch('/api/logout', { method:'POST', headers:{'X-Auth-Token':token} }); } catch(e){}
    _clearProfileLocalStorage();
    clearAuthToken(); location.reload();
}
function _removeOverlay() { const old = document.getElementById('auth-overlay'); if (old) old.remove(); }
function _dismiss() {
    const ov = document.getElementById('auth-overlay');
    if (!ov) return;
    ov.style.transition = 'opacity 0.35s ease'; ov.style.opacity = '0';
    setTimeout(() => { ov.remove(); const app = document.querySelector('.flex.h-screen'); if (app) app.style.display = ''; init(); }, 360);
}
function _shake(el) { el.style.animation='none'; el.offsetHeight; el.style.animation='authShake .4s ease'; }

/* ═══ STYLES ═══ */
const _css = document.createElement('style');
_css.textContent = `
@keyframes authShake { 0%,100%{transform:translateX(0)} 20%,60%{transform:translateX(-6px)} 40%,80%{transform:translateX(6px)} }
@keyframes authFadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes authFloat { 0%,100%{transform:translate(0,0)} 33%{transform:translate(30px,-20px)} 66%{transform:translate(-20px,15px)} }
@keyframes authFloat2 { 0%,100%{transform:translate(0,0)} 33%{transform:translate(-25px,20px)} 66%{transform:translate(15px,-25px)} }
@keyframes authFloat3 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(20px,20px)} }
@keyframes authPulse { 0%,100%{opacity:.18} 50%{opacity:.38} }
@keyframes authGlowLine { 0%{transform:translateX(-100%)} 100%{transform:translateX(200%)} }

.auth-backdrop { position:fixed; inset:0; z-index:9999; display:flex; align-items:center; justify-content:center; overflow:hidden; }
.auth-grid-bg {
    position:absolute; inset:0; pointer-events:none;
    background-image: linear-gradient(var(--grid-c) 1px, transparent 1px), linear-gradient(90deg, var(--grid-c) 1px, transparent 1px);
    background-size:60px 60px;
    mask-image:radial-gradient(ellipse 60% 50% at 50% 50%, black 20%, transparent 70%);
    -webkit-mask-image:radial-gradient(ellipse 60% 50% at 50% 50%, black 20%, transparent 70%);
}
.auth-orb { position:absolute; border-radius:50%; filter:blur(100px); pointer-events:none; animation-timing-function:ease-in-out; animation-iteration-count:infinite; }
.auth-orb-1 { width:550px; height:550px; top:-15%; left:-8%; animation:authFloat 22s infinite, authPulse 8s infinite; }
.auth-orb-2 { width:450px; height:450px; bottom:-10%; right:-8%; animation:authFloat2 28s infinite, authPulse 10s infinite 2s; }
.auth-orb-3 { width:350px; height:350px; top:35%; left:55%; animation:authFloat3 20s infinite, authPulse 12s infinite 4s; }

/* Star canvas field */
.auth-stars { position:absolute; inset:0; pointer-events:none; overflow:hidden; z-index:0; }
#auth-star-canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:all; }

.auth-landing { text-align:center; z-index:1; position:relative; animation:authFadeUp .6s ease; padding:24px; }
.auth-logo-block { display:flex; align-items:center; justify-content:center; gap:18px; margin-bottom:28px; }
.auth-logo-icon { display:flex; align-items:center; }
.auth-brand { font-size:44px; font-weight:800; letter-spacing:14px; color:rgba(226,232,240,.9); text-transform:uppercase; }
.auth-brand-sm { font-size:13px; font-weight:800; letter-spacing:6px; color:rgba(226,232,240,.55); text-transform:uppercase; margin-bottom:10px; }
.auth-tagline-sub { font-size:17px; color:rgba(148,163,184,.7); line-height:1.9; margin-bottom:48px; letter-spacing:.5px; }
.auth-tagline-dim { color:rgba(100,116,139,.5); }

.auth-landing-actions { display:flex; flex-direction:column; align-items:center; gap:14px; max-width:280px; margin:0 auto; }
.auth-btn-hero {
    width:100%; padding:16px 28px; border-radius:12px;
    background:linear-gradient(135deg, rgba(var(--h-rgb),.22), rgba(var(--h-rgb),.10));
    border:1px solid rgba(var(--h-rgb),.35); color:var(--h-accent); font-size:15px; font-weight:600;
    cursor:pointer; transition:background .25s, transform .25s, box-shadow .25s, border-color .25s; font-family:inherit;
    display:flex; align-items:center; justify-content:center; gap:10px; position:relative; overflow:hidden;
}
.auth-btn-hero::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg, transparent, rgba(var(--h-rgb),.5), transparent); animation:authGlowLine 4s ease-in-out infinite; }
.auth-btn-hero:hover { background:linear-gradient(135deg, rgba(var(--h-rgb),.32), rgba(var(--h-rgb),.18)); transform:translateY(-1px); box-shadow:0 8px 32px rgba(var(--h-rgb),.15); }
.auth-btn-secondary {
    width:100%; padding:14px 28px; border-radius:12px; background:rgba(30,41,59,.6);
    border:1px solid rgba(100,116,139,.2); color:rgba(148,163,184,.8); font-size:14px; font-weight:500;
    cursor:pointer; transition:background .25s, transform .25s, box-shadow .25s, border-color .25s; font-family:inherit; display:flex; align-items:center; justify-content:center; gap:10px;
}
.auth-btn-secondary:hover { background:rgba(30,41,59,.9); border-color:rgba(100,116,139,.35); color:rgba(226,232,240,.9); }
.auth-landing-footer { margin-top:52px; }
.auth-version { font-size:10px; color:rgba(100,116,139,.28); letter-spacing:2px; font-weight:600; }

.auth-box { width:100%; max-width:480px; padding:44px 36px 40px; text-align:center; z-index:1; position:relative; animation:authFadeUp .45s ease; }
.auth-back-btn { position:absolute; top:14px; left:10px; background:none; border:none; color:rgba(100,116,139,.65); cursor:pointer; padding:8px; border-radius:8px; transition:color .2s, background .2s; }
.auth-back-btn:hover { color:rgba(226,232,240,.85); background:rgba(255,255,255,.06); }
.auth-screen-title { font-size:26px; font-weight:300; color:rgba(226,232,240,.95); margin-bottom:6px; letter-spacing:.3px; }
.auth-screen-hint { font-size:14px; color:rgba(148,163,184,.7); margin-bottom:28px; min-height:18px; transition:color .3s, opacity .3s; }

.auth-grid { display:flex; flex-wrap:wrap; gap:14px; justify-content:center; margin-bottom:10px; }
.auth-card {
    position:relative; width:148px; padding:26px 14px 22px; border-radius:16px; background:var(--bg);
    border:1px solid rgba(255,255,255,.12); cursor:pointer; transition:transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    display:flex; flex-direction:column; align-items:center; gap:10px; color:inherit; font-family:inherit;
}
.auth-card:hover { transform:translateY(-3px); border-color:color-mix(in srgb,var(--ac) 40%,transparent); box-shadow:0 10px 36px color-mix(in srgb,var(--ac) 12%,transparent); }
.auth-card:active { transform:scale(.97); }
.auth-card-lock { position:absolute; top:9px; right:9px; opacity:.5; color:var(--ac); }
.auth-avatar { width:56px; height:56px; border-radius:50%; border:2.5px solid; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:700; letter-spacing:1px; }
.auth-card-name { font-size:15px; font-weight:600; color:rgba(226,232,240,1); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:128px; }
.auth-card-role { font-size:12px; color:rgba(148,163,184,.75); line-height:1.3; text-align:center; max-height:32px; overflow:hidden; }

.auth-empty-actions { max-width:300px; margin:24px auto 0; }
.auth-empty-hint { font-size:12px; color:rgba(100,116,139,.6); margin-top:10px; margin-bottom:14px; }

.auth-form { max-width:320px; margin:0 auto; text-align:left; }
.auth-field-group { margin-bottom:16px; }
.auth-label { display:block; font-size:12px; font-weight:600; color:rgba(148,163,184,.8); margin-bottom:6px; letter-spacing:.3px; }
.auth-label-hint { font-weight:400; color:rgba(100,116,139,.65); }
.auth-input { width:100%; padding:12px 16px; border-radius:10px; background:rgba(30,41,59,.8); border:1px solid rgba(100,116,139,.3); color:#e2e8f0; font-size:15px; outline:none; transition:border-color .2s; font-family:inherit; box-sizing:border-box; }
.auth-input:focus { border-color:var(--fc, rgba(16,185,129,.5)); }
.auth-pin-input { text-align:center; letter-spacing:5px; font-size:18px; }
.auth-error { font-size:12px; color:#f87171; min-height:20px; margin:8px 0 2px; text-align:center; }
.auth-success { font-size:12px; color:#34d399; min-height:20px; margin:8px 0 2px; text-align:center; }
.pw-input-wrap { position:relative; }
.pw-input-wrap .auth-input { padding-right:42px; }
.pw-eye-btn { position:absolute; right:10px; top:50%; transform:translateY(-50%); background:none; border:none; color:rgba(148,163,184,.5); cursor:pointer; padding:4px; display:flex; align-items:center; justify-content:center; transition:color .2s; }
.pw-eye-btn:hover { color:rgba(148,163,184,.9); }
.pw-strength-bar { height:3px; border-radius:2px; background:rgba(100,116,139,.2); margin-top:6px; overflow:hidden; }
.pw-strength-fill { height:100%; border-radius:2px; transition:width .3s, background .3s; width:0; }
.pw-strength-label { font-size:11px; margin-top:3px; transition:color .3s; }
.pw-match-label { font-size:11px; margin-top:3px; min-height:16px; }

.auth-btn-primary { width:100%; padding:13px; border-radius:10px; margin-top:8px; background:rgba(var(--br, 16,185,129),.18); border:1px solid rgba(var(--br, 16,185,129),.35); color:var(--ba, #6ee7b7); font-size:14px; font-weight:600; cursor:pointer; transition:background .2s, transform .2s, box-shadow .2s, border-color .2s; font-family:inherit; text-align:center; }
.auth-btn-primary:hover { background:rgba(var(--br, 16,185,129),.28); }
.auth-btn-primary:disabled { opacity:.5; cursor:not-allowed; }
.auth-btn-link { background:none; border:none; color:rgba(100,116,139,.65); cursor:pointer; font-size:12px; transition:color .2s; font-family:inherit; margin-top:14px; display:inline-block; }
.auth-btn-link:hover { color:rgba(148,163,184,.9); }
.auth-login-footer { margin-top:18px; }

.auth-pin-header { display:flex; align-items:center; gap:12px; justify-content:center; margin-bottom:18px; }
.auth-pin-av { width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:700; }
.auth-pin-name { font-size:17px; font-weight:600; color:rgba(226,232,240,.92); }
.auth-pin-back { background:none; border:none; color:rgba(100,116,139,.7); cursor:pointer; font-size:18px; padding:2px 8px; border-radius:4px; transition:color .2s; }
.auth-pin-back:hover { color:#e2e8f0; }

@media (max-width:768px) {
    .auth-backdrop { overflow-y:auto; overflow-x:hidden; align-items:flex-start; padding:24px 0; }
    .auth-orb { display:none !important; }
    .auth-stars::after { display:none; }
    .auth-landing { padding:20px 16px; }
    .auth-brand { font-size:32px; letter-spacing:8px; }
    .auth-logo-icon svg { width:36px; height:36px; }
    .auth-logo-block { gap:12px; margin-bottom:20px; }
    .auth-tagline-sub { font-size:15px; margin-bottom:32px; line-height:1.7; }
    .auth-landing-footer { margin-top:32px; }
    .auth-landing-actions { max-width:100%; padding:0 8px; }
    .auth-box { max-width:100%; padding:36px 20px 32px; box-sizing:border-box; }
    .auth-screen-title { font-size:22px; }
    .auth-grid { gap:10px; }
    .auth-card { width:130px; padding:22px 10px 18px; }
}
@media (max-width:400px) {
    .auth-brand { font-size:26px; letter-spacing:6px; }
    .auth-logo-icon svg { width:30px; height:30px; }
    .auth-tagline-sub { font-size:13px; margin-bottom:24px; }
    .auth-box { padding:28px 16px 24px; }
    .auth-card { width:115px; padding:18px 8px 14px; }
    .auth-avatar { width:44px; height:44px; font-size:14px; }
    .auth-card-name { font-size:13px; max-width:100px; }
    .auth-card-role { font-size:11px; }
}
`;
document.head.appendChild(_css);
