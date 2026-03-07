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
    { name:'Sakura', accent:'#f0a0b8', rgb:'240,160,184',
      orb1:'rgba(240,160,184,.14)', orb2:'rgba(200,160,220,.10)', orb3:'rgba(255,220,240,.08)',
      grid:'rgba(240,160,184,.04)', bg:'linear-gradient(160deg,#08050c 0%,#120a1a 50%,#08050c 100%)',
      star1:'rgba(240,160,184,.5)', star2:'rgba(255,220,240,.35)', star3:'rgba(200,160,220,.3)' },
    { name:'Rose', accent:'#f43f5e', rgb:'244,63,94',
      orb1:'rgba(244,63,94,.14)', orb2:'rgba(251,113,133,.10)', orb3:'rgba(225,29,72,.08)',
      grid:'rgba(244,63,94,.04)', bg:'linear-gradient(160deg,#080507 0%,#120a0e 50%,#080507 100%)',
      star1:'rgba(244,63,94,.5)', star2:'rgba(251,113,133,.35)', star3:'rgba(253,164,175,.25)' },
    { name:'Coffee', accent:'#d4943c', rgb:'212,148,60',
      orb1:'rgba(212,148,60,.14)', orb2:'rgba(184,122,46,.10)', orb3:'rgba(232,170,84,.08)',
      grid:'rgba(212,148,60,.04)', bg:'linear-gradient(160deg,#0c0806 0%,#14100a 50%,#0c0806 100%)',
      star1:'rgba(212,148,60,.5)', star2:'rgba(184,122,46,.35)', star3:'rgba(255,240,220,.25)' },
    { name:'Midnight', accent:'#10b981', rgb:'16,185,129',
      orb1:'rgba(16,185,129,.14)', orb2:'rgba(52,211,153,.10)', orb3:'rgba(5,150,105,.07)',
      grid:'rgba(16,185,129,.04)', bg:'linear-gradient(160deg,#050810 0%,#080d18 50%,#050810 100%)',
      star1:'rgba(16,185,129,.5)', star2:'rgba(52,211,153,.35)', star3:'rgba(209,250,229,.25)' },
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
    const _isCosmos = _t.name === 'Cosmos';
    const _isSakura = _t.name === 'Sakura';
    const _isNebula = _t.name === 'Nebula';
    const _isAurora = _t.name === 'Aurora';
    const _isNoir = _t.name === 'Noir';
    const _isRose = _t.name === 'Rose';
    const _isCoffee = _t.name === 'Coffee';
    const _isMidnight = _t.name === 'Midnight';

    // ── Black hole data (nebula auth theme — P1 classic accretion disk) ──
    const _bhParticles = [];
    if (_isNebula) {
        for (let i = 0; i < 400; i++) {
            const band = Math.random();
            const dist = 50 + band * 260;
            const tier = band < 0.2 ? 0 : band < 0.55 ? 1 : 2;
            _bhParticles.push({
                angle: Math.random() * Math.PI * 2,
                dist,
                speed: (0.06 + Math.random() * 0.14) * (180 / (dist + 30)),
                r: tier === 0 ? Math.random() * 2.2 + 0.8 : Math.random() * 1.6 + 0.4,
                a: tier === 0 ? Math.random() * 0.7 + 0.4 : Math.random() * 0.55 + 0.15,
                yOff: (Math.random() - 0.5) * 5,
                tier
            });
        }
    }
    const _BH_TILT = 0.30, _BH_ROT = -0.12;
    const _bhTierColors = [[200,190,255],[167,139,250],[56,189,248]];
    function _bhProject(cx, cy, dist, angle) {
        const x3 = Math.cos(angle) * dist, y3 = Math.sin(angle) * dist;
        const cr = Math.cos(_BH_ROT), sr = Math.sin(_BH_ROT);
        return { x: cx + x3 * cr - y3 * sr, y: cy + (x3 * sr + y3 * cr) * _BH_TILT, depth: Math.sin(angle) };
    }
    function _bhDrawDisk(cx, cy) {
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(_BH_ROT); ctx.scale(1, _BH_TILT);
        for (let ring = 0; ring < 5; ring++) {
            const rd = 70 + ring * 50;
            const alpha = [0.12, 0.09, 0.07, 0.05, 0.035][ring];
            ctx.beginPath(); ctx.arc(0, 0, rd, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(167,139,250,${alpha})`; ctx.lineWidth = 22 + ring * 10; ctx.stroke();
        }
        // Outer nebula haze
        const haze = ctx.createRadialGradient(0, 0, 80, 0, 0, 320);
        haze.addColorStop(0, 'rgba(100,80,200,0.0)');
        haze.addColorStop(0.4, 'rgba(80,60,180,0.04)');
        haze.addColorStop(0.7, 'rgba(56,130,220,0.025)');
        haze.addColorStop(1, 'rgba(56,189,248,0)');
        ctx.fillStyle = haze; ctx.beginPath(); ctx.arc(0, 0, 320, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    }
    function _bhDrawVoid(cx, cy, t) {
        const pulse = 1 + Math.sin(t * 0.4) * 0.03;
        const eventR = 30 * pulse;
        // Gravitational lensing halo
        const lens = ctx.createRadialGradient(cx, cy, eventR, cx, cy, eventR * 5);
        lens.addColorStop(0, 'rgba(167,139,250,0.22)');
        lens.addColorStop(0.25, 'rgba(120,100,220,0.10)');
        lens.addColorStop(0.5, 'rgba(56,189,248,0.04)');
        lens.addColorStop(1, 'rgba(56,189,248,0)');
        ctx.fillStyle = lens; ctx.beginPath(); ctx.arc(cx, cy, eventR * 5, 0, Math.PI * 2); ctx.fill();
        // Photon ring
        const photon = ctx.createRadialGradient(cx, cy, eventR - 3, cx, cy, eventR + 14);
        photon.addColorStop(0, 'rgba(167,139,250,0)');
        photon.addColorStop(0.25, 'rgba(167,139,250,0.55)');
        photon.addColorStop(0.45, 'rgba(220,210,255,0.7)');
        photon.addColorStop(0.65, 'rgba(125,211,252,0.45)');
        photon.addColorStop(1, 'rgba(56,189,248,0)');
        ctx.fillStyle = photon; ctx.beginPath(); ctx.arc(cx, cy, eventR + 14, 0, Math.PI * 2); ctx.fill();
        // Event horizon void
        const voidG = ctx.createRadialGradient(cx, cy, 0, cx, cy, eventR);
        voidG.addColorStop(0, 'rgba(0,0,0,1)');
        voidG.addColorStop(0.8, 'rgba(0,0,0,1)');
        voidG.addColorStop(1, 'rgba(0,0,0,0.6)');
        ctx.fillStyle = voidG; ctx.beginPath(); ctx.arc(cx, cy, eventR, 0, Math.PI * 2); ctx.fill();
    }

    // ── Binary star system (aurora auth theme) ──
    const _BIN_ORBIT_R = 45, _BIN_SPEED = 0.12, _BIN_TILT = 0.38;
    const _binStarA = { r: 22, color: '52,211,153', bright: '200,255,225', glow: '110,231,183' };
    const _binStarB = { r: 16, color: '56,189,248', bright: '200,235,255', glow: '125,211,252' };
    const _binParticles = [];
    if (_isAurora) {
        for (let i = 0; i < 180; i++) {
            const band = Math.random();
            const dist = 55 + band * 180;
            _binParticles.push({
                angle: Math.random() * Math.PI * 2,
                dist,
                speed: (0.04 + Math.random() * 0.06) * (120 / (dist + 30)),
                r: Math.random() * 1.2 + 0.3,
                a: Math.random() * 0.4 + 0.1,
                yOff: (Math.random() - 0.5) * 4,
                colorType: band < 0.4 ? 0 : band < 0.7 ? 1 : 2,
            });
        }
    }
    const _binColors = [[110,231,183],[56,189,248],[200,220,240]];
    function _binProject(cx, cy, dist, angle) {
        return { x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist * _BIN_TILT, depth: Math.sin(angle) };
    }
    function _binDrawSystem(cx, cy, t) {
        const ang = t * _BIN_SPEED;
        const s1x = cx + Math.cos(ang) * _BIN_ORBIT_R;
        const s1y = cy + Math.sin(ang) * _BIN_ORBIT_R * _BIN_TILT;
        const s1d = Math.sin(ang);
        const s2x = cx + Math.cos(ang + Math.PI) * _BIN_ORBIT_R;
        const s2y = cy + Math.sin(ang + Math.PI) * _BIN_ORBIT_R * _BIN_TILT;
        const s2d = Math.sin(ang + Math.PI);

        // Ambient glow
        const amb = ctx.createRadialGradient(cx, cy, 0, cx, cy, 300);
        amb.addColorStop(0, 'rgba(52,211,153,0.06)');
        amb.addColorStop(0.15, 'rgba(56,189,248,0.03)');
        amb.addColorStop(0.35, 'rgba(110,231,183,0.012)');
        amb.addColorStop(0.6, 'rgba(16,185,129,0.004)');
        amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(cx, cy, 300, 0, Math.PI * 2); ctx.fill();

        // Accretion rings
        ctx.save(); ctx.translate(cx, cy); ctx.scale(1, _BIN_TILT);
        const ringDefs = [[60,'110,231,183',0.04],[100,'56,189,248',0.025],[140,'52,211,153',0.015]];
        for (const [rd, rc, ra] of ringDefs) {
            ctx.beginPath(); ctx.arc(0, 0, rd, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(${rc},${ra})`; ctx.lineWidth = 12; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(0, 0, _BIN_ORBIT_R, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(110,231,183,0.035)'; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();

        // Back orbital particles
        const proj = [];
        for (const p of _binParticles) {
            const pr = _binProject(cx, cy, p.dist, p.angle + t * p.speed);
            proj.push({ x: pr.x, y: pr.y + p.yOff, d: pr.depth, r: p.r, a: p.a, ct: p.colorType });
        }
        proj.sort((a, b) => a.d - b.d);
        for (const p of proj) {
            if (p.d > 0.1) continue;
            const c = _binColors[p.ct];
            ctx.globalAlpha = p.a * (0.5 + p.d * 0.5);
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        }

        // Tidal bridge
        const bridgePulse = 0.7 + 0.3 * Math.sin(t * 0.8);
        const bridgeDefs = [['200,255,230',0.06,4],['110,231,183',0.035,8],['56,189,248',0.02,14]];
        for (const [bc, ba, bw] of bridgeDefs) {
            const sOff = Math.sin(t * 0.5 + bw) * 4;
            ctx.beginPath(); ctx.moveTo(s1x, s1y);
            ctx.quadraticCurveTo((s1x+s2x)/2 + sOff, (s1y+s2y)/2 - 5, s2x, s2y);
            ctx.strokeStyle = `rgba(${bc},${ba * bridgePulse})`; ctx.lineWidth = bw; ctx.lineCap = 'round'; ctx.stroke();
        }

        // Draw stars sorted by depth
        const pair = [
            { x: s1x, y: s1y, d: s1d, ...(_binStarA), id: 'A' },
            { x: s2x, y: s2y, d: s2d, ...(_binStarB), id: 'B' },
        ];
        pair.sort((a, b) => a.d - b.d);
        for (const star of pair) {
            const db = 0.75 + 0.25 * star.d;
            const pulse = 1 + Math.sin(t * 0.4 + (star.id === 'A' ? 0 : Math.PI)) * 0.08;
            const sr = star.r * pulse;
            // Outer halo
            const oh = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, sr * 8);
            oh.addColorStop(0, `rgba(${star.glow},${0.08 * db})`);
            oh.addColorStop(0.2, `rgba(${star.color},${0.04 * db})`);
            oh.addColorStop(0.5, `rgba(${star.color},${0.012 * db})`);
            oh.addColorStop(1, 'transparent');
            ctx.fillStyle = oh; ctx.beginPath(); ctx.arc(star.x, star.y, sr * 8, 0, Math.PI * 2); ctx.fill();
            // Inner glow
            const ig = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, sr * 3);
            ig.addColorStop(0, `rgba(${star.bright},${0.25 * db})`);
            ig.addColorStop(0.3, `rgba(${star.glow},${0.15 * db})`);
            ig.addColorStop(0.7, `rgba(${star.color},${0.04 * db})`);
            ig.addColorStop(1, 'transparent');
            ctx.fillStyle = ig; ctx.beginPath(); ctx.arc(star.x, star.y, sr * 3, 0, Math.PI * 2); ctx.fill();
            // Body
            const bd = ctx.createRadialGradient(star.x - sr * 0.1, star.y - sr * 0.1, 0, star.x, star.y, sr);
            bd.addColorStop(0, `rgba(${star.bright},${0.5 * db})`);
            bd.addColorStop(0.3, `rgba(${star.bright},${0.35 * db})`);
            bd.addColorStop(0.7, `rgba(${star.color},${0.2 * db})`);
            bd.addColorStop(1, `rgba(${star.color},${0.05 * db})`);
            ctx.fillStyle = bd; ctx.beginPath(); ctx.arc(star.x, star.y, sr, 0, Math.PI * 2); ctx.fill();
            // Lens flare
            ctx.globalAlpha = 0.08 * db * pulse;
            ctx.strokeStyle = `rgba(${star.bright},0.4)`; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(star.x - sr * 5, star.y); ctx.lineTo(star.x + sr * 5, star.y); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(star.x, star.y - sr * 3.5); ctx.lineTo(star.x, star.y + sr * 3.5); ctx.stroke();
            ctx.globalAlpha = 1;
        }

        // Front particles
        for (const p of proj) {
            if (p.d <= 0.1) continue;
            const c = _binColors[p.ct];
            const da = p.a * (0.6 + p.d * 0.4);
            ctx.globalAlpha = da;
            if (p.ct === 0 && p.r > 0.8) {
                const pg = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 5);
                pg.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${da * 0.25})`);
                pg.addColorStop(1, 'transparent');
                ctx.fillStyle = pg; ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 5, 0, Math.PI * 2); ctx.fill();
            }
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    // Solar system data (cosmos auth theme - tilted perspective)
    const _SS_TILT = 0.38, _SS_ROT = -0.15;
    const _ssPlanets = _isCosmos ? [
        { dist: 72,  r: 3,   color: [176,160,144], speed: 4.2,  phase: Math.random() * Math.PI * 2 },
        { dist: 108, r: 5.2, color: [232,199,122], speed: 1.65, phase: Math.random() * Math.PI * 2 },
        { dist: 150, r: 5.2, color: [70,140,210],  speed: 1.0,  phase: Math.random() * Math.PI * 2, moon: { dist: 15, r: 1.5, speed: 5 } },
        { dist: 198, r: 4.2, color: [210,100,60],  speed: 0.53, phase: Math.random() * Math.PI * 2 },
        { dist: 285, r: 11,  color: [210,165,90],  speed: 0.084,phase: Math.random() * Math.PI * 2 },
        { dist: 372, r: 9,   color: [195,178,115], speed: 0.034,phase: Math.random() * Math.PI * 2, rings: true },
        { dist: 465, r: 6.3, color: [120,195,195], speed: 0.012,phase: Math.random() * Math.PI * 2 },
        { dist: 555, r: 6,   color: [65,100,210],  speed: 0.006,phase: Math.random() * Math.PI * 2 },
    ] : [];
    const _ssAsteroids = [];
    if (_isCosmos) {
        for (let ai = 0; ai < 100; ai++) {
            _ssAsteroids.push({
                dist: 232 + Math.random() * 38,
                angle: Math.random() * Math.PI * 2,
                speed: 0.12 + Math.random() * 0.1,
                r: Math.random() * 0.7 + 0.2,
                a: Math.random() * 0.3 + 0.05,
                yOff: (Math.random() - 0.5) * 4
            });
        }
    }
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
    function _ssDrawSun(cx, cy, t) {
        const pulse = 1 + Math.sin(t * 0.5) * 0.05, r = 30 * pulse;
        const g4 = ctx.createRadialGradient(cx, cy, r, cx, cy, r * 5);
        g4.addColorStop(0, 'rgba(232,185,49,0.06)'); g4.addColorStop(0.5, 'rgba(232,185,49,0.015)'); g4.addColorStop(1, 'rgba(232,185,49,0)');
        ctx.fillStyle = g4; ctx.beginPath(); ctx.arc(cx, cy, r * 5, 0, Math.PI * 2); ctx.fill();
        const g3 = ctx.createRadialGradient(cx, cy, r * 0.8, cx, cy, r * 2.5);
        g3.addColorStop(0, 'rgba(255,220,100,0.3)'); g3.addColorStop(0.5, 'rgba(232,185,49,0.08)'); g3.addColorStop(1, 'rgba(232,185,49,0)');
        ctx.fillStyle = g3; ctx.beginPath(); ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2); ctx.fill();
        const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.3);
        g2.addColorStop(0, 'rgba(255,245,220,0.5)'); g2.addColorStop(1, 'rgba(232,185,49,0)');
        ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(cx, cy, r * 1.3, 0, Math.PI * 2); ctx.fill();
        const g1 = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g1.addColorStop(0, '#fff8e0'); g1.addColorStop(0.2, '#ffe566'); g1.addColorStop(0.5, '#f0c030');
        g1.addColorStop(0.8, '#e8b931'); g1.addColorStop(1, '#c08520');
        ctx.fillStyle = g1; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
        for (let i = 0; i < 3; i++) {
            const a = t * 0.3 + i * 2.1, hx = cx + Math.cos(a) * r * 0.4, hy = cy + Math.sin(a) * r * 0.4;
            const hg = ctx.createRadialGradient(hx, hy, 0, hx, hy, r * 0.3);
            hg.addColorStop(0, 'rgba(255,255,230,0.3)'); hg.addColorStop(1, 'rgba(255,255,230,0)');
            ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(hx, hy, r * 0.3, 0, Math.PI * 2); ctx.fill();
        }
    }
    function _ssDrawPlanet(px, py, p, depth) {
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

    // ── Noir: Pulsar with twin sweeping beams ──
    const _noirPulsarParticles = [];
    if (_isNoir) {
        for (let i = 0; i < 200; i++) {
            const band = Math.random();
            const dist = 30 + band * 200;
            _noirPulsarParticles.push({
                angle: Math.random() * Math.PI * 2,
                dist,
                speed: (0.03 + Math.random() * 0.05) * (100 / (dist + 20)),
                r: Math.random() * 1.0 + 0.3,
                a: Math.random() * 0.35 + 0.08,
                yOff: (Math.random() - 0.5) * 3,
                colorType: band < 0.35 ? 0 : band < 0.65 ? 1 : 2,
            });
        }
    }
    const _noirMagColors = [[167,139,250],[192,132,252],[139,92,246]];
    function _noirDrawPulsar(cx, cy, t) {
        const beamAngle = t * 0.4;
        const beamLen = 350;
        // Twin beams
        for (let b = 0; b < 2; b++) {
            const bAng = beamAngle + b * Math.PI;
            const bx = Math.cos(bAng), by = Math.sin(bAng);
            for (let layer = 0; layer < 4; layer++) {
                const spread = [0.03,0.06,0.1,0.16][layer];
                const alpha = [0.12,0.06,0.03,0.015][layer];
                ctx.beginPath(); ctx.moveTo(cx, cy);
                const perpX = -by, perpY = bx;
                const endX = cx + bx * beamLen, endY = cy + by * beamLen;
                const sd = beamLen * spread;
                ctx.lineTo(endX + perpX * sd, endY + perpY * sd);
                ctx.lineTo(endX - perpX * sd, endY - perpY * sd);
                ctx.closePath();
                const bg = ctx.createLinearGradient(cx, cy, endX, endY);
                bg.addColorStop(0, `rgba(192,132,252,${alpha * 1.5})`);
                bg.addColorStop(0.1, `rgba(167,139,250,${alpha})`);
                bg.addColorStop(0.5, `rgba(139,92,246,${alpha * 0.5})`);
                bg.addColorStop(1, 'transparent');
                ctx.fillStyle = bg; ctx.fill();
            }
        }
        // Magnetic field rings
        ctx.save(); ctx.translate(cx, cy); ctx.scale(1, 0.35);
        for (let ring = 0; ring < 4; ring++) {
            const rd = 40 + ring * 35;
            ctx.beginPath(); ctx.arc(0, 0, rd, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(167,139,250,${0.035 - ring * 0.006})`; ctx.lineWidth = 1.5; ctx.stroke();
        }
        ctx.restore();
        // Back magnetosphere particles
        const proj = [];
        for (const p of _noirPulsarParticles) {
            const a = p.angle + t * p.speed;
            const px2 = cx + Math.cos(a) * p.dist;
            const py2 = cy + Math.sin(a) * p.dist * 0.35 + p.yOff;
            proj.push({ x: px2, y: py2, d: Math.sin(a), r: p.r, a: p.a, ct: p.colorType });
        }
        proj.sort((a2, b2) => a2.d - b2.d);
        for (const p of proj) {
            if (p.d > 0.1) continue;
            const c = _noirMagColors[p.ct];
            ctx.globalAlpha = p.a * (0.5 + p.d * 0.5);
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        }
        // Pulsar core
        const pulse = 1 + Math.sin(t * 2) * 0.15;
        const pr = 14 * pulse;
        const outerG = ctx.createRadialGradient(cx, cy, 0, cx, cy, pr * 10);
        outerG.addColorStop(0, 'rgba(167,139,250,0.08)'); outerG.addColorStop(0.2, 'rgba(139,92,246,0.04)');
        outerG.addColorStop(0.5, 'rgba(99,102,241,0.015)'); outerG.addColorStop(1, 'transparent');
        ctx.globalAlpha = 1;
        ctx.fillStyle = outerG; ctx.beginPath(); ctx.arc(cx, cy, pr * 10, 0, Math.PI * 2); ctx.fill();
        // Pulsing rings
        for (let i = 0; i < 3; i++) {
            const ringT = (t * 0.8 + i * 0.7) % 3.0;
            const ringR = pr + ringT * 40;
            const ringA = Math.max(0, 0.15 * (1 - ringT / 3.0));
            if (ringA < 0.005) continue;
            ctx.beginPath(); ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(192,132,252,${ringA})`; ctx.lineWidth = 2; ctx.stroke();
        }
        // Core
        const innerG = ctx.createRadialGradient(cx, cy, 0, cx, cy, pr * 3);
        innerG.addColorStop(0, `rgba(230,220,255,${0.3 * pulse})`); innerG.addColorStop(0.3, `rgba(192,132,252,${0.15 * pulse})`);
        innerG.addColorStop(0.7, `rgba(139,92,246,${0.04 * pulse})`); innerG.addColorStop(1, 'transparent');
        ctx.fillStyle = innerG; ctx.beginPath(); ctx.arc(cx, cy, pr * 3, 0, Math.PI * 2); ctx.fill();
        const coreG = ctx.createRadialGradient(cx, cy, 0, cx, cy, pr);
        coreG.addColorStop(0, 'rgba(240,235,255,0.6)'); coreG.addColorStop(0.4, 'rgba(192,132,252,0.4)');
        coreG.addColorStop(0.8, 'rgba(139,92,246,0.2)'); coreG.addColorStop(1, 'rgba(99,102,241,0.05)');
        ctx.fillStyle = coreG; ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2); ctx.fill();
        // Front particles
        for (const p of proj) {
            if (p.d <= 0.1) continue;
            const c = _noirMagColors[p.ct];
            ctx.globalAlpha = p.a * (0.6 + p.d * 0.4);
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    // ── Rose: Geometric rose bloom ──
    function _roseDrawBloom(cx, cy, t) {
        const breathe = 1 + 0.06 * Math.sin(t * 0.8);
        const rotation = t * 0.05;
        // Ambient glow
        const amb = ctx.createRadialGradient(cx, cy, 0, cx, cy, 180);
        amb.addColorStop(0, 'rgba(244,63,94,0.08)'); amb.addColorStop(0.3, 'rgba(251,113,133,0.04)');
        amb.addColorStop(0.6, 'rgba(225,29,72,0.015)'); amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(cx, cy, 180, 0, Math.PI * 2); ctx.fill();
        // Pollen particles rising
        for (let i = 0; i < 20; i++) {
            const age = (t * 0.3 + i * 0.5) % 5;
            const px2 = cx + Math.sin(t * 0.2 + i * 1.7) * (10 + age * 8);
            const py2 = cy - age * 25;
            const pa = Math.max(0, 0.3 * (1 - age / 5));
            if (pa < 0.01) continue;
            ctx.globalAlpha = pa;
            ctx.fillStyle = 'rgba(253,164,175,0.8)';
            ctx.beginPath(); ctx.arc(px2, py2, 1 + (1 - age / 5), 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
        // Draw petals — 5 layers, each rotated
        for (let layer = 4; layer >= 0; layer--) {
            const layerScale = (0.5 + layer * 0.12) * breathe;
            const petalCount = 5 + layer;
            const layerRot = rotation + layer * 0.15;
            const openFactor = 0.7 + layer * 0.08;
            for (let i = 0; i < petalCount; i++) {
                const angle = layerRot + (i / petalCount) * Math.PI * 2;
                const pr = 30 * layerScale * openFactor;
                const tipX = cx + Math.cos(angle) * pr;
                const tipY = cy + Math.sin(angle) * pr;
                const cp1Angle = angle - 0.3;
                const cp2Angle = angle + 0.3;
                const cpDist = pr * 0.7;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.quadraticCurveTo(
                    cx + Math.cos(cp1Angle) * cpDist, cy + Math.sin(cp1Angle) * cpDist, tipX, tipY
                );
                ctx.quadraticCurveTo(
                    cx + Math.cos(cp2Angle) * cpDist, cy + Math.sin(cp2Angle) * cpDist, cx, cy
                );
                const bright = 0.5 + 0.5 * Math.sin(t * 0.5 + i + layer);
                const r = layer < 2 ? 244 : 251;
                const g = layer < 2 ? 63 + bright * 30 : 113;
                const b = layer < 2 ? 94 : 133;
                const alpha = (0.12 + layer * 0.04) * bright;
                ctx.fillStyle = `rgba(${r},${g|0},${b},${alpha})`;
                ctx.fill();
                // Petal edge
                ctx.strokeStyle = `rgba(253,164,175,${alpha * 0.5})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
        // Center core
        const coreG = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12 * breathe);
        coreG.addColorStop(0, 'rgba(255,240,243,0.5)'); coreG.addColorStop(0.5, 'rgba(244,63,94,0.3)');
        coreG.addColorStop(1, 'rgba(225,29,72,0)');
        ctx.fillStyle = coreG; ctx.beginPath(); ctx.arc(cx, cy, 12 * breathe, 0, Math.PI * 2); ctx.fill();
    }

    // ── Coffee: Ember with rising sparks ──
    const _coffeeEmbers = [];
    if (_isCoffee) {
        for (let i = 0; i < 50; i++) {
            _coffeeEmbers.push({
                angle: Math.random() * Math.PI * 2,
                speed: 0.3 + Math.random() * 0.7,
                drift: (Math.random() - 0.5) * 0.4,
                size: 0.8 + Math.random() * 2,
                phase: Math.random() * 5,
                bright: Math.random()
            });
        }
    }
    function _coffeeDrawEmber(cx, cy, t) {
        const pulse = 1 + 0.08 * Math.sin(t * 1.2);
        // Wide ambient warmth
        const amb = ctx.createRadialGradient(cx, cy, 0, cx, cy, 200);
        amb.addColorStop(0, 'rgba(212,148,60,0.08)'); amb.addColorStop(0.3, 'rgba(184,122,46,0.04)');
        amb.addColorStop(0.6, 'rgba(232,170,84,0.015)'); amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(cx, cy, 200, 0, Math.PI * 2); ctx.fill();
        // Pulsing warm rings
        for (let i = 0; i < 3; i++) {
            const ringT = (t * 0.6 + i * 1.0) % 4.0;
            const ringR = 20 + ringT * 35;
            const ringA = Math.max(0, 0.08 * (1 - ringT / 4.0));
            if (ringA < 0.003) continue;
            ctx.beginPath(); ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(232,170,84,${ringA})`; ctx.lineWidth = 2; ctx.stroke();
        }
        // Core ember
        const coreR = 22 * pulse;
        const g1 = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 4);
        g1.addColorStop(0, 'rgba(255,240,212,0.15)'); g1.addColorStop(0.2, 'rgba(232,170,84,0.1)');
        g1.addColorStop(0.5, 'rgba(212,148,60,0.04)'); g1.addColorStop(1, 'transparent');
        ctx.fillStyle = g1; ctx.beginPath(); ctx.arc(cx, cy, coreR * 4, 0, Math.PI * 2); ctx.fill();
        const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
        g2.addColorStop(0, 'rgba(255,245,220,0.5)'); g2.addColorStop(0.3, 'rgba(232,170,84,0.35)');
        g2.addColorStop(0.7, 'rgba(212,148,60,0.15)'); g2.addColorStop(1, 'rgba(184,122,46,0.02)');
        ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.fill();
        // Rising sparks
        for (const e of _coffeeEmbers) {
            const age = (t * e.speed + e.phase) % 5;
            const progress = age / 5;
            const sx = cx + Math.sin(t * 0.3 + e.angle) * (8 + age * 12) + e.drift * age * 20;
            const sy = cy - age * 30 - Math.sin(age * 2) * 5;
            const sa = (1 - progress) * 0.6 * (0.5 + e.bright * 0.5);
            if (sa < 0.02) continue;
            const sr = e.size * (1 - progress * 0.5);
            ctx.globalAlpha = sa;
            const sparkColor = e.bright > 0.6 ? '255,240,212' : e.bright > 0.3 ? '232,170,84' : '184,122,46';
            const sg = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr * 4);
            sg.addColorStop(0, `rgba(${sparkColor},${sa * 0.4})`);
            sg.addColorStop(1, 'transparent');
            ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(sx, sy, sr * 4, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = `rgba(${sparkColor},${sa})`;
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    // ── Midnight: Crescent moon with emerald glow ──
    const _midnightFireflies = [];
    if (_isMidnight) {
        for (let i = 0; i < 30; i++) {
            _midnightFireflies.push({
                x: 0.3 + Math.random() * 0.4,
                y: 0.2 + Math.random() * 0.3,
                phase: Math.random() * Math.PI * 2,
                glowSpeed: 0.3 + Math.random() * 0.7,
                wanderRx: 0.01 + Math.random() * 0.02,
                wanderRy: 0.005 + Math.random() * 0.015,
                wanderSpeed: 0.06 + Math.random() * 0.1,
                wanderPhase: Math.random() * Math.PI * 2,
                r: 0.8 + Math.random() * 1.2
            });
        }
    }
    const _midnightClouds = [];
    if (_isMidnight) {
        for (let i = 0; i < 5; i++) {
            _midnightClouds.push({
                x: Math.random(), y: 0.25 + Math.random() * 0.15,
                w: 0.06 + Math.random() * 0.1, h: 0.015 + Math.random() * 0.02,
                speed: 0.001 + Math.random() * 0.002,
                alpha: 0.02 + Math.random() * 0.02,
                phase: Math.random() * Math.PI * 2
            });
        }
    }
    function _midnightDrawMoon(cx, cy, t, cw) {
        const moonR = 35;
        // Emerald halo
        const halo = ctx.createRadialGradient(cx, cy, moonR * 0.8, cx, cy, moonR * 6);
        halo.addColorStop(0, 'rgba(16,185,129,0.06)'); halo.addColorStop(0.3, 'rgba(52,211,153,0.03)');
        halo.addColorStop(0.6, 'rgba(5,150,105,0.01)'); halo.addColorStop(1, 'transparent');
        ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(cx, cy, moonR * 6, 0, Math.PI * 2); ctx.fill();
        // Full moon disk (dark side visible)
        const moonBg = ctx.createRadialGradient(cx - moonR * 0.2, cy - moonR * 0.2, 0, cx, cy, moonR);
        moonBg.addColorStop(0, 'rgba(200,210,230,0.12)'); moonBg.addColorStop(0.5, 'rgba(150,160,180,0.08)');
        moonBg.addColorStop(1, 'rgba(100,110,130,0.04)');
        ctx.fillStyle = moonBg; ctx.beginPath(); ctx.arc(cx, cy, moonR, 0, Math.PI * 2); ctx.fill();
        // Crescent — clip the dark shadow
        ctx.save();
        ctx.beginPath(); ctx.arc(cx, cy, moonR + 1, 0, Math.PI * 2); ctx.clip();
        const shadowOff = moonR * 0.6;
        ctx.fillStyle = 'rgba(5,8,16,0.95)';
        ctx.beginPath(); ctx.arc(cx + shadowOff, cy, moonR * 0.95, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
        // Bright crescent edge glow
        const crescentG = ctx.createRadialGradient(cx - moonR * 0.3, cy, moonR * 0.5, cx, cy, moonR * 1.2);
        crescentG.addColorStop(0, 'rgba(209,250,229,0.25)'); crescentG.addColorStop(0.4, 'rgba(52,211,153,0.08)');
        crescentG.addColorStop(1, 'transparent');
        ctx.fillStyle = crescentG; ctx.beginPath(); ctx.arc(cx, cy, moonR * 1.2, 0, Math.PI * 2); ctx.fill();
        // Subtle crater marks
        const craters = [[0.25,-0.3,0.12],[0.15,0.2,0.08],[-0.1,-0.15,0.06]];
        for (const [ox, oy, cr] of craters) {
            const crx = cx + moonR * ox - moonR * 0.15;
            const cry = cy + moonR * oy;
            ctx.globalAlpha = 0.04;
            ctx.fillStyle = 'rgba(100,130,160,1)';
            ctx.beginPath(); ctx.arc(crx, cry, moonR * cr, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
        // Drifting cloud wisps
        for (const cl of _midnightClouds) {
            cl.x += cl.speed * 0.008;
            if (cl.x > 1.4) cl.x = -0.4;
            const breathe = 0.5 + 0.5 * Math.sin(t * 0.15 + cl.phase);
            const clx = cl.x * cw, cly = cl.y * canvas.height;
            const clw = cl.w * cw;
            const cg = ctx.createRadialGradient(clx, cly, 0, clx, cly, clw);
            cg.addColorStop(0, `rgba(16,185,129,${cl.alpha * breathe})`);
            cg.addColorStop(0.5, `rgba(52,211,153,${cl.alpha * breathe * 0.3})`);
            cg.addColorStop(1, 'transparent');
            ctx.fillStyle = cg;
            ctx.fillRect(clx - clw, cly - clw * cl.h / cl.w, clw * 2, clw * cl.h / cl.w * 2);
        }
        // Emerald fireflies
        for (const f of _midnightFireflies) {
            const wp = t * f.wanderSpeed + f.wanderPhase;
            const fx = (f.x + Math.sin(wp) * f.wanderRx) * cw;
            const fy = (f.y + Math.sin(wp * 2) * f.wanderRy) * canvas.height;
            const rawGlow = Math.sin(t * f.glowSpeed + f.phase);
            const glow = Math.pow(Math.max(0, rawGlow), 1.5);
            if (glow < 0.03) continue;
            ctx.globalAlpha = glow;
            const fHalo = ctx.createRadialGradient(fx, fy, 0, fx, fy, f.r * 12);
            fHalo.addColorStop(0, 'rgba(52,211,153,0.10)'); fHalo.addColorStop(0.3, 'rgba(16,185,129,0.03)');
            fHalo.addColorStop(1, 'transparent');
            ctx.fillStyle = fHalo; ctx.beginPath(); ctx.arc(fx, fy, f.r * 12, 0, Math.PI * 2); ctx.fill();
            const fCore = ctx.createRadialGradient(fx, fy, 0, fx, fy, f.r * 2);
            fCore.addColorStop(0, `rgba(209,250,229,${glow * 0.8})`); fCore.addColorStop(0.4, `rgba(52,211,153,${glow * 0.5})`);
            fCore.addColorStop(1, 'transparent');
            ctx.fillStyle = fCore; ctx.beginPath(); ctx.arc(fx, fy, f.r * 2, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    // ── Sakura: Tree with blossoms ──
    const _sakuraTree = { branches: [], blossoms: [], tips: [] };
    if (_isSakura) {
        let _tSeed = 12345;
        function _tRand() { _tSeed = (_tSeed * 16807 + 0) % 2147483647; return _tSeed / 2147483647; }
        function _tRandR(a, b) { return a + _tRand() * (b - a); }

        function _genSakuraTree(w, h) {
            _tSeed = 12345;
            const br = [], bl = [], tips = [];
            const cx = w * 0.5, baseY = h * 0.58;
            const trunkTop = h * 0.38;

            // Slim trunk with slight lean
            const lean = w * 0.005;
            br.push({ x0:cx-w*0.005, y0:baseY, x1:cx+lean, y1:trunkTop,
                cpx:cx-w*0.002, cpy:h*0.48, w0:w*0.010, d:0 });
            br.push({ x0:cx+w*0.005, y0:baseY, x1:cx+lean, y1:trunkTop,
                cpx:cx+w*0.004, cpy:h*0.48, w0:w*0.010, d:0 });

            // Recursive branch generator — delicate and airy
            function addBranch(sx, sy, angle, length, width, depth) {
                if (depth > 3 || length < h * 0.015) return;
                const curve = _tRandR(-0.3, 0.3);
                const ex = sx + Math.cos(angle) * length;
                const ey = sy + Math.sin(angle) * length;
                const mid = 0.5 + _tRandR(-0.15, 0.15);
                const cpx = sx + Math.cos(angle + curve) * length * mid;
                const cpy = sy + Math.sin(angle + curve) * length * mid;
                br.push({ x0:sx, y0:sy, cpx, cpy, x1:ex, y1:ey, w0:width, d:depth });

                if (depth >= 2) {
                    tips.push({ x: ex, y: ey });
                    bl.push({ x:ex, y:ey, r:_tRandR(4, 9 - depth), phase:_tRand()*Math.PI*2 });
                }

                // Fork into sub-branches
                const forks = depth < 2 ? Math.floor(_tRandR(2, 4)) : Math.floor(_tRandR(1, 3));
                for (let i = 0; i < forks; i++) {
                    const spread = depth < 2 ? _tRandR(0.3, 0.9) : _tRandR(0.2, 0.7);
                    const side = (i % 2 === 0 ? 1 : -1);
                    const childAngle = angle + side * spread + _tRandR(-0.15, 0.15);
                    const childLen = length * _tRandR(0.55, 0.78);
                    const childW = width * _tRandR(0.45, 0.65);
                    addBranch(ex, ey, childAngle, childLen, childW, depth + 1);
                }
            }

            // Main branches — delicate, fewer
            const mainAngles = [
                -Math.PI*0.7 + _tRandR(-0.1,0.1),
                -Math.PI*0.45 + _tRandR(-0.08,0.08),
                -Math.PI*0.3 + _tRandR(-0.05,0.05),
                -Math.PI*0.12 + _tRandR(-0.1,0.1),
            ];
            for (let i = 0; i < mainAngles.length; i++) {
                const len = _tRandR(h*0.07, h*0.13);
                const wid = w * _tRandR(0.003, 0.005);
                addBranch(cx + lean, trunkTop, mainAngles[i], len, wid, 1);
            }

            // One small lower branch
            addBranch(cx, h * 0.48, -Math.PI*0.6 + _tRandR(-0.1,0.1), h*0.05, w*0.002, 2);

            // Generate blossom dots within clusters
            for (const c of bl) {
                c.dots = [];
                const count = Math.floor(_tRandR(3, 6));
                for (let i = 0; i < count; i++) {
                    const ba = _tRand() * Math.PI * 2, bd = _tRand() * c.r;
                    c.dots.push({ ox: Math.cos(ba)*bd, oy: Math.sin(ba)*bd,
                        r: _tRandR(0.8, 2.2), bright: _tRand(), ph: _tRand()*Math.PI*2 });
                }
            }
            return { branches: br, blossoms: bl, tips };
        }
        const _stData = _genSakuraTree(800, 600);
        _sakuraTree.genFn = _genSakuraTree;
        _sakuraTree.branches = _stData.branches;
        _sakuraTree.blossoms = _stData.blossoms;
        _sakuraTree.tips = _stData.tips;
        _sakuraTree.lastW = 800;
        _sakuraTree.lastH = 600;
    }
    function _sakuraDrawTree(cw, ch, t) {
        if (Math.abs(cw - _sakuraTree.lastW) > 50 || Math.abs(ch - _sakuraTree.lastH) > 50) {
            const d = _sakuraTree.genFn(cw, ch);
            _sakuraTree.branches = d.branches;
            _sakuraTree.blossoms = d.blossoms;
            _sakuraTree.tips = d.tips;
            _sakuraTree.lastW = cw;
            _sakuraTree.lastH = ch;
        }
        ctx.save();

        // Gentle wind sway
        const windBase = Math.sin(t * 0.4) * 0.8 + Math.sin(t * 0.7) * 0.4;

        // Very subtle ambient glow
        const gcx = cw * 0.5, gcy = ch * 0.33;
        const pulse = 0.85 + 0.15 * Math.sin(t * 0.6);
        const ambR = Math.min(cw, ch) * 0.18;
        const amb = ctx.createRadialGradient(gcx, gcy, 0, gcx, gcy, ambR);
        amb.addColorStop(0, `rgba(240,180,200,${0.035 * pulse})`);
        amb.addColorStop(0.6, `rgba(220,160,185,${0.015 * pulse})`);
        amb.addColorStop(1, 'transparent');
        ctx.fillStyle = amb; ctx.beginPath(); ctx.arc(gcx, gcy, ambR, 0, Math.PI * 2); ctx.fill();

        // Draw branches — subtle, blended
        const sorted = [..._sakuraTree.branches].sort((a,b) => b.w0 - a.w0);
        for (const b of sorted) {
            const sway = windBase * b.d * 0.8;
            ctx.beginPath();
            ctx.moveTo(b.x0 + (b.d > 0 ? sway * 0.3 : 0), b.y0);
            if (b.cpx !== undefined) {
                ctx.quadraticCurveTo(b.cpx + sway * 0.5, b.cpy, b.x1 + sway, b.y1);
            } else {
                ctx.lineTo(b.x1 + sway, b.y1);
            }
            // Muted, semi-transparent bark
            const alpha = b.d === 0 ? 0.45 : b.d === 1 ? 0.32 : b.d === 2 ? 0.22 : 0.14;
            ctx.strokeStyle = `rgba(60,30,45,${alpha})`;
            ctx.lineWidth = b.w0;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();
        }

        // Draw blossom clusters — very soft, ghostly
        for (const cl of _sakuraTree.blossoms) {
            const clSway = windBase * 1.5;
            const clX = cl.x + clSway, clY = cl.y;

            // Faint cluster glow
            const cg = ctx.createRadialGradient(clX, clY, 0, clX, clY, cl.r * 2);
            cg.addColorStop(0, `rgba(255,200,220,${0.06 * pulse})`);
            cg.addColorStop(1, 'transparent');
            ctx.fillStyle = cg;
            ctx.fillRect(clX - cl.r*2, clY - cl.r*2, cl.r*4, cl.r*4);

            for (const dot of cl.dots) {
                const dp = 0.7 + 0.3 * Math.sin(t * 1.5 + dot.ph);
                const x = clX + dot.ox, y = clY + dot.oy;
                const r = dot.r * (0.85 + 0.15 * dp);
                const c = dot.bright > 0.5 ? [255,215,230] : [245,185,205];
                ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.2 + 0.2*dp})`;
                ctx.fill();
            }
        }
        ctx.restore();
    }

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
        const petal = _isSakura && Math.random() < 0.40;
        // Petals start scattered from tree area downward; stars anywhere
        let initX = Math.random() * canvas.width;
        let initY = Math.random() * canvas.height;
        if (petal && _sakuraTree.tips.length > 0) {
            const tip = _sakuraTree.tips[Math.floor(Math.random() * _sakuraTree.tips.length)];
            // Scale from 800x600 generation size to current canvas
            const sx = canvas.width / 800, sy = canvas.height / 600;
            initX = tip.x * sx + (Math.random() - 0.3) * 60 - Math.random() * canvas.width * 0.3;
            initY = tip.y * sy + Math.random() * canvas.height * 0.6;
        }
        stars.push({
            x: initX,
            y: initY,
            baseX: 0, baseY: 0,
            r: isBright ? Math.random() * 2.0 + 0.8 : Math.random() * 1.2 + 0.3,
            a: isBright ? Math.random() * 0.35 + 0.40 : Math.random() * 0.30 + 0.06,
            speed: Math.random() * 0.15 + 0.03,
            phase: Math.random() * Math.PI * 2,
            cr: pick.c.r, cg: pick.c.g, cb: pick.c.b,
            isBright: isBright,
            petal: petal,
            petalAngle: Math.random() * Math.PI * 2,
            petalSpin: (Math.random() - 0.5) * 0.008,
            petalSize: petal ? Math.random() * 3 + 2.5 : 0,
            petalSway: petal ? Math.random() * 0.4 + 0.2 : 0,
            petalFall: petal ? Math.random() * 0.25 + 0.1 : 0,
            petalAge: petal ? Math.random() * 200 : 0,
            petalSpiralR: petal ? Math.random() * 12 + 6 : 0,
            petalSpiralSpd: petal ? Math.random() * 1.5 + 1.0 : 0
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

        // Solar system (cosmos auth theme - tilted perspective, drawn first)
        if (_isCosmos) {
            const scx = canvas.width * 0.5, scy = canvas.height * 0.28;
            // Tilted orbit lines
            for (const p of _ssPlanets) _ssTiltedOrbit(scx, scy, p.dist, 0.07);
            _ssTiltedOrbit(scx, scy, 232, 0.03); _ssTiltedOrbit(scx, scy, 270, 0.03);
            // Collect renderables for depth sort
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
                else if (o.t === 'p') _ssDrawPlanet(o.x, o.y, o.p, o.d);
            }
        }

        // Black hole (nebula auth theme — P1 classic accretion disk)
        if (_isNebula) {
            const bhx = canvas.width * 0.5, bhy = canvas.height * 0.30;
            // Accretion disk glow rings
            _bhDrawDisk(bhx, bhy);
            // Depth-sorted particles
            const _bp = [];
            for (const p of _bhParticles) {
                const ang = p.angle + t * p.speed;
                const pr = _bhProject(bhx, bhy, p.dist, ang);
                _bp.push({ x: pr.x, y: pr.y + p.yOff, d: pr.depth, r: p.r, a: p.a, tier: p.tier });
            }
            _bp.sort((a, b) => a.d - b.d);
            // Back particles
            for (const p of _bp) {
                if (p.d > 0.1) continue;
                const c = _bhTierColors[p.tier];
                ctx.globalAlpha = p.a * (0.65 + p.d * 0.35);
                ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
            }
            // Void + photon ring + lensing
            _bhDrawVoid(bhx, bhy, t);
            // Front particles
            for (const p of _bp) {
                if (p.d <= 0.1) continue;
                const c = _bhTierColors[p.tier];
                ctx.globalAlpha = p.a * (0.6 + p.d * 0.4);
                if (p.tier === 0) {
                    const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 6);
                    glow.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${p.a * 0.35})`);
                    glow.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0)`);
                    ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 6, 0, Math.PI * 2); ctx.fill();
                }
                ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
            }
            ctx.globalAlpha = 1;
        }

        // Binary star system (aurora auth theme)
        if (_isAurora) {
            _binDrawSystem(canvas.width * 0.5, canvas.height * 0.28, t);
        }

        // Pulsar (noir auth theme)
        if (_isNoir) {
            _noirDrawPulsar(canvas.width * 0.5, canvas.height * 0.30, t);
        }

        // Rose bloom (rose auth theme)
        if (_isRose) {
            _roseDrawBloom(canvas.width * 0.5, canvas.height * 0.30, t);
        }

        // Ember (coffee auth theme)
        if (_isCoffee) {
            _coffeeDrawEmber(canvas.width * 0.5, canvas.height * 0.30, t);
        }

        // Crescent moon (midnight auth theme)
        if (_isMidnight) {
            _midnightDrawMoon(canvas.width * 0.5, canvas.height * 0.28, t, canvas.width);
        }

        // Sakura tree (sakura auth theme)
        if (_isSakura) {
            _sakuraDrawTree(canvas.width, canvas.height, t);
        }

        // Spawn shooting stars
        if (now - lastShooter > SHOOT_INTERVAL + Math.random() * 3000) {
            spawnShooter();
            lastShooter = now;
        }

        // Collect visible stars near mouse for connection lines
        const nearby = [];

        for (let i = 0; i < stars.length; i++) {
            const s = stars[i];

            // Gentle drift — petals fall down, stars drift up
            if (_isSakura && s.petal) {
                s.petalAge += 1;
                s.petalAngle += s.petalSpin;
                // Phase 1: spiral near tree (age < 80) — orbit around spawn point
                // Phase 2: detach and drift left + fall (age >= 80)
                const spiralPhase = Math.min(s.petalAge / 80, 1);
                const spiralFade = 1 - spiralPhase; // 1 at spawn, 0 when fully detached
                // Spiral motion (strong at start, fades out)
                const spiralX = Math.cos(t * s.petalSpiralSpd + s.phase) * s.petalSpiralR * spiralFade;
                const spiralY = Math.sin(t * s.petalSpiralSpd + s.phase) * s.petalSpiralR * spiralFade * 0.6;
                // Drift motion (grows in as spiral fades)
                const windGust = Math.sin(t * 0.3 + s.phase) * 0.15 + Math.sin(t * 0.8 + s.phase * 2) * 0.08;
                const driftX = (-0.3 - windGust) * spiralPhase + Math.sin(t * s.petalSway + s.phase) * 0.15;
                const driftY = s.petalFall * spiralPhase;
                s.baseX += driftX + (spiralX - (s._prevSpiralX || 0));
                s.baseY += driftY + (spiralY - (s._prevSpiralY || 0));
                s._prevSpiralX = spiralX;
                s._prevSpiralY = spiralY;
                if (s.baseY > canvas.height + 15 || s.baseX < -20) {
                    if (_sakuraTree.tips.length > 0) {
                        const tip = _sakuraTree.tips[Math.floor(Math.random() * _sakuraTree.tips.length)];
                        s.baseX = tip.x + (Math.random() - 0.3) * 20;
                        s.baseY = tip.y + (Math.random() - 0.5) * 10;
                    } else {
                        s.baseY = -10;
                        s.baseX = canvas.width * 0.3 + Math.random() * canvas.width * 0.5;
                    }
                    s.y = s.baseY; s.x = s.baseX;
                    s.petalAge = 0;
                    s._prevSpiralX = 0; s._prevSpiralY = 0;
                }
            } else {
                s.baseY -= DRIFT_SPEED;
                if (s.baseY < -10) {
                    s.baseY = canvas.height + 10;
                    s.baseX = Math.random() * canvas.width;
                    s.y = s.baseY; s.x = s.baseX;
                }
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

            // Sakura petals: bezier petal shape with spiral tumble
            if (_isSakura && s.petal) {
                const spiralPhase = Math.min(s.petalAge / 80, 1);
                // Tumble: petals appear to flip in 3D via scale pulsing
                const tumble = 0.7 + 0.3 * Math.sin(t * s.petalSpiralSpd * 1.5 + s.phase);
                const szBase = s.petalSize * ((!isTouch && dist < MOUSE_RADIUS) ? 1 + (1 - dist / MOUSE_RADIUS) * 0.3 : 1);
                const sz = szBase * (spiralPhase < 1 ? tumble : 0.85 + 0.15 * tumble);
                // Petals near tree are more opaque, fade slightly as they drift
                const petalAlpha = alpha * (1 - spiralPhase * 0.15);
                ctx.save();
                ctx.translate(s.x, s.y);
                ctx.rotate(s.petalAngle);
                ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${petalAlpha})`;
                ctx.beginPath();
                ctx.moveTo(0, sz * 1.6);
                ctx.bezierCurveTo(sz * 1.2, sz * 0.8, sz * 1.4, -sz * 0.6, sz * 0.3, -sz * 1.2);
                ctx.quadraticCurveTo(0, -sz * 0.7, -sz * 0.3, -sz * 1.2);
                ctx.bezierCurveTo(-sz * 1.4, -sz * 0.6, -sz * 1.2, sz * 0.8, 0, sz * 1.6);
                ctx.fill();
                // Vein line detail
                ctx.globalAlpha = petalAlpha * 0.12;
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

            // Soft glow on bright stars near cursor
            if (!isTouch && s.isBright && dist < MOUSE_RADIUS * 0.7) {
                ctx.globalAlpha = alpha * 0.12;
                ctx.beginPath();
                ctx.arc(s.x, s.y, radius * 3.5, 0, Math.PI * 2);
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
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
                Back
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
.auth-back-btn { position:absolute; top:14px; left:10px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.08); color:rgba(148,163,184,.85); cursor:pointer; padding:6px 12px 6px 8px; border-radius:8px; transition:color .2s, background .2s; display:flex; align-items:center; gap:4px; font-size:12px; font-weight:500; font-family:inherit; }
.auth-back-btn:hover { color:rgba(226,232,240,.95); background:rgba(255,255,255,.1); border-color:rgba(255,255,255,.15); }
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
