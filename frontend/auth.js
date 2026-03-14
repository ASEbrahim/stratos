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
                    // Show app immediately — don't block on UI settings load
                    const _app = document.querySelector('.flex.h-screen'); if (_app) _app.style.display = '';
                    init();
                    // Load full UI settings in background (non-blocking)
                    if (typeof loadUiSettingsFromServer === 'function') loadUiSettingsFromServer();
                    return;
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
            if (d.authenticated && d.active_profile) { setActiveProfile(d.active_profile); const _app = document.querySelector('.flex.h-screen'); if (_app) _app.style.display = ''; init(); return; }
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
/* Extracted to auth-star-canvas.js */
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
        if (r.ok && d.token) { setAuthToken(d.token); setActiveProfile(d.profile); _clearProfileLocalStorage(); if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state); if (typeof loadUiSettingsFromServer === 'function') await loadUiSettingsFromServer(); _dismiss(); }
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
        if (r.ok && d.token) { setAuthToken(d.token); setActiveProfile(d.profile); _clearProfileLocalStorage(); if (d.ui_state && typeof _applyUiStateFromServer === 'function') _applyUiStateFromServer(d.ui_state); if (typeof loadUiSettingsFromServer === 'function') await loadUiSettingsFromServer(); _dismiss(); }
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
            if (typeof loadUiSettingsFromServer === 'function') await loadUiSettingsFromServer();
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
            if (typeof loadUiSettingsFromServer === 'function') await loadUiSettingsFromServer();
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
            if (typeof loadUiSettingsFromServer === 'function') await loadUiSettingsFromServer();
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
    // Suppress ui-sync during bulk removal to prevent feedback loop
    if (typeof _uiSyncSuppressed !== 'undefined') _uiSyncSuppressed = true;
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
    if (typeof _uiSyncSuppressed !== 'undefined') _uiSyncSuppressed = false;
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
/* Extracted to auth-styles.js */
