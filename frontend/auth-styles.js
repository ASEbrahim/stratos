/**
 * STRAT_OS Auth — Styles
 * Extracted from auth.js. Loaded before auth.js.
 * Injects all auth overlay CSS into the document head.
 */

var _css = document.createElement('style');
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
