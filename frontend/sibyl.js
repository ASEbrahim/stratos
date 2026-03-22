// === SIBYL INTELLIGENCE HUE PANEL ===
// Fullscreen animated hue visualization panel with live data from /api/hue.
// Designed as a StratOS panel (like Markets or Settings).

(function() {
    'use strict';

    // ══════════════════════════════════════════════════
    // STATE
    // ══════════════════════════════════════════════════
    let _sibylInited = false;
    let _sibylVisible = false;
    let _sibylBgAnim = null;
    let _sibylHueAnim = null;
    let _sibylClockTimer = null;
    let _sibylData = null;

    // Canvas refs
    let _bgCanvas, _bgCtx, _hueCanvas, _hueCtx;
    let _W = 0, _H = 0;
    let _hexImg = null, _hexDirty = true;
    let _corridors = [];
    let _scanRings = [];

    // Hue ring state
    let _hV = 0, _TGT = 0, _hT = 0;
    let _sentinelAngle = 0;
    let _hueTimer = null;
    const CW = 680, CH = 680, CX = CW / 2, CY = CH / 2;
    const TAU = Math.PI * 2, SA = -Math.PI / 2;
    const R_OUT = 270, R_DIM = 228, R_WAVE = 200;
    const N_SEG = 60, SEG_GAP = 0.008;

    // Color system
    let _curR = 79, _curG = 195, _curB = 247;
    let _tgtR = 79, _tgtG = 195, _tgtB = 247;
    let _curHex = '#4fc3f7', _curLabel = 'Clear', _curStatus = 'NOMINAL';

    // Dimension values for ring arcs
    let _dims = [
        { val: 0, tier: 'hi' }, { val: 0, tier: 'hi' }, { val: 0, tier: 'mid' },
        { val: 0, tier: 'lo' }, { val: 0, tier: 'hi' }
    ];

    // ══════════════════════════════════════════════════
    // STYLES (injected once)
    // ══════════════════════════════════════════════════
    function injectStyles() {
        if (document.getElementById('sibyl-styles')) return;
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap';
        document.head.appendChild(link);

        const style = document.createElement('style');
        style.id = 'sibyl-styles';
        style.textContent = `
#sibyl-panel{position:relative;width:100%;height:100%;overflow:hidden;background:transparent;color:#dce8f4;font-family:'Rajdhani',sans-serif}
#sibyl-panel canvas.sibyl-bg{position:absolute;top:0;left:0;z-index:0;width:100%;height:100%}
#sibyl-panel .sibyl-scan{position:absolute;top:0;left:0;width:100%;height:1px;z-index:2;pointer-events:none;animation:sibyl-sc 16s linear infinite;opacity:0.7}
@keyframes sibyl-sc{0%{top:-1px}100%{top:100%}}
#sibyl-panel .sibyl-ui{position:absolute;inset:0;z-index:3;display:flex;flex-direction:column;padding:18px 28px 14px}
#sibyl-panel .sibyl-top-bar{display:flex;justify-content:space-between;align-items:center;padding-bottom:14px;flex-shrink:0}
#sibyl-panel .sibyl-sys-id{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:3px;color:rgba(79,195,247,0.35)}
#sibyl-panel .sibyl-sys-id .sibyl-brand{color:rgba(79,195,247,0.7);font-weight:400}
#sibyl-panel .sibyl-clock{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;color:rgba(138,156,184,0.4)}
#sibyl-panel .sibyl-divider{height:1px;flex-shrink:0;background:rgba(79,195,247,0.04)}
#sibyl-panel .sibyl-main{flex:1;display:flex;gap:28px;padding:14px 0;min-height:0;overflow:hidden}
#sibyl-panel .sibyl-col-side{width:300px;flex-shrink:0;display:flex;flex-direction:column;gap:8px;overflow:hidden}
#sibyl-panel .sibyl-col-side::-webkit-scrollbar{display:none}
#sibyl-panel .sibyl-col-center{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:280px;gap:12px}
#sibyl-panel .sibyl-bottom-bar{display:flex;justify-content:space-between;align-items:flex-end;padding-top:8px;flex-shrink:0}
#sibyl-panel .sibyl-status-line{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;color:rgba(79,195,247,0.2)}
#sibyl-panel .sibyl-stats{display:flex;gap:36px}
#sibyl-panel .sibyl-stat{text-align:right}
#sibyl-panel .sibyl-stat-val{font-size:26px;font-weight:600;line-height:1;color:rgba(79,195,247,0.7)}
#sibyl-panel .sibyl-stat-lbl{font-family:'Share Tech Mono',monospace;font-size:8px;letter-spacing:2px;text-transform:uppercase;margin-top:2px;color:rgba(138,156,184,0.35)}
#sibyl-panel .sibyl-panel-card{background:rgba(8,14,30,0.65);border:1px solid rgba(79,195,247,0.05);border-radius:10px;padding:14px 16px;position:relative;backdrop-filter:blur(24px);font-size:var(--sibyl-card-font, 13px)}
#sibyl-panel .sibyl-panel-card .sibyl-edge{position:absolute;top:-1px;left:16px;right:16px;height:1px;background:linear-gradient(90deg,transparent,rgba(79,195,247,0.08),transparent);border-radius:1px}
#sibyl-panel .sibyl-panel-title{font-family:'Share Tech Mono',monospace;font-size:calc(var(--sibyl-card-font, 13px) * 0.7);letter-spacing:3px;text-transform:uppercase;margin-bottom:10px;color:rgba(79,195,247,0.4)}
#sibyl-panel .sibyl-p-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0}
#sibyl-panel .sibyl-p-label{font-size:var(--sibyl-card-font, 13px);font-weight:400;color:rgba(138,156,184,0.7)}
#sibyl-panel .sibyl-p-value{font-family:'Share Tech Mono',monospace;font-size:calc(var(--sibyl-card-font, 13px) * 0.9);color:rgba(220,232,244,0.8)}
#sibyl-panel .sibyl-p-bar{width:100%;height:3px;background:rgba(79,195,247,0.04);border-radius:2px;margin-top:4px;overflow:hidden}
#sibyl-panel .sibyl-p-fill{height:100%;border-radius:2px;transition:width 2s cubic-bezier(0.23,1,0.32,1)}
#sibyl-panel .sibyl-fill-hi{background:linear-gradient(90deg,rgba(79,195,247,0.15),#4fc3f7)}
#sibyl-panel .sibyl-fill-mid{background:linear-gradient(90deg,rgba(129,199,132,0.15),#81c784)}
#sibyl-panel .sibyl-fill-lo{background:linear-gradient(90deg,rgba(255,183,77,0.15),#ffb74d)}
#sibyl-panel .sibyl-insight{font-size:var(--sibyl-card-font, 13px);line-height:1.6;color:rgba(138,156,184,0.5);max-height:4.8em;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
#sibyl-panel .sibyl-insight em{font-style:normal;font-weight:500;color:rgba(79,195,247,0.7)}
#sibyl-panel .sibyl-hue-block{text-align:center}
#sibyl-panel .sibyl-hue-outer{position:relative;display:inline-block;width:340px;height:340px}
#sibyl-panel .sibyl-hue-outer canvas{display:block;width:340px;height:340px}
#sibyl-panel .sibyl-hue-center{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
#sibyl-panel .sibyl-hue-num{font-size:72px;font-weight:700;line-height:1;letter-spacing:-2px;color:#4fc3f7;transition:color 0.8s}
#sibyl-panel .sibyl-hue-label{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:6px;text-transform:uppercase;margin-top:4px;color:rgba(79,195,247,0.5);transition:color 0.8s}
#sibyl-panel .sibyl-hue-sub{font-size:12px;letter-spacing:1px;color:rgba(138,156,184,0.35);margin-top:6px}
#sibyl-panel .sibyl-threat-row{display:flex;gap:6px;justify-content:center;margin-top:12px}
#sibyl-panel .sibyl-t-dot{width:7px;height:7px;border-radius:50%;transition:all 0.8s}
@media(max-width:1100px){
#sibyl-panel .sibyl-main{flex-direction:column;align-items:stretch;overflow-y:auto}
#sibyl-panel .sibyl-col-side{width:100%;flex-direction:row;flex-wrap:wrap;gap:10px}
#sibyl-panel .sibyl-col-side .sibyl-panel-card{flex:1;min-width:240px}
#sibyl-panel .sibyl-col-center{min-height:380px}
}
`;
        document.head.appendChild(style);
    }

    // ══════════════════════════════════════════════════
    // COLOR SYSTEM
    // ══════════════════════════════════════════════════
    function getTheme(v) {
        if (v >= 80) return { r: 79, g: 195, b: 247, label: 'Clear', status: 'NOMINAL', hex: '#4fc3f7' };
        if (v >= 55) return { r: 129, g: 199, b: 132, label: 'Stable', status: 'STABLE', hex: '#81c784' };
        if (v >= 30) return { r: 255, g: 213, b: 79, label: 'Clouded', status: 'ELEVATED', hex: '#ffd54f' };
        if (v >= 12) return { r: 255, g: 138, b: 101, label: 'Turbid', status: 'HIGH ALERT', hex: '#ff8a65' };
        return { r: 229, g: 115, b: 115, label: 'Critical', status: 'CRITICAL', hex: '#e57373' };
    }

    function cRgba(a) { return 'rgba(' + Math.round(_curR) + ',' + Math.round(_curG) + ',' + Math.round(_curB) + ',' + a + ')'; }
    function cStr() { return Math.round(_curR) + ',' + Math.round(_curG) + ',' + Math.round(_curB); }
    function lerpColors() {
        _curR += (_tgtR - _curR) * 0.04;
        _curG += (_tgtG - _curG) * 0.04;
        _curB += (_tgtB - _curB) * 0.04;
    }

    // ══════════════════════════════════════════════════
    // DATA CORRIDOR PARTICLES
    // ══════════════════════════════════════════════════
    function makeUnit(cx, baseSpeed) {
        var u = { cx: cx, baseSpeed: baseSpeed };
        resetUnit(u, true);
        return u;
    }
    function resetUnit(u, init) {
        u.y = init ? Math.random() * _H : _H + 10 + Math.random() * 100;
        u.speed = u.baseSpeed * (0.8 + Math.random() * 0.4);
        var roll = Math.random();
        if (roll < 0.08) { u.type = 2; u.size = 1.8 + Math.random() * 0.8; u.opacity = 0.3 + Math.random() * 0.2; }
        else if (roll < 0.22) { u.type = 1; u.rw = 1 + Math.random(); u.rh = 4 + Math.random() * 5; u.opacity = 0.06 + Math.random() * 0.1; }
        else { u.type = 0; u.size = 0.4 + Math.random() * 0.8; u.opacity = 0.025 + Math.random() * 0.06; }
        u.phase = Math.random() * TAU;
        u.drift = Math.sin(u.cx * 0.01) * 3;
    }
    function updateUnit(u) { u.y -= u.speed; u.phase += 0.008; if (u.y < -20) resetUnit(u, false); }
    function drawUnit(u) {
        var pulse = 0.75 + Math.sin(u.phase) * 0.25;
        var a = u.opacity * pulse;
        var x = u.cx + Math.sin(u.y * 0.002 + u.phase) * u.drift;
        if (u.type === 1) {
            _bgCtx.fillStyle = cRgba(a);
            _bgCtx.fillRect(x - u.rw / 2, u.y - u.rh / 2, u.rw, u.rh);
        } else if (u.type === 2) {
            _bgCtx.fillStyle = cRgba(a);
            _bgCtx.beginPath(); _bgCtx.arc(x, u.y, u.size, 0, TAU); _bgCtx.fill();
            _bgCtx.fillStyle = cRgba(a * 0.1);
            _bgCtx.beginPath(); _bgCtx.arc(x, u.y, u.size * 3, 0, TAU); _bgCtx.fill();
        } else {
            _bgCtx.fillStyle = cRgba(a);
            _bgCtx.beginPath(); _bgCtx.arc(x, u.y, u.size, 0, TAU); _bgCtx.fill();
        }
    }
    function buildCorridors() {
        _corridors = [];
        if (_W < 10 || _H < 10) return;
        var spacing = 48, count = Math.floor(_W / spacing);
        for (var i = 0; i < count; i++) {
            var x = spacing / 2 + i * spacing;
            var density = (i % 3 === 0) ? 4 : (i % 3 === 1) ? 2 : 1;
            var baseSpeed = 0.2 + (i % 5) * 0.08;
            var parts = [];
            for (var j = 0; j < density; j++) parts.push(makeUnit(x, baseSpeed));
            _corridors.push(parts);
        }
    }

    // ══════════════════════════════════════════════════
    // HEX GRID
    // ══════════════════════════════════════════════════
    function drawHex() {
        if (!_hexDirty && _hexImg) { _bgCtx.drawImage(_hexImg, 0, 0); return; }
        _hexImg = document.createElement('canvas'); _hexImg.width = _W; _hexImg.height = _H;
        var h = _hexImg.getContext('2d');
        var s = 48, hh = s * Math.sqrt(3);
        var cx = _W / 2, cy = _H / 2, maxD = Math.max(_W, _H) * 0.6;
        for (var r = -1; r < _H / hh + 2; r++) {
            for (var c = -1; c < _W / s / 1.5 + 2; c++) {
                var px = c * s * 1.5, py = r * hh + (c % 2 ? hh / 2 : 0);
                var dx = px - cx, dy = py - cy, dist = Math.sqrt(dx * dx + dy * dy);
                var fade = Math.min(1, Math.max(0, (dist - 100) / maxD));
                var alpha = 0.016 * fade;
                if (alpha < 0.001) continue;
                h.strokeStyle = cRgba(alpha);
                h.lineWidth = 0.4;
                h.beginPath();
                for (var i = 0; i < 6; i++) {
                    var a = Math.PI / 3 * i - Math.PI / 6;
                    h.lineTo(px + Math.cos(a) * s / 1.8, py + Math.sin(a) * s / 1.8);
                }
                h.closePath(); h.stroke();
            }
        }
        _hexDirty = false;
        _bgCtx.drawImage(_hexImg, 0, 0);
    }

    // ══════════════════════════════════════════════════
    // SCAN RINGS
    // ══════════════════════════════════════════════════
    function makeScanRing() { return resetScanRing({}); }
    function resetScanRing(s) {
        s.cx = _W / 2 + (Math.random() - 0.5) * 200;
        s.cy = _H / 2 + (Math.random() - 0.5) * 200;
        s.r = 0; s.maxR = 80 + Math.random() * 140;
        s.speed = 0.15 + Math.random() * 0.3;
        s.alpha = 0.04 + Math.random() * 0.04;
        return s;
    }
    function updateScanRing(s) { s.r += s.speed; if (s.r > s.maxR) resetScanRing(s); }
    function drawScanRing(s) {
        var life = 1 - s.r / s.maxR, a = s.alpha * life;
        if (a < 0.003) return;
        _bgCtx.save(); _bgCtx.translate(s.cx, s.cy); _bgCtx.beginPath();
        for (var i = 0; i <= 6; i++) {
            var an = i / 6 * TAU;
            if (i === 0) _bgCtx.moveTo(Math.cos(an) * s.r, Math.sin(an) * s.r);
            else _bgCtx.lineTo(Math.cos(an) * s.r, Math.sin(an) * s.r);
        }
        _bgCtx.closePath();
        _bgCtx.strokeStyle = cRgba(a);
        _bgCtx.lineWidth = 0.5; _bgCtx.stroke(); _bgCtx.restore();
    }

    // ══════════════════════════════════════════════════
    // HUE RING DRAW
    // ══════════════════════════════════════════════════
    function drawHueRing() {
        if (!_sibylVisible) return;
        _hueCtx.clearRect(0, 0, CW, CH);
        var filled = Math.floor(_hV / 100 * N_SEG);
        var cs = cStr();

        // 1. Segmented outer arc
        for (var i = 0; i < N_SEG; i++) {
            var a1 = SA + i * (TAU / N_SEG) + SEG_GAP;
            var a2 = SA + (i + 1) * (TAU / N_SEG) - SEG_GAP;
            _hueCtx.beginPath(); _hueCtx.arc(CX, CY, R_OUT, a1, a2);
            if (i < filled) {
                var t = filled > 0 ? i / filled : 0;
                _hueCtx.strokeStyle = 'rgba(' + cs + ',' + (0.12 + t * 0.5) + ')';
                _hueCtx.lineWidth = 4;
            } else {
                _hueCtx.strokeStyle = 'rgba(' + cs + ',0.02)';
                _hueCtx.lineWidth = 4;
            }
            _hueCtx.stroke();
        }

        // 2. Sentinels
        _sentinelAngle += 0.004;
        for (var i = 0; i < 4; i++) {
            var angle = _sentinelAngle + i * (TAU / 4);
            var sx = CX + Math.cos(angle) * R_OUT;
            var sy = CY + Math.sin(angle) * R_OUT;
            for (var t = 1; t <= 6; t++) {
                var ta = angle - 0.004 * t * 3;
                _hueCtx.beginPath(); _hueCtx.arc(CX + Math.cos(ta) * R_OUT, CY + Math.sin(ta) * R_OUT, 1, 0, TAU);
                _hueCtx.fillStyle = 'rgba(' + cs + ',' + (0.12 / t) + ')';
                _hueCtx.fill();
            }
            _hueCtx.beginPath(); _hueCtx.arc(sx, sy, 1.6, 0, TAU);
            _hueCtx.fillStyle = 'rgba(' + cs + ',0.45)';
            _hueCtx.fill();
            var sg = _hueCtx.createRadialGradient(sx, sy, 0, sx, sy, 10);
            sg.addColorStop(0, 'rgba(' + cs + ',0.06)');
            sg.addColorStop(1, 'rgba(' + cs + ',0)');
            _hueCtx.fillStyle = sg; _hueCtx.beginPath(); _hueCtx.arc(sx, sy, 10, 0, TAU); _hueCtx.fill();
        }

        // 3. Lead dot
        if (_hV > 0) {
            var ea = SA + (_hV / 100) * TAU;
            var ex = CX + Math.cos(ea) * R_OUT, ey = CY + Math.sin(ea) * R_OUT;
            var lg = _hueCtx.createRadialGradient(ex, ey, 0, ex, ey, 20);
            lg.addColorStop(0, 'rgba(' + cs + ',' + (0.2 + Math.sin(_hT * 2) * 0.06) + ')');
            lg.addColorStop(1, 'rgba(' + cs + ',0)');
            _hueCtx.fillStyle = lg; _hueCtx.beginPath(); _hueCtx.arc(ex, ey, 20, 0, TAU); _hueCtx.fill();
            _hueCtx.beginPath(); _hueCtx.arc(ex, ey, 3.5, 0, TAU);
            _hueCtx.fillStyle = 'rgba(' + cs + ',0.75)'; _hueCtx.fill();
        }

        // 4. Dimension arcs
        var arcSpan = TAU * 0.165, arcGap = TAU * 0.035;
        for (var i = 0; i < _dims.length; i++) {
            var d = _dims[i], sa = SA + i * (arcSpan + arcGap);
            _hueCtx.beginPath(); _hueCtx.arc(CX, CY, R_DIM, sa, sa + arcSpan);
            _hueCtx.strokeStyle = 'rgba(' + cs + ',0.03)';
            _hueCtx.lineWidth = 5; _hueCtx.lineCap = 'round'; _hueCtx.stroke();
            var fc = d.tier === 'hi' ? cs : d.tier === 'mid' ? '129,199,132' : '255,183,77';
            _hueCtx.beginPath(); _hueCtx.arc(CX, CY, R_DIM, sa, sa + arcSpan * (d.val / 100));
            _hueCtx.strokeStyle = 'rgba(' + fc + ',0.4)';
            _hueCtx.lineWidth = 5; _hueCtx.lineCap = 'round'; _hueCtx.stroke();
        }

        // 5. Waveform
        _hueCtx.beginPath();
        for (var i = 0; i <= 120; i++) {
            var a = (i / 120) * TAU;
            var wave = Math.sin(a * 8 + _hT * 0.8) * 2 + Math.sin(a * 13 - _hT * 0.5) * 1;
            var x = CX + Math.cos(a) * (R_WAVE + wave), y = CY + Math.sin(a) * (R_WAVE + wave);
            if (i === 0) _hueCtx.moveTo(x, y); else _hueCtx.lineTo(x, y);
        }
        _hueCtx.closePath();
        _hueCtx.strokeStyle = 'rgba(' + cs + ',0.03)'; _hueCtx.lineWidth = 0.6; _hueCtx.stroke();

        // 6. Ticks
        [0, 0.25, 0.5, 0.75].forEach(function(p) {
            var a = SA + p * TAU;
            _hueCtx.beginPath();
            _hueCtx.moveTo(CX + Math.cos(a) * (R_OUT - 12), CY + Math.sin(a) * (R_OUT - 12));
            _hueCtx.lineTo(CX + Math.cos(a) * (R_OUT + 12), CY + Math.sin(a) * (R_OUT + 12));
            _hueCtx.strokeStyle = 'rgba(' + cs + ',0.05)'; _hueCtx.lineWidth = 0.8; _hueCtx.stroke();
        });

        // 7. Center glow
        var br = 0.5 + Math.sin(_hT * 0.5) * 0.5;
        var cg = _hueCtx.createRadialGradient(CX, CY, 0, CX, CY, 150);
        cg.addColorStop(0, 'rgba(' + cs + ',' + (0.01 + br * 0.008) + ')');
        cg.addColorStop(0.5, 'rgba(' + cs + ',' + (0.003 + br * 0.002) + ')');
        cg.addColorStop(1, 'rgba(' + cs + ',0)');
        _hueCtx.fillStyle = cg; _hueCtx.beginPath(); _hueCtx.arc(CX, CY, 150, 0, TAU); _hueCtx.fill();
        _hueCtx.beginPath(); _hueCtx.arc(CX, CY, 175, 0, TAU);
        _hueCtx.strokeStyle = 'rgba(' + cs + ',0.012)'; _hueCtx.lineWidth = 0.3; _hueCtx.stroke();

        _hT += 0.02;
        _sibylHueAnim = requestAnimationFrame(drawHueRing);
    }

    // ══════════════════════════════════════════════════
    // BACKGROUND ANIMATION
    // ══════════════════════════════════════════════════
    function animBg() {
        if (!_sibylVisible) return;
        _bgCtx.clearRect(0, 0, _W, _H);
        // Semi-transparent overlay so hex grid/particles are visible but theme bg shows through
        _bgCtx.fillStyle = 'rgba(2, 6, 16, 0.3)';
        _bgCtx.fillRect(0, 0, _W, _H);
        lerpColors();
        drawHex();
        for (var i = 0; i < _scanRings.length; i++) { updateScanRing(_scanRings[i]); drawScanRing(_scanRings[i]); }
        for (var i = 0; i < _corridors.length; i++) {
            var parts = _corridors[i];
            for (var j = 0; j < parts.length; j++) { updateUnit(parts[j]); drawUnit(parts[j]); }
        }
        _sibylBgAnim = requestAnimationFrame(animBg);
    }

    // ══════════════════════════════════════════════════
    // RESIZE
    // ══════════════════════════════════════════════════
    function sibylResize() {
        var panel = document.getElementById('sibyl-panel');
        if (!panel || !_bgCanvas) return;
        _W = _bgCanvas.width = panel.offsetWidth;
        _H = _bgCanvas.height = panel.offsetHeight;
        _hexDirty = true;
        buildCorridors();
        _scanRings = [];
        for (var i = 0; i < 3; i++) _scanRings.push(makeScanRing());
    }

    // ══════════════════════════════════════════════════
    // THEME APPLICATION
    // ══════════════════════════════════════════════════
    function applyTheme() {
        var numEl = document.getElementById('sibyl-hue-num');
        var labelEl = document.getElementById('sibyl-hue-label');
        var brandEl = document.getElementById('sibyl-brand');
        var statusEl = document.getElementById('sibyl-status-line');
        if (numEl) numEl.style.color = _curHex;
        if (labelEl) { labelEl.style.color = _curHex + '80'; labelEl.textContent = _curLabel; }
        if (brandEl) brandEl.style.color = _curHex + 'B3';
        if (statusEl) statusEl.textContent = 'STRATOS :: SIBYL :: AREA STRESS \u2014 ' + _curStatus;

        // Threat dots
        var thresholds = [80, 55, 30, 12, 0];
        var dotColors = ['#4fc3f7', '#81c784', '#ffd54f', '#ff8a65', '#e57373'];
        for (var i = 0; i < 5; i++) {
            var el = document.getElementById('sibyl-td' + i);
            if (!el) continue;
            el.style.background = dotColors[i];
            var on = _hV >= thresholds[i];
            el.style.opacity = on ? '1' : '0.12';
            el.style.boxShadow = on ? '0 0 8px ' + dotColors[i] : 'none';
        }

        // Scan line color
        var scanEl = document.getElementById('sibyl-scan-line');
        if (scanEl) scanEl.style.background = 'linear-gradient(90deg, transparent, ' + cRgba(0.06) + ', transparent)';
    }

    // ══════════════════════════════════════════════════
    // COUNT-UP ANIMATION
    // ══════════════════════════════════════════════════
    function tickUp() {
        if (_hV < _TGT) {
            _hV++;
            var numEl = document.getElementById('sibyl-hue-num');
            if (numEl) numEl.textContent = _hV;
            applyTheme();
            _hueTimer = setTimeout(tickUp, 18);
        }
    }
    function tickDown() {
        if (_hV > _TGT) {
            _hV--;
            var numEl = document.getElementById('sibyl-hue-num');
            if (numEl) numEl.textContent = _hV;
            applyTheme();
            _hueTimer = setTimeout(tickDown, 18);
        }
    }

    function setHue(target) {
        _TGT = target;
        var info = getTheme(target);
        _tgtR = info.r; _tgtG = info.g; _tgtB = info.b;
        _curLabel = info.label; _curStatus = info.status; _curHex = info.hex;
        clearTimeout(_hueTimer);
        if (_hV < target) tickUp(); else tickDown();
        _hexDirty = true;
    }

    // ══════════════════════════════════════════════════
    // ESCAPE HTML
    // ══════════════════════════════════════════════════
    function esc(s) {
        var d = document.createElement('div');
        d.textContent = s || '';
        return d.innerHTML;
    }

    // ══════════════════════════════════════════════════
    // FILL CLASS BASED ON VALUE
    // ══════════════════════════════════════════════════
    function fillClass(v) {
        if (v >= 60) return 'sibyl-fill-hi';
        if (v >= 35) return 'sibyl-fill-mid';
        return 'sibyl-fill-lo';
    }

    function valueColor(v) {
        if (v >= 60) return 'color:rgba(79,195,247,0.7)';
        if (v >= 35) return '';
        return 'color:rgba(255,183,77,0.6)';
    }

    // ══════════════════════════════════════════════════
    // BUILD PANEL HTML
    // ══════════════════════════════════════════════════
    function buildPanelHTML() {
        return `
<canvas class="sibyl-bg" id="sibyl-bg-canvas"></canvas>
<div class="sibyl-scan" id="sibyl-scan-line"></div>
<div class="sibyl-ui">
  <div class="sibyl-top-bar">
    <div class="sibyl-sys-id"><span class="sibyl-brand" id="sibyl-brand">SIBYL</span> SYSTEM 4.2 \u2014 STRATOS INTELLIGENCE PLATFORM</div>
    <div class="sibyl-clock" id="sibyl-clock"></div>
  </div>
  <div class="sibyl-divider"></div>
  <div class="sibyl-main">
    <div class="sibyl-col-side" id="sibyl-left-panels"></div>
    <div class="sibyl-col-center">
      <div class="sibyl-hue-block">
        <div class="sibyl-hue-outer">
          <canvas id="sibyl-hue-canvas" width="680" height="680"></canvas>
          <div class="sibyl-hue-center">
            <div class="sibyl-hue-num" id="sibyl-hue-num">0</div>
            <div class="sibyl-hue-label" id="sibyl-hue-label">\u2014</div>
          </div>
        </div>
        <div class="sibyl-hue-sub">Intelligence hue reading</div>
        <div class="sibyl-threat-row">
          <div class="sibyl-t-dot" id="sibyl-td0"></div>
          <div class="sibyl-t-dot" id="sibyl-td1"></div>
          <div class="sibyl-t-dot" id="sibyl-td2"></div>
          <div class="sibyl-t-dot" id="sibyl-td3"></div>
          <div class="sibyl-t-dot" id="sibyl-td4"></div>
        </div>
      </div>
    </div>
    <div class="sibyl-col-side" id="sibyl-right-panels"></div>
  </div>
  <div class="sibyl-divider"></div>
  <div class="sibyl-bottom-bar">
    <div class="sibyl-status-line" id="sibyl-status-line">STRATOS :: SIBYL :: AREA STRESS \u2014 NOMINAL</div>
    <div class="sibyl-stats" id="sibyl-stats"></div>
  </div>
</div>`;
    }

    // ══════════════════════════════════════════════════
    // POPULATE FROM API DATA
    // ══════════════════════════════════════════════════
    function populateFromData(data) {
        _sibylData = data;
        var hue = data.hue || {};
        var bs = data.behavior_summary || {};
        var overall = hue.overall != null ? hue.overall : 0;
        var dims = hue.dimensions || {};
        var nudges = hue.nudges || [];

        // Extract from behavior_summary (actual API shape)
        var topCats = bs.top_categories || {};
        var topSources = bs.top_sources || {};
        var usage = bs.usage_patterns || {};
        var trajectory = bs.trajectory || 'unknown';
        var alignment = bs.alignment || {};
        var articleCount = bs.article_count || 0;
        var feedbackCount = bs.feedback_count || 0;

        // Set dimension values for ring arcs
        var freshness = dims.freshness || 0;
        var diversity = dims.diversity || 0;
        var coverage = dims.coverage || 0;
        var signal = dims.signal_strength || 0;
        var engagementDim = dims.engagement || 0;

        function tierOf(v) { return v >= 60 ? 'hi' : v >= 35 ? 'mid' : 'lo'; }
        _dims = [
            { val: freshness, tier: tierOf(freshness) },
            { val: diversity, tier: tierOf(diversity) },
            { val: coverage, tier: tierOf(coverage) },
            { val: signal, tier: tierOf(signal) },
            { val: engagementDim, tier: tierOf(engagementDim) }
        ];

        // ── Left panels: Feed Diagnostics + Source Reliability ──
        var leftHTML = '';

        // Feed Diagnostics
        leftHTML += '<div class="sibyl-panel-card"><span class="sibyl-edge"></span><div class="sibyl-panel-title">Feed diagnostics</div>';
        [['Freshness', freshness], ['Diversity', diversity], ['Coverage', coverage],
         ['Signal strength', signal], ['Engagement', engagementDim]].forEach(function(d) {
            leftHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">' + d[0] + '</span><span class="sibyl-p-value">' + d[1] + '</span></div>';
            leftHTML += '<div class="sibyl-p-bar"><div class="sibyl-p-fill ' + fillClass(d[1]) + '" style="width:' + d[1] + '%"></div></div>';
        });
        leftHTML += '</div>';

        // Source Reliability (from top_sources object: { "Serper/Google": { count, avg_score }, ... })
        leftHTML += '<div class="sibyl-panel-card"><span class="sibyl-edge"></span><div class="sibyl-panel-title">Source reliability</div>';
        var sourceEntries = Object.entries(topSources).slice(0, 5);
        if (sourceEntries.length === 0) {
            leftHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label" style="color:rgba(138,156,184,0.4)">No sources tracked</span></div>';
        } else {
            sourceEntries.forEach(function(entry) {
                var name = esc(entry[0]);
                var info = entry[1] || {};
                var avgScore = info.avg_score || 0;
                // Convert avg_score (0-10) to a reliability percentage, clamped
                var pct = Math.min(100, Math.max(0, Math.round(avgScore * 10)));
                leftHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">' + name + '</span><span class="sibyl-p-value" style="' + valueColor(pct) + '">' + pct + '%</span></div>';
                leftHTML += '<div class="sibyl-p-bar"><div class="sibyl-p-fill ' + fillClass(pct) + '" style="width:' + pct + '%"></div></div>';
            });
        }
        leftHTML += '</div>';
        document.getElementById('sibyl-left-panels').innerHTML = leftHTML;

        // ── Right panels: Behavioral Profile + Category Engagement + Inspector Note ──
        var rightHTML = '';

        // Behavioral Profile
        rightHTML += '<div class="sibyl-panel-card"><span class="sibyl-edge"></span><div class="sibyl-panel-title">Behavioral profile</div>';

        // Find top engaged category by count
        var catEntries = Object.entries(topCats);
        catEntries.sort(function(a, b) { return (b[1].count || 0) - (a[1].count || 0); });
        var topEngaged = catEntries.length > 0 ? catEntries[0][0] : 'N/A';

        // Compute overall click rate from top categories
        var totalFb = 0, totalClicks = 0;
        catEntries.forEach(function(e) {
            var v = e[1];
            totalClicks += (v.clicks || 0);
            totalFb += (v.clicks || 0) + (v.saves || 0) + (v.dismisses || 0) + (v.rates || 0);
        });
        var clickRate = totalFb > 0 ? Math.round((totalClicks / totalFb) * 100) : 0;

        // Alignment percentage
        var wellAligned = (alignment.well_aligned || []).length;
        var overDeclared = (alignment.over_declared || []).length;
        var totalDeclared = wellAligned + overDeclared;
        var alignPct = totalDeclared > 0 ? Math.round((wellAligned / totalDeclared) * 100) : (articleCount > 0 ? 60 : 0);

        var trajLabel = trajectory.charAt(0).toUpperCase() + trajectory.slice(1).replace(/_/g, ' ');
        var trajArrow = trajectory.indexOf('ris') >= 0 ? '\u25B2' : trajectory.indexOf('declin') >= 0 ? '\u25BC' : '\u25C6';
        var trajColor = trajectory.indexOf('ris') >= 0 ? '#81c784' : trajectory.indexOf('declin') >= 0 ? '#ff8a65' : 'rgba(220,232,244,0.8)';

        rightHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">Top engaged</span><span class="sibyl-p-value" style="color:rgba(79,195,247,0.8)">' + esc(topEngaged) + '</span></div>';
        rightHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">Click rate</span><span class="sibyl-p-value">' + clickRate + '%</span></div>';
        rightHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">Trajectory</span><span class="sibyl-p-value" style="color:' + trajColor + '">' + trajArrow + ' ' + esc(trajLabel) + '</span></div>';
        rightHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">Alignment</span><span class="sibyl-p-value">' + alignPct + '%</span></div>';
        rightHTML += '<div class="sibyl-p-bar"><div class="sibyl-p-fill ' + fillClass(alignPct) + '" style="width:' + alignPct + '%"></div></div>';
        rightHTML += '</div>';

        // Category Engagement (from top_categories: { "cat_name": { count, avg_score, clicks, ... } })
        rightHTML += '<div class="sibyl-panel-card"><span class="sibyl-edge"></span><div class="sibyl-panel-title">Category engagement</div>';
        var topCatSlice = catEntries.slice(0, 4);
        if (topCatSlice.length === 0) {
            rightHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label" style="color:rgba(138,156,184,0.4)">No categories</span></div>';
        } else {
            var maxCount = topCatSlice[0] ? (topCatSlice[0][1].count || 1) : 1;
            topCatSlice.forEach(function(entry) {
                var name = esc(entry[0]);
                var info = entry[1] || {};
                var pct = Math.round(((info.count || 0) / maxCount) * 100);
                var scoreStr = (info.avg_score || 0).toFixed(1);
                rightHTML += '<div class="sibyl-p-row"><span class="sibyl-p-label">' + name + '</span><span class="sibyl-p-value" style="' + valueColor(pct) + '">' + scoreStr + ' avg</span></div>';
                rightHTML += '<div class="sibyl-p-bar"><div class="sibyl-p-fill ' + fillClass(pct) + '" style="width:' + pct + '%"></div></div>';
            });
        }
        rightHTML += '</div>';

        // Inspector Note (from nudges — constrained height, no flex grow)
        rightHTML += '<div class="sibyl-panel-card"><span class="sibyl-edge"></span><div class="sibyl-panel-title">Inspector note</div>';
        if (nudges.length > 0) {
            var noteHTML = nudges.map(function(n) {
                var msg = esc(n.message || '');
                msg = msg.replace(/(\b(?:freshness|diversity|coverage|signal|engagement|tech|ai|market|finance|regional|scan|sources?|keywords?)\b)/gi, '<em>$1</em>');
                return msg;
            }).join(' ');
            rightHTML += '<div class="sibyl-insight">' + noteHTML + '</div>';
        } else {
            rightHTML += '<div class="sibyl-insight">All intelligence feeds operating within normal parameters. No actionable nudges at this time.</div>';
        }
        rightHTML += '</div>';
        document.getElementById('sibyl-right-panels').innerHTML = rightHTML;

        // Bottom stats
        var sourceCount = Object.keys(topSources).length;
        var lastScanHrs = usage.hours_since_last_scan;
        var lastScanStr = (lastScanHrs != null && lastScanHrs < 999) ? lastScanHrs.toFixed(1) + 'h' : '\u2014';

        var statsHTML = '';
        statsHTML += '<div class="sibyl-stat"><div class="sibyl-stat-val">' + articleCount + '</div><div class="sibyl-stat-lbl">Signals</div></div>';
        statsHTML += '<div class="sibyl-stat"><div class="sibyl-stat-val">' + sourceCount + '</div><div class="sibyl-stat-lbl">Sources</div></div>';
        statsHTML += '<div class="sibyl-stat"><div class="sibyl-stat-val">' + esc(lastScanStr) + '</div><div class="sibyl-stat-lbl">Last scan</div></div>';
        statsHTML += '<div class="sibyl-stat"><div class="sibyl-stat-val">' + feedbackCount + '</div><div class="sibyl-stat-lbl">Feedback</div></div>';
        document.getElementById('sibyl-stats').innerHTML = statsHTML;

        // Trigger hue count-up from 0 to actual value
        _hV = 0;
        setHue(Math.round(overall));
    }

    // ══════════════════════════════════════════════════
    // CLOCK
    // ══════════════════════════════════════════════════
    function startClock() {
        function tick() {
            var el = document.getElementById('sibyl-clock');
            if (!el) return;
            var n = new Date();
            el.textContent = n.toISOString().slice(0, 10) + ' ' + n.toTimeString().slice(0, 8) + ' UTC+3';
        }
        tick();
        _sibylClockTimer = setInterval(tick, 1000);
    }

    // ══════════════════════════════════════════════════
    // INIT SIBYL PANEL
    // ══════════════════════════════════════════════════
    window.initSibylPanel = function() {
        injectStyles();

        var panel = document.getElementById('sibyl-panel');
        if (!panel) return;

        // Build the inner HTML if not already built
        if (!_sibylInited) {
            panel.innerHTML = buildPanelHTML();
            _sibylInited = true;
        }

        _sibylVisible = true;

        // Setup canvases
        _bgCanvas = document.getElementById('sibyl-bg-canvas');
        _hueCanvas = document.getElementById('sibyl-hue-canvas');
        if (!_bgCanvas || !_hueCanvas) return;
        _bgCtx = _bgCanvas.getContext('2d');
        _hueCtx = _hueCanvas.getContext('2d');

        // Resize
        sibylResize();

        // Start animations
        if (_sibylBgAnim) cancelAnimationFrame(_sibylBgAnim);
        if (_sibylHueAnim) cancelAnimationFrame(_sibylHueAnim);
        animBg();
        drawHueRing();

        // Start clock
        if (_sibylClockTimer) clearInterval(_sibylClockTimer);
        startClock();

        // Fetch data
        var token = typeof getAuthToken === 'function' ? getAuthToken() : '';
        fetch('/api/hue', {
            headers: { 'X-Auth-Token': token }
        })
        .then(function(res) { if (!res.ok) throw new Error('API error'); return res.json(); })
        .then(function(data) { populateFromData(data); })
        .catch(function(e) {
            console.warn('[Sibyl] Failed to fetch hue data:', e);
            // Show fallback with zeros
            populateFromData({ hue: { overall: 0, label: 'No data', dimensions: {} }, sources: [], categories: [], engagement: {}, stats: {} });
        });
    };

    window.hideSibylPanel = function() {
        _sibylVisible = false;
        if (_sibylBgAnim) { cancelAnimationFrame(_sibylBgAnim); _sibylBgAnim = null; }
        if (_sibylHueAnim) { cancelAnimationFrame(_sibylHueAnim); _sibylHueAnim = null; }
        if (_sibylClockTimer) { clearInterval(_sibylClockTimer); _sibylClockTimer = null; }
        clearTimeout(_hueTimer);
    };

    // Handle window resize when visible
    window.addEventListener('resize', function() {
        if (_sibylVisible) {
            sibylResize();
            _hexDirty = true;
        }
    });

})();
