/**
 * STRAT_OS Auth — Interactive Star Canvas Engine
 * Extracted from auth.js. Loaded before auth.js.
 * Depends on: _t (defined in auth.js, available by the time _initStarParallax is called)
 */

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
    const _isSibyl = _t.name === 'Sibyl';

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
        ctx.strokeStyle = `rgba(120,150,210,${alpha})`; ctx.lineWidth = 0.8; ctx.stroke(); ctx.restore();
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
    // ── Noir: Pendulum clock ──
    const _noirNumerals = [];
    const _noirTrail = [];
    const _noirRipples = [];
    if (_isNoir) {
        const nums = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII'];
        for (let i = 0; i < 12; i++) {
            _noirNumerals.push({
                text: nums[i % nums.length],
                x: (Math.random() - 0.5) * 0.3,
                y: (Math.random() - 0.5) * 0.2,
                vx: (Math.random() - 0.5) * 0.08,
                vy: (Math.random() - 0.5) * 0.05,
                alpha: Math.random() * 0.1 + 0.02,
                size: Math.random() * 12 + 7,
                rotation: Math.random() * Math.PI * 2,
                rotSpeed: (Math.random() - 0.5) * 0.004
            });
        }
    }
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
        const time = t;
        const pendulumLen = 90;
        const maxAngle = 0.4;
        const angle = maxAngle * Math.sin(time * 1.2);
        const angVel = maxAngle * 1.2 * Math.cos(time * 1.2);
        const pivotX = cx, pivotY = cy - 45;
        const bobX = pivotX + Math.sin(angle) * pendulumLen;
        const bobY = pivotY + Math.cos(angle) * pendulumLen;

        // Mouse proximity boost — clock glows brighter when mouse is near
        const mdx = mouseX - pivotX, mdy = mouseY - pivotY;
        const mDist = Math.sqrt(mdx * mdx + mdy * mdy);
        const mBoost = mDist < 200 ? (1 - mDist / 200) * 0.4 : 0;

        // Clock face tick marks
        const clockR = 60;
        for (let i = 0; i < 12; i++) {
            const a = (i / 12) * Math.PI * 2 - Math.PI / 2;
            const iR = clockR - 7, oR = clockR;
            const ta = 0.18 + 0.1 * Math.sin(time * 2 + i * 0.5) + mBoost;
            ctx.beginPath();
            ctx.moveTo(pivotX + Math.cos(a) * iR, pivotY + Math.sin(a) * iR);
            ctx.lineTo(pivotX + Math.cos(a) * oR, pivotY + Math.sin(a) * oR);
            ctx.strokeStyle = `rgba(196,181,253,${ta})`; ctx.lineWidth = i % 3 === 0 ? 2 : 1; ctx.stroke();
            ctx.beginPath(); ctx.arc(pivotX + Math.cos(a) * oR, pivotY + Math.sin(a) * oR, 1.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(167,139,250,${ta + 0.05})`; ctx.fill();
        }

        // Sub-tick marks (60 positions) for clarity
        for (let i = 0; i < 60; i++) {
            if (i % 5 === 0) continue;
            const a = (i / 60) * Math.PI * 2 - Math.PI / 2;
            const sta = 0.06 + 0.04 * Math.sin(time * 1.5 + i * 0.3) + mBoost * 0.5;
            ctx.beginPath();
            ctx.moveTo(pivotX + Math.cos(a) * (clockR - 3), pivotY + Math.sin(a) * (clockR - 3));
            ctx.lineTo(pivotX + Math.cos(a) * clockR, pivotY + Math.sin(a) * clockR);
            ctx.strokeStyle = `rgba(139,92,246,${sta})`; ctx.lineWidth = 0.5; ctx.stroke();
        }

        // Clock face circle outline
        ctx.beginPath(); ctx.arc(pivotX, pivotY, clockR + 2, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(167,139,250,${0.08 + mBoost * 0.5})`; ctx.lineWidth = 0.6; ctx.stroke();

        // Floating roman numerals
        for (const n of _noirNumerals) {
            n.x += n.vx * 0.001; n.y += n.vy * 0.001; n.rotation += n.rotSpeed;
            if (Math.abs(n.x) > 0.25) n.vx *= -1;
            if (Math.abs(n.y) > 0.2) n.vy *= -1;
            ctx.save(); ctx.translate(cx + n.x * _cw, cy + n.y * _ch);
            ctx.rotate(n.rotation);
            ctx.font = `${n.size}px "Times New Roman", serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillStyle = `rgba(167,139,250,${n.alpha + mBoost * 0.3})`; ctx.fillText(n.text, 0, 0);
            ctx.restore();
        }

        // Gears — more visible
        const gs = time * 0.8;
        _noirDrawGear(pivotX - 16, pivotY - 8, 11, 15, 8, gs, 0.3 + mBoost);
        _noirDrawGear(pivotX + 14, pivotY - 6, 8, 11, 6, -gs * 1.33, 0.25 + mBoost);
        _noirDrawGear(pivotX + 4, pivotY - 20, 6, 9, 7, gs * 1.14, 0.2 + mBoost);
        _noirDrawGear(pivotX - 8, pivotY + 12, 5, 8, 5, -gs * 1.6, 0.18 + mBoost);

        // Ripples at swing extremes
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

        // Afterimage trail
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

        // Rod
        ctx.beginPath(); ctx.moveTo(pivotX, pivotY); ctx.lineTo(bobX, bobY);
        ctx.strokeStyle = `rgba(196,181,253,${0.4 + mBoost})`; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pivotX, pivotY); ctx.lineTo(bobX, bobY);
        ctx.strokeStyle = `rgba(167,139,250,${0.1 + mBoost * 0.3})`; ctx.lineWidth = 5; ctx.stroke();

        // Pivot dot
        ctx.beginPath(); ctx.arc(pivotX, pivotY, 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(196,181,253,${0.6 + mBoost})`; ctx.fill();

        // Bob glow — brighter, reactive
        const bp = 0.7 + 0.3 * Math.sin(time * 3);
        const bobDx = mouseX - bobX, bobDy = mouseY - bobY;
        const bobDist = Math.sqrt(bobDx * bobDx + bobDy * bobDy);
        const bobMB = bobDist < 100 ? (1 - bobDist / 100) * 0.3 : 0;
        const bg2 = ctx.createRadialGradient(bobX, bobY, 0, bobX, bobY, 35);
        bg2.addColorStop(0, `rgba(167,139,250,${0.2 + bobMB})`);
        bg2.addColorStop(0.5, `rgba(139,92,246,${0.08 + bobMB * 0.5})`);
        bg2.addColorStop(1, 'transparent');
        ctx.beginPath(); ctx.arc(bobX, bobY, 35, 0, Math.PI * 2); ctx.fillStyle = bg2; ctx.fill();
        const bg1 = ctx.createRadialGradient(bobX, bobY, 0, bobX, bobY, 14);
        bg1.addColorStop(0, `rgba(232,224,255,${(0.55 + bobMB) * bp})`);
        bg1.addColorStop(0.4, `rgba(167,139,250,${(0.35 + bobMB) * bp})`);
        bg1.addColorStop(1, 'rgba(139,92,246,0)');
        ctx.beginPath(); ctx.arc(bobX, bobY, 14, 0, Math.PI * 2); ctx.fillStyle = bg1; ctx.fill();
        ctx.beginPath(); ctx.arc(bobX, bobY, 5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232,224,255,${(0.65 + bobMB) * bp})`; ctx.fill();
        ctx.beginPath(); ctx.arc(bobX, bobY, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${(0.5 + bobMB) * bp})`; ctx.fill();
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

    // ── Coffee: Cup with rising steam ──
    const _coffeeSteam = [];
    const _coffeeBeans = [];
    if (_isCoffee) {
        const numStreams = 4;
        for (let s = 0; s < numStreams; s++) {
            for (let i = 0; i < 25; i++) {
                _coffeeSteam.push({
                    stream: s, streams: numStreams,
                    life: Math.floor(Math.random() * 180),
                    maxLife: Math.random() * 180 + 120,
                    speed: Math.random() * 0.25 + 0.12,
                    amplitude: Math.random() * 20 + 10,
                    frequency: Math.random() * 0.008 + 0.004,
                    phaseOff: Math.random() * Math.PI * 2,
                    size: Math.random() * 1.8 + 0.8,
                    baseX: 0, x: 0, y: 0,
                    bright: Math.random()
                });
            }
        }
        for (let i = 0; i < 12; i++) {
            _coffeeBeans.push({
                x: Math.random(), y: Math.random(),
                size: Math.random() * 4 + 2,
                rotation: Math.random() * Math.PI * 2,
                rotSpeed: (Math.random() - 0.5) * 0.003,
                vx: (Math.random() - 0.5) * 0.0001,
                vy: (Math.random() - 0.5) * 0.0001,
                alpha: Math.random() * 0.12 + 0.03
            });
        }
    }
    function _coffeeDrawCup(cx, cy, t) {
        const cw = _cw, ch = _ch;
        // Warm ambient glow behind steam
        const warmG = ctx.createRadialGradient(cx, cy - 20, 0, cx, cy - 20, 120);
        warmG.addColorStop(0, 'rgba(212,148,60,0.04)');
        warmG.addColorStop(0.5, 'rgba(184,122,46,0.02)');
        warmG.addColorStop(1, 'transparent');
        ctx.fillStyle = warmG; ctx.beginPath(); ctx.arc(cx, cy - 20, 120, 0, Math.PI * 2); ctx.fill();

        // Coffee beans drifting
        for (const b of _coffeeBeans) {
            b.x += b.vx; b.y += b.vy; b.rotation += b.rotSpeed;
            if (b.x < -0.05 || b.x > 1.05) b.vx *= -1;
            if (b.y < -0.05 || b.y > 1.05) b.vy *= -1;
            ctx.save(); ctx.translate(b.x * cw, b.y * ch); ctx.rotate(b.rotation);
            ctx.fillStyle = `rgba(184,122,46,${b.alpha})`;
            ctx.beginPath(); ctx.ellipse(-b.size * 0.15, 0, b.size * 0.4, b.size * 0.7, 0, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.ellipse(b.size * 0.15, 0, b.size * 0.4, b.size * 0.7, 0, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = `rgba(12,8,6,${b.alpha * 1.2})`; ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(0, -b.size * 0.5);
            ctx.bezierCurveTo(-b.size * 0.1, -b.size * 0.15, b.size * 0.1, b.size * 0.15, 0, b.size * 0.5);
            ctx.stroke(); ctx.restore();
        }

        // Cup position — above center, over logo
        const cupCx = cx, cupTop = cy - 30;
        const cupBot = cupTop + 40;
        const cupW = 26, cupWB = 21;

        // Saucer
        ctx.beginPath(); ctx.ellipse(cupCx, cupBot + 3, cupW * 1.3, 5, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(18,14,10,0.7)'; ctx.fill();
        ctx.strokeStyle = 'rgba(184,122,46,0.15)'; ctx.lineWidth = 0.8; ctx.stroke();

        // Cup body
        ctx.beginPath();
        ctx.moveTo(cupCx - cupW, cupTop); ctx.lineTo(cupCx - cupWB, cupBot);
        ctx.quadraticCurveTo(cupCx, cupBot + 7, cupCx + cupWB, cupBot);
        ctx.lineTo(cupCx + cupW, cupTop); ctx.closePath();
        ctx.fillStyle = 'rgba(18,14,10,0.8)'; ctx.fill();
        ctx.strokeStyle = 'rgba(184,122,46,0.25)'; ctx.lineWidth = 1; ctx.stroke();

        // Cup rim
        ctx.beginPath(); ctx.ellipse(cupCx, cupTop, cupW, 4, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(22,18,12,0.8)'; ctx.fill();
        ctx.strokeStyle = 'rgba(212,148,60,0.3)'; ctx.lineWidth = 0.8; ctx.stroke();

        // Handle
        ctx.beginPath();
        ctx.ellipse(cupCx + cupW + 8, cupTop + 14, 8, 13, 0, -Math.PI * 0.4, Math.PI * 0.5);
        ctx.strokeStyle = 'rgba(184,122,46,0.2)'; ctx.lineWidth = 1.5; ctx.stroke();

        // Steam particles — rise from cup
        const steamOriginY = cupTop - 3;
        const steamSpread = cupW * 0.6;
        for (const p of _coffeeSteam) {
            p.life++;
            if (p.life > p.maxLife) {
                p.life = 0;
                p.baseX = cupCx + ((p.stream / p.streams) - 0.5) * steamSpread * 2 + (Math.random() - 0.5) * 8;
                p.x = p.baseX; p.y = steamOriginY;
            }
            p.y -= p.speed;
            const prog = p.life / p.maxLife;
            p.x = p.baseX + Math.sin(p.life * p.frequency + p.phaseOff) * p.amplitude * (1 + prog);
            p.baseX += (Math.random() - 0.5) * 0.2;

            let alpha = 1;
            if (prog < 0.1) alpha = prog / 0.1;
            else if (prog > 0.5) alpha = 1 - (prog - 0.5) / 0.5;
            alpha *= 0.35;
            const sz = p.size * (1 + prog * 1.2);
            const col = p.bright > 0.7 ? '255,240,212' : p.bright > 0.4 ? '232,170,84' : '212,148,60';
            const sg = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, sz);
            sg.addColorStop(0, `rgba(${col},${alpha * 0.7})`);
            sg.addColorStop(0.5, `rgba(${col},${alpha * 0.25})`);
            sg.addColorStop(1, `rgba(${col},0)`);
            ctx.beginPath(); ctx.arc(p.x, p.y, sz, 0, Math.PI * 2);
            ctx.fillStyle = sg; ctx.fill();
        }

        // Tiny warm glow at cup center
        const pulse = 0.7 + 0.3 * Math.sin(t * 1.5);
        const tg = ctx.createRadialGradient(cupCx, cupTop, 0, cupCx, cupTop, 35);
        tg.addColorStop(0, `rgba(255,240,212,${0.03 * pulse})`);
        tg.addColorStop(1, 'transparent');
        ctx.fillStyle = tg; ctx.beginPath(); ctx.arc(cupCx, cupTop, 35, 0, Math.PI * 2); ctx.fill();
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
            const clx = cl.x * cw, cly = cl.y * _ch;
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
            const fy = (f.y + Math.sin(wp * 2) * f.wanderRy) * _ch;
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

    // ── Sibyl: Neural network brain ──
    let _sbBp = [], _sbBs, _sbBx, _sbBy;
    let _sbNodes = [], _sbLinks = [], _sbPulses = [];
    let _sbNextMajorPulse = 1;
    let _sbNextMinorPulse = 0.3;
    let _sbLastMajorHub = -1;
    let _sbSweepPos = 0;

    if (_isSibyl) {
        // Pre-computed brain outline points (no SVG DOM parsing needed)
        (function() {
            const _d=[3050,1361,3046,1362,3042,1362,3038,1363,3034,1363,3030,1364,3026,1364,3022,1365,3018,1365,3014,1365,3010,1366,3006,1366,3002,1366,2998,1367,2994,1367,2990,1367,2986,1368,2982,1368,2978,1368,2974,1369,2970,1369,2966,1369,2962,1370,2958,1370,2954,1370,2950,1371,2946,1371,2943,1371,2939,1371,2935,1372,2931,1372,2927,1372,2923,1372,2919,1373,2915,1373,2911,1373,2907,1373,2903,1374,2899,1374,2895,1374,2891,1374,2887,1375,2883,1375,2879,1375,2875,1375,2871,1375,2867,1376,2863,1376,2859,1376,2855,1376,2851,1377,2847,1377,2843,1377,2839,1377,2835,1377,2831,1378,2827,1378,2823,1378,2819,1378,2815,1378,2811,1378,2807,1379,2803,1379,2799,1379,2795,1379,2791,1379,2787,1380,2783,1380,2779,1380,2775,1380,2771,1380,2767,1380,2763,1381,2759,1381,2755,1381,2751,1381,2747,1381,2743,1381,2739,1382,2735,1382,2731,1382,2727,1382,2723,1382,2719,1383,2715,1383,2711,1383,2707,1383,2703,1384,2699,1384,2695,1384,2691,1384,2687,1385,2683,1385,2679,1385,2675,1385,2671,1386,2667,1386,2663,1386,2659,1387,2655,1387,2651,1387,2647,1388,2643,1388,2639,1388,2635,1389,2631,1389,2627,1389,2623,1390,2619,1390,2615,1390,2611,1391,2607,1391,2603,1392,2599,1392,2595,1392,2591,1393,2587,1393,2583,1394,2579,1394,2575,1394,2571,1395,2567,1395,2563,1396,2559,1396,2555,1397,2551,1397,2548,1398,2544,1398,2540,1399,2536,1399,2532,1400,2528,1400,2524,1401,2520,1402,2516,1402,2512,1403,2508,1403,2504,1404,2500,1405,2496,1405,2492,1406,2488,1406,2484,1407,2480,1408,2476,1408,2472,1409,2468,1410,2464,1410,2461,1411,2457,1412,2453,1413,2449,1413,2445,1414,2441,1415,2437,1416,2433,1416,2429,1417,2425,1418,2421,1419,2417,1420,2413,1421,2410,1421,2406,1422,2402,1423,2398,1424,2394,1425,2390,1426,2386,1427,2382,1428,2378,1429,2375,1430,2371,1431,2367,1432,2363,1433,2359,1434,2355,1435,2351,1436,2348,1437,2344,1438,2340,1440,2336,1441,2332,1442,2328,1443,2325,1444,2321,1445,2317,1447,2313,1448,2309,1449,2306,1451,2302,1452,2298,1453,2294,1455,2291,1456,2287,1457,2283,1459,2279,1460,2276,1462,2272,1463,2268,1465,2265,1466,2261,1468,2257,1469,2253,1471,2250,1473,2246,1474,2243,1476,2239,1478,2235,1479,2232,1481,2228,1483,2225,1485,2221,1486,2217,1488,2214,1490,2210,1492,2207,1494,2203,1496,2200,1498,2196,1500,2193,1502,2190,1504,2186,1506,2183,1508,2179,1510,2176,1513,2173,1515,2169,1517,2166,1519,2163,1522,2160,1524,2156,1526,2153,1529,2150,1531,2147,1533,2144,1536,2141,1538,2137,1541,2134,1544,2131,1546,2128,1549,2125,1551,2122,1554,2120,1557,2117,1560,2114,1562,2111,1565,2108,1568,2105,1571,2103,1574,2100,1577,2097,1580,2094,1583,2092,1586,2089,1589,2086,1591,2083,1594,2080,1597,2078,1600,2075,1603,2072,1605,2069,1608,2066,1610,2062,1613,2059,1615,2056,1617,2052,1620,2049,1622,2046,1624,2042,1625,2038,1627,2035,1628,2031,1630,2027,1631,2023,1632,2019,1633,2015,1633,2011,1634,2007,1635,2003,1636,2000,1636,1996,1637,1992,1638,1988,1639,1984,1640,1980,1641,1976,1643,1973,1644,1969,1645,1965,1647,1961,1648,1958,1650,1954,1651,1950,1653,1947,1654,1943,1656,1939,1658,1936,1660,1932,1662,1929,1663,1925,1665,1922,1667,1918,1669,1915,1672,1912,1674,1908,1676,1905,1678,1902,1680,1898,1682,1895,1685,1892,1687,1888,1689,1885,1692,1882,1694,1879,1696,1876,1699,1872,1701,1869,1704,1866,1706,1863,1709,1860,1711,1857,1714,1854,1716,1851,1719,1847,1721,1844,1724,1841,1727,1838,1729,1835,1732,1832,1734,1829,1737,1826,1740,1823,1742,1820,1745,1817,1748,1814,1750,1811,1753,1808,1756,1805,1758,1802,1761,1800,1764,1797,1766,1794,1769,1791,1772,1788,1775,1785,1777,1782,1780,1779,1783,1776,1786,1773,1788,1770,1791,1767,1794,1765,1797,1762,1799,1759,1802,1756,1805,1753,1808,1750,1811,1747,1813,1744,1816,1742,1819,1739,1822,1736,1824,1733,1827,1730,1830,1727,1833,1724,1836,1722,1839,1719,1841,1716,1844,1713,1847,1710,1850,1707,1853,1705,1855,1702,1858,1699,1861,1696,1864,1693,1867,1690,1870,1688,1872,1685,1875,1682,1878,1679,1881,1676,1884,1673,1887,1671,1889,1668,1892,1665,1895,1662,1898,1659,1901,1657,1904,1654,1907,1651,1909,1648,1912,1645,1915,1643,1918,1640,1921,1637,1924,1634,1927,1632,1929,1629,1932,1626,1935,1623,1938,1621,1941,1618,1944,1615,1947,1612,1950,1610,1953,1607,1956,1604,1959,1602,1962,1599,1965,1596,1968,1593,1971,1591,1974,1588,1977,1586,1980,1583,1983,1580,1986,1578,1989,1575,1992,1573,1995,1570,1998,1567,2001,1565,2004,1562,2007,1560,2010,1558,2013,1555,2017,1553,2020,1550,2023,1548,2026,1546,2030,1544,2033,1541,2036,1539,2040,1537,2043,1535,2046,1533,2050,1531,2053,1529,2057,1527,2060,1525,2064,1523,2067,1522,2071,1520,2075,1519,2078,1517,2082,1516,2086,1514,2090,1513,2093,1512,2097,1511,2101,1510,2105,1509,2109,1508,2113,1507,2117,1506,2121,1506,2125,1505,2129,1504,2132,1504,2136,1503,2140,1503,2144,1502,2148,1502,2152,1502,2156,1501,2160,1501,2164,1500,2168,1500,2172,1500,2176,1499,2180,1499,2184,1498,2188,1497,2192,1496,2196,1495,2200,1493,2203,1491,2207,1488,2210,1486,2213,1482,2215,1479,2217,1476,2219,1472,2221,1469,2223,1465,2225,1461,2226,1458,2228,1454,2229,1450,2231,1447,2233,1443,2235,1440,2238,1437,2240,1434,2242,1431,2245,1427,2247,1424,2250,1421,2253,1419,2255,1416,2258,1413,2261,1410,2264,1407,2267,1404,2269,1402,2272,1399,2275,1396,2278,1394,2281,1391,2284,1388,2287,1386,2290,1383,2293,1381,2296,1378,2300,1376,2303,1373,2306,1371,2309,1368,2312,1366,2315,1363,2318,1361,2322,1358,2325,1356,2328,1354,2331,1351,2335,1349,2338,1347,2341,1344,2344,1342,2348,1340,2351,1338,2354,1335,2357,1333,2361,1331,2364,1329,2367,1326,2371,1324,2374,1322,2377,1320,2381,1318,2384,1316,2388,1314,2391,1311,2394,1309,2398,1307,2401,1305,2405,1303,2408,1301,2411,1299,2415,1297,2418,1295,2422,1293,2425,1291,2429,1289,2432,1287,2436,1285,2439,1283,2443,1281,2446,1279,2450,1277,2453,1275,2457,1273,2460,1271,2464,1270,2467,1268,2471,1266,2474,1264,2478,1262,2481,1261,2485,1259,2489,1257,2492,1255,2496,1253,2499,1252,2503,1250,2507,1248,2510,1247,2514,1245,2518,1243,2521,1242,2525,1240,2529,1239,2532,1237,2536,1236,2540,1234,2543,1233,2547,1231,2551,1230,2554,1228,2558,1227,2562,1226,2566,1224,2569,1223,2573,1222,2577,1220,2581,1219,2585,1218,2589,1217,2592,1216,2596,1215,2600,1214,2604,1213,2608,1213,2612,1212,2616,1211,2620,1211,2624,1210,2628,1210,2632,1210,2636,1210,2640,1210,2644,1209,2648,1209,2652,1209,2656,1208,2660,1207,2664,1207,2667,1206,2671,1205,2675,1204,2679,1203,2683,1202,2687,1201,2691,1199,2694,1198,2698,1197,2702,1195,2706,1194,2710,1192,2713,1191,2717,1189,2721,1188,2724,1186,2728,1184,2731,1182,2735,1181,2739,1179,2742,1177,2746,1175,2749,1174,2753,1172,2757,1170,2760,1169,2764,1167,2768,1166,2771,1164,2775,1162,2779,1161,2782,1159,2786,1158,2790,1157,2794,1155,2797,1154,2801,1152,2805,1151,2809,1150,2812,1148,2816,1147,2820,1146,2824,1145,2828,1144,2831,1142,2835,1141,2839,1140,2843,1139,2847,1138,2851,1137,2854,1136,2858,1135,2862,1134,2866,1133,2870,1132,2874,1131,2878,1130,2882,1129,2886,1129,2890,1128,2893,1127,2897,1126,2901,1126,2905,1125,2909,1124,2913,1123,2917,1123,2921,1122,2925,1122,2929,1121,2933,1120,2937,1120,2941,1119,2945,1119,2949,1118,2953,1118,2957,1118,2961,1117,2965,1117,2969,1117,2973,1116,2977,1116,2981,1116,2985,1115,2989,1115,2993,1115,2997,1115,3001,1115,3005,1114,3009,1114,3013,1114,3017,1114,3021,1114,3025,1114,3029,1114,3033,1114,3037,1114,3041,1114,3045,1114,3049,1115,3053,1115,3057,1115,3060,1115,3064,1115,3068,1116,3072,1116,3076,1116,3080,1116,3084,1117,3088,1117,3092,1118,3096,1118,3100,1118,3104,1119,3108,1119,3112,1120,3116,1120,3120,1121,3124,1122,3128,1122,3132,1123,3136,1123,3140,1124,3144,1125,3148,1126,3152,1126,3156,1127,3160,1128,3164,1129,3167,1130,3171,1131,3175,1132,3179,1133,3183,1134,3187,1135,3191,1136,3195,1137,3198,1138,3202,1139,3206,1140,3210,1141,3214,1142,3218,1144,3221,1145,3225,1146,3229,1148,3233,1149,3237,1150,3240,1152,3244,1153,3248,1155,3251,1156,3255,1158,3259,1159,3263,1161,3266,1163,3270,1164,3274,1166,3277,1168,3281,1169,3284,1171,3288,1173,3291,1175,3295,1177,3299,1179,3302,1181,3306,1183,3309,1185,3312,1187,3316,1189,3319,1191,3323,1193,3326,1195,3329,1198,3333,1200,3336,1202,3339,1205,3342,1207,3346,1209,3349,1212,3352,1214,3355,1217,3358,1219,3361,1222,3364,1225,3367,1227,3370,1230,3373,1233,3376,1235,3379,1238,3382,1241,3385,1243,3388,1246,3391,1249,3394,1251,3397,1254,3400,1256,3403,1259,3407,1261,3410,1264,3413,1266,3416,1268,3420,1270,3423,1272,3427,1273,3430,1275,3434,1276,3438,1277,3442,1279,3445,1280,3449,1281,3453,1281,3457,1282,3461,1283,3465,1284,3469,1284,3473,1285,3477,1285,3481,1286,3485,1286,3489,1287,3493,1288,3497,1288,3501,1289,3505,1289,3509,1290,3513,1290,3517,1291,3520,1291,3524,1292,3528,1292,3532,1293,3536,1293,3540,1294,3544,1294,3548,1295,3552,1295,3556,1296,3560,1297,3564,1297,3568,1298,3572,1299,3576,1299,3580,1300,3584,1300,3588,1301,3592,1302,3596,1303,3600,1303,3604,1304,3607,1305,3611,1305,3615,1306,3619,1307,3623,1308,3627,1308,3631,1309,3635,1310,3639,1311,3643,1312,3647,1313,3651,1313,3655,1314,3658,1315,3662,1316,3666,1317,3670,1318,3674,1319,3678,1320,3682,1320,3686,1321,3690,1322,3693,1323,3697,1324,3701,1325,3705,1326,3709,1327,3713,1328,3717,1329,3721,1330,3724,1331,3728,1333,3732,1334,3736,1335,3740,1336,3744,1337,3748,1338,3751,1339,3755,1340,3759,1341,3763,1343,3767,1344,3771,1345,3774,1346,3778,1347,3782,1349,3786,1350,3790,1351,3793,1352,3797,1354,3801,1355,3805,1356,3809,1358,3812,1359,3816,1360,3820,1362,3824,1363,3827,1364,3831,1366,3835,1367,3839,1369,3842,1370,3846,1372,3850,1373,3854,1375,3857,1376,3861,1378,3865,1379,3868,1381,3872,1382,3876,1384,3879,1385,3883,1387,3887,1388,3890,1390,3894,1392,3898,1393,3901,1395,3905,1397,3909,1398,3912,1400,3916,1402,3920,1403,3923,1405,3927,1407,3930,1409,3934,1410,3938,1412,3941,1414,3945,1416,3948,1418,3952,1420,3955,1421,3959,1423,3962,1425,3966,1427,3969,1429,3973,1431,3976,1433,3980,1435,3983,1437,3987,1439,3990,1441,3994,1443,3997,1445,4001,1447,4004,1449,4008,1451,4011,1453,4014,1455,4018,1458,4021,1460,4025,1462,4028,1464,4031,1466,4035,1469,4038,1471,4041,1473,4045,1475,4048,1478,4051,1480,4054,1482,4058,1484,4061,1487,4064,1489,4067,1491,4071,1494,4074,1496,4077,1499,4080,1501,4083,1504,4087,1506,4090,1509,4093,1511,4096,1514,4099,1516,4102,1519,4105,1521,4108,1524,4111,1526,4114,1529,4118,1532,4121,1534,4124,1537,4127,1539,4130,1542,4132,1545,4135,1548,4138,1550,4141,1553,4144,1556,4147,1559,4150,1561,4153,1564,4156,1567,4158,1570,4161,1573,4164,1576,4167,1579,4170,1581,4172,1584,4175,1587,4178,1590,4181,1593,4183,1596,4186,1599,4189,1602,4191,1605,4194,1608,4196,1611,4199,1614,4202,1617,4204,1621,4207,1624,4209,1627,4212,1630,4214,1633,4216,1636,4219,1640,4221,1643,4224,1646,4226,1649,4228,1652,4231,1656,4233,1659,4235,1662,4238,1666,4240,1669,4242,1672,4244,1676,4247,1679,4249,1682,4251,1686,4253,1689,4255,1693,4257,1696,4259,1699,4261,1703,4263,1706,4265,1710,4267,1713,4269,1717,4271,1720,4273,1724,4275,1727,4277,1731,4279,1734,4280,1738,4282,1742,4284,1745,4286,1749,4288,1752,4289,1756,4291,1760,4293,1763,4294,1767,4296,1771,4297,1774,4299,1778,4301,1782,4302,1785,4304,1789,4305,1793,4306,1797,4308,1800,4309,1804,4311,1808,4312,1812,4313,1815,4315,1819,4316,1823,4317,1827,4318,1831,4320,1834,4321,1838,4322,1842,4323,1846,4324,1850,4325,1854,4326,1858,4327,1861,4328,1865,4329,1869,4330,1873,4331,1877,4332,1881,4333,1885,4334,1889,4335,1893,4336,1896,4337,1900,4337,1904,4338,1908,4339,1912,4340,1916,4340,1920,4341,1924,4342,1928,4342,1932,4343,1936,4343,1940,4344,1944,4344,1948,4345,1952,4345,1956,4346,1960,4346,1964,4347,1968,4347,1972,4347,1976,4348,1980,4348,1984,4348,1988,4349,1992,4349,1996,4349,2000,4349,2004,4349,2008,4350,2012,4350,2016,4350,2020,4350,2024,4350,2028,4350,2032,4350,2036,4350,2040,4350,2044,4350,2048,4350,2052,4350,2056,4351,2060,4352,2063,4353,2067,4354,2070,4356,2073,4359,2076,4362,2078,4366,2080,4369,2081,4373,2082,4377,2083,4381,2084,4385,2085,4388,2085,4392,2086,4396,2087,4400,2087,4404,2087,4408,2088,4412,2088,4416,2088,4420,2089,4424,2089,4428,2089,4432,2089,4436,2089,4440,2090,4444,2090,4448,2090,4452,2090,4456,2090,4460,2090,4464,2090,4468,2090,4472,2091,4476,2091,4480,2091,4484,2091,4488,2091,4492,2091,4496,2091,4500,2091,4504,2091,4508,2091,4512,2091,4516,2091,4520,2091,4524,2091,4528,2091,4532,2091,4536,2092,4540,2092,4544,2092,4548,2092,4552,2092,4556,2092,4560,2092,4564,2092,4568,2092,4572,2092,4576,2092,4580,2092,4584,2092,4588,2092,4592,2092,4596,2092,4600,2092,4604,2092,4608,2092,4612,2092,4616,2092,4620,2092,4624,2092,4628,2092,4632,2092,4636,2093,4640,2093,4644,2093,4648,2093,4652,2093,4656,2093,4660,2093,4664,2093,4668,2093,4672,2093,4676,2093,4680,2093,4684,2093,4688,2093,4692,2093,4696,2093,4700,2094,4704,2094,4708,2094,4712,2094,4716,2094,4720,2094,4724,2094,4728,2094,4732,2094,4736,2095,4740,2095,4744,2095,4748,2095,4752,2095,4756,2095,4760,2096,4764,2096,4768,2096,4772,2096,4776,2097,4780,2097,4784,2097,4788,2098,4792,2098,4796,2098,4800,2099,4804,2100,4808,2100,4812,2101,4816,2102,4820,2103,4824,2104,4827,2105,4831,2107,4835,2109,4838,2111,4842,2114,4845,2117,4847,2120,4849,2124,4851,2128,4853,2131,4854,2135,4855,2139,4857,2143,4858,2147,4859,2150,4860,2154,4861,2158,4863,2162,4864,2165,4866,2169,4867,2173,4869,2176,4871,2180,4873,2183,4875,2187,4877,2190,4879,2193,4881,2197,4883,2200,4886,2203,4888,2206,4890,2210,4893,2213,4895,2216,4898,2219,4900,2222,4903,2225,4905,2228,4908,2231,4911,2234,4913,2237,4916,2240,4919,2243,4921,2246,4924,2249,4927,2252,4929,2255,4932,2258,4935,2261,4938,2264,4940,2266,4943,2269,4946,2272,4949,2275,4951,2278,4954,2281,4957,2284,4960,2287,4963,2290,4965,2292,4968,2295,4971,2298,4974,2301,4976,2304,4979,2307,4982,2310,4985,2313,4987,2316,4990,2319,4993,2321,4996,2324,4998,2327,5001,2330,5004,2333,5006,2336,5009,2339,5012,2342,5015,2345,5017,2348,5020,2351,5022,2354,5025,2357,5028,2360,5030,2363,5033,2366,5035,2370,5038,2373,5040,2376,5042,2379,5045,2383,5047,2386,5049,2389,5051,2393,5053,2397,5055,2400,5057,2404,5058,2407,5060,2411,5061,2415,5063,2419,5064,2423,5065,2427,5066,2431,5066,2434,5067,2438,5067,2442,5068,2446,5068,2450,5069,2454,5069,2458,5069,2462,5069,2466,5070,2470,5070,2474,5070,2478,5070,2482,5070,2486,5070,2490,5070,2494,5070,2498,5070,2502,5070,2506,5070,2510,5070,2514,5070,2518,5070,2522,5070,2526,5070,2530,5070,2534,5070,2538,5070,2542,5070,2546,5070,2550,5070,2554,5069,2558,5069,2562,5069,2566,5069,2570,5068,2574,5068,2578,5067,2582,5066,2586,5065,2590,5064,2593,5062,2597,5060,2600,5058,2603,5055,2606,5052,2608,5049,2610,5046,2613,5042,2615,5039,2617,5035,2619,5032,2620,5028,2622,5025,2624,5021,2626,5018,2628,5014,2629,5010,2631,5007,2633,5003,2635,5000,2637,4996,2639,4993,2640,4989,2642,4986,2644,4982,2646,4979,2648,4975,2650,4972,2652,4968,2654,4965,2656,4961,2658,4958,2660,4954,2662,4951,2664,4947,2666,4944,2668,4940,2670,4937,2672,4933,2674,4930,2676,4927,2678,4923,2680,4920,2682,4916,2684,4913,2686,4909,2689,4906,2691,4903,2693,4899,2695,4896,2697,4893,2699,4889,2701,4886,2704,4883,2706,4879,2708,4876,2710,4873,2713,4869,2715,4866,2717,4863,2719,4859,2721,4856,2724,4853,2726,4849,2728,4846,2731,4843,2733,4839,2735,4836,2738,4833,2740,4830,2742,4826,2744,4823,2747,4820,2749,4817,2752,4814,2754,4810,2756,4807,2759,4804,2761,4801,2764,4797,2766,4794,2768,4791,2771,4788,2773,4785,2776,4782,2778,4778,2781,4775,2783,4772,2785,4769,2788,4766,2790,4763,2793,4760,2795,4756,2798,4753,2800,4750,2803,4747,2806,4744,2808,4741,2811,4738,2813,4735,2816,4732,2818,4729,2821,4726,2823,4723,2826,4719,2829,4716,2831,4713,2834,4710,2837,4707,2839,4704,2842,4701,2844,4698,2847,4695,2850,4692,2852,4689,2855,4686,2858,4683,2860,4680,2863,4677,2866,4675,2869,4672,2871,4669,2874,4666,2877,4663,2879,4660,2882,4657,2885,4654,2888,4651,2890,4648,2893,4645,2896,4642,2899,4640,2902,4637,2904,4634,2907,4631,2910,4628,2913,4625,2916,4622,2918,4620,2921,4617,2924,4614,2927,4611,2930,4608,2933,4606,2936,4603,2938,4600,2941,4597,2944,4595,2947,4592,2950,4589,2953,4586,2956,4584,2959,4581,2962,4578,2965,4575,2968,4573,2970,4570,2973,4567,2976,4565,2979,4562,2982,4559,2985,4557,2988,4554,2991,4551,2994,4549,2997,4546,3000,4543,3003,4541,3006,4538,3009,4535,3012,4533,3016,4530,3019,4528,3022,4525,3025,4523,3028,4520,3031,4517,3034,4515,3037,4512,3040,4510,3043,4507,3046,4505,3049,4502,3053,4500,3056,4497,3059,4495,3062,4492,3065,4490,3068,4487,3071,4484,3074,4482,3077,4479,3080,4477,3083,4474,3086,4471,3089,4469,3092,4466,3095,4464,3098,4461,3101,4458,3104,4456,3107,4453,3110,4450,3113,4448,3116,4445,3119,4442,3122,4440,3125,4437,3128,4434,3131,4432,3134,4429,3137,4426,3140,4423,3143,4421,3146,4418,3149,4415,3151,4412,3154,4410,3157,4407,3160,4404,3163,4401,3166,4398,3168,4396,3171,4393,3174,4390,3177,4387,3180,4384,3182,4381,3185,4378,3188,4375,3191,4372,3193,4369,3196,4366,3199,4363,3201,4360,3204,4357,3207,4354,3209,4351,3212,4348,3214,4345,3217,4342,3219,4339,3222,4336,3224,4333,3227,4330,3229,4326,3232,4323,3234,4320,3236,4317,3239,4314,3241,4310,3243,4307,3245,4304,3247,4300,3250,4297,3252,4293,3254,4290,3256,4287,3258,4283,3260,4280,3262,4276,3264,4273,3265,4269,3267,4265,3269,4262,3271,4258,3273,4255,3274,4251,3276,4248,3278,4244,3280,4241,3283,4238,3285,4234,3288,4232,3291,4229,3294,4226,3298,4224,3301,4222,3305,4221,3309,4220,3312,4218,3316,4217,3320,4216,3324,4216,3328,4215,3332,4214,3336,4214,3340,4213,3344,4213,3348,4213,3352,4212,3356,4212,3360,4212,3364,4212,3368,4211,3372,4211,3376,4211,3380,4211,3384,4211,3388,4211,3392,4211,3396,4211,3400,4211,3404,4210,3408,4210,3412,4210,3416,4210,3420,4210,3424,4210,3428,4210,3432,4210,3436,4210,3440,4210,3444,4210,3448,4210,3452,4210,3456,4210,3460,4210,3464,4210,3468,4210,3472,4210,3476,4210,3480,4210,3484,4210,3488,4210,3492,4210,3496,4210,3500,4210,3504,4210,3508,4210,3512,4210,3516,4210,3520,4210,3524,4210,3528,4210,3532,4210,3536,4210,3540,4210,3544,4210,3548,4210,3552,4210,3556,4210,3560,4210,3564,4210,3568,4210,3572,4210,3576,4210,3580,4210,3584,4210,3588,4210,3592,4210,3596,4210,3600,4210,3604,4210,3608,4210,3612,4210,3616,4210,3620,4210,3624,4210,3628,4210,3632,4209,3636,4209,3640,4209,3644,4209,3648,4209,3652,4209,3656,4209,3660,4209,3664,4209,3668,4209,3672,4209,3676,4208,3680,4208,3684,4208,3688,4208,3692,4207,3696,4207,3700,4207,3704,4207,3708,4206,3712,4206,3716,4205,3720,4205,3724,4204,3728,4203,3731,4202,3735,4201,3739,4200,3743,4199,3747,4198,3751,4197,3754,4195,3758,4194,3762,4192,3765,4190,3769,4189,3772,4187,3776,4185,3780,4183,3783,4181,3787,4180,3790,4178,3794,4176,3798,4175,3801,4173,3805,4172,3809,4170,3812,4169,3816,4167,3820,4166,3824,4164,3828,4163,3831,4162,3835,4161,3839,4160,3843,4160,3847,4160,3851,4159,3855,4158,3859,4157,3863,4156,3867,4155,3870,4154,3874,4153,3878,4151,3882,4150,3885,4149,3889,4147,3893,4146,3897,4144,3900,4143,3904,4141,3908,4140,3911,4138,3915,4137,3919,4135,3922,4133,3926,4132,3930,4130,3933,4128,3937,4127,3941,4125,3944,4123,3948,4122,3951,4120,3955,4118,3959,4116,3962,4114,3966,4113,3969,4111,3973,4109,3976,4107,3980,4105,3984,4104,3987,4102,3991,4100,3994,4098,3998,4096,4001,4094,4005,4092,4008,4090,4012,4089,4015,4087,4019,4085,4022,4083,4026,4081,4029,4079,4033,4077,4036,4075,4040,4073,4043,4071,4047,4069,4050,4067,4054,4065,4057,4063,4061,4061,4064,4059,4067,4057,4071,4055,4074,4053,4078,4051,4081,4049,4085,4047,4088,4045,4091,4042,4095,4040,4098,4038,4102,4036,4105,4034,4108,4032,4112,4030,4115,4027,4118,4025,4122,4023,4125,4021,4128,4019,4132,4016,4135,4014,4138,4012,4142,4010,4145,4007,4148,4005,4151,4003,4155,4000,4158,3998,4161,3996,4164,3993,4168,3991,4171,3989,4174,3986,4177,3984,4180,3981,4183,3979,4186,3976,4189,3973,4192,3971,4195,3968,4198,3965,4201,3963,4204,3960,4207,3957,4210,3954,4213,3951,4215,3949,4218,3946,4221,3943,4224,3940,4226,3937,4229,3934,4231,3931,4234,3928,4236,3924,4239,3921,4241,3918,4244,3915,4246,3912,4248,3908,4250,3905,4252,3901,4255,3898,4257,3895,4258,3891,4260,3888,4262,3884,4264,3880,4266,3877,4267,3873,4269,3869,4270,3866,4272,3862,4273,3858,4274,3854,4275,3851,4277,3847,4278,3843,4279,3839,4280,3835,4281,3831,4281,3827,4282,3823,4283,3819,4284,3816,4284,3812,4285,3808,4286,3804,4286,3800,4287,3796,4287,3792,4287,3788,4288,3784,4288,3780,4288,3776,4289,3772,4289,3768,4289,3764,4289,3760,4290,3756,4290,3752,4290,3748,4290,3744,4290,3740,4290,3736,4290,3732,4290,3728,4290,3724,4290,3720,4290,3716,4290,3712,4290,3708,4290,3704,4290,3700,4290,3696,4291,3692,4291,3688,4291,3684,4292,3680,4293,3676,4294,3672,4295,3668,4296,3665,4298,3661,4300,3658,4302,3654,4305,3652,4308,3649,4312,3647,4315,3645,4319,3643,4322,3641,4326,3640,4330,3639,4334,3638,4338,3637,4342,3636,4346,3636,4350,3635,4354,3635,4357,3634,4361,3634,4365,3633,4369,3633,4373,3633,4377,3632,4381,3632,4385,3632,4389,3632,4393,3631,4397,3631,4401,3631,4405,3631,4409,3631,4413,3631,4417,3631,4421,3631,4425,3631,4429,3630,4433,3630,4437,3630,4441,3630,4445,3630,4449,3630,4453,3630,4457,3630,4461,3630,4465,3630,4469,3630,4473,3630,4477,3630,4481,3630,4485,3630,4489,3630,4493,3630,4497,3630,4501,3630,4505,3630,4509,3630,4513,3630,4517,3630,4521,3630,4525,3630,4529,3630,4533,3630,4537,3630,4541,3630,4545,3630,4549,3630,4553,3630,4557,3630,4561,3630,4565,3630,4569,3630,4573,3630,4577,3630,4581,3630,4585,3630,4589,3630,4593,3630,4597,3630,4601,3630,4605,3630,4609,3629,4613,3629,4617,3629,4621,3629,4625,3629,4629,3629,4633,3629,4637,3629,4641,3629,4645,3629,4649,3629,4653,3629,4657,3629,4661,3628,4665,3628,4669,3628,4673,3628,4677,3628,4681,3628,4685,3628,4689,3628,4693,3627,4697,3627,4701,3627,4705,3627,4709,3627,4713,3627,4717,3626,4721,3626,4725,3626,4729,3626,4733,3626,4737,3626,4741,3625,4745,3625,4749,3625,4753,3625,4757,3624,4761,3624,4765,3624,4769,3624,4773,3623,4777,3623,4781,3623,4785,3623,4789,3622,4793,3622,4797,3622,4801,3621,4805,3621,4809,3621,4813,3620,4817,3620,4821,3620,4825,3619,4829,3619,4833,3619,4837,3618,4841,3618,4845,3617,4849,3617,4853,3617,4857,3616,4861,3616,4865,3615,4869,3615,4873,3614,4877,3614,4881,3613,4885,3613,4889,3612,4893,3612,4897,3611,4901,3611,4905,3610,4908,3609,4912,3609,4916,3608,4920,3608,4924,3607,4928,3606,4932,3606,4936,3605,4940,3604,4944,3603,4948,3603,4952,3602,4956,3601,4960,3600,4964,3600,4968,3599,4971,3598,4975,3597,4979,3596,4983,3595,4987,3595,4991,3594,4995,3593,4999,3592,5003,3591,5006,3590,5010,3589,5014,3588,5018,3587,5022,3586,5026,3584,5030,3583,5033,3582,5037,3581,5041,3580,5045,3579,5049,3577,5052,3576,5056,3575,5060,3574,5064,3572,5068,3571,5071,3569,5075,3568,5079,3567,5083,3565,5086,3564,5090,3562,5094,3561,5097,3559,5101,3558,5105,3556,5108,3554,5112,3553,5116,3551,5119,3549,5123,3548,5126,3546,5130,3544,5134,3542,5137,3540,5141,3539,5144,3537,5148,3535,5151,3533,5155,3531,5158,3529,5162,3527,5165,3525,5169,3523,5172,3521,5175,3519,5179,3516,5182,3514,5185,3512,5189,3510,5192,3508,5195,3505,5199,3503,5202,3501,5205,3499,5209,3496,5212,3494,5215,3491,5218,3489,5221,3487,5225,3484,5228,3482,5231,3479,5234,3477,5237,3474,5240,3472,5243,3469,5246,3467,5249,3464,5252,3461,5255,3459,5258,3456,5261,3454,5264,3451,5267,3448,5270,3445,5273,3443,5276,3440,5279,3437,5282,3435,5285,3432,5288,3429,5291,3426,5294,3424,5297,3421,5299,3418,5302,3415,5305,3412,5308,3410,5311,3407,5313,3404,5316,3401,5319,3398,5322,3395,5325,3392,5327,3389,5330,3386,5333,3383,5335,3380,5338,3377,5341,3374,5343,3371,5346,3368,5349,3365,5351,3362,5354,3359,5356,3356,5359,3353,5361,3350,5364,3347,5366,3344,5369,3341,5371,3338,5374,3334,5376,3331,5379,3328,5381,3325,5383,3322,5386,3318,5388,3315,5390,3312,5392,3308,5395,3305,5397,3302,5399,3298,5401,3295,5404,3292,5406,3288,5408,3285,5410,3282,5412,3278,5414,3275,5416,3271,5418,3268,5420,3264,5422,3261,5424,3257,5426,3254,5428,3250,5429,3247,5431,3243,5433,3240,5435,3236,5436,3232,5438,3229,5440,3225,5441,3221,5443,3218,5445,3214,5446,3210,5448,3207,5449,3203,5450,3199,5452,3195,5453,3192,5455,3188,5456,3184,5457,3180,5458,3177,5460,3173,5461,3169,5462,3165,5463,3161,5464,3157,5465,3153,5466,3150,5467,3146,5468,3142,5469,3138,5469,3134,5470,3130,5471,3126,5472,3122,5472,3118,5473,3114,5474,3110,5475,3106,5476,3103,5477,3099,5478,3095,5479,3091,5480,3087,5481,3083,5482,3079,5483,3076,5484,3072,5485,3068,5486,3064,5488,3060,5489,3056,5490,3053,5491,3049,5493,3045,5494,3041,5496,3038,5497,3034,5499,3030,5500,3026,5502,3023,5503,3019,5505,3016,5507,3012,5508,3008,5510,3005,5512,3001,5513,2997,5514,2993,5515,2990,5516,2986,5517,2982,5518,2978,5519,2974,5520,2970,5521,2966,5521,2962,5522,2958,5522,2954,5523,2950,5523,2946,5524,2942,5524,2938,5524,2934,5525,2930,5525,2926,5525,2922,5526,2918,5526,2914,5526,2910,5526,2906,5526,2902,5527,2898,5527,2894,5527,2890,5527,2886,5527,2882,5527,2878,5527,2874,5527,2870,5527,2866,5528,2862,5528,2858,5528,2854,5528,2850,5528,2846,5528,2842,5528,2838,5528,2834,5528,2830,5528,2826,5528,2822,5528,2818,5529,2814,5529,2810,5529,2806,5529,2802,5529,2798,5529,2794,5529,2790,5529,2786,5529,2782,5529,2778,5529,2774,5528,2770,5528,2766,5528,2762,5528,2758,5528,2754,5528,2750,5528,2747,5527,2743,5527,2739,5527,2735,5527,2731,5526,2727,5526,2723,5526,2719,5525,2715,5525,2711,5524,2707,5524,2703,5523,2699,5523,2695,5522,2691,5521,2687,5521,2683,5520,2679,5519,2675,5518,2671,5517,2667,5516,2664,5515,2660,5514,2656,5512,2652,5511,2648,5509,2645,5508,2641,5506,2637,5505,2634,5503,2630,5501,2627,5499,2623,5497,2620,5495,2616,5493,2613,5490,2610,5488,2606,5485,2603,5483,2600,5480,2597,5478,2594,5475,2591,5472,2588,5470,2585,5467,2582,5464,2580,5461,2577,5458,2574,5455,2571,5452,2569,5449,2566,5446,2563,5444,2560,5441,2558,5438,2555,5435,2552,5432,2549,5430,2546,5427,2543,5425,2540,5423,2536,5421,2533,5419,2529,5417,2526,5416,2522,5415,2518,5414,2514,5413,2510,5412,2506,5412,2502,5411,2498,5411,2494,5410,2490,5410,2486,5409,2482,5409,2478,5409,2474,5408,2470,5408,2466,5407,2462,5407,2459,5407,2455,5407,2451,5406,2447,5406,2443,5406,2439,5405,2435,5405,2431,5404,2427,5404,2423,5403,2419,5403,2415,5402,2411,5402,2407,5401,2403,5401,2399,5400,2395,5399,2391,5398,2387,5398,2383,5397,2379,5396,2375,5395,2371,5394,2368,5393,2364,5393,2360,5392,2356,5390,2352,5389,2348,5388,2344,5387,2340,5386,2337,5385,2333,5384,2329,5382,2325,5381,2321,5380,2318,5378,2314,5377,2310,5375,2306,5374,2303,5372,2299,5371,2295,5369,2292,5367,2288,5366,2285,5364,2281,5362,2277,5360,2274,5358,2270,5356,2267,5354,2263,5352,2260,5350,2256,5348,2253,5346,2250,5344,2246,5342,2243,5339,2240,5337,2236,5335,2233,5332,2230,5330,2227,5327,2224,5325,2220,5322,2217,5320,2214,5317,2211,5315,2208,5312,2205,5309,2202,5306,2199,5304,2197,5301,2194,5298,2191,5295,2188,5292,2185,5289,2183,5286,2180,5283,2177,5280,2175,5277,2172,5274,2170,5271,2167,5268,2165,5265,2162,5262,2160,5258,2157,5255,2155,5252,2153,5249,2150,5245,2148,5242,2146,5239,2144,5235,2142,5232,2140,5228,2137,5225,2135,5222,2133,5218,2131,5215,2129,5211,2127,5208,2125,5204,2124,5201,2122,5197,2120,5193,2118,5190,2116,5186,2115,5183,2113,5179,2111,5175,2109,5172,2108,5168,2106,5165,2104,5161,2102,5158,2100,5154,2098,5151,2096,5147,2094,5144,2092,5140,2090,5137,2088,5133,2086,5130,2084,5127,2082,5124,2079,5120,2077,5117,2074,5114,2072,5111,2069,5108,2066,5106,2064,5103,2061,5100,2058,5098,2055,5095,2051,5093,2048,5090,2045,5088,2042,5086,2039,5083,2035,5081,2032,5079,2029,5077,2025,5074,2022,5072,2019,5070,2015,5068,2012,5065,2009,5063,2006,5060,2003,5058,1999,5055,1996,5053,1993,5050,1990,5048,1987,5045,1984,5042,1981,5040,1978,5037,1975,5034,1972,5032,1969,5029,1966,5026,1963,5023,1961,5020,1958,5018,1955,5015,1952,5012,1949,5009,1947,5006,1944,5003,1941,5000,1939,4997,1936,4994,1933,4991,1931,4988,1928,4985,1926,4982,1923,4979,1920,4976,1918,4973,1915,4969,1913,4966,1910,4963,1908,4960,1906,4957,1903,4954,1901,4950,1898,4947,1896,4944,1894,4941,1891,4937,1889,4934,1887,4931,1884,4927,1882,4924,1880,4921,1878,4917,1876,4914,1873,4911,1871,4907,1869,4904,1867,4901,1865,4897,1863,4894,1861,4890,1859,4887,1857,4883,1854,4880,1852,4877,1850,4873,1848,4870,1846,4866,1845,4863,1843,4859,1841,4856,1839,4852,1837,4849,1835,4845,1833,4841,1831,4838,1830,4834,1828,4831,1826,4827,1824,4824,1822,4820,1821,4816,1819,4813,1817,4809,1816,4805,1814,4802,1812,4798,1811,4794,1809,4791,1807,4787,1806,4783,1804,4780,1803,4776,1801,4772,1800,4769,1798,4765,1797,4761,1795,4757,1794,4754,1792,4750,1791,4746,1790,4742,1788,4739,1787,4735,1786,4731,1784,4727,1783,4724,1782,4720,1781,4716,1779,4712,1778,4708,1777,4705,1775,4701,1774,4697,1773,4693,1771,4690,1770,4686,1768,4682,1767,4679,1765,4675,1763,4671,1762,4668,1760,4664,1758,4661,1756,4657,1754,4654,1752,4650,1750,4647,1748,4644,1746,4640,1744,4637,1741,4634,1739,4630,1737,4627,1734,4624,1732,4621,1730,4618,1727,4615,1725,4611,1722,4608,1719,4605,1717,4602,1714,4599,1712,4596,1709,4593,1706,4590,1704,4587,1701,4584,1698,4581,1696,4578,1693,4575,1690,4572,1688,4569,1685,4566,1682,4563,1680,4560,1677,4557,1674,4554,1672,4551,1669,4548,1667,4545,1664,4542,1662,4539,1659,4536,1657,4533,1654,4529,1652,4526,1649,4523,1647,4520,1645,4517,1642,4513,1640,4510,1637,4507,1635,4504,1633,4500,1631,4497,1628,4494,1626,4490,1624,4487,1622,4484,1620,4480,1618,4477,1616,4473,1614,4470,1611,4466,1609,4463,1608,4459,1606,4456,1604,4452,1602,4449,1600,4445,1598,4442,1596,4438,1595,4434,1593,4431,1591,4427,1590,4424,1588,4420,1586,4416,1585,4412,1583,4409,1582,4405,1580,4401,1579,4398,1578,4394,1576,4390,1575,4386,1574,4382,1572,4379,1571,4375,1570,4371,1569,4367,1568,4363,1567,4359,1566,4355,1565,4352,1564,4348,1563,4344,1562,4340,1561,4336,1560,4332,1560,4328,1559,4324,1558,4320,1557,4316,1557,4312,1556,4308,1556,4304,1555,4300,1554,4296,1554,4292,1554,4288,1553,4285,1553,4281,1552,4277,1552,4273,1552,4269,1551,4265,1551,4261,1551,4257,1551,4253,1551,4249,1550,4245,1550,4241,1550,4237,1550,4233,1550,4229,1550,4225,1550,4221,1550,4217,1550,4213,1549,4209,1549,4205,1548,4201,1548,4197,1546,4193,1545,4189,1544,4186,1542,4183,1540,4179,1537,4176,1534,4174,1531,4171,1528,4169,1525,4166,1522,4163,1520,4160,1517,4157,1514,4154,1512,4151,1509,4148,1507,4144,1504,4141,1502,4138,1500,4135,1498,4131,1495,4128,1493,4125,1491,4121,1489,4118,1487,4114,1485,4111,1483,4107,1481,4104,1479,4100,1477,4097,1475,4094,1473,4090,1471,4086,1469,4083,1467,4079,1465,4076,1463,4072,1462,4069,1460,4065,1458,4062,1456,4058,1454,4054,1453,4051,1451,4047,1449,4044,1447,4040,1446,4036,1444,4033,1442,4029,1441,4025,1439,4022,1437,4018,1436,4014,1434,4011,1433,4007,1431,4003,1430,4000,1428,3996,1427,3992,1425,3988,1424,3985,1422,3981,1421,3977,1420,3973,1418,3970,1417,3966,1416,3962,1414,3958,1413,3954,1412,3951,1411,3947,1410,3943,1409,3939,1408,3935,1407,3931,1407,3927,1406,3923,1405,3919,1404,3915,1404,3911,1403,3907,1403,3903,1402,3899,1401,3896,1401,3892,1401,3888,1400,3884,1400,3880,1399,3876,1399,3872,1398,3868,1398,3864,1398,3860,1397,3856,1397,3852,1397,3848,1396,3844,1396,3840,1396,3836,1395,3832,1395,3828,1395,3824,1395,3820,1394,3816,1394,3812,1394,3808,1394,3804,1393,3800,1393,3796,1393,3792,1393,3788,1393,3784,1392,3780,1392,3776,1392,3772,1392,3768,1392,3764,1391,3760,1391,3756,1391,3752,1391,3748,1391,3744,1391,3740,1390,3736,1390,3732,1390,3728,1390,3724,1390,3720,1389,3716,1389,3712,1389,3708,1389,3704,1389,3700,1388,3696,1388,3692,1388,3688,1388,3684,1388,3680,1387,3676,1387,3672,1387,3668,1387,3664,1386,3660,1386,3656,1386,3652,1386,3648,1385,3644,1385,3640,1385,3636,1384,3632,1384,3628,1384,3624,1384,3620,1383,3616,1383,3612,1383,3608,1382,3604,1382,3600,1382,3596,1381,3592,1381,3588,1381,3584,1380,3580,1380,3576,1380,3572,1379,3568,1379,3564,1379,3560,1378,3556,1378,3552,1377,3548,1377,3544,1377,3540,1376,3537,1376,3533,1375,3529,1375,3525,1374,3521,1373,3517,1373,3513,1372,3509,1371,3505,1370,3501,1369,3497,1368,3493,1368,3489,1367,3485,1366,3481,1366,3477,1365,3473,1365,3469,1364,3466,1364,3462,1364,3458,1363,3454,1363,3450,1363,3446,1362,3442,1362,3438,1362,3434,1361,3430,1361,3426,1361,3422,1361,3418,1360,3414,1360,3410,1360,3406,1360,3402,1359,3398,1359,3394,1359,3390,1359,3386,1359,3382,1358,3378,1358,3374,1358,3370,1358,3366,1358,3362,1357,3358,1357,3354,1357,3350,1357,3346,1357,3342,1357,3338,1357,3334,1356,3330,1356,3326,1356,3322,1356,3318,1356,3314,1356,3310,1356,3306,1356,3302,1355,3298,1355,3294,1355,3290,1355,3286,1355,3282,1355,3278,1355,3274,1355,3270,1355,3266,1355,3262,1355,3258,1355,3254,1355,3250,1355,3246,1354,3242,1354,3238,1354,3234,1354,3230,1354,3226,1354,3222,1354,3218,1354,3214,1354,3210,1354,3206,1354,3202,1354,3198,1354,3194,1354,3190,1354,3186,1354,3182,1354,3178,1354,3174,1354,3170,1354,3166,1354,3162,1354,3158,1354,3154,1355,3150,1355,3146,1355,3142,1355,3138,1355,3134,1355,3130,1355,3126,1355,3122,1355,3118,1355,3114,1356,3110,1356,3106,1356,3102,1356,3098,1356,3094,1356,3090,1357,3086,1357,3082,1357,3078,1357,3074,1358,3070,1358,3066,1359,3062,1359,3058,1360,3054,1360,3050,1361];
            _sbBp=[];for(let i=0;i<_d.length;i+=2)_sbBp.push({x:_d[i]/10,y:_d[i+1]/10});
        })();

        function _sbGetTransform() {
            const th = _ch * .28;
            _sbBs = th / 649;
            _sbBx = _cw / 2 - 660 * _sbBs / 2;
            _sbBy = _ch * .18 - th / 2; // brain sits above the logo
        }

        function _sbSpt(px, py) { return { x: _sbBx + px * _sbBs, y: _sbBy + py * _sbBs }; }

        function _sbInside(sx, sy) {
            const px = (sx - _sbBx) / _sbBs, py = (sy - _sbBy) / _sbBs; let ins = false;
            for (let i = 0, j = _sbBp.length - 1; i < _sbBp.length; j = i++) {
                const xi = _sbBp[i].x, yi = _sbBp[i].y, xj = _sbBp[j].x, yj = _sbBp[j].y;
                if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) ins = !ins;
            } return ins;
        }

        function _sbBuildNetwork() {
            _sbNodes = []; _sbLinks = [];
            const spacing = 22 * _sbBs * 660 / 300;
            const mx = _sbBx + 3, MX = _sbBx + 660 * _sbBs - 3, my = _sbBy + 3, MY = _sbBy + 649 * _sbBs - 3;
            let row = 0;
            for (let sy = my; sy < MY; sy += spacing * (.7 + Math.random() * .35)) {
                const off = (row % 2) ? (spacing * (.3 + Math.random() * .4)) : 0;
                for (let sx = mx + off; sx < MX; sx += spacing * (.75 + Math.random() * .5)) {
                    const jx = sx + (Math.random() - .5) * spacing * .55;
                    const jy = sy + (Math.random() - .5) * spacing * .55;
                    if (!_sbInside(jx, jy)) continue;
                    if (Math.random() < .15) continue;
                    _sbNodes.push({
                        x: jx, y: jy, bx: jx, by: jy,
                        r: .6 + Math.random() * .3,
                        activity: 0, connections: 0, threat: 0,
                        phase: Math.random() * Math.PI * 2,
                        pulseRing: 0, isHub: false,
                    });
                }
                row++;
            }
            const maxDist = spacing * 1.5;
            for (let i = 0; i < _sbNodes.length; i++) {
                const dists = [];
                for (let j = i + 1; j < _sbNodes.length; j++) {
                    const dx = _sbNodes[i].x - _sbNodes[j].x, dy = _sbNodes[i].y - _sbNodes[j].y;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < maxDist) dists.push({ j, d });
                }
                dists.sort((a, b) => a.d - b.d);
                for (let k = 0; k < Math.min(3, dists.length); k++) {
                    _sbLinks.push({ a: i, b: dists[k].j, dist: dists[k].d, pulseAlpha: 0, pulseDecay: 0 });
                    _sbNodes[i].connections++;
                    _sbNodes[dists[k].j].connections++;
                }
            }
            const avgConn = _sbNodes.reduce((s, n) => s + n.connections, 0) / Math.max(_sbNodes.length, 1);
            for (const n of _sbNodes) {
                if (n.connections >= avgConn * 1.3) { n.isHub = true; n.r = 1.0 + Math.random() * .4; }
            }
        }

        // Pre-computed screen points (rebuilt on transform change)
        let _sbScreenPts = [];
        function _sbCacheScreenPts() {
            _sbScreenPts = _sbBp.map(p => _sbSpt(p.x, p.y));
        }

        function _sbDrawOutline(t) {
            if (_sbScreenPts.length === 0) return;
            if (_sbScreenPts.length < 3) return;
            _sbSweepPos = (_sbSweepPos + .0008) % 1;
            const sweepIdx = Math.floor(_sbSweepPos * _sbScreenPts.length);
            const sweepW = _sbScreenPts.length * .12;
            const segLen = 5, gapLen = 3, totalSeg = segLen + gapLen;
            const bpLen = _sbScreenPts.length;
            const PI2 = Math.PI * 2;

            // Per-segment drawing with sweep boost (matches standalone)
            for (let i = 0; i < bpLen - 1; i++) {
                if (i % totalSeg >= segLen) continue;
                const pt = _sbScreenPts[i], pt2 = _sbScreenPts[i + 1];
                let dist = i - sweepIdx;
                if (dist < 0) dist += bpLen;
                if (dist > bpLen / 2) dist = bpLen - dist;
                const sweepBoost = dist < sweepW ? (.2 * (1 - dist / sweepW)) : 0;
                ctx.strokeStyle = 'rgba(79,195,247,' + (.09 + sweepBoost) + ')';
                ctx.lineWidth = sweepBoost > .06 ? 1.4 : .8;
                ctx.beginPath(); ctx.moveTo(pt.x, pt.y); ctx.lineTo(pt2.x, pt2.y); ctx.stroke();
                if (sweepBoost > .08) {
                    ctx.strokeStyle = 'rgba(160,230,255,' + (sweepBoost * .5) + ')';
                    ctx.lineWidth = .3;
                    ctx.beginPath(); ctx.moveTo(pt.x, pt.y); ctx.lineTo(pt2.x, pt2.y); ctx.stroke();
                }
            }

            // Sweep head glow (subtle radial gradient)
            const hp = _sbScreenPts[sweepIdx];
            const hg = ctx.createRadialGradient(hp.x,hp.y,0,hp.x,hp.y,10);
            hg.addColorStop(0,'rgba(79,195,247,.1)'); hg.addColorStop(.5,'rgba(79,195,247,.03)'); hg.addColorStop(1,'rgba(79,195,247,0)');
            ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(hp.x, hp.y, 10, 0, PI2); ctx.fill();
            ctx.fillStyle = 'rgba(200,240,255,.5)';
            ctx.beginPath(); ctx.arc(hp.x, hp.y, 1, 0, PI2); ctx.fill();

            // Second sweep (opposite, dimmer)
            const s2 = Math.floor(((_sbSweepPos + .5) % 1) * bpLen);
            const h2 = _sbScreenPts[s2];
            const hg2 = ctx.createRadialGradient(h2.x,h2.y,0,h2.x,h2.y,6);
            hg2.addColorStop(0,'rgba(79,195,247,.05)'); hg2.addColorStop(1,'rgba(79,195,247,0)');
            ctx.fillStyle = hg2; ctx.beginPath(); ctx.arc(h2.x, h2.y, 6, 0, PI2); ctx.fill();

            // Inner fill
            ctx.beginPath();
            ctx.moveTo(_sbScreenPts[0].x, _sbScreenPts[0].y);
            for (let i = 1; i < bpLen; i++) ctx.lineTo(_sbScreenPts[i].x, _sbScreenPts[i].y);
            ctx.closePath();
            ctx.fillStyle = 'rgba(79,195,247,.006)';
            ctx.fill();
        }

        function _sbDrawNetwork(t) {
            if (t > _sbNextMajorPulse) {
                const hubs = [];
                for (let i = 0; i < _sbNodes.length; i++) if (_sbNodes[i].isHub) hubs.push(i);
                let srcIdx;
                if (hubs.length > 1) {
                    const candidates = hubs.filter(h => h !== _sbLastMajorHub);
                    srcIdx = candidates[Math.floor(Math.random() * candidates.length)];
                } else {
                    srcIdx = hubs.length > 0 ? hubs[0] : Math.floor(Math.random() * _sbNodes.length);
                }
                _sbLastMajorHub = srcIdx;
                if (_sbNodes[srcIdx]) {
                    _sbNodes[srcIdx].activity = .6;
                    _sbNodes[srcIdx].pulseRing = .8;
                    let delay = 0;
                    const connLinks = [];
                    for (let li = 0; li < _sbLinks.length; li++) {
                        const l = _sbLinks[li];
                        if (l.a === srcIdx || l.b === srcIdx) connLinks.push(li);
                    }
                    for (let i = connLinks.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [connLinks[i], connLinks[j]] = [connLinks[j], connLinks[i]]; }
                    const initialCount = 1 + Math.floor(Math.random() * 2);
                    for (let ci = 0; ci < Math.min(initialCount, connLinks.length); ci++) {
                        const li = connLinks[ci];
                        const l = _sbLinks[li];
                        const toIdx = l.a === srcIdx ? l.b : l.a;
                        _sbPulses.push({ from: srcIdx, to: toIdx, linkIdx: li, progress: 0, speed: .002 + Math.random() * .002, strength: .9, generation: 0, delay: delay });
                        delay += 12 + Math.floor(Math.random() * 8);
                    }
                }
                _sbNextMajorPulse = t + 3.5;
            }
            if (t > _sbNextMinorPulse) {
                const idx = Math.floor(Math.random() * _sbNodes.length);
                if (_sbNodes[idx] && !_sbNodes[idx].isHub) {
                    _sbNodes[idx].activity = Math.max(_sbNodes[idx].activity, .15);
                    const connLinks = [];
                    for (let li = 0; li < _sbLinks.length; li++) {
                        if (_sbLinks[li].a === idx || _sbLinks[li].b === idx) connLinks.push(li);
                    }
                    if (connLinks.length > 0) {
                        const li = connLinks[Math.floor(Math.random() * connLinks.length)];
                        const l = _sbLinks[li];
                        _sbPulses.push({ from: idx, to: l.a === idx ? l.b : l.a, linkIdx: li, progress: 0, speed: .003 + Math.random() * .003, strength: .3, generation: 0, delay: 10 });
                    }
                }
                _sbNextMinorPulse = t + 1.0 + Math.random() * .5;
            }
            for (const l of _sbLinks) {
                const a = _sbNodes[l.a], b = _sbNodes[l.b];
                const act = Math.max(a.activity, b.activity);
                const baseA = .018 + act * .03 + l.pulseAlpha * .15;
                ctx.strokeStyle = `rgba(79,195,247,${baseA})`;
                ctx.lineWidth = .3 + l.pulseAlpha * .6;
                ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
                if (l.pulseAlpha > .01) {
                    ctx.strokeStyle = `rgba(180,235,255,${l.pulseAlpha * .5})`;
                    ctx.lineWidth = .2 + l.pulseAlpha * .4;
                    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
                }
                if (l.pulseDecay > 0) {
                    l.pulseAlpha = l.pulseDecay;
                    l.pulseDecay *= .92;
                    if (l.pulseDecay < .005) l.pulseDecay = 0;
                }
            }
            for (let i = _sbPulses.length - 1; i >= 0; i--) {
                const p = _sbPulses[i];
                if (p.delay && p.delay > 0) {
                    p.delay--;
                    const src = _sbNodes[p.from];
                    if (src) src.activity = Math.max(src.activity, p.strength * .3);
                    continue;
                }
                p.progress += p.speed;
                if (p.linkIdx >= 0 && p.linkIdx < _sbLinks.length) {
                    const targetAlpha = p.strength * .7 * p.progress;
                    _sbLinks[p.linkIdx].pulseDecay = Math.max(_sbLinks[p.linkIdx].pulseDecay, targetAlpha);
                    _sbLinks[p.linkIdx].pulseAlpha += (targetAlpha - _sbLinks[p.linkIdx].pulseAlpha) * .08;
                }
                if (p.progress >= 1) {
                    const tgt = _sbNodes[p.to];
                    tgt.activity = Math.min(1, tgt.activity + .3 * p.strength);
                    tgt.pulseRing = Math.max(tgt.pulseRing, p.strength * .7);
                    const cascadeProb = .6 - p.generation * .03;
                    if (p.generation < 12 && Math.random() < cascadeProb) {
                        const cascadeLinks = [];
                        for (let li = 0; li < _sbLinks.length; li++) {
                            const l = _sbLinks[li];
                            if ((l.a === p.to || l.b === p.to) && l.a !== p.from && l.b !== p.from) cascadeLinks.push(li);
                        }
                        for (let ci = cascadeLinks.length - 1; ci > 0; ci--) { const cj = Math.floor(Math.random() * (ci + 1)); [cascadeLinks[ci], cascadeLinks[cj]] = [cascadeLinks[cj], cascadeLinks[ci]]; }
                        const maxBranch = p.generation < 3 ? (1 + Math.floor(Math.random() * 3)) : (Math.random() < .6 ? 1 : 2);
                        let branched = 0;
                        let cascadeDelay = 10 + Math.floor(Math.random() * 12);
                        for (const li of cascadeLinks) {
                            if (branched >= maxBranch) break;
                            const l = _sbLinks[li];
                            _sbPulses.push({ from: p.to, to: l.a === p.to ? l.b : l.a, linkIdx: li, progress: 0, speed: .002 + Math.random() * .002, strength: p.strength * (.55 + Math.random() * .15), generation: p.generation + 1, delay: cascadeDelay });
                            cascadeDelay += 6 + Math.floor(Math.random() * 8);
                            branched++;
                        }
                    }
                    _sbPulses.splice(i, 1);
                    continue;
                }
                const a = _sbNodes[p.from], b = _sbNodes[p.to];
                if (!a || !b) { _sbPulses.splice(i, 1); continue; }
                const px = a.x + (b.x - a.x) * p.progress;
                const py = a.y + (b.y - a.y) * p.progress;
                const alpha = p.strength * (1 - Math.abs(p.progress - .5) * 1.5);
                const clampA = Math.max(0, Math.min(1, alpha));
                ctx.fillStyle = `rgba(79,195,247,${clampA * .08})`;
                ctx.beginPath(); ctx.arc(px, py, 10, 0, Math.PI * 2); ctx.fill();
                ctx.fillStyle = `rgba(120,215,255,${clampA * .25})`;
                ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
                ctx.fillStyle = `rgba(220,245,255,${clampA * .8})`;
                ctx.beginPath(); ctx.arc(px, py, 1.8, 0, Math.PI * 2); ctx.fill();
            }
            for (const n of _sbNodes) {
                n.x = n.bx + Math.sin(t * .03 + n.phase) * .3;
                n.y = n.by + Math.cos(t * .025 + n.phase) * .2;
                if (n.activity > 0) n.activity *= .985;
                if (n.pulseRing > 0) n.pulseRing *= .97;
                if (n.threat > 0) n.threat *= .985;
                const r = n.r + n.activity * .6;
                if (n.pulseRing > .02) {
                    const ringR = r + 12 * (1 - n.pulseRing);
                    ctx.beginPath(); ctx.arc(n.x, n.y, ringR, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(79,195,247,${n.pulseRing * .2})`;
                    ctx.lineWidth = .6; ctx.stroke();
                }
                if (n.activity > .05) {
                    // Simple halo circle instead of expensive radialGradient
                    const hR = r * (n.isHub ? 5 : 3.5);
                    ctx.fillStyle = `rgba(79,195,247,${n.activity * (n.isHub ? .04 : .025)})`;
                    ctx.beginPath(); ctx.arc(n.x, n.y, hR, 0, Math.PI * 2); ctx.fill();
                }
                if (n.isHub) {
                    ctx.beginPath(); ctx.arc(n.x, n.y, r + 2, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(79,195,247,${.05 + n.activity * .1})`;
                    ctx.lineWidth = .35; ctx.stroke();
                    ctx.beginPath(); ctx.arc(n.x, n.y, r + .8, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(120,210,255,${.06 + n.activity * .12})`;
                    ctx.lineWidth = .25; ctx.stroke();
                }
                // Threat glow — Crime Coefficient spike (red/orange)
                if (n.threat > .05) {
                    ctx.fillStyle = `rgba(255,${Math.floor(100 + n.threat * 60)},60,${n.threat * .2})`;
                    ctx.beginPath(); ctx.arc(n.x, n.y, r * 5, 0, Math.PI * 2); ctx.fill();
                }
                const somaA = n.isHub ? (.08 + n.activity * .4) : (.05 + n.activity * .3);
                ctx.fillStyle = n.threat > .1 ? `rgba(255,160,80,${somaA + n.threat * .3})` : `rgba(79,195,247,${somaA})`;
                ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
                if (n.activity > .1) {
                    ctx.fillStyle = `rgba(200,240,255,${n.activity * .4})`;
                    ctx.beginPath(); ctx.arc(n.x, n.y, r * .4, 0, Math.PI * 2); ctx.fill();
                }
            }
        }

        // ── Scan rings, data lines, threat — deferred init (need _cw/_ch) ──
        const _sbScanRings = [];
        const _sbDataLines = [];
        let _sbNextThreat = 0;
        let _sbExtrasInited = false;

        function _sbInitExtras() {
            if (_sbExtrasInited) return;
            _sbExtrasInited = true;
            _sbNextThreat = Date.now() * .001 + 8 + Math.random() * 10;
            // Scan rings — across full screen like in-app Sibyl theme
            for (let i = 0; i < 4; i++) {
                _sbScanRings.push({
                    x: _cw * (.15 + Math.random() * .7), y: _ch * (.1 + Math.random() * .8),
                    radius: Math.random() * 50, maxRadius: 80 + Math.random() * 160,
                    speed: .4 + Math.random() * .35, baseAlpha: .02 + Math.random() * .02,
                });
            }
            // Data lines — scattered across entire background
            for (let i = 0; i < 20; i++) {
                _sbDataLines.push({
                    x: Math.random() * _cw, y: Math.random() * _ch,
                    w: 8 + Math.random() * 28,
                    speed: .08 + Math.random() * .12,
                    opacity: .015 + Math.random() * .03,
                    drift: (Math.random() - .5) * .25,
                });
            }
        }

        function _sbDrawExtras(t) {
            // Scan rings — illuminate nodes they pass over
            for (const ring of _sbScanRings) {
                ring.radius += ring.speed;
                const progress = ring.radius / ring.maxRadius;
                if (progress >= 1) {
                    ring.radius = 0;
                    ring.x = _cw * (.1 + Math.random() * .8);
                    ring.y = _ch * (.05 + Math.random() * .9);
                    ring.maxRadius = 80 + Math.random() * 160;
                    ring.speed = .4 + Math.random() * .35;
                    continue;
                }
                const alpha = ring.baseAlpha * (1 - progress);
                ctx.strokeStyle = `rgba(79,195,247,${alpha})`;
                ctx.lineWidth = .5;
                ctx.beginPath(); ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2); ctx.stroke();
                // Illuminate nodes caught in scan wave
                for (const node of _sbNodes) {
                    const dx = node.x - ring.x, dy = node.y - ring.y;
                    const nd = Math.sqrt(dx * dx + dy * dy);
                    if (Math.abs(nd - ring.radius) < 10) {
                        node.activity = Math.max(node.activity, alpha * 5);
                    }
                }
            }

            // Threat flashes — rare red/orange Crime Coefficient spikes
            if (t > _sbNextThreat && _sbNodes.length > 0) {
                const idx = Math.floor(Math.random() * _sbNodes.length);
                _sbNodes[idx].threat = 1;
                _sbNextThreat = t + 8 + Math.random() * 14;
            }

            // Holographic data lines
            ctx.lineWidth = .8;
            for (const dl of _sbDataLines) {
                dl.y -= dl.speed;
                dl.x += dl.drift;
                if (dl.y < -10) { dl.y = _ch + 10; dl.x = Math.random() * _cw; }
                ctx.strokeStyle = `rgba(79,195,247,${dl.opacity})`;
                ctx.beginPath(); ctx.moveTo(dl.x, dl.y); ctx.lineTo(dl.x + dl.w, dl.y); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(dl.x + dl.w, dl.y - 1.2); ctx.lineTo(dl.x + dl.w, dl.y + 1.2); ctx.stroke();
            }
        }

        function _sibylDrawBrain(t) {
            _sbDrawExtras(t);
            _sbDrawOutline(t);
            _sbDrawNetwork(t);
        }

        // Init deferred — _cw/_ch not yet available. Called after resize().
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
            const sc = 0.75; // tree scale factor
            const cx = w * 0.5, baseY = h * 0.55;
            const trunkTop = baseY - (h * 0.15) * sc;

            // Thin trunk with slight lean
            const lean = w * 0.004 * sc;
            br.push({ x0:cx-w*0.003*sc, y0:baseY, x1:cx+lean, y1:trunkTop,
                cpx:cx-w*0.001*sc, cpy:baseY-(baseY-trunkTop)*0.53, w0:w*0.005*sc, d:0 });
            br.push({ x0:cx+w*0.003*sc, y0:baseY, x1:cx+lean, y1:trunkTop,
                cpx:cx+w*0.002*sc, cpy:baseY-(baseY-trunkTop)*0.53, w0:w*0.005*sc, d:0 });

            // Recursive branch generator — delicate and airy
            function addBranch(sx, sy, angle, length, width, depth) {
                if (depth > 3 || length < h * 0.012) return;
                const curve = _tRandR(-0.3, 0.3);
                const ex = sx + Math.cos(angle) * length;
                const ey = sy + Math.sin(angle) * length;
                const mid = 0.5 + _tRandR(-0.15, 0.15);
                const cpx = sx + Math.cos(angle + curve) * length * mid;
                const cpy = sy + Math.sin(angle + curve) * length * mid;
                br.push({ x0:sx, y0:sy, cpx, cpy, x1:ex, y1:ey, w0:width, d:depth });

                if (depth >= 1) {
                    tips.push({ x: ex, y: ey });
                    bl.push({ x:ex, y:ey, r:_tRandR(6, 14 - depth*2), phase:_tRand()*Math.PI*2 });
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
                const len = _tRandR(h*0.06, h*0.11) * sc;
                const wid = w * _tRandR(0.0015, 0.003) * sc;
                addBranch(cx + lean, trunkTop, mainAngles[i], len, wid, 1);
            }

            // One small lower branch
            addBranch(cx, baseY - (baseY - trunkTop) * 0.53, -Math.PI*0.6 + _tRandR(-0.1,0.1), h*0.04*sc, w*0.001*sc, 2);

            // Generate blossom dots within clusters
            for (const c of bl) {
                c.dots = [];
                const count = Math.floor(_tRandR(14, 28));
                for (let i = 0; i < count; i++) {
                    const ba = _tRand() * Math.PI * 2, bd = _tRand() * c.r * 1.2;
                    c.dots.push({ ox: Math.cos(ba)*bd, oy: Math.sin(ba)*bd,
                        r: _tRandR(0.6, 2.2), bright: _tRand(), ph: _tRand()*Math.PI*2 });
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
        _sakuraTree.offsetX = 0;
        _sakuraTree.offsetY = 0;
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
            // Wispy, thin bark — subtle opacity
            const alpha = b.d === 0 ? 0.22 : b.d === 1 ? 0.16 : b.d === 2 ? 0.11 : 0.06;
            ctx.strokeStyle = `rgba(60,30,45,${alpha})`;
            ctx.lineWidth = b.w0;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();
        }

        // Draw blossom clusters — glowing, full canopy
        for (const cl of _sakuraTree.blossoms) {
            const clSway = windBase * 1.5;
            const clX = cl.x + clSway, clY = cl.y;

            // Cluster glow — visible and warm
            const cg = ctx.createRadialGradient(clX, clY, 0, clX, clY, cl.r * 2.5);
            cg.addColorStop(0, `rgba(255,190,215,${0.12 * pulse})`);
            cg.addColorStop(0.5, `rgba(245,170,200,${0.05 * pulse})`);
            cg.addColorStop(1, 'transparent');
            ctx.fillStyle = cg;
            ctx.fillRect(clX - cl.r*2.5, clY - cl.r*2.5, cl.r*5, cl.r*5);

            for (const dot of cl.dots) {
                const dp = 0.7 + 0.3 * Math.sin(t * 1.5 + dot.ph);
                const x = clX + dot.ox, y = clY + dot.oy;
                const r = dot.r * (0.85 + 0.15 * dp);
                const c = dot.bright > 0.6 ? [255,225,240] : dot.bright > 0.3 ? [255,200,220] : [245,175,200];
                // Small halo per dot
                const dg = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
                dg.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${0.15 * dp})`);
                dg.addColorStop(1, 'transparent');
                ctx.fillStyle = dg; ctx.fillRect(x - r*2, y - r*2, r*4, r*4);
                ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.3 + 0.3*dp})`;
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

    const _dpr = window.devicePixelRatio || 1;
    let _cw = 0, _ch = 0; // logical CSS dimensions
    function resize() {
        _cw = canvas.parentElement.offsetWidth;
        _ch = canvas.parentElement.offsetHeight;
        canvas.width = _cw * _dpr;
        canvas.height = _ch * _dpr;
        canvas.style.width = _cw + 'px';
        canvas.style.height = _ch + 'px';
        ctx.setTransform(_dpr, 0, 0, _dpr, 0, 0);
    }
    resize();

    // Sibyl brain init (needs _cw/_ch from resize)
    if (_isSibyl && typeof _sbGetTransform === 'function') {
        _sbGetTransform();
        _sbCacheScreenPts();
        _sbBuildNetwork();
        _sbInitExtras();
    }

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
        const petal = _isSakura && Math.random() < 0.55;
        // Petals start scattered from tree area downward; stars anywhere
        let initX = Math.random() * _cw;
        let initY = Math.random() * _ch;
        if (petal) {
            if (_sakuraTree.tips.length > 0 && Math.random() < 0.8) {
                const tip = _sakuraTree.tips[Math.floor(Math.random() * _sakuraTree.tips.length)];
                const sx = _cw / 800, sy = _ch / 600;
                const initOffY = isMobile ? _ch * 0.28 : -_ch * 0.12;
                initX = tip.x * sx + (Math.random() - 0.5) * 20;
                initY = tip.y * sy + initOffY + (Math.random() - 0.3) * _ch * 0.15;
            }
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
            x: Math.random() * _cw * 0.7,
            y: Math.random() * _ch * 0.35,
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
        ctx.clearRect(0, 0, _cw, _ch);
        const t = Date.now() * 0.001;
        const now = Date.now();
        const _elScale = isMobile ? Math.min(_cw / 400, 0.7) : 1;

        // Solar system (cosmos auth theme - tilted perspective, drawn first)
        if (_isCosmos) {
            const scx = _cw * 0.5, scy = isMobile ? _ch * 0.62 : _ch * 0.28;
            if (isMobile) { ctx.save(); ctx.translate(scx, scy); ctx.scale(_elScale, _elScale); ctx.translate(-scx, -scy); }
            // Tilted orbit lines
            for (const p of _ssPlanets) _ssTiltedOrbit(scx, scy, p.dist, 0.15);
            _ssTiltedOrbit(scx, scy, 232, 0.07); _ssTiltedOrbit(scx, scy, 270, 0.07);
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
            if (isMobile) { ctx.restore(); }
        }

        // Black hole (nebula auth theme — P1 classic accretion disk)
        if (_isNebula) {
            const bhx = _cw * 0.5, bhy = isMobile ? _ch * 0.62 : _ch * 0.30;
            if (isMobile) { ctx.save(); ctx.translate(bhx, bhy); ctx.scale(_elScale, _elScale); ctx.translate(-bhx, -bhy); }
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
            if (isMobile) { ctx.restore(); }
        }

        // Binary star system (aurora auth theme)
        if (_isAurora) {
            const _aurCx = _cw * 0.5, _aurCy = isMobile ? _ch * 0.62 : _ch * 0.28;
            if (isMobile) { ctx.save(); ctx.translate(_aurCx, _aurCy); ctx.scale(_elScale, _elScale); ctx.translate(-_aurCx, -_aurCy); }
            _binDrawSystem(_aurCx, _aurCy, t);
            if (isMobile) { ctx.restore(); }
        }

        // Theme center elements — on mobile, position at middle-bottom and scale to fit
        const _elCx = _cw * 0.5;
        const _elCy = isMobile ? _ch * 0.62 : _ch * 0.30;

        if (_isNoir) {
            ctx.save(); ctx.translate(_elCx, _elCy); ctx.scale(_elScale, _elScale); ctx.translate(-_elCx, -_elCy);
            _noirDrawPendulum(_elCx, _elCy, t);
            ctx.restore();
        }
        if (_isRose) {
            ctx.save(); ctx.translate(_elCx, _elCy); ctx.scale(_elScale, _elScale); ctx.translate(-_elCx, -_elCy);
            _roseDrawBloom(_elCx, _elCy, t);
            ctx.restore();
        }
        if (_isCoffee) {
            ctx.save(); ctx.translate(_elCx, _elCy); ctx.scale(_elScale, _elScale); ctx.translate(-_elCx, -_elCy);
            _coffeeDrawCup(_elCx, _elCy, t);
            ctx.restore();
        }
        if (_isMidnight) {
            ctx.save(); ctx.translate(_elCx, _elCy); ctx.scale(_elScale, _elScale); ctx.translate(-_elCx, -_elCy);
            _midnightDrawMoon(_elCx, isMobile ? _elCy : _ch * 0.28, t, _cw);
            ctx.restore();
        }
        if (_isSibyl) {
            _sibylDrawBrain(t);
        }
        if (_isSakura) {
            if (isMobile) {
                const _skOffY = _ch * 0.28;
                _sakuraTree.offsetY = _skOffY;
                _sakuraTree.offsetX = 0;
                const _skPivY = _ch * 0.90;
                ctx.save(); ctx.translate(_elCx, _skPivY); ctx.scale(_elScale, _elScale); ctx.translate(-_elCx, -_skPivY);
                ctx.translate(0, _skOffY);
                _sakuraDrawTree(_cw, _ch, t);
                ctx.restore();
            } else {
                // Shift tree slightly above the logo center
                const _skOffY = -_ch * 0.12;
                _sakuraTree.offsetY = _skOffY;
                _sakuraTree.offsetX = 0;
                ctx.save(); ctx.translate(0, _skOffY);
                _sakuraDrawTree(_cw, _ch, t);
                ctx.restore();
            }
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
                const windGust = Math.sin(t * 0.3 + s.phase) * 0.2 + Math.sin(t * 0.8 + s.phase * 2) * 0.12;
                const driftX = (-0.55 - windGust) * spiralPhase + Math.sin(t * s.petalSway + s.phase) * 0.1;
                const driftY = s.petalFall * 1.2 * spiralPhase;
                s.baseX += driftX + (spiralX - (s._prevSpiralX || 0));
                s.baseY += driftY + (spiralY - (s._prevSpiralY || 0));
                s._prevSpiralX = spiralX;
                s._prevSpiralY = spiralY;
                if (s.baseY > _ch + 15 || s.baseX < -20) {
                    if (_sakuraTree.tips.length > 0 && Math.random() < 0.55) {
                        // From tree branch tips (apply visual offset)
                        const tip = _sakuraTree.tips[Math.floor(Math.random() * _sakuraTree.tips.length)];
                        const offY = _sakuraTree.offsetY || 0;
                        s.baseX = tip.x + (Math.random() - 0.5) * 10;
                        s.baseY = tip.y + offY + (Math.random() - 0.5) * 6;
                    } else {
                        // From random sky positions — petals drifting in from above
                        s.baseY = -10 - Math.random() * 30;
                        s.baseX = Math.random() * _cw;
                    }
                    s.y = s.baseY; s.x = s.baseX;
                    s.petalAge = Math.random() < 0.45 ? 0 : 80 + Math.random() * 40;
                    s._prevSpiralX = 0; s._prevSpiralY = 0;
                }
            } else {
                s.baseY -= DRIFT_SPEED;
                if (s.baseY < -10) {
                    s.baseY = _ch + 10;
                    s.baseX = Math.random() * _cw;
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

            if (s.y < -15 || s.y > _ch + 15) continue;

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
            if (sh.life <= 0 || sh.x > _cw + 60 || sh.y > _ch + 60) {
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
            stars[i].baseX = Math.random() * _cw;
            stars[i].baseY = Math.random() * _ch;
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
