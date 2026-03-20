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
    const _sbSVG_PATH = "M3050 5129 c-25 -5 -148 -14 -275 -19 -357 -14 -566 -75 -677 -199 -32 -35 -53 -49 -80 -54 -85 -13 -147 -57 -309 -218 -171 -170 -198 -210 -207 -301 -5 -57 -6 -59 -47 -77 -80 -35 -245 -309 -245 -407 0 -29 -10 -64 -29 -102 -105 -208 -84 -490 47 -633 50 -55 51 -57 62 -145 64 -517 347 -834 744 -834 56 0 56 0 58 -247 3 -247 3 -247 52 -261 39 -11 67 -31 140 -102 114 -109 116 -110 225 -110 91 0 91 0 120 59 92 181 241 363 411 501 124 101 193 175 227 244 27 56 27 56 243 56 211 0 218 1 266 25 27 14 57 25 66 25 50 0 272 120 349 188 75 67 99 125 99 244 0 98 0 98 243 98 411 0 575 40 729 177 119 106 189 215 209 328 6 33 20 80 32 105 19 38 23 66 25 178 4 163 -9 212 -72 270 -43 40 -43 40 -50 128 -13 160 -84 265 -223 331 -60 29 -77 43 -104 84 -64 101 -209 200 -361 249 -55 18 -82 34 -143 90 -109 99 -211 140 -349 140 -28 0 -43 6 -56 23 -22 28 -116 80 -197 109 -46 16 -96 22 -243 28 -102 5 -203 13 -225 20 -52 15 -397 22 -455 9z";
    let _sbBp = [], _sbBs, _sbBx, _sbBy;
    let _sbNodes = [], _sbLinks = [], _sbPulses = [];
    let _sbNextMajorPulse = 1;
    let _sbNextMinorPulse = 0.3;
    let _sbLastMajorHub = -1;
    let _sbSweepPos = 0;

    if (_isSibyl) {
        // Parse SVG path to extract brain outline points
        (function _sbParsePath() {
            const ns = "http://www.w3.org/2000/svg";
            const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 660 649");
            const g = document.createElementNS(ns, "g"); g.setAttribute("transform", "translate(0,649) scale(0.1,-0.1)");
            const p = document.createElementNS(ns, "path"); p.setAttribute("d", _sbSVG_PATH);
            g.appendChild(p); svg.appendChild(g); document.body.appendChild(svg);
            svg.style.cssText = 'position:absolute;visibility:hidden;width:660px;height:649px';
            _sbBp = []; const len = p.getTotalLength();
            for (let i = 0; i < len; i += 2) { const pt = p.getPointAtLength(i); _sbBp.push({ x: pt.x * .1, y: 649 + pt.y * -.1 }); }
            document.body.removeChild(svg);
        })();

        function _sbGetTransform() {
            const th = _ch * .28;
            _sbBs = th / 649;
            _sbBx = _cw / 2 - 660 * _sbBs / 2;
            _sbBy = _ch * .28 - th / 2;
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
                        activity: 0, connections: 0,
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

        function _sbDrawOutline(t) {
            _sbSweepPos = (_sbSweepPos + .0008) % 1;
            const sweepIdx = Math.floor(_sbSweepPos * _sbBp.length);
            const sweepW = _sbBp.length * .12;
            const segLen = 5, gapLen = 3, totalSeg = segLen + gapLen;
            for (let i = 0; i < _sbBp.length - 1; i++) {
                if (i % totalSeg >= segLen) continue;
                const pt = _sbSpt(_sbBp[i].x, _sbBp[i].y);
                const pt2 = _sbSpt(_sbBp[(i + 1) % _sbBp.length].x, _sbBp[(i + 1) % _sbBp.length].y);
                let dist = i - sweepIdx;
                if (dist < 0) dist += _sbBp.length;
                if (dist > _sbBp.length / 2) dist = _sbBp.length - dist;
                const sweepBoost = dist < sweepW ? (.2 * (1 - dist / sweepW)) : 0;
                const baseAlpha = .05 + sweepBoost;
                ctx.strokeStyle = `rgba(79,195,247,${baseAlpha})`;
                ctx.lineWidth = sweepBoost > .06 ? 1.4 : .7;
                ctx.beginPath(); ctx.moveTo(pt.x, pt.y); ctx.lineTo(pt2.x, pt2.y); ctx.stroke();
                if (sweepBoost > .08) {
                    ctx.strokeStyle = `rgba(160,230,255,${sweepBoost * .5})`;
                    ctx.lineWidth = .3;
                    ctx.beginPath(); ctx.moveTo(pt.x, pt.y); ctx.lineTo(pt2.x, pt2.y); ctx.stroke();
                }
            }
            const hp = _sbSpt(_sbBp[sweepIdx].x, _sbBp[sweepIdx].y);
            const hg = ctx.createRadialGradient(hp.x, hp.y, 0, hp.x, hp.y, 14);
            hg.addColorStop(0, 'rgba(79,195,247,.18)'); hg.addColorStop(.4, 'rgba(79,195,247,.05)'); hg.addColorStop(1, 'rgba(79,195,247,0)');
            ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(hp.x, hp.y, 14, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = 'rgba(200,240,255,.7)'; ctx.beginPath(); ctx.arc(hp.x, hp.y, 1.5, 0, Math.PI * 2); ctx.fill();
            const s2 = Math.floor(((_sbSweepPos + .5) % 1) * _sbBp.length);
            const h2 = _sbSpt(_sbBp[s2].x, _sbBp[s2].y);
            const hg2 = ctx.createRadialGradient(h2.x, h2.y, 0, h2.x, h2.y, 8);
            hg2.addColorStop(0, 'rgba(79,195,247,.08)'); hg2.addColorStop(1, 'rgba(79,195,247,0)');
            ctx.fillStyle = hg2; ctx.beginPath(); ctx.arc(h2.x, h2.y, 8, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath();
            let p0 = _sbSpt(_sbBp[0].x, _sbBp[0].y); ctx.moveTo(p0.x, p0.y);
            for (let i = 1; i < _sbBp.length; i++) { const pt = _sbSpt(_sbBp[i].x, _sbBp[i].y); ctx.lineTo(pt.x, pt.y); }
            ctx.closePath(); ctx.fillStyle = 'rgba(79,195,247,.006)'; ctx.fill();
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
                const r = n.r + n.activity * .6;
                if (n.pulseRing > .02) {
                    const ringR = r + 12 * (1 - n.pulseRing);
                    ctx.beginPath(); ctx.arc(n.x, n.y, ringR, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(79,195,247,${n.pulseRing * .2})`;
                    ctx.lineWidth = .6; ctx.stroke();
                }
                if (n.activity > .03) {
                    const hg = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * (n.isHub ? 6 : 4));
                    hg.addColorStop(0, `rgba(79,195,247,${n.activity * (n.isHub ? .08 : .05)})`);
                    hg.addColorStop(1, 'rgba(79,195,247,0)');
                    ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(n.x, n.y, r * (n.isHub ? 6 : 4), 0, Math.PI * 2); ctx.fill();
                }
                if (n.isHub) {
                    ctx.beginPath(); ctx.arc(n.x, n.y, r + 2, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(79,195,247,${.05 + n.activity * .1})`;
                    ctx.lineWidth = .35; ctx.stroke();
                    ctx.beginPath(); ctx.arc(n.x, n.y, r + .8, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(120,210,255,${.06 + n.activity * .12})`;
                    ctx.lineWidth = .25; ctx.stroke();
                }
                const somaA = n.isHub ? (.08 + n.activity * .4) : (.05 + n.activity * .3);
                ctx.fillStyle = `rgba(79,195,247,${somaA})`;
                ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
                if (n.activity > .1) {
                    ctx.fillStyle = `rgba(200,240,255,${n.activity * .4})`;
                    ctx.beginPath(); ctx.arc(n.x, n.y, r * .4, 0, Math.PI * 2); ctx.fill();
                }
            }
        }

        function _sibylDrawBrain(t) {
            _sbDrawOutline(t);
            _sbDrawNetwork(t);
        }

        _sbGetTransform();
        _sbBuildNetwork();
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
            ctx.save(); ctx.translate(_elCx, _elCy); ctx.scale(_elScale, _elScale); ctx.translate(-_elCx, -_elCy);
            _sibylDrawBrain(t);
            ctx.restore();
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
