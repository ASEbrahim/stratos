// ═══════════════════════════════════════════════════════════
// STRAT_OS MARKET — TradingView Lightweight Charts
// Yahoo Finance timeframes: 1D, 5D, 1M, 6M, 1Y, 5Y
// ═══════════════════════════════════════════════════════════

const ASSET_NAMES = {
    'GC=F':'Gold','SI=F':'Silver','HG=F':'Copper','PL=F':'Platinum',
    'CL=F':'Crude Oil','NG=F':'Natural Gas','BZ=F':'Brent Crude',
    'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','XRP-USD':'XRP',
    'SOL-USD':'Solana','ADA-USD':'Cardano','DOGE-USD':'Dogecoin',
    'NVDA':'NVIDIA','AMD':'AMD','AAPL':'Apple','MSFT':'Microsoft',
    'GOOGL':'Alphabet','GOOG':'Alphabet','META':'Meta','AMZN':'Amazon',
    'TSLA':'Tesla','INTC':'Intel','TSM':'TSMC',
    'XOM':'ExxonMobil','CVX':'Chevron','SLB':'SLB','HAL':'Halliburton',
    'BKR':'Baker Hughes','COP':'ConocoPhillips','OXY':'Occidental',
    'JPM':'JPMorgan','BAC':'Bank of America','GS':'Goldman Sachs',
};
function getAssetName(sym, fallback) { return ASSET_NAMES[sym] || fallback || sym; }

// ── Timeframe → data key mapping (Yahoo Finance intervals) ──
// Frontend button  | Backend key | yfinance interval | deep period
// 1D               | "1m"        | 1m                | 7d
// 5D               | "5m"        | 5m                | 60d
// 1M               | "1d_1mo"    | 1d                | 1y
// 1Y               | "1d_1y"     | 1d                | 5y
// 5Y               | "1wk"       | 1wk               | 10y

const TF_LABELS = { '1m':'1-min candles (today)', '5m':'5-min candles (15 days)', '1d_1mo':'Daily candles (1 year)', '1d_1y':'Daily candles (5 years)', '1wk':'Weekly candles' };
// Each timeframe fetches deeper history than the label suggests — scroll left to explore
// Fallback order if a timeframe has no data
const TF_FALLBACKS = ['1m','5m','1d_1mo','1d_1y','1wk'];

// ── State ──
let _tvChart = null;
let _tvSeries = null;
let _tvPrevLine = null;
let _tvTrendLines = [];
let _chartType = 'line';
let _crosshairOn = true;
let _autoTrendOn = false;
let _drawMode = false;
let _drawColor = '#34d399';
let _drawLines = [];
let _drawPending = null;
let _drawCtx = null;

// ═══════════════════════════════════════════════════════════
// CHART INIT
// ═══════════════════════════════════════════════════════════

let _chartRO = null;  // Track ResizeObserver for cleanup

function initChart(forceRecreate) {
    // VERIFIED: LightweightCharts is event-driven, not timer-based. A hidden chart
    // (mobile sidebar) has near-zero idle CPU — ResizeObserver won't fire on hidden
    // elements, crosshair/click subscriptions only trigger on user interaction.
    const el = document.getElementById('tv-chart-container');
    if (!el) return;

    // If chart already exists and no force-recreate, skip — updateChart handles data
    if (_tvChart && !forceRecreate) {
        return;
    }

    // Cleanup previous chart properly
    if (_tvChart) {
        try { _tvChart.remove(); } catch(e) {}
        _tvChart = null;
    }
    if (_chartRO) {
        try { _chartRO.disconnect(); } catch(e) {}
        _chartRO = null;
    }

    el.innerHTML = '';

    const cs = getComputedStyle(document.documentElement);
    const border = cs.getPropertyValue('--border-strong').trim() || 'rgba(51,65,85,0.5)';
    const txt = cs.getPropertyValue('--text-muted').trim() || '#64748b';

    _tvChart = LightweightCharts.createChart(el, {
        width: el.clientWidth, height: el.clientHeight,
        layout: { background:{type:'solid',color:'transparent'}, textColor:txt, fontSize:10, fontFamily:"'ui-monospace','SFMono-Regular',monospace" },
        grid: { vertLines:{color:'rgba(148,163,184,0.06)'}, horzLines:{color:'rgba(148,163,184,0.06)'} },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine:{color:'rgba(148,163,184,0.3)',width:1,style:LightweightCharts.LineStyle.Dashed,labelBackgroundColor:'#1e293b'},
            horzLine:{color:'rgba(148,163,184,0.3)',width:1,style:LightweightCharts.LineStyle.Dashed,labelBackgroundColor:'#1e293b'},
        },
        rightPriceScale: { borderColor:border, scaleMargins:{top:0.1,bottom:0.1} },
        timeScale: { borderColor:border, timeVisible:true, secondsVisible:false, rightOffset:5, minBarSpacing:3 },
        handleScroll:{mouseWheel:false,pressedMouseMove:false,horzTouchDrag:true,vertTouchDrag:false},
        handleScale:{axisPressedMouseMove:false,mouseWheel:false,pinch:true},
    });

    _createSeries();

    // Responsive resize (debounced to one frame — avoids thrashing during drag-resize)
    let _roTimer = null;
    _chartRO = new ResizeObserver(() => {
        if (_roTimer) cancelAnimationFrame(_roTimer);
        _roTimer = requestAnimationFrame(() => {
            if (_tvChart) _tvChart.applyOptions({width:el.clientWidth,height:el.clientHeight});
            _redrawOverlay();
        });
    });
    _chartRO.observe(el);

    // Add zoom toolbar
    _addMainChartToolbar(el);

    // Init drawing canvas
    const canvas = document.getElementById('draw-overlay');
    if (canvas) { canvas.width=el.clientWidth; canvas.height=el.clientHeight; _drawCtx=canvas.getContext('2d'); }

    // Drawing click
    _tvChart.subscribeClick(param => {
        if (!_drawMode || !param.time || !param.point) return;
        const price = _tvSeries.coordinateToPrice(param.point.y);
        if (price == null) return;
        if (!_drawPending) {
            _drawPending = {time:param.time, price};
            _setDrawStatus('Click second point...');
        } else {
            _drawLines.push({p1:_drawPending, p2:{time:param.time,price}, color:_drawColor});
            _drawPending = null;
            _setDrawStatus('Line drawn — click to start another');
            _redrawOverlay();
        }
    });

    // Live preview
    _tvChart.subscribeCrosshairMove(param => {
        if (!_drawMode || !_drawPending || !param.point) return;
        _redrawOverlay();
        const c = document.getElementById('draw-overlay');
        if (!c || !_drawCtx) return;
        const x1 = _tvChart.timeScale().timeToCoordinate(_drawPending.time);
        const y1 = _tvSeries.priceToCoordinate(_drawPending.price);
        if (x1==null||y1==null) return;
        _drawCtx.beginPath(); _drawCtx.moveTo(x1,y1); _drawCtx.lineTo(param.point.x,param.point.y);
        _drawCtx.strokeStyle=_drawColor; _drawCtx.lineWidth=1.5; _drawCtx.setLineDash([4,3]); _drawCtx.stroke(); _drawCtx.setLineDash([]);
    });

    // Redraw drawings on scroll/zoom
    _tvChart.timeScale().subscribeVisibleTimeRangeChange(() => _redrawOverlay());

    // OHLC legend on crosshair move
    _tvChart.subscribeCrosshairMove(param => {
        if (_drawMode && _drawPending) return; // Don't interfere with drawing preview
        const legendO = document.getElementById('legend-o');
        const legendH = document.getElementById('legend-h');
        const legendL = document.getElementById('legend-l');
        const legendC = document.getElementById('legend-c');
        const legendV = document.getElementById('legend-v');
        if (!legendO) return;

        if (!param.time || !param.seriesData || !param.seriesData.has(_tvSeries)) {
            legendO.textContent = legendH.textContent = legendL.textContent = legendC.textContent = legendV.textContent = '';
            return;
        }
        const d = param.seriesData.get(_tvSeries);
        if (!d) return;

        const fp = v => v != null ? v.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}) : '--';

        if (d.open != null) {
            // Candlestick data
            legendO.textContent = 'O ' + fp(d.open);
            legendH.textContent = 'H ' + fp(d.high);
            legendL.textContent = 'L ' + fp(d.low);
            legendC.textContent = 'C ' + fp(d.close);
            legendC.className = 'text-[10px] font-mono ' + (d.close >= d.open ? 'text-emerald-400' : 'text-red-400');
        } else if (d.value != null) {
            // Line data
            legendO.textContent = '';
            legendH.textContent = '';
            legendL.textContent = '';
            legendC.textContent = '$' + fp(d.value);
            legendC.className = 'text-[10px] font-mono text-slate-300';
        }
        legendV.textContent = '';
    });

    // Remove TradingView logo/watermark
    _removeTVLogo(el);
}

function _removeTVLogo(container) {
    // Hide only the TradingView attribution links, NOT their parent table/tr
    // (the chart canvas lives inside the same table structure)
    setTimeout(() => {
        container.querySelectorAll('a[href*="tradingview"], a[href*="lightweight-charts"]').forEach(a => {
            a.style.display = 'none';
        });
    }, 200);
}

function _createSeries() {
    if (!_tvChart) return;
    if (_tvSeries) try{_tvChart.removeSeries(_tvSeries);}catch(e){}
    if (_tvPrevLine) try{_tvChart.removeSeries(_tvPrevLine);}catch(e){}
    _clearTrendSeries();

    const accent = getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim() ||
                   getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#10b981';

    if (_chartType === 'candle') {
        _tvSeries = _tvChart.addCandlestickSeries({
            upColor:'#22c55e', downColor:'#ef4444', borderUpColor:'#22c55e', borderDownColor:'#ef4444',
            wickUpColor:'#22c55e80', wickDownColor:'#ef444480',
        });
    } else {
        _tvSeries = _tvChart.addLineSeries({
            color:accent, lineWidth:2, crosshairMarkerVisible:true, crosshairMarkerRadius:4,
            lastValueVisible:true, priceLineVisible:true,
        });
    }
    _tvPrevLine = _tvChart.addLineSeries({
        color:'rgba(148,163,184,0.35)', lineWidth:1, lineStyle:LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible:false, lastValueVisible:false, priceLineVisible:false,
    });
}

// ═══════════════════════════════════════════════════════════
// TOOLBAR ACTIONS
// ═══════════════════════════════════════════════════════════

function setChartType(type) {
    _chartType = type;
    document.getElementById('chart-type-line')?.classList.toggle('active', type==='line');
    document.getElementById('chart-type-candle')?.classList.toggle('active', type==='candle');
    _createSeries();
    updateChart(currentSymbol);
}

function toggleCrosshair() {
    _crosshairOn = !_crosshairOn;
    document.getElementById('crosshair-btn')?.classList.toggle('active', _crosshairOn);
    if (_tvChart) _tvChart.applyOptions({ crosshair:{ mode: _crosshairOn ? LightweightCharts.CrosshairMode.Normal : LightweightCharts.CrosshairMode.Hidden }});
}

function toggleAutoTrend() {
    _autoTrendOn = !_autoTrendOn;
    document.getElementById('auto-trend-btn')?.classList.toggle('active', _autoTrendOn);
    _autoTrendOn ? _computeAutoTrend() : _clearTrendSeries();
}

function toggleDrawMode() {
    _drawMode = !_drawMode; _drawPending = null;
    document.getElementById('draw-btn')?.classList.toggle('active', _drawMode);
    document.getElementById('draw-colors')?.classList.toggle('hidden', !_drawMode);
    _setDrawStatus(_drawMode ? 'Click first point on chart...' : '');
    if (_tvChart) {
        _tvChart.applyOptions({
            handleScroll:{mouseWheel:!_drawMode,pressedMouseMove:!_drawMode,horzTouchDrag:!_drawMode},
            handleScale:{mouseWheel:!_drawMode,axisPressedMouseMove:!_drawMode},
        });
    }
}

function setDrawColor(c) {
    _drawColor = c;
    document.querySelectorAll('#draw-colors button[onclick^="setDrawColor"]').forEach(btn => {
        const m = btn.getAttribute('onclick').match(/'(.+?)'/);
        btn.style.borderColor = (m && m[1]===c) ? 'rgba(255,255,255,0.7)' : 'transparent';
    });
}

function clearDrawings() {
    _drawLines=[]; _drawPending=null; _redrawOverlay();
    _setDrawStatus(_drawMode ? 'Cleared — click first point...' : '');
}

function _setDrawStatus(t) {
    const el = document.getElementById('draw-status');
    if (el) { el.textContent=t; el.classList.toggle('hidden',!t); }
}

// ═══════════════════════════════════════════════════════════
// DRAWING OVERLAY
// ═══════════════════════════════════════════════════════════

function _redrawOverlay() {
    const canvas = document.getElementById('draw-overlay');
    const cont = document.getElementById('tv-chart-container');
    if (!canvas||!cont||!_tvChart||!_tvSeries) return;
    canvas.width=cont.clientWidth; canvas.height=cont.clientHeight;
    const ctx = canvas.getContext('2d'); _drawCtx=ctx;
    ctx.clearRect(0,0,canvas.width,canvas.height);

    for (const ln of _drawLines) {
        const x1=_tvChart.timeScale().timeToCoordinate(ln.p1.time);
        const y1=_tvSeries.priceToCoordinate(ln.p1.price);
        const x2=_tvChart.timeScale().timeToCoordinate(ln.p2.time);
        const y2=_tvSeries.priceToCoordinate(ln.p2.price);
        if (x1==null||y1==null||x2==null||y2==null) continue;
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle=ln.color; ctx.lineWidth=2; ctx.stroke();
        [{x:x1,y:y1},{x:x2,y:y2}].forEach(pt => {
            ctx.beginPath(); ctx.arc(pt.x,pt.y,3.5,0,Math.PI*2); ctx.fillStyle=ln.color; ctx.fill();
        });
    }

    _renderFocusDrawings(ctx, canvas.width, canvas.height, _tvChart, _tvSeries, currentSymbol);
}

// Render focus mode drawings (from localStorage) on any chart overlay.
// Accepts chart/series/symbol so markets-panel.js can reuse this.
function _renderFocusDrawings(ctx, w, h, chart, series, symbol) {
    if (!chart || !series || !symbol) return;
    var raw;
    try { raw = localStorage.getItem('stratos_drawings_' + symbol); } catch(e) { return; }
    if (!raw) return;
    var drawings;
    try { drawings = JSON.parse(raw); } catch(e) { return; }
    if (!drawings || !drawings.length) return;

    var ts = chart.timeScale();
    var chartW = ts.width();
    if (chartW <= 0 || chartW > w) chartW = w;

    // Clip to plot area (excludes price scale)
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, chartW, h); ctx.clip();

    var fmtP = typeof _fsFmtPrice === 'function' ? _fsFmtPrice : function(v) {
        if (v == null) return '--';
        if (Math.abs(v) >= 100) return v.toFixed(2);
        if (Math.abs(v) >= 1) return v.toFixed(4);
        return v.toFixed(6);
    };

    drawings.forEach(function(d) {
        if (d.type === 'hline') {
            var y = series.priceToCoordinate(d.price);
            if (y == null) return;
            ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1;
            ctx.setLineDash([5, 3]);
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#F0B90B'; ctx.font = '10px monospace';
            ctx.fillText(fmtP(d.price), 4, y - 4);

        } else if (d.type === 'hray') {
            var y = series.priceToCoordinate(d.price);
            var sx = ts.logicalToCoordinate(d.startIdx);
            if (y == null || sx == null) return;
            ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1;
            ctx.setLineDash([5, 3]);
            ctx.beginPath(); ctx.moveTo(sx, y); ctx.lineTo(chartW, y); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#F0B90B'; ctx.font = '10px monospace';
            ctx.fillText(fmtP(d.price), sx + 4, y - 4);

        } else if (d.type === 'vline') {
            var x = ts.logicalToCoordinate(d.timeIdx);
            if (x == null) return;
            ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1;
            ctx.setLineDash([5, 3]);
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            ctx.setLineDash([]);

        } else if (d.type === 'text') {
            var x = ts.logicalToCoordinate(d.timeIdx);
            var y = series.priceToCoordinate(d.price);
            if (x == null || y == null) return;
            ctx.fillStyle = '#eaecef'; ctx.font = '12px sans-serif';
            ctx.fillText(d.text, x, y);

        } else if (d.type === 'trend') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            ctx.strokeStyle = '#2196F3'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();

        } else if (d.type === 'ray') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            ctx.strokeStyle = '#2196F3'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
            var dx = ex - sx, dy = ey - sy, len = Math.sqrt(dx*dx + dy*dy);
            if (len > 0) {
                var ext = Math.max(w, h) * 3 / len;
                ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + dx*ext, sy + dy*ext); ctx.stroke();
            }

        } else if (d.type === 'fib') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            var fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
            var fibColors = ['#787B86','#F6C343','#4CAF50','#2196F3','#9C27B0','#E91E63','#787B86'];
            var range = d.endPrice - d.startPrice;
            fibLevels.forEach(function(lvl, idx) {
                var price = d.startPrice + range * lvl;
                var y = series.priceToCoordinate(price);
                if (y == null) return;
                ctx.strokeStyle = fibColors[idx] || '#787B86'; ctx.lineWidth = 1;
                ctx.setLineDash(lvl === 0 || lvl === 1 ? [] : [4, 2]);
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = fibColors[idx] || '#787B86'; ctx.font = '9px monospace';
                ctx.fillText((lvl * 100).toFixed(1) + '% ' + fmtP(price), 4, y - 3);
            });
            var yTop = series.priceToCoordinate(Math.max(d.startPrice, d.endPrice));
            var yBot = series.priceToCoordinate(Math.min(d.startPrice, d.endPrice));
            if (yTop != null && yBot != null) {
                ctx.fillStyle = 'rgba(33,150,243,0.05)';
                ctx.fillRect(Math.min(sx, ex), yTop, Math.abs(ex - sx), yBot - yTop);
            }

        } else if (d.type === 'rect') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1; ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(240,185,11,0.08)';
            ctx.fillRect(sx, sy, ex - sx, ey - sy);
            ctx.strokeRect(sx, sy, ex - sx, ey - sy);

        } else if (d.type === 'channel') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            var offPriceY = series.priceToCoordinate(d.endPrice + d.offset);
            var offsetPx = (offPriceY != null) ? (offPriceY - ey) : 0;
            ctx.strokeStyle = '#2196F3'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(sx, sy + offsetPx); ctx.lineTo(ex, ey + offsetPx); ctx.stroke();
            ctx.fillStyle = 'rgba(33,150,243,0.06)';
            ctx.beginPath();
            ctx.moveTo(sx, sy); ctx.lineTo(ex, ey);
            ctx.lineTo(ex, ey + offsetPx); ctx.lineTo(sx, sy + offsetPx);
            ctx.closePath(); ctx.fill();

        } else if (d.type === 'longpos') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            var entryY = sy, tpY = ey;
            var slY = entryY + (entryY - tpY);
            var slPrice = d.startPrice - (d.endPrice - d.startPrice);
            var boxL = Math.min(sx, ex), boxR = Math.max(sx, ex);
            if (boxR - boxL < 60) { boxL = 0; boxR = chartW; }
            ctx.fillStyle = 'rgba(14,203,129,0.1)';
            ctx.fillRect(boxL, Math.min(entryY, tpY), boxR - boxL, Math.abs(tpY - entryY));
            ctx.fillStyle = 'rgba(246,70,93,0.1)';
            ctx.fillRect(boxL, entryY, boxR - boxL, Math.abs(slY - entryY));
            ctx.lineWidth = 1; ctx.setLineDash([]);
            ctx.strokeStyle = '#888';
            ctx.beginPath(); ctx.moveTo(boxL, entryY); ctx.lineTo(boxR, entryY); ctx.stroke();
            ctx.strokeStyle = '#0ECB81';
            ctx.beginPath(); ctx.moveTo(boxL, tpY); ctx.lineTo(boxR, tpY); ctx.stroke();
            ctx.strokeStyle = '#F6465D';
            ctx.beginPath(); ctx.moveTo(boxL, slY); ctx.lineTo(boxR, slY); ctx.stroke();
            ctx.font = '10px monospace';
            ctx.fillStyle = '#888';
            ctx.fillText('Entry ' + fmtP(d.startPrice), boxL + 4, entryY - 4);
            ctx.fillStyle = '#0ECB81';
            var tpPct = d.startPrice ? Math.abs(((d.endPrice - d.startPrice) / d.startPrice) * 100) : 0;
            ctx.fillText('TP ' + fmtP(d.endPrice) + ' (+' + tpPct.toFixed(2) + '%)', boxL + 4, tpY - 4);
            ctx.fillStyle = '#F6465D';
            var slPct = d.startPrice ? Math.abs(((slPrice - d.startPrice) / d.startPrice) * 100) : 0;
            ctx.fillText('SL ' + fmtP(slPrice) + ' (-' + slPct.toFixed(2) + '%)', boxL + 4, slY + 14);
            var rr = Math.abs(d.endPrice - d.startPrice) / Math.max(0.0001, Math.abs(d.startPrice - slPrice));
            ctx.fillStyle = '#ddd';
            ctx.fillText('R/R 1:' + rr.toFixed(1), boxL + 4, entryY + 14);

        } else if (d.type === 'shortpos') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            var entryY = sy, tpY = ey;
            var slY = entryY - (tpY - entryY);
            var slPrice = d.startPrice + (d.startPrice - d.endPrice);
            var boxL = Math.min(sx, ex), boxR = Math.max(sx, ex);
            if (boxR - boxL < 60) { boxL = 0; boxR = chartW; }
            ctx.fillStyle = 'rgba(14,203,129,0.1)';
            ctx.fillRect(boxL, Math.min(entryY, tpY), boxR - boxL, Math.abs(tpY - entryY));
            ctx.fillStyle = 'rgba(246,70,93,0.1)';
            ctx.fillRect(boxL, Math.min(entryY, slY), boxR - boxL, Math.abs(slY - entryY));
            ctx.lineWidth = 1; ctx.setLineDash([]);
            ctx.strokeStyle = '#888';
            ctx.beginPath(); ctx.moveTo(boxL, entryY); ctx.lineTo(boxR, entryY); ctx.stroke();
            ctx.strokeStyle = '#0ECB81';
            ctx.beginPath(); ctx.moveTo(boxL, tpY); ctx.lineTo(boxR, tpY); ctx.stroke();
            ctx.strokeStyle = '#F6465D';
            ctx.beginPath(); ctx.moveTo(boxL, slY); ctx.lineTo(boxR, slY); ctx.stroke();
            ctx.font = '10px monospace';
            ctx.fillStyle = '#888';
            ctx.fillText('Entry ' + fmtP(d.startPrice), boxL + 4, entryY - 4);
            ctx.fillStyle = '#0ECB81';
            var tpPct = d.startPrice ? Math.abs(((d.startPrice - d.endPrice) / d.startPrice) * 100) : 0;
            ctx.fillText('TP ' + fmtP(d.endPrice) + ' (+' + tpPct.toFixed(2) + '%)', boxL + 4, tpY + 14);
            ctx.fillStyle = '#F6465D';
            var slPct = d.startPrice ? Math.abs(((slPrice - d.startPrice) / d.startPrice) * 100) : 0;
            ctx.fillText('SL ' + fmtP(slPrice) + ' (-' + slPct.toFixed(2) + '%)', boxL + 4, slY - 4);
            var rr = Math.abs(d.startPrice - d.endPrice) / Math.max(0.0001, Math.abs(slPrice - d.startPrice));
            ctx.fillStyle = '#ddd';
            ctx.fillText('R/R 1:' + rr.toFixed(1), boxL + 4, entryY + 14);

        } else if (d.type === 'measure') {
            var sx = ts.logicalToCoordinate(d.startIdx), sy = series.priceToCoordinate(d.startPrice);
            var ex = ts.logicalToCoordinate(d.endIdx), ey = series.priceToCoordinate(d.endPrice);
            if (sx == null || sy == null || ex == null || ey == null) return;
            ctx.strokeStyle = '#aaa'; ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
            ctx.setLineDash([]);
            var pDiff = d.endPrice - d.startPrice;
            var pPct = d.startPrice ? ((pDiff / d.startPrice) * 100) : 0;
            ctx.fillStyle = '#ddd'; ctx.font = '10px monospace';
            ctx.fillText(fmtP(pDiff) + ' (' + pPct.toFixed(2) + '%)', (sx+ex)/2 + 4, (sy+ey)/2 - 6);
        }
    });

    ctx.restore();
}

// ═══════════════════════════════════════════════════════════
// AUTO TREND LINES
// ═══════════════════════════════════════════════════════════

function _clearTrendSeries() {
    if (!_tvChart) return;
    _tvTrendLines.forEach(s => { try{_tvChart.removeSeries(s);}catch(e){} });
    _tvTrendLines = [];
}

function _computeAutoTrend() {
    _clearTrendSeries();
    if (!_tvChart || !_autoTrendOn || !marketData[currentSymbol]) return;

    const ad = _resolveData(currentSymbol, currentTimeframe);
    if (!ad || !ad.timestamps || ad.timestamps.length < 15) return;

    const ts=ad.timestamps, hi=ad.highs||ad.history, lo=ad.lows||ad.history, cl=ad.history;
    const W = Math.max(3, Math.min(7, Math.floor(ts.length/15)));
    const swHi=[], swLo=[];
    for (let i=W;i<cl.length-W;i++) {
        let isH=true, isL=true;
        for (let j=i-W;j<=i+W;j++) { if(j===i)continue; if((hi[j]||cl[j])>=(hi[i]||cl[i]))isH=false; if((lo[j]||cl[j])<=(lo[i]||cl[i]))isL=false; }
        if (isH) swHi.push({i, price:hi[i]||cl[i], time:_toUnix(ts[i])});
        if (isL) swLo.push({i, price:lo[i]||cl[i], time:_toUnix(ts[i])});
    }

    if (swHi.length>=2) {
        const pts=swHi.slice(-2);
        const s=_tvChart.addLineSeries({color:'#ef4444',lineWidth:1,lineStyle:LightweightCharts.LineStyle.LargeDashed,crosshairMarkerVisible:false,lastValueVisible:false,priceLineVisible:false});
        s.setData([{time:pts[0].time,value:pts[0].price},{time:pts[1].time,value:pts[1].price}]);
        _tvTrendLines.push(s);
    }
    if (swLo.length>=2) {
        const pts=swLo.slice(-2);
        const s=_tvChart.addLineSeries({color:'#22c55e',lineWidth:1,lineStyle:LightweightCharts.LineStyle.LargeDashed,crosshairMarkerVisible:false,lastValueVisible:false,priceLineVisible:false});
        s.setData([{time:pts[0].time,value:pts[0].price},{time:pts[1].time,value:pts[1].price}]);
        _tvTrendLines.push(s);
    }
}

// Convert timestamp to Unix seconds for Lightweight Charts display.
// LC renders timestamps as-is (assumes UTC), so we shift by the browser's
// UTC offset to make the x-axis show the user's local time.
var _tzOffsetSec = -(new Date().getTimezoneOffset() * 60); // e.g., +10800 for UTC+3
function _toUnix(ts) {
    if (typeof ts === 'number') {
        // Integer from backend = UTC Unix seconds → shift to local
        return ts + _tzOffsetSec;
    }
    // String (legacy "YYYY-MM-DDTHH:MM" in exchange-local time):
    // new Date(str) interprets as browser-local; adding _tzOffsetSec cancels
    // that out so LC displays the raw exchange time as-is.
    return Math.floor(new Date(ts).getTime() / 1000) + _tzOffsetSec;
}

// ═══════════════════════════════════════════════════════════
// DATA RESOLUTION — find best available data for a timeframe
// ═══════════════════════════════════════════════════════════

function _resolveData(symbol, tf) {
    if (!marketData[symbol]?.data) return null;
    const d = marketData[symbol].data;
    // Try exact match first, then walk fallbacks
    if (d[tf]) return d[tf];
    for (const fb of TF_FALLBACKS) {
        if (d[fb]) return d[fb];
    }
    return null;
}

// ═══════════════════════════════════════════════════════════
// TIMEFRAME & UPDATE CHART
// ═══════════════════════════════════════════════════════════

function setTimeframe(tf) {
    currentTimeframe = tf;
    document.querySelectorAll('.time-btn').forEach(b => {
        b.className = `time-btn text-[10px] px-2 py-0.5 rounded transition-all ${b.dataset.tf===tf?'bg-slate-600 text-white':'text-slate-400'}`;
    });
    renderTickerButtons();
    renderMarketOverview();
    _refreshAllCompareCharts();
    updateChart(currentSymbol);
}

function updateChart(symbol) {
    currentSymbol = symbol;
    const ad = _resolveData(symbol, currentTimeframe);
    if (!ad) return;

    const meta = marketData[symbol];
    const history = ad.history || [];
    const price = ad.price || (history.length ? history[history.length-1] : 0);
    const change = ad.change || 0;
    candleData = { opens:ad.opens||[], highs:ad.highs||[], lows:ad.lows||[], volumes:ad.volumes||[], timestamps:ad.timestamps||[] };

    const open = ad.open || (history[0] ?? price);
    const high = ad.high || (history.length ? Math.max(...history) : price);
    const low = ad.low || (history.length ? Math.min(...history) : price);
    const prevClose = ad.prev_close || open;
    const volume = ad.volume || 0;
    const changeAbs = price - prevClose;

    // Header
    const dn = getAssetName(symbol, meta.name);
    const cs = symbol.replace('-USD','').replace('=F','');
    document.getElementById('selected-asset-name').textContent = `${dn} (${cs})`;
    document.getElementById('selected-price-large').textContent = '$'+price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
    const cd = document.getElementById('selected-change-detail');
    const sg = changeAbs>=0?'+':'';
    cd.textContent = `${sg}${changeAbs.toFixed(2)} (${sg}${change.toFixed(2)}%)`;
    cd.style.color = change>=0 ? (getComputedStyle(document.documentElement).getPropertyValue('--accent-light').trim()||'#34d399') : '#f87171';
    document.getElementById('selected-market-status').textContent = TF_LABELS[currentTimeframe] || '';

    // Stats
    const fmt = v => v ? v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '--';
    document.getElementById('stat-open').textContent = fmt(open);
    document.getElementById('stat-high').textContent = fmt(high);
    document.getElementById('stat-high').className = 'text-[11px] font-medium text-emerald-400';
    document.getElementById('stat-low').textContent = fmt(low);
    document.getElementById('stat-low').className = 'text-[11px] font-medium text-red-400';
    document.getElementById('stat-prev-close').textContent = fmt(prevClose);
    document.getElementById('stat-volume').textContent = volume>0 ? formatVolume(volume) : '--';
    document.getElementById('stat-day-range').textContent = (low&&high) ? `${fmt(low)} - ${fmt(high)}` : '--';

    // ── Build chart data ──
    const timestamps = ad.timestamps || [];
    const hasOHLC = candleData.opens.length===history.length && candleData.highs.length===history.length;
    let chartData = [];

    for (let i=0; i<history.length; i++) {
        const t = timestamps[i] != null ? _toUnix(timestamps[i]) : (Math.floor(Date.now()/1000) + _tzOffsetSec) - (history.length-1-i)*60;
        if (_chartType==='candle' && hasOHLC) {
            chartData.push({ time:t, open:candleData.opens[i]||history[i], high:candleData.highs[i]||history[i], low:candleData.lows[i]||history[i], close:history[i] });
        } else {
            chartData.push({ time:t, value:history[i] });
        }
    }

    // Deduplicate & sort (Lightweight Charts requires strictly increasing time)
    const seen = new Set();
    chartData = chartData.filter(d => { if(seen.has(d.time))return false; seen.add(d.time); return true; });
    chartData.sort((a,b) => a.time-b.time);

    // Line color based on direction
    if (_chartType==='line') {
        const accent = getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim()||'#10b981';
        _tvSeries.applyOptions({ color: change>=0 ? accent : '#ef4444' });
    }

    _tvSeries.setData(chartData);

    // Prev close line (only for intraday views where it makes sense)
    if (chartData.length >= 2 && (currentTimeframe==='1m' || currentTimeframe==='5m')) {
        _tvPrevLine.setData([
            {time:chartData[0].time, value:prevClose},
            {time:chartData[chartData.length-1].time, value:prevClose},
        ]);
    } else {
        _tvPrevLine.setData([]);
    }

    // ── Set visible range: show last N bars (bar-count approach) ──
    // Keep candles readable: ~3-6px per bar at typical chart widths
    const _barsForSymbol = (tf, sym) => {
        if (tf === '1m') return 60;    // ~1 hour (tighter view)
        if (tf === '5m') return 48;    // ~4 hours
        return { '1d_1mo': 30, '1d_1y': 60, '1wk': 52 }[tf] || 40;
    };
    if (chartData.length >= 2) {
        const n = Math.min(_barsForSymbol(currentTimeframe, symbol), chartData.length);
        _tvChart.timeScale().setVisibleLogicalRange({
            from: chartData.length - n,
            to: chartData.length
        });
    } else {
        _tvChart.timeScale().fitContent();
    }
    if (_autoTrendOn) _computeAutoTrend();
    setTimeout(() => _redrawOverlay(), 100);

    // Remove TV logo again (chart recreations can re-add it)
    const el = document.getElementById('tv-chart-container');
    if (el) _removeTVLogo(el);

    // Highlight active ticker
    document.querySelectorAll('.ticker-btn').forEach(btn => {
        const active = btn.dataset.symbol===symbol;
        btn.classList.toggle('bg-slate-700',active); btn.classList.toggle('border-slate-500',active);
        btn.classList.toggle('bg-slate-800',!active); btn.classList.toggle('border-slate-700',!active);
    });

    updateAssetAnalysis(symbol, price, change, changeAbs, open, high, low, prevClose, volume);
    renderMarketOverview();
}

function formatVolume(v) {
    if(v>=1e9)return(v/1e9).toFixed(2)+'B'; if(v>=1e6)return(v/1e6).toFixed(2)+'M';
    if(v>=1e3)return(v/1e3).toFixed(1)+'K'; return v.toLocaleString();
}

// ═══════════════════════════════════════════════════════════
// COMPARE CHARTS
// ═══════════════════════════════════════════════════════════

let _compareCharts = {}; // { symbol: { chart, series } }

function openCompareMenu(e) {
    const menu = document.getElementById('compare-menu');
    const items = document.getElementById('compare-menu-items');
    
    // If already visible, just hide and return
    if (!menu.classList.contains('hidden')) {
        menu.classList.add('hidden');
        return;
    }
    
    const keys = Object.keys(marketData).filter(s => s !== currentSymbol && !_compareCharts[s]);
    if (!keys.length) { items.innerHTML = '<div class="text-[10px] text-slate-500 px-2 py-1">All tickers shown</div>'; }
    else {
        items.innerHTML = keys.map(sym => {
            const lbl = sym.replace('-USD','').replace('=F','');
            const ad = _resolveData(sym, currentTimeframe);
            const chg = ad?.change || 0;
            const up = chg >= 0;
            return `<button onclick="addCompareChart('${sym}')" class="w-full text-left px-2 py-1.5 rounded text-[11px] hover:bg-slate-800 transition-colors flex items-center justify-between gap-2">
                <span class="text-slate-300 font-mono font-bold">${lbl}</span>
                <span class="font-mono text-[10px] ${up?'text-emerald-400':'text-red-400'}">${up?'+':''}${chg.toFixed(1)}%</span>
            </button>`;
        }).join('');
    }
    
    // Show menu
    menu.classList.remove('hidden');
    
    // Position near button — open upward so it doesn't go under Asset Analysis
    const btn = e.currentTarget; // Capture before event object resets currentTarget
    const rect = btn.getBoundingClientRect();
    menu.style.position = 'fixed';
    menu.style.right = (window.innerWidth - rect.right) + 'px';
    menu.style.left = 'auto';
    // Open upward: place bottom of menu at top of button
    menu.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
    menu.style.top = 'auto';

    // Close on outside click/tap (delayed to avoid catching this click)
    const handler = (ev) => {
        if (!menu.contains(ev.target) && !btn.contains(ev.target)) {
            menu.classList.add('hidden');
            document.removeEventListener('mousedown', handler);
            document.removeEventListener('touchstart', handler);
        }
    };
    // Remove any old handler first
    if (window._compareCloseHandler) {
        document.removeEventListener('mousedown', window._compareCloseHandler);
        document.removeEventListener('touchstart', window._compareCloseHandler);
    }
    window._compareCloseHandler = handler;
    setTimeout(() => {
        document.addEventListener('mousedown', handler);
        document.addEventListener('touchstart', handler);
    }, 50);
}

function addCompareChart(symbol) {
    document.getElementById('compare-menu').classList.add('hidden');
    if (_compareCharts[symbol]) return;

    const grid = document.getElementById('compare-grid');
    const id = 'compare-' + symbol.replace(/[^a-zA-Z0-9]/g, '_');

    // Create container
    const wrapper = document.createElement('div');
    wrapper.id = id;
    wrapper.className = 'bg-slate-900/60 border border-slate-800/60 rounded-lg p-2 relative';
    wrapper.innerHTML = `
        <div class="flex items-center justify-between mb-1">
            <button onclick="updateChart('${symbol}')" class="text-[10px] font-mono font-bold text-slate-300 hover:text-emerald-400 transition-colors cursor-pointer">${symbol.replace('-USD','').replace('=F','')}</button>
            <button onclick="removeCompareChart('${symbol}')" class="text-slate-600 hover:text-red-400 transition-colors"><i data-lucide="x" class="w-3 h-3"></i></button>
        </div>
        <div class="flex items-baseline gap-1.5 mb-1">
            <span id="${id}-price" class="text-[11px] font-bold text-white">--</span>
            <span id="${id}-change" class="text-[9px] font-mono">--</span>
        </div>
        <div id="${id}-chart" style="width:100%;height:80px;"></div>
    `;
    grid.appendChild(wrapper);
    lucide.createIcons();

    // Create mini chart
    const cs = getComputedStyle(document.documentElement);
    const chartEl = document.getElementById(id + '-chart');
    const chart = LightweightCharts.createChart(chartEl, {
        width: chartEl.clientWidth, height: 80,
        layout: { background:{type:'solid',color:'transparent'}, textColor:'#475569', fontSize:9 },
        grid: { vertLines:{visible:false}, horzLines:{color:'rgba(148,163,184,0.04)'} },
        rightPriceScale: { visible:false },
        timeScale: { visible:false },
        crosshair: { mode: LightweightCharts.CrosshairMode.Hidden },
        handleScroll: false, handleScale: false,
    });
    const series = chart.addAreaSeries({
        lineColor:'#10b981', topColor:'rgba(16,185,129,0.15)', bottomColor:'rgba(16,185,129,0.01)',
        lineWidth:1.5, crosshairMarkerVisible:false, lastValueVisible:false, priceLineVisible:false,
    });

    _compareCharts[symbol] = { chart, series, wrapper };

    // Resize observer
    new ResizeObserver(() => chart.applyOptions({width:chartEl.clientWidth})).observe(chartEl);

    _updateCompareChart(symbol);
}

function _updateCompareChart(symbol) {
    const cc = _compareCharts[symbol];
    if (!cc) return;
    const ad = _resolveData(symbol, currentTimeframe);
    if (!ad) return;

    const id = 'compare-' + symbol.replace(/[^a-zA-Z0-9]/g, '_');
    const price = ad.price || 0;
    const change = ad.change || 0;
    const up = change >= 0;

    const priceEl = document.getElementById(id + '-price');
    const changeEl = document.getElementById(id + '-change');
    if (priceEl) priceEl.textContent = '$' + price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
    if (changeEl) { changeEl.textContent = `${up?'+':''}${change.toFixed(2)}%`; changeEl.className = `text-[9px] font-mono font-bold ${up?'text-emerald-400':'text-red-400'}`; }

    cc.series.applyOptions({
        lineColor: up ? '#10b981' : '#ef4444',
        topColor: up ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
        bottomColor: up ? 'rgba(16,185,129,0.01)' : 'rgba(239,68,68,0.01)',
    });

    const ts = ad.timestamps || [];
    const hist = ad.history || [];
    const chartData = [];
    const seen = new Set();
    for (let i = 0; i < hist.length; i++) {
        const t = ts[i] != null ? _toUnix(ts[i]) : (Math.floor(Date.now()/1000) + _tzOffsetSec) - (hist.length-1-i)*60;
        if (!seen.has(t)) { seen.add(t); chartData.push({time:t, value:hist[i]}); }
    }
    chartData.sort((a,b) => a.time - b.time);
    cc.series.setData(chartData);
    cc.chart.timeScale().fitContent();
}

function removeCompareChart(symbol) {
    const cc = _compareCharts[symbol];
    if (!cc) return;
    try { cc.chart.remove(); } catch(e) {}
    cc.wrapper.remove();
    delete _compareCharts[symbol];
}

function _refreshAllCompareCharts() {
    for (const sym of Object.keys(_compareCharts)) _updateCompareChart(sym);
}

// ═══════════════════════════════════════════════════════════
// ASSET ANALYSIS (Enhanced)
// ═══════════════════════════════════════════════════════════

function updateAssetAnalysis(symbol, price, change, changeAbs, open, high, low, prevClose, volume) {
    const container = document.getElementById('analysis-content');
    const meta = marketData[symbol]||{};
    const dn = getAssetName(symbol, meta.name);
    const cs = symbol.replace('-USD','').replace('=F','');
    const isUp = change>=0, mag = Math.abs(change);
    const fp = v => v!=null ? v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '--';

    // Compute metrics from history
    const ad = _resolveData(symbol, currentTimeframe);
    const hist = ad?.history || [];
    const highs = ad?.highs || hist;
    const lows = ad?.lows || hist;

    // Momentum (rate of change last 14 bars vs 14 before that)
    let momentum = 0, momentumLabel = 'Neutral';
    if (hist.length >= 28) {
        const recent = hist[hist.length-1] / hist[hist.length-14] - 1;
        const prior = hist[hist.length-14] / hist[hist.length-28] - 1;
        momentum = recent - prior;
        if (momentum > 0.02) momentumLabel = 'Accelerating';
        else if (momentum > 0.005) momentumLabel = 'Building';
        else if (momentum < -0.02) momentumLabel = 'Decelerating';
        else if (momentum < -0.005) momentumLabel = 'Fading';
    }

    // Volatility (standard deviation of % returns, last 20 bars)
    let volatility = 0;
    const volWindow = Math.min(20, hist.length - 1);
    if (volWindow >= 5) {
        const returns = [];
        for (let i = hist.length - volWindow; i < hist.length; i++) {
            if (hist[i-1] > 0) returns.push((hist[i] - hist[i-1]) / hist[i-1]);
        }
        const mean = returns.reduce((a,b) => a+b, 0) / returns.length;
        volatility = Math.sqrt(returns.reduce((a,r) => a + (r-mean)**2, 0) / returns.length) * 100;
    }
    let volLabel = 'Low';
    if (volatility > 3) volLabel = 'Extreme';
    else if (volatility > 1.5) volLabel = 'High';
    else if (volatility > 0.5) volLabel = 'Moderate';

    // Support / Resistance (recent swing highs/lows)
    let support = null, resistance = null;
    if (hist.length >= 10) {
        const lookback = Math.min(60, hist.length);
        const recentHi = highs.slice(-lookback);
        const recentLo = lows.slice(-lookback);
        const swingHighs = [], swingLows = [];
        for (let i = 2; i < recentHi.length - 2; i++) {
            if (recentHi[i] > recentHi[i-1] && recentHi[i] > recentHi[i-2] && recentHi[i] > recentHi[i+1] && recentHi[i] > recentHi[i+2])
                swingHighs.push(recentHi[i]);
            if (recentLo[i] < recentLo[i-1] && recentLo[i] < recentLo[i-2] && recentLo[i] < recentLo[i+1] && recentLo[i] < recentLo[i+2])
                swingLows.push(recentLo[i]);
        }
        if (swingHighs.length) resistance = swingHighs.filter(h => h > price).sort((a,b) => a-b)[0] || Math.max(...swingHighs);
        if (swingLows.length) support = swingLows.filter(l => l < price).sort((a,b) => b-a)[0] || Math.min(...swingLows);
    }

    // Distance from period high/low
    const distFromHigh = high > 0 ? ((price - high) / high * 100) : 0;
    const distFromLow = low > 0 ? ((price - low) / low * 100) : 0;

    // Range position (0% = at low, 100% = at high)
    const rangePos = (high > low) ? ((price - low) / (high - low) * 100) : 50;

    // Trend description
    let trend = 'flat';
    if (mag > 5) trend = isUp ? 'surging' : 'plunging';
    else if (mag > 2) trend = isUp ? 'rising strongly' : 'falling sharply';
    else if (mag > 0.5) trend = isUp ? 'trending up' : 'trending down';

    // Category context
    const isCrypto = symbol.includes('-USD');
    const isMetal = ['GC=F','SI=F','HG=F','PL=F'].includes(symbol);

    let contextLine = '';
    if (isMetal && mag > 2) contextLine = `<span class="text-amber-400/70">Significant precious metals move — often signals macro shifts.</span>`;
    else if (isCrypto && mag > 5) contextLine = `<span class="text-purple-400/70">Elevated volatility — typical crypto conditions.</span>`;
    else if (mag > 3) contextLine = `<span class="text-sky-400/70">Notable move — worth monitoring for continuation or reversal.</span>`;

    const momentumColor = momentum > 0.005 ? 'text-emerald-400' : momentum < -0.005 ? 'text-red-400' : 'text-slate-400';
    const volColor = volatility > 1.5 ? 'text-amber-400' : volatility > 0.5 ? 'text-yellow-400' : 'text-slate-400';

    container.innerHTML = `
        <div class="bg-slate-900/60 border border-slate-800/60 rounded-lg p-3">
            <!-- Header -->
            <div class="flex items-center gap-2 mb-2">
                <span class="text-[10px] font-bold ${isUp?'text-emerald-400':'text-red-400'} uppercase">${isUp?'↑ Bullish':'↓ Bearish'}</span>
                <span class="text-[10px] text-slate-600">•</span>
                <span class="text-[10px] text-slate-500">${cs}</span>
                <span class="text-[10px] text-slate-600">•</span>
                <span class="text-[10px] ${momentumColor}">${momentumLabel}</span>
            </div>
            <p class="text-xs text-slate-300 leading-relaxed mb-3">${dn} is ${trend} at $${fp(price)}, ${isUp?'up':'down'} ${mag.toFixed(2)}% from $${fp(prevClose)}.</p>
            ${contextLine ? `<p class="text-[10px] leading-relaxed mb-3">${contextLine}</p>` : ''}

            <!-- Metrics Grid -->
            <div class="grid grid-cols-2 gap-x-4 gap-y-2 mb-3">
                <div class="flex justify-between items-center">
                    <span class="text-[10px] text-slate-500">Momentum</span>
                    <span class="text-[10px] font-mono font-bold ${momentumColor}">${momentumLabel}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-[10px] text-slate-500">Volatility</span>
                    <span class="text-[10px] font-mono font-bold ${volColor}">${volLabel} (${volatility.toFixed(2)}%)</span>
                </div>
                ${support != null ? `<div class="flex justify-between items-center">
                    <span class="text-[10px] text-slate-500">Support</span>
                    <span class="text-[10px] font-mono font-bold text-emerald-400">$${fp(support)}</span>
                </div>` : ''}
                ${resistance != null ? `<div class="flex justify-between items-center">
                    <span class="text-[10px] text-slate-500">Resistance</span>
                    <span class="text-[10px] font-mono font-bold text-red-400">$${fp(resistance)}</span>
                </div>` : ''}
            </div>

            <!-- Range Position Bar -->
            <div class="mb-3">
                <div class="flex justify-between text-[9px] text-slate-500 mb-1">
                    <span>$${fp(low)}</span>
                    <span class="text-slate-400 font-bold">${rangePos.toFixed(0)}% of range</span>
                    <span>$${fp(high)}</span>
                </div>
                <div class="w-full h-1.5 bg-slate-800 rounded-full relative overflow-hidden">
                    <div class="absolute inset-0 rounded-full" style="background:linear-gradient(90deg,#ef4444,#fbbf24,#22c55e);opacity:0.3;"></div>
                    <div class="absolute top-0 h-full w-1 rounded-full bg-white shadow-sm shadow-white/50" style="left:${Math.min(99,Math.max(1,rangePos))}%;transform:translateX(-50%);"></div>
                </div>
            </div>

            <!-- Key Levels -->
            <div class="space-y-1">
                <div class="flex justify-between text-[10px]">
                    <span class="text-slate-500">From High</span>
                    <span class="font-mono ${distFromHigh >= 0 ? 'text-emerald-400' : 'text-red-400'}">${distFromHigh >= 0 ? '+' : ''}${distFromHigh.toFixed(2)}%</span>
                </div>
                <div class="flex justify-between text-[10px]">
                    <span class="text-slate-500">From Low</span>
                    <span class="font-mono ${distFromLow >= 0 ? 'text-emerald-400' : 'text-red-400'}">${distFromLow >= 0 ? '+' : ''}${distFromLow.toFixed(2)}%</span>
                </div>
                ${volume > 0 ? `<div class="flex justify-between text-[10px]">
                    <span class="text-slate-500">Volume</span>
                    <span class="font-mono text-slate-300">${formatVolume(volume)}</span>
                </div>` : ''}
            </div>
        </div>
        <a href="https://finance.yahoo.com/quote/${symbol}" target="_blank"
           class="flex items-center justify-center gap-2 text-[11px] text-slate-500 hover:text-emerald-400 transition-colors mt-2 py-1.5 rounded border border-slate-800/50 hover:border-emerald-500/30">
            <i data-lucide="external-link" class="w-3 h-3"></i> View on Yahoo Finance
        </a>`;
    lucide.createIcons();
}

// ═══════════════════════════════════════════════════════════
// MARKET OVERVIEW
// ═══════════════════════════════════════════════════════════

function renderMarketOverview() {
    const container = document.getElementById('overview-content');
    if (!container) return;
    const keys = Object.keys(marketData);
    if (!keys.length) { container.innerHTML = '<div class="text-xs text-slate-500 italic">No market data</div>'; return; }

    const sorted = keys.map(sym => {
        const ad = _resolveData(sym, currentTimeframe);
        return { sym, ad, change: ad?.change || 0 };
    }).sort((a,b) => b.change - a.change);

    const maxAbs = Math.max(1, Math.max(...sorted.map(i => Math.abs(i.change))));
    const fp = v => v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});

    let html = '<div class="heatmap-row" style="display:flex;flex-wrap:wrap;gap:4px;">';

    for (const {sym, ad} of sorted) {
        if (!ad) continue;
        const lbl = sym.replace('-USD','').replace('=F','');
        const price = ad.price || 0;
        const change = ad.change || 0;
        const up = change >= 0;
        const intensity = Math.min(1, Math.abs(change) / maxAbs);

        const bg = up
            ? `rgba(0,${Math.round(180*intensity)},80,${(0.05+intensity*0.14).toFixed(2)})`
            : `rgba(${Math.round(200*intensity)},0,40,${(0.05+intensity*0.14).toFixed(2)})`;
        const border = up
            ? `rgba(34,197,94,${(0.1+intensity*0.2).toFixed(2)})`
            : `rgba(239,68,68,${(0.1+intensity*0.2).toFixed(2)})`;

        html += `<div onclick="updateChart('${sym}')" class="cursor-pointer rounded-md transition-all hover:scale-[1.03]" style="background:${bg};border:1px solid ${border};padding:4px 8px;min-width:0;flex:1 1 0;">`;
        html += `<div style="display:flex;align-items:center;justify-content:space-between;gap:6px;"><span class="text-[10px] font-mono font-bold text-white">${lbl}</span><span class="text-[9px] font-mono font-bold ${up?'text-emerald-400':'text-red-400'}">${up?'+':''}${change.toFixed(1)}%</span></div>`;
        html += `<div class="text-[9px] font-mono text-slate-400">$${fp(price)}</div>`;
        html += `</div>`;
    }

    html += '</div>';
    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════
// MANUAL SEARCH TOGGLE
// ═══════════════════════════════════════════════════════════

let manualSearchCollapsed = false;
function toggleManualSearch() {}
function applyManualSearchState() {}

// ═══════════════════════════════════════════════════════════
// TICKER BUTTONS + DRAG REORDER
// ═══════════════════════════════════════════════════════════

function renderTickerButtons() {
    const container = document.getElementById('ticker-buttons');
    let keys = Object.keys(marketData);
    if (!keys.length) { container.innerHTML='<span class="text-xs text-slate-500">No market data</span>'; return; }
    const saved = _getTickerOrder();
    if (saved.length) { const o=saved.filter(s=>keys.includes(s)), r=keys.filter(s=>!o.includes(s)); keys=[...o,...r]; }

    container.innerHTML = keys.map(sym => {
        const asset = marketData[sym];
        // Use current timeframe for change%, fallback through chain
        const ad = _resolveData(sym, currentTimeframe);
        const chg = ad?.change || 0;
        const up = chg>=0, lbl = sym.replace('-USD','').replace('=F','');
        return `<button draggable="true" data-symbol="${sym}" ondragstart="_tickerDragStart(event)" ondragover="_tickerDragOver(event)" ondrop="_tickerDrop(event)" ondragend="_tickerDragEnd(event)" onclick="updateChart('${sym}')" class="ticker-btn flex-shrink-0 bg-slate-800 border border-slate-700 rounded px-2.5 py-1 hover:border-slate-500 transition-all flex items-center gap-1.5 cursor-grab active:cursor-grabbing"><span class="text-xs font-mono font-bold text-slate-300">${lbl}</span><span class="text-[10px] font-mono font-bold ${up?'text-emerald-400':'text-red-400'}">${up?'+':''}${chg.toFixed(1)}%</span></button>`;
    }).join('');
}

const _TICKER_ORDER_KEY = 'stratos_ticker_order';
let _tickerDragSym = null;
function _getTickerOrder(){try{return JSON.parse(localStorage.getItem(_TICKER_ORDER_KEY)||'[]');}catch(e){return[];}}
function _saveTickerOrder(){const b=document.querySelectorAll('#ticker-buttons .ticker-btn[data-symbol]');localStorage.setItem(_TICKER_ORDER_KEY,JSON.stringify(Array.from(b).map(b=>b.dataset.symbol)));}
function _tickerDragStart(e){_tickerDragSym=e.currentTarget.dataset.symbol;e.currentTarget.style.opacity='0.4';e.dataTransfer.effectAllowed='move';}
function _tickerDragOver(e){e.preventDefault();e.dataTransfer.dropEffect='move';e.currentTarget.style.borderColor='rgba(52,211,153,0.5)';}
function _tickerDrop(e){
    e.preventDefault();const t=e.currentTarget;t.style.borderColor='';const ts=t.dataset.symbol;
    if(!_tickerDragSym||_tickerDragSym===ts)return;
    const c=document.getElementById('ticker-buttons'),btns=Array.from(c.querySelectorAll('.ticker-btn[data-symbol]'));
    const db=btns.find(b=>b.dataset.symbol===_tickerDragSym),tb=btns.find(b=>b.dataset.symbol===ts);
    if(!db||!tb)return;
    btns.indexOf(db)<btns.indexOf(tb)?c.insertBefore(db,tb.nextSibling):c.insertBefore(db,tb);
    _saveTickerOrder();
}
function _tickerDragEnd(e){e.currentTarget.style.opacity='';document.querySelectorAll('#ticker-buttons .ticker-btn').forEach(b=>b.style.borderColor='');}

// ═══════════════════════════════════════════════════════════
// CLICKABLE TICKERS IN AGENT CHAT
// ═══════════════════════════════════════════════════════════

function navigateToTicker(symbol) {
    // Switch to a tab with a chart if on settings
    if (typeof setActive==='function' && typeof activeRoot!=='undefined') {
        if (activeRoot === 'settings') setActive('dashboard');
    }
    // Scroll to markets widget
    const widget = document.getElementById('markets-widget');
    if (widget && widget.offsetParent) widget.scrollIntoView({behavior:'smooth',block:'start'});
    // Find the actual symbol key
    setTimeout(() => {
        const tryKeys = [symbol, symbol+'-USD', symbol+'=F',
            symbol.replace('-USD',''), symbol.replace('=F','')];
        for (const v of tryKeys) {
            if (marketData[v]) { updateChart(v); return; }
        }
    }, 200);
}

// ═══════════════════════════════════════════════════════════
// DEBUG EXPORT
// ═══════════════════════════════════════════════════════════

function exportMarketDebug() {
    const dump = {
        _meta: {
            exported: new Date().toISOString(),
            currentSymbol,
            currentTimeframe,
            chartType: _chartType,
            screenWidth: document.getElementById('tv-chart-container')?.clientWidth,
            screenHeight: document.getElementById('tv-chart-container')?.clientHeight,
        },
        marketData,
    };
    const blob = new Blob([JSON.stringify(dump, null, 2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'stratos_market_debug.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

// ═══════════════════════════════════════════════════════════
// CHART EXPORT (PNG Screenshot)
// ═══════════════════════════════════════════════════════════

function exportChartPNG() {
    if (!_tvChart) return;
    try {
        const canvas = _tvChart.takeScreenshot();
        const a = document.createElement('a');
        const sym = (currentSymbol || 'chart').replace(/[^a-zA-Z0-9_-]/g, '_');
        const tf = currentTimeframe || 'unknown';
        a.download = `STRAT_OS_${sym}_${tf}_${new Date().toISOString().slice(0,10)}.png`;
        a.href = canvas.toDataURL('image/png');
        a.click();
    } catch(e) {
        if (typeof showToast === 'function') showToast('Screenshot failed', 'error', 2000);
    }
}

// ═══════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS (active only on dashboard/market view)
// ═══════════════════════════════════════════════════════════

document.addEventListener('keydown', function(e) {
    // Skip if user is typing in an input, textarea, or contenteditable
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
    // Only active on dashboard view
    if (typeof activeRoot !== 'undefined' && activeRoot !== 'dashboard') return;

    const TFS = ['1m','5m','1d_1mo','1d_1y','1wk'];

    switch(e.key) {
        // 1-5: switch timeframes
        case '1': e.preventDefault(); setTimeframe(TFS[0]); break;
        case '2': e.preventDefault(); setTimeframe(TFS[1]); break;
        case '3': e.preventDefault(); setTimeframe(TFS[2]); break;
        case '4': e.preventDefault(); setTimeframe(TFS[3]); break;
        case '5': e.preventDefault(); setTimeframe(TFS[4]); break;
        // j/k: cycle tickers
        case 'j': case 'k': {
            const keys = Object.keys(marketData || {});
            if (keys.length < 2) break;
            const idx = keys.indexOf(currentSymbol);
            const next = e.key === 'j'
                ? keys[(idx + 1) % keys.length]
                : keys[(idx - 1 + keys.length) % keys.length];
            updateChart(next);
            break;
        }
        // d: toggle draw mode
        case 'd': e.preventDefault(); toggleDrawMode(); break;
        // c: toggle candle/line
        case 'c': e.preventDefault(); setChartType(_chartType === 'candle' ? 'line' : 'candle'); break;
        // t: toggle auto-trend
        case 't': e.preventDefault(); toggleAutoTrend(); break;
        // x: toggle crosshair
        case 'x': e.preventDefault(); toggleCrosshair(); break;
        // s: screenshot
        case 's': if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); exportChartPNG(); } break;
        // Escape: cancel drawing / clear draw mode
        case 'Escape':
            if (_drawMode) { toggleDrawMode(); }
            break;
    }
});

/* ═══════════════════════════════════════════════════
   CHART ZOOM TOOLBAR
   ═══════════════════════════════════════════════════ */

function _addMainChartToolbar(container) {
    var wrapper = container.parentElement || document.getElementById('chart-wrapper');
    if (!wrapper || wrapper.querySelector('.chart-toolbar')) return;
    wrapper.style.position = 'relative';

    var bar = document.createElement('div');
    bar.className = 'chart-toolbar';
    bar.innerHTML =
        '<button onclick="_mainChartZoom(1)" title="Zoom In (scroll wheel also works)">+</button>' +
        '<button onclick="_mainChartZoom(-1)" title="Zoom Out">−</button>' +
        '<button onclick="_mainChartFit()" title="Fit all data">⟲</button>' +
        '<button onclick="_mainChartVZoom(1)" title="Expand price scale">↕</button>';
    wrapper.appendChild(bar);
}

function _mainChartZoom(dir) {
    if (!_tvChart) return;
    var ts = _tvChart.timeScale();
    var range = ts.getVisibleLogicalRange();
    if (!range) return;
    var center = (range.from + range.to) / 2;
    var span = range.to - range.from;
    var factor = dir > 0 ? 0.7 : 1.4;
    var newSpan = Math.max(5, span * factor);
    ts.setVisibleLogicalRange({ from: center - newSpan / 2, to: center + newSpan / 2 });
}

function _mainChartFit() {
    if (!_tvChart) return;
    _tvChart.timeScale().fitContent();
    _tvChart.priceScale('right').applyOptions({ autoScale: true });
}

function _mainChartVZoom(dir) {
    if (!_tvChart || !_tvSeries) return;
    var ps = _tvChart.priceScale('right');
    ps.applyOptions({ autoScale: false });
    var el = _tvChart.chartElement();
    if (!el) return;
    var h = el.clientHeight;
    var topPrice = _tvSeries.coordinateToPrice(0);
    var botPrice = _tvSeries.coordinateToPrice(h);
    if (topPrice == null || botPrice == null) { ps.applyOptions({autoScale:true}); return; }
    var half = Math.abs(topPrice - botPrice) / 2;
    var factor = dir > 0 ? 0.75 : 1.33;
    var newHalf = half * factor;
    // Adjust scaleMargins to simulate vertical zoom
    var topMargin = 0.5 - (newHalf / (2 * half)) * 0.5;
    topMargin = Math.max(0.01, Math.min(0.45, topMargin));
    ps.applyOptions({ scaleMargins: { top: topMargin, bottom: topMargin } });
}
