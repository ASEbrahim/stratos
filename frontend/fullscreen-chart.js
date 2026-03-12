/**
 * fullscreen-chart.js — StratOS Fullscreen Chart Mode
 * Extracted from mobile.js. Works on ALL devices (desktop + mobile).
 */
(function() {
'use strict';

/* ── Utility guards (shared with mobile.js) ── */
const isMobile = () => window.innerWidth <= 1024;
const isSmall  = () => window.innerWidth <= 768;

/* ── Init fullscreen charts on DOMContentLoaded ── */
document.addEventListener('DOMContentLoaded', () => {
    initChartFullscreen();
});

/* ═══════════════════════════════════════════════
   E.  FULLSCREEN CHART MODE (mobile + desktop)
   ═══════════════════════════════════════════════ */

function initChartFullscreen() {
    /* No size guard — works on mobile AND desktop */
    /* Focus buttons are now built into each chart panel's toolbar (markets-panel.js)
       and inline in the summary tab toolbar (index.html). No overlay icons needed. */
}

function _addFullscreenBtn(container, titleFn) {
    if (container.querySelector('.chart-fs-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'chart-fs-btn';
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></svg>';
    btn.title = 'Fullscreen';
    Object.assign(btn.style, {
        position: 'absolute', top: '6px', right: '6px', zIndex: '25',
        width: '28px', height: '28px', borderRadius: '6px',
        background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.1)',
        color: 'var(--text-secondary)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
        backdropFilter: 'blur(4px)'
    });
    container.style.position = 'relative';
    container.appendChild(btn);

    btn.addEventListener('click', e => {
        e.stopPropagation();
        _openFullscreenChart(container, titleFn());
    });
}

/* ═══════════════════════════════════════════════════════════
   BINANCE-STYLE FULLSCREEN CHART
   Three zones: Header bar, Drawing toolbar, Chart area
   ═══════════════════════════════════════════════════════════ */

var _fs = null; /* Active fullscreen chart state */

/* ── Chart utilities moved to fullscreen-chart-utils.js ── */
/* _fsTfMap, _fsTzOff, _calcSMA, _fsFormatVol, _fsFmtPrice, _fsBuildData
   are declared globally in fullscreen-chart-utils.js (loaded before this file). */

/* ── Open Fullscreen Chart (Binance-style) ── */
/* Expose to global scope so inline onclick handlers (index.html, markets-panel.js) can call it */
window._openFullscreenChart = _openFullscreenChart;

function _openFullscreenChart(sourceEl, title) {
    /* Reset stuck state from a previous crashed attempt */
    if (_fs) {
        try {
            if (_fs.elements) {
                if (_fs.elements.root) try { _fs.elements.root.remove(); } catch(e){}
                if (_fs.elements.bg) try { _fs.elements.bg.remove(); } catch(e){}
                if (_fs.elements.hotbar) try { _fs.elements.hotbar.remove(); } catch(e){}
                if (_fs.elements.fullPanel) try { _fs.elements.fullPanel.remove(); } catch(e){}
            }
            if (_fs.chart) try { _fs.chart.remove(); } catch(e){}
        } catch(e) {}
        _fs = null;
    }

    if (!sourceEl) {
        if (typeof showToast === 'function') showToast('Chart element not found', 'error');
        return;
    }

    try { return _openFullscreenChartInner(sourceEl, title); }
    catch(err) {
        console.error('[Focus] Fullscreen chart error:', err);
        if (typeof showToast === 'function') showToast('Focus mode error: ' + err.message, 'error');
        /* Clean up partial state */
        if (_fs) {
            try {
                if (_fs.elements) {
                    if (_fs.elements.root) try { _fs.elements.root.remove(); } catch(e){}
                    if (_fs.elements.bg) try { _fs.elements.bg.remove(); } catch(e){}
                    if (_fs.elements.hotbar) try { _fs.elements.hotbar.remove(); } catch(e){}
                    if (_fs.elements.fullPanel) try { _fs.elements.fullPanel.remove(); } catch(e){}
                }
                if (_fs.chart) try { _fs.chart.remove(); } catch(e){}
            } catch(e) {}
            _fs = null;
        }
    }
}

function _openFullscreenChartInner(sourceEl, title) {
    /* Determine symbol and timeframe */
    var symbol = '', tfKey = '1m', mpEntry = null;
    var isMain = sourceEl.id === 'chart-wrapper' || sourceEl.contains(document.getElementById('tv-chart-container'));
    if (isMain && typeof currentSymbol !== 'undefined') {
        symbol = currentSymbol;
        tfKey = (typeof currentTimeframe !== 'undefined') ? currentTimeframe : '1m';
    }
    if (!symbol && typeof _mpCharts !== 'undefined') {
        for (var i = 0; i < _mpCharts.length; i++) {
            var wrap = document.getElementById(_mpCharts[i].id + '-chartwrap');
            if (wrap && (sourceEl === wrap || sourceEl.contains(wrap) || wrap.contains(sourceEl))) {
                symbol = _mpCharts[i].symbol;
                tfKey = _mpCharts[i].timeframe || '1m';
                mpEntry = _mpCharts[i];
                break;
            }
        }
    }
    if (!symbol) { if (typeof showToast === 'function') showToast('No chart data', 'error'); return; }

    var built = _fsBuildData(symbol, tfKey);
    if (!built || built.data.length === 0) { if (typeof showToast === 'function') showToast('No chart data to display', 'error'); return; }

    var isMobile = typeof isSmall === 'function' && isSmall();
    var ad = built.ad;
    var price = ad.price || (ad.history ? ad.history[ad.history.length-1] : 0);
    var change = ad.change || 0;
    var assetName = (typeof marketData !== 'undefined' && marketData[symbol]) ? (marketData[symbol].name || symbol) : symbol;

    /* ── State ── */
    _fs = {
        symbol: symbol,
        tfKey: tfKey,
        chart: null,
        series: null,
        maSeries: [],
        drawCanvas: null,
        drawCtx: null,
        drawings: [],
        drawMode: null, /* null, 'trend', 'ray', 'hline', 'hray', 'vline', 'channel', 'fib', 'rect', 'longpos', 'shortpos', 'measure', 'text', 'eraser' */
        drawStart: null,
        crosshairOn: true,
        magnetOn: false,
        lockOn: false,
        hideOn: false,
        channelBase: null,
        crosshairSub: null,
        isMobile: isMobile,
        elements: {},
        selectedIdx: -1,   /* Index of currently selected drawing (-1 = none) */
        dragAnchor: null,   /* 'start'|'end'|'body' — which part is being dragged */
        dragging: false,    /* Whether we are currently dragging a drawing */
        dragStart: null,    /* {timeIdx, price} at drag start */
        hoverIdx: -1,       /* Index of drawing under cursor (for delete key) */
    };
    window._fs = _fs;

    /* ── Backdrop ── */
    var bg = document.createElement('div');
    bg.className = 'cfs-bg';
    document.body.appendChild(bg);
    _fs.elements.bg = bg;

    /* ── Root container ── */
    var root = document.createElement('div');
    root.className = 'cfs-root';
    document.body.appendChild(root);
    _fs.elements.root = root;

    /* ── ZONE 1: Header bar ── */
    var header = document.createElement('div');
    header.className = 'cfs-header';

    /* Left: Ticker + Price + Change + 24h stats */
    var priceColor = change >= 0 ? '#0ECB81' : '#F6465D';
    var changeStr = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';

    /* Check for Binance 24h stats */
    var stats24h = ad.stats_24h || null;
    var statsHTML = '';
    if (stats24h) {
        var chg24 = stats24h.change_pct || 0;
        var col24 = chg24 >= 0 ? '#0ECB81' : '#F6465D';
        statsHTML = '<span class="cfs-24h" style="color:#848e9c;font-size:10px;margin-left:6px">' +
            '24h <span style="color:' + col24 + '">' + (chg24 >= 0 ? '+' : '') + chg24.toFixed(2) + '%</span>' +
            '  H <span style="color:#ddd">' + _fsFmtPrice(stats24h.high) + '</span>' +
            '  L <span style="color:#ddd">' + _fsFmtPrice(stats24h.low) + '</span>' +
            '  Vol <span style="color:#ddd">' + _fsFormatVol(stats24h.volume) + '</span>' +
            '</span>';
    }

    var tickerSymLabel = symbol.replace(/-USD/,'').replace(/=F/,'').replace(/=X/,'');
    var tickerFullName = (typeof getAssetName === 'function') ? getAssetName(symbol, assetName) : assetName;
    var showFullName = tickerFullName && tickerFullName !== symbol && tickerFullName !== tickerSymLabel;
    var leftHTML = '<div class="cfs-hdr-left" style="position:relative;">' +
        '<button class="cfs-ticker cfs-ticker-btn" title="Switch ticker">' +
        '<span class="cfs-ticker-sym">' + tickerSymLabel + '</span>' +
        (showFullName ? '<span class="cfs-ticker-name">' + tickerFullName + '</span>' : '') +
        ' <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="vertical-align:middle;opacity:0.6;"><path d="M6 9l6 6 6-6"/></svg></button>' +
        '<span class="cfs-price" style="color:' + priceColor + '">' + _fsFmtPrice(price) + '</span>' +
        '<span class="cfs-change" style="color:' + priceColor + '">' + changeStr + '</span>' +
        statsHTML +
        '</div>';

    /* Center: Timeframe buttons */
    var tfHTML = '<div class="cfs-tf-row">';
    _fsTfMap.forEach(function(tf) {
        var cls = tf.key === tfKey ? 'cfs-tf active' : 'cfs-tf';
        tfHTML += '<button class="' + cls + '" data-tf="' + tf.key + '">' + tf.label + '</button>';
    });
    tfHTML += '</div>';

    /* Right: Close */
    var rightHTML = '<button class="cfs-close-btn" title="Close (Esc)">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>' +
        '</button>';

    header.innerHTML = leftHTML + tfHTML + rightHTML;
    root.appendChild(header);

    /* ── Ticker dropdown menu ── */
    var tickerMenu = document.createElement('div');
    tickerMenu.className = 'cfs-ticker-menu';
    tickerMenu.style.cssText = 'display:none; position:absolute; top:calc(100% + 4px); left:0; min-width:240px; max-height:340px; overflow-y:auto; background:#252930; border:1px solid rgba(240,185,11,0.25); border-radius:10px; box-shadow:0 12px 32px rgba(0,0,0,0.7); z-index:50; padding:6px; backdrop-filter:blur(12px);';
    header.querySelector('.cfs-hdr-left').appendChild(tickerMenu);

    function _fsPopulateTickerMenu() {
        var syms = (typeof marketData !== 'undefined') ? Object.keys(marketData) : [];
        tickerMenu.innerHTML = syms.map(function(s) {
            var ad = (typeof _resolveData === 'function') ? _resolveData(s, _fs.tfKey) : null;
            var chg = ad ? (ad.change || 0) : 0;
            var up = chg >= 0;
            var lbl = s.replace('-USD','').replace('=F','');
            var name = (typeof getAssetName === 'function') ? getAssetName(s, marketData[s]?.name) : s;
            var isActive = s === _fs.symbol;
            return '<button data-sym="' + s + '" style="' +
                'display:flex; align-items:center; justify-content:space-between; width:100%; padding:6px 10px; border:none; border-radius:6px; cursor:pointer; transition:background 0.1s;' +
                'background:' + (isActive ? 'rgba(255,255,255,0.08)' : 'transparent') + '; color:#eaecef; font-size:12px;' +
                '" onmouseenter="this.style.background=\'rgba(255,255,255,0.1)\'" onmouseleave="this.style.background=\'' + (isActive ? 'rgba(255,255,255,0.08)' : 'transparent') + '\'">' +
                '<span style="font-family:ui-monospace,monospace; font-weight:700;">' + lbl + ' <span style="font-weight:400;color:#848e9c;font-size:11px;">' + name + '</span></span>' +
                '<span style="font-family:ui-monospace,monospace; font-size:11px; color:' + (up ? '#0ECB81' : '#F6465D') + ';">' + (up ? '+' : '') + chg.toFixed(1) + '%</span>' +
                '</button>';
        }).join('');
    }

    /* Toggle dropdown on ticker button click */
    var tickerBtn = header.querySelector('.cfs-ticker-btn');
    tickerBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        var vis = tickerMenu.style.display !== 'none';
        if (vis) {
            tickerMenu.style.display = 'none';
        } else {
            _fsPopulateTickerMenu();
            tickerMenu.style.display = 'block';
        }
    });

    /* Select a ticker from the dropdown */
    tickerMenu.addEventListener('click', function(e) {
        var btn = e.target.closest('[data-sym]');
        if (!btn) return;
        var newSym = btn.dataset.sym;
        if (newSym === _fs.symbol) { tickerMenu.style.display = 'none'; return; }

        var newBuilt = _fsBuildData(newSym, _fs.tfKey);
        if (!newBuilt || newBuilt.data.length === 0) {
            if (typeof showToast === 'function') showToast('No data for ' + newSym, 'error');
            return;
        }

        _fs.symbol = newSym;
        var newAd = newBuilt.ad;
        var newPrice = newAd.price || 0;
        var newChange = newAd.change || 0;
        var pc = newChange >= 0 ? '#0ECB81' : '#F6465D';
        var newName = (typeof marketData !== 'undefined' && marketData[newSym]) ? (marketData[newSym].name || newSym) : newSym;
        if (typeof getAssetName === 'function') newName = getAssetName(newSym, newName);

        /* Update header elements */
        var tEl = header.querySelector('.cfs-ticker-btn');
        if (tEl) {
            var symLbl = newSym.replace(/-USD/,'').replace(/=F/,'').replace(/=X/,'');
            var fullN = (typeof getAssetName === 'function') ? getAssetName(newSym, newName) : newName;
            var showN = fullN && fullN !== newSym && fullN !== symLbl;
            tEl.innerHTML = '<span class="cfs-ticker-sym">' + symLbl + '</span>' +
                (showN ? '<span class="cfs-ticker-name">' + fullN + '</span>' : '') +
                ' <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="vertical-align:middle;opacity:0.6;"><path d="M6 9l6 6 6-6"/></svg>';
        }
        var pEl = header.querySelector('.cfs-price');
        var cE = header.querySelector('.cfs-change');
        if (pEl) { pEl.textContent = _fsFmtPrice(newPrice); pEl.style.color = pc; }
        if (cE) { cE.textContent = (newChange >= 0 ? '+' : '') + newChange.toFixed(2) + '%'; cE.style.color = pc; }

        /* Rebuild chart */
        _fsCreateSeries(newBuilt.data, newBuilt.hasOHLC);

        /* Reset drawing state & load drawings for new symbol */
        _fs.drawStart = null;
        _fs.previewEnd = null;
        _fs.channelBase = null;
        _fsLoadDrawings();
        _fsRenderAnalysis();

        /* Restart live refresh for new ticker */
        _fsFetchAndUpdate();
        _fsStartLive();

        /* Update agent placeholder */
        var agentInp = document.getElementById('cfs-agent-input');
        if (agentInp) agentInp.placeholder = 'Ask about ' + newSym.replace(/-USD/,'').replace(/=F/,'') + '...';

        tickerMenu.style.display = 'none';
    });

    /* Close dropdown on outside click */
    root.addEventListener('click', function(e) {
        if (!tickerMenu.contains(e.target) && !tickerBtn.contains(e.target)) {
            tickerMenu.style.display = 'none';
        }
    });

    /* ── ZONE 2: Drawing toolbar (desktop sidebar / mobile toggle) ── */
    var _sz = 22; /* Icon size */
    var _s = ' viewBox="0 0 24 24" width="'+_sz+'" height="'+_sz+'" fill="none" stroke="currentColor" stroke-width="2">';
    var toolbarData = [
        /* Cursors */
        { id:'cursor',    icon:'<svg'+_s+'<path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/></svg>', tip:'Pointer (Esc)' },
        { id:'crosshair', icon:'<svg'+_s+'<line x1="12" y1="3" x2="12" y2="21"/><line x1="3" y1="12" x2="21" y2="12"/><circle cx="12" cy="12" r="2" stroke-width="1.5"/></svg>', tip:'Crosshair (C)', toggle:true },
        { id:'_div1', icon:'', tip:'' },
        /* Lines */
        { id:'trend',  icon:'<svg'+_s+'<line x1="4" y1="20" x2="20" y2="4"/></svg>', tip:'Trend Line' },
        { id:'ray',    icon:'<svg'+_s+'<line x1="4" y1="20" x2="20" y2="4"/><polyline points="16 4 20 4 20 8" stroke-width="1.5"/></svg>', tip:'Ray' },
        { id:'hline',  icon:'<svg'+_s+'<line x1="3" y1="12" x2="21" y2="12"/></svg>', tip:'Horizontal Line' },
        { id:'hray',   icon:'<svg'+_s+'<line x1="5" y1="12" x2="21" y2="12"/><circle cx="5" cy="12" r="2" fill="currentColor" stroke="none"/></svg>', tip:'Horizontal Ray' },
        { id:'vline',  icon:'<svg'+_s+'<line x1="12" y1="3" x2="12" y2="21"/></svg>', tip:'Vertical Line' },
        { id:'_div2', icon:'', tip:'' },
        /* Channels & Fib */
        { id:'channel', icon:'<svg'+_s+'<line x1="4" y1="18" x2="20" y2="8"/><line x1="4" y1="14" x2="20" y2="4" stroke-dasharray="3"/></svg>', tip:'Parallel Channel' },
        { id:'fib',    icon:'<svg'+_s+'<line x1="3" y1="5" x2="21" y2="5"/><line x1="3" y1="9" x2="21" y2="9" stroke-dasharray="3"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="16" x2="21" y2="16" stroke-dasharray="3"/><line x1="3" y1="19" x2="21" y2="19"/></svg>', tip:'Fibonacci' },
        { id:'_div3', icon:'', tip:'' },
        /* Shapes & Position */
        { id:'rect',     icon:'<svg'+_s+'<rect x="4" y="6" width="16" height="12" rx="1"/></svg>', tip:'Rectangle' },
        { id:'longpos',  icon:'<svg'+_s+'<line x1="4" y1="14" x2="20" y2="14" stroke="#888"/><line x1="4" y1="7" x2="20" y2="7" stroke="#0ECB81"/><line x1="4" y1="20" x2="20" y2="20" stroke="#F6465D"/><polyline points="10 10 12 7 14 10" stroke="#0ECB81" fill="none"/></svg>', tip:'Long Position' },
        { id:'shortpos', icon:'<svg'+_s+'<line x1="4" y1="10" x2="20" y2="10" stroke="#888"/><line x1="4" y1="17" x2="20" y2="17" stroke="#0ECB81"/><line x1="4" y1="4" x2="20" y2="4" stroke="#F6465D"/><polyline points="10 14 12 17 14 14" stroke="#F6465D" fill="none"/></svg>', tip:'Short Position' },
        { id:'measure',  icon:'<svg'+_s+'<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.4 2.4 0 0 1 0-3.4l2.6-2.6a2.4 2.4 0 0 1 3.4 0z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/></svg>', tip:'Measure' },
        { id:'text',     icon:'<svg'+_s+'<text x="7" y="18" font-size="16" font-weight="700" fill="currentColor" stroke="none">T</text></svg>', tip:'Text' },
        { id:'_div4', icon:'', tip:'' },
        /* Utility */
        { id:'magnet',     icon:'<svg'+_s+'<path d="M6 15V9a6 6 0 1 1 12 0v6"/><line x1="6" y1="12" x2="6" y2="15"/><line x1="18" y1="12" x2="18" y2="15"/><rect x="3" y="15" width="6" height="4" rx="1" fill="currentColor" stroke="none" opacity="0.3"/><rect x="15" y="15" width="6" height="4" rx="1" fill="currentColor" stroke="none" opacity="0.3"/></svg>', tip:'Magnet', toggle:true },
        { id:'eraser',    icon:'<svg'+_s+'<path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg>', tip:'Eraser' },
        { id:'trash',     icon:'<svg'+_s+'<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>', tip:'Trash All' },
        { id:'lock',      icon:'<svg'+_s+'<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>', tip:'Lock Drawings', toggle:true },
        { id:'hide',      icon:'<svg'+_s+'<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>', tip:'Hide/Show', toggle:true },
        { id:'screenshot', icon:'<svg'+_s+'<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>', tip:'Screenshot' },
    ];

    /* ── Tool shortcut + tooltip data ── */
    var toolShortcuts = {
        cursor:    { name:'Pointer',           desc:'Default cursor for panning and zooming',           key:'Esc' },
        crosshair: { name:'Crosshair',         desc:'Shows price and time at cursor position',          key:'C' },
        trend:     { name:'Trend Line',         desc:'Draw a line between two points',                   key:'T' },
        ray:       { name:'Ray',                desc:'Line extending infinitely in one direction',       key:'R' },
        hline:     { name:'Horizontal Line',    desc:'Mark a price level across the chart',              key:'H' },
        hray:      { name:'Horizontal Ray',     desc:'Price level extending right only',                 key:'Shift+H' },
        vline:     { name:'Vertical Line',      desc:'Mark a point in time',                             key:'V' },
        channel:   { name:'Parallel Channel',   desc:'Two parallel trend lines forming a range',         key:'P' },
        fib:       { name:'Fibonacci',          desc:'Draw retracement levels between two prices',       key:'F' },
        rect:      { name:'Rectangle',          desc:'Highlight a price/time zone',                      key:'Shift+R' },
        longpos:   { name:'Long Position',      desc:'Plan a trade with entry, TP, and SL',              key:'L' },
        shortpos:  { name:'Short Position',     desc:'Plan a short with entry, TP, and SL',              key:'Shift+L' },
        measure:   { name:'Measure',            desc:'Measure price change, %, and bars between points', key:'M' },
        text:      { name:'Text',               desc:'Place a text label on the chart',                  key:'X' },
        magnet:    { name:'Magnet',             desc:'Snap drawing points to candle OHLC values',        key:'G' },
        eraser:    { name:'Eraser',             desc:'Click any drawing to delete it',                   key:'E' },
        trash:     { name:'Trash All',          desc:'Remove all drawings from chart',                   key:'Shift+Del' },
        lock:      { name:'Lock Drawings',      desc:'Prevent drawings from being moved or deleted',     key:'K' },
        hide:      { name:'Hide/Show',          desc:'Toggle visibility of all drawings',                key:'Shift+K' },
        screenshot:{ name:'Screenshot',         desc:'Save chart as image',                              key:null },
    };

    if (!isMobile) {
        /* Desktop: left sidebar */
        var sidebar = document.createElement('div');
        sidebar.className = 'cfs-sidebar';
        toolbarData.forEach(function(t) {
            if (t.id.charAt(0) === '_') {
                /* Divider */
                var hr = document.createElement('hr');
                hr.className = 'cfs-divider';
                sidebar.appendChild(hr);
                return;
            }
            var btn = document.createElement('button');
            btn.className = 'cfs-tool' + (t.id === 'cursor' ? ' active' : '') + (t.id === 'crosshair' ? ' active' : '');
            btn.dataset.tool = t.id;
            btn.innerHTML = t.icon;
            sidebar.appendChild(btn);
        });
        root.appendChild(sidebar);
        _fs.elements.sidebar = sidebar;

        /* ── Desktop tooltip on hover ── */
        var tooltip = document.createElement('div');
        tooltip.className = 'cfs-tooltip';
        root.appendChild(tooltip);
        var _ttTimer = null;

        sidebar.addEventListener('mouseover', function(e) {
            var btn = e.target.closest('.cfs-tool');
            if (!btn) return;
            var tid = btn.dataset.tool;
            var info = toolShortcuts[tid];
            if (!info) return;
            clearTimeout(_ttTimer);
            _ttTimer = setTimeout(function() {
                var html = '<div class="cfs-tooltip-name">' + info.name;
                if (info.key) html += ' <span class="cfs-tooltip-key">' + info.key + '</span>';
                html += '</div><div class="cfs-tooltip-desc">' + info.desc + '</div>';
                tooltip.innerHTML = html;
                var rect = btn.getBoundingClientRect();
                var rootRect = root.getBoundingClientRect();
                tooltip.style.left = (rect.right - rootRect.left + 8) + 'px';
                tooltip.style.top = (rect.top - rootRect.top) + 'px';
                tooltip.classList.add('visible');
            }, 400);
        });
        sidebar.addEventListener('mouseout', function(e) {
            clearTimeout(_ttTimer);
            tooltip.classList.remove('visible');
        });
    } else {
        /* Mobile: Two-tier toolbar — hotbar (always visible) + full panel (drag up) */
        var hotbarTools = ['cursor','trend','hline','fib','measure','eraser'];
        var _hbSz = 20;
        var _hbS = ' viewBox="0 0 24 24" width="'+_hbSz+'" height="'+_hbSz+'" fill="none" stroke="currentColor" stroke-width="2">';

        /* ── Tier 1: Mini Hotbar (always visible, 48px) ── */
        var hotbar = document.createElement('div');
        hotbar.className = 'cfs-hotbar';
        hotbarTools.forEach(function(tid) {
            var tData = null;
            for (var i = 0; i < toolbarData.length; i++) { if (toolbarData[i].id === tid) { tData = toolbarData[i]; break; } }
            if (!tData) return;
            var btn = document.createElement('button');
            btn.className = 'hotbar-tool' + (tid === 'cursor' ? ' active' : '');
            btn.dataset.tool = tid;
            btn.innerHTML = tData.icon.replace(/width="\d+"/,'width="'+_hbSz+'"').replace(/height="\d+"/,'height="'+_hbSz+'"');
            hotbar.appendChild(btn);
        });
        /* Grip handle to open full panel */
        var grip = document.createElement('div');
        grip.className = 'hotbar-grip';
        grip.innerHTML = '<span></span><span></span><span></span>';
        hotbar.appendChild(grip);

        /* ── Tier 2: Full Panel (hidden, drag/tap to reveal) ── */
        var fullPanel = document.createElement('div');
        fullPanel.className = 'cfs-full-panel';
        /* Panel grip handle at top */
        var panelGrip = document.createElement('div');
        panelGrip.className = 'cfs-panel-grip';
        panelGrip.innerHTML = '<span></span>';
        fullPanel.appendChild(panelGrip);
        /* 4-column grid with ALL tools */
        var panelGrid = document.createElement('div');
        panelGrid.className = 'cfs-panel-grid';
        var toolLabels = { cursor:'Pointer', crosshair:'Crosshair', trend:'Trend', ray:'Ray',
            hline:'H-Line', hray:'H-Ray', vline:'V-Line', channel:'Channel',
            fib:'Fib', rect:'Rect', longpos:'Long', shortpos:'Short',
            measure:'Measure', text:'Text', magnet:'Magnet', eraser:'Eraser',
            trash:'Trash', lock:'Lock', hide:'Hide', screenshot:'Screenshot' };
        toolbarData.forEach(function(t) {
            if (t.id.charAt(0) === '_') return; /* skip dividers */
            var btn = document.createElement('button');
            btn.className = 'cfs-tool' + (t.id === 'cursor' ? ' active' : '') + (t.id === 'crosshair' ? ' active' : '');
            btn.dataset.tool = t.id;
            btn.innerHTML = t.icon + '<span class="cfs-tool-label">' + (toolLabels[t.id] || t.tip) + '</span>';
            panelGrid.appendChild(btn);
        });
        /* Done button in grid */
        var doneBtn = document.createElement('button');
        doneBtn.className = 'cfs-tool cfs-done-btn';
        doneBtn.dataset.tool = 'done';
        doneBtn.innerHTML = '<svg viewBox="0 0 24 24" width="'+_sz+'" height="'+_sz+'" fill="none" stroke="#0ECB81" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg><span class="cfs-tool-label" style="color:#0ECB81">Done</span>';
        panelGrid.appendChild(doneBtn);
        fullPanel.appendChild(panelGrid);

        _fs.elements.hotbar = hotbar;
        _fs.elements.fullPanel = fullPanel;
        _fs.elements.doneBtn = doneBtn;
        _fs.elements.panelGrip = panelGrip;
        _fs.elements.hotbarGrip = grip;
    }

    /* ── ZONE 3: Chart area ── */
    var chartWrap = document.createElement('div');
    chartWrap.className = 'cfs-chart-area' + (isMobile ? '' : ' with-sidebar');

    /* OHLC data strip */
    var ohlcStrip = document.createElement('div');
    ohlcStrip.className = 'cfs-ohlc-strip';
    ohlcStrip.innerHTML = '<span class="cfs-ohlc-label">O <span id="cfs-o">--</span> H <span id="cfs-h">--</span> L <span id="cfs-l">--</span> C <span id="cfs-c">--</span> <span id="cfs-chg"></span></span>';
    chartWrap.appendChild(ohlcStrip);

    /* MA legend */
    var maLegend = document.createElement('div');
    maLegend.className = 'cfs-ma-legend';
    maLegend.innerHTML = '<span style="color:#F0B90B">MA10 <b id="cfs-ma10">--</b></span> ' +
        '<span style="color:#F6465D">MA20 <b id="cfs-ma20">--</b></span> ' +
        '<span style="color:#9B59B6">MA50 <b id="cfs-ma50">--</b></span> ' +
        '<span style="color:#26A69A">MA200 <b id="cfs-ma200">--</b></span>';
    chartWrap.appendChild(maLegend);

    /* Chart container */
    var chartEl = document.createElement('div');
    chartEl.className = 'cfs-chart-container';
    chartWrap.appendChild(chartEl);

    /* ── ZONE 4: Bottom Intel Panel (Analysis + Agent) ── */
    var intelPanel = document.createElement('div');
    intelPanel.className = 'cfs-intel-panel';
    var savedIntelH = parseInt(localStorage.getItem('cfsIntelHeight')) || 0;
    intelPanel.style.height = savedIntelH > 28 ? savedIntelH + 'px' : '28px';

    var intelGrip = document.createElement('div');
    intelGrip.className = 'cfs-intel-grip';
    intelGrip.innerHTML = '<span class="cfs-intel-grip-bar"></span>';
    intelPanel.appendChild(intelGrip);

    var intelBody = document.createElement('div');
    intelBody.className = 'cfs-intel-body';
    intelBody.style.display = savedIntelH > 28 ? '' : 'none';

    /* Left column: Analysis */
    var intelAnalysis = document.createElement('div');
    intelAnalysis.className = 'cfs-intel-col';
    intelAnalysis.id = 'cfs-intel-analysis';
    intelAnalysis.innerHTML = '<h4><span class="dot"></span>Analysis</h4><div class="cfs-intel-metrics"></div>';

    /* Right column: Strat Market Agent */
    var intelAgent = document.createElement('div');
    intelAgent.className = 'cfs-intel-col';
    intelAgent.id = 'cfs-intel-agent';
    intelAgent.innerHTML = '<h4><span class="dot" style="background:#3b82f6"></span>Strat Market</h4>' +
        '<div class="cfs-intel-agent-msgs" id="cfs-agent-msgs"></div>' +
        '<div class="cfs-intel-agent-input">' +
            '<input type="text" id="cfs-agent-input" placeholder="Ask about ' + symbol.replace(/-USD/,'').replace(/=F/,'') + '...">' +
            '<button id="cfs-agent-send"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg></button>' +
        '</div>';

    intelBody.appendChild(intelAnalysis);
    intelBody.appendChild(intelAgent);
    intelPanel.appendChild(intelBody);
    chartWrap.appendChild(intelPanel);

    /* Store references */
    _fs.elements.intelPanel = intelPanel;
    _fs.elements.intelBody = intelBody;
    _fs.elements.intelGrip = intelGrip;

    /* ── Intel Panel: Drag-to-resize ── */
    (function() {
        var _ipDragging = false, _ipStartY = 0, _ipStartH = 0;
        var MIN_H = 28, MAX_H_RATIO = 0.5;

        function _ipStart(y) {
            _ipDragging = true;
            _ipStartY = y;
            _ipStartH = intelPanel.offsetHeight;
            intelPanel.style.transition = 'none';
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';
        }
        function _ipMove(y) {
            if (!_ipDragging) return;
            var dy = _ipStartY - y; /* drag up = increase height */
            var maxH = Math.floor(root.clientHeight * MAX_H_RATIO);
            var newH = Math.min(maxH, Math.max(MIN_H, _ipStartH + dy));
            intelPanel.style.height = newH + 'px';
            intelBody.style.display = newH > MIN_H + 4 ? '' : 'none';
        }
        function _ipEnd() {
            if (!_ipDragging) return;
            _ipDragging = false;
            intelPanel.style.transition = '';
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            var h = intelPanel.offsetHeight;
            localStorage.setItem('cfsIntelHeight', h);
            intelBody.style.display = h > MIN_H + 4 ? '' : 'none';
            if (h > MIN_H + 4) _fsRenderAnalysis();
        }

        /* Mouse */
        intelGrip.addEventListener('mousedown', function(e) {
            e.preventDefault();
            _ipStart(e.clientY);
        });
        document.addEventListener('mousemove', function(e) { _ipMove(e.clientY); });
        document.addEventListener('mouseup', _ipEnd);

        /* Touch */
        intelGrip.addEventListener('touchstart', function(e) {
            _ipStart(e.touches[0].clientY);
        }, { passive: true });
        document.addEventListener('touchmove', function(e) {
            if (_ipDragging) _ipMove(e.touches[0].clientY);
        }, { passive: true });
        document.addEventListener('touchend', _ipEnd);

        /* Double-click toggle */
        intelGrip.addEventListener('dblclick', function() {
            var h = intelPanel.offsetHeight;
            var targetH = h > MIN_H + 10 ? MIN_H : 240;
            intelPanel.style.height = targetH + 'px';
            intelBody.style.display = targetH > MIN_H + 4 ? '' : 'none';
            localStorage.setItem('cfsIntelHeight', targetH);
            if (targetH > MIN_H + 4) _fsRenderAnalysis();
        });
    })();

    /* ── Intel Panel: Analysis Renderer ── */
    function _fsRenderAnalysis() {
        var el = document.getElementById('cfs-intel-analysis');
        if (!el) return;
        var metricsEl = el.querySelector('.cfs-intel-metrics');
        if (!metricsEl) return;

        var sym = _fs.symbol;
        var ad = (typeof _resolveData === 'function') ? _resolveData(sym, _fs.tfKey) : null;
        if (!ad || !ad.history || ad.history.length < 5) {
            metricsEl.innerHTML = '<div style="font-size:10px;color:#64748b;padding:8px 0;">No data available</div>';
            return;
        }

        var hist = ad.history;
        var highs = ad.highs || hist;
        var lows = ad.lows || hist;
        var price = ad.price || hist[hist.length - 1];
        var prevClose = hist.length > 1 ? hist[hist.length - 2] : price;
        var chg = prevClose ? ((price - prevClose) / prevClose * 100) : 0;
        var isUp = chg >= 0;
        var fp = function(v) { return v != null ? v.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}) : '--'; };

        /* Momentum (rate of change last 14 vs 14 before that) */
        var momentum = 0, momentumLabel = 'Neutral';
        if (hist.length >= 28) {
            var recent = hist[hist.length-1] / hist[hist.length-14] - 1;
            var prior = hist[hist.length-14] / hist[hist.length-28] - 1;
            momentum = recent - prior;
            if (momentum > 0.02) momentumLabel = 'Accelerating';
            else if (momentum > 0.005) momentumLabel = 'Building';
            else if (momentum < -0.02) momentumLabel = 'Decelerating';
            else if (momentum < -0.005) momentumLabel = 'Fading';
        }

        /* Volatility (std dev of % returns, last 20 bars) */
        var volatility = 0;
        var volWindow = Math.min(20, hist.length - 1);
        if (volWindow >= 5) {
            var returns = [];
            for (var i = hist.length - volWindow; i < hist.length; i++) {
                if (hist[i-1] > 0) returns.push((hist[i] - hist[i-1]) / hist[i-1]);
            }
            var mean = returns.reduce(function(a,b){return a+b;}, 0) / returns.length;
            volatility = Math.sqrt(returns.reduce(function(a,r){return a + Math.pow(r-mean,2);}, 0) / returns.length) * 100;
        }
        var volLabel = 'Low';
        if (volatility > 3) volLabel = 'Extreme';
        else if (volatility > 1.5) volLabel = 'High';
        else if (volatility > 0.5) volLabel = 'Moderate';

        /* Support / Resistance (recent swing highs/lows) */
        var support = null, resistance = null;
        if (hist.length >= 10) {
            var lookback = Math.min(60, hist.length);
            var recentHi = highs.slice(-lookback);
            var recentLo = lows.slice(-lookback);
            var swingHighs = [], swingLows = [];
            for (var j = 2; j < recentHi.length - 2; j++) {
                if (recentHi[j] > recentHi[j-1] && recentHi[j] > recentHi[j-2] && recentHi[j] > recentHi[j+1] && recentHi[j] > recentHi[j+2])
                    swingHighs.push(recentHi[j]);
                if (recentLo[j] < recentLo[j-1] && recentLo[j] < recentLo[j-2] && recentLo[j] < recentLo[j+1] && recentLo[j] < recentLo[j+2])
                    swingLows.push(recentLo[j]);
            }
            if (swingHighs.length) resistance = swingHighs.filter(function(h){return h > price;}).sort(function(a,b){return a-b;})[0] || Math.max.apply(null, swingHighs);
            if (swingLows.length) support = swingLows.filter(function(l){return l < price;}).sort(function(a,b){return b-a;})[0] || Math.min.apply(null, swingLows);
        }

        /* Range position */
        var pHigh = ad.high || Math.max.apply(null, highs.slice(-60));
        var pLow = ad.low || Math.min.apply(null, lows.slice(-60));
        var rangePos = (pHigh > pLow) ? ((price - pLow) / (pHigh - pLow) * 100) : 50;
        var distHigh = pHigh > 0 ? ((price - pHigh) / pHigh * 100) : 0;
        var distLow = pLow > 0 ? ((price - pLow) / pLow * 100) : 0;

        /* Trend */
        var mag = Math.abs(chg);
        var trend = 'Flat';
        if (mag > 5) trend = isUp ? 'Surging' : 'Plunging';
        else if (mag > 2) trend = isUp ? 'Rising' : 'Falling';
        else if (mag > 0.5) trend = isUp ? 'Up' : 'Down';

        /* Fib levels from drawings (if any) */
        var fibHtml = '';
        if (_fs.drawings && _fs.drawings.length) {
            var fibDrawing = null;
            for (var fi = _fs.drawings.length - 1; fi >= 0; fi--) {
                if (_fs.drawings[fi].type === 'fib') { fibDrawing = _fs.drawings[fi]; break; }
            }
            if (fibDrawing) {
                var fibHigh = Math.max(fibDrawing.startPrice, fibDrawing.endPrice);
                var fibLow = Math.min(fibDrawing.startPrice, fibDrawing.endPrice);
                var fibRange = fibHigh - fibLow;
                var fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                fibHtml = '<div style="margin-top:6px;"><div style="font-size:9px;font-weight:700;color:#848e9c;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Fibonacci</div>';
                fibLevels.forEach(function(lv) {
                    var val = fibHigh - fibRange * lv;
                    var clr = price > val ? '#10b981' : '#ef4444';
                    if (Math.abs(price - val) / price < 0.005) clr = '#f0b90b';
                    fibHtml += '<div class="cfs-intel-metric"><span class="label">' + (lv * 100).toFixed(1) + '%</span><span class="value" style="color:' + clr + '">$' + fp(val) + '</span></div>';
                });
                fibHtml += '</div>';
            }
        }

        var momColor = momentum > 0.005 ? '#10b981' : momentum < -0.005 ? '#ef4444' : '#94a3b8';
        var volColor = volatility > 1.5 ? '#f59e0b' : volatility > 0.5 ? '#eab308' : '#94a3b8';
        var trendColor = isUp ? '#10b981' : '#ef4444';

        metricsEl.innerHTML =
            '<div class="cfs-intel-metric"><span class="label">Trend</span><span class="value" style="color:' + trendColor + '">' + trend + ' (' + (isUp?'+':'') + chg.toFixed(2) + '%)</span></div>' +
            '<div class="cfs-intel-metric"><span class="label">Momentum</span><span class="value" style="color:' + momColor + '">' + momentumLabel + '</span></div>' +
            '<div class="cfs-intel-metric"><span class="label">Volatility</span><span class="value" style="color:' + volColor + '">' + volLabel + ' (' + volatility.toFixed(2) + '%)</span></div>' +
            (support != null ? '<div class="cfs-intel-metric"><span class="label">Support</span><span class="value" style="color:#10b981">$' + fp(support) + '</span></div>' : '') +
            (resistance != null ? '<div class="cfs-intel-metric"><span class="label">Resistance</span><span class="value" style="color:#ef4444">$' + fp(resistance) + '</span></div>' : '') +
            '<div class="cfs-intel-metric"><span class="label">From High</span><span class="value" style="color:' + (distHigh >= 0 ? '#10b981' : '#ef4444') + '">' + (distHigh >= 0 ? '+' : '') + distHigh.toFixed(2) + '%</span></div>' +
            '<div class="cfs-intel-metric"><span class="label">From Low</span><span class="value" style="color:' + (distLow >= 0 ? '#10b981' : '#ef4444') + '">' + (distLow >= 0 ? '+' : '') + distLow.toFixed(2) + '%</span></div>' +
            '<div style="margin-top:4px;">' +
                '<div style="display:flex;justify-content:space-between;font-size:9px;color:#64748b;margin-bottom:2px;"><span>$' + fp(pLow) + '</span><span style="color:#94a3b8;font-weight:600;">' + rangePos.toFixed(0) + '%</span><span>$' + fp(pHigh) + '</span></div>' +
                '<div class="cfs-intel-range-bar"><div class="cfs-intel-range-fill"></div><div class="cfs-intel-range-dot" style="left:' + Math.min(99,Math.max(1,rangePos)) + '%;"></div></div>' +
            '</div>' +
            fibHtml;
    }

    /* ── Intel Panel: Agent Chat ── */
    var _cfsAgentHistory = [];
    var _cfsAgentStreaming = false;

    function _cfsAgentAppend(role, content, elId) {
        var msgs = document.getElementById('cfs-agent-msgs');
        if (!msgs) return null;
        var div = document.createElement('div');
        if (elId) div.id = elId;
        var time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

        if (role === 'user') {
            div.className = 'flex justify-end';
            div.innerHTML = '<div style="max-width:85%;"><div style="border-radius:12px 12px 2px 12px;padding:5px 8px;font-size:10px;line-height:1.4;color:#e2e8f0;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.15);">' + _cfsEsc(content) + '</div></div>';
        } else if (role === 'typing') {
            div.innerHTML = '<div style="display:flex;align-items:center;gap:4px;padding:4px 0;"><span style="width:5px;height:5px;border-radius:50%;background:#34d399;animation:bounce 1s infinite;"></span><span style="width:5px;height:5px;border-radius:50%;background:#34d399;animation:bounce 1s infinite 0.15s;"></span><span style="width:5px;height:5px;border-radius:50%;background:#34d399;animation:bounce 1s infinite 0.3s;"></span><span style="font-size:9px;color:#64748b;margin-left:4px;">Analyzing...</span></div>';
        } else {
            div.innerHTML = '<div style="font-size:10px;line-height:1.5;color:#cbd5e1;">' + content + '</div>';
        }
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
        return div;
    }

    function _cfsEsc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

    function _cfsSendAgent() {
        if (_cfsAgentStreaming) return;
        var input = document.getElementById('cfs-agent-input');
        var msg = input ? input.value.trim() : '';
        if (!msg) return;
        input.value = '';

        _cfsAgentAppend('user', msg);
        _cfsAgentHistory.push({role:'user', content: msg});

        var typingId = 'cfs-typing-' + Date.now();
        _cfsAgentAppend('typing', '', typingId);
        _cfsAgentStreaming = true;

        /* Build context with current ticker data */
        var sym = _fs ? _fs.symbol : '';
        var ad = (typeof _resolveData === 'function') ? _resolveData(sym, _fs ? _fs.tfKey : '1m') : null;
        var ctx = '';
        if (ad && ad.price) {
            var clean = sym.replace(/-USD/,'').replace(/=F/,'').replace(/=X/,'');
            var name = (typeof getAssetName === 'function') ? getAssetName(sym) : clean;
            ctx = '\n[Context: Viewing ' + clean + ' (' + name + ') at $' + (ad.price||0).toFixed(2) + ', change ' + (ad.change||0).toFixed(2) + '%';
            if (ad.high) ctx += ', H:$' + ad.high.toFixed(2);
            if (ad.low) ctx += ', L:$' + ad.low.toFixed(2);
            ctx += ']';
        }
        var contextMsg = msg + ctx + '\n[Instructions: Be concise and actionable. Max 3-4 bullet points. Focus on the specific ticker being viewed.]';

        fetch('/api/agent-chat', {
            method:'POST',
            headers:{
                'Content-Type':'application/json',
                'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
                'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
            },
            body: JSON.stringify({ message: contextMsg, history: _cfsAgentHistory.slice(-12) })
        }).then(function(res) {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            var reader = res.body.getReader();
            var decoder = new TextDecoder();
            var full = '';
            var typing = document.getElementById(typingId);
            if (typing) typing.remove();
            var respEl = _cfsAgentAppend('assistant', '');

            function pump() {
                return reader.read().then(function(result) {
                    if (result.done) {
                        full = full.replace(/[\u4e00-\u9fff\u3400-\u4dbf]+/g, '').replace(/\s{2,}/g, ' ').trim();
                        _cfsAgentHistory.push({role:'assistant', content: full});
                        if (respEl && typeof formatAgentText === 'function') {
                            respEl.querySelector('div').innerHTML = formatAgentText(full);
                        }
                        _cfsAgentStreaming = false;
                        var sendBtn = document.getElementById('cfs-agent-send');
                        if (sendBtn) sendBtn.disabled = false;
                        var inp = document.getElementById('cfs-agent-input');
                        if (inp) { inp.disabled = false; inp.focus(); }
                        return;
                    }
                    var chunk = decoder.decode(result.value, {stream:true});
                    chunk.split('\n').forEach(function(line) {
                        if (!line.startsWith('data: ')) return;
                        var raw = line.slice(6);
                        if (raw === '[DONE]') return;
                        try {
                            var payload = JSON.parse(raw);
                            if (payload.token) {
                                full += payload.token;
                                if (respEl) respEl.querySelector('div').textContent = full;
                            }
                        } catch(e){}
                    });
                    return pump();
                });
            }
            return pump();
        }).catch(function(err) {
            var typing = document.getElementById(typingId);
            if (typing) typing.remove();
            _cfsAgentAppend('assistant', '<span style="color:#f59e0b;">Error: ' + _cfsEsc(err.message) + '</span>');
            _cfsAgentStreaming = false;
        });
    }

    /* Wire agent send button and Enter key (use querySelector on detached DOM, not getElementById) */
    var agentSendBtn = intelAgent.querySelector('#cfs-agent-send');
    var agentInput = intelAgent.querySelector('#cfs-agent-input');
    if (agentSendBtn) agentSendBtn.addEventListener('click', _cfsSendAgent);
    if (agentInput) agentInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); _cfsSendAgent(); }
    });

    /* Initial analysis render if panel is expanded */
    if (savedIntelH > 28) setTimeout(_fsRenderAnalysis, 100);

    root.appendChild(chartWrap);

    /* Append mobile hotbar + full panel */
    if (isMobile) {
        document.body.appendChild(_fs.elements.hotbar);
        document.body.appendChild(_fs.elements.fullPanel);
        /* Reserve space for hotbar at bottom */
        chartWrap.style.paddingBottom = '48px';
    }

    /* ── Create Lightweight Charts instance ── */
    var fsChart = LightweightCharts.createChart(chartEl, {
        width: chartEl.clientWidth,
        height: chartEl.clientHeight,
        layout: { background: {type:'solid', color:'transparent'}, textColor: '#848e9c', fontSize: 11, fontFamily: "'ui-monospace','SFMono-Regular','Menlo',monospace" },
        grid: { vertLines: {color:'rgba(42,46,57,0.6)'}, horzLines: {color:'rgba(42,46,57,0.6)'} },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal, vertLine: {color:'rgba(255,255,255,0.2)', style:3}, horzLine: {color:'rgba(255,255,255,0.2)', style:3} },
        rightPriceScale: { borderColor: '#2a2e39', scaleMargins: {top:0.1, bottom:0.08}, entireTextOnly: true },
        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false, rightOffset: 5, minBarSpacing: isMobile ? 3 : 2 },
        handleScroll: { mouseWheel:true, pressedMouseMove:true, horzTouchDrag:true, vertTouchDrag:true },
        handleScale: { axisPressedMouseMove:true, mouseWheel:true, pinch:true },
    });
    _fs.chart = fsChart;

    /* ── MA refresh (reusable — called on create AND live tick) ── */
    var _fsMaConfigs = [
        { period:10,  color:'#F0B90B' },
        { period:20,  color:'#F6465D' },
        { period:50,  color:'#9B59B6' },
        { period:200, color:'#26A69A' },
    ];
    function _fsRefreshMAs(data) {
        var closes = data.map(function(d) { return d.close != null ? d.close : d.value; });
        var times = data.map(function(d) { return d.time; });
        var maIdx = 0;
        _fsMaConfigs.forEach(function(mc) {
            if (closes.length < mc.period) return;
            var smaVals = _calcSMA(closes, mc.period);
            var maData = [];
            for (var i = 0; i < smaVals.length; i++) {
                if (smaVals[i] != null) maData.push({ time: times[i], value: smaVals[i] });
            }
            if (maData.length > 0) {
                if (maIdx < _fs.maSeries.length) {
                    _fs.maSeries[maIdx].setData(maData);
                } else {
                    var maSeries = fsChart.addLineSeries({
                        color: mc.color, lineWidth: 1,
                        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
                    });
                    maSeries.setData(maData);
                    _fs.maSeries.push(maSeries);
                }
                maIdx++;
            }
        });
    }

    /* ── Create candle series ── */
    function _fsCreateSeries(data, hasOHLC) {
        /* Remove old series */
        if (_fs.series) { try { fsChart.removeSeries(_fs.series); } catch(e){} }
        _fs.maSeries.forEach(function(s) { try { fsChart.removeSeries(s); } catch(e){} });
        _fs.maSeries = [];

        if (hasOHLC && data.length > 0 && data[0].open != null) {
            _fs.series = fsChart.addCandlestickSeries({
                upColor:'#0ECB81', downColor:'#F6465D',
                borderUpColor:'#0ECB81', borderDownColor:'#F6465D',
                wickUpColor:'#0ECB8180', wickDownColor:'#F6465D80',
            });
        } else {
            _fs.series = fsChart.addLineSeries({
                color: change >= 0 ? '#0ECB81' : '#F6465D',
                lineWidth: 2, crosshairMarkerVisible: true,
            });
        }
        _fs.series.setData(data);
        _fs.chartData = data;  /* Store for magnet OHLC snapping */

        _fsRefreshMAs(data);

        /* Calculate visible bars — enough for readable candles, not all data */
        if (data.length > 2) {
            var n = 80;
            var tf = _fs ? _fs.tfKey : '';
            if (tf === '1m') { n = isMobile ? 100 : 200; }      /* ~2-3 hours */
            else if (tf === '5m') { n = isMobile ? 60 : 100; }   /* ~5-8 hours */
            else if (tf === '1d_1mo') { n = 60; }
            else if (tf === '1d_1y') { n = 120; }
            else if (tf === '1wk') { n = 100; }
            if (n > data.length) n = data.length;
            fsChart.timeScale().setVisibleLogicalRange({ from: data.length - n, to: data.length });
        } else {
            fsChart.timeScale().fitContent();
        }
    }

    _fsCreateSeries(built.data, built.hasOHLC);

    /* ── OHLC + MA Crosshair tooltip ── */
    _fs.crosshairSub = fsChart.subscribeCrosshairMove(function(param) {
        var oEl = document.getElementById('cfs-o');
        var hEl = document.getElementById('cfs-h');
        var lEl = document.getElementById('cfs-l');
        var cEl = document.getElementById('cfs-c');
        var chgEl = document.getElementById('cfs-chg');
        if (!oEl) return;

        var candleVal = param.seriesData ? param.seriesData.get(_fs.series) : null;
        if (candleVal && candleVal.open != null) {
            oEl.textContent = _fsFmtPrice(candleVal.open);
            hEl.textContent = _fsFmtPrice(candleVal.high);
            lEl.textContent = _fsFmtPrice(candleVal.low);
            cEl.textContent = _fsFmtPrice(candleVal.close);
            var diff = candleVal.close - candleVal.open;
            var pct = candleVal.open ? ((diff / candleVal.open) * 100) : 0;
            var col = diff >= 0 ? '#0ECB81' : '#F6465D';
            chgEl.innerHTML = '<span style="color:' + col + '">' + (diff >= 0 ? '+' : '') + _fsFmtPrice(diff) + ' (' + pct.toFixed(2) + '%)</span>';
        } else if (candleVal && candleVal.value != null) {
            oEl.textContent = '--'; hEl.textContent = '--'; lEl.textContent = '--';
            cEl.textContent = _fsFmtPrice(candleVal.value);
            chgEl.textContent = '';
        }

        /* Update MA legend */
        _fs.maSeries.forEach(function(s, idx) {
            var maVal = param.seriesData ? param.seriesData.get(s) : null;
            var elId = ['cfs-ma10','cfs-ma20','cfs-ma50','cfs-ma200'][idx];
            var el = document.getElementById(elId);
            if (el) el.textContent = (maVal && maVal.value != null) ? _fsFmtPrice(maVal.value) : '--';
        });
    });

    /* ── Timeframe switching ── */
    header.addEventListener('click', function(e) {
        var btn = e.target.closest('.cfs-tf');
        if (!btn) return;
        var newTf = btn.dataset.tf;
        if (newTf === _fs.tfKey) return;

        var newBuilt = _fsBuildData(_fs.symbol, newTf);
        if (!newBuilt || newBuilt.data.length === 0) {
            if (typeof showToast === 'function') showToast('No data for this timeframe', 'error');
            return;
        }

        _fs.tfKey = newTf;
        /* Update active button */
        header.querySelectorAll('.cfs-tf').forEach(function(b) {
            b.classList.toggle('active', b.dataset.tf === newTf);
        });

        /* Update price in header */
        var newAd = newBuilt.ad;
        var newPrice = newAd.price || 0;
        var newChange = newAd.change || 0;
        var pc = newChange >= 0 ? '#0ECB81' : '#F6465D';
        var pEl = header.querySelector('.cfs-price');
        var cE = header.querySelector('.cfs-change');
        if (pEl) { pEl.textContent = _fsFmtPrice(newPrice); pEl.style.color = pc; }
        if (cE) { cE.textContent = (newChange >= 0 ? '+' : '') + newChange.toFixed(2) + '%'; cE.style.color = pc; }

        /* Rebuild chart series + MAs */
        _fsCreateSeries(newBuilt.data, newBuilt.hasOHLC);
        /* Cancel in-progress drawing on tf change, load saved drawings for new tf */
        _fs.drawStart = null;
        _fs.previewEnd = null;
        _fs.channelBase = null;
        _fs.selectedIdx = -1;
        _fs.dragging = false;
        _fsLoadDrawings();
        _fsRedrawCanvas();
        _fsRenderAnalysis();
        /* Restart live auto-refresh for new timeframe */
        _fsStartLive();
    });

    /* ── Drawing tools — click-click interaction ── */
    /* Create canvas overlay for drawings + preview */
    var drawCanvas = document.createElement('canvas');
    drawCanvas.className = 'cfs-draw-canvas';
    chartEl.appendChild(drawCanvas);
    _fs.drawCanvas = drawCanvas;
    _fs.drawCtx = drawCanvas.getContext('2d');
    _fs.drawStart = null;  /* First click anchor for two-point tools */
    _fs.previewEnd = null; /* Current mouse/touch position for rubber-band */

    function _fsResizeCanvas() {
        drawCanvas.width = chartEl.clientWidth;
        drawCanvas.height = chartEl.clientHeight;
        _fsRedrawCanvas();
    }

    /* Get chart coordinates from a mouse/touch event */
    function _fsGetChartCoords(e) {
        var rect = chartEl.getBoundingClientRect();
        var cx, cy;
        if (e.touches && e.touches.length > 0) { cx = e.touches[0].clientX; cy = e.touches[0].clientY; }
        else if (e.changedTouches && e.changedTouches.length > 0) { cx = e.changedTouches[0].clientX; cy = e.changedTouches[0].clientY; }
        else { cx = e.clientX; cy = e.clientY; }
        var x = cx - rect.left, y = cy - rect.top;
        var price = _fs.series ? _fs.series.coordinateToPrice(y) : null;
        var timeIdx = fsChart.timeScale().coordinateToLogical(x);

        /* Magnet: snap to nearest candle OHLC value */
        if (_fs.magnetOn && _fs.chartData && _fs.series && price != null && timeIdx != null) {
            var idx = Math.round(timeIdx);
            if (idx >= 0 && idx < _fs.chartData.length) {
                var candle = _fs.chartData[idx];
                if (candle && candle.open != null) {
                    var ohlc = [candle.open, candle.high, candle.low, candle.close];
                    var bestPrice = ohlc[0], bestDist = Math.abs(price - ohlc[0]);
                    for (var i = 1; i < ohlc.length; i++) {
                        var d = Math.abs(price - ohlc[i]);
                        if (d < bestDist) { bestDist = d; bestPrice = ohlc[i]; }
                    }
                    price = bestPrice;
                    var snapY = _fs.series.priceToCoordinate(bestPrice);
                    if (snapY != null) y = snapY;
                }
                timeIdx = idx;  /* Snap to exact candle index */
                var snapX = fsChart.timeScale().logicalToCoordinate(idx);
                if (snapX != null) x = snapX;
            }
        }

        return { x:x, y:y, price:price, timeIdx:timeIdx };
    }

    /* ── Tool selection ── */
    function _fsSelectTool(toolId) {
        /* Cancel any in-progress drawing */
        _fs.drawStart = null;
        _fs.previewEnd = null;
        _fs.channelBase = null;

        /* Toggle tools: crosshair, magnet, lock, hide */
        if (toolId === 'crosshair') {
            _fs.crosshairOn = !_fs.crosshairOn;
            _fsUpdateToggleBtn('crosshair', _fs.crosshairOn);
            fsChart.applyOptions({ crosshair: { mode: _fs.crosshairOn ? LightweightCharts.CrosshairMode.Normal : LightweightCharts.CrosshairMode.Hidden } });
            return;
        }
        if (toolId === 'magnet') {
            _fs.magnetOn = !_fs.magnetOn;
            _fsUpdateToggleBtn('magnet', _fs.magnetOn);
            return;
        }
        if (toolId === 'lock') {
            _fs.lockOn = !_fs.lockOn;
            _fsUpdateToggleBtn('lock', _fs.lockOn);
            return;
        }
        if (toolId === 'hide') {
            _fs.hideOn = !_fs.hideOn;
            _fsUpdateToggleBtn('hide', _fs.hideOn);
            _fsRedrawCanvas();
            return;
        }
        if (toolId === 'trash') {
            if (_fs.drawings.length === 0) return;
            stratosConfirm('Delete all drawings on this chart?', { title: 'Clear Drawings', okText: 'Delete All', cancelText: 'Cancel' }).then(ok => {
                if (!ok) return;
                _fs.drawings = [];
                _fs.selectedIdx = -1;
                _fsSaveDrawings();
                _fsRedrawCanvas();
            });
            return;
        }
        if (toolId === 'screenshot') {
            _fsScreenshot();
            return;
        }
        if (toolId === 'cursor') {
            _fs.drawMode = null;
            _fs.selectedIdx = -1;
            _fs.dragging = false;
        } else {
            _fs.drawMode = toolId;
            _fs.selectedIdx = -1;
            _fs.dragging = false;
        }
        /* Update active state on drawing tool buttons (skip toggles) */
        var activeId = _fs.drawMode || 'cursor';
        root.querySelectorAll('.cfs-tool').forEach(function(b) {
            if (b.classList.contains('cfs-done-btn')) return;
            var t = b.dataset.tool;
            if (t === 'crosshair' || t === 'magnet' || t === 'lock' || t === 'hide') return;
            b.classList.toggle('active', t === activeId);
        });
        /* Sync hotbar active state (mobile) */
        if (isMobile && _fs.elements.hotbar) {
            _fs.elements.hotbar.querySelectorAll('.hotbar-tool').forEach(function(b) {
                b.classList.toggle('active', b.dataset.tool === activeId);
            });
        }
        _fsUpdateCanvasPointer();
        _fsRedrawCanvas();
    }

    function _fsUpdateToggleBtn(toolId, isOn) {
        root.querySelectorAll('.cfs-tool').forEach(function(b) {
            if (b.dataset.tool === toolId) b.classList.toggle('active', isOn);
        });
    }

    /* Toggle canvas pointer events + cursor style.
       Canvas always has pointer-events:none so the chart underneath handles scroll/zoom/pan.
       We capture click events on the chart container instead. */
    function _fsUpdateCanvasPointer() {
        drawCanvas.style.pointerEvents = 'none';
        drawCanvas.style.cursor = 'default';
    }
    drawCanvas.style.pointerEvents = 'none';

    /* Attach tool click handlers */
    root.querySelectorAll('.cfs-tool').forEach(function(btn) {
        if (btn.classList.contains('cfs-done-btn')) return;
        btn.onclick = function(e) { e.stopPropagation(); _fsSelectTool(btn.dataset.tool); };
    });

    /* ── Mobile two-tier toolbar: hotbar + full panel ── */
    if (isMobile && _fs.elements.hotbar) {
        var _panelOpen = false;
        var _panelLocked = false;
        var _panelStartY = 0;
        var _panelCurrentY = 0;
        var _panelStartTime = 0;
        var _panelDragging = false;
        var _panelH = 0;
        var panel = _fs.elements.fullPanel;
        var hotbar = _fs.elements.hotbar;

        function _panelMeasure() {
            panel.style.visibility = 'hidden';
            panel.classList.add('open');
            _panelH = panel.offsetHeight || Math.round(window.innerHeight * 0.5);
            panel.classList.remove('open');
            panel.style.visibility = '';
        }

        function _openPanel() {
            if (!_panelH) _panelMeasure();
            _panelOpen = true;
            panel.classList.add('open');
            panel.style.transform = '';
            /* Adjust chart to make room */
            chartWrap.style.transition = 'padding-bottom 200ms ease-out';
            chartWrap.style.paddingBottom = _panelH + 'px';
            /* Hide hotbar behind panel */
            hotbar.style.transition = 'opacity 150ms';
            hotbar.style.opacity = '0';
            hotbar.style.pointerEvents = 'none';
            /* Lock after animation */
            setTimeout(function() {
                _panelLocked = true;
                if (_fs && _fs.chart) _fs.chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
                _fsResizeCanvas();
            }, 220);
        }

        function _closePanel() {
            _panelLocked = false;
            _panelOpen = false;
            panel.classList.remove('open');
            panel.style.transform = '';
            /* Restore chart */
            chartWrap.style.transition = 'padding-bottom 200ms ease-out';
            chartWrap.style.paddingBottom = '48px';
            /* Show hotbar */
            hotbar.style.transition = 'opacity 150ms';
            hotbar.style.opacity = '1';
            hotbar.style.pointerEvents = '';
            setTimeout(function() {
                if (_fs && _fs.chart) _fs.chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
                _fsResizeCanvas();
            }, 220);
        }

        /* ── Hotbar tool taps → activate immediately, no panel ── */
        hotbar.querySelectorAll('.hotbar-tool').forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                _fsSelectTool(btn.dataset.tool);
            });
        });

        /* ── Grip handle in hotbar → tap to toggle panel ── */
        _fs.elements.hotbarGrip.addEventListener('click', function(e) {
            e.stopPropagation();
            if (_panelOpen) _closePanel(); else _openPanel();
        });

        /* ── Panel grip drag to close ── */
        var panelGrip = _fs.elements.panelGrip;

        panelGrip.addEventListener('touchstart', function(e) {
            if (!_panelH) _panelMeasure();
            _panelStartY = e.touches[0].clientY;
            _panelCurrentY = _panelStartY;
            _panelStartTime = Date.now();
            _panelDragging = true;
            panel.style.transition = 'none';
            chartWrap.style.transition = 'none';
        }, { passive: true });

        panelGrip.addEventListener('touchmove', function(e) {
            if (!_panelDragging) return;
            _panelCurrentY = e.touches[0].clientY;
            var deltaY = _panelCurrentY - _panelStartY; /* positive = dragging down */
            if (deltaY > 0) {
                panel.style.transform = 'translateY(' + deltaY + 'px)';
                var newPad = Math.max(48, _panelH - deltaY);
                chartWrap.style.paddingBottom = newPad + 'px';
            }
        }, { passive: true });

        panelGrip.addEventListener('touchend', function(e) {
            if (!_panelDragging) return;
            _panelDragging = false;
            var deltaY = _panelCurrentY - _panelStartY;
            var elapsed = Date.now() - _panelStartTime;
            var velocity = elapsed > 0 ? deltaY / elapsed : 0;
            var threshold = _panelH * 0.25;
            if (deltaY > threshold || velocity > 0.5) {
                _closePanel();
            } else {
                _openPanel();
            }
        }, { passive: true });

        /* ── Tap panel grip to toggle ── */
        panelGrip.addEventListener('click', function(e) {
            e.stopPropagation();
            if (_panelOpen) _closePanel(); else _openPanel();
        });

        /* ── Panel tool click handlers ── */
        panel.querySelectorAll('.cfs-tool').forEach(function(btn) {
            if (btn.classList.contains('cfs-done-btn')) return;
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                _fsSelectTool(btn.dataset.tool);
                /* Panel stays open — do NOT close */
            });
        });

        /* ── Done button → close panel + deselect tool ── */
        _fs.elements.doneBtn.onclick = function(e) {
            e.stopPropagation();
            _fsSelectTool('cursor');
            _closePanel();
        };

        /* ── Hotbar grip drag-up gesture to open panel ── */
        _fs.elements.hotbarGrip.addEventListener('touchstart', function(e) {
            if (_panelOpen) return;
            if (!_panelH) _panelMeasure();
            _panelStartY = e.touches[0].clientY;
            _panelCurrentY = _panelStartY;
            _panelStartTime = Date.now();
            _panelDragging = true;
            /* Pre-show panel offscreen */
            panel.style.transition = 'none';
            panel.style.transform = 'translateY(100%)';
            panel.classList.add('open');
            chartWrap.style.transition = 'none';
        }, { passive: true });

        _fs.elements.hotbarGrip.addEventListener('touchmove', function(e) {
            if (!_panelDragging || _panelOpen) return;
            _panelCurrentY = e.touches[0].clientY;
            var deltaY = _panelStartY - _panelCurrentY; /* positive = up */
            if (deltaY > 0) {
                var progress = Math.min(deltaY, _panelH);
                panel.style.transform = 'translateY(' + (_panelH - progress) + 'px)';
                chartWrap.style.paddingBottom = (48 + progress) + 'px';
            }
        }, { passive: true });

        _fs.elements.hotbarGrip.addEventListener('touchend', function(e) {
            if (!_panelDragging) return;
            _panelDragging = false;
            var deltaY = _panelStartY - _panelCurrentY;
            var elapsed = Date.now() - _panelStartTime;
            var velocity = elapsed > 0 ? deltaY / elapsed : 0;
            var threshold = _panelH * 0.25;
            if (deltaY > threshold || velocity > 0.5) {
                panel.style.transition = 'transform 200ms ease-out';
                _openPanel();
            } else {
                panel.classList.remove('open');
                panel.style.transform = '';
                chartWrap.style.paddingBottom = '48px';
            }
        }, { passive: true });

        /* Recalculate on orientation change */
        window.addEventListener('resize', function() { _panelH = 0; });
    }

    /* ── Point-to-line-segment distance (for eraser hit-test) ── */
    function _pointToLineDist(px, py, x1, y1, x2, y2) {
        var A = px - x1, B = py - y1, C = x2 - x1, D = y2 - y1;
        var dot = A*C + B*D, lenSq = C*C + D*D;
        var param = lenSq ? dot / lenSq : -1;
        var xx, yy;
        if (param < 0) { xx = x1; yy = y1; }
        else if (param > 1) { xx = x2; yy = y2; }
        else { xx = x1 + param*C; yy = y1 + param*D; }
        return Math.sqrt((px-xx)*(px-xx) + (py-yy)*(py-yy));
    }

    /* ── Hit-test: find drawing nearest to click (for eraser/select) ── */
    function _fsHitTest(coords) {
        var T = 14;
        for (var i = _fs.drawings.length - 1; i >= 0; i--) {
            var d = _fs.drawings[i];
            if (d.type === 'hline') {
                var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
                if (y != null && Math.abs(coords.y - y) < T) return i;
            } else if (d.type === 'hray') {
                var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
                var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
                if (y != null && Math.abs(coords.y - y) < T && sx != null && coords.x >= sx - T) return i;
            } else if (d.type === 'vline') {
                var x = fsChart.timeScale().logicalToCoordinate(d.timeIdx);
                if (x != null && Math.abs(coords.x - x) < T) return i;
            } else if (d.type === 'text') {
                var x = fsChart.timeScale().logicalToCoordinate(d.timeIdx);
                var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
                if (x != null && y != null && Math.abs(coords.x - x) < 40 && Math.abs(coords.y - y) < 15) return i;
            } else {
                var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
                var sy = _fs.series ? _fs.series.priceToCoordinate(d.startPrice) : null;
                var ex = fsChart.timeScale().logicalToCoordinate(d.endIdx);
                var ey = _fs.series ? _fs.series.priceToCoordinate(d.endPrice) : null;
                if (sx == null || sy == null || ex == null || ey == null) continue;
                if (d.type === 'rect' || d.type === 'longpos' || d.type === 'shortpos') {
                    var mnX = Math.min(sx, ex), mxX = Math.max(sx, ex);
                    var mnY = Math.min(sy, ey), mxY = Math.max(sy, ey);
                    if (coords.x >= mnX-T && coords.x <= mxX+T && coords.y >= mnY-T && coords.y <= mxY+T) return i;
                } else {
                    if (_pointToLineDist(coords.x, coords.y, sx, sy, ex, ey) < T) return i;
                }
            }
        }
        return -1;
    }

    /* Determine which anchor of a drawing the pointer is near: 'start', 'end', or 'body' */
    function _fsAnchorHitTest(d, coords) {
        var A = 16; /* Anchor grab radius (larger than body hit) */
        if (d.type === 'hline') {
            return 'body'; /* Only moveable as a whole */
        } else if (d.type === 'hray') {
            var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
            var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
            if (sx != null && y != null && Math.abs(coords.x - sx) < A && Math.abs(coords.y - y) < A) return 'start';
            return 'body';
        } else if (d.type === 'vline') {
            return 'body';
        } else if (d.type === 'text') {
            return 'body';
        } else {
            /* Two-point drawing types */
            var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
            var sy = _fs.series ? _fs.series.priceToCoordinate(d.startPrice) : null;
            var ex = fsChart.timeScale().logicalToCoordinate(d.endIdx);
            var ey = _fs.series ? _fs.series.priceToCoordinate(d.endPrice) : null;
            if (sx != null && sy != null && Math.abs(coords.x - sx) < A && Math.abs(coords.y - sy) < A) return 'start';
            if (ex != null && ey != null && Math.abs(coords.x - ex) < A && Math.abs(coords.y - ey) < A) return 'end';
            return 'body';
        }
    }

    /* ── Click-click drawing interaction ── */
    function _fsOnClick(e) {
        if (!_fs) return;

        /* Cursor mode (no drawMode): click to select/deselect drawings */
        if (!_fs.drawMode) {
            if (_fs.dragging) return;
            if (_fs.drawings.length === 0) return; /* No drawings, let chart handle click */
            var coords = _fsGetChartCoords(e);
            if (coords.price == null || coords.timeIdx == null) { _fs.selectedIdx = -1; _fsRedrawCanvas(); return; }
            var hitIdx = _fsHitTest(coords);
            if (hitIdx < 0 && _fs.selectedIdx < 0) return; /* Nothing to select/deselect */
            _fs.selectedIdx = (hitIdx === _fs.selectedIdx) ? -1 : hitIdx;
            _fsRedrawCanvas();
            return;
        }

        if (_fs.lockOn && _fs.drawMode !== 'eraser') return;
        var coords = _fsGetChartCoords(e);
        if (coords.price == null || coords.timeIdx == null) return;

        /* Eraser: click to delete nearest drawing */
        if (_fs.drawMode === 'eraser') {
            var idx = _fsHitTest(coords);
            if (idx >= 0) {
                if (_fs.selectedIdx === idx) _fs.selectedIdx = -1;
                else if (_fs.selectedIdx > idx) _fs.selectedIdx--;
                _fs.drawings.splice(idx, 1);
                _fsSaveDrawings();
                _fsRedrawCanvas();
            }
            return;
        }

        /* ── Single-click tools ── */
        if (_fs.drawMode === 'hline') {
            _fs.drawings.push({ type:'hline', price: coords.price });
            _fsSaveDrawings(); _fsRedrawCanvas(); return;
        }
        if (_fs.drawMode === 'hray') {
            _fs.drawings.push({ type:'hray', price: coords.price, startIdx: coords.timeIdx });
            _fsSaveDrawings(); _fsRedrawCanvas(); return;
        }
        if (_fs.drawMode === 'vline') {
            _fs.drawings.push({ type:'vline', timeIdx: coords.timeIdx });
            _fsSaveDrawings(); _fsRedrawCanvas(); return;
        }
        if (_fs.drawMode === 'text') {
            (async () => {
                const txt = await stratosPrompt({ title: 'Text Annotation', label: 'Annotation text', placeholder: 'Support level...' });
                if (txt) {
                    _fs.drawings.push({ type:'text', text: txt, price: coords.price, timeIdx: coords.timeIdx });
                    _fsSaveDrawings(); _fsRedrawCanvas();
                }
            })();
            return;
        }

        /* ── Channel: 3-click (base line + offset) ── */
        if (_fs.drawMode === 'channel') {
            if (!_fs.drawStart) {
                /* Click 1: start of base line */
                _fs.drawStart = coords;
                _fs.previewEnd = coords;
                _fs.channelBase = null;
                _fsRedrawCanvas();
            } else if (!_fs.channelBase) {
                /* Click 2: end of base line → enter offset mode */
                _fs.channelBase = { startPrice:_fs.drawStart.price, startIdx:_fs.drawStart.timeIdx, endPrice:coords.price, endIdx:coords.timeIdx };
                _fs.drawStart = coords;
                _fs.previewEnd = coords;
                _fsRedrawCanvas();
            } else {
                /* Click 3: set channel offset → finalize */
                var offsetPrice = coords.price - _fs.channelBase.endPrice;
                _fs.drawings.push({
                    type:'channel',
                    startPrice:_fs.channelBase.startPrice, startIdx:_fs.channelBase.startIdx,
                    endPrice:_fs.channelBase.endPrice, endIdx:_fs.channelBase.endIdx,
                    offset: offsetPrice
                });
                _fs.channelBase = null;
                _fs.drawStart = null;
                _fs.previewEnd = null;
                _fsSaveDrawings(); _fsRedrawCanvas();
            }
            return;
        }

        /* ── Two-point tools: trend, ray, fib, rect, longpos, shortpos, measure ── */
        if (!_fs.drawStart) {
            _fs.drawStart = coords;
            _fs.previewEnd = coords;
            _fsRedrawCanvas();
        } else {
            var s = _fs.drawStart;
            var drawing = { type:_fs.drawMode, startPrice:s.price, startIdx:s.timeIdx, endPrice:coords.price, endIdx:coords.timeIdx };
            if (_fs.drawMode === 'measure') {
                drawing.sx = s.x; drawing.sy = s.y; drawing.ex = coords.x; drawing.ey = coords.y;
            }
            _fs.drawings.push(drawing);
            _fs.drawStart = null;
            _fs.previewEnd = null;
            _fsSaveDrawings(); _fsRedrawCanvas();
        }
    }

    /* Rubber-band preview on mouse/touch move */
    function _fsOnMove(e) {
        if (!_fs || !_fs.drawMode || !_fs.drawStart) return;
        _fs.previewEnd = _fsGetChartCoords(e);
        _fsRedrawCanvas();
    }

    /* ── Drag helpers for moving/reshaping selected drawings ── */
    var _dragDidMove = false; /* True if mouse/touch moved during drag (suppresses click) */

    function _fsDragStart(e) {
        if (!_fs || _fs.drawMode) return; /* Only drag in cursor mode */
        if (_fs.selectedIdx < 0 || _fs.selectedIdx >= _fs.drawings.length) return;
        var coords = _fsGetChartCoords(e);
        if (coords.price == null || coords.timeIdx == null) return;
        /* Only start drag if pointer is near the selected drawing */
        var hitIdx = _fsHitTest(coords);
        if (hitIdx !== _fs.selectedIdx) return;
        /* Detect which anchor we grabbed */
        var d = _fs.drawings[_fs.selectedIdx];
        _fs.dragAnchor = _fsAnchorHitTest(d, coords);
        _fs.dragging = true;
        _fs.dragStart = { timeIdx: coords.timeIdx, price: coords.price };
        _dragDidMove = false;
        /* Disable chart pan/zoom during drag */
        fsChart.applyOptions({ handleScroll: false, handleScale: false });
    }

    function _fsDragMove(e) {
        if (!_fs) return;
        /* Handle rubber-band preview for drawing tools */
        if (_fs.drawMode && _fs.drawStart) {
            _fs.previewEnd = _fsGetChartCoords(e);
            _fsRedrawCanvas();
            return;
        }
        /* Handle drag-move for selected drawing */
        if (!_fs.dragging || _fs.selectedIdx < 0) return;
        e.preventDefault();
        _dragDidMove = true;
        var coords = _fsGetChartCoords(e);
        if (coords.price == null || coords.timeIdx == null) return;
        var dTime = coords.timeIdx - _fs.dragStart.timeIdx;
        var dPrice = coords.price - _fs.dragStart.price;
        var d = _fs.drawings[_fs.selectedIdx];
        if (!d) return;
        var anchor = _fs.dragAnchor || 'body';

        if (anchor === 'start') {
            /* Move only start endpoint */
            if (d.type === 'hray') {
                /* hray uses startIdx + price (not startPrice/endPrice) */
                d.startIdx += dTime;
                d.price += dPrice;
            } else {
                if (d.startIdx != null) d.startIdx += dTime;
                if (d.startPrice != null) d.startPrice += dPrice;
            }
        } else if (anchor === 'end') {
            /* Move only end endpoint */
            if (d.endIdx != null) d.endIdx += dTime;
            if (d.endPrice != null) d.endPrice += dPrice;
        } else {
            /* Move whole drawing (body drag) */
            if (d.type === 'hline') {
                d.price += dPrice;
            } else if (d.type === 'hray') {
                d.price += dPrice;
                d.startIdx += dTime;
            } else if (d.type === 'vline') {
                d.timeIdx += dTime;
            } else if (d.type === 'text') {
                d.price += dPrice;
                d.timeIdx += dTime;
            } else {
                d.startIdx += dTime; d.startPrice += dPrice;
                d.endIdx += dTime;   d.endPrice += dPrice;
            }
        }
        _fs.dragStart = { timeIdx: coords.timeIdx, price: coords.price };
        _fsRedrawCanvas();
    }

    function _fsDragEnd(e) {
        if (!_fs || !_fs.dragging) return;
        _fs.dragging = false;
        _fs.dragStart = null;
        _fs.dragAnchor = null;
        /* Re-enable chart pan/zoom */
        fsChart.applyOptions({ handleScroll: true, handleScale: true });
        if (_dragDidMove) {
            _fsSaveDrawings();
            _fsRedrawCanvas();
        }
    }

    /* Event listeners on chartEl (not canvas) so chart scroll/zoom/pan always works.
       Canvas has pointer-events:none — events pass through to chart, we listen on chartEl. */

    /* Desktop: mousedown for drag, click for select/draw, mousemove for drag/preview */
    chartEl.addEventListener('mousedown', function(e) {
        if (!_fs) return;
        if (!_fs.drawMode && _fs.selectedIdx >= 0) {
            _fsDragStart(e);
        }
    });
    chartEl.addEventListener('click', function(e) {
        if (!_fs) return;
        if (_dragDidMove) { _dragDidMove = false; return; }
        _fsOnClick(e);
    });
    chartEl.addEventListener('mousemove', function(e) {
        if (!_fs) return;
        _fsDragMove(e);
        /* Update cursor + hoverIdx */
        if (_fs.drawMode) {
            _fs.hoverIdx = -1;
            chartEl.style.cursor = _fs.drawStart ? 'crosshair' : (_fs.drawMode === 'eraser' ? 'pointer' : 'crosshair');
            return;
        }
        if (_fs.dragging) { chartEl.style.cursor = 'grabbing'; return; }
        var coords = _fsGetChartCoords(e);
        var hitIdx = _fsHitTest(coords);
        _fs.hoverIdx = hitIdx;
        if (hitIdx >= 0 && hitIdx === _fs.selectedIdx) {
            var d = _fs.drawings[hitIdx];
            var anchor = _fsAnchorHitTest(d, coords);
            chartEl.style.cursor = (anchor === 'start' || anchor === 'end') ? 'crosshair' : 'grab';
        } else if (hitIdx >= 0) {
            chartEl.style.cursor = 'pointer';
        } else {
            chartEl.style.cursor = '';
        }
    });
    chartEl.addEventListener('mouseup', function(e) {
        _fsDragEnd(e);
        if (!_fs || !_fs.dragging) chartEl.style.cursor = '';
    });

    /* Mobile: touch events on chartEl */
    var _touchNearDrawing = false;
    chartEl.addEventListener('touchstart', function(e) {
        if (!_fs) return;
        _touchNearDrawing = false;
        if (!_fs.drawMode) {
            var coords = _fsGetChartCoords(e);
            var hitIdx = _fsHitTest(coords);
            if (hitIdx >= 0) {
                _touchNearDrawing = true;
                if (_fs.selectedIdx >= 0 && hitIdx === _fs.selectedIdx) {
                    _fsDragStart(e);
                }
            }
            return;
        }
    }, {passive:true});
    chartEl.addEventListener('touchend', function(e) {
        if (!_fs) return;
        if (_fs.dragging) { _fsDragEnd(e); return; }
        if (_dragDidMove) { _dragDidMove = false; return; }
        if (!_fs.drawMode) {
            if (_touchNearDrawing) _fsOnClick(e);
            return;
        }
        _fsOnClick(e);
    }, {passive:true});
    chartEl.addEventListener('touchmove', function(e) {
        if (!_fs) return;
        if (_fs.dragging) { _fsDragMove(e); return; }
        if (_fs.drawMode && _fs.drawStart) {
            _fs.previewEnd = _fsGetChartCoords(e);
            _fsRedrawCanvas();
        }
    }, {passive:true});

    /* Redraw drawings on chart scroll/zoom */
    fsChart.timeScale().subscribeVisibleLogicalRangeChange(function() { _fsRedrawCanvas(); });

    /* Right-click / Escape: cancel in-progress drawing */
    chartEl.addEventListener('contextmenu', function(e) {
        if (_fs && (_fs.drawStart || _fs.channelBase)) {
            e.preventDefault();
            _fs.drawStart = null;
            _fs.previewEnd = null;
            _fs.channelBase = null;
            _fsRedrawCanvas();
        }
    });
    root.addEventListener('keydown', function(e) {
        if (!_fs) return;
        /* Escape: cancel in-progress drawing OR deselect */
        if (e.key === 'Escape') {
            if (_fs.drawStart || _fs.channelBase) {
                _fs.drawStart = null;
                _fs.previewEnd = null;
                _fs.channelBase = null;
                _fsRedrawCanvas();
            } else if (_fs.selectedIdx >= 0) {
                _fs.selectedIdx = -1;
                _fsRedrawCanvas();
            }
            return;
        }
        /* Delete/Backspace: delete selected or hovered drawing */
        if (e.key === 'Delete' || e.key === 'Backspace') {
            var idx = _fs.selectedIdx >= 0 ? _fs.selectedIdx : _fs.hoverIdx;
            if (idx >= 0 && idx < _fs.drawings.length) {
                _fs.drawings.splice(idx, 1);
                _fs.selectedIdx = -1;
                _fs.hoverIdx = -1;
                _fsSaveDrawings();
                _fsRedrawCanvas();
                _fsUpdateCanvasPointer();
                e.preventDefault();
            }
        }
    });
    /* Make root focusable for keyboard events */
    root.setAttribute('tabindex', '-1');
    root.style.outline = 'none';
    root.focus();

    /* ── Drawing Canvas Renderer (finalized + preview) ── */
    function _fsDrawAnchor(ctx, x, y) {
        /* Small solid circle at anchor point */
        ctx.fillStyle = 'var(--accent, #F0B90B)';
        ctx.fillStyle = '#F0B90B';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        /* Outer ring */
        ctx.strokeStyle = 'rgba(240,185,11,0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.stroke();
    }

    function _fsDrawShape(ctx, type, s, e, w, isPreview) {
        /* Draws a single shape given start/end coords. w = chartW (plot area, excludes price scale). isPreview = dashed/transparent */
        var alpha = isPreview ? 0.5 : 1;
        var h = drawCanvas.height;
        /* Clip to plot area so nothing bleeds into price scale */
        ctx.save();
        ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();

        if (type === 'trend') {
            ctx.strokeStyle = isPreview ? 'rgba(33,150,243,0.5)' : '#2196F3';
            ctx.lineWidth = 1.5;
            ctx.setLineDash(isPreview ? [6, 4] : []);
            ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
            ctx.setLineDash([]);

        } else if (type === 'ray') {
            /* Line from s through e, extending to canvas edge */
            ctx.strokeStyle = isPreview ? 'rgba(33,150,243,0.5)' : '#2196F3';
            ctx.lineWidth = 1.5;
            ctx.setLineDash(isPreview ? [6, 4] : []);
            var dx = e.x - s.x, dy = e.y - s.y;
            var len = Math.sqrt(dx*dx + dy*dy);
            if (len > 0) {
                var ext = Math.max(w, h) * 3 / len;
                ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(s.x + dx*ext, s.y + dy*ext); ctx.stroke();
            }
            ctx.setLineDash([]);

        } else if (type === 'channel') {
            /* Two parallel lines + fill between */
            var offsetPx = (e.channelOffsetPx != null) ? e.channelOffsetPx : 0;
            ctx.strokeStyle = isPreview ? 'rgba(33,150,243,0.5)' : '#2196F3';
            ctx.lineWidth = 1.5;
            ctx.setLineDash(isPreview ? [6, 4] : []);
            ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(s.x, s.y + offsetPx); ctx.lineTo(e.x, e.y + offsetPx); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = isPreview ? 'rgba(33,150,243,0.03)' : 'rgba(33,150,243,0.06)';
            ctx.beginPath();
            ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y);
            ctx.lineTo(e.x, e.y + offsetPx); ctx.lineTo(s.x, s.y + offsetPx);
            ctx.closePath(); ctx.fill();

        } else if (type === 'fib') {
            var fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
            var fibColors = ['#787B86','#F6C343','#4CAF50','#2196F3','#9C27B0','#E91E63','#787B86'];
            var range = e.price - s.price;
            fibLevels.forEach(function(lvl, idx) {
                var price = s.price + range * lvl;
                var y = _fs.series ? _fs.series.priceToCoordinate(price) : null;
                if (y == null) return;
                ctx.globalAlpha = alpha;
                ctx.strokeStyle = fibColors[idx] || '#787B86';
                ctx.lineWidth = 1;
                ctx.setLineDash(lvl === 0 || lvl === 1 ? [] : [4, 2]);
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = fibColors[idx] || '#787B86';
                ctx.font = '9px monospace';
                ctx.fillText((lvl * 100).toFixed(1) + '% ' + _fsFmtPrice(price), 4, y - 3);
                ctx.globalAlpha = 1;
            });
            var yTop = _fs.series ? _fs.series.priceToCoordinate(Math.max(s.price, e.price)) : null;
            var yBot = _fs.series ? _fs.series.priceToCoordinate(Math.min(s.price, e.price)) : null;
            if (yTop != null && yBot != null) {
                ctx.fillStyle = isPreview ? 'rgba(33,150,243,0.03)' : 'rgba(33,150,243,0.05)';
                ctx.fillRect(Math.min(s.x, e.x), yTop, Math.abs(e.x - s.x), yBot - yTop);
            }

        } else if (type === 'rect') {
            ctx.strokeStyle = isPreview ? 'rgba(240,185,11,0.5)' : '#F0B90B';
            ctx.lineWidth = 1;
            ctx.setLineDash(isPreview ? [4, 3] : []);
            ctx.fillStyle = isPreview ? 'rgba(240,185,11,0.04)' : 'rgba(240,185,11,0.08)';
            var rw = e.x - s.x, rh = e.y - s.y;
            ctx.fillRect(s.x, s.y, rw, rh);
            ctx.strokeRect(s.x, s.y, rw, rh);
            ctx.setLineDash([]);

        } else if (type === 'longpos') {
            /* Long position: entry at s, TP at e (above). SL mirrored below entry. */
            var entryY = s.y, tpY = e.y;
            var slY = entryY + (entryY - tpY);
            var slPrice = s.price - (e.price - s.price);
            var boxL = Math.min(s.x, e.x), boxR = Math.max(s.x, e.x);
            if (boxR - boxL < 60) { boxL = 0; boxR = w; }
            /* TP zone (green) */
            ctx.fillStyle = isPreview ? 'rgba(14,203,129,0.06)' : 'rgba(14,203,129,0.1)';
            ctx.fillRect(boxL, Math.min(entryY, tpY), boxR - boxL, Math.abs(tpY - entryY));
            /* SL zone (red) */
            ctx.fillStyle = isPreview ? 'rgba(246,70,93,0.06)' : 'rgba(246,70,93,0.1)';
            ctx.fillRect(boxL, entryY, boxR - boxL, Math.abs(slY - entryY));
            /* Lines */
            ctx.lineWidth = 1; ctx.setLineDash([]);
            ctx.strokeStyle = '#888';
            ctx.beginPath(); ctx.moveTo(boxL, entryY); ctx.lineTo(boxR, entryY); ctx.stroke();
            ctx.strokeStyle = '#0ECB81';
            ctx.beginPath(); ctx.moveTo(boxL, tpY); ctx.lineTo(boxR, tpY); ctx.stroke();
            ctx.strokeStyle = '#F6465D';
            ctx.beginPath(); ctx.moveTo(boxL, slY); ctx.lineTo(boxR, slY); ctx.stroke();
            /* Labels */
            ctx.font = '10px monospace'; ctx.globalAlpha = alpha;
            ctx.fillStyle = '#888';
            ctx.fillText('Entry ' + _fsFmtPrice(s.price), boxL + 4, entryY - 4);
            ctx.fillStyle = '#0ECB81';
            var tpPct = s.price ? Math.abs(((e.price - s.price) / s.price) * 100) : 0;
            ctx.fillText('TP ' + _fsFmtPrice(e.price) + ' (+' + tpPct.toFixed(2) + '%)', boxL + 4, tpY - 4);
            ctx.fillStyle = '#F6465D';
            var slPct = s.price ? Math.abs(((slPrice - s.price) / s.price) * 100) : 0;
            ctx.fillText('SL ' + _fsFmtPrice(slPrice) + ' (-' + slPct.toFixed(2) + '%)', boxL + 4, slY + 14);
            var rr = Math.abs(e.price - s.price) / Math.max(0.0001, Math.abs(s.price - slPrice));
            ctx.fillStyle = '#ddd';
            ctx.fillText('R/R 1:' + rr.toFixed(1), boxL + 4, entryY + 14);
            ctx.globalAlpha = 1;

        } else if (type === 'shortpos') {
            /* Short position: entry at s, TP at e (below). SL mirrored above entry. */
            var entryY = s.y, tpY = e.y;
            var slY = entryY - (tpY - entryY);
            var slPrice = s.price + (s.price - e.price);
            var boxL = Math.min(s.x, e.x), boxR = Math.max(s.x, e.x);
            if (boxR - boxL < 60) { boxL = 0; boxR = w; }
            /* TP zone (green, below) */
            ctx.fillStyle = isPreview ? 'rgba(14,203,129,0.06)' : 'rgba(14,203,129,0.1)';
            ctx.fillRect(boxL, Math.min(entryY, tpY), boxR - boxL, Math.abs(tpY - entryY));
            /* SL zone (red, above) */
            ctx.fillStyle = isPreview ? 'rgba(246,70,93,0.06)' : 'rgba(246,70,93,0.1)';
            ctx.fillRect(boxL, Math.min(entryY, slY), boxR - boxL, Math.abs(slY - entryY));
            /* Lines */
            ctx.lineWidth = 1; ctx.setLineDash([]);
            ctx.strokeStyle = '#888';
            ctx.beginPath(); ctx.moveTo(boxL, entryY); ctx.lineTo(boxR, entryY); ctx.stroke();
            ctx.strokeStyle = '#0ECB81';
            ctx.beginPath(); ctx.moveTo(boxL, tpY); ctx.lineTo(boxR, tpY); ctx.stroke();
            ctx.strokeStyle = '#F6465D';
            ctx.beginPath(); ctx.moveTo(boxL, slY); ctx.lineTo(boxR, slY); ctx.stroke();
            /* Labels */
            ctx.font = '10px monospace'; ctx.globalAlpha = alpha;
            ctx.fillStyle = '#888';
            ctx.fillText('Entry ' + _fsFmtPrice(s.price), boxL + 4, entryY - 4);
            ctx.fillStyle = '#0ECB81';
            var tpPct = s.price ? Math.abs(((s.price - e.price) / s.price) * 100) : 0;
            ctx.fillText('TP ' + _fsFmtPrice(e.price) + ' (+' + tpPct.toFixed(2) + '%)', boxL + 4, tpY + 14);
            ctx.fillStyle = '#F6465D';
            var slPct = s.price ? Math.abs(((slPrice - s.price) / s.price) * 100) : 0;
            ctx.fillText('SL ' + _fsFmtPrice(slPrice) + ' (-' + slPct.toFixed(2) + '%)', boxL + 4, slY - 4);
            var rr = Math.abs(s.price - e.price) / Math.max(0.0001, Math.abs(slPrice - s.price));
            ctx.fillStyle = '#ddd';
            ctx.fillText('R/R 1:' + rr.toFixed(1), boxL + 4, entryY + 14);
            ctx.globalAlpha = 1;

        } else if (type === 'measure') {
            ctx.strokeStyle = isPreview ? 'rgba(170,170,170,0.5)' : '#aaa';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
            ctx.setLineDash([]);
            var pDiff = e.price - s.price;
            var pPct = s.price ? ((pDiff / s.price) * 100) : 0;
            ctx.fillStyle = isPreview ? 'rgba(221,221,221,0.5)' : '#ddd';
            ctx.font = '10px monospace';
            ctx.fillText(_fsFmtPrice(pDiff) + ' (' + pPct.toFixed(2) + '%)', (s.x+e.x)/2 + 4, (s.y+e.y)/2 - 6);
        }
        ctx.restore(); /* Undo clip rect */
    }

    function _fsRedrawCanvas() {
        if (!_fs || !_fs.drawCtx) return;
        var ctx = _fs.drawCtx;
        var w = drawCanvas.width, h = drawCanvas.height;
        /* Plot area width (excludes price scale on right) */
        var chartW = (fsChart && fsChart.timeScale()) ? fsChart.timeScale().width() : w;
        if (chartW <= 0 || chartW > w) chartW = w;
        ctx.clearRect(0, 0, w, h);

        /* If hide toggle is on, skip all drawings */
        if (_fs.hideOn) return;

        /* Clip all drawing to plot area (excludes price scale) */
        ctx.save();
        ctx.beginPath(); ctx.rect(0, 0, chartW, h); ctx.clip();

        /* Draw all finalized drawings */
        _fs.drawings.forEach(function(d) {
            if (d.type === 'hline') {
                var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
                if (y == null) return;
                ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1;
                ctx.setLineDash([5, 3]);
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#F0B90B'; ctx.font = '10px monospace';
                ctx.fillText(_fsFmtPrice(d.price), 4, y - 4);
            } else if (d.type === 'hray') {
                var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
                var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
                if (y == null || sx == null) return;
                ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1;
                ctx.setLineDash([5, 3]);
                ctx.beginPath(); ctx.moveTo(sx, y); ctx.lineTo(chartW, y); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#F0B90B'; ctx.font = '10px monospace';
                ctx.fillText(_fsFmtPrice(d.price), sx + 4, y - 4);
            } else if (d.type === 'vline') {
                var x = fsChart.timeScale().logicalToCoordinate(d.timeIdx);
                if (x == null) return;
                ctx.strokeStyle = '#F0B90B'; ctx.lineWidth = 1;
                ctx.setLineDash([5, 3]);
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
                ctx.setLineDash([]);
            } else if (d.type === 'text') {
                var x = fsChart.timeScale().logicalToCoordinate(d.timeIdx);
                var y = _fs.series ? _fs.series.priceToCoordinate(d.price) : null;
                if (x == null || y == null) return;
                ctx.fillStyle = '#eaecef'; ctx.font = '12px sans-serif';
                ctx.fillText(d.text, x, y);
            } else if (d.type === 'channel') {
                var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
                var sy = _fs.series ? _fs.series.priceToCoordinate(d.startPrice) : null;
                var ex = fsChart.timeScale().logicalToCoordinate(d.endIdx);
                var ey = _fs.series ? _fs.series.priceToCoordinate(d.endPrice) : null;
                if (sx == null || sy == null || ex == null || ey == null) return;
                var offPriceY = _fs.series ? _fs.series.priceToCoordinate(d.endPrice + d.offset) : null;
                var offsetPx = (offPriceY != null) ? (offPriceY - ey) : 0;
                _fsDrawShape(ctx, 'channel', {x:sx, y:sy, price:d.startPrice}, {x:ex, y:ey, price:d.endPrice, channelOffsetPx: offsetPx}, chartW, false);
            } else {
                /* Two-point tools: trend, ray, fib, rect, longpos, shortpos, measure */
                var sx = fsChart.timeScale().logicalToCoordinate(d.startIdx);
                var sy = _fs.series ? _fs.series.priceToCoordinate(d.startPrice) : null;
                var ex = fsChart.timeScale().logicalToCoordinate(d.endIdx);
                var ey = _fs.series ? _fs.series.priceToCoordinate(d.endPrice) : null;
                if (sx == null || sy == null || ex == null || ey == null) return;
                _fsDrawShape(ctx, d.type, {x:sx, y:sy, price:d.startPrice}, {x:ex, y:ey, price:d.endPrice}, chartW, false);
            }
        });

        /* Draw selection highlight + handles on selected drawing */
        if (_fs.selectedIdx >= 0 && _fs.selectedIdx < _fs.drawings.length) {
            var sel = _fs.drawings[_fs.selectedIdx];
            ctx.save();
            ctx.strokeStyle = 'rgba(240,185,11,0.7)';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 3]);
            if (sel.type === 'hline') {
                var y = _fs.series ? _fs.series.priceToCoordinate(sel.price) : null;
                if (y != null) {
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke();
                    _fsDrawAnchor(ctx, chartW / 2, y);
                }
            } else if (sel.type === 'hray') {
                var y = _fs.series ? _fs.series.priceToCoordinate(sel.price) : null;
                var sx = fsChart.timeScale().logicalToCoordinate(sel.startIdx);
                if (y != null && sx != null) {
                    ctx.beginPath(); ctx.moveTo(sx, y); ctx.lineTo(chartW, y); ctx.stroke();
                    _fsDrawAnchor(ctx, sx, y);
                }
            } else if (sel.type === 'vline') {
                var x = fsChart.timeScale().logicalToCoordinate(sel.timeIdx);
                if (x != null) {
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
                    _fsDrawAnchor(ctx, x, h / 2);
                }
            } else if (sel.type === 'text') {
                var x = fsChart.timeScale().logicalToCoordinate(sel.timeIdx);
                var y = _fs.series ? _fs.series.priceToCoordinate(sel.price) : null;
                if (x != null && y != null) _fsDrawAnchor(ctx, x, y);
            } else {
                var sx = fsChart.timeScale().logicalToCoordinate(sel.startIdx);
                var sy = _fs.series ? _fs.series.priceToCoordinate(sel.startPrice) : null;
                var ex = fsChart.timeScale().logicalToCoordinate(sel.endIdx);
                var ey = _fs.series ? _fs.series.priceToCoordinate(sel.endPrice) : null;
                if (sx != null && sy != null && ex != null && ey != null) {
                    _fsDrawAnchor(ctx, sx, sy);
                    _fsDrawAnchor(ctx, ex, ey);
                }
            }
            ctx.restore();
        }

        /* Draw preview (rubber-band) if first click is set */
        if (_fs.drawStart && _fs.previewEnd && _fs.drawMode) {
            var mode = _fs.drawMode;
            /* Skip preview for single-click and utility tools */
            if (mode !== 'hline' && mode !== 'hray' && mode !== 'vline' && mode !== 'text' && mode !== 'eraser') {
                var sx = fsChart.timeScale().logicalToCoordinate(_fs.drawStart.timeIdx);
                var sy = _fs.series ? _fs.series.priceToCoordinate(_fs.drawStart.price) : null;
                var ex = fsChart.timeScale().logicalToCoordinate(_fs.previewEnd.timeIdx);
                var ey = _fs.series ? _fs.series.priceToCoordinate(_fs.previewEnd.price) : null;
                if (sx != null && sy != null && ex != null && ey != null) {
                    if (mode === 'channel' && _fs.channelBase) {
                        /* Channel phase 2: show base line + moving offset */
                        var bsx = fsChart.timeScale().logicalToCoordinate(_fs.channelBase.startIdx);
                        var bsy = _fs.series ? _fs.series.priceToCoordinate(_fs.channelBase.startPrice) : null;
                        var bex = fsChart.timeScale().logicalToCoordinate(_fs.channelBase.endIdx);
                        var bey = _fs.series ? _fs.series.priceToCoordinate(_fs.channelBase.endPrice) : null;
                        if (bsx != null && bsy != null && bex != null && bey != null) {
                            var offPY = _fs.series ? _fs.series.priceToCoordinate(_fs.channelBase.endPrice + (_fs.previewEnd.price - _fs.channelBase.endPrice)) : null;
                            var offsetPx = (offPY != null) ? (offPY - bey) : (ey - bey);
                            _fsDrawShape(ctx, 'channel', {x:bsx, y:bsy, price:_fs.channelBase.startPrice}, {x:bex, y:bey, price:_fs.channelBase.endPrice, channelOffsetPx: offsetPx}, chartW, true);
                        }
                    } else {
                        _fsDrawShape(ctx, mode, {x:sx, y:sy, price:_fs.drawStart.price}, {x:ex, y:ey, price:_fs.previewEnd.price}, chartW, true);
                    }
                }
                /* Draw anchor marker at first click point */
                if (sx != null && sy != null) {
                    _fsDrawAnchor(ctx, sx, sy);
                }
                /* For channel phase 2, also draw anchor at base start */
                if (mode === 'channel' && _fs.channelBase) {
                    var bsx2 = fsChart.timeScale().logicalToCoordinate(_fs.channelBase.startIdx);
                    var bsy2 = _fs.series ? _fs.series.priceToCoordinate(_fs.channelBase.startPrice) : null;
                    if (bsx2 != null && bsy2 != null) _fsDrawAnchor(ctx, bsx2, bsy2);
                }
            }
        }
        ctx.restore(); /* Undo plot-area clip rect */
    }
    _fs._fsRedrawCanvas = _fsRedrawCanvas;

    /* Screenshot — composites chart + drawings + header info + active tools */
    function _fsScreenshot() {
        try {
            /* Find the main chart canvas (TV lightweight-charts) */
            var tvCanvases = chartEl.querySelectorAll('canvas');
            var mainCanvas = null;
            for (var ci = 0; ci < tvCanvases.length; ci++) {
                if (!tvCanvases[ci].classList.contains('cfs-draw-canvas') && tvCanvases[ci].width > 100) {
                    mainCanvas = tvCanvases[ci]; break;
                }
            }
            if (!mainCanvas) return;

            var cw = mainCanvas.width, ch = mainCanvas.height;
            var headerH = 48, footerH = 32, pad = 16;
            var totalW = cw + pad * 2;
            var totalH = ch + headerH + footerH + pad * 2;

            var out = document.createElement('canvas');
            out.width = totalW; out.height = totalH;
            var ctx = out.getContext('2d');

            /* Background */
            ctx.fillStyle = '#0f1218';
            ctx.fillRect(0, 0, totalW, totalH);

            /* Header: ticker + price + change + timeframe */
            ctx.fillStyle = '#1a1f2e';
            ctx.fillRect(0, 0, totalW, headerH);
            ctx.fillStyle = '#848e9c';
            ctx.font = 'bold 14px ui-monospace, monospace';
            var sym = _fs.symbol.replace(/-USD/,'').replace(/=F/,'').replace(/=X/,'');
            ctx.fillText(sym, pad, 28);

            var ad = (typeof _resolveData === 'function') ? _resolveData(_fs.symbol, _fs.tfKey) : null;
            var price = ad ? (ad.price || 0) : 0;
            var change = ad ? (ad.change || 0) : 0;
            var symW = ctx.measureText(sym).width;
            ctx.font = '13px ui-monospace, monospace';
            ctx.fillStyle = change >= 0 ? '#0ECB81' : '#F6465D';
            var priceStr = '$' + price.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
            ctx.fillText(priceStr + '  ' + (change >= 0 ? '+' : '') + change.toFixed(2) + '%', pad + symW + 12, 28);

            /* Timeframe label */
            var tfLabel = (_fs.tfKey || '').replace('_','/').toUpperCase();
            ctx.fillStyle = '#F0B90B';
            ctx.font = 'bold 11px ui-monospace, monospace';
            var tfW = ctx.measureText(tfLabel).width;
            ctx.fillText(tfLabel, totalW - pad - tfW, 28);

            /* Chart canvas */
            ctx.drawImage(mainCanvas, pad, headerH);

            /* Drawing overlay canvas */
            if (_fs.drawCanvas && _fs.drawCanvas.width > 0) {
                ctx.drawImage(_fs.drawCanvas, pad, headerH);
            }

            /* Footer: active drawing tools used */
            ctx.fillStyle = '#1a1f2e';
            ctx.fillRect(0, headerH + ch + pad, totalW, footerH + pad);
            var tools = [];
            if (_fs.drawings && _fs.drawings.length > 0) {
                var types = {};
                _fs.drawings.forEach(function(d) { if (d.type) types[d.type] = (types[d.type] || 0) + 1; });
                var names = { trend:'Trend', ray:'Ray', hline:'H-Line', hray:'H-Ray', vline:'V-Line', channel:'Channel', fib:'Fib', rect:'Rect', longpos:'Long', shortpos:'Short', measure:'Measure', text:'Text' };
                Object.keys(types).forEach(function(t) {
                    tools.push((names[t] || t) + (types[t] > 1 ? ' x' + types[t] : ''));
                });
            }
            if (_fs.maSeries && _fs.maSeries.length > 0) tools.push('MA');
            if (_fs.crosshairOn) tools.push('Crosshair');
            if (_fs.magnetOn) tools.push('Magnet');

            ctx.font = '10px ui-monospace, monospace';
            ctx.fillStyle = '#848e9c';
            var toolStr = tools.length ? tools.join('  \u00b7  ') : 'No tools active';
            ctx.fillText(toolStr, pad, headerH + ch + pad + 20);

            /* Branding */
            ctx.fillStyle = '#3a3f4c';
            ctx.font = '9px ui-monospace, monospace';
            var brand = 'STRAT_OS';
            ctx.fillText(brand, totalW - pad - ctx.measureText(brand).width, headerH + ch + pad + 20);

            /* Download */
            var link = document.createElement('a');
            link.download = 'STRAT_OS_' + sym + '_' + (_fs.tfKey || 'chart') + '_' + new Date().toISOString().slice(0,10) + '.png';
            link.href = out.toDataURL('image/png');
            link.click();
            if (typeof showToast === 'function') showToast('Chart saved as PNG', 'success');
        } catch(e) {
            console.error('[Focus] Screenshot error:', e);
            if (typeof showToast === 'function') showToast('Screenshot failed', 'error');
        }
    }

    /* ── Drawing persistence (localStorage per symbol) ── */
    function _fsSaveDrawings() {
        if (!_fs || !_fs.symbol) return;
        try {
            localStorage.setItem('stratos_drawings_' + _fs.symbol, JSON.stringify(_fs.drawings));
        } catch(e) {}
        /* Refresh analysis panel (e.g. fib levels) when drawings change */
        _fsRenderAnalysis();
    }
    function _fsLoadDrawings() {
        if (!_fs || !_fs.symbol) return;
        try {
            var saved = localStorage.getItem('stratos_drawings_' + _fs.symbol);
            if (saved) {
                _fs.drawings = JSON.parse(saved);
                _fsRedrawCanvas();
            }
        } catch(e) { _fs.drawings = []; }
    }
    /* Load saved drawings after chart is ready */
    setTimeout(_fsLoadDrawings, 100);

    /* ── Double-tap to fit (mobile) — pinch-safe ── */
    if (isMobile) {
        var lastTap = 0;
        var wasPinch = false;
        chartEl.addEventListener('touchstart', function(e) {
            if (e.touches.length > 1) wasPinch = true;
        }, {passive: true});
        chartEl.addEventListener('touchend', function(e) {
            if (_fs.drawMode) return;
            /* Skip if this touchend is from a pinch gesture */
            if (wasPinch) { wasPinch = (e.touches.length > 0); return; }
            /* Require single-finger tap only */
            if (e.changedTouches.length !== 1) return;
            var now = Date.now();
            if (now - lastTap < 300) {
                fsChart.timeScale().fitContent();
                fsChart.priceScale('right').applyOptions({ autoScale: true });
                lastTap = 0; /* Reset so triple-tap doesn't re-trigger */
            } else {
                lastTap = now;
            }
        });
    }

    /* ── ResizeObserver ── */
    var fsRO = new ResizeObserver(function() {
        fsChart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
        _fsResizeCanvas();
    });
    fsRO.observe(chartEl);
    /* Initial canvas size */
    setTimeout(_fsResizeCanvas, 50);

    /* ── Live auto-refresh for ALL timeframes in focus mode ── */
    var _fsLiveTimer = null;
    function _fsFetchAndUpdate() {
        if (!_fs || document.hidden) return;
        var sym = _fs.symbol;
        var tf = _fs.tfKey;
        var token = typeof getAuthToken === 'function' ? getAuthToken() : localStorage.getItem('stratos_auth_token');
        fetch('/api/market-tick?symbol=' + encodeURIComponent(sym) + '&interval=' + tf, {
            headers: token ? {'X-Auth-Token': token} : {}
        })
        .then(function(r) { return r.ok ? r.json() : null; })
        .then(function(tick) {
            if (!tick || !_fs || _fs.symbol !== sym || _fs.tfKey !== tf) return;
            var td = tick[sym];
            if (!td || !td.data || !td.data[tf]) return;
            var idata = td.data[tf];
            /* Update marketData in-place */
            if (typeof marketData !== 'undefined' && marketData[sym] && marketData[sym].data) {
                marketData[sym].data[tf] = idata;
            }
            /* Rebuild chart data and update series */
            var newBuilt = _fsBuildData(sym, tf);
            if (newBuilt && newBuilt.data.length > 0) {
                _fs.series.setData(newBuilt.data);
                _fs.chartData = newBuilt.data;
                _fsRefreshMAs(newBuilt.data);
                /* Update header price */
                var p = idata.price || 0;
                var c = idata.change || 0;
                var pc = c >= 0 ? '#0ECB81' : '#F6465D';
                var pEl = header.querySelector('.cfs-price');
                var cE = header.querySelector('.cfs-change');
                if (pEl) { pEl.textContent = _fsFmtPrice(p); pEl.style.color = pc; }
                if (cE) { cE.textContent = (c >= 0 ? '+' : '') + c.toFixed(2) + '%'; cE.style.color = pc; }
            }
        })
        .catch(function() {});
    }
    function _fsStartLive() {
        _fsStopLive();
        if (!_fs) return;
        var tf = _fs.tfKey;
        /* Refresh interval: 30s for 1m, 60s for all others */
        var intervalMs = tf === '1m' ? 30000 : 60000;
        _fsLiveTimer = setInterval(_fsFetchAndUpdate, intervalMs);
    }
    function _fsStopLive() {
        if (_fsLiveTimer) { clearInterval(_fsLiveTimer); _fsLiveTimer = null; }
    }
    /* Immediately fetch fresh data on open, then start interval */
    _fsFetchAndUpdate();
    _fsStartLive();
    /* Restart on timeframe switch */
    var _origTfHandler = null;

    /* ── Close handler ── */
    function _fsClose() {
        var closingSym = _fs ? _fs.symbol : null;
        _fsStopLive();
        if (_fs.crosshairSub) { try { fsChart.unsubscribeCrosshairMove(_fs.crosshairSub); } catch(e){} }
        fsRO.disconnect();
        try { fsChart.remove(); } catch(e) {}
        /* Remove mobile hotbar + full panel from document.body */
        if (_fs.elements && _fs.elements.hotbar && _fs.elements.hotbar.parentNode) {
            _fs.elements.hotbar.parentNode.removeChild(_fs.elements.hotbar);
        }
        if (_fs.elements && _fs.elements.fullPanel && _fs.elements.fullPanel.parentNode) {
            _fs.elements.fullPanel.parentNode.removeChild(_fs.elements.fullPanel);
        }
        root.remove();
        bg.remove();
        window.removeEventListener('popstate', onPop);
        document.removeEventListener('keydown', onEsc);
        window._fs = null;
        _fs = null;
        /* Sync main chart with latest data from focus mode */
        if (closingSym && typeof updateChart === 'function') updateChart(closingSym);
    }

    header.querySelector('.cfs-close-btn').addEventListener('click', _fsClose);
    bg.addEventListener('click', _fsClose);

    var onEsc = function(e) {
        if (e.key === 'Escape') {
            if (!_fs) return;
            /* If drawing in progress, cancel it first */
            if (_fs.drawStart) {
                _fs.drawStart = null;
                _fs.previewEnd = null;
                _fsRedrawCanvas();
                return;
            }
            /* If a drawing tool is active, switch to cursor first */
            if (_fs.drawMode) {
                _fsSelectTool('cursor');
                return;
            }
            /* Otherwise close the fullscreen chart */
            _fsClose();
        }
    };
    document.addEventListener('keydown', onEsc);

    var onPop = function() { _fsClose(); };
    history.pushState({ chartFs: true }, '');
    window.addEventListener('popstate', onPop);

    /* ── Desktop keyboard shortcuts in fullscreen ── */
    if (!isMobile) {
        document.addEventListener('keydown', function _fsKeys(e) {
            if (!_fs) { document.removeEventListener('keydown', _fsKeys); return; }
            /* Ctrl+Z undo drawing */
            if (e.ctrlKey && e.key === 'z') {
                _fs.drawings.pop();
                _fs.drawStart = null;
                _fs.previewEnd = null;
                _fs.channelBase = null;
                _fsSaveDrawings();
                _fsRedrawCanvas();
                return;
            }
            /* Don't capture shortcuts when typing in inputs */
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
            if (e.ctrlKey || e.altKey || e.metaKey) return; /* only bare keys or shift+ */

            var toolId = null;
            switch (e.key.toLowerCase()) {
                case 'c': toolId = 'crosshair'; break;
                case 't': toolId = 'trend'; break;
                case 'r': toolId = e.shiftKey ? 'rect' : 'ray'; break;
                case 'h': toolId = e.shiftKey ? 'hray' : 'hline'; break;
                case 'v': toolId = 'vline'; break;
                case 'p': toolId = 'channel'; break;
                case 'f': toolId = 'fib'; break;
                case 'l': toolId = e.shiftKey ? 'shortpos' : 'longpos'; break;
                case 'x': toolId = 'text'; break;
                case 'm': toolId = 'measure'; break;
                case 'g': toolId = 'magnet'; break;
                case 'e': toolId = 'eraser'; break;
                case 'k': toolId = e.shiftKey ? 'hide' : 'lock'; break;
                case 'delete': if (e.shiftKey) toolId = 'trash'; break;
            }
            if (toolId) {
                e.preventDefault();
                _fsSelectTool(toolId);
                /* Flash animation on the activated button */
                var flashBtn = root.querySelector('.cfs-tool[data-tool="' + toolId + '"]');
                if (flashBtn) {
                    flashBtn.classList.remove('cfs-tool-flash');
                    void flashBtn.offsetWidth; /* reflow to restart animation */
                    flashBtn.classList.add('cfs-tool-flash');
                    setTimeout(function() { flashBtn.classList.remove('cfs-tool-flash'); }, 250);
                }
            }
        });
    }
}

/* _extractSeriesData and _extractMpChartData moved to fullscreen-chart-utils.js */

})();
