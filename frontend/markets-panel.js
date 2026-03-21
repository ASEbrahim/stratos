// ═══════════════════════════════════════════════════════════
// STRAT_OS — MARKETS PANEL
// ═══════════════════════════════════════════════════════════

// roundRect polyfill for older browsers
if (typeof CanvasRenderingContext2D !== 'undefined' && !CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        if (typeof r === 'number') r = [r,r,r,r];
        var tl=r[0]||0; this.beginPath(); this.moveTo(x+tl,y);
        this.lineTo(x+w-tl,y); this.quadraticCurveTo(x+w,y,x+w,y+tl);
        this.lineTo(x+w,y+h-tl); this.quadraticCurveTo(x+w,y+h,x+w-tl,y+h);
        this.lineTo(x+tl,y+h); this.quadraticCurveTo(x,y+h,x,y+h-tl);
        this.lineTo(x,y+tl); this.quadraticCurveTo(x,y,x+tl,y); this.closePath(); return this;
    };
}

var _mpTzOff = -(new Date().getTimezoneOffset() * 60); // browser TZ offset for chart display
var MP_MAX_CHARTS = 6;
var _mpCharts = [];
var _mpSwapSelected = null; // click-to-swap: selected chart id
var _mpCounter = 0;
var _mpSectionCollapsed = {};
var _mpActiveTicker = null;
var _mpIntelArticles = []; // stored articles for safe Ask AI referencing

function _mpSaveLayout() { try { localStorage.setItem('mp_layout', JSON.stringify(_mpCharts.map(function(c) { return { symbol: c.symbol, timeframe: c.timeframe, chartType: c.chartType||'candle', height: c.chartHeight||220 }; }))); } catch(e){} }
function _mpLoadLayout() { try { return JSON.parse(localStorage.getItem('mp_layout') || '[]'); } catch(e) { return []; } }
function _mpLoadSections() { try { _mpSectionCollapsed = JSON.parse(localStorage.getItem('mp_sec2') || '{}'); } catch(e) { _mpSectionCollapsed = {}; } }
function _mpSaveSections() { try { localStorage.setItem('mp_sec2', JSON.stringify(_mpSectionCollapsed)); } catch(e){} }
function _mpFp(v) { return v != null ? v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'; }

function mpToggleSection(name) {
    _mpSectionCollapsed[name] = !_mpSectionCollapsed[name];
    _mpSaveSections();
    _mpApplySection(name);
}
function _mpApplySection(name) {
    var collapsed = !!_mpSectionCollapsed[name];
    var map = { overview: 'mp-overview-content', stats: 'mp-stats-table', news: 'mp-news-content' };
    var el = document.getElementById(map[name]);
    var chev = document.getElementById('mp-chev-' + name);
    if (el) { el.style.maxHeight = collapsed ? '0' : '2000px'; el.style.opacity = collapsed ? '0' : '1'; el.style.marginTop = collapsed ? '0' : ''; }
    if (chev) chev.style.transform = collapsed ? 'rotate(-90deg)' : '';
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════

var _mpSectionDragInited = false;

function initMarketsPanel() {
    if (!document.getElementById('mp-chart-grid')) return;
    _mpLoadSections();
    if (!_mpSectionDragInited) { _mpInitSectionDrag(); _mpSectionDragInited = true; }

    if (_mpCharts.length > 0) {
        _mpRefreshAll(); _mpRenderOverview(); _mpRenderStatsTable(); _mpRenderNews();
        ['overview','stats','news'].forEach(function(n) { _mpApplySection(n); });
        _mpRestoreBottomLayout();
        return;
    }

    var saved = _mpLoadLayout();
    if (saved.length > 0) saved.forEach(function(s) { mpAddChart(s.symbol, s.timeframe, true, s.chartType, s.height); });
    else { var k = Object.keys(marketData); if (k[0]) mpAddChart(k[0], '1m', true); if (k[1]) mpAddChart(k[1], '1m', true); }

    _mpRenderOverview(); _mpRenderStatsTable(); _mpRenderNews();
    ['overview','stats','news'].forEach(function(n) { _mpApplySection(n); });
    _mpRestoreBottomLayout();

    if (typeof financeNewsData !== 'undefined' && financeNewsData.length === 0 && typeof loadExtraFeeds === 'function') {
        loadExtraFeeds('finance').then(function() { _mpRenderNews(); }).catch(function(){});
    }
}

// ═══════════════════════════════════════════════════════════
// SVG ICONS (inline to avoid Lucide replacement issues)
// ═══════════════════════════════════════════════════════════

var _gripSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="text-slate-600"><circle cx="9" cy="5" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="9" cy="12" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="19" r="1"/></svg>';
var _xSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
var _icoLine  = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>';
var _icoCandle= '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="18" y1="4" x2="18" y2="2"/><rect x="15" y="4" width="6" height="6" rx="1"/><line x1="6" y1="22" x2="6" y2="16"/><line x1="6" y1="10" x2="6" y2="8"/><rect x="3" y="10" width="6" height="6" rx="1"/></svg>';
var _icoArea  = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 20L7 14L11 16L16 8L21 12L21 20Z" fill="currentColor" opacity="0.15"/><polyline points="3 20 7 14 11 16 16 8 21 12"/></svg>';
var _icoCross = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>';
var _icoFib = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="22" x2="21" y2="2"/><line x1="3" y1="5" x2="21" y2="5" stroke-dasharray="2 2" opacity="0.5"/><line x1="3" y1="9" x2="21" y2="9" stroke-dasharray="2 2" opacity="0.5"/><line x1="3" y1="14" x2="21" y2="14" stroke-dasharray="2 2" opacity="0.5"/><line x1="3" y1="18" x2="21" y2="18" stroke-dasharray="2 2" opacity="0.5"/></svg>';
var _icoDraw = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.85 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>';
var _icoTrash = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>';
var _icoExport = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';

// ═══════════════════════════════════════════════════════════
// SERIES CREATION & CHART TYPE SWITCHING
// ═══════════════════════════════════════════════════════════

function _mpCreateSeries(chart, type) {
    if (type === 'candle') {
        return chart.addCandlestickSeries({ upColor:'#22c55e', downColor:'#ef4444', borderUpColor:'#22c55e', borderDownColor:'#ef4444', wickUpColor:'#22c55e80', wickDownColor:'#ef444480' });
    } else if (type === 'area') {
        return chart.addAreaSeries({ lineColor:'#10b981', topColor:'rgba(16,185,129,0.18)', bottomColor:'rgba(16,185,129,0.01)', lineWidth:2, crosshairMarkerVisible:true, crosshairMarkerRadius:3, lastValueVisible:true, priceLineVisible:true });
    } else {
        return chart.addLineSeries({ color:'#10b981', lineWidth:2, crosshairMarkerVisible:true, crosshairMarkerRadius:4, lastValueVisible:true, priceLineVisible:true });
    }
}

function mpSetChartType(id, type) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e || e.chartType === type) return;
    try { e.chart.removeSeries(e.series); } catch(x) {}
    e.series = _mpCreateSeries(e.chart, type);
    e.chartType = type;
    // Redraw drawings with new series coordinates after data loads
    ['line','candle','area'].forEach(function(t) {
        var btn = document.getElementById(id+'-btn-'+t);
        if (btn) { if (t===type) btn.classList.add('active'); else btn.classList.remove('active'); }
    });
    _mpUpdateChart(e);
    _mpSaveLayout();
    // Redraw canvas after chart updates (slight delay for series to render)
    setTimeout(function() { _mpRedrawCanvas(e); }, 100);
}

function mpToggleCrosshair(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e) return;
    e.crosshair = !e.crosshair;
    var btn = document.getElementById(id+'-btn-cross');
    if (btn) { if (e.crosshair) btn.classList.add('active'); else btn.classList.remove('active'); }
    e.chart.applyOptions({ crosshair: { mode: e.crosshair ? LightweightCharts.CrosshairMode.Normal : LightweightCharts.CrosshairMode.Hidden }});
}

// ═══════════════════════════════════════════════════════════
// CHART GRID
// ═══════════════════════════════════════════════════════════

function mpAddChart(symbol, timeframe, skipSave, chartType, chartHeight) {
    if (_mpCharts.length >= MP_MAX_CHARTS) return;
    if (!symbol) { var used = _mpCharts.map(function(c){return c.symbol;}); symbol = Object.keys(marketData).find(function(s){return !used.includes(s);}) || Object.keys(marketData)[0]; }
    if (!symbol || !marketData[symbol]) return;
    timeframe = timeframe || '1m';
    chartType = chartType || 'candle';
    chartHeight = chartHeight || 220;

    var id = 'mp-' + (++_mpCounter);
    var grid = document.getElementById('mp-chart-grid');
    var panel = document.createElement('div');
    panel.id = id;
    panel.className = 'mp-chart-panel glass-panel rounded-xl p-3 relative overflow-hidden';
    panel.style.cssText = 'min-width:0; border:1px solid rgba(51,65,85,0.4);';
    panel.dataset.mpId = id;

    var tfBtns = ['1m','5m','1d_1mo','1d_1y','1wk'].map(function(tf) {
        var labels = {'1m':'1D','5m':'5D','1d_1mo':'1M','1d_1y':'1Y','1wk':'5Y'};
        return '<button onclick="mpChangeTimeframe(\'' + id + '\',\'' + tf + '\')" data-mptf="' + tf + '" class="mp-tf-btn text-[9px] px-1.5 py-0.5 rounded transition-all ' + (tf===timeframe?'bg-slate-600 text-white':'text-slate-500 hover:text-slate-300') + '">' + labels[tf] + '</button>';
    }).join('');
    var opts = Object.keys(marketData).map(function(s){return '<option value="' + s + '" ' + (s===symbol?'selected':'') + '>' + s.replace('-USD','').replace('=F','') + ' \u2014 ' + getAssetName(s,marketData[s]?.name) + '</option>';}).join('');

    // Chart tools toolbar
    var tools = '<div class="flex items-center justify-between gap-1 mb-1.5">';
    tools += '<div class="flex items-center gap-1">';
    tools += '<button onclick="mpSetChartType(\'' + id + '\',\'line\')" id="' + id + '-btn-line" class="chart-tool-btn px-2 py-1 rounded text-[10px]' + (chartType==='line'?' active':'') + '" title="Line">' + _icoLine + '</button>';
    tools += '<button onclick="mpSetChartType(\'' + id + '\',\'candle\')" id="' + id + '-btn-candle" class="chart-tool-btn px-2 py-1 rounded text-[10px]' + (chartType==='candle'?' active':'') + '" title="Candlestick">' + _icoCandle + '</button>';
    tools += '<button onclick="mpSetChartType(\'' + id + '\',\'area\')" id="' + id + '-btn-area" class="chart-tool-btn px-2 py-1 rounded text-[10px]' + (chartType==='area'?' active':'') + '" title="Area">' + _icoArea + '</button>';
    tools += '<div class="w-px h-4 mx-0.5" style="background:rgba(51,65,85,0.5);"></div>';
    tools += '<button onclick="mpToggleCrosshair(\'' + id + '\')" id="' + id + '-btn-cross" class="chart-tool-btn active px-2 py-1 rounded text-[10px]" title="Crosshair">' + _icoCross + '</button>';
    tools += '<div class="w-px h-4 mx-0.5" style="background:rgba(51,65,85,0.5);"></div>';
    tools += '<button onclick="mpToggleFib(\'' + id + '\')" id="' + id + '-btn-fib" class="chart-tool-btn px-2 py-1 rounded text-[10px]" title="Fibonacci / Pattern Analysis">' + _icoFib + '</button>';
    tools += '<div class="w-px h-4 mx-0.5" style="background:rgba(51,65,85,0.5);"></div>';
    tools += '<button onclick="mpToggleDraw(\'' + id + '\')" id="' + id + '-btn-draw" class="chart-tool-btn px-2 py-1 rounded text-[10px]" title="Draw trend lines">' + _icoDraw + '</button>';
    tools += '<div id="' + id + '-draw-colors" class="hidden flex items-center gap-1 ml-0.5">';
    tools += '<button onclick="mpSetDrawColor(\'' + id + '\',\'#34d399\')" class="w-3.5 h-3.5 rounded-full border border-transparent hover:border-white/50" style="background:#34d399;" title="Green"></button>';
    tools += '<button onclick="mpSetDrawColor(\'' + id + '\',\'#f87171\')" class="w-3.5 h-3.5 rounded-full border border-transparent hover:border-white/50" style="background:#f87171;" title="Red"></button>';
    tools += '<button onclick="mpSetDrawColor(\'' + id + '\',\'#60a5fa\')" class="w-3.5 h-3.5 rounded-full border border-transparent hover:border-white/50" style="background:#60a5fa;" title="Blue"></button>';
    tools += '<button onclick="mpSetDrawColor(\'' + id + '\',\'#fbbf24\')" class="w-3.5 h-3.5 rounded-full border border-transparent hover:border-white/50" style="background:#fbbf24;" title="Yellow"></button>';
    tools += '<button onclick="mpClearDraw(\'' + id + '\')" class="chart-tool-btn px-1.5 py-0.5 rounded text-[9px] ml-0.5" title="Clear drawings">' + _icoTrash + '</button>';
    tools += '</div>';
    tools += '<div class="w-px h-4 mx-0.5" style="background:rgba(51,65,85,0.5);"></div>';
    tools += '<button onclick="mpExportChartPNG(\'' + id + '\')" class="chart-tool-btn px-2 py-1 rounded text-[10px] flex items-center gap-1" title="Export chart as PNG (S)">' + _icoExport + ' <span class="text-[9px]">Export</span></button>';
    tools += '</div>';
    // Focus Mode button
    tools += '<button onclick="_openFullscreenChart(document.getElementById(\'' + id + '-chartwrap\'), document.querySelector(\'#' + id + ' select\')?.selectedOptions[0]?.textContent || \'Chart\')" class="text-[10px] font-semibold px-2 py-1 rounded-md transition-all flex items-center gap-1 flex-shrink-0" style="color:var(--text-secondary); border:1px solid var(--border-strong); background:var(--bg-hover);" onmouseenter="this.style.color=\'var(--accent)\';this.style.borderColor=\'var(--accent)\';this.style.background=\'rgba(var(--accent-rgb,52,211,153),0.1)\'" onmouseleave="this.style.color=\'var(--text-secondary)\';this.style.borderColor=\'var(--border-strong)\';this.style.background=\'var(--bg-hover)\'" title="Focus mode"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></svg> Focus</button>';
    tools += '</div>';

    // Row 1: grip + ticker + timeframe + close
    var row1 = '<div class="flex items-center justify-between mb-2">';
    row1 += '<div class="flex items-center gap-2" style="min-width:0;">';
    row1 += '<div class="mp-chart-grip cursor-grab active:cursor-grabbing p-1 -ml-1 rounded hover:bg-slate-800/50" title="Click, then click another to swap">' + _gripSvg + '</div>';
    row1 += '<select onchange="mpChangeTicker(\'' + id + '\',this.value)" class="bg-transparent text-xs font-mono font-bold text-slate-200 border-none outline-none cursor-pointer" style="min-width:0;max-width:160px;appearance:none;-webkit-appearance:none;background-image:url(\'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2210%22 height=%226%22><path d=%22M0 0l5 6 5-6z%22 fill=%22%2364748b%22/></svg>\');background-repeat:no-repeat;background-position:right center;padding-right:14px;">' + opts + '</select>';
    row1 += '</div>';
    row1 += '<div class="flex items-center gap-1 flex-shrink-0">';
    row1 += '<div class="flex rounded p-0.5" style="background:rgba(15,23,42,0.5);">' + tfBtns + '</div>';
    row1 += '<button onclick="_mpRefreshSingle(\'' + id + '\')" class="mp-refresh-btn" title="Refresh this ticker" style="color:var(--text-muted);cursor:pointer;background:none;border:none;padding:2px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg></button>';
    row1 += '<button onclick="mpRemoveChart(\'' + id + '\')" class="ml-1 text-slate-600 hover:text-red-400 transition-colors p-0.5">' + _xSvg + '</button>';
    row1 += '</div></div>';

    // Row 2: price + change
    var row2 = '<div class="flex items-baseline gap-2 mb-1">';
    row2 += '<span id="' + id + '-price" class="text-lg font-bold text-white">--</span>';
    row2 += '<span id="' + id + '-change" class="text-xs font-mono font-bold">--</span>';
    row2 += '<span id="' + id + '-tflabel" class="text-[10px] text-slate-500"></span>';
    row2 += '</div>';

    // Chart container
    var chartDiv = '<div id="' + id + '-chartwrap" class="relative w-full rounded overflow-hidden mp-chartwrap" style="height:'+chartHeight+'px;border:1px solid rgba(51,65,85,0.3);">';
    chartDiv += '<div id="' + id + '-chart" style="width:100%;height:100%;"></div>';
    chartDiv += '<canvas id="' + id + '-drawcanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;" width="0" height="0"></canvas>';
    // Scroll hint overlays
    chartDiv += '<div class="mp-scroll-fade-l"></div><div class="mp-scroll-fade-r"></div>';
    chartDiv += '<div class="mp-scroll-track"><div class="mp-scroll-thumb" id="' + id + '-scrollthumb"></div></div>';
    chartDiv += '</div>';
    chartDiv += '<div class="mp-resize-handle flex items-center justify-center cursor-ns-resize select-none" data-mpid="'+id+'" style="height:8px;margin-top:-1px;"><div style="width:32px;height:3px;border-radius:2px;background:rgba(100,116,139,0.25);transition:background 0.15s;"></div></div>';

    // Stats grid
    var stats = '<div class="grid grid-cols-3 gap-x-3 gap-y-1 mt-2 text-[10px]">';
    stats += '<div class="flex justify-between"><span class="text-slate-500">Open</span><span id="' + id + '-open" class="text-slate-300 font-mono">--</span></div>';
    stats += '<div class="flex justify-between"><span class="text-slate-500">High</span><span id="' + id + '-high" class="text-emerald-400 font-mono">--</span></div>';
    stats += '<div class="flex justify-between"><span class="text-slate-500">Low</span><span id="' + id + '-low" class="text-red-400 font-mono">--</span></div>';
    stats += '<div class="flex justify-between"><span class="text-slate-500">Prev Close</span><span id="' + id + '-prev" class="text-slate-300 font-mono">--</span></div>';
    stats += '<div class="flex justify-between"><span class="text-slate-500">Volume</span><span id="' + id + '-vol" class="text-slate-300 font-mono">--</span></div>';
    stats += '<div class="flex justify-between"><span class="text-slate-500">Range</span><span id="' + id + '-range" class="text-slate-300 font-mono">--</span></div>';
    stats += '</div>';

    // Analysis
    var analysis = '<div id="' + id + '-analysis" class="mt-2 pt-2 border-t border-slate-800/30"></div>';
    // Fib panel lives here (separate from analysis so refreshes don't destroy it)
    var fibContainer = '<div id="' + id + '-fibpanel" class=""></div>';

    panel.innerHTML = row1 + row2 + tools + chartDiv + stats + analysis + fibContainer;
    grid.appendChild(panel);

    // Click-to-swap: click a grip to select, click another to swap
    var grip = panel.querySelector('.mp-chart-grip');
    if (grip) {
        grip.addEventListener('click', function(e) {
            e.stopPropagation();
            if (_mpSwapSelected === id) {
                // Deselect
                _mpSwapSelected = null;
                panel.style.borderColor = '';
                panel.style.boxShadow = '';
            } else if (_mpSwapSelected) {
                // Swap with selected
                var fi = _mpCharts.findIndex(function(c){return c.id===_mpSwapSelected;});
                var ti = _mpCharts.findIndex(function(c){return c.id===id;});
                if (fi >= 0 && ti >= 0) {
                    var tmp = _mpCharts[fi]; _mpCharts[fi] = _mpCharts[ti]; _mpCharts[ti] = tmp;
                    grid.innerHTML = ''; _mpCharts.forEach(function(c){grid.appendChild(c.el);});
                    _mpUpdateGridCols(); _mpSaveLayout();
                }
                // Clear selection state on all panels
                _mpSwapSelected = null;
                document.querySelectorAll('.mp-chart-panel').forEach(function(p) { p.style.borderColor = ''; p.style.boxShadow = ''; });
            } else {
                // Select this panel
                _mpSwapSelected = id;
                panel.style.borderColor = 'rgba(16,185,129,0.5)';
                panel.style.boxShadow = '0 0 0 1px rgba(16,185,129,0.3), inset 0 0 12px rgba(16,185,129,0.03)';
            }
        });
    }

    lucide.createIcons({ nameAttr: 'data-lucide' });

    var chartEl = document.getElementById(id + '-chart');
    var chart = LightweightCharts.createChart(chartEl, {
        width: chartEl.clientWidth, height: chartHeight,
        layout: { background:{type:'solid',color:'transparent'}, textColor:'#64748b', fontSize:9, fontFamily:"'ui-monospace','SFMono-Regular',monospace" },
        grid: { vertLines:{color:'rgba(148,163,184,0.04)'}, horzLines:{color:'rgba(148,163,184,0.04)'} },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal, vertLine:{color:'rgba(148,163,184,0.2)',width:1,style:2,labelBackgroundColor:'#1e293b'}, horzLine:{color:'rgba(148,163,184,0.2)',width:1,style:2,labelBackgroundColor:'#1e293b'} },
        rightPriceScale: { borderColor:'rgba(51,65,85,0.3)', scaleMargins:{top:0.08,bottom:0.08} },
        timeScale: { borderColor:'rgba(51,65,85,0.3)', timeVisible:true, secondsVisible:false, rightOffset:3, minBarSpacing:2.5 },
        handleScroll:{mouseWheel:true,pressedMouseMove:true}, handleScale:{mouseWheel:true,pinch:true},
    });

    var series = _mpCreateSeries(chart, chartType);

    var _mpRoTimer = null;
    var ro = new ResizeObserver(function() {
        if (_mpRoTimer) cancelAnimationFrame(_mpRoTimer);
        _mpRoTimer = requestAnimationFrame(function() {
            if (chart) chart.applyOptions({width: chartEl.clientWidth});
            var cv = document.getElementById(id+'-drawcanvas');
            var wrap = document.getElementById(id+'-chartwrap');
            if (cv && wrap) { cv.width = wrap.clientWidth; cv.height = wrap.clientHeight; }
            var ent = _mpCharts.find(function(c){return c.id===id;});
            if (ent && ent.drawLines.length) _mpRedrawCanvas(ent);
        });
    });
    ro.observe(chartEl);
    _mpKillLogo(chartEl);

    // Draw + Fib mode: click to place points
    chart.subscribeClick(function(param) {
        var ent = _mpCharts.find(function(c){return c.id===id;});
        if (!ent || !param.time || !param.point) return;
        var pr = ent.series.coordinateToPrice(param.point.y);
        if (pr == null) return;

        // Fibonacci mode
        if (ent.fibMode) {
            if (!ent.fibP1) {
                ent.fibP1 = {time:param.time, price:pr};
            } else {
                ent.fibP2 = {time:param.time, price:pr};
                ent.fibMode = false;
                var btn = document.getElementById(id+'-btn-fib');
                if (btn) btn.classList.remove('active');
                // Re-enable scroll
                ent.chart.applyOptions({handleScroll:{mouseWheel:true,pressedMouseMove:true},handleScale:{mouseWheel:true,axisPressedMouseMove:true}});
                _mpDrawFib(ent);
            }
            return;
        }

        // Draw mode
        if (ent.drawMode) {
            if (!ent.drawPending) {
                ent.drawPending = {time:param.time, price:pr};
            } else {
                ent.drawLines.push({p1:ent.drawPending, p2:{time:param.time,price:pr}, color:ent.drawColor});
                ent.drawPending = null;
                _mpRedrawCanvas(ent);
            }
        }
    });

    // Live preview line for draw AND fib modes
    chart.subscribeCrosshairMove(function(param) {
        var ent = _mpCharts.find(function(c){return c.id===id;});
        if (!ent || !param.point) return;
        
        // Fib preview
        if (ent.fibMode && ent.fibP1) {
            _mpRedrawCanvas(ent);
            var cv = document.getElementById(id+'-drawcanvas');
            if (!cv) return;
            var ctx = cv.getContext('2d');
            var x1 = ent.chart.timeScale().timeToCoordinate(ent.fibP1.time);
            var y1 = ent.series.priceToCoordinate(ent.fibP1.price);
            if (x1==null||y1==null) return;
            ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(param.point.x,param.point.y);
            ctx.strokeStyle='rgba(251,191,36,0.6)'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]);
            // Draw dot at first point
            ctx.beginPath(); ctx.arc(x1,y1,4,0,Math.PI*2); ctx.fillStyle='#fbbf24'; ctx.fill();
            return;
        }
        
        // Draw preview
        if (ent.drawMode && ent.drawPending) {
            _mpRedrawCanvas(ent);
            var cv2 = document.getElementById(id+'-drawcanvas');
            if (!cv2) return;
            var ctx2 = cv2.getContext('2d');
            var dx1 = ent.chart.timeScale().timeToCoordinate(ent.drawPending.time);
            var dy1 = ent.series.priceToCoordinate(ent.drawPending.price);
            if (dx1==null||dy1==null) return;
            ctx2.beginPath(); ctx2.moveTo(dx1,dy1); ctx2.lineTo(param.point.x,param.point.y);
            ctx2.strokeStyle=ent.drawColor; ctx2.lineWidth=1.5; ctx2.setLineDash([4,3]); ctx2.stroke(); ctx2.setLineDash([]);
        }
    });

    // Redraw on scroll/zoom
    chart.timeScale().subscribeVisibleTimeRangeChange(function() {
        var ent = _mpCharts.find(function(c){return c.id===id;});
        if (!ent) return;
        if (ent._fibPrices) _mpDrawFibCanvas(ent);
        else _mpRedrawCanvas(ent);
        _mpUpdateScrollThumb(ent);
    });

    var entry = { id:id, symbol:symbol, timeframe:timeframe, chart:chart, series:series, el:panel, ro:ro, chartType:chartType, chartHeight:chartHeight, crosshair:true, drawMode:false, drawColor:'#34d399', drawLines:[], drawPending:null, fibMode:false, fibPinned:false, fibP1:null, fibP2:null, fibSeries:[], pinnedFibs:[] };
    _mpCharts.push(entry); _mpUpdateGridCols(); _mpUpdateChart(entry);
    if (!skipSave) _mpSaveLayout(); _mpUpdateAddBtn();

    // Resize handle interaction
    var handle = panel.querySelector('.mp-resize-handle');
    if (handle) {
        handle.addEventListener('mousedown', function(re) {
            re.preventDefault(); re.stopPropagation();
            var startY = re.clientY, startH = entry.chartHeight;
            var bar = handle.firstElementChild;
            if (bar) bar.style.background = 'rgba(16,185,129,0.5)';
            var onMove = function(me) {
                var newH = Math.max(120, Math.min(600, startH + me.clientY - startY));
                entry.chartHeight = newH;
                var wrap = document.getElementById(entry.id + '-chartwrap');
                if (wrap) wrap.style.height = newH + 'px';
                entry.chart.applyOptions({height: newH});
            };
            var onUp = function() {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                if (bar) bar.style.background = '';
                _mpSaveLayout();
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
        handle.addEventListener('mouseenter', function() { handle.firstElementChild.style.background = 'rgba(100,116,139,0.5)'; });
        handle.addEventListener('mouseleave', function() { handle.firstElementChild.style.background = ''; });
    }
}

function mpToggleChart(symbol, timeframe) {
    var existing = _mpCharts.find(function(c) { return c.symbol === symbol; });
    if (existing) {
        mpRemoveChart(existing.id);
    } else {
        mpAddChart(symbol, timeframe);
    }
}

function mpRemoveChart(id) {
    var idx = _mpCharts.findIndex(function(c){return c.id===id;}); if (idx < 0) return;
    var e = _mpCharts[idx]; try{e.ro.disconnect();}catch(x){} try{e.chart.remove();}catch(x){} e.el.remove();
    _mpCharts.splice(idx, 1); _mpUpdateGridCols(); _mpSaveLayout(); _mpUpdateAddBtn();
}
function mpChangeTicker(id, sym) { var e = _mpCharts.find(function(c){return c.id===id;}); if (!e) return; e.symbol = sym; _mpUpdateChart(e); _mpSaveLayout(); }
function mpChangeTimeframe(id, tf) {
    var e = _mpCharts.find(function(c){return c.id===id;}); if (!e) return; e.timeframe = tf;
    e.el.querySelectorAll('.mp-tf-btn').forEach(function(b) { b.className = 'mp-tf-btn text-[9px] px-1.5 py-0.5 rounded transition-all ' + (b.dataset.mptf===tf?'bg-slate-600 text-white':'text-slate-500 hover:text-slate-300'); });
    _mpUpdateChart(e); _mpSaveLayout();
}

// ═══════════════════════════════════════════════════════════
// PER-CARD REFRESH — fetches latest data for a single ticker
// ═══════════════════════════════════════════════════════════

window._mpRefreshSingle = function(id) {
    var entry = _mpCharts.find(function(c) { return c.id === id; });
    if (!entry) return;
    var btn = document.querySelector('#mp-card-' + id + ' .mp-refresh-btn svg');
    if (btn) btn.classList.add('animate-spin');

    var sym = entry.symbol || id;
    var interval = entry.interval || entry.timeframe || '1d';
    fetch('/api/market-tick?symbol=' + encodeURIComponent(sym) + '&interval=' + encodeURIComponent(interval))
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data && data.data) {
                marketData[sym] = data.data;
                _mpUpdateChart(entry);
                if (typeof showToast === 'function') showToast(sym + ' refreshed', 'success');
            }
        })
        .catch(function(err) {
            console.warn('[MP] Refresh failed for', sym, err);
            if (typeof showToast === 'function') showToast('Failed to refresh ' + sym, 'error');
        })
        .finally(function() {
            if (btn) btn.classList.remove('animate-spin');
        });
};

// ═══════════════════════════════════════════════════════════
// UPDATE CHART DATA — handles candle, line, and area types
// ═══════════════════════════════════════════════════════════

function _mpUpdateChart(entry) {
    var ad = _resolveData(entry.symbol, entry.timeframe); if (!ad) return;
    var price=ad.price||0, change=ad.change||0, up=change>=0;
    var tfL={'1m':'Today','5m':'Past 5 Days','1d_1mo':'Past Month','1d_1y':'Past Year','1wk':'Past 5 Years'};
    var s=function(sfx,v){var el=document.getElementById(entry.id+sfx);if(el)el.textContent=v;};
    s('-price','$'+_mpFp(price));
    var ch=document.getElementById(entry.id+'-change'); if(ch){ch.textContent=(up?'+':'')+change.toFixed(2)+'%';ch.className='text-xs font-mono font-bold '+(up?'text-emerald-400':'text-red-400');}
    s('-tflabel',tfL[entry.timeframe]||''); s('-open',_mpFp(ad.open)); s('-high',_mpFp(ad.high)); s('-low',_mpFp(ad.low));
    s('-prev',_mpFp(ad.prev_close)); s('-vol',ad.volume?formatVolume(ad.volume):'--');
    s('-range',ad.low&&ad.high?_mpFp(ad.low)+' \u2013 '+_mpFp(ad.high):'--');

    var hist=ad.history||[],ts=ad.timestamps||[],op=ad.opens||[],hi=ad.highs||[],lo=ad.lows||[];
    var hasOHLC=op.length===hist.length&&hi.length===hist.length;
    var isCandle = entry.chartType === 'candle';
    var seen={}; var cd=[];
    for(var i=0;i<hist.length;i++){
        var t=ts[i]!=null?(typeof ts[i]==='number'?ts[i]+_mpTzOff:Math.floor(new Date(ts[i]).getTime()/1000)+_mpTzOff):Math.floor(Date.now()/1000)+_mpTzOff-(hist.length-1-i)*60;
        if(seen[t])continue;seen[t]=1;
        if (isCandle) {
            cd.push(hasOHLC?{time:t,open:op[i]||hist[i],high:hi[i]||hist[i],low:lo[i]||hist[i],close:hist[i]}:{time:t,open:hist[i],high:hist[i],low:hist[i],close:hist[i]});
        } else {
            cd.push({time:t, value:hist[i]});
        }
    }
    cd.sort(function(a,b){return a.time-b.time;});

    entry.series.setData(cd);
    // Use logical range (bar count) for readable candle sizing
    var NB={'1m':120,'5m':78,'1d_1mo':60,'1d_1y':120,'1wk':100};
    var nb=Math.min(NB[entry.timeframe]||80, cd.length);
    if(cd.length>=2){entry.chart.timeScale().setVisibleLogicalRange({from:cd.length-nb,to:cd.length});}else entry.chart.timeScale().fitContent();
    _mpKillLogo(document.getElementById(entry.id+'-chart'));
    _mpAnalysis(entry,ad);
    _mpUpdateScrollThumb(entry);
}

function _mpAnalysis(entry, ad) {
    var el=document.getElementById(entry.id+'-analysis'); if(!el)return;
    var hist=ad.history||[],price=ad.price||0,change=ad.change||0,up=change>=0,mag=Math.abs(change),high=ad.high||0,low=ad.low||0;
    var mom='Neutral',mc='text-slate-400';
    if(hist.length>=28){var r=hist[hist.length-1]/hist[hist.length-14]-1,p=hist[hist.length-14]/hist[hist.length-28]-1,m=r-p;if(m>0.02){mom='Accelerating';mc='text-emerald-400';}else if(m>0.005){mom='Building';mc='text-emerald-400/70';}else if(m<-0.02){mom='Decelerating';mc='text-red-400';}else if(m<-0.005){mom='Fading';mc='text-red-400/70';}}
    var vv=0,vw=Math.min(20,hist.length-1);if(vw>=5){var rets=[];for(var i=hist.length-vw;i<hist.length;i++){if(hist[i-1]>0)rets.push((hist[i]-hist[i-1])/hist[i-1]);}var mn=rets.reduce(function(a,b){return a+b;},0)/rets.length;vv=Math.sqrt(rets.reduce(function(a,v){return a+(v-mn)*(v-mn);},0)/rets.length)*100;}
    var vl='Low',vc='text-slate-400';if(vv>3){vl='Extreme';vc='text-red-400';}else if(vv>1.5){vl='High';vc='text-amber-400';}else if(vv>0.5){vl='Moderate';vc='text-yellow-400';}
    var rp=(high>low)?((price-low)/(high-low)*100):50;
    var tr='flat';if(mag>5)tr=up?'surging':'plunging';else if(mag>2)tr=up?'rising':'falling';else if(mag>0.5)tr=up?'up':'down';
    var sym=entry.symbol.replace('-USD','').replace('=F','');
    el.innerHTML='<div class="flex items-center gap-2 mb-1.5 flex-wrap"><span class="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded '+(up?'bg-emerald-500/10 text-emerald-400':'bg-red-500/10 text-red-400')+'">'+(up?'\u2191 Bull':'\u2193 Bear')+'</span><span class="text-[9px] '+mc+' font-bold">'+mom+'</span><span class="text-[9px] text-slate-600">\u00b7</span><span class="text-[9px] '+vc+'">Vol: '+vl+'</span></div><div class="mb-1.5"><div class="flex justify-between text-[9px] text-slate-500 mb-0.5"><span>$'+_mpFp(low)+'</span><span class="text-slate-400">'+rp.toFixed(0)+'%</span><span>$'+_mpFp(high)+'</span></div><div class="w-full h-1 bg-slate-800 rounded-full relative overflow-hidden"><div class="absolute inset-0 rounded-full" style="background:linear-gradient(90deg,#ef4444,#fbbf24,#22c55e);opacity:0.25;"></div><div class="absolute top-0 h-full w-0.5 rounded-full bg-white" style="left:'+Math.min(99,Math.max(1,rp))+'%;"></div></div></div><p class="text-[10px] text-slate-500">'+sym+' '+tr+' at $'+_mpFp(price)+', '+(up?'up':'down')+' '+mag.toFixed(2)+'%.</p>';
}

function _mpRefreshAll() { _mpCharts.forEach(function(c){_mpUpdateChart(c);}); }

function _mpUpdateGridCols() {
    var grid=document.getElementById('mp-chart-grid'); if(!grid)return;
    var n=_mpCharts.length;
    _mpCharts.forEach(function(c){c.el.style.gridColumn='';});
    if(n<=1){grid.style.gridTemplateColumns='1fr';return;}
    var cols=n>=5?3:2; grid.style.gridTemplateColumns='repeat('+cols+',minmax(0,1fr))';
    if(n%cols===1&&n>1)_mpCharts[n-1].el.style.gridColumn='1/-1';
}

function _mpUpdateAddBtn(){var b=document.getElementById('mp-add-btn');if(!b)return;var f=_mpCharts.length>=MP_MAX_CHARTS;b.disabled=f;b.style.opacity=f?'0.3':'';}
function _mpKillLogo(el){if(!el)return;var k=function(){el.querySelectorAll('a[href*="tradingview"],a[href*="lightweight-charts"]').forEach(function(a){a.style.display='none';});};setTimeout(k,200);setTimeout(k,600);setTimeout(k,1200);}

// ═══════════════════════════════════════════════════════════
// MARKET OVERVIEW — Merged overview + heatmap
// ═══════════════════════════════════════════════════════════

function _mpRenderOverview() {
    var c = document.getElementById('mp-overview-content');
    if (!c || !Object.keys(marketData).length) return;

    var items = Object.keys(marketData).map(function(s) {
        var a = _resolveData(s,'1m') || _resolveData(s,'5m') || {};
        return { sym:s, ad:a, change:a.change||0, price:a.price||0 };
    }).sort(function(a,b) { return b.change - a.change; });

    var maxAbs = Math.max(1, Math.max.apply(null, items.map(function(i){return Math.abs(i.change);})));

    var h = '<div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-2">';
    items.forEach(function(item) {
        var ad = item.ad; if (!ad) return;
        var l = item.sym.replace('-USD','').replace('=F','');
        var name = getAssetName(item.sym, (marketData[item.sym]||{}).name);
        var ch = item.change, up = ch >= 0, price = item.price;
        var intensity = Math.min(1, Math.abs(ch) / maxAbs);
        var hist = ad.history || [];

        var bg = up
            ? 'rgba(0,'+Math.round(180*intensity)+',80,'+(0.06+intensity*0.18)+')'
            : 'rgba('+Math.round(200*intensity)+',0,40,'+(0.06+intensity*0.18)+')';
        var border = up
            ? 'rgba(34,197,94,'+(0.12+intensity*0.25)+')'
            : 'rgba(239,68,68,'+(0.12+intensity*0.25)+')';

        var spark = '';
        if (hist.length >= 4) {
            var pts = hist.slice(-24), mn = Math.min.apply(null,pts), mx = Math.max.apply(null,pts), rng = mx-mn||1, w = 56, ht = 18;
            spark = '<svg width="'+w+'" height="'+ht+'" class="mt-1 opacity-70"><path d="'+pts.map(function(v,i){return(i===0?'M':'L')+(i/(pts.length-1)*w).toFixed(1)+','+(ht-2-((v-mn)/rng)*(ht-4)).toFixed(1);}).join(' ')+'" fill="none" stroke="'+(up?'#10b981':'#ef4444')+'" stroke-width="1.2"/></svg>';
        }

        h += '<div onclick="mpToggleChart(\'' + item.sym + '\',\'1m\')" class="cursor-pointer rounded-lg p-2 transition-all hover:scale-[1.02]" style="background:'+bg+';border:1px solid '+border+';">';
        h += '<div class="flex items-center justify-between"><span class="text-[11px] font-mono font-bold text-white">'+l+'</span><span class="text-[10px] font-mono font-bold '+(up?'text-emerald-400':'text-red-400')+'">'+(up?'+':'')+ch.toFixed(2)+'%</span></div>';
        h += '<div class="text-[9px] text-slate-500 truncate">'+_mpEsc(name)+'</div>';
        h += '<div class="text-[10px] font-mono text-slate-300 mt-0.5">$'+_mpFp(price)+'</div>';
        h += spark;
        h += '</div>';
    });
    h += '</div>';
    c.innerHTML = h;
}


// ═══════════════════════════════════════════════════════════
// MARKET AGENT — embedded chat for market analysis
// ═══════════════════════════════════════════════════════════

var _mpAgentHistory = [];
var _mpAgentStreaming = false;
function _mpCapHistory() { if (_mpAgentHistory.length > 100) _mpAgentHistory.splice(0, _mpAgentHistory.length - 100); }

function _mpRenderStatsTable() {
    _mpAgentInitSuggestions();
    _mpAgentCheckStatus();
    _mpAgentInitResize();
    try { lucide.createIcons({ nameAttr:'data-lucide' }); } catch(e){}
}

// ── Market context builder ──
function _mpBuildMarketContext() {
    var ctx = '';
    var syms = Object.keys(marketData);
    if (syms.length) {
        ctx += '\n[Current Watchlist Prices]\n';
        syms.forEach(function(sym) {
            var ad = _resolveData(sym,'1m') || _resolveData(sym,'5m') || {};
            var clean = sym.replace('-USD','').replace('=F','');
            var name = getAssetName(sym, (marketData[sym]||{}).name);
            if (ad.price) {
                var ch = ad.change || 0, up = ch >= 0;
                ctx += clean + ' (' + name + '): $' + _mpFp(ad.price) + ' ' + (up?'+':'') + ch.toFixed(2) + '%';
                if (ad.high && ad.low) ctx += ' H:$' + _mpFp(ad.high) + ' L:$' + _mpFp(ad.low);
                ctx += '\n';
            }
        });
    }
    return ctx;
}

// ── Suggestions ──
function _mpAgentInitSuggestions() {
    var el = document.getElementById('mp-agent-suggestions');
    if (!el) return;
    var suggestions = [];
    var syms = Object.keys(typeof marketData !== 'undefined' ? marketData : {}).map(function(s){ return s.replace('-USD','').replace('=F',''); });
    suggestions.push('\uD83D\uDCCA Market summary');
    if (syms.length >= 2) {
        suggestions.push('Compare ' + syms[0] + ' and ' + syms[1]);
        suggestions.push('How is ' + syms[0] + ' doing?');
        suggestions.push('Chart ' + syms[0]);
    }
    suggestions.push('Show my watchlist');
    suggestions.push('Any buying opportunities?');
    suggestions.push('Show my categories');
    if (syms.length) suggestions.push('Analyze ' + syms[Math.floor(Math.random()*syms.length)] + ' trend');

    var picks = suggestions.sort(function(){ return Math.random()-0.5; }).slice(0,5);
    var btns = '';
    picks.forEach(function(s) {
        btns += '<button onclick="mpSendMarketAgent(this.textContent)" class="text-[9px] px-2 py-1 rounded-lg transition-all cursor-pointer whitespace-nowrap" style="border:1px solid rgba(51,65,85,0.5);color:#64748b;background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor=\'#34d399\';this.style.color=\'#34d399\';this.style.background=\'rgba(16,185,129,0.06)\'" onmouseleave="this.style.borderColor=\'rgba(51,65,85,0.5)\';this.style.color=\'#64748b\';this.style.background=\'rgba(255,255,255,0.02)\'">' + _mpEsc(s) + '</button>';
    });
    el.innerHTML = btns;
}

// ── Status check ──
function _mpAgentCheckStatus() {
    var dot = document.getElementById('mp-agent-status-dot');
    var badge = document.getElementById('mp-agent-model-badge');
    var serperBadge = document.getElementById('mp-agent-serper-badge');
    // Check agent availability
    fetch('/api/agent-status').then(function(r){ return r.json(); }).then(function(d) {
        if (dot) dot.style.background = d.available ? 'rgb(16,185,129)' : 'rgb(245,158,11)';
        if (badge && d.model) badge.textContent = d.model.split(':')[0];
    }).catch(function() {
        if (dot) dot.style.background = 'rgb(239,68,68)';
    });
    // Check serper from config (same as main agent)
    var headers = {
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
        'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
    };
    fetch('/api/config', {headers:headers}).then(function(r){ return r.json(); }).then(function(cfg) {
        var hasSerper = cfg.search && cfg.search.serper_api_key && cfg.search.serper_api_key !== 'YOUR_SERPER_API_KEY';
        if (serperBadge) {
            if (hasSerper) {
                serperBadge.innerHTML = '<span class="inline-block w-1.5 h-1.5 rounded-full mr-0.5" style="background:#10b981;"></span> Web';
                serperBadge.style.color = '#34d399';
            } else {
                serperBadge.innerHTML = '<span class="inline-block w-1.5 h-1.5 rounded-full mr-0.5" style="background:var(--text-muted,#64748b);opacity:0.4;"></span> No web';
                serperBadge.style.color = 'var(--text-muted,#64748b)';
            }
        }
    }).catch(function(){});
}

// ── Resize handle ──
function _mpAgentInitResize() {
    var handle = document.getElementById('mp-agent-resize');
    var msgs = document.getElementById('mp-agent-messages');
    if (!handle || !msgs || handle._mpResizeInit) return;
    handle._mpResizeInit = true;
    var startY, startH;
    handle.addEventListener('mousedown', function(e) {
        e.preventDefault();
        startY = e.clientY; startH = msgs.offsetHeight;
        var onMove = function(me) { msgs.style.height = Math.min(600, Math.max(100, startH + (me.clientY - startY))) + 'px'; };
        var onUp = function() { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); document.body.style.cursor = ''; document.body.style.userSelect = ''; };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
    });
}

// ── Export ──
function mpExportMarketAgent() {
    if (!_mpAgentHistory.length) { if (typeof showToast === 'function') showToast('No chat to export','warning'); return; }
    var data = { version:1, agent:'strat-market', exported_at:new Date().toISOString(), messages:_mpAgentHistory };
    var blob = new Blob([JSON.stringify(data,null,2)], {type:'application/json'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = 'strat-market-chat_' + new Date().toISOString().slice(0,10) + '.json'; a.click();
    URL.revokeObjectURL(url);
    if (typeof showToast === 'function') showToast('Market chat exported','success');
}
function mpImportMarketAgent() {
    var inp = document.createElement('input');
    inp.type = 'file'; inp.accept = '.json,.txt';
    inp.onchange = function() {
        var file = inp.files && inp.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function(ev) {
            var raw = (ev.target.result || '').trim();
            if (!raw) return;
            try {
                var data = JSON.parse(raw);
                if (data.messages && Array.isArray(data.messages) && data.messages.length) {
                    _mpAgentHistory = data.messages;
                    var msgs = document.getElementById('mp-agent-messages');
                    if (!msgs) return;
                    msgs.innerHTML = '';
                    data.messages.forEach(function(m) {
                        if (m.role === 'assistant') {
                            _mpAgentAppend('assistant', typeof formatAgentText === 'function' ? formatAgentText(m.content) : _mpEsc(m.content));
                        } else {
                            _mpAgentAppend('user', m.content);
                        }
                    });
                    if (typeof showToast === 'function') showToast('Loaded ' + data.messages.length + ' messages', 'success');
                    return;
                }
            } catch(e) {}
            // Treat as context file
            var text = raw.slice(0, 5000);
            var contextMsg = '[Imported context from "' + file.name + '"]\n\n' + text;
            _mpAgentHistory.push({role:'user', content:contextMsg}); _mpCapHistory();
            var welcome = document.getElementById('mp-agent-welcome');
            if (welcome) welcome.remove();
            _mpAgentAppend('user', '\uD83D\uDCC4 Context loaded: ' + file.name);
            if (typeof showToast === 'function') showToast('Context loaded from ' + file.name, 'success');
        };
        reader.readAsText(file);
    };
    inp.click();
}


function mpHandleAgentImport(event) {
    var file = event.target.files && event.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function(e) {
        var raw = (e.target.result || '').trim();
        if (!raw) return;
        try {
            var data = JSON.parse(raw);
            if (data.messages && Array.isArray(data.messages) && data.messages.length) {
                _mpAgentHistory = data.messages;
                var msgs = document.getElementById('mp-agent-messages');
                if (!msgs) return;
                msgs.innerHTML = '';
                _mpAgentHistory.forEach(function(m) {
                    if (m.role === 'assistant') {
                        _mpAgentAppend('assistant', typeof formatAgentText === 'function' ? formatAgentText(m.content) : _mpEsc(m.content));
                    } else {
                        _mpAgentAppend('user', m.content);
                    }
                });
                msgs.scrollTop = msgs.scrollHeight;
                var info = data.exported_at ? ' from ' + new Date(data.exported_at).toLocaleDateString() : '';
                if (typeof showToast === 'function') showToast('Loaded ' + data.messages.length + ' messages' + info, 'success');
                return;
            }
        } catch(err) { /* not valid chat JSON */ }
        // Treat as context file
        var text = raw.slice(0, 5000);
        var contextMsg = '[Imported context from "' + (file.name || 'file') + '"]\n\n' + text;
        _mpAgentHistory.push({role:'user', content: contextMsg}); _mpCapHistory();
        var welcome = document.getElementById('mp-agent-welcome');
        if (welcome) welcome.remove();
        var preview = text.length > 200 ? text.slice(0, 200) + '\u2026' : text;
        _mpAgentAppend('assistant', '<div class="rounded-lg px-2.5 py-2 text-[10px]" style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);"><div class="flex items-center gap-1.5 mb-1"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><span class="text-emerald-400 font-semibold">Context loaded: ' + _mpEsc(file.name) + '</span></div><div class="text-slate-400 whitespace-pre-wrap" style="max-height:80px;overflow:hidden;">' + _mpEsc(preview) + '</div></div>');
        if (typeof showToast === 'function') showToast('Context loaded from ' + file.name, 'success');
        event.target.value = '';
    };
    reader.readAsText(file);
}

// ── Quick local commands (handled without AI) ──
function _mpHandleQuickCommand(msg) {
    var lower = msg.toLowerCase().trim();

    // Market summary — generates visual widget locally
    if (/^(market\s*summary|summary|top\s*movers|how.*market|what.*market)/i.test(lower)) {
        if (typeof marketData === 'undefined' || !marketData || !Object.keys(marketData).length) {
            return { handled:true, html:'<span class="text-slate-400">No market data loaded yet.</span>' };
        }
        var items = Object.keys(marketData).map(function(sym) {
            var ad = _resolveData(sym, '1m') || _resolveData(sym, '5m') || {};
            return { sym:sym, price:ad.price||0, change:ad.change||0 };
        }).sort(function(a,b){ return b.change - a.change; });

        var maxAbs = Math.max(1, Math.max.apply(null, items.map(function(i){return Math.abs(i.change);})));
        var html = '<div class="mb-1.5"><span class="text-[10px] font-bold text-slate-300">\uD83D\uDCCA Market Summary</span></div><div class="space-y-1">';
        items.forEach(function(item) {
            var label = item.sym.replace('-USD','').replace('=F','');
            var name = getAssetName(item.sym, (marketData[item.sym]||{}).name);
            var up = item.change >= 0;
            var barW = Math.min(100, (Math.abs(item.change) / maxAbs) * 100);
            html += '<div class="flex items-center gap-2 text-[10px]">';
            html += '<span class="font-mono font-bold text-slate-200 w-10 flex-shrink-0">' + label + '</span>';
            html += '<span class="text-slate-500 w-16 truncate flex-shrink-0">' + _mpEsc(name) + '</span>';
            html += '<span class="font-mono w-16 text-right text-slate-300 flex-shrink-0">$' + _mpFp(item.price) + '</span>';
            html += '<div class="flex-1 h-2.5 rounded-full overflow-hidden" style="background:rgba(51,65,85,0.3);min-width:40px;"><div class="h-full rounded-full" style="width:' + barW + '%;background:' + (up ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)') + ';"></div></div>';
            html += '<span class="font-mono font-bold w-14 text-right flex-shrink-0 ' + (up?'text-emerald-400':'text-red-400') + '">' + (up?'+':'') + item.change.toFixed(2) + '%</span>';
            html += '</div>';
        });
        html += '</div>';
        var top = items[0], bot = items[items.length-1];
        if (top && bot && items.length > 1) {
            html += '<div class="mt-1.5 pt-1.5 border-t border-slate-800/30 text-[10px] text-slate-500">';
            html += '\uD83C\uDFC6 Top: <span class="text-emerald-400 font-bold">' + top.sym.replace('-USD','').replace('=F','') + '</span> (' + (top.change>=0?'+':'') + top.change.toFixed(2) + '%)';
            html += ' &nbsp;\u00b7&nbsp; \uD83D\uDCC9 Bottom: <span class="text-red-400 font-bold">' + bot.sym.replace('-USD','').replace('=F','') + '</span> (' + bot.change.toFixed(2) + '%)';
            html += '</div>';
        }
        return { handled:true, html:html, plain:'Market summary' };
    }

    // Chart command: "chart NVDA" → opens chart panel
    var chartMatch = msg.match(/^chart\s+(.+)/i);
    if (chartMatch) {
        var sym = chartMatch[1].trim().toUpperCase();
        var actualSym = null;
        Object.keys(typeof marketData !== 'undefined' ? marketData : {}).forEach(function(k) {
            if (k === sym || k.replace('-USD','').replace('=F','') === sym) actualSym = k;
        });
        if (actualSym) {
            mpAddChart(actualSym, '1m');
            return { handled:true, html:'<span class="text-emerald-400">\uD83D\uDCC8 Opened chart for <span class="font-bold">' + _mpEsc(sym) + '</span></span>', plain:'Chart opened' };
        }
        return { handled:true, html:'<span class="text-amber-400">\u26A0 ' + _mpEsc(sym) + ' not in your watchlist. Try <span class="font-mono">"add ticker ' + _mpEsc(sym) + '"</span> first.</span>' };
    }

    return null;
}
function mpSendMarketAgent(prefill) {
    if (_mpAgentStreaming) return;
    var input = document.getElementById('mp-agent-input');
    var msg = prefill || (input ? input.value.trim() : '');
    if (!msg) return;
    if (input) input.value = '';

    // Remove welcome
    var welcome = document.getElementById('mp-agent-welcome');
    if (welcome) welcome.remove();

    // Add user bubble
    _mpAgentAppend('user', msg);

    // ── Quick local commands (no AI needed) ──
    var quick = _mpHandleQuickCommand(msg);
    if (quick && quick.handled) {
        _mpAgentHistory.push({role:'user', content:msg});
        _mpAgentHistory.push({role:'assistant', content: quick.plain || 'OK'}); _mpCapHistory();
        _mpAgentAppend('assistant', quick.html || _mpEsc(quick.plain || ''));
        return;
    }

    // ── Ticker/category command interception ──
    if (typeof _parseTickerCommand === 'function') {
        var cmd = _parseTickerCommand(msg);
        if (cmd) {
            _mpAgentHistory.push({role:'user', content:msg}); _mpCapHistory();
            var typingId2 = 'mp-typing-' + Date.now();
            _mpAgentAppend('typing', '', typingId2);
            _executeTickerCommand(cmd).then(function(result) {
                var t2 = document.getElementById(typingId2); if (t2) t2.remove();
                var text = result.plain || (typeof result === 'string' ? result : JSON.stringify(result));
                var html = result.html || (typeof formatAgentText === 'function' ? formatAgentText(text) : _mpEsc(text));
                _mpAgentHistory.push({role:'assistant', content:text}); _mpCapHistory();
                _mpAgentAppend('assistant', html);
            }).catch(function(err) {
                var t2 = document.getElementById(typingId2); if (t2) t2.remove();
                _mpAgentAppend('assistant', '<span class="text-amber-400">\u26A0 ' + _mpEsc(err.message || 'Command failed') + '</span>');
            });
            return;
        }
    }

    _mpAgentHistory.push({role:'user', content:msg}); _mpCapHistory();

    // Add typing indicator
    var typingId = 'mp-typing-' + Date.now();
    _mpAgentAppend('typing', '', typingId);

    _mpAgentStreaming = true;
    var sendBtn = document.getElementById('mp-agent-send');
    if (sendBtn) { sendBtn.disabled = true; sendBtn.style.opacity = '0.5'; }
    if (input) input.disabled = true;

    // Build context-enhanced message with current market data
    var contextMsg = msg;
    var mktCtx = _mpBuildMarketContext();
    var sysHint = '\n[Instructions: Be concise and actionable. Max 3-4 bullet points. Focus on specific impacts to the user\'s watchlist. Skip generic advice. No repeating data the user can already see.]';
    if (mktCtx) contextMsg = msg + '\n' + mktCtx + sysHint;
    else contextMsg = msg + sysHint;

    fetch('/api/agent-chat', {
        method:'POST',
        headers:{
            'Content-Type':'application/json',
            'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
            'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
        },
        body: JSON.stringify({ message: contextMsg, history: _mpAgentHistory.slice(-16) })
    }).then(function(res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var full = '';

        var typing = document.getElementById(typingId);
        if (typing) typing.remove();
        var respEl = _mpAgentAppend('assistant', '');

        function pump() {
            return reader.read().then(function(result) {
                if (result.done) {
                    full = full.replace(/[\u4e00-\u9fff\u3400-\u4dbf]+/g, '').replace(/\s{2,}/g, ' ').replace(/\.(\s*\.)+/g, '.').trim();
                    _mpAgentHistory.push({role:'assistant', content: full}); _mpCapHistory();
                    var rd = respEl.querySelector('.mp-agent-resp');
                    if (rd && typeof formatAgentText === 'function' && typeof wrapWithShowMore === 'function') rd.innerHTML = wrapWithShowMore(full, formatAgentText(full));
                    else if (rd && typeof formatAgentText === 'function') rd.innerHTML = formatAgentText(full);
                    _mpAgentDone();
                    return;
                }
                var chunk = decoder.decode(result.value, {stream:true});
                var lines = chunk.split('\n');
                lines.forEach(function(line) {
                    if (!line.startsWith('data: ')) return;
                    var raw = line.slice(6);
                    if (raw === '[DONE]') return;
                    try {
                        var payload = JSON.parse(raw);
                        if (payload.token) {
                            full += payload.token;
                        }
                        if (payload.status) {
                            var rd3 = respEl.querySelector('.mp-agent-resp');
                            if (rd3) {
                                var isSearch = payload.status.indexOf('web_search') >= 0;
                                var icon = isSearch ? '\uD83D\uDD0D' : '\u2699\uFE0F';
                                var label = payload.status.replace(/^[\uD83D\uDD0D\u2699\uFE0F]\s*/, '');
                                rd3.innerHTML = '<div class="flex items-center gap-1.5 py-1"><div class="flex gap-1"><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#34d399;animation-delay:0ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#34d399;animation-delay:150ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#34d399;animation-delay:300ms;"></span></div><span class="text-[9px] font-mono text-emerald-400">' + icon + ' ' + _mpEsc(label) + '</span></div>';
                            }
                        }
                        if (payload.error) {
                            full = '\u26A0 ' + payload.error;
                        }
                    } catch(e) {}
                });
                var rd2 = respEl.querySelector('.mp-agent-resp');
                if (rd2 && full) rd2.innerHTML = typeof formatAgentText === 'function' ? formatAgentText(full) : _mpEsc(full);
                var msgs = document.getElementById('mp-agent-messages');
                if (msgs) msgs.scrollTop = msgs.scrollHeight;
                return pump();
            });
        }
        return pump();
    }).catch(function(err) {
        var typing2 = document.getElementById(typingId);
        if (typing2) typing2.remove();
        _mpAgentAppend('assistant', '<span class="text-amber-400">\u26A0 ' + _mpEsc(err.message || 'Could not reach agent') + '</span>');
        _mpAgentDone();
    });
}

function _mpAgentDone() {
    _mpAgentStreaming = false;
    var sendBtn = document.getElementById('mp-agent-send');
    var input = document.getElementById('mp-agent-input');
    if (sendBtn) { sendBtn.disabled = false; sendBtn.style.opacity = ''; }
    if (input) { input.disabled = false; input.focus(); }
}

function _mpAgentAppend(role, content, elId) {
    var msgs = document.getElementById('mp-agent-messages');
    if (!msgs) return null;
    var div = document.createElement('div');
    if (elId) div.id = elId;
    var time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

    if (role === 'user') {
        div.className = 'flex justify-end';
        div.innerHTML = '<div class="max-w-[85%]"><div class="rounded-2xl rounded-br-sm px-3 py-2 text-[11px] leading-relaxed" style="color:var(--text-heading,#e2e8f0);background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.15);">' + _mpEsc(content) + '</div><div class="text-[8px] mt-0.5 text-right" style="color:var(--text-muted);opacity:0.4;">' + time + '</div></div>';
    } else if (role === 'typing') {
        div.className = 'flex gap-2';
        div.innerHTML = '<div class="w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 mt-0.5" style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.15);"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2"><path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 007.92 12.446A9 9 0 1112 3z"/><path d="M19 3v4"/><path d="M21 5h-4"/></svg></div><div class="flex-1"><div class="flex items-center gap-1.5 py-1.5"><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#34d399;animation-delay:0ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#34d399;animation-delay:150ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:#34d399;animation-delay:300ms;"></span><span class="text-[9px] text-slate-500 ml-1">Analyzing...</span></div></div>';
    } else {
        div.className = 'flex gap-2';
        div.innerHTML = '<div class="w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 mt-0.5" style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.15);"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2"><path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 007.92 12.446A9 9 0 1112 3z"/><path d="M19 3v4"/><path d="M21 5h-4"/></svg></div><div class="flex-1 min-w-0"><div class="mp-agent-resp text-[11px] leading-relaxed" style="color:var(--text-body,#cbd5e1);">' + content + '</div><div class="text-[8px] mt-0.5" style="color:var(--text-muted);opacity:0.4;">' + time + '</div></div>';
    }

    div.style.opacity = '0'; div.style.transform = 'translateY(6px)';
    msgs.appendChild(div);
    requestAnimationFrame(function() {
        div.style.transition = 'opacity 0.2s,transform 0.2s';
        div.style.opacity = '1'; div.style.transform = 'translateY(0)';
    });
    msgs.scrollTop = msgs.scrollHeight;
    return div;
}

function mpClearMarketAgent() {
    _mpAgentHistory = [];
    var msgs = document.getElementById('mp-agent-messages');
    if (!msgs) return;
    msgs.innerHTML = '<div id="mp-agent-welcome" class="flex flex-col items-center py-4 px-2"><div class="w-10 h-10 rounded-2xl flex items-center justify-center mb-3" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2"><path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 007.92 12.446A9 9 0 1112 3z"/><path d="M19 3v4"/><path d="M21 5h-4"/></svg></div><p class="text-[11px] font-semibold mb-1 text-slate-300">Market Intelligence Agent</p><p class="text-[10px] text-slate-500 text-center max-w-xs mb-3">Analyze articles, manage your watchlist, compare tickers, or ask about market trends.</p><div id="mp-agent-suggestions" class="flex flex-wrap gap-1.5 justify-center max-w-md"></div></div>';
    _mpAgentInitSuggestions();
}

// ═══════════════════════════════════════════════════════════
// TICKER INTEL — Tab buttons + per-ticker news
// ═══════════════════════════════════════════════════════════

function _mpRenderNews() {
    var container = document.getElementById('mp-news-content');
    if (!container) return;

    var tickers = Object.keys(marketData);
    if (!tickers.length) {
        container.innerHTML = '<div class="text-[10px] text-slate-500 italic py-3 text-center">No tickers loaded</div>';
        return;
    }

    var tabsHtml = '<div class="flex flex-wrap gap-1.5 mb-3" id="mp-ticker-tabs">';
    tickers.forEach(function(sym) {
        var clean = sym.replace('-USD','').replace('=F','');
        var ad = _resolveData(sym,'1m') || _resolveData(sym,'5m') || {};
        var ch = ad.change || 0, up = ch >= 0;
        var isActive = _mpActiveTicker === sym;
        var cls = isActive
            ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
            : 'bg-slate-800/40 border-slate-700/50 text-slate-400 hover:border-slate-600 hover:text-slate-200';
        tabsHtml += '<button onclick="_mpSelectTicker(\'' + sym + '\')" class="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-[10px] font-mono font-bold transition-all ' + cls + '">';
        tabsHtml += '<span>' + clean + '</span>';
        tabsHtml += '<span class="text-[9px] ' + (up?'text-emerald-400':'text-red-400') + '">' + (up?'+':'') + ch.toFixed(1) + '%</span>';
        tabsHtml += '</button>';
    });
    tabsHtml += '</div>';

    container.innerHTML = tabsHtml + '<div id="mp-ticker-detail"></div>';

    if (!_mpActiveTicker || !marketData[_mpActiveTicker]) _mpActiveTicker = tickers[0];
    _mpRenderTickerDetail();
}

function _mpSelectTicker(sym) {
    _mpActiveTicker = sym;
    // Update tab highlight
    var tabs = document.getElementById('mp-ticker-tabs');
    if (tabs) {
        tabs.querySelectorAll('button').forEach(function(btn) {
            var lbl = _mpActiveTicker.replace('-USD','').replace('=F','');
            var txt = btn.querySelector('span');
            var isCur = txt && txt.textContent.trim() === lbl;
            btn.classList.remove('bg-emerald-500/15','border-emerald-500/40','text-emerald-300','bg-slate-800/40','border-slate-700/50','text-slate-400');
            if (isCur) { btn.classList.add('bg-emerald-500/15','border-emerald-500/40','text-emerald-300'); }
            else { btn.classList.add('bg-slate-800/40','border-slate-700/50','text-slate-400'); }
        });
    }
    _mpRenderTickerDetail();
}

function _mpRenderTickerDetail() {
    var detail = document.getElementById('mp-ticker-detail');
    if (!detail || !_mpActiveTicker) return;

    var sym = _mpActiveTicker;
    var clean = sym.replace('-USD','').replace('=F','');
    var name = getAssetName(sym, (marketData[sym]||{}).name);
    var ad = _resolveData(sym,'1m') || _resolveData(sym,'5m') || {};
    var price = ad.price||0, change = ad.change||0, up = change>=0;

    // Search terms
    var terms = [clean.toLowerCase()];
    if (name.toLowerCase() !== clean.toLowerCase() && name.length > 2) {
        terms.push(name.toLowerCase());
        name.toLowerCase().split(/[\s,]+/).forEach(function(w) { if (w.length >= 4) terms.push(w); });
    }

    // Collect pool
    var pool = [];
    if (typeof financeNewsData !== 'undefined') pool = pool.concat(financeNewsData);
    if (typeof newsData !== 'undefined') pool = pool.concat(newsData.filter(function(n){return (n.score||0)>=5;}));

    var seen = {};
    var matched = [];
    pool.forEach(function(item) {
        var text = ((item.title||'') + ' ' + (item.summary||'')).toLowerCase();
        var key = (item.title||'').toLowerCase().replace(/[^a-z0-9]/g,'').slice(0,50);
        if (!key || key.length < 8 || seen[key]) return;
        seen[key] = 1;
        if (terms.some(function(t) { return text.indexOf(t) >= 0; })) matched.push(item);
    });

    matched.sort(function(a,b) { return (b.score||0) - (a.score||0); });
    matched = matched.slice(0, 10);

    // Sparkline (bigger)
    var hist = ad.history || [];
    var spark = '';
    if (hist.length >= 4) {
        var pts=hist.slice(-40),mn=Math.min.apply(null,pts),mx=Math.max.apply(null,pts),rng=mx-mn||1,w=110,ht=32;
        var pathD = pts.map(function(v,i){return(i===0?'M':'L')+(i/(pts.length-1)*w).toFixed(1)+','+(ht-2-((v-mn)/rng)*(ht-4)).toFixed(1);}).join(' ');
        var fillD = pathD + ' L'+w+','+(ht)+' L0,'+ht+' Z';
        spark='<svg width="'+w+'" height="'+ht+'" class="flex-shrink-0">';
        spark+='<defs><linearGradient id="sg-'+clean+'" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="'+(up?'#10b981':'#ef4444')+'" stop-opacity="0.2"/><stop offset="100%" stop-color="'+(up?'#10b981':'#ef4444')+'" stop-opacity="0"/></linearGradient></defs>';
        spark+='<path d="'+fillD+'" fill="url(#sg-'+clean+')"/>';
        spark+='<path d="'+pathD+'" fill="none" stroke="'+(up?'#10b981':'#ef4444')+'" stroke-width="1.5"/>';
        spark+='</svg>';
    }

    var html = '';

    // ── Hero bar ──
    html += '<div class="flex items-center gap-3 bg-slate-900/60 rounded-xl px-4 py-3 mb-3 border border-slate-800/40">';
    html += '<div class="flex-1 min-w-0">';
    html += '<div class="flex items-baseline gap-2"><span class="text-base font-mono font-bold text-white">' + clean + '</span><span class="text-[10px] text-slate-500">' + _mpEsc(name) + '</span></div>';
    html += '<div class="flex items-baseline gap-2 mt-0.5"><span class="text-lg font-mono font-bold text-slate-100">$' + _mpFp(price) + '</span>';
    html += '<span class="text-xs font-mono font-bold px-1.5 py-0.5 rounded ' + (up?'bg-emerald-500/10 text-emerald-400':'bg-red-500/10 text-red-400') + '">' + (up?'\u25B2 +':'\u25BC ') + Math.abs(change).toFixed(2) + '%</span>';
    html += '</div></div>';
    html += spark;
    html += '</div>';

    // ── Article count ──
    html += '<div class="flex items-center justify-between mb-2">';
    html += '<span class="text-[10px] text-slate-500">' + matched.length + ' article' + (matched.length!==1?'s':'') + ' found</span>';
    if (matched.length) html += '<span class="text-[9px] text-slate-600">Click \u2728 Ask AI to analyze with StratOS agent</span>';
    html += '</div>';

    if (!matched.length) {
        html += '<div class="text-center py-8 rounded-xl border border-dashed border-slate-800/50">';
        html += '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="text-slate-700 mx-auto mb-3"><path d="M4 22h16a2 2 0 002-2V4a2 2 0 00-2-2H8a2 2 0 00-2 2v16a2 2 0 01-2 2zm0 0a2 2 0 01-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8z"/></svg>';
        html += '<p class="text-[11px] text-slate-400 font-medium">No intel found for ' + clean + '</p>';
        html += '<p class="text-[10px] text-slate-600 mt-1">Try loading finance feeds from the sidebar</p>';
        html += '</div>';
    } else {
        _mpIntelArticles = matched;
        html += '<div class="space-y-2">';
        matched.forEach(function(item, idx) {
            var age = item.timestamp ? _mpTimeAgo(item.timestamp) : '';
            var sc = item.score || 0;
            var summary = item.summary || '';
            if (summary.length > 250) summary = summary.slice(0,250) + '\u2026';
            var title = item.title || 'Untitled';
            var src = item.source || item.feed_name || '';

            // Score-based accent
            var accent = '#64748b';
            var accentBg = 'rgba(100,116,139,0.06)';
            if (sc >= 9) { accent = '#10b981'; accentBg = 'rgba(16,185,129,0.04)'; }
            else if (sc >= 8) { accent = '#3b82f6'; accentBg = 'rgba(59,130,246,0.04)'; }
            else if (sc >= 7) { accent = '#f59e0b'; accentBg = 'rgba(245,158,11,0.04)'; }

            // Score badge
            var scBadge = '';
            if (sc >= 6) {
                var scCls = sc >= 9 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20' : sc >= 8 ? 'bg-blue-500/15 text-blue-400 border-blue-500/20' : sc >= 7 ? 'bg-amber-500/15 text-amber-400 border-amber-500/20' : 'bg-slate-500/10 text-slate-400 border-slate-500/20';
                scBadge = '<span class="text-[9px] font-bold px-1.5 py-0.5 rounded border ' + scCls + '">' + sc.toFixed(1) + '</span>';
            }

            html += '<div class="group rounded-xl overflow-hidden transition-all hover:shadow-lg hover:shadow-black/10" style="background:'+accentBg+';border-left:3px solid '+accent+';border-top:1px solid rgba(51,65,85,0.3);border-right:1px solid rgba(51,65,85,0.3);border-bottom:1px solid rgba(51,65,85,0.3);">';

            // Top row: source + age + score
            html += '<div class="flex items-center justify-between px-3 pt-2.5 pb-1">';
            html += '<div class="flex items-center gap-1.5">';
            html += '<span class="text-[9px] font-bold text-slate-500 uppercase tracking-wider">' + _mpEsc(src) + '</span>';
            if (age) html += '<span class="text-[8px] text-slate-600 bg-slate-800/30 px-1.5 py-0.5 rounded">' + age + '</span>';
            html += '</div>';
            html += scBadge;
            html += '</div>';

            // Title
            html += '<div class="px-3 pb-1"><a href="' + (item.url||'#') + '" target="_blank" rel="noopener" class="text-[12px] font-semibold text-slate-200 group-hover:text-white leading-snug transition-colors" style="text-decoration:none;">' + _mpEsc(title) + '</a></div>';

            // Summary
            if (summary) {
                html += '<div class="px-3 pb-2"><p class="text-[10.5px] text-slate-400 leading-relaxed">' + _mpEsc(summary) + '</p></div>';
            }

            // Action bar
            html += '<div class="flex items-center gap-2 px-3 py-2 border-t" style="border-color:rgba(51,65,85,0.2);background:rgba(15,23,42,0.3);">';
            html += '<a href="' + (item.url||'#') + '" target="_blank" rel="noopener" class="text-[10px] text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1" style="text-decoration:none;"><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>Open</a>';
            html += '<div class="flex-1"></div>';
            html += '<button onclick="_mpAskAI(' + idx + ')" class="flex items-center gap-1.5 text-[10px] px-2.5 py-1 rounded-lg transition-all font-medium" style="background:rgba(16,185,129,0.08);color:#34d399;border:1px solid rgba(16,185,129,0.2);" onmouseenter="this.style.background=\'rgba(16,185,129,0.15)\'" onmouseleave="this.style.background=\'rgba(16,185,129,0.08)\'">';
            html += '<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 007.92 12.446A9 9 0 1112 3z"/><path d="M19 3v4"/><path d="M21 5h-4"/></svg>';
            html += '\u2728 Ask AI</button>';
            html += '</div>';

            html += '</div>';
        });
        html += '</div>';
    }

    detail.innerHTML = html;
}

function _mpTimeAgo(ts){try{var d=new Date(ts),diff=(Date.now()-d.getTime())/1000;if(diff<60)return'now';if(diff<3600)return Math.floor(diff/60)+'m';if(diff<86400)return Math.floor(diff/3600)+'h';if(diff<604800)return Math.floor(diff/86400)+'d';return d.toLocaleDateString(undefined,{month:'short',day:'numeric'});}catch(e){return'';}}
function _mpEsc(s){var d=document.createElement('div');d.textContent=s||'';return d.innerHTML;}


// ═══════════════════════════════════════════════════════════
// DRAW MODE — per chart
// ═══════════════════════════════════════════════════════════

function mpToggleDraw(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e) return;
    e.drawMode = !e.drawMode;
    e.drawPending = null;
    var btn = document.getElementById(id+'-btn-draw');
    var colors = document.getElementById(id+'-draw-colors');
    if (btn) { if (e.drawMode) btn.classList.add('active'); else btn.classList.remove('active'); }
    if (colors) { if (e.drawMode) colors.classList.remove('hidden'); else colors.classList.add('hidden'); }
    // Toggle scroll/zoom
    e.chart.applyOptions({
        handleScroll:{mouseWheel:!e.drawMode,pressedMouseMove:!e.drawMode},
        handleScale:{mouseWheel:!e.drawMode,axisPressedMouseMove:!e.drawMode}
    });
    // Toggle canvas pointer events
    var cv = document.getElementById(id+'-drawcanvas');
    if (cv) cv.style.pointerEvents = e.drawMode ? 'none' : 'none'; // chart needs clicks, canvas is overlay only
}

function mpSetDrawColor(id, color) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (e) e.drawColor = color;
}

function mpClearDraw(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e) return;
    e.drawLines = [];
    e.drawPending = null;
    _mpRedrawCanvas(e);
}

function _mpRedrawCanvas(entry) {
    var cv = document.getElementById(entry.id+'-drawcanvas');
    var wrap = document.getElementById(entry.id+'-chartwrap');
    if (!cv || !wrap) return;
    cv.width = wrap.clientWidth;
    cv.height = wrap.clientHeight;
    var ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);

    entry.drawLines.forEach(function(ln) {
        var x1 = entry.chart.timeScale().timeToCoordinate(ln.p1.time);
        var y1 = entry.series.priceToCoordinate(ln.p1.price);
        var x2 = entry.chart.timeScale().timeToCoordinate(ln.p2.time);
        var y2 = entry.series.priceToCoordinate(ln.p2.price);
        if (x1==null||y1==null||x2==null||y2==null) return;
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle = ln.color; ctx.lineWidth = 2; ctx.stroke();
        [{x:x1,y:y1},{x:x2,y:y2}].forEach(function(pt) {
            ctx.beginPath(); ctx.arc(pt.x,pt.y,3,0,Math.PI*2); ctx.fillStyle=ln.color; ctx.fill();
        });
    });

    // Render focus mode drawings from localStorage
    if (typeof _renderFocusDrawings === 'function') {
        _renderFocusDrawings(ctx, cv.width, cv.height, entry.chart, entry.series, entry.symbol);
    }
}

// ═══════════════════════════════════════════════════════════
// SCROLL INDICATOR — shows visible portion of chart data
// ═══════════════════════════════════════════════════════════

function _mpUpdateScrollThumb(entry) {
    var thumb = document.getElementById(entry.id + '-scrollthumb');
    if (!thumb) return;
    try {
        var ts = entry.chart.timeScale();
        var vr = ts.getVisibleLogicalRange();
        if (!vr) return;
        var ad = _resolveData(entry.symbol, entry.timeframe);
        var total = ad && ad.history ? ad.history.length : 100;
        if (total < 2) return;
        var start = Math.max(0, vr.from) / total;
        var end = Math.min(total, vr.to) / total;
        var w = Math.max(0.1, end - start);
        thumb.style.left = (start * 100) + '%';
        thumb.style.width = (w * 100) + '%';
    } catch(e) {}
}

// ═══════════════════════════════════════════════════════════
// ASK AI — sends article context to agent
// ═══════════════════════════════════════════════════════════

function _mpAskAI(idx) {
    var item = _mpIntelArticles[idx];
    if (!item) return;
    var title = item.title || 'Untitled';
    var summary = item.summary || '';
    var source = item.source || item.feed_name || '';

    var prompt = 'Analyze this market intel and tell me what it means for my portfolio/strategy:\n\n';
    prompt += '**' + title + '**\n';
    if (summary) prompt += summary + '\n';
    if (source) prompt += '(Source: ' + source + ')';

    // Ensure market agent section is visible
    if (_mpSectionCollapsed['stats']) {
        _mpSectionCollapsed['stats'] = false;
        _mpSaveSections();
        _mpApplySection('stats');
    }

    // Send directly to the embedded market agent
    mpSendMarketAgent(prompt);

    // Scroll to agent
    var agentEl = document.getElementById('mp-sec-stats');
    if (agentEl) agentEl.scrollIntoView({behavior:'smooth', block:'nearest'});
}

// ═══════════════════════════════════════════════════════════
// SECTION DRAG — reorder overview, ticker intel, comparison
// ═══════════════════════════════════════════════════════════

function _mpInitSectionDrag() {
    var root = document.getElementById('mp-sections');
    if (!root) return;
    var selected = null;

    root.addEventListener('click', function(e) {
        var grip = e.target.closest('.mp-grip');
        if (!grip) return;
        e.stopPropagation();
        var sec = grip.closest('.mp-dragsec');
        if (!sec) return;

        if (selected === sec) {
            // Deselect
            selected = null;
            sec.style.outline = '';
            sec.style.outlineOffset = '';
        } else if (selected) {
            // Swap the two sections
            var parent1 = selected.parentNode, parent2 = sec.parentNode;
            var p1 = document.createComment(''), p2 = document.createComment('');
            parent1.replaceChild(p1, selected);
            parent2.replaceChild(p2, sec);
            p1.parentNode.replaceChild(sec, p1);
            p2.parentNode.replaceChild(selected, p2);
            selected.style.outline = '';
            selected.style.outlineOffset = '';
            selected = null;
        } else {
            // Select
            selected = sec;
            sec.style.outline = '1.5px solid rgba(16,185,129,0.5)';
            sec.style.outlineOffset = '-1px';
        }
    });
}

// ═══════════════════════════════════════════════════════════
// FIBONACCI RETRACEMENT & PATTERN ANALYSIS
// ═══════════════════════════════════════════════════════════

function mpToggleFib(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e) return;

    // Cancel if already in selection mode
    if (e.fibMode) {
        e.fibMode = false;
        e.fibP1 = null; e.fibP2 = null;
        var btn0 = document.getElementById(id+'-btn-fib');
        if (btn0) btn0.classList.remove('active');
        e.chart.applyOptions({handleScroll:{mouseWheel:true,pressedMouseMove:true},handleScale:{mouseWheel:true,axisPressedMouseMove:true}});
        _mpRedrawCanvas(e);
        return;
    }

    // If active fib exists and is pinned — save to pinnedFibs, start new
    if (e.fibP1 && e.fibP2 && e.fibPinned) {
        e.pinnedFibs.push({
            p1: e.fibP1, p2: e.fibP2,
            prices: e._fibPrices ? e._fibPrices.slice() : [],
            series: e.fibSeries.slice(),
            pattern: e._lastPattern || {name:'',icon:'',cls:'',desc:''}
        });
        // Don't clear the chart series — they stay drawn
        e.fibSeries = [];
        e._fibPrices = null;
        e.fibP1 = null; e.fibP2 = null; e.fibPinned = false;
    }

    // If active fib exists but NOT pinned — clear it
    if (e.fibP1 && e.fibP2) {
        _mpClearFibSeries(e);
        e.fibP1 = null; e.fibP2 = null;
    }

    // Enter fib selection mode
    e.fibMode = true;
    e.fibPinned = false;
    var btn = document.getElementById(id+'-btn-fib');
    if (btn) btn.classList.add('active');

    e.chart.applyOptions({
        handleScroll:{mouseWheel:false,pressedMouseMove:false},
        handleScale:{mouseWheel:false,axisPressedMouseMove:false}
    });
}

function mpToggleFibPin(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e || !e.fibP1 || !e.fibP2) return;
    e.fibPinned = !e.fibPinned;
    var pinBtn = document.getElementById(id + '-fib-pin');
    if (pinBtn) {
        if (e.fibPinned) { pinBtn.innerHTML = '\u{1F4CC} Pinned'; pinBtn.classList.add('text-amber-400'); pinBtn.classList.remove('text-slate-500'); }
        else { pinBtn.innerHTML = '\u{1F4CC} Pin'; pinBtn.classList.remove('text-amber-400'); pinBtn.classList.add('text-slate-500'); }
    }
}

function mpClearFib(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e) return;
    _mpClearFibSeries(e);
    // Clear all pinned fibs too
    if (e.pinnedFibs) {
        e.pinnedFibs.forEach(function(pf) {
            if (pf.series) pf.series.forEach(function(s) { try{e.chart.removeSeries(s);}catch(x){} });
        });
        e.pinnedFibs = [];
    }
    e.fibP1 = null; e.fibP2 = null; e.fibMode = false; e.fibPinned = false;
    e._fibPrices = null; e._lastPattern = null;
    var btn = document.getElementById(id+'-btn-fib');
    if (btn) btn.classList.remove('active');
    var fp = document.getElementById(id + '-fibpanel');
    if (fp) fp.innerHTML = '';
    _mpRedrawCanvas(e);
    e.chart.applyOptions({handleScroll:{mouseWheel:true,pressedMouseMove:true},handleScale:{mouseWheel:true,axisPressedMouseMove:true}});
}

function mpClearPinnedFib(chartId, idx) {
    var e = _mpCharts.find(function(c){return c.id===chartId;});
    if (!e || !e.pinnedFibs || !e.pinnedFibs[idx]) return;
    var pf = e.pinnedFibs[idx];
    if (pf.series) pf.series.forEach(function(s) { try{e.chart.removeSeries(s);}catch(x){} });
    e.pinnedFibs.splice(idx, 1);
    _mpShowFibPanel(e, e._lastPattern || {name:'',icon:'',cls:'',desc:''});
    _mpRedrawCanvas(e);
}

var _fibLevels = [
    {level:0,    label:'0%',      color:'#ef4444', bg:'rgba(239,68,68,0.06)'},
    {level:0.236,label:'23.6%',   color:'#f59e0b', bg:'rgba(245,158,11,0.05)'},
    {level:0.382,label:'38.2%',   color:'#eab308', bg:'rgba(234,179,8,0.06)'},
    {level:0.5,  label:'50.0%',   color:'#94a3b8', bg:'rgba(148,163,184,0.06)'},
    {level:0.618,label:'61.8%',   color:'#22c55e', bg:'rgba(34,197,94,0.06)'},
    {level:0.786,label:'78.6%',   color:'#10b981', bg:'rgba(16,185,129,0.05)'},
    {level:1,    label:'100%',    color:'#3b82f6', bg:'rgba(59,130,246,0.06)'}
];

function _mpDrawFib(entry) {
    _mpClearFibSeries(entry);
    if (!entry.fibP1 || !entry.fibP2) return;

    var p1 = entry.fibP1.price, p2 = entry.fibP2.price;
    var t1 = entry.fibP1.time, t2 = entry.fibP2.time;
    var high = Math.max(p1, p2), low = Math.min(p1, p2);
    var range = high - low;
    if (range < 0.001) return;

    var isUpswing = p1 < p2;
    var tMin = Math.min(t1, t2), tMax = Math.max(t1, t2);
    var ext = Math.max(1, Math.floor((tMax - tMin) * 0.6));

    // Store computed level prices for detail panel
    entry._fibPrices = [];

    // Draw fib levels as series lines (thicker, more visible)
    _fibLevels.forEach(function(f) {
        var price = isUpswing ? (p2 - range * f.level) : (p2 + range * f.level);
        entry._fibPrices.push({label:f.label, level:f.level, price:price, color:f.color});
        try {
            var s = entry.chart.addLineSeries({
                color: f.color, lineWidth: f.level === 0.5 || f.level === 0.618 ? 2 : 1,
                lineStyle: f.level === 0.5 ? 0 : 2,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            s.setData([{time:tMin, value:price}, {time:tMax+ext, value:price}]);
            entry.fibSeries.push(s);
        } catch(x) {}
    });

    // Draw the swing line (golden, thick)
    try {
        var swing = entry.chart.addLineSeries({
            color:'#fbbf24', lineWidth:2.5, lineStyle:0,
            crosshairMarkerVisible:false, lastValueVisible:false, priceLineVisible:false,
        });
        swing.setData([{time:t1, value:p1}, {time:t2, value:p2}]);
        entry.fibSeries.push(swing);
    } catch(x) {}

    // Draw fib zones + labels on canvas
    _mpDrawFibCanvas(entry);

    // Detect pattern & show detail panel
    var pattern = _mpDetectPattern(entry, t1, t2);
    entry._lastPattern = pattern;
    _mpShowFibPanel(entry, pattern);

    // Redraw on scroll/zoom
    entry._fibRedraw = function() { _mpDrawFibCanvas(entry); };
}

function _mpDrawFibCanvas(entry) {
    var cv = document.getElementById(entry.id+'-drawcanvas');
    var wrap = document.getElementById(entry.id+'-chartwrap');
    if (!cv || !wrap || !entry._fibPrices) return;
    cv.width = wrap.clientWidth; cv.height = wrap.clientHeight;
    var ctx = cv.getContext('2d');

    // Redraw any hand-drawn lines first
    entry.drawLines.forEach(function(ln) {
        var x1=entry.chart.timeScale().timeToCoordinate(ln.p1.time);
        var y1=entry.series.priceToCoordinate(ln.p1.price);
        var x2=entry.chart.timeScale().timeToCoordinate(ln.p2.time);
        var y2=entry.series.priceToCoordinate(ln.p2.price);
        if(x1==null||y1==null||x2==null||y2==null)return;
        ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
        ctx.strokeStyle=ln.color;ctx.lineWidth=2;ctx.stroke();
        [{x:x1,y:y1},{x:x2,y:y2}].forEach(function(pt) {
            ctx.beginPath();ctx.arc(pt.x,pt.y,3,0,Math.PI*2);ctx.fillStyle=ln.color;ctx.fill();
        });
    });

    // Draw shaded zones between fib levels
    var prices = entry._fibPrices;
    for (var i = 0; i < prices.length - 1; i++) {
        var y1 = entry.series.priceToCoordinate(prices[i].price);
        var y2 = entry.series.priceToCoordinate(prices[i+1].price);
        if (y1 == null || y2 == null) continue;
        var top = Math.min(y1, y2), btm = Math.max(y1, y2);
        ctx.fillStyle = _fibLevels[i].bg;
        ctx.fillRect(0, top, cv.width, btm - top);
    }

    // Draw labels with background pill
    ctx.font = 'bold 10px ui-monospace, SFMono-Regular, monospace';
    ctx.textBaseline = 'middle';

    prices.forEach(function(f) {
        var y = entry.series.priceToCoordinate(f.price);
        if (y == null) return;
        var txt = f.label + '  $' + f.price.toFixed(2);
        var tw = ctx.measureText(txt).width;
        // Pill background
        ctx.fillStyle = 'rgba(15,23,42,0.85)';
        ctx.beginPath();
        ctx.roundRect(4, y-8, tw+10, 16, 4);
        ctx.fill();
        ctx.strokeStyle = f.color;
        ctx.lineWidth = 1;
        ctx.stroke();
        // Text
        ctx.fillStyle = f.color;
        ctx.fillText(txt, 9, y);
    });

    // Draw swing arrows at P1 and P2
    if (entry.fibP1 && entry.fibP2) {
        var sx1 = entry.chart.timeScale().timeToCoordinate(entry.fibP1.time);
        var sy1 = entry.series.priceToCoordinate(entry.fibP1.price);
        var sx2 = entry.chart.timeScale().timeToCoordinate(entry.fibP2.time);
        var sy2 = entry.series.priceToCoordinate(entry.fibP2.price);
        [[sx1,sy1,'P1'],[sx2,sy2,'P2']].forEach(function(pt) {
            if (pt[0]==null||pt[1]==null) return;
            // Dot
            ctx.beginPath(); ctx.arc(pt[0],pt[1],5,0,Math.PI*2);
            ctx.fillStyle='#fbbf24'; ctx.fill();
            ctx.strokeStyle='rgba(0,0,0,0.5)'; ctx.lineWidth=1; ctx.stroke();
            // Label
            ctx.fillStyle='rgba(15,23,42,0.85)';
            ctx.beginPath(); ctx.roundRect(pt[0]+8, pt[1]-8, 22, 16, 3); ctx.fill();
            ctx.fillStyle='#fbbf24'; ctx.font='bold 9px monospace';
            ctx.fillText(pt[2], pt[0]+11, pt[1]);
        });
    }
}

function _mpShowFibPanel(entry, pattern) {
    var el = document.getElementById(entry.id + '-fibpanel');
    if (!el) return;

    var ad = _resolveData(entry.symbol, entry.timeframe);
    var currentPrice = ad ? (ad.price || 0) : 0;

    // Build array of all fibs to render: pinned first, then active
    var allFibs = [];
    if (entry.pinnedFibs) {
        entry.pinnedFibs.forEach(function(pf, i) {
            allFibs.push({p1:pf.p1, p2:pf.p2, prices:pf.prices, pattern:pf.pattern, pinned:true, idx:i});
        });
    }
    if (entry.fibP1 && entry.fibP2 && entry._fibPrices) {
        allFibs.push({p1:entry.fibP1, p2:entry.fibP2, prices:entry._fibPrices, pattern:pattern, pinned:false, idx:-1});
    }
    if (!allFibs.length) { el.innerHTML = ''; return; }

    // Flex container for side-by-side
    var multi = allFibs.length > 1;
    var html = '<div class="mt-2 pt-2 border-t border-amber-500/20' + (multi ? ' flex gap-3 overflow-x-auto' : '') + '">';

    allFibs.forEach(function(fib) {
        var p1p = fib.p1.price, p2p = fib.p2.price;
        var high = Math.max(p1p, p2p), low = Math.min(p1p, p2p);
        var range = high - low;
        var swing = (range / low * 100).toFixed(2);
        var isUp = p1p < p2p;

        html += '<div class="' + (multi ? 'flex-1 min-w-[260px]' : '') + '">';

        // Header
        html += '<div class="flex items-center justify-between mb-1.5">';
        html += '<div class="flex items-center gap-1.5 flex-wrap">';
        html += '<span class="text-[8px] font-bold px-1 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">\uD83D\uDCC0 Fib</span>';
        if (fib.pattern && fib.pattern.name) html += '<span class="text-[9px] font-bold ' + fib.pattern.cls + '">' + fib.pattern.icon + ' ' + fib.pattern.name + '</span>';
        if (fib.pinned) html += '<span class="text-[8px] text-amber-400">\uD83D\uDCCC</span>';
        html += '</div>';
        html += '<div class="flex items-center gap-1">';
        if (fib.pinned) {
            html += '<button onclick="mpClearPinnedFib(\'' + entry.id + '\',' + fib.idx + ')" class="text-[8px] px-1 py-0.5 rounded border border-red-500/20 text-red-400 hover:bg-red-500/10 transition-all">\u2715</button>';
        } else {
            html += '<button id="'+entry.id+'-fib-pin" onclick="mpToggleFibPin(\'' + entry.id + '\')" class="text-[8px] px-1 py-0.5 rounded border border-slate-700/30 ' + (entry.fibPinned ? 'text-amber-400 border-amber-500/20' : 'text-slate-500 hover:text-amber-400 hover:border-amber-500/20') + ' transition-all">' + (entry.fibPinned ? '\uD83D\uDCCC Pinned' : '\uD83D\uDCCC Pin') + '</button>';
            html += '<button onclick="mpClearFib(\'' + entry.id + '\')" class="text-[8px] px-1 py-0.5 rounded border border-red-500/20 text-red-400 hover:bg-red-500/10 transition-all">Clear</button>';
        }
        html += '</div></div>';

        // Swing info
        html += '<div class="flex items-center gap-2 mb-1.5 text-[9px] flex-wrap">';
        html += '<span class="text-slate-500">Swing: <span class="font-bold '+(isUp?'text-emerald-400':'text-red-400')+'">'+swing+'%</span></span>';
        html += '<span class="text-slate-500">H: <span class="font-bold text-emerald-400">$'+_mpFp(high)+'</span></span>';
        html += '<span class="text-slate-500">L: <span class="font-bold text-red-400">$'+_mpFp(low)+'</span></span>';
        html += '</div>';

        // Levels table
        html += '<div class="grid grid-cols-4 gap-x-2 gap-y-0.5 text-[9px]">';
        html += '<span class="text-slate-600 font-bold">Level</span><span class="text-slate-600 font-bold text-right">Price</span><span class="text-slate-600 font-bold text-right">Diff</span><span class="text-slate-600 font-bold text-right">Status</span>';
        (fib.prices||[]).forEach(function(f) {
            var diff = currentPrice - f.price;
            var pct = currentPrice > 0 ? ((diff / currentPrice) * 100).toFixed(1) : '0.0';
            var isNear = range > 0 && Math.abs(diff / range) < 0.03;
            var statusCls = isNear ? 'text-amber-400 font-bold' : (diff > 0 ? 'text-emerald-400' : 'text-red-400');
            var status = isNear ? '\u2022 AT' : (diff > 0 ? '\u2191 Above' : '\u2193 Below');
            html += '<span style="color:'+f.color+';" class="font-bold">'+f.label+'</span>';
            html += '<span class="text-slate-300 font-mono text-right">$'+f.price.toFixed(2)+'</span>';
            html += '<span class="text-slate-400 font-mono text-right">'+pct+'%</span>';
            html += '<span class="'+statusCls+' text-right">'+status+'</span>';
        });
        html += '</div>';

        // Nearest level
        var nearest = null, nearestDist = Infinity;
        (fib.prices||[]).forEach(function(f) {
            var d = Math.abs(currentPrice - f.price);
            if (d < nearestDist) { nearestDist = d; nearest = f; }
        });
        if (nearest && currentPrice > 0) {
            html += '<div class="mt-1 text-[9px] text-slate-500">Price <span class="text-white font-bold">$'+_mpFp(currentPrice)+'</span> nearest <span style="color:'+nearest.color+';" class="font-bold">'+nearest.label+'</span></div>';
        }

        html += '</div>';
    });

    html += '</div>';
    el.innerHTML = html;
}

function _mpDetectPattern(entry, t1, t2) {
    var ad = _resolveData(entry.symbol, entry.timeframe);
    if (!ad || !ad.history || !ad.timestamps) return {name:'',icon:'',cls:'',desc:''};

    var hist = ad.history, ts = ad.timestamps;
    var tMin = Math.min(t1, t2), tMax = Math.max(t1, t2);

    var rangeData = [];
    for (var i = 0; i < hist.length; i++) {
        var t = ts[i] ? Math.floor(new Date(ts[i]).getTime()/1000) : 0;
        if (t >= tMin && t <= tMax) rangeData.push(hist[i]);
    }
    if (rangeData.length < 5) return {name:'Insufficient Data',icon:'\u2753',cls:'text-slate-400',desc:'Select a wider range for pattern detection.'};

    var n = rangeData.length;
    var maxVal = Math.max.apply(null, rangeData);
    var minVal = Math.min.apply(null, rangeData);
    var startP = rangeData[0], endP = rangeData[n-1];
    var mid = Math.floor(n / 2);
    var tolerance = (maxVal - minVal) * 0.1;

    var fh = rangeData.slice(0, mid), sh = rangeData.slice(mid);
    var fhMax = Math.max.apply(null, fh), shMax = Math.max.apply(null, sh);
    var fhMin = Math.min.apply(null, fh), shMin = Math.min.apply(null, sh);
    var midData = rangeData.slice(Math.floor(n*0.3), Math.floor(n*0.7));
    var midMin = Math.min.apply(null, midData), midMax = Math.max.apply(null, midData);

    // Head and shoulders: 3 peaks, middle highest
    var third = Math.floor(n/3);
    var t1d = rangeData.slice(0,third), t2d = rangeData.slice(third,2*third), t3d = rangeData.slice(2*third);
    var p1Max = Math.max.apply(null,t1d), p2Max = Math.max.apply(null,t2d), p3Max = Math.max.apply(null,t3d);
    if (p2Max > p1Max && p2Max > p3Max && Math.abs(p1Max-p3Max) < tolerance*2 && p2Max > p1Max+tolerance) {
        return {name:'Head & Shoulders',icon:'\u{1F451}',cls:'text-red-400',desc:'Bearish reversal pattern. The middle peak (head) is higher than both shoulders. Watch for neckline break below $'+Math.min(Math.min.apply(null,t1d.slice(-3)),Math.min.apply(null,t3d.slice(0,3))).toFixed(2)+'.'};
    }

    // Inverse H&S
    var p1Min = Math.min.apply(null,t1d), p2Min = Math.min.apply(null,t2d), p3Min = Math.min.apply(null,t3d);
    if (p2Min < p1Min && p2Min < p3Min && Math.abs(p1Min-p3Min) < tolerance*2 && p2Min < p1Min-tolerance) {
        return {name:'Inverse H&S',icon:'\u{1F451}',cls:'text-emerald-400',desc:'Bullish reversal pattern. The middle trough is lower than both shoulders. Watch for neckline break above $'+Math.max(Math.max.apply(null,t1d.slice(-3)),Math.max.apply(null,t3d.slice(0,3))).toFixed(2)+'.'};
    }

    // Double Top
    if (Math.abs(fhMax - shMax) < tolerance && midMin < fhMax - tolerance*2) {
        return {name:'Double Top',icon:'\u26A0',cls:'text-red-400',desc:'Bearish reversal pattern. Two peaks at similar levels (~$'+fhMax.toFixed(2)+') with a valley between them. Support at $'+midMin.toFixed(2)+'.'};
    }

    // Double Bottom
    if (Math.abs(fhMin - shMin) < tolerance && midMax > fhMin + tolerance*2) {
        return {name:'Double Bottom',icon:'\u26A0',cls:'text-emerald-400',desc:'Bullish reversal pattern. Two troughs at similar levels (~$'+fhMin.toFixed(2)+') with a peak between them. Resistance at $'+midMax.toFixed(2)+'.'};
    }

    // Triangles
    var q1 = rangeData.slice(0, third), q3 = rangeData.slice(2*third);
    var q1MaxV = Math.max.apply(null,q1), q3MaxV = Math.max.apply(null,q3);
    var q1MinV = Math.min.apply(null,q1), q3MinV = Math.min.apply(null,q3);

    if (Math.abs(q1MaxV - q3MaxV) < tolerance && q3MinV > q1MinV + tolerance) {
        return {name:'Ascending Triangle',icon:'\u25B3',cls:'text-emerald-400',desc:'Bullish continuation. Flat resistance at ~$'+q1MaxV.toFixed(2)+' with rising support. Expect breakout above resistance.'};
    }
    if (Math.abs(q1MinV - q3MinV) < tolerance && q3MaxV < q1MaxV - tolerance) {
        return {name:'Descending Triangle',icon:'\u25BD',cls:'text-red-400',desc:'Bearish continuation. Flat support at ~$'+q1MinV.toFixed(2)+' with falling resistance. Expect breakdown below support.'};
    }
    if (q3MaxV < q1MaxV - tolerance*0.5 && q3MinV > q1MinV + tolerance*0.5) {
        return {name:'Symmetrical Triangle',icon:'\u25C7',cls:'text-amber-400',desc:'Converging highs and lows indicate compression. Direction of breakout will determine trend. Volume typically decreases during formation.'};
    }

    // Trend detection
    if (endP > startP * 1.01) {
        var mag = ((endP/startP - 1)*100).toFixed(1);
        return {name:'Uptrend',icon:'\u2191',cls:'text-emerald-400',desc:'Price moved up '+mag+'% across this range. Look for support at key Fibonacci levels (38.2%, 50%, 61.8%) on pullbacks.'};
    }
    if (endP < startP * 0.99) {
        var dmag = ((1 - endP/startP)*100).toFixed(1);
        return {name:'Downtrend',icon:'\u2193',cls:'text-red-400',desc:'Price moved down '+dmag+'% across this range. Watch for resistance at Fibonacci retracement levels on bounces.'};
    }

    return {name:'Consolidation',icon:'\u2194',cls:'text-slate-400',desc:'Sideways price action. The 50% level ($'+(entry._fibPrices[3]||{}).price?.toFixed(2)+') often acts as a pivot in consolidation ranges.'};
}

function _mpClearFibSeries(entry) {
    if (entry.fibSeries) {
        entry.fibSeries.forEach(function(s) { try{entry.chart.removeSeries(s);}catch(x){} });
        entry.fibSeries = [];
    }
    entry._fibPrices = null;
    entry._fibRedraw = null;
}

function mpOpenPicker(){if(_mpCharts.length<MP_MAX_CHARTS)mpAddChart(null,'1m');}

// Toggle bottom row layout between side-by-side and stacked
function mpToggleBottomLayout() {
    var row = document.getElementById('mp-bottom-row');
    var label = document.getElementById('mp-layout-label');
    if (!row) return;
    var isSideBySide = row.classList.contains('lg:grid-cols-2');
    if (isSideBySide) {
        row.classList.remove('lg:grid-cols-2');
        row.classList.add('grid-cols-1');
        if (label) label.textContent = 'Stacked';
    } else {
        row.classList.add('lg:grid-cols-2');
        row.classList.remove('grid-cols-1');
        if (label) label.textContent = 'Side-by-side';
    }
    try { localStorage.setItem('mp_bottom_layout', isSideBySide ? 'stacked' : 'side'); } catch(e) {}
}

// Restore bottom layout on init
function _mpRestoreBottomLayout() {
    try {
        var saved = localStorage.getItem('mp_bottom_layout');
        if (saved === 'stacked') {
            var row = document.getElementById('mp-bottom-row');
            var label = document.getElementById('mp-layout-label');
            if (row) { row.classList.remove('lg:grid-cols-2'); row.classList.add('grid-cols-1'); }
            if (label) label.textContent = 'Stacked';
        }
    } catch(e) {}
}
function cleanupMarketsPanel(){}
function destroyMarketsPanel(){_mpCharts.forEach(function(c){try{c.ro.disconnect();}catch(e){}try{c.chart.remove();}catch(e){}});_mpCharts=[];_mpCounter=0;}

// ═══════════════════════════════════════════════════════════
// EXPORT CHART AS PNG
// ═══════════════════════════════════════════════════════════

function mpExportChartPNG(id) {
    var e = _mpCharts.find(function(c){return c.id===id;});
    if (!e || !e.chart) return;
    try {
        var canvas = e.chart.takeScreenshot();
        var a = document.createElement('a');
        var sym = (e.symbol || 'chart').replace(/[^a-zA-Z0-9_-]/g, '_');
        var tf = e.timeframe || 'unknown';
        a.download = 'STRAT_OS_' + sym + '_' + tf + '_' + new Date().toISOString().slice(0,10) + '.png';
        a.href = canvas.toDataURL('image/png');
        a.click();
    } catch(err) {
        if (typeof showToast === 'function') showToast('Screenshot failed', 'error', 2000);
    }
}

// ═══════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS (Markets panel)
// ═══════════════════════════════════════════════════════════

// Get the focused chart: the one matching _mpActiveTicker, or first chart
function _mpGetFocusedChart() {
    if (_mpCharts.length === 0) return null;
    if (_mpActiveTicker) {
        var match = _mpCharts.find(function(c){return c.symbol === _mpActiveTicker;});
        if (match) return match;
    }
    return _mpCharts[0];
}

document.addEventListener('keydown', function(e) {
    // Skip if typing in input/textarea/contenteditable
    var tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
    // Only active on markets view
    if (typeof activeRoot === 'undefined' || activeRoot !== 'markets_view') return;
    // Let browser handle Ctrl/Cmd combos (copy, paste, undo, etc.)
    if (e.ctrlKey || e.metaKey) return;

    var fc = _mpGetFocusedChart();
    if (!fc) return;

    var TFS = ['1m','5m','1d_1mo','1d_1y','1wk'];

    switch(e.key) {
        // 1-5: switch timeframes on focused chart
        case '1': e.preventDefault(); mpChangeTimeframe(fc.id, TFS[0]); break;
        case '2': e.preventDefault(); mpChangeTimeframe(fc.id, TFS[1]); break;
        case '3': e.preventDefault(); mpChangeTimeframe(fc.id, TFS[2]); break;
        case '4': e.preventDefault(); mpChangeTimeframe(fc.id, TFS[3]); break;
        case '5': e.preventDefault(); mpChangeTimeframe(fc.id, TFS[4]); break;
        // j/k: cycle through charts
        case 'j': case 'k': {
            if (_mpCharts.length < 2) break;
            var idx = _mpCharts.indexOf(fc);
            var next = e.key === 'j'
                ? _mpCharts[(idx + 1) % _mpCharts.length]
                : _mpCharts[(idx - 1 + _mpCharts.length) % _mpCharts.length];
            _mpActiveTicker = next.symbol;
            // Scroll the chart into view
            next.el.scrollIntoView({behavior:'smooth', block:'nearest'});
            // Flash highlight
            next.el.style.outline = '2px solid rgba(16,185,129,0.5)';
            setTimeout(function(){next.el.style.outline='';}, 800);
            break;
        }
        // d: toggle draw mode
        case 'd': e.preventDefault(); mpToggleDraw(fc.id); break;
        // c: cycle chart type
        case 'c': {
            e.preventDefault();
            var types = ['line','candle','area'];
            var ci = types.indexOf(fc.chartType);
            var nextType = types[(ci + 1) % types.length];
            mpSetChartType(fc.id, nextType);
            break;
        }
        // f: toggle Fibonacci
        case 'f': e.preventDefault(); mpToggleFib(fc.id); break;
        // x: toggle crosshair
        case 'x': e.preventDefault(); mpToggleCrosshair(fc.id); break;
        // s: screenshot
        case 's': if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); mpExportChartPNG(fc.id); } break;
        // Escape: cancel drawing
        case 'Escape':
            if (fc.drawMode) mpToggleDraw(fc.id);
            if (fc.fibMode) mpToggleFib(fc.id);
            break;
    }
});

/* ═══════════════════════════════════════════════════════════
   MOBILE SPLIT VIEW TOGGLE
   ═══════════════════════════════════════════════════════════ */
function mpToggleMobileCompact() {
    var grid = document.getElementById('mp-chart-grid');
    if (!grid) return;
    var isCompact = grid.classList.toggle('compact-mobile');
    var label = document.getElementById('mp-compact-label');
    if (label) label.textContent = isCompact ? 'Full' : 'Split';
    localStorage.setItem('mp_mobile_compact', isCompact ? '1' : '0');
    // Trigger chart resize after layout shift
    setTimeout(function() {
        _mpCharts.forEach(function(c) {
            if (c.chart) {
                var wrap = c.el.querySelector('.mp-chartwrap');
                if (wrap) c.chart.applyOptions({ width: wrap.clientWidth, height: wrap.clientHeight });
            }
        });
    }, 150);
}

// ═══════════════════════════════════════════════════════════
// SYNC — refresh markets panel when market data updates
// ═══════════════════════════════════════════════════════════

window.addEventListener('stratos-market-data-refreshed', function() {
    // Only refresh if the markets panel is visible (has charts loaded)
    if (_mpCharts.length > 0) {
        _mpRefreshAll();
        _mpRenderOverview();
        _mpRenderNews();
    }
});

// Restore saved compact mode on init
(function() {
    if (window.innerWidth <= 768 && localStorage.getItem('mp_mobile_compact') === '1') {
        document.addEventListener('DOMContentLoaded', function() {
            var grid = document.getElementById('mp-chart-grid');
            var label = document.getElementById('mp-compact-label');
            if (grid) grid.classList.add('compact-mobile');
            if (label) label.textContent = 'Full';
        });
    }
})();
