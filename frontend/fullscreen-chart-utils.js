/**
 * fullscreen-chart-utils.js — Shared utilities for StratOS fullscreen chart
 *
 * Extracted from fullscreen-chart.js.
 * Must be loaded BEFORE fullscreen-chart.js.
 * All declarations are top-level vars/functions (globally accessible, no IIFE).
 */

/* Timeframe mapping: button label → data key */
var _fsTfMap = [
    { label:'1m',   key:'1m' },
    { label:'5m',   key:'5m' },
    { label:'D',    key:'1d_1mo' },
    { label:'D\u00b75Y', key:'1d_1y' },
    { label:'W',    key:'1wk' },
];

/* Browser timezone offset in seconds (for chart timestamp alignment) */
var _fsTzOff = -(new Date().getTimezoneOffset() * 60);

/* ── SMA Calculation ── */
function _calcSMA(closes, period) {
    var result = [];
    for (var i = 0; i < closes.length; i++) {
        if (i < period - 1) { result.push(null); continue; }
        var sum = 0;
        for (var j = i - period + 1; j <= i; j++) sum += closes[j];
        result.push(sum / period);
    }
    return result;
}

/* ── Format volume with K/M/B suffixes ── */
function _fsFormatVol(v) {
    if (v == null) return '--';
    if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
    if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return v.toFixed(0);
}

/* ── Format price with appropriate precision ── */
function _fsFmtPrice(v) {
    if (v == null) return '--';
    if (Math.abs(v) >= 100) return v.toFixed(2);
    if (Math.abs(v) >= 1) return v.toFixed(4);
    return v.toFixed(6);
}

/* ── Build OHLC chart data from _resolveData for a given symbol+tf ── */
function _fsBuildData(symbol, tfKey) {
    if (typeof _resolveData !== 'function') return null;
    var ad = _resolveData(symbol, tfKey);
    if (!ad || !ad.history || ad.history.length === 0) return null;
    var ts = ad.timestamps || [];
    var hist = ad.history;
    var hasOHLC = ad.opens && ad.opens.length === hist.length;
    var data = [], seen = {};
    for (var i = 0; i < hist.length; i++) {
        var t = ts[i] != null ? (typeof ts[i]==='number' ? ts[i]+_fsTzOff : Math.floor(new Date(ts[i]).getTime()/1000)+_fsTzOff) : Math.floor(Date.now()/1000)+_fsTzOff-(hist.length-1-i)*60;
        if (seen[t]) continue;
        seen[t] = true;
        if (hasOHLC) {
            data.push({ time:t, open:ad.opens[i]||hist[i], high:(ad.highs||[])[i]||hist[i], low:(ad.lows||[])[i]||hist[i], close:hist[i] });
        } else {
            data.push({ time:t, value:hist[i] });
        }
    }
    data.sort(function(a,b){return a.time-b.time;});
    return { data:data, ad:ad, hasOHLC:hasOHLC };
}

/* ── Extract data points for the main chart series (from marketData) ── */
function _extractSeriesData(series) {
    if (!series) return [];
    /* Fallback: re-build from marketData */
    try {
        if (typeof currentSymbol !== 'undefined' && typeof currentTimeframe !== 'undefined' && typeof _resolveData === 'function') {
            var ad = _resolveData(currentSymbol, currentTimeframe);
            if (ad && ad.history) {
                var ts = ad.timestamps || [];
                var hist = ad.history;
                var hasOHLC = ad.opens && ad.opens.length === hist.length;
                var data = [];
                var seen = {};
                for (var i = 0; i < hist.length; i++) {
                    var t = ts[i] != null ? (typeof ts[i]==='number' ? ts[i]+_fsTzOff : Math.floor(new Date(ts[i]).getTime()/1000)+_fsTzOff) : Math.floor(Date.now()/1000)+_fsTzOff-(hist.length-1-i)*60;
                    if (seen[t]) continue;
                    seen[t] = true;
                    if (hasOHLC) {
                        data.push({ time:t, open:ad.opens[i]||hist[i], high:(ad.highs||[])[i]||hist[i], low:(ad.lows||[])[i]||hist[i], close:hist[i] });
                    } else {
                        data.push({ time:t, value:hist[i] });
                    }
                }
                data.sort(function(a,b){return a.time-b.time;});
                return data;
            }
        }
    } catch(e) {}
    return [];
}

/* ── Extract chart data for a markets-panel entry ── */
function _extractMpChartData(entry) {
    if (!entry) return [];
    try {
        var ad = (typeof _resolveData === 'function') ? _resolveData(entry.symbol, entry.timeframe) : null;
        if (!ad || !ad.history) return [];
        var ts = ad.timestamps || [];
        var hist = ad.history;
        var hasOHLC = ad.opens && ad.opens.length === hist.length;
        var data = [];
        var seen = {};
        for (var i = 0; i < hist.length; i++) {
            var t = ts[i] != null ? (typeof ts[i]==='number' ? ts[i]+_fsTzOff : Math.floor(new Date(ts[i]).getTime()/1000)+_fsTzOff) : Math.floor(Date.now()/1000)+_fsTzOff-(hist.length-1-i)*60;
            if (seen[t]) continue;
            seen[t] = true;
            if (hasOHLC && entry.chartType === 'candle') {
                data.push({ time:t, open:ad.opens[i]||hist[i], high:(ad.highs||[])[i]||hist[i], low:(ad.lows||[])[i]||hist[i], close:hist[i] });
            } else {
                data.push({ time:t, value:hist[i] });
            }
        }
        data.sort(function(a,b){return a.time-b.time;});
        return data;
    } catch(e) { return []; }
}
