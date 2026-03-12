// ═══════════════════════════════════════════════════════════
// AGENT TICKERS — Watchlist widget, ticker/category commands
// Split from agent.js
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// WATCHLIST WIDGET — Rich card display for ticker list
// ═══════════════════════════════════════════════════════════

function _buildWatchlistWidget(tickers) {
    var fp = function(v) { return v != null ? v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '--'; };

    var cards = '';
    for (var idx = 0; idx < tickers.length; idx++) {
        var sym = tickers[idx];
        var lbl = sym.replace('-USD','').replace('=F','');
        // Try to find market data (exact, -USD, =F variants)
        var md = (typeof marketData !== 'undefined') ? (marketData[sym] || marketData[sym+'-USD'] || marketData[sym+'=F']) : null;
        var ad = md?.data ? (_resolveData(sym, '1m') || _resolveData(sym+'-USD', '1m') || _resolveData(sym+'=F', '1m') ||
                               _resolveData(sym, '5m') || _resolveData(sym+'-USD', '5m') || _resolveData(sym+'=F', '5m')) : null;

        var realSym = marketData[sym] ? sym : marketData[sym+'-USD'] ? sym+'-USD' : marketData[sym+'=F'] ? sym+'=F' : sym;
        var name = (typeof getAssetName === 'function') ? getAssetName(realSym, md?.name || lbl) : lbl;
        var price = ad?.price || 0;
        var change = ad?.change || 0;
        var up = change >= 0;
        var hasData = price > 0;

        // Mini sparkline SVG
        var spark = '';
        var hist = ad?.history || [];
        if (hist.length >= 4) {
            var pts = hist.slice(-24);
            var mn = Math.min.apply(null, pts), mx = Math.max.apply(null, pts), rng = mx - mn || 1;
            var w = 48, h = 18;
            var path = pts.map(function(v, i) { return (i===0?'M':'L') + (i/(pts.length-1)*w).toFixed(1) + ',' + (h-((v-mn)/rng)*h).toFixed(1); }).join(' ');
            spark = '<svg width="' + w + '" height="' + h + '" style="flex-shrink:0;"><path d="' + path + '" fill="none" stroke="' + (up?'#10b981':'#ef4444') + '" stroke-width="1.5"/></svg>';
        }

        var clickAction = hasData ? "navigateToTicker('" + realSym + "')" : '';
        var cursor = hasData ? 'cursor-pointer' : 'cursor-default';
        var hoverBorder = hasData ? 'hover:border-slate-600' : '';

        cards += '\
        <div onclick="' + clickAction + '" class="' + cursor + ' bg-slate-900/50 border border-slate-800/60 rounded-lg p-2.5 transition-all ' + hoverBorder + ' flex items-center gap-2.5" style="min-width:0;">\
            <div class="flex-1 min-w-0">\
                <div class="flex items-center gap-1.5">\
                    <span class="text-[11px] font-mono font-bold ' + (hasData ? 'text-cyan-400' : 'text-slate-300') + '">' + lbl + '</span>\
                    <span class="text-[9px] text-slate-500 truncate">' + (name !== lbl ? name : '') + '</span>\
                </div>\
                ' + (hasData ? '\
                <div class="flex items-center gap-2 mt-0.5">\
                    <span class="text-[11px] font-bold text-white font-mono">$' + fp(price) + '</span>\
                    <span class="text-[10px] font-mono font-bold ' + (up?'text-emerald-400':'text-red-400') + '">' + (up?'+':'') + change.toFixed(2) + '%</span>\
                </div>' : '<div class="text-[10px] text-slate-500 mt-0.5 italic">No data loaded</div>') + '\
            </div>\
            ' + spark + '\
        </div>';
    }

    var html = '\
        <div style="margin-bottom:8px;">\
            <span class="text-[11px] font-bold" style="color:var(--text-heading,#e2e8f0);">Your watchlist (' + tickers.length + '):</span>\
        </div>\
        <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:6px; margin-bottom:8px;">\
            ' + cards + '\
        </div>\
        <div class="text-[10px]" style="color:var(--text-muted,#64748b);">\
            Say <b>"add ticker TSLA"</b> or <b>"remove ticker CVX"</b> to modify.\
        </div>';

    var plain = 'Your watchlist (' + tickers.length + '): ' + tickers.join(', ');
    return { html: html, plain: plain };
}

// ═══════════════════════════════════════════════════════════
// TICKER COMMANDS — add/remove/list tickers via agent chat
// ═══════════════════════════════════════════════════════════

function _parseTickerCommand(msg) {
    var m = msg.trim();
    var lower = m.toLowerCase();

    // ── Ticker commands ──

    if (/^(show|list|what('?s| are)?)\s*(my\s*)?(tickers|watchlist|market\s*tickers)/i.test(lower) ||
        /^my\s*(tickers|watchlist)$/i.test(lower)) {
        return { action: 'list' };
    }

    // "add ticker TSLA" / "add TSLA to watchlist" (no "to <other>")
    var addTickerMatch = m.match(/^add\s+(?:ticker[s]?\s+)(.+?)(?:\s+to\s+(?:my\s+)?(?:watchlist|tickers|market))?$/i);
    if (addTickerMatch) {
        var syms = addTickerMatch[1].split(/[,\s]+/).map(function(s) { return s.trim().toUpperCase(); }).filter(function(s) { return s && /^[A-Z0-9.\-=^]{1,15}$/.test(s); });
        if (syms.length) return { action: 'add', symbols: syms };
    }

    // "remove ticker CVX" / "remove CVX from watchlist"
    var rmTickerMatch = m.match(/^(?:remove|delete|drop)\s+(?:ticker[s]?\s+)(.+?)(?:\s+from\s+(?:my\s+)?(?:watchlist|tickers|market))?$/i);
    if (rmTickerMatch) {
        var syms2 = rmTickerMatch[1].split(/[,\s]+/).map(function(s) { return s.trim().toUpperCase(); }).filter(function(s) { return s && /^[A-Z0-9.\-=^]{1,15}$/.test(s); });
        if (syms2.length) return { action: 'remove', symbols: syms2 };
    }

    // ── Category / keyword commands ──

    if (/^(show|list|what('?s| are)?)\s*(my\s*)?(categories|search\s*(keywords|queries)|feeds)/i.test(lower) ||
        /^my\s*(categories|feeds)$/i.test(lower)) {
        return { action: 'list_categories' };
    }

    // "show keywords in <category>"
    var showKwMatch = m.match(/^(?:show|list|what('?s| are)?)\s*(?:the\s+)?(?:keywords?|items?|queries?)\s*(?:in|for|of)\s+(.+)/i);
    if (showKwMatch) {
        return { action: 'show_keywords', category: showKwMatch[2].trim() };
    }

    // "add <keyword> to <category>"
    var addKwMatch = m.match(/^add\s+(.+?)\s+to\s+(.+)/i);
    if (addKwMatch) {
        var cat = addKwMatch[2].replace(/\s*(?:category|feed|keywords?)\s*/i, '').trim();
        if (cat.length > 1 && !/^(?:my\s+)?(?:watchlist|tickers|market)$/i.test(cat)) {
            var keywords = addKwMatch[1].split(/,/).map(function(s) { return s.trim(); }).filter(function(s) { return s.length > 0; });
            if (keywords.length) return { action: 'add_keyword', keywords: keywords, category: cat };
        }
    }

    // "remove <keyword> from <category>"
    var rmKwMatch = m.match(/^(?:remove|delete|drop)\s+(.+?)\s+from\s+(.+)/i);
    if (rmKwMatch) {
        var cat2 = rmKwMatch[2].replace(/\s*(?:category|feed|keywords?)\s*/i, '').trim();
        if (cat2.length > 1 && !/^(?:my\s+)?(?:watchlist|tickers|market)$/i.test(cat2)) {
            var keywords2 = rmKwMatch[1].split(/,/).map(function(s) { return s.trim(); }).filter(function(s) { return s.length > 0; });
            if (keywords2.length) return { action: 'remove_keyword', keywords: keywords2, category: cat2 };
        }
    }

    // "enable/disable <category>"
    var toggleMatch = m.match(/^(enable|disable|turn\s+(?:on|off))\s+(.+?)(?:\s+category)?$/i);
    if (toggleMatch) {
        var on = /enable|on/i.test(toggleMatch[1]);
        return { action: 'toggle_category', category: toggleMatch[2].trim(), enabled: on };
    }

    // ── Fallback: bare "add X" / "remove X" still treated as ticker ──
    var bareAdd = m.match(/^add\s+(.+)$/i);
    if (bareAdd && !bareAdd[1].match(/\s+to\s+/i)) {
        var syms3 = bareAdd[1].split(/[,\s]+/).map(function(s) { return s.trim().toUpperCase(); }).filter(function(s) { return s && /^[A-Z0-9.\-=^]{1,15}$/.test(s); });
        if (syms3.length) return { action: 'add', symbols: syms3 };
    }
    var bareRm = m.match(/^(?:remove|delete|drop)\s+(.+)$/i);
    if (bareRm && !bareRm[1].match(/\s+from\s+/i)) {
        var syms4 = bareRm[1].split(/[,\s]+/).map(function(s) { return s.trim().toUpperCase(); }).filter(function(s) { return s && /^[A-Z0-9.\-=^]{1,15}$/.test(s); });
        if (syms4.length) return { action: 'remove', symbols: syms4 };
    }

    return null;
}

// ── Fuzzy-match a user string to a category id or label ──
function _findCategory(categories, query) {
    var q = query.toLowerCase().replace(/[^a-z0-9\s]/g, '');
    // Exact match on id or label
    var match = categories.find(function(c) { return c.id?.toLowerCase() === q || c.label?.toLowerCase() === q; });
    if (match) return match;
    // Partial / fuzzy
    match = categories.find(function(c) {
        return c.label?.toLowerCase().includes(q) || q.includes(c.label?.toLowerCase()) ||
            c.id?.toLowerCase().includes(q) || q.includes(c.id?.toLowerCase());
    });
    return match || null;
}

async function _executeTickerCommand(cmd) {
    var headers = {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
        'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
    };

    try {
        var cfgRes = await fetch('/api/config', { headers: headers });
        if (!cfgRes.ok) throw new Error('Failed to load config');
        var cfg = await cfgRes.json();
        var tickers = (cfg.market?.tickers || []).map(function(t) { return t.symbol; });
        var categories = cfg.dynamic_categories || [];

        // ── Ticker: list ──
        if (cmd.action === 'list') {
            if (!tickers.length) return { plain: "Your watchlist is empty.", html: formatAgentText("Your watchlist is empty. Say **\"add ticker TSLA\"** to add one.") };
            return _buildWatchlistWidget(tickers);
        }

        // ── Ticker: add ──
        if (cmd.action === 'add') {
            var already = cmd.symbols.filter(function(s) { return tickers.includes(s); });
            var adding = cmd.symbols.filter(function(s) { return !tickers.includes(s); });
            tickers = [].concat(tickers, adding);
            await _saveConfig(headers, { tickers: tickers });
            var msg = '';
            if (adding.length) msg += 'Added **' + adding.join(', ') + '** to your watchlist.';
            if (already.length) msg += (msg ? ' ' : '') + '**' + already.join(', ') + '** already there.';
            msg += '\n\nWatchlist: **' + tickers.join(', ') + '** (' + tickers.length + ')\nRefresh market data to see them on the dashboard.';
            _syncSimpleTickers(tickers);
            return msg;
        }

        // ── Ticker: remove ──
        if (cmd.action === 'remove') {
            var removing = cmd.symbols.filter(function(s) { return tickers.includes(s); });
            var notFound = cmd.symbols.filter(function(s) { return !tickers.includes(s); });
            tickers = tickers.filter(function(t) { return !cmd.symbols.includes(t); });
            await _saveConfig(headers, { tickers: tickers });
            var msg2 = '';
            if (removing.length) msg2 += 'Removed **' + removing.join(', ') + '**.';
            if (notFound.length) msg2 += (msg2 ? ' ' : '') + '**' + notFound.join(', ') + "** wasn't in your watchlist.";
            msg2 += tickers.length ? '\n\nWatchlist: **' + tickers.join(', ') + '** (' + tickers.length + ')' : '\n\nWatchlist is now empty.';
            _syncSimpleTickers(tickers);
            return msg2;
        }

        // ── Categories: list ──
        if (cmd.action === 'list_categories') {
            if (!categories.length) return "No categories configured. Set them up in **Settings**.";
            var lines = categories.map(function(c) {
                var status = c.enabled === false ? '\ud83d\udd34' : '\ud83d\udfe2';
                var count = (c.items || []).length;
                return status + ' **' + c.label + '** \u2014 ' + count + ' keyword' + (count !== 1 ? 's' : '');
            });
            return '**Your categories (' + categories.length + '):**\n\n' + lines.join('\n') + '\n\nSay **"show keywords in [category]"** for details, or **"add [keyword] to [category]"** to modify.';
        }

        // ── Categories: show keywords ──
        if (cmd.action === 'show_keywords') {
            var cat = _findCategory(categories, cmd.category);
            if (!cat) return 'Couldn\'t find a category matching **"' + cmd.category + '"**.\n\nYour categories: ' + categories.map(function(c) { return '**' + c.label + '**'; }).join(', ');
            var items = cat.items || [];
            if (!items.length) return '**' + cat.label + '** has no keywords yet. Say **"add [keyword] to ' + cat.label + '"** to add some.';
            return '**' + cat.label + '** (' + items.length + ' keywords):\n\n' + items.map(function(i) { return '\u2022 ' + i; }).join('\n') + '\n\nSay **"add [keyword] to ' + cat.label + '"** or **"remove [keyword] from ' + cat.label + '"**.';
        }

        // ── Categories: add keyword ──
        if (cmd.action === 'add_keyword') {
            var cat2 = _findCategory(categories, cmd.category);
            if (!cat2) return 'Couldn\'t find a category matching **"' + cmd.category + '"**.\n\nYour categories: ' + categories.map(function(c) { return '**' + c.label + '**'; }).join(', ');
            var existing = (cat2.items || []).map(function(i) { return i.toLowerCase(); });
            var adding2 = cmd.keywords.filter(function(k) { return !existing.includes(k.toLowerCase()); });
            var already2 = cmd.keywords.filter(function(k) { return existing.includes(k.toLowerCase()); });
            cat2.items = [].concat(cat2.items || [], adding2);
            await _saveConfig(headers, { dynamic_categories: categories });
            var msg3 = '';
            if (adding2.length) msg3 += 'Added **' + adding2.join(', ') + '** to **' + cat2.label + '**.';
            if (already2.length) msg3 += (msg3 ? ' ' : '') + '**' + already2.join(', ') + '** already there.';
            msg3 += '\n\n**' + cat2.label + '** now has ' + cat2.items.length + ' keywords. Run a new scan to fetch results.';
            return msg3;
        }

        // ── Categories: remove keyword ──
        if (cmd.action === 'remove_keyword') {
            var cat3 = _findCategory(categories, cmd.category);
            if (!cat3) return 'Couldn\'t find a category matching **"' + cmd.category + '"**.\n\nYour categories: ' + categories.map(function(c) { return '**' + c.label + '**'; }).join(', ');
            var removing2 = cmd.keywords.filter(function(k) { return (cat3.items || []).some(function(i) { return i.toLowerCase() === k.toLowerCase(); }); });
            var notFound2 = cmd.keywords.filter(function(k) { return !(cat3.items || []).some(function(i) { return i.toLowerCase() === k.toLowerCase(); }); });
            cat3.items = (cat3.items || []).filter(function(i) { return !cmd.keywords.some(function(k) { return k.toLowerCase() === i.toLowerCase(); }); });
            await _saveConfig(headers, { dynamic_categories: categories });
            var msg4 = '';
            if (removing2.length) msg4 += 'Removed **' + removing2.join(', ') + '** from **' + cat3.label + '**.';
            if (notFound2.length) msg4 += (msg4 ? ' ' : '') + '**' + notFound2.join(', ') + "** wasn't found.";
            msg4 += '\n\n**' + cat3.label + '** now has ' + cat3.items.length + ' keywords.';
            return msg4;
        }

        // ── Categories: enable/disable ──
        if (cmd.action === 'toggle_category') {
            var cat4 = _findCategory(categories, cmd.category);
            if (!cat4) return 'Couldn\'t find a category matching **"' + cmd.category + '"**.\n\nYour categories: ' + categories.map(function(c) { return '**' + c.label + '**'; }).join(', ');
            cat4.enabled = cmd.enabled;
            await _saveConfig(headers, { dynamic_categories: categories });
            return '**' + cat4.label + '** is now **' + (cmd.enabled ? 'enabled \ud83d\udfe2' : 'disabled \ud83d\udd34') + '**.' + (!cmd.enabled ? " It won't be included in future scans." : ' It will be included in the next scan.');
        }

    } catch (e) {
        return 'Failed to update: ' + e.message;
    }
}

async function _saveConfig(headers, data) {
    var res = await fetch('/api/config', { method: 'POST', headers: headers, body: JSON.stringify(data) });
    if (!res.ok) throw new Error('Config save failed');
}

function _syncSimpleTickers(tickers) {
    if (typeof simpleTickers !== 'undefined') {
        simpleTickers.length = 0;
        tickers.forEach(function(t) { simpleTickers.push(t); });
        if (typeof renderSimpleTickers === 'function') renderSimpleTickers();
    }
}
