// ═══════════════════════════════════════════════════════════
// AGENT SUGGESTIONS — Persona-aware suggestion chips
// Split from agent.js
// ═══════════════════════════════════════════════════════════

// ── Clickable suggestion chips (dynamically generated from profile) ──
var _PERSONA_SUGGESTIONS = {
    intelligence: [
        "What are today's top stories?",
        "What's the most critical alert right now?",
        "Summarize today's news in 3 bullets",
        "What should I pay attention to today?",
        "What's the single best signal for me?",
        "Anything I should be concerned about?",
        "What trends am I seeing this week?",
    ],
    market: [
        "How are the markets doing?",
        "Which assets are up and which are down?",
        "Show my watchlist",
        "Compare BTC and ETH performance",
        "What are the top movers today?",
        "Any earnings reports coming up?",
        "What's the market sentiment?",
    ],
    scholarly: [
        "Tell me about the history of Baghdad",
        "Explain the concept of dialectics",
        "What are the key schools of Islamic philosophy?",
        "Summarize Plato's Republic",
        "What were the major achievements of the Islamic Golden Age?",
        "What were the causes of WWI?",
    ],
    gaming: [
        "Start a new adventure",
        "Describe the scene around me",
        "What are my options right now?",
        "Check my character stats",
        "Talk to the nearest NPC",
        "Set up a new scenario",
        "Continue the story",
    ],
    anime: [
        "What anime is airing this season?",
        "Top rated manga this year",
        "Recommend something like Attack on Titan",
        "What's new on Crunchyroll?",
    ],
    tcg: [
        "What are the most valuable Pokmon cards?",
        "Latest Magic: The Gathering sets",
        "Yu-Gi-Oh meta decks right now",
        "Best TCG investments this month",
    ],
};
var _GENERIC_SUGGESTIONS = _PERSONA_SUGGESTIONS.intelligence;

function _buildDynamicSuggestions() {
    var suggestions = [];
    try {
        // Try to build context-driven suggestions from profile data first
        var profileSugs = _buildProfileSuggestions();
        if (profileSugs.length >= 3) {
            suggestions = profileSugs;
        } else {
            // Fall back to static persona suggestions
            var personaSugs = _PERSONA_SUGGESTIONS[currentPersona] || _PERSONA_SUGGESTIONS.intelligence;
            suggestions = [].concat(personaSugs);
        }
    } catch(e) {
        var personaSugs = _PERSONA_SUGGESTIONS[currentPersona] || _PERSONA_SUGGESTIONS.intelligence;
        suggestions = [].concat(personaSugs);
    }
    try {
        // Gaming-specific: scenario-aware suggestions
        if (currentPersona === 'gaming' && typeof _gamesGetState === 'function') {
            var gs = _gamesGetState();
            if (gs.activeScenario) {
                suggestions.push("Describe my surroundings");
                suggestions.push("What happens next?");
                if (gs.rpMode === 'gm') {
                    suggestions.push("Roll for initiative");
                    suggestions.push("Show my inventory");
                    suggestions.push("What quests are available?");
                    suggestions.push("Level up my character");
                } else if (gs.rpMode === 'immersive') {
                    if (gs.activeNpc) {
                        suggestions.push("Ask " + gs.activeNpc + " about their past");
                        suggestions.push("Challenge " + gs.activeNpc);
                        suggestions.push("Tell " + gs.activeNpc + " a secret");
                    }
                    suggestions.push("Express how I feel about this");
                    suggestions.push("Look around the room");
                }
            }
            return suggestions; // skip non-gaming suggestions
        }

        // Anime, TCG, scholarly — only their own base suggestions
        if (currentPersona === 'anime' || currentPersona === 'tcg' || currentPersona === 'scholarly') {
            return suggestions;
        }

        // Intelligence and market personas get enriched suggestions
        // Ticker management hints
        suggestions.push("Show my tickers");
        suggestions.push("Add ticker TSLA to my watchlist");

        // Web search suggestions
        suggestions.push("Search for Equate hiring in Kuwait");
        suggestions.push("Search for Warba Bank student offers");
        suggestions.push("Search latest NVIDIA news");

        // Pull category names from the sidebar nav
        var navItems = document.querySelectorAll('#nav-menu button[data-section]');
        var categories = [];
        navItems.forEach(function(btn) {
            var label = btn.querySelector('.sidebar-label')?.textContent?.trim();
            var section = btn.dataset.section;
            if (label && section && !['all','settings','saved'].includes(section)) {
                categories.push(label);
            }
        });

        // Category management hints
        suggestions.push("Show my categories");
        if (categories.length) {
            suggestions.push("Show keywords in " + categories[0]);
        }

        // Add category-specific suggestions
        categories.forEach(function(cat) {
            suggestions.push("What's new in " + cat + "?");
            suggestions.push("Any critical signals in " + cat + "?");
        });

        // Pull ticker symbols from market data if available
        if (typeof marketData !== 'undefined' && marketData) {
            var syms = Object.keys(marketData).slice(0, 4).map(function(s) { return s.replace('-USD','').replace('=F',''); });
            if (syms.length >= 2) {
                suggestions.push("Compare " + syms[0] + " and " + syms[1] + " performance");
                suggestions.push("How is " + syms[0] + " doing today?");
            }
        }
    } catch(e) {}
    return suggestions;
}

function renderAgentSuggestions() {
    var container = document.getElementById('agent-suggestions');
    if (!container) return;

    // Use cached dynamic suggestions for this persona if available
    var cached = _personaSuggestions[currentPersona];
    var picks;
    var isDynamic = false;
    if (cached && cached.length > 0) {
        picks = cached.slice(0, 6);
        isDynamic = true;
    } else {
        var suggestions = _buildDynamicSuggestions();
        var shuffled = [].concat(suggestions).sort(function() { return Math.random() - 0.5; });
        picks = shuffled.slice(0, 6);
    }

    // Map suggestions to icons
    var iconMap = {
        'top': 'trending-up', 'critical': 'alert-triangle', 'market': 'bar-chart-2',
        'ticker': 'list', 'categor': 'tag', 'keyword': 'hash',
        'search': 'globe', 'compare': 'git-compare', 'summarize': 'file-text',
        'signal': 'zap', 'recommend': 'lightbulb', 'concern': 'shield-alert',
        'trend': 'activity', 'add': 'plus', 'show': 'eye'
    };

    container.innerHTML = picks.map(function(s) {
        var lower = s.toLowerCase();
        var icon = isDynamic ? 'sparkles' : 'message-circle';
        if (!isDynamic) {
            for (var key in iconMap) {
                if (lower.includes(key)) { icon = iconMap[key]; break; }
            }
        }
        var safeS = s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/</g, '&lt;').replace(/>/g, '&gt;');
        var onclick = isDynamic
            ? "document.getElementById('agent-input').value='" + safeS + "';sendAgentMessage()"
            : "sendSuggestion(this)";
        return '<button onclick="' + onclick + '" class="text-[10px] px-2.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer whitespace-nowrap" style="border:1px solid var(--border-strong); color:var(--text-muted); background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor=\'var(--accent,#34d399)\';this.style.color=\'var(--accent,#34d399)\';this.style.background=\'rgba(16,185,129,0.06)\'" onmouseleave="this.style.borderColor=\'var(--border-strong)\';this.style.color=\'var(--text-muted)\';this.style.background=\'rgba(255,255,255,0.02)\'"><i data-lucide="' + icon + '" class="w-3 h-3 opacity-60"></i>' + safeS + '</button>';
    }).join('');
    lucide.createIcons();
}

function sendSuggestion(btn) {
    var text = btn.textContent;
    // Remove the welcome/suggestions block
    var welcome = document.getElementById('agent-welcome');
    if (welcome) welcome.remove();
    // Set input and send
    var input = document.getElementById('agent-input');
    if (input) input.value = text;
    sendAgentMessage();
}

// ═══════════════════════════════════════════════════════════
// RESPONSE SUGGESTION CHIPS
// ═══════════════════════════════════════════════════════════

function _generateResponseChips(response, userMsg) {
    var chips = [];
    // Response chips are only useful for intelligence and market personas
    if (typeof currentPersona !== 'undefined' &&
        !['intelligence', 'market'].includes(currentPersona)) {
        return chips;
    }
    var rLower = response.toLowerCase();
    var mLower = userMsg.toLowerCase();

    // Detect ticker mentions (3-5 uppercase letters)
    var tickerMatches = response.match(/\b([A-Z]{2,5})\b/g);
    if (tickerMatches) {
        var currentTickers = (typeof configData !== 'undefined' && configData?.market?.tickers)
            ? configData.market.tickers.map(function(t) { return (t.symbol || t).toUpperCase(); }) : [];
        var seen = new Set();
        for (var i = 0; i < tickerMatches.length; i++) {
            var t = tickerMatches[i];
            if (seen.has(t) || currentTickers.includes(t)) continue;
            // Skip common words that look like tickers
            if (['THE','AND','FOR','ARE','BUT','NOT','YOU','ALL','CAN','HAS','HER','WAS','ONE','OUR','OUT','DAY','HAD','HIS','HOW','ITS','MAY','NEW','NOW','OLD','SEE','WAY','WHO','DID','GET','LET','SAY','SHE','TOO','USE','KEY','GDP','CEO','IPO','ETF','USD','EUR','GBP','RSS','API'].includes(t)) continue;
            seen.add(t);
            if (chips.length < 3) {
                chips.push({
                    label: 'Track ' + t,
                    icon: 'plus-circle',
                    action: "document.getElementById('agent-input').value='Add " + t + " to watchlist';sendAgentMessage()",
                    tip: 'Add ' + t + ' to your watchlist'
                });
            }
        }
    }

    // Suggest web search follow-up
    if (rLower.includes('search') || rLower.includes('look up') || rLower.includes('find more')) {
        chips.push({
            label: 'Search more',
            icon: 'search',
            action: "document.getElementById('agent-input').value='Search for more details on this topic';sendAgentMessage()",
            tip: 'Search the web for more information'
        });
    }

    // If response mentions news/market topics, suggest drilling deeper
    if (rLower.includes('market') || rLower.includes('price') || rLower.includes('stock')) {
        if (!chips.some(function(c) { return c.label.includes('Track'); })) {
            chips.push({
                label: 'Market analysis',
                icon: 'bar-chart-2',
                action: "document.getElementById('agent-input').value='Give me a detailed market analysis';sendAgentMessage()",
                tip: 'Get deeper market analysis'
            });
        }
    }

    // Suggest summarize if response is long
    if (response.length > 800) {
        chips.push({
            label: 'Summarize',
            icon: 'align-left',
            action: "document.getElementById('agent-input').value='Summarize the above in 3 bullet points';sendAgentMessage()",
            tip: 'Get a shorter summary'
        });
    }

    return chips.slice(0, 4); // Max 4 chips
}

function _applyAgentChip(action) {
    var input = document.getElementById('agent-input');
    if (input) {
        input.value = action;
        sendAgentMessage();
    }
}

// Build context-driven suggestions from user's profile, categories, and feed data
function _buildProfileSuggestions() {
    var sugs = [];
    try {
        var role = (typeof configData !== 'undefined' && configData?.profile?.role) || '';
        var categories = [];
        // Get dynamic categories from config
        if (typeof simpleCategories !== 'undefined' && Array.isArray(simpleCategories)) {
            categories = simpleCategories.filter(function(c) { return c.enabled !== false; }).map(function(c) { return c.label || c.name || ''; }).filter(Boolean);
        }
        // Get top scored articles for topical suggestions
        var topArticles = [];
        if (typeof newsData !== 'undefined' && Array.isArray(newsData)) {
            topArticles = newsData.filter(function(n) { return (n.score || 0) >= 7; }).slice(0, 5);
        }

        if (currentPersona === 'intelligence' || currentPersona === 'market') {
            if (topArticles.length > 0) {
                sugs.push("What's the most important signal today?");
                var topCat = topArticles[0].category || topArticles[0].root || '';
                if (topCat) sugs.push("Deep dive into " + topCat + " signals");
            }
            if (categories.length > 0) {
                sugs.push("What's trending in " + categories[0] + "?");
                if (categories.length > 1) sugs.push("Compare " + categories[0] + " vs " + categories[1] + " activity");
            }
            if (role) sugs.push("How do today's signals affect my role as " + role.split(' ')[0] + "?");
            sugs.push("Summarize today's feed in 3 bullets");
            if (topArticles.length >= 3) {
                sugs.push("Any patterns across today's top stories?");
            }
        }
    } catch(e) {}
    return sugs;
}

// ═══════════════════════════════════════════════════════════
// INTERACTIVE ONBOARDING — hardcoded multi-step setup flow
// No LLM dependency. Each step: bot message → chips/input → save → next.
// ═══════════════════════════════════════════════════════════

var _obStep = 0; // current onboarding step
var _obData = { role: '', location: '', categories: [], tickers: [] };

var _OB_CATEGORIES = [
    { label: 'Tech & AI', id: 'tech_ai', icon: 'cpu', items: ['artificial intelligence','machine learning','LLM','GPT','software development','cloud computing','startups','open source'], tickers: ['NVDA','MSFT','GOOGL','META'] },
    { label: 'Finance & Markets', id: 'finance', icon: 'bar-chart-3', items: ['stock market','cryptocurrency','federal reserve','interest rates','IPO','earnings','economic policy','inflation'], tickers: ['SPY','BTC-USD','ETH-USD','GC=F'] },
    { label: 'Career & Jobs', id: 'career', icon: 'briefcase', items: ['job market','hiring trends','remote work','layoffs','professional development','salary','recruitment','workforce'], tickers: [] },
    { label: 'Energy & Oil', id: 'energy', icon: 'zap', items: ['oil prices','OPEC','renewable energy','solar','natural gas','energy policy','electric vehicles','crude oil'], tickers: ['CL=F','XOM','TSLA'] },
    { label: 'Regional News', id: 'regional', icon: 'globe', items: ['Kuwait','GCC','Saudi Arabia','UAE','Middle East','Gulf','MENA','Arab'], tickers: [] },
    { label: 'Cybersecurity', id: 'cybersecurity', icon: 'shield', items: ['cybersecurity','data breach','vulnerability','ransomware','zero day','malware','InfoSec','threat intelligence'], tickers: ['CRWD','PANW','FTNT'] },
    { label: 'Science', id: 'science', icon: 'flask-conical', items: ['research','space','physics','biology','climate change','medical research','quantum','NASA'], tickers: [] },
    { label: 'Gaming & Entertainment', id: 'gaming', icon: 'gamepad-2', items: ['video games','esports','streaming','game development','Nintendo','PlayStation','Xbox','indie games'], tickers: ['NTDOY','SONY','EA'] },
];

var _OB_ROLES = ['Software Engineer','Data Scientist','Product Manager','Financial Analyst','Student','Researcher','Entrepreneur','Journalist','Marketing Manager','Designer'];
var _OB_LOCATIONS = ['United States','United Kingdom','Kuwait','UAE','Saudi Arabia','Canada','Germany','India','Japan','Australia'];

function _obChipHtml(label, onclick, opts) {
    var selected = opts && opts.selected;
    var accent = opts && opts.accent;
    var bg = selected ? 'rgba(52,211,153,0.15)' : (accent ? 'var(--accent,#10b981)' : 'rgba(52,211,153,0.04)');
    var color = selected ? 'var(--accent,#34d399)' : (accent ? 'var(--bg-primary,#0f172a)' : 'var(--text-heading)');
    var border = selected ? 'rgba(52,211,153,0.5)' : (accent ? 'var(--accent,#10b981)' : 'rgba(52,211,153,0.2)');
    var weight = (selected || accent) ? '600' : '500';
    var shadow = accent ? 'box-shadow:0 2px 12px rgba(16,185,129,0.25);' : '';
    return '<button onclick="' + onclick.replace(/"/g, '&quot;') + '"'
        + ' class="text-[13px] px-4 py-2.5 rounded-xl flex items-center gap-2 transition-all cursor-pointer"'
        + ' style="border:1px solid ' + border + '; color:' + color + '; background:' + bg + '; font-weight:' + weight + ';' + shadow + '"'
        + ' onmouseenter="this.style.borderColor=\'rgba(52,211,153,0.5)\';this.style.boxShadow=\'0 0 12px rgba(52,211,153,0.15)\';this.style.transform=\'translateY(-1px)\'"'
        + ' onmouseleave="this.style.borderColor=\'' + border + '\';this.style.boxShadow=\'' + (accent ? '0 2px 12px rgba(16,185,129,0.25)' : 'none') + '\';this.style.transform=\'none\'">'
        + label + '</button>';
}

function _obMsg(text) {
    if (typeof appendAgentMessage === 'function') {
        appendAgentMessage('assistant', typeof formatAgentText === 'function' ? formatAgentText(text) : text);
    }
}

function _obUserMsg(text) {
    if (typeof appendAgentMessage === 'function') {
        appendAgentMessage('user', text);
    }
}

function _obChips(html) {
    var el = document.getElementById('agent-messages');
    if (el) {
        el.insertAdjacentHTML('beforeend', '<div class="agent-onboarding-chips" style="display:flex;flex-wrap:wrap;gap:8px;padding:12px 0;">' + html + '</div>');
        el.scrollTop = el.scrollHeight;
    }
}

function _obClearChips() {
    var chips = document.querySelectorAll('.agent-onboarding-chips');
    for (var i = 0; i < chips.length; i++) chips[i].remove();
}

function _obToken() {
    return typeof getAuthToken === 'function' ? getAuthToken() : '';
}

function _obSaveConfig(payload) {
    return fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Auth-Token': _obToken() },
        body: JSON.stringify(payload)
    });
}

// ── STEP 1: Ask role ──
function _obStep1() {
    _obStep = 1;
    console.log('onboarding: step 1 — role');

    // Terminal-style greeting (custom HTML, not appendAgentMessage)
    var el = document.getElementById('agent-messages');
    if (el) {
        var greetHtml = '<div class="flex gap-3 mb-3">'
            + '<div class="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5" style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.15);">'
            + '<i data-lucide="bot" class="w-3.5 h-3.5 text-emerald-400"></i></div>'
            + '<div class="flex-1 min-w-0"><div class="ob-greeting">'
            + '<div class="ob-line1">Intelligence profile initialized.</div>'
            + '<div class="ob-line2">I\'m your StratOS agent.<span class="ob-cursor"></span></div>'
            + '<div class="ob-body">Let\'s set up your feed. First \u2014 what\'s your role or profession?</div>'
            + '</div></div></div>';
        el.insertAdjacentHTML('beforeend', greetHtml);
        if (typeof lucide !== 'undefined') lucide.createIcons();
        el.scrollTop = el.scrollHeight;
    }

    var html = '';
    for (var i = 0; i < _OB_ROLES.length; i++) {
        html += _obChipHtml(_OB_ROLES[i], '_obSelectRole(\'' + _OB_ROLES[i].replace(/'/g, "\\'") + '\')');
    }
    _obChips(html);
    window._obExpectingInput = 'role';
}

function _obSelectRole(role) {
    window._obExpectingInput = null;
    _obClearChips();
    _obUserMsg(role);
    _obData.role = role;
    console.log('onboarding: role =', role);
    setTimeout(_obStep2, 400);
}

// ── STEP 2: Ask location ──
function _obStep2() {
    _obStep = 2;
    console.log('onboarding: step 2 — location');
    _obMsg('Got it, ' + _obData.role + '. Where are you based?');
    var html = '';
    for (var i = 0; i < _OB_LOCATIONS.length; i++) {
        html += _obChipHtml(_OB_LOCATIONS[i], '_obSelectLocation(\'' + _OB_LOCATIONS[i].replace(/'/g, "\\'") + '\')');
    }
    _obChips(html);
    window._obExpectingInput = 'location';
}

function _obSelectLocation(loc) {
    window._obExpectingInput = null;
    _obClearChips();
    _obUserMsg(loc);
    _obData.location = loc;
    console.log('onboarding: location =', loc);
    // Save role + location now
    _obSaveConfig({ profile: { role: _obData.role, location: _obData.location } }).then(function() {
        console.log('onboarding: profile saved');
    });
    setTimeout(_obStep3, 400);
}

// ── STEP 3: Pick interests (multi-select) ──
function _obStep3() {
    _obStep = 3;
    console.log('onboarding: step 3 — categories');
    _obMsg('What topics do you want to track? Pick as many as you like, then hit **Done**.');
    _obData.categories = [];
    _obRenderCategoryChips();
}

function _obRenderCategoryChips() {
    _obClearChips();
    var html = '';
    for (var i = 0; i < _OB_CATEGORIES.length; i++) {
        var cat = _OB_CATEGORIES[i];
        var sel = _obData.categories.indexOf(cat.id) >= 0;
        html += _obChipHtml((sel ? '\u2713 ' : '') + cat.label, '_obToggleCategory(\'' + cat.id + '\')', { selected: sel });
    }
    if (_obData.categories.length > 0) {
        html += _obChipHtml('Done \u2192', '_obCategoriesDone()', { accent: true });
    }
    _obChips(html);
}

function _obToggleCategory(id) {
    var idx = _obData.categories.indexOf(id);
    if (idx >= 0) {
        _obData.categories.splice(idx, 1);
    } else {
        _obData.categories.push(id);
    }
    _obRenderCategoryChips();
}

function _obCategoriesDone() {
    _obClearChips();
    var labels = _obData.categories.map(function(id) {
        var cat = _OB_CATEGORIES.find(function(c) { return c.id === id; });
        return cat ? cat.label : id;
    });
    _obUserMsg(labels.join(', '));

    // Build and save categories
    var cats = [];
    var tickers = [];
    for (var i = 0; i < _obData.categories.length; i++) {
        var cat = _OB_CATEGORIES.find(function(c) { return c.id === _obData.categories[i]; });
        if (cat) {
            cats.push({ label: cat.label, id: cat.id, icon: cat.icon, items: cat.items, enabled: true });
            tickers = tickers.concat(cat.tickers || []);
        }
    }
    // Dedupe tickers
    _obData.tickers = tickers.filter(function(t, i) { return tickers.indexOf(t) === i; });

    _obSaveConfig({ dynamic_categories: cats }).then(function() {
        console.log('onboarding: categories saved');
    });

    setTimeout(_obStep4, 400);
}

// ── STEP 4: Market tickers ──
function _obStep4() {
    _obStep = 4;
    console.log('onboarding: step 4 — tickers');
    if (_obData.tickers.length > 0) {
        _obMsg('Based on your interests, I\'d suggest tracking these markets:\n\n**' + _obData.tickers.join(', ') + '**\n\nWant to add these to your watchlist?');
        var html = _obChipHtml('Yes, add them', '_obTickersAccept()', { accent: true });
        html += _obChipHtml('Skip for now', '_obTickersSkip()');
        html += _obChipHtml('Let me choose', '_obTickersCustom()');
        _obChips(html);
    } else {
        _obMsg('Want to track any market tickers? (stocks, crypto, commodities)');
        var html = _obChipHtml('Skip for now', '_obTickersSkip()');
        html += _obChipHtml('Let me choose', '_obTickersCustom()');
        _obChips(html);
    }
}

function _obTickersAccept() {
    _obClearChips();
    _obUserMsg('Add ' + _obData.tickers.join(', '));
    _obSaveTickers(_obData.tickers);
    setTimeout(_obStep5, 400);
}

function _obTickersSkip() {
    _obClearChips();
    _obUserMsg('Skip tickers');
    _obMsg('No problem \u2014 you can add tickers anytime from the Markets tab.');
    setTimeout(_obStep5, 600);
}

function _obTickersCustom() {
    _obClearChips();
    _obMsg('Type ticker symbols separated by commas (e.g. NVDA, BTC-USD, TSLA):');
    window._obExpectingInput = 'tickers';
}

function _obSaveTickers(symbols) {
    var tickerObjs = symbols.map(function(s) { return { symbol: s.trim().toUpperCase() }; });
    _obSaveConfig({ market: { tickers: tickerObjs } }).then(function() {
        console.log('onboarding: tickers saved');
    });
}

// ── STEP 5: Run scan ──
function _obStep5() {
    _obStep = 5;
    console.log('onboarding: step 5 — scan');
    _obMsg('Your profile is set up:\n\n- **Role:** ' + (_obData.role || 'Not set') + '\n- **Location:** ' + (_obData.location || 'Not set') + '\n- **Categories:** ' + (_obData.categories.length || 0) + ' topics\n- **Tickers:** ' + (_obData.tickers.length > 0 ? _obData.tickers.join(', ') : 'None') + '\n\nReady to run your first scan? This will fetch and score articles matching your interests.');
    var html = _obChipHtml('Run First Scan', '_obRunScan()', { accent: true });
    _obChips(html);
}

function _obRunScan() {
    _obClearChips();
    _obUserMsg('Run my first scan');
    _obMsg('Scanning now. I\'ll pull articles matching your profile and score them for relevance. This takes about a minute \u2014 I\'ll summarize the results when it\'s done.');
    window._onboardingScanListener = true;
    if (typeof toggleScan === 'function') toggleScan();
    // Mark onboarding seen (scan listener handles completion)
    localStorage.setItem('stratos-onboarding-seen', 'true');
}

// ── Handle typed input during onboarding ──
function _obHandleUserInput(msg) {
    if (!window._obExpectingInput) return false;
    var field = window._obExpectingInput;
    window._obExpectingInput = null;

    if (field === 'role') {
        _obData.role = msg;
        _obClearChips();
        console.log('onboarding: role (typed) =', msg);
        setTimeout(_obStep2, 400);
        return true;
    }
    if (field === 'location') {
        _obData.location = msg;
        _obClearChips();
        console.log('onboarding: location (typed) =', msg);
        _obSaveConfig({ profile: { role: _obData.role, location: _obData.location } });
        setTimeout(_obStep3, 400);
        return true;
    }
    if (field === 'tickers') {
        var symbols = msg.split(/[,\s]+/).filter(function(s) { return s.length >= 1; });
        _obData.tickers = symbols.map(function(s) { return s.toUpperCase(); });
        _obSaveTickers(_obData.tickers);
        setTimeout(_obStep5, 400);
        return true;
    }
    return false;
}
