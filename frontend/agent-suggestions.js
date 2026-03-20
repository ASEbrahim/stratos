// ═══════════════════════════════════════════════════════════
// AGENT SUGGESTIONS — Dynamic, DB-backed suggestion chips
// No hardcoded suggestions — all generated from profile data or LLM
// ═══════════════════════════════════════════════════════════

function _buildDynamicSuggestions() {
    var suggestions = [];
    try {
        // Gaming-specific: scenario-aware suggestions (no character names — those come from LLM)
        if (currentPersona === 'gaming' && typeof _gamesGetState === 'function') {
            var gs = _gamesGetState();
            if (gs.activeScenario) {
                if (gs.rpMode === 'gm') {
                    suggestions.push("Describe my surroundings");
                    suggestions.push("What happens next?");
                    suggestions.push("What quests are available?");
                }
                // Immersive mode: only show NPC suggestions from DB/LLM cache, not hardcoded
            }
            return suggestions;
        }

        // Build from live profile data
        var profileSugs = _buildProfileSuggestions();
        if (profileSugs.length > 0) {
            suggestions = suggestions.concat(profileSugs);
        }
    } catch(e) {}
    return suggestions;
}

function renderAgentSuggestions() {
    var container = document.getElementById('agent-suggestions');
    if (!container) return;

    // Use cached dynamic suggestions (DB-backed) for this persona if available
    var cached = _personaSuggestions[currentPersona];
    var picks;
    var isDynamic = false;
    if (cached && cached.length > 0) {
        picks = cached.slice(0, 6);
        isDynamic = true;
    } else {
        var suggestions = _buildDynamicSuggestions();
        if (suggestions.length === 0) {
            container.innerHTML = '';
            return;
        }
        var shuffled = [].concat(suggestions).sort(function() { return Math.random() - 0.5; });
        picks = shuffled.slice(0, 6);
    }

    // Map suggestions to icons
    var iconMap = {
        'top': 'trending-up', 'critical': 'alert-triangle', 'market': 'bar-chart-2',
        'ticker': 'list', 'categor': 'tag', 'keyword': 'hash',
        'search': 'globe', 'compare': 'git-compare', 'summarize': 'file-text',
        'signal': 'zap', 'recommend': 'lightbulb', 'concern': 'shield-alert',
        'trend': 'activity', 'add': 'plus', 'show': 'eye',
        'quest': 'compass', 'surround': 'map', 'next': 'arrow-right'
    };

    container.innerHTML = picks.map(function(s) {
        // Handle rich suggestions {label, prompt} from gaming
        var label = typeof s === 'object' ? s.label : s;
        var prompt = typeof s === 'object' ? s.prompt : s;
        var lower = label.toLowerCase();
        var icon = isDynamic ? 'sparkles' : 'message-circle';
        if (!isDynamic) {
            for (var key in iconMap) {
                if (lower.includes(key)) { icon = iconMap[key]; break; }
            }
        }
        var onclick = isDynamic
            ? "document.getElementById('agent-input').value='" + prompt.replace(/'/g,"\\'") + "';sendAgentMessage()"
            : "sendSuggestion(this)";
        return '<button onclick="' + onclick + '" class="text-[10px] px-2.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer whitespace-nowrap" style="border:1px solid var(--border-strong); color:var(--text-muted); background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor=\'var(--accent,#34d399)\';this.style.color=\'var(--accent,#34d399)\';this.style.background=\'rgba(16,185,129,0.06)\'" onmouseleave="this.style.borderColor=\'var(--border-strong)\';this.style.color=\'var(--text-muted)\';this.style.background=\'rgba(255,255,255,0.02)\'"><i data-lucide="' + icon + '" class="w-3 h-3 opacity-60"></i>' + label + '</button>';
    }).join('');
    lucide.createIcons();
}

function sendSuggestion(btn) {
    var text = btn.textContent;
    var welcome = document.getElementById('agent-welcome');
    if (welcome) welcome.remove();
    var input = document.getElementById('agent-input');
    if (input) input.value = text;
    sendAgentMessage();
}

// ═══════════════════════════════════════════════════════════
// RESPONSE SUGGESTION CHIPS
// ═══════════════════════════════════════════════════════════

function _generateResponseChips(response, userMsg) {
    var chips = [];
    if (typeof currentPersona !== 'undefined' &&
        !['intelligence', 'market'].includes(currentPersona)) {
        return chips;
    }
    var rLower = response.toLowerCase();

    // Detect ticker mentions (3-5 uppercase letters)
    var tickerMatches = response.match(/\b([A-Z]{2,5})\b/g);
    if (tickerMatches) {
        var currentTickers = (typeof configData !== 'undefined' && configData?.market?.tickers)
            ? configData.market.tickers.map(function(t) { return (t.symbol || t).toUpperCase(); }) : [];
        var seen = new Set();
        for (var i = 0; i < tickerMatches.length; i++) {
            var t = tickerMatches[i];
            if (seen.has(t) || currentTickers.includes(t)) continue;
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

    if (rLower.includes('search') || rLower.includes('look up') || rLower.includes('find more')) {
        chips.push({
            label: 'Search more',
            icon: 'search',
            action: "document.getElementById('agent-input').value='Search for more details on this topic';sendAgentMessage()",
            tip: 'Search the web for more information'
        });
    }

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

    if (response.length > 800) {
        chips.push({
            label: 'Summarize',
            icon: 'align-left',
            action: "document.getElementById('agent-input').value='Summarize the above in 3 bullet points';sendAgentMessage()",
            tip: 'Get a shorter summary'
        });
    }

    return chips.slice(0, 4);
}

function _applyAgentChip(action) {
    var input = document.getElementById('agent-input');
    if (input) {
        input.value = action;
        sendAgentMessage();
    }
}

// Build context-driven suggestions from user's live profile, categories, and feed data
function _buildProfileSuggestions() {
    var sugs = [];
    try {
        var categories = [];
        if (typeof simpleCategories !== 'undefined' && Array.isArray(simpleCategories)) {
            categories = simpleCategories.filter(function(c) { return c.enabled !== false; }).map(function(c) { return c.label || c.name || ''; }).filter(Boolean);
        }
        var topArticles = [];
        if (typeof newsData !== 'undefined' && Array.isArray(newsData)) {
            topArticles = newsData.filter(function(n) { return (n.score || 0) >= 7; }).slice(0, 5);
        }

        if (currentPersona === 'intelligence' || currentPersona === 'market') {
            if (topArticles.length > 0) {
                var topCat = topArticles[0].category || topArticles[0].root || '';
                if (topCat) sugs.push("Deep dive into " + topCat + " signals");
            }
            if (categories.length > 0) {
                sugs.push("What's trending in " + categories[0] + "?");
                if (categories.length > 1) sugs.push("Compare " + categories[0] + " vs " + categories[1] + " activity");
            }
        }
    } catch(e) {}
    return sugs;
}
