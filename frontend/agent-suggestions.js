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
// ONBOARDING CHIPS — suggestion shortcuts for new users
// Clicking sends the prompt as a normal agent message to the LLM.
// The LLM handles the conversation naturally via onboarding system prompt.
// ═══════════════════════════════════════════════════════════


var _ONBOARDING_CHIPS = [
    { label: 'Tech & AI', prompt: "I'm interested in Tech & AI — artificial intelligence, software, and emerging technology" },
    { label: 'Finance & Markets', prompt: "I'm interested in Finance & Markets — stocks, crypto, and economic trends" },
    { label: 'Career & Jobs', prompt: "I'm interested in Career & Jobs — job market, hiring, and professional development" },
    { label: 'Energy & Oil', prompt: "I'm interested in Energy & Oil — oil prices, renewables, and energy policy" },
    { label: 'Regional News', prompt: "I'm interested in Regional News — Kuwait, GCC, and Middle East" },
    { label: 'Cybersecurity', prompt: "I'm interested in Cybersecurity — threats, vulnerabilities, and InfoSec" },
    { label: 'Something else...', prompt: null, action: 'focus_input' },
];

function renderOnboardingChips(container) {
    var chips = _ONBOARDING_CHIPS;
    var html = '<div class="agent-onboarding-chips" style="display:flex;flex-wrap:wrap;gap:8px;padding:12px 0;">';
    for (var i = 0; i < chips.length; i++) {
        var chip = chips[i];
        var safeLabel = chip.label.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        if (chip.action === 'focus_input') {
            html += '<button onclick="document.getElementById(\'agent-input\')?.focus();document.getElementById(\'agent-input\').placeholder=\'Tell me about yourself...\';"'
                + ' class="text-[13px] px-4 py-2.5 rounded-xl flex items-center gap-2 transition-all cursor-pointer whitespace-nowrap"'
                + ' style="border:1px solid var(--border-strong); color:var(--text-muted); background:rgba(255,255,255,0.02); font-weight:500;"'
                + ' onmouseenter="this.style.borderColor=\'var(--accent)\';this.style.color=\'var(--accent)\'"'
                + ' onmouseleave="this.style.borderColor=\'var(--border-strong)\';this.style.color=\'var(--text-muted)\'">'
                + safeLabel + '</button>';
        } else {
            var safePrompt = chip.prompt.replace(/'/g, "\\'").replace(/\\/g, '\\\\');
            html += '<button onclick="_sendOnboardingChip(\'' + safePrompt + '\')"'
                + ' class="text-[13px] px-4 py-2.5 rounded-xl flex items-center gap-2 transition-all cursor-pointer whitespace-nowrap"'
                + ' style="border:1px solid rgba(52,211,153,0.2); color:var(--text-heading); background:rgba(52,211,153,0.04); font-weight:500;"'
                + ' onmouseenter="this.style.borderColor=\'rgba(52,211,153,0.5)\';this.style.background=\'rgba(52,211,153,0.1)\';this.style.boxShadow=\'0 0 12px rgba(52,211,153,0.15)\';this.style.transform=\'translateY(-1px)\'"'
                + ' onmouseleave="this.style.borderColor=\'rgba(52,211,153,0.2)\';this.style.background=\'rgba(52,211,153,0.04)\';this.style.boxShadow=\'none\';this.style.transform=\'none\'">'
                + safeLabel + '</button>';
        }
    }
    html += '</div>';
    container.insertAdjacentHTML('beforeend', html);
}

function _sendOnboardingChip(prompt) {
    // Remove chips after selection
    document.querySelectorAll('.agent-onboarding-chips').forEach(function(el) { el.remove(); });
    // Send as normal agent message — LLM handles it
    var input = document.getElementById('agent-input');
    if (input) input.value = prompt;
    if (typeof sendAgentMessage === 'function') sendAgentMessage();
}
