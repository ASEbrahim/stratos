// ═══════════════════════════════════════════════════════════
// STRAT AGENT — AI Chat over scraped intelligence data
// ═══════════════════════════════════════════════════════════

let agentHistory = [];   // {role:'user'|'assistant', content:string}
let agentOpen = false;
let agentStreaming = false;
let agentMode = 'structured'; // 'structured' or 'free'
let _agentAbortController = null;
let currentPersona = 'intelligence';
let selectedPersonas = ['intelligence']; // Multi-persona selection (max 3)
let availablePersonas = [];
let _personaSuggestions = {};  // Per-persona dynamic suggestion cache
try { _personaSuggestions = JSON.parse(localStorage.getItem('stratos_persona_suggestions') || '{}'); } catch(e) {}

// ── Conversation management (DB-backed via /api/conversations) ──
let _agentConvList = [];       // [{id, persona, title, is_active, message_count, ...}]
let _agentActiveConvId = null;  // current active conversation DB id

function _agentHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

async function _loadConversations(persona) {
    try {
        const r = await fetch(`/api/conversations?persona=${encodeURIComponent(persona || currentPersona)}`, { headers: _agentHeaders() });
        if (r.ok) {
            const d = await r.json();
            _agentConvList = d.conversations || [];
            // Find active conversation
            const active = _agentConvList.find(c => c.is_active);
            if (active) {
                _agentActiveConvId = active.id;
            } else if (_agentConvList.length > 0) {
                _agentActiveConvId = _agentConvList[0].id;
            } else {
                _agentActiveConvId = null;
            }
        }
    } catch(e) { _agentConvList = []; _agentActiveConvId = null; }
}

async function _loadConvMessages(convId) {
    try {
        const r = await fetch(`/api/conversations/${convId}`, { headers: _agentHeaders() });
        if (r.ok) {
            const d = await r.json();
            return d.messages || [];
        }
    } catch(e) {}
    return [];
}

async function _switchConversation(convId) {
    if (agentStreaming) return;
    _agentActiveConvId = convId;
    // Set active on server
    fetch(`/api/conversations/${convId}`, {
        method: 'PUT', headers: _agentHeaders(),
        body: JSON.stringify({ is_active: true })
    }).catch(() => {});
    // Load messages
    const messages = await _loadConvMessages(convId);
    agentHistory = messages;
    // Re-render chat
    const msgs = document.getElementById('agent-messages');
    if (msgs) msgs.innerHTML = '';
    if (agentHistory.length === 0) {
        _updatePersonaWelcome();
    } else {
        const welcome = document.getElementById('agent-welcome');
        if (welcome) welcome.style.display = 'none';
        _renderRestoredHistory();
    }
    _renderConvTabs();
    if (_agentFullscreen) _refreshFsSidebar();
}
window._switchConversation = _switchConversation;

async function newAgentChat() {
    if (agentStreaming) return;
    // Don't create if current conv is empty
    if (_agentActiveConvId && agentHistory.length === 0) return;
    try {
        const r = await fetch('/api/conversations', {
            method: 'POST', headers: _agentHeaders(),
            body: JSON.stringify({ persona: currentPersona, title: 'New Chat' })
        });
        if (r.ok) {
            const d = await r.json();
            _agentActiveConvId = d.id;
            agentHistory = [];
            await _loadConversations(currentPersona);
            // Clear UI
            const msgs = document.getElementById('agent-messages');
            if (msgs) msgs.innerHTML = '';
            _updatePersonaWelcome();
            _renderConvTabs();
            if (_agentFullscreen) _refreshFsSidebar();
            renderAgentSuggestions();
        }
    } catch(e) {}
}
window.newAgentChat = newAgentChat;

async function _deleteConversation(convId) {
    try {
        await fetch(`/api/conversations/${convId}`, { method: 'DELETE', headers: _agentHeaders() });
        await _loadConversations(currentPersona);
        if (_agentActiveConvId === convId || !_agentConvList.find(c => c.id === _agentActiveConvId)) {
            if (_agentConvList.length > 0) {
                await _switchConversation(_agentConvList[0].id);
            } else {
                await newAgentChat();
            }
        }
        _renderConvTabs();
        if (_agentFullscreen) _refreshFsSidebar();
    } catch(e) {}
}
window._deleteConversation = _deleteConversation;

function _renderConvTabs() {
    const container = document.getElementById('agent-conv-tabs');
    if (!container) return;
    if (_agentConvList.length <= 1 && (!_agentConvList[0] || _agentConvList[0].message_count === 0)) {
        container.innerHTML = '';
        return;
    }
    const theme = PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence;
    container.innerHTML = _agentConvList.map(c => {
        const active = c.id === _agentActiveConvId;
        const title = (c.title || 'New Chat').length > 24 ? (c.title || 'New Chat').slice(0, 22) + '…' : (c.title || 'New Chat');
        return `<button onclick="_switchConversation(${c.id})" ondblclick="event.stopPropagation();_renameConversation(${c.id})" class="flex items-center gap-1 px-2 py-1 rounded-md text-[9px] font-medium whitespace-nowrap transition-all flex-shrink-0 group" style="background:${active ? theme.bg : 'transparent'};border:1px solid ${active ? theme.color + '40' : 'transparent'};color:${active ? theme.color : 'var(--text-muted)'};" title="Double-click to rename" onmouseenter="if(!${active})this.style.background='rgba(255,255,255,0.03)'" onmouseleave="if(!${active})this.style.background='transparent'">${escAgent(title)}${_agentConvList.length > 1 ? `<span onclick="event.stopPropagation();_deleteConversation(${c.id})" class="opacity-0 group-hover:opacity-100 ml-1 transition-opacity" style="color:var(--text-muted);" title="Delete">×</span>` : ''}</button>`;
    }).join('') + `<button onclick="newAgentChat()" class="px-1.5 py-1 rounded-md text-[9px] transition-all flex-shrink-0" style="color:var(--text-muted);" title="New chat" onmouseenter="this.style.color='${theme.color}'" onmouseleave="this.style.color='var(--text-muted)'"><i data-lucide="plus" class="w-3 h-3"></i></button>`;
    lucide.createIcons();
}

async function _renameConversation(convId) {
    const conv = _agentConvList.find(c => c.id === convId);
    if (!conv) return;
    const newTitle = prompt('Rename conversation:', conv.title || 'New Chat');
    if (newTitle === null || !newTitle.trim()) return;
    try {
        await fetch(`/api/conversations/${convId}`, {
            method: 'PUT', headers: _agentHeaders(),
            body: JSON.stringify({ title: newTitle.trim() })
        });
        await _loadConversations(currentPersona);
        _renderConvTabs();
        if (_agentFullscreen) _refreshFsSidebar();
    } catch(e) {}
}
window._renameConversation = _renameConversation;

// Save agent history to DB (called after each exchange)
async function _saveAgentHistory() {
    if (!_agentActiveConvId) return;
    try {
        const messages = agentHistory.slice(-40);
        // Auto-title from first user message
        const firstUser = messages.find(m => m.role === 'user');
        const title = firstUser ? firstUser.content.slice(0, 40).trim() || 'Chat' : 'New Chat';
        await fetch(`/api/conversations/${_agentActiveConvId}`, {
            method: 'PUT', headers: _agentHeaders(),
            body: JSON.stringify({ messages, title })
        });
        // Refresh list to update title/counts in tabs
        await _loadConversations(currentPersona);
        _renderConvTabs();
        if (_agentFullscreen) _refreshFsSidebar();
    } catch(e) {}
}

async function _restoreAgentHistory() {
    // Migrate old localStorage conversations to DB (one-time)
    try {
        const old = localStorage.getItem('stratos_agent_conversations');
        if (old) {
            const parsed = JSON.parse(old);
            for (const [persona, convs] of Object.entries(parsed)) {
                if (!Array.isArray(convs)) continue;
                for (const conv of convs) {
                    if (!conv.messages || conv.messages.length === 0) continue;
                    try {
                        const r = await fetch('/api/conversations', {
                            method: 'POST', headers: _agentHeaders(),
                            body: JSON.stringify({ persona, title: conv.title || 'Migrated Chat' })
                        });
                        if (r.ok) {
                            const d = await r.json();
                            await fetch(`/api/conversations/${d.id}`, {
                                method: 'PUT', headers: _agentHeaders(),
                                body: JSON.stringify({ messages: conv.messages })
                            });
                        }
                    } catch(e) {}
                }
            }
            localStorage.removeItem('stratos_agent_conversations');
            localStorage.removeItem('stratos_agent_active_conv');
            localStorage.removeItem('stratos_agent_history');
        }
    } catch(e) {}

    await _loadConversations(currentPersona);
    if (_agentActiveConvId) {
        agentHistory = await _loadConvMessages(_agentActiveConvId);
        if (agentHistory.length > 0) {
            _renderRestoredHistory();
        }
    } else {
        // Create first conversation if none exist
        await newAgentChat();
    }
    _renderConvTabs();
}

function _renderRestoredHistory() {
    const msgs = document.getElementById('agent-messages');
    if (!msgs || agentHistory.length === 0) return;
    const welcome = document.getElementById('agent-welcome');
    if (welcome) welcome.style.display = 'none';
    for (const h of agentHistory) {
        const div = document.createElement('div');
        div.className = h.role === 'user'
            ? 'flex justify-end mb-2'
            : 'flex justify-start mb-2';
        const bubble = document.createElement('div');
        bubble.className = h.role === 'user'
            ? 'agent-bubble-user max-w-[85%] rounded-2xl px-3 py-2 text-sm'
            : 'agent-bubble-ai max-w-[85%] rounded-2xl px-3 py-2 text-sm';
        bubble.innerHTML = h.role === 'assistant' ? formatAgentText(h.content) : escAgent(h.content);
        div.appendChild(bubble);
        msgs.appendChild(div);
    }
    msgs.scrollTop = msgs.scrollHeight;
}

function _autoResizeAgentInput(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 96) + 'px';
}

// Mode toggle removed — always use structured mode with tools.
// Free mode added confusion without meaningful value (Problems 6 & 7).
function toggleAgentMode() {
    // No-op: structured mode is always active
}

const PERSONA_SUBTITLES = {
    intelligence: 'Search the web, manage your feed, analyze signals',
    market: 'Market data, price analysis, watchlist management',
    scholarly: 'History, language, philosophy, academic discussion',
    anime: 'Anime & manga tracking (coming soon)',
    tcg: 'Trading card games (coming soon)',
    gaming: 'Gaming news & deals (coming soon)',
};
const PERSONA_WELCOMES = {
    intelligence: { title: 'How can I help?', desc: 'I can search the web, manage your watchlist & categories, and analyze your feed data.' },
    market: { title: 'Market Intelligence', desc: 'Ask about prices, manage your watchlist, compare assets, or get market analysis.' },
    scholarly: { title: 'Scholarly Assistant', desc: 'Ask about history, philosophy, language, or any academic topic.' },
    gaming: { title: 'Gaming Hub', desc: 'Game deals, news, releases, and recommendations.' },
    anime: { title: 'Anime & Manga', desc: 'Seasonal anime, manga recommendations, and otaku culture.' },
    tcg: { title: 'Trading Card Games', desc: 'Card valuations, meta decks, set releases, and TCG news.' },
};

async function switchPersona(name) {
    // Save current conversation before switching
    _saveAgentHistory();
    currentPersona = name;
    selectedPersonas = [name];
    _updatePersonaPickerLabel();
    const subtitle = document.querySelector('#agent-panel .text-\\[10px\\].mt-0\\.5');
    if (subtitle) subtitle.textContent = PERSONA_SUBTITLES[name] || PERSONA_SUBTITLES.intelligence;
    const msgs = document.getElementById('agent-messages');
    if (msgs) {
        // Brief transition animation
        const theme = PERSONA_THEMES[name] || PERSONA_THEMES.intelligence;
        msgs.innerHTML = `<div class="flex flex-col items-center justify-center py-8" style="animation:fadeIn 0.3s ease;">
            <i data-lucide="${theme.icon}" class="w-8 h-8 mb-2" style="color:${theme.color};"></i>
            <div class="text-[11px] font-semibold" style="color:${theme.color};">${theme.label}</div>
            <div class="text-[9px] mt-0.5" style="color:var(--text-muted);">Loading context...</div>
        </div>`;
        lucide.createIcons();
        // Load persona's conversations from DB
        await _loadConversations(name);
        if (_agentActiveConvId) {
            agentHistory = await _loadConvMessages(_agentActiveConvId);
        } else {
            agentHistory = [];
            // Create a new conversation for this persona
            await newAgentChat();
        }
        setTimeout(() => {
            msgs.innerHTML = '';
            if (agentHistory.length > 0) {
                const welcome = document.getElementById('agent-welcome');
                if (welcome) welcome.style.display = 'none';
                _renderRestoredHistory();
            } else {
                _updatePersonaWelcome();
            }
        }, 400);
    }
    _renderConvTabs();
    if (_agentFullscreen) _refreshFsSidebar();
    renderAgentSuggestions();
    // Update context indicator
    _updateContextBadge(name);
    if (typeof _onPersonaChanged === 'function') _onPersonaChanged(name);
    if (typeof updateScenarioBar === 'function') updateScenarioBar();
}

// ── Persona theme data ──
const PERSONA_THEMES = {
    intelligence: { icon: 'radar', color: '#34d399', bg: 'rgba(16,185,129,0.1)', label: 'Intelligence' },
    market:       { icon: 'trending-up', color: '#60a5fa', bg: 'rgba(96,165,250,0.1)', label: 'Market' },
    scholarly:    { icon: 'book-open', color: '#c084fc', bg: 'rgba(192,132,252,0.1)', label: 'Scholarly' },
    gaming:       { icon: 'gamepad-2', color: '#f472b6', bg: 'rgba(244,114,182,0.1)', label: 'Gaming' },
    anime:        { icon: 'sparkles', color: '#fb923c', bg: 'rgba(251,146,60,0.1)', label: 'Anime' },
    tcg:          { icon: 'layers', color: '#fbbf24', bg: 'rgba(251,191,36,0.1)', label: 'TCG' },
};

// ── Multi-Persona Picker ──
function _togglePersonaPicker() {
    const dd = document.getElementById('persona-picker-dropdown');
    if (!dd) return;
    if (dd.classList.contains('hidden')) {
        _renderPersonaPicker();
        dd.classList.remove('hidden');
        // Close on outside click
        setTimeout(() => document.addEventListener('click', _closePersonaPicker, { once: true }), 0);
    } else {
        dd.classList.add('hidden');
    }
}
window._togglePersonaPicker = _togglePersonaPicker;

function _closePersonaPicker() {
    const dd = document.getElementById('persona-picker-dropdown');
    if (dd) dd.classList.add('hidden');
}

function _renderPersonaPicker() {
    const dd = document.getElementById('persona-picker-dropdown');
    if (!dd) return;
    const personas = availablePersonas.length ? availablePersonas : [
        {name:'intelligence'},{name:'market'},{name:'scholarly'},{name:'gaming'},{name:'anime'},{name:'tcg'}
    ];
    dd.innerHTML = personas.map(p => {
        const checked = selectedPersonas.includes(p.name);
        const disabled = !checked && selectedPersonas.length >= 3;
        const theme = PERSONA_THEMES[p.name] || PERSONA_THEMES.intelligence;
        return `<label class="flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-colors text-[10px] ${disabled ? 'opacity-40' : ''}" style="color:var(--text-secondary)" onmouseenter="this.style.background='var(--bg-hover)'" onmouseleave="this.style.background='transparent'">
            <input type="checkbox" ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''} value="${p.name}" onchange="_onPersonaCheckChange(this)" style="width:12px;height:12px;accent-color:${theme.color};">
            <i data-lucide="${theme.icon}" class="w-3 h-3" style="color:${theme.color};"></i>
            <span>${theme.label}</span>
        </label>`;
    }).join('');
    lucide.createIcons();
}

function _onPersonaCheckChange(cb) {
    const name = cb.value;
    if (cb.checked) {
        if (selectedPersonas.length >= 3) { cb.checked = false; return; }
        if (!selectedPersonas.includes(name)) selectedPersonas.push(name);
    } else {
        selectedPersonas = selectedPersonas.filter(p => p !== name);
        if (selectedPersonas.length === 0) {
            selectedPersonas = ['intelligence'];
            cb.checked = false;
        }
    }
    currentPersona = selectedPersonas[0];
    _updatePersonaPickerLabel();
    _renderPersonaPicker();
    const subtitle = document.querySelector('#agent-panel .text-\\[10px\\].mt-0\\.5');
    if (subtitle) subtitle.textContent = selectedPersonas.length > 1
        ? `Multi-agent: ${selectedPersonas.map(p => (PERSONA_THEMES[p]||PERSONA_THEMES.intelligence).label).join(' + ')}`
        : (PERSONA_SUBTITLES[currentPersona] || PERSONA_SUBTITLES.intelligence);
    _updatePersonaWelcome();
    renderAgentSuggestions();
    if (typeof _onPersonaChanged === 'function') _onPersonaChanged(currentPersona);
    if (typeof updateScenarioBar === 'function') updateScenarioBar();
}
window._onPersonaCheckChange = _onPersonaCheckChange;

function _updatePersonaPickerLabel() {
    const label = document.getElementById('persona-picker-label');
    if (!label) return;
    if (selectedPersonas.length === 1) {
        const theme = PERSONA_THEMES[selectedPersonas[0]] || PERSONA_THEMES.intelligence;
        label.innerHTML = `<span style="color:${theme.color};">${theme.label}</span>`;
    } else {
        label.innerHTML = selectedPersonas.map(p => {
            const t = PERSONA_THEMES[p] || PERSONA_THEMES.intelligence;
            return `<span class="px-1 py-0.5 rounded" style="background:${t.bg};color:${t.color};">${t.label}</span>`;
        }).join(' ');
    }
    // Update the picker button border to match primary persona
    const btn = label.closest('button');
    const primaryTheme = PERSONA_THEMES[selectedPersonas[0]] || PERSONA_THEMES.intelligence;
    if (btn) {
        btn.style.borderColor = primaryTheme.color + '30';
        btn.style.background = primaryTheme.bg;
    }
}

async function loadPersonas() {
    try {
        const r = await fetch('/api/agent-personas', {
            headers: { 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' }
        });
        if (r.ok) {
            const d = await r.json();
            availablePersonas = d.personas || [];
            _updatePersonaPickerLabel();
        }
    } catch (e) { /* ignore — fallback persona list in picker */ }
}

// ── Clickable suggestion chips (dynamically generated from profile) ──
const _PERSONA_SUGGESTIONS = {
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
        "Compare Sunni and Shia jurisprudence",
        "What were the causes of WWI?",
    ],
    gaming: [
        "What are the top gaming deals right now?",
        "Any new game releases this week?",
        "What's trending on Steam?",
        "Best indie games of this year",
        "Compare PS5 vs Xbox Series X specs",
    ],
    anime: [
        "What anime is airing this season?",
        "Top rated manga this year",
        "Recommend something like Attack on Titan",
        "What's new on Crunchyroll?",
    ],
    tcg: [
        "What are the most valuable Pokémon cards?",
        "Latest Magic: The Gathering sets",
        "Yu-Gi-Oh meta decks right now",
        "Best TCG investments this month",
    ],
};
const _GENERIC_SUGGESTIONS = _PERSONA_SUGGESTIONS.intelligence;

function _buildDynamicSuggestions() {
    const personaSugs = _PERSONA_SUGGESTIONS[currentPersona] || _PERSONA_SUGGESTIONS.intelligence;
    const suggestions = [...personaSugs];
    try {
        // Ticker management hints
        suggestions.push("Show my tickers");
        suggestions.push("Add ticker TSLA to my watchlist");

        // Web search suggestions
        suggestions.push("Search for Equate hiring in Kuwait");
        suggestions.push("Search for Warba Bank student offers");
        suggestions.push("Search latest NVIDIA news");

        // Pull category names from the sidebar nav
        const navItems = document.querySelectorAll('#nav-menu button[data-section]');
        const categories = [];
        navItems.forEach(btn => {
            const label = btn.querySelector('.sidebar-label')?.textContent?.trim();
            const section = btn.dataset.section;
            if (label && section && !['all','settings','saved'].includes(section)) {
                categories.push(label);
            }
        });

        // Category management hints
        suggestions.push("Show my categories");
        if (categories.length) {
            suggestions.push(`Show keywords in ${categories[0]}`);
        }

        // Add category-specific suggestions
        categories.forEach(cat => {
            suggestions.push(`What's new in ${cat}?`);
            suggestions.push(`Any critical signals in ${cat}?`);
        });

        // Pull ticker symbols from market data if available
        if (typeof marketData !== 'undefined' && marketData) {
            const syms = Object.keys(marketData).slice(0, 4).map(s => s.replace('-USD','').replace('=F',''));
            if (syms.length >= 2) {
                suggestions.push(`Compare ${syms[0]} and ${syms[1]} performance`);
                suggestions.push(`How is ${syms[0]} doing today?`);
            }
        }
    } catch(e) {}
    return suggestions;
}

function renderAgentSuggestions() {
    const container = document.getElementById('agent-suggestions');
    if (!container) return;

    // Use cached dynamic suggestions for this persona if available
    const cached = _personaSuggestions[currentPersona];
    let picks;
    let isDynamic = false;
    if (cached && cached.length > 0) {
        picks = cached.slice(0, 6);
        isDynamic = true;
    } else {
        const suggestions = _buildDynamicSuggestions();
        const shuffled = [...suggestions].sort(() => Math.random() - 0.5);
        picks = shuffled.slice(0, 6);
    }

    // Map suggestions to icons
    const iconMap = {
        'top': 'trending-up', 'critical': 'alert-triangle', 'market': 'bar-chart-2',
        'ticker': 'list', 'categor': 'tag', 'keyword': 'hash',
        'search': 'globe', 'compare': 'git-compare', 'summarize': 'file-text',
        'signal': 'zap', 'recommend': 'lightbulb', 'concern': 'shield-alert',
        'trend': 'activity', 'add': 'plus', 'show': 'eye'
    };

    container.innerHTML = picks.map(s => {
        const lower = s.toLowerCase();
        let icon = isDynamic ? 'sparkles' : 'message-circle';
        if (!isDynamic) {
            for (const [key, val] of Object.entries(iconMap)) {
                if (lower.includes(key)) { icon = val; break; }
            }
        }
        const onclick = isDynamic
            ? `document.getElementById('agent-input').value='${s.replace(/'/g,"\\'")}';sendAgentMessage()`
            : `sendSuggestion(this)`;
        return `<button onclick="${onclick}" class="text-[10px] px-2.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer whitespace-nowrap" style="border:1px solid var(--border-strong); color:var(--text-muted); background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor='var(--accent,#34d399)';this.style.color='var(--accent,#34d399)';this.style.background='rgba(16,185,129,0.06)'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.color='var(--text-muted)';this.style.background='rgba(255,255,255,0.02)'"><i data-lucide="${icon}" class="w-3 h-3 opacity-60"></i>${s}</button>`;
    }).join('');
    lucide.createIcons();
}

function sendSuggestion(btn) {
    const text = btn.textContent;
    // Remove the welcome/suggestions block
    const welcome = document.getElementById('agent-welcome');
    if (welcome) welcome.remove();
    // Set input and send
    const input = document.getElementById('agent-input');
    if (input) input.value = text;
    sendAgentMessage();
}

function _updatePersonaWelcome() {
    const welcome = document.getElementById('agent-welcome');
    if (!welcome) return; // Chat has messages, no welcome to update
    const theme = PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence;
    const w = PERSONA_WELCOMES[currentPersona] || PERSONA_WELCOMES.intelligence;
    const iconDiv = welcome.querySelector('.w-12.h-12 [data-lucide]');
    if (iconDiv) {
        iconDiv.setAttribute('data-lucide', theme.icon);
        iconDiv.style.color = theme.color;
        lucide.createIcons();
    }
    const titleEl = welcome.querySelector('.text-sm.font-semibold');
    if (titleEl) titleEl.textContent = w.title;
    const descEl = welcome.querySelector('.text-\\[11px\\]');
    if (descEl) descEl.textContent = w.desc;
}

function _updateContextBadge(persona) {
    // Color the context button with persona theme
    const ctxBtn = document.querySelector('[onclick*="toggleContextEditor"]');
    const theme = PERSONA_THEMES[persona] || PERSONA_THEMES.intelligence;
    if (ctxBtn) {
        ctxBtn.title = `Edit ${theme.label} context`;
    }
    // Color the file browser button
    const fileBtn = document.querySelector('[onclick*="toggleFileBrowser"]');
    if (fileBtn) {
        fileBtn.title = `${theme.label} files`;
    }
}

function toggleAgentChat() {
    agentOpen = !agentOpen;
    const body = document.getElementById('agent-body');
    const chevron = document.getElementById('agent-chevron');
    if (agentOpen) {
        body.classList.remove('hidden');
        chevron.style.transform = 'rotate(180deg)';
        document.getElementById('agent-input')?.focus();
    } else {
        body.classList.add('hidden');
        chevron.style.transform = '';
    }
}

async function clearAgentChat() {
    agentHistory = [];
    await _saveAgentHistory();
    const msgs = document.getElementById('agent-messages');
    if (!msgs) return;
    msgs.innerHTML = `<div id="agent-welcome" class="flex flex-col items-center py-6 px-2">
        <div class="w-12 h-12 rounded-2xl flex items-center justify-center mb-4" style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.15);">
            <i data-lucide="sparkles" class="w-6 h-6 text-emerald-400"></i>
        </div>
        <p class="text-sm font-semibold mb-1" style="color:var(--text-heading);">Chat cleared</p>
        <p class="text-[11px] mb-5 text-center" style="color:var(--text-muted);">Try one of these to get started:</p>
        <div id="agent-suggestions" class="flex flex-wrap gap-1.5 justify-center max-w-md"></div>
    </div>`;
    renderAgentSuggestions();
    lucide.createIcons();
}

function showAgentPanel(show) {
    const panel = document.getElementById('agent-panel');
    if (!panel) return;
    if (show) {
        panel.classList.remove('hidden');
        // Update status dot based on Ollama availability
        checkAgentStatus();
        loadPersonas();
    } else {
        panel.classList.add('hidden');
    }
}

async function checkAgentStatus() {
    const dot = document.getElementById('agent-status-dot');
    const modelBadge = document.getElementById('agent-model-badge');
    const serperBadge = document.getElementById('agent-serper-badge');
    
    try {
        const r = await fetch('/api/agent-status');
        if (r.ok) {
            const d = await r.json();
            if (dot) {
                dot.style.background = d.available ? 'rgb(16,185,129)' : 'rgb(245,158,11)';
                dot.title = d.available ? `Online (${d.model})` : 'Ollama not reachable';
            }
            if (modelBadge) {
                modelBadge.textContent = d.model || '?';
                modelBadge.style.display = '';
            }
        }
    } catch(e) {
        if (dot) {
            dot.style.background = 'rgb(239,68,68)';
            dot.title = 'Backend unreachable';
        }
    }
    
    // Check Serper availability
    try {
        const cfgR = await fetch('/api/config');
        if (cfgR.ok) {
            const cfg = await cfgR.json();
            const hasSerper = cfg.search?.serper_api_key && cfg.search.serper_api_key !== 'YOUR_SERPER_API_KEY';
            if (serperBadge) {
                if (hasSerper) {
                    serperBadge.innerHTML = `<span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-0.5"></span> Web search enabled`;
                    serperBadge.style.color = 'var(--accent,#34d399)';
                } else {
                    serperBadge.innerHTML = `<span class="inline-block w-1.5 h-1.5 rounded-full mr-0.5" style="background:var(--text-muted);opacity:0.4;"></span> Web search off`;
                    serperBadge.style.color = 'var(--text-muted)';
                }
            }
        }
    } catch(e) {}
}

function appendAgentMessage(role, content) {
    const msgs = document.getElementById('agent-messages');
    if (!msgs) return null;
    
    const wrapper = document.createElement('div');
    const time = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    
    if (role === 'user') {
        wrapper.className = 'flex justify-end mb-3';
        wrapper.innerHTML = `
            <div class="max-w-[82%]">
                <div class="rounded-2xl rounded-br-sm px-4 py-2.5 text-xs leading-relaxed" style="color:var(--text-heading); background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.15);">
                    ${escAgent(content)}
                </div>
                <div class="text-[9px] mt-1 text-right" style="color:var(--text-muted); opacity:0.4;">${time}</div>
            </div>`;
    } else {
        wrapper.className = 'flex gap-3 mb-3 group/msg';
        wrapper.innerHTML = `
            <div class="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5" style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.15);">
                <i data-lucide="bot" class="w-3.5 h-3.5 text-emerald-400"></i>
            </div>
            <div class="flex-1 min-w-0">
                <div class="agent-response text-xs leading-relaxed" style="color:var(--text-body,#cbd5e1);">${content}</div>
                <div class="flex items-center gap-2 mt-1 agent-msg-actions">
                    <span class="text-[9px]" style="color:var(--text-muted); opacity:0.4;">${time}</span>
                    <button onclick="_copyAgentMessage(this)" class="p-0.5 rounded" title="Copy message">
                        <i data-lucide="copy" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    <button onclick="speakMessage(this.closest('.group\\/msg').querySelector('.agent-response').innerText, this)" class="speak-btn p-0.5 rounded" title="Read aloud">
                        <i data-lucide="volume-2" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                </div>
            </div>`;
    }
    
    // Animate in
    wrapper.style.opacity = '0';
    wrapper.style.transform = 'translateY(8px)';
    msgs.appendChild(wrapper);
    lucide.createIcons();
    requestAnimationFrame(() => {
        wrapper.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
        wrapper.style.opacity = '1';
        wrapper.style.transform = 'translateY(0)';
    });
    msgs.scrollTop = msgs.scrollHeight;
    return wrapper;
}

// ── TTS (Text-to-Speech) ──
let _currentTTSAudio = null;

async function speakMessage(text, btn) {
    // Toggle off if already playing
    if (_currentTTSAudio && !_currentTTSAudio.paused) {
        _currentTTSAudio.pause();
        _currentTTSAudio.currentTime = 0;
        _currentTTSAudio = null;
        const icon = btn.querySelector('[data-lucide]');
        if (icon) { icon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
        return;
    }

    const origIcon = btn.querySelector('[data-lucide]');
    if (origIcon) { origIcon.setAttribute('data-lucide', 'loader'); lucide.createIcons(); }
    btn.disabled = true;

    try {
        const resp = await fetch('/api/tts', {
            method: 'POST',
            headers: _agentHeaders(),
            body: JSON.stringify({ text: text.substring(0, 5000) })
        });
        if (!resp.ok) throw new Error('TTS ' + resp.status);

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        _currentTTSAudio = new Audio(url);

        _currentTTSAudio.onended = () => {
            if (origIcon) { origIcon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
            btn.disabled = false;
            URL.revokeObjectURL(url);
            _currentTTSAudio = null;
        };
        _currentTTSAudio.onerror = () => {
            if (origIcon) { origIcon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
            btn.disabled = false;
            URL.revokeObjectURL(url);
            _currentTTSAudio = null;
        };

        if (origIcon) { origIcon.setAttribute('data-lucide', 'square'); lucide.createIcons(); }
        btn.disabled = false;
        _currentTTSAudio.play();
    } catch (e) {
        console.error('TTS error:', e);
        if (origIcon) { origIcon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
        btn.disabled = false;
    }
}

function _copyAgentMessage(btn) {
    const resp = btn.closest('.group\\/msg')?.querySelector('.agent-response');
    if (!resp) return;
    const text = resp.innerText || resp.textContent || '';
    const showCheck = () => {
        const icon = btn.querySelector('[data-lucide]');
        if (icon) { icon.setAttribute('data-lucide', 'check'); lucide.createIcons(); }
        btn.style.opacity = '1';
        setTimeout(() => {
            if (icon) { icon.setAttribute('data-lucide', 'copy'); lucide.createIcons(); }
        }, 1500);
    };
    if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(showCheck).catch(() => {});
    } else {
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select(); document.execCommand('copy');
        document.body.removeChild(ta); showCheck();
    }
}

function escAgent(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Markdown-style formatting for agent responses with smart color highlighting
function formatAgentText(text) {
    // Extract fenced code blocks before escaping HTML
    const codeBlocks = [];
    let processed = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push({ lang, code: code.replace(/\n$/, '') });
        return `\x00CODEBLOCK_${idx}\x00`;
    });

    // ── Detect clickable option blocks (2+ consecutive numbered/lettered lines, optionally preceded by a question) ──
    const optionBlocks = [];
    // Pass 1: with question trigger line
    processed = processed.replace(
        /((?:^.*(?:\?|choose|select|would you like|options|what do you want|what would you)[^\n]*\n))((?:^(?:\d+|[a-zA-Z])[.):\-]\s+.+\n?){2,})/gmi,
        (match, questionLine, optionLines) => {
            const idx = optionBlocks.length;
            const options = [];
            let seq = 1;
            optionLines.replace(/^(?:\d+|[a-zA-Z])[.):\-]\s+(.+)$/gm, (_, text) => {
                options.push({ num: seq++, text: text.trim() });
            });
            optionBlocks.push({ questionLine: questionLine.trim(), options });
            return `\x00OPTBLOCK_${idx}\x00\n`;
        }
    );
    // Pass 2: option blocks without a question trigger (3+ consecutive to avoid false positives)
    processed = processed.replace(
        /^((?:(?:\d+|[a-zA-Z])[.):\-]\s+.+\n?){3,})/gm,
        (match) => {
            const idx = optionBlocks.length;
            const options = [];
            let seq = 1;
            match.replace(/^(?:\d+|[a-zA-Z])[.):\-]\s+(.+)$/gm, (_, text) => {
                options.push({ num: seq++, text: text.trim() });
            });
            if (options.length >= 3) {
                optionBlocks.push({ questionLine: '', options });
                return `\x00OPTBLOCK_${idx}\x00\n`;
            }
            return match;
        }
    );

    let html = processed
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong class="text-slate-100">$1</strong>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-slate-800 rounded text-emerald-300 text-[10px]">$1</code>')
        // Headers (### or ---TITLE)
        .replace(/^#{1,3}\s+(.+)$/gm, '<div class="text-slate-200 font-semibold mt-3 mb-1">$1</div>')
        .replace(/^---(.+?)---?\s*$/gm, '<div class="text-slate-200 font-semibold mt-3 mb-1">$1</div>')
        // Numbered lists: "1. text" or "1) text"
        .replace(/^(\d+)[.)]\s+(.+)$/gm, '<div class="pl-3 mb-1"><span class="text-emerald-400/70 mr-1">$1.</span> $2</div>')
        // Bullet lists: "- text"
        .replace(/^[-•]\s+(.+)$/gm, '<div class="pl-3 mb-1"><span class="text-slate-500 mr-1">·</span> $1</div>')
        // Double newlines → paragraph break (spacing)
        .replace(/\n\n+/g, '<div class="h-2"></div>')
        // Single newlines → line break
        .replace(/\n/g, '<br>');

    // ── Smart color highlighting (post-HTML) ──

    // Percentages: green for positive, red for negative
    html = html.replace(/([+-]?\d+\.?\d*)%/g, (match, num) => {
        const val = parseFloat(num);
        if (val > 0) return `<span class="text-emerald-400 font-semibold">+${Math.abs(val)}%</span>`;
        if (val < 0) return `<span class="text-red-400 font-semibold">${val}%</span>`;
        return `<span class="text-slate-400">0%</span>`;
    });

    // Money amounts: $XXX or KD XXX
    html = html.replace(/\$(\d[\d,]*\.?\d*)/g, '<span class="text-amber-300 font-mono font-semibold">$$$1</span>');
    html = html.replace(/KD\s*(\d[\d,]*\.?\d*)/g, '<span class="text-amber-300 font-mono font-semibold">KD $1</span>');

    // Score values: [8.5] or Score: 8.5
    html = html.replace(/\[(\d+\.?\d*)\]/g, (m, s) => {
        const v = parseFloat(s);
        const c = v >= 9 ? 'text-emerald-400' : v >= 7 ? 'text-blue-400' : v >= 5 ? 'text-amber-400' : 'text-red-400';
        return `<span class="${c} font-semibold">[${s}]</span>`;
    });

    // Ticker symbols — dynamically built from loaded market data + config + well-known list
    const staticTickers = ['NVDA','AMD','ARM','TSLA','AAPL','MSFT','GOOGL','AMZN','META',
        'SLB','HAL','XOM','CVX','BKR','QCOM','INTC','TSM','IBM','ORCL','ASML',
        'BTC','ETH','XRP','BNB','SOL','ADA','DOGE','IONQ','ENPH','BATS'];
    // Add all symbols from loaded market data (handles any user-added tickers)
    const dynamicTickers = Object.keys(typeof marketData !== 'undefined' ? marketData : {}).map(s => s.replace('-USD','').replace('=F',''));
    // Add tickers from config (they may not be loaded in marketData yet)
    const cfgTickers = (typeof configData !== 'undefined' && configData?.market?.tickers)
        ? configData.market.tickers.map(t => (t.symbol||'').replace('-USD','').replace('=F',''))
        : [];
    const allTickers = [...new Set([...staticTickers, ...dynamicTickers, ...cfgTickers])].filter(t => t.length >= 2);
    const tickerPattern = new RegExp(
        `(?<=\\s|^|>|\\()(?:${allTickers.join('|')}|[A-Z]{2,5}-USD|[A-Z]{1,4}=F)(?=\\s|$|<|[.,;:!?)])`, 'g'
    );
    html = html.replace(tickerPattern, '<span class="text-cyan-400 font-mono font-semibold cursor-pointer hover:underline" onclick="navigateToTicker(\'$&\')" title="View $& chart">$&</span>');

    // Company names — subtle highlight
    const companies = ['NVIDIA','Nvidia','Qualcomm','Samsung','Apple','Microsoft','Google',
        'Schlumberger','Halliburton','Baker Hughes','Equate','KNPC','KOC','KIPIC',
        'Warba Bank','Boubyan','NBK','KFH','Kuwait Finance House',
        'Arm Holdings','Exxon Mobil','ExxonMobil','Chevron','TotalEnergies'];
    companies.forEach(name => {
        // Only replace if not already inside an HTML tag
        const re = new RegExp(`(?<!<[^>]*)\\b(${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})\\b(?![^<]*>)`, 'g');
        html = html.replace(re, '<span class="text-slate-100">$1</span>');
    });

    // Re-inject fenced code blocks
    codeBlocks.forEach((block, i) => {
        const escaped = block.code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const langBadge = block.lang ? `<span class="absolute top-1.5 right-2 text-[9px] font-mono" style="color:var(--text-muted);opacity:0.5;">${block.lang}</span>` : '';
        html = html.replace(`\x00CODEBLOCK_${i}\x00`,
            `<div class="relative my-2 rounded-lg overflow-hidden" style="background:rgba(0,0,0,0.3);border:1px solid var(--border-strong);">${langBadge}<pre class="p-3 overflow-x-auto text-[11px] leading-relaxed font-mono" style="color:#e2e8f0;"><code>${escaped}</code></pre></div>`);
    });

    // Re-inject clickable option blocks
    optionBlocks.forEach((block, i) => {
        const theme = PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence;
        const qHtml = block.questionLine
            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong class="text-slate-100">$1</strong>');
        const buttonsHtml = block.options.map((opt, j) => {
            const safeText = opt.text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
            return `<button onclick="sendAgentOption(this, '${safeText}')" class="agent-option-btn flex items-center gap-2 w-full text-left px-3 py-2 rounded-lg text-[11px] transition-all" style="background:rgba(255,255,255,0.03);border:1px solid var(--border-strong);color:var(--text-secondary);animation:fadeIn ${0.15 + j * 0.08}s ease;" onmouseenter="this.style.borderColor='${theme.color}';this.style.background='${theme.bg}';this.style.color='${theme.color}'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.background='rgba(255,255,255,0.03)';this.style.color='var(--text-secondary)'">
                <span class="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold" style="background:${theme.bg};color:${theme.color};border:1px solid ${theme.color}40;">${opt.num}</span>
                <span>${opt.text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</span>
            </button>`;
        }).join('');
        const qBlock = qHtml.trim() ? `<div class="mb-1">${qHtml}</div>` : '';
        html = html.replace(`\x00OPTBLOCK_${i}\x00`,
            `${qBlock}<div class="agent-options flex flex-col gap-1.5 my-2">${buttonsHtml}</div>`);
    });

    return html;
}

function sendAgentOption(btn, text) {
    // Disable all option buttons in the same group
    const group = btn.closest('.agent-options');
    if (group) {
        group.querySelectorAll('.agent-option-btn').forEach(b => {
            b.style.pointerEvents = 'none';
            b.style.opacity = '0.4';
        });
        // Highlight selected
        btn.style.opacity = '1';
        btn.style.borderColor = (PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence).color;
        btn.style.background = (PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence).bg;
    }
    const input = document.getElementById('agent-input');
    if (input) {
        input.value = text;
        sendAgentMessage();
    }
}
window.sendAgentOption = sendAgentOption;

// Show more/less for long agent responses (> 500 chars raw text)
var _showMoreCounter = 0;
function wrapWithShowMore(rawText, formattedHtml) {
    if (rawText.length <= 500) return formattedHtml;
    var id = 'showmore-' + (++_showMoreCounter);
    // Find a cut point near 500 chars in the raw text — try to cut at a paragraph or sentence boundary
    var cutAt = 500;
    var nlPos = rawText.indexOf('\n\n', 400);
    if (nlPos > 0 && nlPos < 600) cutAt = nlPos;
    else {
        var dotPos = rawText.indexOf('. ', 400);
        if (dotPos > 0 && dotPos < 600) cutAt = dotPos + 1;
    }
    var shortText = rawText.slice(0, cutAt).trim();
    var shortHtml = formatAgentText(shortText);
    return '<div id="' + id + '">'
        + '<div class="showmore-short">' + shortHtml
        + '<div class="mt-2"><button onclick="document.querySelector(\'#' + id + ' .showmore-short\').style.display=\'none\';document.querySelector(\'#' + id + ' .showmore-full\').style.display=\'block\';" '
        + 'class="text-[9px] px-2 py-0.5 rounded transition-colors cursor-pointer" '
        + 'style="color:#34d399;border:1px solid rgba(52,211,153,0.3);background:rgba(52,211,153,0.05);"'
        + '>Show more &#x25BC;</button></div></div>'
        + '<div class="showmore-full" style="display:none;">' + formattedHtml
        + '<div class="mt-2"><button onclick="document.querySelector(\'#' + id + ' .showmore-full\').style.display=\'none\';document.querySelector(\'#' + id + ' .showmore-short\').style.display=\'block\';" '
        + 'class="text-[9px] px-2 py-0.5 rounded transition-colors cursor-pointer" '
        + 'style="color:#34d399;border:1px solid rgba(52,211,153,0.3);background:rgba(52,211,153,0.05);"'
        + '>Show less &#x25B2;</button></div></div>'
        + '</div>';
}

async function sendAgentMessage() {
    if (agentStreaming) return;
    const input = document.getElementById('agent-input');
    const sendBtn = document.getElementById('agent-send-btn');
    const msg = (input?.value || '').trim();
    if (!msg) return;
    
    input.value = '';
    input.style.height = 'auto';

    // Remove welcome suggestions on first message
    const welcome = document.getElementById('agent-welcome');
    if (welcome) welcome.remove();
    
    // ── Ticker command interception ──
    const tickerCmd = _parseTickerCommand(msg);
    if (tickerCmd) {
        agentHistory.push({ role: 'user', content: msg });
        appendAgentMessage('user', msg);
        const result = await _executeTickerCommand(tickerCmd);
        agentHistory.push({ role: 'assistant', content: result.plain || result });
        // Use rich HTML if available, otherwise format markdown
        appendAgentMessage('assistant', result.html || formatAgentText(result));
        _saveAgentHistory();
        return;
    }
    
    agentHistory.push({ role: 'user', content: msg });
    appendAgentMessage('user', msg);
    
    // Show typing indicator with bouncing dots
    const typingEl = appendAgentMessage('assistant', `<div class="flex items-center gap-2.5"><div class="agent-thinking-dots flex gap-1"><span></span><span></span><span></span></div><span class="text-[10px]" style="color:var(--text-muted);">Thinking...</span></div>`);
    
    agentStreaming = true;
    _agentAbortController = new AbortController();
    sendBtn.disabled = false;
    input.disabled = true;
    sendBtn.style.opacity = '1';
    sendBtn.innerHTML = '<i data-lucide="square" class="w-4 h-4"></i>';
    sendBtn.title = 'Stop generating';
    sendBtn.onclick = function() { cancelAgentStream(); };
    lucide.createIcons();

    try {
        const response = await fetch('/api/agent-chat', {
            method: 'POST',
            signal: _agentAbortController.signal,
            headers: {
                'Content-Type': 'application/json',
                'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
                'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
            },
            body: JSON.stringify({
                message: msg,
                history: agentHistory.slice(-20),
                mode: agentMode,
                persona: currentPersona,
                ...(selectedPersonas.length > 1 ? { personas: selectedPersonas } : {})
            })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${response.status}`);
        }
        
        // Stream the response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let dynamicSuggestions = [];
        
        // Replace typing indicator with empty response div
        const respDiv = typingEl.querySelector('.agent-response');
        if (respDiv) respDiv.innerHTML = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            // Parse SSE lines
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const payload = JSON.parse(line.slice(6));
                        if (payload.token) {
                            fullResponse += payload.token;
                            if (respDiv) {
                                respDiv.innerHTML = formatAgentText(fullResponse);
                                const msgs = document.getElementById('agent-messages');
                                if (msgs) msgs.scrollTop = msgs.scrollHeight;
                            }
                        }
                        if (payload.suggestions && Array.isArray(payload.suggestions)) {
                            dynamicSuggestions = payload.suggestions;
                            _personaSuggestions[currentPersona] = payload.suggestions;
                            try { localStorage.setItem('stratos_persona_suggestions', JSON.stringify(_personaSuggestions)); } catch(e) {}
                        }
                        if (payload.status) {
                            // Tool usage indicator — animated status bar
                            if (respDiv) {
                                const isSearch = payload.status.includes('web_search');
                                const icon = isSearch ? '🔍' : '⚙️';
                                const label = payload.status.replace(/^[🔍⚙️]\s*/, '');
                                respDiv.innerHTML = `<div class="flex items-center gap-2.5 py-1"><div class="agent-thinking-dots flex gap-1"><span></span><span></span><span></span></div><span class="text-[10px] font-mono" style="color:var(--accent,#34d399);">${icon} ${escAgent(label)}</span></div>`;
                                const msgs = document.getElementById('agent-messages');
                                if (msgs) msgs.scrollTop = msgs.scrollHeight;
                            }
                        }
                        if (payload.error) {
                            fullResponse = '⚠ ' + payload.error;
                            if (respDiv) respDiv.innerHTML = `<span class="text-amber-400">${escAgent(fullResponse)}</span>`;
                        }
                    } catch(e) {}
                }
            }
        }

        // Clean non-English text (Qwen model sometimes leaks Chinese/Arabic characters)
        fullResponse = fullResponse.replace(/[\u4e00-\u9fff\u3400-\u4dbf\u{20000}-\u{2a6df}]+/gu, '').trim();
        // Remove orphaned punctuation from cleanup — but PRESERVE newlines
        fullResponse = fullResponse.replace(/[^\S\n]{2,}/g, ' ').replace(/\.\s*\./g, '.').trim();
        // Clean up blank lines (3+ newlines → 2)
        fullResponse = fullResponse.replace(/\n{3,}/g, '\n\n');

        // Final render with cleaned text + show more/less for long responses
        const finalDiv = typingEl?.querySelector('.agent-response');
        if (finalDiv) {
            finalDiv.innerHTML = wrapWithShowMore(fullResponse, formatAgentText(fullResponse));
            // Add suggestion chips — prefer LLM-generated, fallback to rule-based
            const chipSuggestions = dynamicSuggestions.length > 0
                ? dynamicSuggestions.map(s => ({ label: s, icon: 'sparkles', action: s, tip: s }))
                : _generateResponseChips(fullResponse, msg);
            if (chipSuggestions.length) {
                const chipHtml = chipSuggestions.map(c => {
                    const isDynamic = dynamicSuggestions.length > 0;
                    const onclick = isDynamic
                        ? `document.getElementById('agent-input').value='${escAgent(c.action).replace(/'/g,"\\'")}';sendAgentMessage()`
                        : escAgent(c.action);
                    return `<button onclick="${onclick}" class="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all hover:scale-105" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);color:var(--accent,#34d399);" title="${escAgent(c.tip || '')}">
                        <i data-lucide="${c.icon}" class="w-3 h-3"></i> ${escAgent(c.label)}
                    </button>`;
                }).join('');
                finalDiv.insertAdjacentHTML('afterend',
                    `<div class="flex flex-wrap gap-1.5 mt-2 agent-chips" style="animation:fadeIn 0.3s ease">${chipHtml}</div>`);
                lucide.createIcons();
            }
        }

        agentHistory.push({ role: 'assistant', content: fullResponse });
        
    } catch(e) {
        const respDiv = typingEl?.querySelector('.agent-response');
        const raw = e.message || 'Failed to reach agent';
        let friendly;
        if (raw.includes('Failed to fetch') || raw.includes('NetworkError') || raw.includes('network'))
            friendly = 'Network error — check your connection and try again.';
        else if (raw.includes('timeout') || raw.includes('Timeout') || raw.includes('504'))
            friendly = 'The AI model is taking too long to respond. Try a shorter question or check if Ollama is running.';
        else if (raw.includes('503') || raw.includes('model') || raw.includes('ollama'))
            friendly = 'The AI model appears to be offline. Make sure Ollama is running.';
        else if (raw.includes('429') || raw.includes('Too Many'))
            friendly = 'Too many requests — please wait a moment and try again.';
        else
            friendly = raw;
        if (respDiv) {
            respDiv.innerHTML = `<div class="flex items-start gap-2 p-2 rounded-lg" style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);"><i data-lucide="alert-triangle" class="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5"></i><span class="text-amber-300 text-xs">${escAgent(friendly)}</span></div>`;
            lucide.createIcons();
        }
        agentHistory.push({ role: 'assistant', content: friendly });
    } finally {
        _saveAgentHistory();
        agentStreaming = false;
        _agentAbortController = null;
        sendBtn.disabled = false;
        input.disabled = false;
        sendBtn.style.opacity = '1';
        sendBtn.innerHTML = '<i data-lucide="arrow-up" class="w-4 h-4"></i>';
        sendBtn.title = 'Send message';
        sendBtn.onclick = function() { sendAgentMessage(); };
        lucide.createIcons();
        input.focus();
    }
}

function cancelAgentStream() {
    if (_agentAbortController) {
        _agentAbortController.abort();
        _agentAbortController = null;
    }
}
window.cancelAgentStream = cancelAgentStream;

// ═══════════════════════════════════════════════════════════
// F2: RESPONSE SUGGESTION CHIPS
// ═══════════════════════════════════════════════════════════

function _generateResponseChips(response, userMsg) {
    const chips = [];
    const rLower = response.toLowerCase();
    const mLower = userMsg.toLowerCase();

    // Detect ticker mentions (3-5 uppercase letters)
    const tickerMatches = response.match(/\b([A-Z]{2,5})\b/g);
    if (tickerMatches) {
        const currentTickers = (typeof configData !== 'undefined' && configData?.market?.tickers)
            ? configData.market.tickers.map(t => (t.symbol || t).toUpperCase()) : [];
        const seen = new Set();
        for (const t of tickerMatches) {
            if (seen.has(t) || currentTickers.includes(t)) continue;
            // Skip common words that look like tickers
            if (['THE','AND','FOR','ARE','BUT','NOT','YOU','ALL','CAN','HAS','HER','WAS','ONE','OUR','OUT','DAY','HAD','HIS','HOW','ITS','MAY','NEW','NOW','OLD','SEE','WAY','WHO','DID','GET','LET','SAY','SHE','TOO','USE','KEY','GDP','CEO','IPO','ETF','USD','EUR','GBP','RSS','API'].includes(t)) continue;
            seen.add(t);
            if (chips.length < 3) {
                chips.push({
                    label: `Track ${t}`,
                    icon: 'plus-circle',
                    action: `document.getElementById('agent-input').value='Add ${t} to watchlist';sendAgentMessage()`,
                    tip: `Add ${t} to your watchlist`
                });
            }
        }
    }

    // Suggest web search follow-up
    if (rLower.includes('search') || rLower.includes('look up') || rLower.includes('find more')) {
        chips.push({
            label: 'Search more',
            icon: 'search',
            action: `document.getElementById('agent-input').value='Search for more details on this topic';sendAgentMessage()`,
            tip: 'Search the web for more information'
        });
    }

    // If response mentions news/market topics, suggest drilling deeper
    if (rLower.includes('market') || rLower.includes('price') || rLower.includes('stock')) {
        if (!chips.some(c => c.label.includes('Track'))) {
            chips.push({
                label: 'Market analysis',
                icon: 'bar-chart-2',
                action: `document.getElementById('agent-input').value='Give me a detailed market analysis';sendAgentMessage()`,
                tip: 'Get deeper market analysis'
            });
        }
    }

    // Suggest summarize if response is long
    if (response.length > 800) {
        chips.push({
            label: 'Summarize',
            icon: 'align-left',
            action: `document.getElementById('agent-input').value='Summarize the above in 3 bullet points';sendAgentMessage()`,
            tip: 'Get a shorter summary'
        });
    }

    return chips.slice(0, 4); // Max 4 chips
}

function _applyAgentChip(action) {
    const input = document.getElementById('agent-input');
    if (input) {
        input.value = action;
        sendAgentMessage();
    }
}

// ═══════════════════════════════════════════════════════════
// AGENT CHAT EXPORT / IMPORT
// ═══════════════════════════════════════════════════════════

function exportAgentChat() {
    if (agentHistory.length === 0) {
        showToast('No chat to export', 'warning');
        return;
    }
    const exportData = {
        version: 1,
        exported_at: new Date().toISOString(),
        profile: typeof configData !== 'undefined' ? configData?.profile?.role || '' : '',
        messages: agentHistory
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const date = new Date().toISOString().slice(0,10);
    a.href = url;
    a.download = `strat-agent-chat_${date}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Chat exported', 'success');
}

function importAgentChat() {
    document.getElementById('agent-import-file')?.click();
}

function handleAgentImport(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const raw = e.target.result?.trim();
        if (!raw) { showToast('File is empty', 'error'); return; }

        // Try parsing as chat export JSON first
        let chatLoaded = false;
        try {
            const data = JSON.parse(raw);
            if (data.messages && Array.isArray(data.messages)) {
                const valid = data.messages.every(m =>
                    m && typeof m === 'object' &&
                    (m.role === 'user' || m.role === 'assistant') &&
                    typeof m.content === 'string'
                );
                if (valid && data.messages.length > 0) {
                    // ── Load as chat history ──
                    agentHistory = data.messages;
                    const msgs = document.getElementById('agent-messages');
                    if (!msgs) return;
                    msgs.innerHTML = '';
                    for (const msg of agentHistory) {
                        if (msg.role === 'assistant') {
                            appendAgentMessage('assistant', formatAgentText(msg.content));
                        } else {
                            appendAgentMessage('user', msg.content);
                        }
                    }
                    _openAgentPanel();
                    msgs.scrollTop = msgs.scrollHeight;
                    const info = data.exported_at ? ` from ${new Date(data.exported_at).toLocaleDateString()}` : '';
                    showToast(`Loaded ${data.messages.length} messages${info}`, 'success');
                    chatLoaded = true;
                }
            }
        } catch (_) { /* not valid JSON or not chat format — fall through to context */ }

        if (!chatLoaded) {
            // ── Treat as context file ──
            const text = raw.slice(0, 5000); // Cap at 5k chars
            const fileName = file.name || 'file';

            // Inject into agent history as a context message
            const contextMsg = `[Imported context from "${fileName}"]\n\n${text}`;
            agentHistory.push({ role: 'user', content: contextMsg });
            _saveAgentHistory();

            // Show a compact context-loaded block in the chat
            const msgs = document.getElementById('agent-messages');
            if (msgs) {
                const div = document.createElement('div');
                div.className = 'flex justify-end mb-2';
                const preview = text.length > 200 ? text.slice(0, 200) + '…' : text;
                div.innerHTML = `
                    <div class="max-w-[80%] rounded-lg px-3 py-2 text-xs" style="background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.2);">
                        <div class="flex items-center gap-1.5 mb-1">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-emerald-400"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                            <span class="text-emerald-400 font-semibold">Context loaded: ${fileName}</span>
                        </div>
                        <div class="text-slate-400 whitespace-pre-wrap">${preview}</div>
                    </div>`;
                msgs.appendChild(div);
                _openAgentPanel();
                msgs.scrollTop = msgs.scrollHeight;
            }
            showToast(`Context loaded from ${fileName} — the agent can now reference it`, 'success');
        }
    };
    reader.readAsText(file);
    event.target.value = '';
}

function _openAgentPanel() {
    const body = document.getElementById('agent-body');
    const chevron = document.getElementById('agent-chevron');
    if (body) body.classList.remove('hidden');
    if (chevron) chevron.style.transform = 'rotate(180deg)';
    agentOpen = true;
}

var _agentFullscreen = false;
var _agentFsSidebarOpen = true;

function _buildFsSidebar() {
    const theme = PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence;
    const modelName = document.getElementById('agent-model-badge')?.textContent || 'qwen3.5:9b';
    // Build conversation list
    const convItems = _agentConvList.map(c => {
        const active = c.id === _agentActiveConvId;
        const title = (c.title || 'New Chat').length > 28 ? (c.title || 'New Chat').slice(0, 26) + '…' : (c.title || 'New Chat');
        return `<div class="group flex items-center gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all" style="background:${active ? 'rgba(255,255,255,0.06)' : 'transparent'};" onclick="_switchConversation(${c.id})" ondblclick="event.stopPropagation();_renameConversation(${c.id})" title="Double-click to rename" onmouseenter="if(!${active})this.style.background='rgba(255,255,255,0.03)'" onmouseleave="if(!${active})this.style.background='${active ? 'rgba(255,255,255,0.06)' : 'transparent'}'">
            <i data-lucide="message-square" class="w-4 h-4 flex-shrink-0" style="color:${active ? theme.color : 'var(--text-muted)'};"></i>
            <span class="flex-1 text-[13px] truncate" style="color:${active ? 'var(--text-heading)' : 'var(--text-muted)'};">${escAgent(title)}</span>
            <div class="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                <span onclick="event.stopPropagation();_renameConversation(${c.id})" class="p-1 rounded transition-all" style="color:var(--text-muted);" title="Rename" onmouseenter="this.style.color='${theme.color}'" onmouseleave="this.style.color='var(--text-muted)'"><i data-lucide="pencil" class="w-3 h-3"></i></span>
                ${_agentConvList.length > 1 ? `<span onclick="event.stopPropagation();_deleteConversation(${c.id})" class="p-1 rounded transition-all" style="color:var(--text-muted);" title="Delete" onmouseenter="this.style.color='#f87171'" onmouseleave="this.style.color='var(--text-muted)'"><i data-lucide="trash-2" class="w-3 h-3"></i></span>` : ''}
            </div>
        </div>`;
    }).join('');
    // Build persona options
    const personaItems = Object.entries(PERSONA_THEMES).map(([key, t]) => {
        const active = key === currentPersona;
        return `<button onclick="switchPersona('${key}')" class="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-left transition-all" style="background:${active ? t.bg : 'transparent'};color:${active ? t.color : 'var(--text-muted)'};" onmouseenter="if(!${active})this.style.background='rgba(255,255,255,0.03)'" onmouseleave="if(!${active})this.style.background='${active ? t.bg : 'transparent'}'">
            <i data-lucide="${t.icon}" class="w-4 h-4"></i>
            <span class="text-[13px] font-medium">${t.label}</span>
        </button>`;
    }).join('');

    return `
    <div class="agent-fs-sidebar" style="width:280px;min-width:280px;height:100%;display:flex;flex-direction:column;border-right:1px solid rgba(255,255,255,0.05);">
        <!-- New Chat button -->
        <div class="px-3 pt-4 pb-2">
            <button onclick="newAgentChat()" class="w-full flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-[13px] font-medium transition-all" style="border:1px solid var(--border-strong);color:var(--text-heading);background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor='${theme.color}';this.style.background='${theme.bg}'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.background='rgba(255,255,255,0.02)'">
                <i data-lucide="plus" class="w-4.5 h-4.5"></i> New Chat
            </button>
        </div>
        <!-- Persona selector -->
        <div class="px-3 pb-2">
            <div class="text-[10px] font-bold uppercase tracking-wider mb-2 px-1" style="color:var(--text-muted);">Persona</div>
            <div class="space-y-0.5">${personaItems}</div>
        </div>
        <div class="mx-3 mb-2" style="height:1px;background:var(--border-strong);"></div>
        <!-- Conversation list -->
        <div class="flex-1 overflow-y-auto px-2" style="scrollbar-width:thin;">
            <div class="text-[10px] font-bold uppercase tracking-wider mb-2 px-1" style="color:var(--text-muted);">Chats</div>
            <div class="space-y-0.5">${convItems}</div>
        </div>
        <div class="mx-3 my-1" style="height:1px;background:var(--border-strong);"></div>
        <!-- Bottom actions -->
        <div class="px-3 pb-4 pt-2 space-y-1.5">
            <div class="flex items-center gap-1">
                <button onclick="importAgentChat()" class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--text-muted);border:1px solid transparent;" title="Import chat" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="upload" class="w-4 h-4"></i> Import
                </button>
                <button onclick="exportAgentChat()" class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--text-muted);border:1px solid transparent;" title="Export chat" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="download" class="w-4 h-4"></i> Export
                </button>
            </div>
            <div class="flex items-center gap-1">
                <button onclick="toggleContextEditor()" class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--text-muted);" title="Edit context" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="file-cog" class="w-4 h-4"></i> Context
                </button>
                <button onclick="toggleFileBrowser()" class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--text-muted);" title="Browse files" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="folder-open" class="w-4 h-4"></i> Files
                </button>
            </div>
            <div class="flex items-center gap-1">
                <button onclick="clearAgentChat()" class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--text-muted);" title="Clear current chat" onmouseenter="this.style.color='#f87171';this.style.background='rgba(239,68,68,0.06)'" onmouseleave="this.style.color='var(--text-muted)';this.style.background='transparent'">
                    <i data-lucide="trash-2" class="w-4 h-4"></i> Clear Chat
                </button>
                <button onclick="toggleAgentFullscreen()" class="flex-1 flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--text-muted);" title="Exit fullscreen" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="minimize-2" class="w-4 h-4"></i> Collapse
                </button>
            </div>
            <button onclick="_toggleFsCustomizer()" class="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[12px] transition-all" style="color:var(--accent);" title="Customize fullscreen appearance" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                <i data-lucide="palette" class="w-4 h-4"></i> Customize
            </button>
            <!-- Model badge -->
            <div class="flex items-center gap-2 px-3 py-2 mt-1">
                <i data-lucide="cpu" class="w-3.5 h-3.5" style="color:${theme.color};"></i>
                <span class="text-[11px] font-mono" style="color:var(--text-muted);">${escAgent(modelName)}</span>
            </div>
        </div>
    </div>`;
}

// ── Fullscreen Customizer ──
var _fsCustomizerOpen = false;
var _fsCustomDefaults = {
    chatOpacity: 0.7, chatBlur: 16, sidebarOpacity: 0.75, sidebarBlur: 20,
    fontSize: 15, lineHeight: 1.65, bubblePadding: 14, bubbleRadius: 16,
    chatWidth: 800, inputHeight: 52, sendSize: 44, uiScale: 1, gradBar: true
};

function _loadFsCustom() {
    try { return Object.assign({}, _fsCustomDefaults, JSON.parse(localStorage.getItem('stratos-fs-custom') || '{}')); }
    catch { return Object.assign({}, _fsCustomDefaults); }
}
function _saveFsCustom(c) { localStorage.setItem('stratos-fs-custom', JSON.stringify(c)); }

function _applyFsCustom(c) {
    const panel = document.querySelector('.agent-fullscreen');
    if (!panel) return;
    const inner = panel.querySelector('[data-agent-inner]');
    const sidebar = panel.querySelector('.agent-fs-sidebar');
    if (inner) {
        inner.style.background = `rgba(8,8,26,${c.chatOpacity})`;
        inner.style.backdropFilter = `blur(${c.chatBlur}px)`;
        inner.style.webkitBackdropFilter = `blur(${c.chatBlur}px)`;
        inner.style.maxWidth = c.chatWidth + 'px';
    }
    if (sidebar) {
        sidebar.style.setProperty('background', `rgba(8,8,26,${c.sidebarOpacity})`, 'important');
        sidebar.style.setProperty('backdrop-filter', `blur(${c.sidebarBlur}px)`, 'important');
        sidebar.style.setProperty('-webkit-backdrop-filter', `blur(${c.sidebarBlur}px)`, 'important');
    }
    // Use CSS custom properties so they apply to all current + future elements
    panel.style.setProperty('--fs-font-size', c.fontSize + 'px');
    panel.style.setProperty('--fs-line-height', String(c.lineHeight));
    panel.style.setProperty('--fs-bubble-padding', c.bubblePadding + 'px ' + Math.round(c.bubblePadding * 1.3) + 'px');
    panel.style.setProperty('--fs-bubble-radius', c.bubbleRadius + 'px');
    panel.style.setProperty('--fs-input-height', c.inputHeight + 'px');
    panel.style.setProperty('--fs-send-size', c.sendSize + 'px');
    panel.style.setProperty('--fs-ui-scale', String(c.uiScale));
    panel.style.setProperty('--fs-grad-display', c.gradBar ? 'block' : 'none');
}

function _toggleFsCustomizer() {
    _fsCustomizerOpen = !_fsCustomizerOpen;
    let p = document.getElementById('agent-fs-customizer');
    if (_fsCustomizerOpen) {
        if (p) p.remove();
        const c = _loadFsCustom();
        const themes = ['midnight','coffee','rose','noir','aurora','cosmos','sakura','nebula'];
        const curTheme = document.documentElement.getAttribute('data-theme') || 'midnight';
        const themeColors = {midnight:'#34d399',coffee:'#fbbf24',rose:'#fb7185',noir:'#a78bfa',aurora:'#34d399',cosmos:'#e8b931',sakura:'#f0a0b8',nebula:'#38bdf8'};
        const themeBtns = themes.map(t =>
            `<button onclick="_fsCustSetTheme('${t}')" class="px-2 py-1 rounded text-[10px] font-bold transition-all" style="color:${t===curTheme?themeColors[t]:'var(--text-muted)'};border:1px solid ${t===curTheme?themeColors[t]+'60':'var(--border-strong)'};background:${t===curTheme?themeColors[t]+'15':'transparent'};" onmouseenter="this.style.background='rgba(255,255,255,0.05)'" onmouseleave="this.style.background='${t===curTheme?themeColors[t]+'15':'transparent'}'">${t.charAt(0).toUpperCase()+t.slice(1)}</button>`
        ).join('');

        const html = `
        <div id="agent-fs-customizer" class="agent-fs-customizer" style="position:absolute;top:0;left:280px;width:300px;height:100%;background:rgba(8,8,26,0.92);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border-left:1px solid rgba(255,255,255,0.06);z-index:20;display:flex;flex-direction:column;animation:slideInLeft 0.2s ease;">
            <div style="padding:16px 16px 8px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.06);flex-shrink:0;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <i data-lucide="palette" class="w-4 h-4" style="color:var(--accent)"></i>
                    <span class="text-[13px] font-bold" style="color:var(--text-heading);">Customize</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;">
                    <button onclick="_fsCustReset()" class="strat-tip text-[10px] px-2 py-1 rounded transition-all" style="color:var(--text-muted);border:1px solid var(--border-strong);" data-tip="Restore all settings to their default values" onmouseenter="this.style.color='var(--accent)'" onmouseleave="this.style.color='var(--text-muted)'">Reset</button>
                    <button onclick="_toggleFsCustomizer()" class="p-1 rounded transition-all strat-tip" style="color:var(--text-muted);" data-tip="Close customizer" onmouseenter="this.style.color='var(--text-heading)'" onmouseleave="this.style.color='var(--text-muted)'">&times;</button>
                </div>
            </div>
            <div style="padding:12px 16px;flex:1;overflow-y:auto;" class="fs-cust-body">
                <!-- Theme -->
                <div class="fs-cust-group">
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Switch the base color theme for the entire app">Theme</div>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">${themeBtns}</div>
                </div>
                <!-- Chat Area -->
                <div class="fs-cust-group" style="margin-top:14px;">
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Control the chat panel background transparency and blur">Chat Area</div>
                    <label class="fs-cust-label strat-tip" data-tip="How transparent the chat background is. Lower values let more of the theme show through">Opacity</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-chat-opacity" min="0" max="1" step="0.05" value="${c.chatOpacity}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-chat-opacity-val">${Math.round(c.chatOpacity*100)}%</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Frosted glass blur intensity behind the chat panel">Glass Blur</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-chat-blur" min="0" max="40" step="1" value="${c.chatBlur}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-chat-blur-val">${c.chatBlur}px</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Maximum width of the conversation area. Wider = more horizontal space for messages">Max Width</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-chat-width" min="600" max="1400" step="20" value="${c.chatWidth}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-chat-width-val">${c.chatWidth}px</span>
                    </div>
                </div>
                <!-- Sidebar -->
                <div class="fs-cust-group" style="margin-top:14px;">
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Control the sidebar panel transparency and blur">Sidebar</div>
                    <label class="fs-cust-label strat-tip" data-tip="How transparent the sidebar background is">Opacity</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-sidebar-opacity" min="0" max="1" step="0.05" value="${c.sidebarOpacity}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-sidebar-opacity-val">${Math.round(c.sidebarOpacity*100)}%</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Frosted glass blur intensity behind the sidebar">Glass Blur</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-sidebar-blur" min="0" max="40" step="1" value="${c.sidebarBlur}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-sidebar-blur-val">${c.sidebarBlur}px</span>
                    </div>
                </div>
                <!-- Typography -->
                <div class="fs-cust-group" style="margin-top:14px;">
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Adjust text sizing for messages, input, and all UI elements">Typography</div>
                    <label class="fs-cust-label strat-tip" data-tip="Scale everything — sidebar, messages, buttons, input, all text and UI. This zooms the entire fullscreen view">UI Scale</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-ui-scale" min="0.85" max="1.3" step="0.01" value="${c.uiScale}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-ui-scale-val">${Math.round(c.uiScale*100)}%</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Base font size for message bubbles, input field, welcome text, and option buttons">Message Font</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-font-size" min="12" max="24" step="1" value="${c.fontSize}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-font-size-val">${c.fontSize}px</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Space between lines of text in messages. Higher values make text more spread out and easier to read">Line Height</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-line-height" min="1.2" max="2.2" step="0.05" value="${c.lineHeight}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-line-height-val">${c.lineHeight.toFixed(2)}</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Height of the text input box at the bottom of the chat">Input Height</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-input-height" min="36" max="80" step="2" value="${c.inputHeight}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-input-height-val">${c.inputHeight}px</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Size of the send button next to the input field">Send Button</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-send-size" min="30" max="60" step="2" value="${c.sendSize}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-send-size-val">${c.sendSize}px</span>
                    </div>
                </div>
                <!-- Bubbles -->
                <div class="fs-cust-group" style="margin-top:14px;">
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Adjust the shape and spacing of chat message bubbles">Bubbles</div>
                    <label class="fs-cust-label strat-tip" data-tip="How rounded the corners of message bubbles are. 0 = sharp squares, 28 = pill-shaped">Corner Radius</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-bubble-radius" min="0" max="28" step="1" value="${c.bubbleRadius}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-bubble-radius-val">${c.bubbleRadius}px</span>
                    </div>
                    <label class="fs-cust-label strat-tip" data-tip="Internal spacing inside each message bubble. More padding = more breathing room around text">Padding</label>
                    <div class="fs-cust-row">
                        <input type="range" class="fs-cust-slider" id="fsc-bubble-padding" min="6" max="28" step="1" value="${c.bubblePadding}" oninput="_fsCustUpdate()">
                        <span class="fs-cust-val" id="fsc-bubble-padding-val">${c.bubblePadding}px</span>
                    </div>
                </div>
                <!-- Effects -->
                <div class="fs-cust-group" style="margin-top:14px;">
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Visual effects and decorative elements">Effects</div>
                    <label class="fs-cust-label strat-tip" style="display:flex;align-items:center;gap:8px;cursor:pointer;" data-tip="Animated accent-colored gradient line at the top of the fullscreen view">
                        <input type="checkbox" id="fsc-grad-bar" ${c.gradBar?'checked':''} onchange="_fsCustUpdate()" style="accent-color:var(--accent);">
                        Gradient accent bar
                    </label>
                </div>
            </div>
        </div>`;
        const wrapper = document.querySelector('.agent-fs-wrapper');
        if (wrapper) {
            wrapper.insertAdjacentHTML('afterbegin', html);
            lucide.createIcons();
            _initFsCustTooltips();
        }
    } else {
        if (p) p.remove();
        const tt = document.getElementById('fsc-tooltip');
        if (tt) tt.remove();
    }
}

// Floating tooltip for customizer (avoids overflow clipping)
function _initFsCustTooltips() {
    let tt = document.getElementById('fsc-tooltip');
    if (!tt) {
        tt = document.createElement('div');
        tt.id = 'fsc-tooltip';
        tt.style.cssText = 'position:fixed;z-index:999999;pointer-events:none;opacity:0;transition:opacity 0.15s;background:rgba(2,6,18,0.97);border:1px solid rgba(var(--accent-rgb,52,211,153),0.25);box-shadow:0 4px 20px rgba(0,0,0,0.6);font-size:11px;padding:8px 12px;border-radius:8px;color:#e2e8f0;line-height:1.45;max-width:230px;white-space:normal;';
        document.body.appendChild(tt);
    }
    const cust = document.getElementById('agent-fs-customizer');
    if (!cust) return;
    cust.addEventListener('mouseover', function(e) {
        const tip = e.target.closest('[data-tip]');
        if (!tip) { tt.style.opacity = '0'; return; }
        tt.textContent = tip.dataset.tip;
        tt.style.opacity = '1';
        const r = tip.getBoundingClientRect();
        // Position to the right of the customizer panel
        const custRect = cust.getBoundingClientRect();
        let left = custRect.right + 8;
        let top = r.top + r.height / 2;
        // If it would overflow the viewport right, position to the left instead
        if (left + 240 > window.innerWidth) left = custRect.left - 240;
        tt.style.left = left + 'px';
        tt.style.top = top + 'px';
        tt.style.transform = 'translateY(-50%)';
    });
    cust.addEventListener('mouseout', function(e) {
        if (!e.relatedTarget || !cust.contains(e.relatedTarget)) tt.style.opacity = '0';
    });
    cust.addEventListener('mousemove', function(e) {
        const tip = e.target.closest('[data-tip]');
        if (!tip) { tt.style.opacity = '0'; return; }
        if (tt.textContent !== tip.dataset.tip) {
            tt.textContent = tip.dataset.tip;
            const r = tip.getBoundingClientRect();
            const custRect = cust.getBoundingClientRect();
            let left = custRect.right + 8;
            if (left + 240 > window.innerWidth) left = custRect.left - 240;
            tt.style.left = left + 'px';
            tt.style.top = (r.top + r.height / 2) + 'px';
        }
    });
}

function _fsCustUpdate() {
    const c = _loadFsCustom();
    const get = id => document.getElementById(id);
    c.chatOpacity = parseFloat(get('fsc-chat-opacity')?.value ?? c.chatOpacity);
    c.chatBlur = parseInt(get('fsc-chat-blur')?.value ?? c.chatBlur);
    c.chatWidth = parseInt(get('fsc-chat-width')?.value ?? c.chatWidth);
    c.sidebarOpacity = parseFloat(get('fsc-sidebar-opacity')?.value ?? c.sidebarOpacity);
    c.sidebarBlur = parseInt(get('fsc-sidebar-blur')?.value ?? c.sidebarBlur);
    c.fontSize = parseInt(get('fsc-font-size')?.value ?? c.fontSize);
    c.lineHeight = parseFloat(get('fsc-line-height')?.value ?? c.lineHeight);
    c.bubblePadding = parseInt(get('fsc-bubble-padding')?.value ?? c.bubblePadding);
    c.bubbleRadius = parseInt(get('fsc-bubble-radius')?.value ?? c.bubbleRadius);
    c.inputHeight = parseInt(get('fsc-input-height')?.value ?? c.inputHeight);
    c.sendSize = parseInt(get('fsc-send-size')?.value ?? c.sendSize);
    c.uiScale = parseFloat(get('fsc-ui-scale')?.value ?? c.uiScale);
    c.gradBar = get('fsc-grad-bar')?.checked ?? c.gradBar;
    // Update labels
    const setVal = (id, v) => { const el = get(id); if (el) el.textContent = v; };
    setVal('fsc-chat-opacity-val', Math.round(c.chatOpacity * 100) + '%');
    setVal('fsc-chat-blur-val', c.chatBlur + 'px');
    setVal('fsc-chat-width-val', c.chatWidth + 'px');
    setVal('fsc-sidebar-opacity-val', Math.round(c.sidebarOpacity * 100) + '%');
    setVal('fsc-sidebar-blur-val', c.sidebarBlur + 'px');
    setVal('fsc-font-size-val', c.fontSize + 'px');
    setVal('fsc-line-height-val', c.lineHeight.toFixed(2));
    setVal('fsc-bubble-padding-val', c.bubblePadding + 'px');
    setVal('fsc-bubble-radius-val', c.bubbleRadius + 'px');
    setVal('fsc-input-height-val', c.inputHeight + 'px');
    setVal('fsc-send-size-val', c.sendSize + 'px');
    setVal('fsc-ui-scale-val', Math.round(c.uiScale * 100) + '%');
    _saveFsCustom(c);
    _applyFsCustom(c);
}

function _fsCustReset() {
    localStorage.removeItem('stratos-fs-custom');
    _applyFsCustom(_fsCustomDefaults);
    // Refresh customizer panel if open
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; _toggleFsCustomizer(); }
}

function _fsCustSetTheme(t) {
    if (typeof setTheme === 'function') setTheme(t);
    // Refresh customizer to update active theme highlight
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; setTimeout(() => _toggleFsCustomizer(), 100); }
}

function _refreshFsSidebar() {
    if (!_agentFullscreen) return;
    const existing = document.querySelector('.agent-fs-sidebar');
    if (!existing) return;
    // Rebuild sidebar from current _agentConvList (already loaded by caller)
    const wrapper = existing.parentElement;
    if (wrapper) {
        existing.remove();
        wrapper.insertAdjacentHTML('afterbegin', _buildFsSidebar());
        lucide.createIcons();
    }
}

function toggleAgentFullscreen() {
    const panel = document.getElementById('agent-panel');
    const msgs = document.getElementById('agent-messages');
    const btn = document.getElementById('agent-fs-btn');
    if (!panel) return;

    _agentFullscreen = !_agentFullscreen;

    if (_agentFullscreen) {
        _openAgentPanel();

        // Reparent panel to <body> so it escapes #main-content opacity/stacking
        panel._origParent = panel.parentNode;
        panel._origNext = panel.nextElementSibling;
        document.body.appendChild(panel);

        panel.dataset.origStyle = panel.getAttribute('style') || '';
        panel.classList.add('agent-fullscreen');

        // Tag the main chat content div so we can find it reliably on exit
        const inner = panel.querySelector(':scope > .rounded-xl');
        if (inner && !panel.querySelector('.agent-fs-sidebar')) {
            inner.dataset.agentInner = '1';
            const wrapper = document.createElement('div');
            wrapper.className = 'agent-fs-wrapper';
            wrapper.style.cssText = 'display:flex;height:100%;width:100%;';
            panel.appendChild(wrapper);
            wrapper.insertAdjacentHTML('afterbegin', _buildFsSidebar());
            wrapper.appendChild(inner);
            lucide.createIcons();
        }

        if (msgs) { msgs.style.height = ''; msgs.style.maxHeight = 'none'; }
        if (btn) btn.title = 'Exit fullscreen (Esc)';
        const icon = btn?.querySelector('[data-lucide]');
        if (icon) { icon.setAttribute('data-lucide', 'minimize-2'); lucide.createIcons(); }
        const tabs = document.getElementById('agent-conv-tabs');
        if (tabs) tabs.style.display = 'none';
        // Apply saved fullscreen customizations
        setTimeout(() => {
            _applyFsCustom(_loadFsCustom());
            document.getElementById('agent-input')?.focus();
        }, 300);
    } else {
        // Close customizer if open
        if (_fsCustomizerOpen) { _fsCustomizerOpen = false; const cp = document.getElementById('agent-fs-customizer'); if (cp) cp.remove(); }
        // Remove sidebar wrapper, restore normal layout — find inner by data attribute
        const wrapper = panel.querySelector('.agent-fs-wrapper');
        const inner = wrapper?.querySelector('[data-agent-inner]');
        if (wrapper && inner) {
            delete inner.dataset.agentInner;
            panel.appendChild(inner);
            wrapper.remove();
        } else if (wrapper) {
            // Fallback: move all non-sidebar children out
            const sidebar = wrapper.querySelector('.agent-fs-sidebar');
            if (sidebar) sidebar.remove();
            while (wrapper.firstChild) panel.appendChild(wrapper.firstChild);
            wrapper.remove();
        }
        panel.classList.remove('agent-fullscreen');

        // Reparent panel back to its original location
        if (panel._origParent) {
            if (panel._origNext && panel._origNext.parentNode === panel._origParent) {
                panel._origParent.insertBefore(panel, panel._origNext);
            } else {
                panel._origParent.appendChild(panel);
            }
            delete panel._origParent;
            delete panel._origNext;
        }

        if (msgs) { msgs.style.height = '280px'; msgs.style.maxHeight = '600px'; }
        if (btn) btn.title = 'Fullscreen';
        const icon = btn?.querySelector('[data-lucide]');
        if (icon) { icon.setAttribute('data-lucide', 'maximize-2'); lucide.createIcons(); }
        const tabs = document.getElementById('agent-conv-tabs');
        if (tabs) tabs.style.display = '';
        _renderConvTabs();
    }
    setTimeout(() => { if (msgs) msgs.scrollTop = msgs.scrollHeight; }, 100);
}

// Escape key exits agent fullscreen
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && _agentFullscreen) {
        toggleAgentFullscreen();
    }
});

// ═══════════════════════════════════════════════════════════
// WATCHLIST WIDGET — Rich card display for ticker list
// ═══════════════════════════════════════════════════════════

function _buildWatchlistWidget(tickers) {
    const fp = v => v != null ? v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '--';

    let cards = '';
    for (const sym of tickers) {
        const lbl = sym.replace('-USD','').replace('=F','');
        // Try to find market data (exact, -USD, =F variants)
        const md = (typeof marketData !== 'undefined') ? (marketData[sym] || marketData[sym+'-USD'] || marketData[sym+'=F']) : null;
        const ad = md?.data ? (_resolveData(sym, '1m') || _resolveData(sym+'-USD', '1m') || _resolveData(sym+'=F', '1m') ||
                               _resolveData(sym, '5m') || _resolveData(sym+'-USD', '5m') || _resolveData(sym+'=F', '5m')) : null;

        const realSym = marketData[sym] ? sym : marketData[sym+'-USD'] ? sym+'-USD' : marketData[sym+'=F'] ? sym+'=F' : sym;
        const name = (typeof getAssetName === 'function') ? getAssetName(realSym, md?.name || lbl) : lbl;
        const price = ad?.price || 0;
        const change = ad?.change || 0;
        const up = change >= 0;
        const hasData = price > 0;

        // Mini sparkline SVG
        let spark = '';
        const hist = ad?.history || [];
        if (hist.length >= 4) {
            const pts = hist.slice(-24);
            const mn = Math.min(...pts), mx = Math.max(...pts), rng = mx - mn || 1;
            const w = 48, h = 18;
            const path = pts.map((v, i) => `${i===0?'M':'L'}${(i/(pts.length-1)*w).toFixed(1)},${(h-((v-mn)/rng)*h).toFixed(1)}`).join(' ');
            spark = `<svg width="${w}" height="${h}" style="flex-shrink:0;"><path d="${path}" fill="none" stroke="${up?'#10b981':'#ef4444'}" stroke-width="1.5"/></svg>`;
        }

        const clickAction = hasData ? `navigateToTicker('${realSym}')` : '';
        const cursor = hasData ? 'cursor-pointer' : 'cursor-default';
        const hoverBorder = hasData ? 'hover:border-slate-600' : '';

        cards += `
        <div onclick="${clickAction}" class="${cursor} bg-slate-900/50 border border-slate-800/60 rounded-lg p-2.5 transition-all ${hoverBorder} flex items-center gap-2.5" style="min-width:0;">
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-1.5">
                    <span class="text-[11px] font-mono font-bold ${hasData ? 'text-cyan-400' : 'text-slate-300'}">${lbl}</span>
                    <span class="text-[9px] text-slate-500 truncate">${name !== lbl ? name : ''}</span>
                </div>
                ${hasData ? `
                <div class="flex items-center gap-2 mt-0.5">
                    <span class="text-[11px] font-bold text-white font-mono">$${fp(price)}</span>
                    <span class="text-[10px] font-mono font-bold ${up?'text-emerald-400':'text-red-400'}">${up?'+':''}${change.toFixed(2)}%</span>
                </div>` : `<div class="text-[10px] text-slate-500 mt-0.5 italic">No data loaded</div>`}
            </div>
            ${spark}
        </div>`;
    }

    const html = `
        <div style="margin-bottom:8px;">
            <span class="text-[11px] font-bold" style="color:var(--text-heading,#e2e8f0);">Your watchlist (${tickers.length}):</span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:6px; margin-bottom:8px;">
            ${cards}
        </div>
        <div class="text-[10px]" style="color:var(--text-muted,#64748b);">
            Say <b>"add ticker TSLA"</b> or <b>"remove ticker CVX"</b> to modify.
        </div>`;

    const plain = `Your watchlist (${tickers.length}): ${tickers.join(', ')}`;
    return { html, plain };
}

// ═══════════════════════════════════════════════════════════
// TICKER COMMANDS — add/remove/list tickers via agent chat
// ═══════════════════════════════════════════════════════════

function _parseTickerCommand(msg) {
    const m = msg.trim();
    const lower = m.toLowerCase();

    // ── Ticker commands ──

    if (/^(show|list|what('?s| are)?)\s*(my\s*)?(tickers|watchlist|market\s*tickers)/i.test(lower) ||
        /^my\s*(tickers|watchlist)$/i.test(lower)) {
        return { action: 'list' };
    }

    // "add ticker TSLA" / "add TSLA to watchlist" (no "to <other>")
    const addTickerMatch = m.match(/^add\s+(?:ticker[s]?\s+)(.+?)(?:\s+to\s+(?:my\s+)?(?:watchlist|tickers|market))?$/i);
    if (addTickerMatch) {
        const syms = addTickerMatch[1].split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(s => s && /^[A-Z0-9.\-=^]{1,15}$/.test(s));
        if (syms.length) return { action: 'add', symbols: syms };
    }

    // "remove ticker CVX" / "remove CVX from watchlist"
    const rmTickerMatch = m.match(/^(?:remove|delete|drop)\s+(?:ticker[s]?\s+)(.+?)(?:\s+from\s+(?:my\s+)?(?:watchlist|tickers|market))?$/i);
    if (rmTickerMatch) {
        const syms = rmTickerMatch[1].split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(s => s && /^[A-Z0-9.\-=^]{1,15}$/.test(s));
        if (syms.length) return { action: 'remove', symbols: syms };
    }

    // ── Category / keyword commands ──

    if (/^(show|list|what('?s| are)?)\s*(my\s*)?(categories|search\s*(keywords|queries)|feeds)/i.test(lower) ||
        /^my\s*(categories|feeds)$/i.test(lower)) {
        return { action: 'list_categories' };
    }

    // "show keywords in <category>"
    const showKwMatch = m.match(/^(?:show|list|what('?s| are)?)\s*(?:the\s+)?(?:keywords?|items?|queries?)\s*(?:in|for|of)\s+(.+)/i);
    if (showKwMatch) {
        return { action: 'show_keywords', category: showKwMatch[2].trim() };
    }

    // "add <keyword> to <category>"
    const addKwMatch = m.match(/^add\s+(.+?)\s+to\s+(.+)/i);
    if (addKwMatch) {
        const cat = addKwMatch[2].replace(/\s*(?:category|feed|keywords?)\s*/i, '').trim();
        if (cat.length > 1 && !/^(?:my\s+)?(?:watchlist|tickers|market)$/i.test(cat)) {
            const keywords = addKwMatch[1].split(/,/).map(s => s.trim()).filter(s => s.length > 0);
            if (keywords.length) return { action: 'add_keyword', keywords, category: cat };
        }
    }

    // "remove <keyword> from <category>"
    const rmKwMatch = m.match(/^(?:remove|delete|drop)\s+(.+?)\s+from\s+(.+)/i);
    if (rmKwMatch) {
        const cat = rmKwMatch[2].replace(/\s*(?:category|feed|keywords?)\s*/i, '').trim();
        if (cat.length > 1 && !/^(?:my\s+)?(?:watchlist|tickers|market)$/i.test(cat)) {
            const keywords = rmKwMatch[1].split(/,/).map(s => s.trim()).filter(s => s.length > 0);
            if (keywords.length) return { action: 'remove_keyword', keywords, category: cat };
        }
    }

    // "enable/disable <category>"
    const toggleMatch = m.match(/^(enable|disable|turn\s+(?:on|off))\s+(.+?)(?:\s+category)?$/i);
    if (toggleMatch) {
        const on = /enable|on/i.test(toggleMatch[1]);
        return { action: 'toggle_category', category: toggleMatch[2].trim(), enabled: on };
    }

    // ── Fallback: bare "add X" / "remove X" still treated as ticker ──
    const bareAdd = m.match(/^add\s+(.+)$/i);
    if (bareAdd && !bareAdd[1].match(/\s+to\s+/i)) {
        const syms = bareAdd[1].split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(s => s && /^[A-Z0-9.\-=^]{1,15}$/.test(s));
        if (syms.length) return { action: 'add', symbols: syms };
    }
    const bareRm = m.match(/^(?:remove|delete|drop)\s+(.+)$/i);
    if (bareRm && !bareRm[1].match(/\s+from\s+/i)) {
        const syms = bareRm[1].split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(s => s && /^[A-Z0-9.\-=^]{1,15}$/.test(s));
        if (syms.length) return { action: 'remove', symbols: syms };
    }

    return null;
}

// ── Fuzzy-match a user string to a category id or label ──
function _findCategory(categories, query) {
    const q = query.toLowerCase().replace(/[^a-z0-9\s]/g, '');
    // Exact match on id or label
    let match = categories.find(c => c.id?.toLowerCase() === q || c.label?.toLowerCase() === q);
    if (match) return match;
    // Partial / fuzzy
    match = categories.find(c =>
        c.label?.toLowerCase().includes(q) || q.includes(c.label?.toLowerCase()) ||
        c.id?.toLowerCase().includes(q) || q.includes(c.id?.toLowerCase())
    );
    return match || null;
}

async function _executeTickerCommand(cmd) {
    const headers = {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
        'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
    };

    try {
        const cfgRes = await fetch('/api/config', { headers });
        if (!cfgRes.ok) throw new Error('Failed to load config');
        const cfg = await cfgRes.json();
        let tickers = (cfg.market?.tickers || []).map(t => t.symbol);
        let categories = cfg.dynamic_categories || [];

        // ── Ticker: list ──
        if (cmd.action === 'list') {
            if (!tickers.length) return { plain: "Your watchlist is empty.", html: formatAgentText("Your watchlist is empty. Say **\"add ticker TSLA\"** to add one.") };
            return _buildWatchlistWidget(tickers);
        }

        // ── Ticker: add ──
        if (cmd.action === 'add') {
            const already = cmd.symbols.filter(s => tickers.includes(s));
            const adding = cmd.symbols.filter(s => !tickers.includes(s));
            tickers = [...tickers, ...adding];
            await _saveConfig(headers, { tickers });
            let msg = '';
            if (adding.length) msg += `Added **${adding.join(', ')}** to your watchlist.`;
            if (already.length) msg += `${msg ? ' ' : ''}**${already.join(', ')}** already there.`;
            msg += `\n\nWatchlist: **${tickers.join(', ')}** (${tickers.length})\nRefresh market data to see them on the dashboard.`;
            _syncSimpleTickers(tickers);
            return msg;
        }

        // ── Ticker: remove ──
        if (cmd.action === 'remove') {
            const removing = cmd.symbols.filter(s => tickers.includes(s));
            const notFound = cmd.symbols.filter(s => !tickers.includes(s));
            tickers = tickers.filter(t => !cmd.symbols.includes(t));
            await _saveConfig(headers, { tickers });
            let msg = '';
            if (removing.length) msg += `Removed **${removing.join(', ')}**.`;
            if (notFound.length) msg += `${msg ? ' ' : ''}**${notFound.join(', ')}** wasn't in your watchlist.`;
            msg += tickers.length ? `\n\nWatchlist: **${tickers.join(', ')}** (${tickers.length})` : '\n\nWatchlist is now empty.';
            _syncSimpleTickers(tickers);
            return msg;
        }

        // ── Categories: list ──
        if (cmd.action === 'list_categories') {
            if (!categories.length) return "No categories configured. Set them up in **Settings**.";
            const lines = categories.map(c => {
                const status = c.enabled === false ? '🔴' : '🟢';
                const count = (c.items || []).length;
                return `${status} **${c.label}** — ${count} keyword${count !== 1 ? 's' : ''}`;
            });
            return `**Your categories (${categories.length}):**\n\n${lines.join('\n')}\n\nSay **\"show keywords in [category]\"** for details, or **\"add [keyword] to [category]\"** to modify.`;
        }

        // ── Categories: show keywords ──
        if (cmd.action === 'show_keywords') {
            const cat = _findCategory(categories, cmd.category);
            if (!cat) return `Couldn't find a category matching **"${cmd.category}"**.\n\nYour categories: ${categories.map(c => `**${c.label}**`).join(', ')}`;
            const items = cat.items || [];
            if (!items.length) return `**${cat.label}** has no keywords yet. Say **\"add [keyword] to ${cat.label}\"** to add some.`;
            return `**${cat.label}** (${items.length} keywords):\n\n${items.map(i => `• ${i}`).join('\n')}\n\nSay **\"add [keyword] to ${cat.label}\"** or **\"remove [keyword] from ${cat.label}\"**.`;
        }

        // ── Categories: add keyword ──
        if (cmd.action === 'add_keyword') {
            const cat = _findCategory(categories, cmd.category);
            if (!cat) return `Couldn't find a category matching **"${cmd.category}"**.\n\nYour categories: ${categories.map(c => `**${c.label}**`).join(', ')}`;
            const existing = (cat.items || []).map(i => i.toLowerCase());
            const adding = cmd.keywords.filter(k => !existing.includes(k.toLowerCase()));
            const already = cmd.keywords.filter(k => existing.includes(k.toLowerCase()));
            cat.items = [...(cat.items || []), ...adding];
            await _saveConfig(headers, { dynamic_categories: categories });
            let msg = '';
            if (adding.length) msg += `Added **${adding.join(', ')}** to **${cat.label}**.`;
            if (already.length) msg += `${msg ? ' ' : ''}**${already.join(', ')}** already there.`;
            msg += `\n\n**${cat.label}** now has ${cat.items.length} keywords. Run a new scan to fetch results.`;
            return msg;
        }

        // ── Categories: remove keyword ──
        if (cmd.action === 'remove_keyword') {
            const cat = _findCategory(categories, cmd.category);
            if (!cat) return `Couldn't find a category matching **"${cmd.category}"**.\n\nYour categories: ${categories.map(c => `**${c.label}**`).join(', ')}`;
            const removing = cmd.keywords.filter(k => (cat.items || []).some(i => i.toLowerCase() === k.toLowerCase()));
            const notFound = cmd.keywords.filter(k => !(cat.items || []).some(i => i.toLowerCase() === k.toLowerCase()));
            cat.items = (cat.items || []).filter(i => !cmd.keywords.some(k => k.toLowerCase() === i.toLowerCase()));
            await _saveConfig(headers, { dynamic_categories: categories });
            let msg = '';
            if (removing.length) msg += `Removed **${removing.join(', ')}** from **${cat.label}**.`;
            if (notFound.length) msg += `${msg ? ' ' : ''}**${notFound.join(', ')}** wasn't found.`;
            msg += `\n\n**${cat.label}** now has ${cat.items.length} keywords.`;
            return msg;
        }

        // ── Categories: enable/disable ──
        if (cmd.action === 'toggle_category') {
            const cat = _findCategory(categories, cmd.category);
            if (!cat) return `Couldn't find a category matching **"${cmd.category}"**.\n\nYour categories: ${categories.map(c => `**${c.label}**`).join(', ')}`;
            cat.enabled = cmd.enabled;
            await _saveConfig(headers, { dynamic_categories: categories });
            return `**${cat.label}** is now **${cmd.enabled ? 'enabled 🟢' : 'disabled 🔴'}**.${!cmd.enabled ? " It won't be included in future scans." : ' It will be included in the next scan.'}`;
        }

    } catch (e) {
        return `Failed to update: ${e.message}`;
    }
}

async function _saveConfig(headers, data) {
    const res = await fetch('/api/config', { method: 'POST', headers, body: JSON.stringify(data) });
    if (!res.ok) throw new Error('Config save failed');
}

function _syncSimpleTickers(tickers) {
    if (typeof simpleTickers !== 'undefined') {
        simpleTickers.length = 0;
        tickers.forEach(t => simpleTickers.push(t));
        if (typeof renderSimpleTickers === 'function') renderSimpleTickers();
    }
}

// ═══════════════════════════════════════════════════════════
// AGENT CHAT RESIZE
// ═══════════════════════════════════════════════════════════
(function initAgentResize() {
    document.addEventListener('DOMContentLoaded', () => {
        // Restore chat history from localStorage
        _restoreAgentHistory();
        // Render initial suggestion chips
        renderAgentSuggestions();
        // Restore TTS preference
        if (localStorage.getItem('stratos_tts_enabled') === '0') {
            document.body.classList.add('tts-disabled');
        }
        
        setTimeout(() => {
            const handle = document.getElementById('agent-resize-handle');
            const msgs = document.getElementById('agent-messages');
            if (!handle || !msgs) return;
            
            let startY, startH;
            
            handle.addEventListener('mousedown', (e) => {
                e.preventDefault();
                startY = e.clientY;
                startH = msgs.offsetHeight;
                document.addEventListener('mousemove', onDrag);
                document.addEventListener('mouseup', onStop);
                document.body.style.cursor = 'ns-resize';
                document.body.style.userSelect = 'none';
            });
            
            handle.addEventListener('touchstart', (e) => {
                const t = e.touches[0];
                startY = t.clientY;
                startH = msgs.offsetHeight;
                document.addEventListener('touchmove', onTouchDrag, { passive: false });
                document.addEventListener('touchend', onStop);
            }, { passive: true });
            
            function onDrag(e) {
                const newH = Math.min(600, Math.max(120, startH + (e.clientY - startY)));
                msgs.style.height = newH + 'px';
            }
            
            function onTouchDrag(e) {
                e.preventDefault();
                const t = e.touches[0];
                const newH = Math.min(600, Math.max(120, startH + (t.clientY - startY)));
                msgs.style.height = newH + 'px';
            }
            
            function onStop() {
                document.removeEventListener('mousemove', onDrag);
                document.removeEventListener('mouseup', onStop);
                document.removeEventListener('touchmove', onTouchDrag);
                document.removeEventListener('touchend', onStop);
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        }, 500);
    });
})();
