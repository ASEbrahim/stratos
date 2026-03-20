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
let _personaSuggestions = {};  // Per-persona dynamic suggestion cache (profile-scoped)
function _suggestionsKey() {
    const p = typeof getActiveProfile === 'function' ? getActiveProfile() : '';
    return p ? 'stratos_persona_suggestions_' + p : 'stratos_persona_suggestions';
}
try { _personaSuggestions = JSON.parse(localStorage.getItem(_suggestionsKey()) || '{}'); } catch(e) {}
let _agentFreeLength = false;  // Extended response mode
let _agentAllScans = false;    // Use all stored scan data vs current only
let _agentInjectSignals = true; // Signal injection toggle (feed data in context)

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

function _toggleFreeLength() {
    _agentFreeLength = !_agentFreeLength;
    window._agentFreeLength = _agentFreeLength;
    const btn = document.getElementById('agent-free-length-btn');
    if (btn) {
        btn.textContent = _agentFreeLength ? 'Long' : 'Short';
        btn.style.background = _agentFreeLength ? 'rgba(52,211,153,0.12)' : 'rgba(255,255,255,0.03)';
        btn.style.color = _agentFreeLength ? 'var(--accent,#34d399)' : 'var(--text-muted)';
        btn.style.borderColor = _agentFreeLength ? 'rgba(52,211,153,0.3)' : 'rgba(255,255,255,0.1)';
    }
}
window._toggleFreeLength = _toggleFreeLength;

function _toggleAllScans() {
    _agentAllScans = !_agentAllScans;
    window._agentAllScans = _agentAllScans;
    const btn = document.getElementById('agent-all-scans-btn');
    if (btn) {
        btn.textContent = _agentAllScans ? 'All Scans' : 'Current';
        btn.style.background = _agentAllScans ? 'rgba(96,165,250,0.12)' : 'rgba(255,255,255,0.03)';
        btn.style.color = _agentAllScans ? '#60a5fa' : 'var(--text-muted)';
        btn.style.borderColor = _agentAllScans ? 'rgba(96,165,250,0.3)' : 'rgba(255,255,255,0.1)';
    }
}
window._toggleAllScans = _toggleAllScans;

function _toggleSignalInjection() {
    _agentInjectSignals = !_agentInjectSignals;
    window._agentInjectSignals = _agentInjectSignals;
    const btn = document.getElementById('agent-signal-toggle-btn');
    if (btn) {
        btn.textContent = _agentInjectSignals ? 'Signals: ON' : 'Signals: OFF';
        btn.style.background = _agentInjectSignals ? 'rgba(251,191,36,0.12)' : 'rgba(255,255,255,0.03)';
        btn.style.color = _agentInjectSignals ? '#fbbf24' : 'var(--text-muted)';
        btn.style.borderColor = _agentInjectSignals ? 'rgba(251,191,36,0.3)' : 'rgba(255,255,255,0.1)';
    }
}
window._toggleSignalInjection = _toggleSignalInjection;

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
    // Also refresh compact sidebar if open
    if (typeof _refreshCompactSidebar === 'function') _refreshCompactSidebar();
}

async function _renameConversation(convId) {
    const conv = _agentConvList.find(c => c.id === convId);
    if (!conv) return;
    const newTitle = await stratosPrompt({ title: 'Rename Conversation', label: 'Title', defaultValue: conv.title || 'New Chat' });
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
        if (h.role === 'assistant') {
            appendAgentMessage('assistant', formatAgentText(h.content));
        } else {
            appendAgentMessage('user', h.content);
        }
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
    // Show/hide RP-specific controls (director's note button)
    _updateRPControls(name);
}

function _updateRPControls(persona) {
    const isRP = persona === 'roleplay' || persona === 'gaming';
    let dirBtn = document.getElementById('rp-director-btn');
    if (isRP && !dirBtn) {
        const uploadBtn = document.getElementById('agent-upload-btn');
        if (uploadBtn) {
            dirBtn = document.createElement('button');
            dirBtn.id = 'rp-director-btn';
            dirBtn.className = 'w-9 h-9 rounded-xl flex items-center justify-center transition-all flex-shrink-0';
            dirBtn.style.cssText = 'color:var(--text-muted);border:1px solid var(--border-strong);background:rgba(255,255,255,0.02);';
            dirBtn.title = "Director's Note — steer the AI";
            dirBtn.setAttribute('aria-label', "Director's Note");
            dirBtn.setAttribute('onmouseenter', "this.style.borderColor='var(--accent)';this.style.color='var(--accent)'");
            dirBtn.setAttribute('onmouseleave', "this.style.borderColor='var(--border-strong)';this.style.color='var(--text-muted)'");
            dirBtn.onclick = _rpToggleDirector;
            dirBtn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i>';
            uploadBtn.parentElement.insertBefore(dirBtn, uploadBtn);
            lucide.createIcons();
        }
    } else if (!isRP && dirBtn) {
        dirBtn.remove();
        const panel = document.getElementById('rp-director-panel');
        if (panel) panel.remove();
    }
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

function _showPersonaGuide() {
    const existing = document.getElementById('persona-guide-overlay');
    if (existing) { existing.remove(); return; }

    const overlay = document.createElement('div');
    overlay.id = 'persona-guide-overlay';
    overlay.className = 'pg-overlay';

    const modes = [
        { key: 'intelligence', color: '#34d399', rgb: '52,211,153', title: 'Intelligence',
          sub: 'News analysis & signal detection', badge: 'Core', badgeBg: 'rgba(52,211,153,0.1)',
          desc: 'Your personal news analyst. Monitors your scored feed for critical signals, summarizes trends, and alerts when high-relevance stories break.',
          pills: ['Web Search','Feed Scoring','Categories','Trends'],
          examples: ['"What\'s critical today?"','"Search NVIDIA earnings"'] },
        { key: 'market', color: '#60a5fa', rgb: '96,165,250', title: 'Market',
          sub: 'Live financial data & watchlist', badge: 'Live', badgeBg: 'rgba(96,165,250,0.1)',
          desc: 'Financial analyst with live market feeds. Pulls real-time prices, compares assets side-by-side, and identifies top movers with sentiment context.',
          pills: ['Live Prices','Watchlist','Compare','Movers'],
          examples: ['"How\'s NVDA doing?"','"Compare BTC vs ETH"'] },
        { key: 'scholarly', color: '#c084fc', rgb: '192,132,252', title: 'Scholarly',
          sub: 'Research & knowledge base', badge: 'Research', badgeBg: 'rgba(192,132,252,0.1)',
          desc: 'Academic companion with deep knowledge base access. Draws from YouTube transcripts, video insights, and documents for thorough, sourced research.',
          pills: ['Knowledge Base','Videos','Web','Files'],
          examples: ['"Ottoman Empire fall"','"Hegel\'s dialectics"'] },
        { key: 'gaming', color: '#f472b6', rgb: '244,114,182', title: 'Gaming',
          sub: 'Interactive RPG & world builder', badge: 'Interactive', badgeBg: 'rgba(244,114,182,0.1)',
          desc: 'Full RPG engine. <b style="color:#f472b6">GM Mode</b>: narration, stats, dice. <b style="color:#f472b6">Immersive</b>: NPCs respond in-character. Build or import entire fictional universes.',
          pills: ['GM & Immersive RP','Scenarios','NPCs','Files'],
          examples: ['"Start an adventure"','"Talk to the merchant"'] },
        { key: 'anime', color: '#fb923c', rgb: '251,146,60', title: 'Anime',
          sub: 'Seasonal tracking & taste-based picks', badge: null,
          desc: 'Your otaku companion. Tracks seasonal anime, curates manga picks based on your taste, and dives into studio histories, voice actors, and genre evolution.',
          pills: ['Seasonal','Recommendations','Genres'],
          examples: ['"Airing this season?"','"Like Vinland Saga"'] },
        { key: 'tcg', color: '#fbbf24', rgb: '251,191,36', title: 'TCG',
          sub: 'Card values, meta & set releases', badge: null,
          desc: 'Covers Pokémon, Magic: The Gathering, Yu-Gi-Oh!, and more. Tracks card values, analyzes competitive meta shifts, and keeps you current on new sets and chase cards.',
          pills: ['Values','Meta','Releases'],
          examples: ['"Best Pokémon cards"','"MTG standard meta"'] },
    ];

    const svgIcons = {
        intelligence: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>',
        market: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        scholarly: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>',
        gaming: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f472b6" stroke-width="2"><line x1="6" y1="11" x2="10" y2="11"/><line x1="8" y1="9" x2="8" y2="13"/><path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258A4 4 0 0 0 17.32 5z"/></svg>',
        anime: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fb923c" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
        tcg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2"/><rect x="4" y="3" width="16" height="14" rx="2" opacity="0.5"/></svg>',
    };

    function buildCard(m) {
        const badgeHtml = m.badge ? `<span class="pg-card-badge" style="background:${m.badgeBg};color:${m.color};border:1px solid ${m.color}33;">${m.badge}</span>` : '';
        const pillsHtml = m.pills.map(p =>
            `<span class="pg-pill" style="background:${m.color}14;color:${m.color};border:1px solid ${m.color}1f;">${p}</span>`
        ).join('');
        const exHtml = m.examples.map(e =>
            `<span class="pg-ex" data-persona="${m.key}" data-prompt="${e.replace(/"/g,'&quot;')}">${e}</span>`
        ).join('');
        return `<div class="pg-card" style="--card-color:${m.color}4d;--card-rgb:${m.rgb};" data-persona-key="${m.key}">
            <div class="pg-card-top">
                <div class="pg-card-icon" style="background:${m.color}1a;">${svgIcons[m.key]}</div>
                <div class="pg-card-meta">
                    <div class="pg-card-title" style="color:${m.color};">${m.title}</div>
                    <div class="pg-card-sub">${m.sub}</div>
                </div>
                ${badgeHtml}
            </div>
            <div class="pg-card-desc">${m.desc}</div>
            <div class="pg-card-row">${pillsHtml}</div>
            <div class="pg-card-row">${exHtml}</div>
        </div>`;
    }

    // Row pairs: [0,1], [2,3], world-import, [4,5]
    const row1 = `<div class="pg-row">${buildCard(modes[0])}${buildCard(modes[1])}</div>`;
    const row2 = `<div class="pg-row">${buildCard(modes[2])}${buildCard(modes[3])}</div>`;
    const worldImport = `<div class="pg-row"><div class="pg-highlight">
        <div class="pg-hl-icon">&#127759;</div>
        <div class="pg-hl-content">
            <div class="pg-hl-title">World Import <span class="pg-hl-new">New</span></div>
            <div class="pg-hl-desc">Name any anime, game, show, or book and the <em>Gaming</em> persona fetches canon details from the web — then auto-populates your scenario with world lore, characters, locations, and items.</div>
            <div class="pg-hl-examples">
                <span class="pg-hl-ex" data-prompt="Implement the SAO world into this scenario">"Implement the SAO world"</span>
                <span class="pg-hl-ex" data-prompt="Build a Witcher scenario with all canon details">"Build a Witcher scenario"</span>
                <span class="pg-hl-ex" data-prompt="Set up the Avatar universe">"Set up the Avatar universe"</span>
                <span class="pg-hl-ex" data-prompt="Create a Naruto world with all characters">"Create a Naruto world"</span>
            </div>
        </div>
    </div></div>`;
    const row3 = `<div class="pg-row">${buildCard(modes[4])}${buildCard(modes[5])}</div>`;

    overlay.innerHTML = `<div class="pg-container">
        <canvas class="pg-stars" id="pg-stars-canvas"></canvas>
        <div class="pg-titlebar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent,#34d399)" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>
            <span class="pg-title">Agent Personas</span>
            <span class="pg-subtitle">Each persona is a specialized AI mode with its own tools, memory, and data access.</span>
            <button class="pg-close" id="pg-close-btn">&times;</button>
        </div>
        <div class="pg-body">${row1}${row2}${worldImport}${row3}</div>
        <div class="pg-footer"><b>Combine up to 3 personas</b> from the picker to merge tools in one conversation.</div>
    </div>`;

    function closeGuide() { overlay.remove(); _pgStarRaf && cancelAnimationFrame(_pgStarRaf); _pgStarRaf = null; }
    overlay.addEventListener('click', e => { if (e.target === overlay) closeGuide(); });
    overlay.querySelector('#pg-close-btn').addEventListener('click', closeGuide);
    document.addEventListener('keydown', function esc(e) { if (e.key === 'Escape') { closeGuide(); document.removeEventListener('keydown', esc); } });

    // Click card → switch persona
    overlay.querySelectorAll('.pg-card[data-persona-key]').forEach(card => {
        card.addEventListener('click', e => {
            if (e.target.closest('.pg-ex')) return; // let example clicks handle themselves
            const key = card.dataset.personaKey;
            if (key && typeof _onPersonaCheckChange === 'function') {
                selectedPersonas = [key];
                currentPersona = key;
                _updatePersonaPickerLabel();
                _renderPersonaPicker();
                _updatePersonaWelcome();
                if (typeof renderAgentSuggestions === 'function') renderAgentSuggestions();
                if (typeof _onPersonaChanged === 'function') _onPersonaChanged(currentPersona);
                if (typeof updateScenarioBar === 'function') updateScenarioBar();
            }
            closeGuide();
        });
    });

    // Click example → send as prompt
    overlay.querySelectorAll('.pg-ex[data-prompt], .pg-hl-ex[data-prompt]').forEach(el => {
        el.addEventListener('click', e => {
            e.stopPropagation();
            const prompt = el.dataset.prompt.replace(/^"|"$/g, '');
            const personaKey = el.dataset.persona || 'gaming';
            if (personaKey && !selectedPersonas.includes(personaKey)) {
                selectedPersonas = [personaKey];
                currentPersona = personaKey;
                _updatePersonaPickerLabel();
                _renderPersonaPicker();
                _updatePersonaWelcome();
                if (typeof _onPersonaChanged === 'function') _onPersonaChanged(currentPersona);
                if (typeof updateScenarioBar === 'function') updateScenarioBar();
            }
            closeGuide();
            const input = document.getElementById('agent-input');
            if (input) { input.value = prompt; input.focus(); }
        });
    });

    document.body.appendChild(overlay);

    // Star parallax engine
    let _pgStarRaf = null;
    const canvas = document.getElementById('pg-stars-canvas');
    if (canvas) {
        const container = canvas.parentElement;
        const ctx = canvas.getContext('2d');
        canvas.width = container.offsetWidth; canvas.height = container.offsetHeight;
        const ar = 52, ag = 211, ab = 153, COUNT = 100;
        const stars = [];
        for (let i = 0; i < COUNT; i++) {
            stars.push({ x: Math.random()*canvas.width, y: Math.random()*canvas.height, r: Math.random()*1.4+0.3, a: Math.random()*0.4+0.08, speed: Math.random()*0.06+0.01, phase: Math.random()*Math.PI*2, cr: Math.random()<0.3?ar:255, cg: Math.random()<0.3?ag:255, cb: Math.random()<0.3?ab:255 });
        }
        const shooters = []; let lastShoot = Date.now();
        function drawStars() {
            ctx.clearRect(0,0,canvas.width,canvas.height);
            const t = Date.now()*0.001;
            for (const s of stars) {
                s.y -= s.speed; if (s.y < -2) { s.y = canvas.height+2; s.x = Math.random()*canvas.width; }
                const f = 0.6+0.4*Math.sin(t*2+s.phase), alpha = s.a*f;
                ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,Math.PI*2); ctx.fillStyle=`rgba(${s.cr},${s.cg},${s.cb},${alpha})`; ctx.fill();
                if(s.r>1){ctx.beginPath();ctx.arc(s.x,s.y,s.r*3,0,Math.PI*2);ctx.fillStyle=`rgba(${s.cr},${s.cg},${s.cb},${alpha*0.15})`;ctx.fill();}
            }
            for(let i=0;i<stars.length;i++){for(let j=i+1;j<stars.length;j++){const dx=stars[i].x-stars[j].x,dy=stars[i].y-stars[j].y,dist=Math.sqrt(dx*dx+dy*dy);if(dist<90){ctx.beginPath();ctx.moveTo(stars[i].x,stars[i].y);ctx.lineTo(stars[j].x,stars[j].y);ctx.strokeStyle=`rgba(${ar},${ag},${ab},${0.05*(1-dist/90)})`;ctx.lineWidth=0.5;ctx.stroke();}}}
            if(Date.now()-lastShoot>5000+Math.random()*4000){lastShoot=Date.now();shooters.push({x:Math.random()*canvas.width*0.6,y:Math.random()*canvas.height*0.3,vx:Math.cos(0.35)*4,vy:Math.sin(0.35)*4,life:1,len:Math.random()*30+20});}
            for(let i=shooters.length-1;i>=0;i--){const sh=shooters[i];sh.x+=sh.vx;sh.y+=sh.vy;sh.life-=0.015;if(sh.life<=0){shooters.splice(i,1);continue;}const g=ctx.createLinearGradient(sh.x,sh.y,sh.x-sh.vx*sh.len/5,sh.y-sh.vy*sh.len/5);g.addColorStop(0,`rgba(255,255,255,${sh.life*0.7})`);g.addColorStop(1,'rgba(255,255,255,0)');ctx.beginPath();ctx.moveTo(sh.x,sh.y);ctx.lineTo(sh.x-sh.vx*sh.len/5,sh.y-sh.vy*sh.len/5);ctx.strokeStyle=g;ctx.lineWidth=1.5;ctx.stroke();}
            _pgStarRaf = requestAnimationFrame(drawStars);
        }
        _pgStarRaf = requestAnimationFrame(drawStars);
    }
}
window._showPersonaGuide = _showPersonaGuide;

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

// Suggestions moved to agent-suggestions.js


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
        wrapper.className = 'flex justify-end mb-3 group/usermsg';
        wrapper.innerHTML = `
            <div class="max-w-[82%]">
                <div class="agent-bubble-user rounded-2xl rounded-br-sm px-4 py-2.5 text-xs leading-relaxed" style="color:var(--text-heading); background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.18); backdrop-filter:blur(6px);">
                    ${escAgent(content)}
                </div>
                <div class="flex items-center justify-end gap-1.5 mt-1">
                    <span class="text-[9px]" style="color:var(--text-muted); opacity:0.4;">${time}</span>
                    <button onclick="_editUserMessage(this)" class="p-0.5 rounded opacity-0 group-hover/usermsg:opacity-100 transition-opacity" title="Edit message">
                        <i data-lucide="pencil" class="w-2.5 h-2.5" style="color:var(--text-muted);"></i>
                    </button>
                </div>
            </div>`;
    } else {
        wrapper.className = 'flex gap-3 mb-3 group/msg';
        wrapper.innerHTML = `
            <div class="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5" style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.15);">
                <i data-lucide="bot" class="w-3.5 h-3.5 text-emerald-400"></i>
            </div>
            <div class="flex-1 min-w-0">
                <div class="agent-response text-sm leading-relaxed" style="color:var(--text-body,#cbd5e1);">${content}</div><!-- content is pre-sanitized via formatAgentText() which HTML-escapes before markdown formatting -->
                <div class="flex items-center gap-2 mt-1 agent-msg-actions">
                    <span class="text-[9px]" style="color:var(--text-muted); opacity:0.4;">${time}</span>
                    <button onclick="_copyAgentMessage(this)" class="p-0.5 rounded" title="Copy message">
                        <i data-lucide="copy" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    <button onclick="speakMessage(this.closest('.agent-msg-actions').parentElement.querySelector('.agent-response').innerText, this)" class="speak-btn p-0.5 rounded" title="Read aloud">
                        <i data-lucide="volume-2" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    <button onclick="_editAssistantMessage(this)" class="p-0.5 rounded opacity-0 group-hover/msg:opacity-100 transition-opacity" title="Edit response">
                        <i data-lucide="pencil" class="w-2.5 h-2.5" style="color:var(--text-muted);"></i>
                    </button>
                    ${(currentPersona === 'roleplay' || currentPersona === 'gaming') ? `
                    <span style="color:var(--text-muted);opacity:0.2;">|</span>
                    <button onclick="_rpThumbsFeedback(this, 'thumbs_up')" class="p-0.5 rounded rp-feedback-btn" title="Good response" data-feedback="">
                        <i data-lucide="thumbs-up" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    <button onclick="_rpThumbsFeedback(this, 'thumbs_down')" class="p-0.5 rounded rp-feedback-btn" title="Bad response" data-feedback="">
                        <i data-lucide="thumbs-down" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    <button onclick="_rpEditMessage(this)" class="p-0.5 rounded" title="Edit response (RP)">
                        <i data-lucide="pencil" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    <button onclick="_rpRegenerateMessage(this)" class="p-0.5 rounded" title="Regenerate response">
                        <i data-lucide="refresh-cw" class="w-3 h-3" style="color:var(--text-muted);"></i>
                    </button>
                    ` : ''}
                </div>
            </div>`;
    }
    
    // Animate in
    wrapper.style.opacity = '0';
    wrapper.style.transform = 'translateY(8px)';
    msgs.appendChild(wrapper);
    lucide.createIcons();
    // Scroll BEFORE animation to prevent visual jump
    msgs.scrollTop = msgs.scrollHeight;
    requestAnimationFrame(() => {
        wrapper.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
        wrapper.style.opacity = '1';
        wrapper.style.transform = 'translateY(0)';
        // Re-scroll after animation starts to catch any layout shift
        msgs.scrollTop = msgs.scrollHeight;
    });
    // UX-01: Hyperlink signal titles in assistant messages
    if (role === 'assistant') {
        const respEl = wrapper.querySelector('.agent-response');
        if (respEl) _hyperlinkSignals(respEl);
    }
    // Fire hook for mobile agent sync (replaces polling)
    if (typeof _onAgentMessageHook === 'function') _onAgentMessageHook(role, content);
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
        btn.title = 'Read aloud';
        return;
    }

    const origIcon = btn.querySelector('[data-lucide]');
    if (origIcon) { origIcon.setAttribute('data-lucide', 'loader'); lucide.createIcons(); }
    btn.disabled = true;

    // Determine voice from persona + user preferences
    const persona = typeof _currentPersona !== 'undefined' ? _currentPersona : '';
    const voice = localStorage.getItem(`stratos_tts_voice_${persona}`)
                || localStorage.getItem('stratos_tts_voice')
                || null;
    const speed = parseFloat(localStorage.getItem('stratos_tts_speed') || '1.0');

    try {
        const resp = await fetch('/api/tts', {
            method: 'POST',
            headers: _agentHeaders(),
            body: JSON.stringify({ text: text.substring(0, 5000), voice, speed, persona })
        });
        if (!resp.ok) {
            const errBody = await resp.json().catch(() => ({}));
            throw new Error(errBody.error || 'TTS ' + resp.status);
        }

        // Read engine info from headers for tooltip
        const engine = resp.headers.get('X-TTS-Engine') || '';
        const processing = resp.headers.get('X-TTS-Processing') || '';

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        _currentTTSAudio = new Audio(url);

        _currentTTSAudio.onended = () => {
            if (origIcon) { origIcon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
            btn.disabled = false;
            btn.title = engine && processing ? `${engine} · ${processing}s` : 'Read aloud';
            URL.revokeObjectURL(url);
            _currentTTSAudio = null;
        };
        _currentTTSAudio.onerror = () => {
            if (origIcon) { origIcon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
            btn.disabled = false;
            btn.title = 'Read aloud';
            URL.revokeObjectURL(url);
            _currentTTSAudio = null;
        };

        if (origIcon) { origIcon.setAttribute('data-lucide', 'square'); lucide.createIcons(); }
        btn.disabled = false;
        btn.title = engine && processing ? `Playing · ${engine} · ${processing}s` : 'Playing...';
        _currentTTSAudio.preload = 'auto';
        await new Promise(resolve => {
            _currentTTSAudio.oncanplaythrough = resolve;
            _currentTTSAudio.load();
        });
        _currentTTSAudio.play();
    } catch (e) {
        console.error('TTS error:', e);
        if (origIcon) { origIcon.setAttribute('data-lucide', 'volume-2'); lucide.createIcons(); }
        btn.disabled = false;
        if (typeof showToast === 'function') {
            const msg = e.message && e.message.includes('Nothing to speak') ? e.message :
                        e.message && e.message.includes('503') ? 'TTS unavailable — no engine installed' :
                        e.message && e.message.includes('401') ? 'TTS failed — session expired, please refresh' :
                        'TTS failed — ' + (e.message || 'unknown error');
            showToast(msg, 'error');
        }
    }
}

function _copyAgentMessage(btn) {
    const resp = btn.closest('.agent-msg-actions')?.parentElement?.querySelector('.agent-response');
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

// ── UX-04: Edit user messages with response regeneration ──

function _editUserMessage(btn) {
    if (agentStreaming) return;
    const wrapper = btn.closest('.group\\/usermsg');
    if (!wrapper) return;
    const bubble = wrapper.querySelector('.agent-bubble-user');
    if (!bubble) return;
    const original = bubble.textContent.trim();

    // Find the index of this user message in the DOM
    const msgs = document.getElementById('agent-messages');
    if (!msgs) return;
    const allWrappers = Array.from(msgs.children);
    const wrapperIdx = allWrappers.indexOf(wrapper);

    // Replace bubble with editable textarea
    const editDiv = document.createElement('div');
    editDiv.className = 'w-full';
    editDiv.innerHTML = `
        <textarea class="w-full rounded-lg p-3 text-xs" style="background:rgba(59,130,246,0.08); color:var(--text-heading); border:1px solid rgba(59,130,246,0.3); min-height:60px; resize:vertical;">${escAgent(original)}</textarea>
        <div class="flex items-center justify-end gap-2 mt-1.5">
            <button class="agent-edit-cancel text-[10px] px-3 py-1 rounded" style="background:var(--bg-tertiary,rgba(255,255,255,0.03)); color:var(--text-muted);">Cancel</button>
            <button class="agent-edit-save text-[10px] px-3 py-1 rounded" style="background:rgba(59,130,246,0.2); color:#60a5fa; border:1px solid rgba(59,130,246,0.3);">Save & Resend</button>
        </div>`;

    const container = bubble.parentElement;
    const origBubbleHTML = bubble.outerHTML;
    const origActionsHTML = container.querySelector('.flex.items-center.justify-end')?.outerHTML || '';
    bubble.style.display = 'none';
    const actionsRow = container.querySelector('.flex.items-center.justify-end');
    if (actionsRow) actionsRow.style.display = 'none';
    container.appendChild(editDiv);

    const textarea = editDiv.querySelector('textarea');
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    // Cancel
    editDiv.querySelector('.agent-edit-cancel').addEventListener('click', function() {
        editDiv.remove();
        bubble.style.display = '';
        if (actionsRow) actionsRow.style.display = '';
    });

    // Save & Resend
    editDiv.querySelector('.agent-edit-save').addEventListener('click', function() {
        const newText = textarea.value.trim();
        if (!newText) return;

        // Restore bubble with new text
        bubble.textContent = newText;
        bubble.style.display = '';
        if (actionsRow) actionsRow.style.display = '';
        editDiv.remove();

        // Find the history index: count user messages up to this DOM position
        let historyIdx = -1;
        let userCount = 0;
        for (let i = 0; i < allWrappers.length; i++) {
            if (allWrappers[i].classList.contains('group/usermsg')) {
                if (i === wrapperIdx) {
                    historyIdx = userCount * 2; // user messages are at even indices in interleaved history
                    // Actually find by matching — more robust
                    break;
                }
                userCount++;
            }
        }

        // Find the corresponding index in agentHistory by scanning
        let foundIdx = -1;
        let userSeen = 0;
        for (let i = 0; i < agentHistory.length; i++) {
            if (agentHistory[i].role === 'user') {
                if (userSeen === userCount) {
                    foundIdx = i;
                    break;
                }
                userSeen++;
            }
        }

        if (foundIdx === -1) return;

        // Truncate agentHistory to the edit point (remove edited msg and everything after)
        agentHistory = agentHistory.slice(0, foundIdx);
        agentHistory.push({ role: 'user', content: newText });

        // Remove all DOM elements after the edited user message wrapper
        while (wrapper.nextSibling) {
            wrapper.nextSibling.remove();
        }

        // Re-send the edited message via the streaming path
        const input = document.getElementById('agent-input');
        if (input) input.value = '';

        // Trigger the send flow with the edited message already in history
        _resendFromEdit(newText);
    });
}

async function _resendFromEdit(msg) {
    if (agentStreaming) return;
    const input = document.getElementById('agent-input');
    const sendBtn = document.getElementById('agent-send-btn');

    // Show typing indicator
    const typingEl = appendAgentMessage('assistant', `<div class="flex items-center gap-2.5"><div class="agent-thinking-dots flex gap-1"><span></span><span></span><span></span></div><span class="text-[10px]" style="color:var(--text-muted);">Thinking...</span></div>`);

    agentStreaming = true;
    _agentAbortController = new AbortController();
    if (sendBtn) {
        sendBtn.disabled = false;
        if (input) input.disabled = true;
        sendBtn.style.opacity = '1';
        sendBtn.innerHTML = '<i data-lucide="square" class="w-4 h-4"></i>';
        sendBtn.title = 'Stop generating';
        sendBtn.onclick = function() { cancelAgentStream(); };
        lucide.createIcons();
    }

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
                ...(selectedPersonas.length > 1 ? { personas: selectedPersonas } : {}),
                ...(_agentFreeLength ? { free_length: true } : {}),
                ...(_agentAllScans ? { use_all_scans: true } : {}),
                ...(!_agentInjectSignals ? { inject_signals: false } : {}),
                ...(_rpDirectorNote ? { director_note: _rpDirectorNote } : {}),
                ...(currentPersona === 'gaming' && typeof _gamesGetState === 'function' ? {
                    rp_mode: _gamesGetState().rpMode,
                    active_npc: _gamesGetState().activeNpc,
                    active_scenario: _gamesGetState().activeScenario || ''
                } : {})
            })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        const respDiv = typingEl.querySelector('.agent-response');
        if (respDiv) respDiv.innerHTML = '';

        let _streamRenderPending = false;
        let _streamRenderTimer = null;
        const STREAM_RENDER_INTERVAL = 80;

        function _scheduleStreamRender() {
            if (_streamRenderPending) return;
            _streamRenderPending = true;
            _streamRenderTimer = setTimeout(() => {
                _streamRenderPending = false;
                if (respDiv) {
                    respDiv.innerHTML = formatAgentText(fullResponse);
                    const msgs = document.getElementById('agent-messages');
                    if (msgs) msgs.scrollTop = msgs.scrollHeight;
                }
            }, STREAM_RENDER_INTERVAL);
        }

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const payload = JSON.parse(line.slice(6));
                        if (payload.token) {
                            fullResponse += payload.token;
                            _scheduleStreamRender();
                        }
                        if (payload.status && respDiv) {
                            const isSearch = payload.status.includes('web_search');
                            const icon = isSearch ? '🔍' : '⚙️';
                            const label = payload.status.replace(/^[🔍⚙️]\s*/, '');
                            respDiv.innerHTML = `<div class="flex items-center gap-2.5 py-1"><div class="agent-thinking-dots flex gap-1"><span></span><span></span><span></span></div><span class="text-[10px] font-mono" style="color:var(--accent,#34d399);">${icon} ${escAgent(label)}</span></div>`;
                        }
                        if (payload.error) {
                            fullResponse = '\u26a0 ' + payload.error;
                            if (respDiv) respDiv.innerHTML = `<span class="text-amber-400">${escAgent(fullResponse)}</span>`;
                        }
                    } catch(e) {}
                }
            }
        }

        if (_streamRenderTimer) clearTimeout(_streamRenderTimer);

        fullResponse = fullResponse.replace(/[\u4e00-\u9fff\u3400-\u4dbf\u{20000}-\u{2a6df}]+/gu, '').trim();
        fullResponse = fullResponse.replace(/[^\S\n]{2,}/g, ' ').replace(/\.\s*\./g, '.').trim();
        fullResponse = fullResponse.replace(/\n{3,}/g, '\n\n');

        const finalDiv = typingEl?.querySelector('.agent-response');
        if (finalDiv) {
            finalDiv.innerHTML = wrapWithShowMore(fullResponse, formatAgentText(fullResponse));
            _hyperlinkSignals(finalDiv);
        }

        agentHistory.push({ role: 'assistant', content: fullResponse });

    } catch(e) {
        const respDiv = typingEl?.querySelector('.agent-response');
        const raw = e.message || 'Failed to reach agent';
        if (respDiv) {
            respDiv.innerHTML = `<div class="flex items-start gap-2 p-2 rounded-lg" style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);"><i data-lucide="alert-triangle" class="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5"></i><span class="text-amber-300 text-xs">${escAgent(raw)}</span></div>`;
            lucide.createIcons();
        }
        agentHistory.push({ role: 'assistant', content: raw });
    } finally {
        _saveAgentHistory();
        agentStreaming = false;
        _agentAbortController = null;
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.style.opacity = '1';
            sendBtn.innerHTML = '<i data-lucide="arrow-up" class="w-4 h-4"></i>';
            sendBtn.title = 'Send message';
            sendBtn.onclick = function() { sendAgentMessage(); };
            lucide.createIcons();
        }
        if (input) { input.disabled = false; input.focus(); }
        if (typeof _onAgentStreamEndHook === 'function') _onAgentStreamEndHook();
    }
}

// ── UX-05: Edit AI messages as learning signal ──

function _editAssistantMessage(btn) {
    if (agentStreaming) return;
    const actionsRow = btn.closest('.agent-msg-actions');
    const contentDiv = actionsRow?.parentElement?.querySelector('.agent-response');
    if (!contentDiv) return;
    const originalText = contentDiv.innerText || '';
    const wrapper = btn.closest('.group\\/msg');
    if (!wrapper) return;

    // Create inline edit UI
    const editArea = document.createElement('div');
    editArea.className = 'agent-ai-edit-area mt-2';
    editArea.innerHTML = `
        <textarea class="w-full rounded-lg p-3 text-xs" style="background:rgba(16,185,129,0.05); color:var(--text-body); border:1px solid rgba(16,185,129,0.25); min-height:100px; resize:vertical;">${escAgent(originalText)}</textarea>
        <div class="flex items-center gap-2 mt-1.5">
            <button class="agent-ai-edit-save text-[10px] px-3 py-1 rounded" style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3);">Save correction</button>
            <button class="agent-ai-edit-cancel text-[10px] px-3 py-1 rounded" style="background:var(--bg-tertiary,rgba(255,255,255,0.03)); color:var(--text-muted);">Cancel</button>
        </div>`;

    contentDiv.style.display = 'none';
    actionsRow.style.display = 'none';
    contentDiv.parentElement.appendChild(editArea);

    const textarea = editArea.querySelector('textarea');
    textarea.focus();

    // Cancel
    editArea.querySelector('.agent-ai-edit-cancel').addEventListener('click', function() {
        editArea.remove();
        contentDiv.style.display = '';
        actionsRow.style.display = '';
    });

    // Save
    editArea.querySelector('.agent-ai-edit-save').addEventListener('click', function() {
        const editedText = textarea.value.trim();
        if (!editedText || editedText === originalText) {
            editArea.remove();
            contentDiv.style.display = '';
            actionsRow.style.display = '';
            return;
        }

        // Update the displayed message
        contentDiv.innerHTML = formatAgentText(editedText);
        _hyperlinkSignals(contentDiv);
        contentDiv.style.display = '';
        actionsRow.style.display = '';
        editArea.remove();
        if (typeof lucide !== 'undefined') lucide.createIcons();

        // Find and update this assistant message in agentHistory
        const msgs = document.getElementById('agent-messages');
        if (msgs) {
            const allAssistantWrappers = Array.from(msgs.querySelectorAll('.group\\/msg'));
            const wrapperIdx = allAssistantWrappers.indexOf(wrapper);
            // Map to agentHistory index
            let assistantSeen = 0;
            for (let i = 0; i < agentHistory.length; i++) {
                if (agentHistory[i].role === 'assistant') {
                    if (assistantSeen === wrapperIdx) {
                        agentHistory[i].content = editedText;
                        break;
                    }
                    assistantSeen++;
                }
            }
        }

        // Inject a hidden context note so future messages benefit from the correction
        const correctionNote = `[The user corrected your previous response. Their preferred version: ${editedText.slice(0, 500)}. Adjust accordingly.]`;
        agentHistory.push({ role: 'user', content: correctionNote });
        // Immediately add a synthetic acknowledgment so history stays balanced
        agentHistory.push({ role: 'assistant', content: '(Noted — I\'ll adjust my responses accordingly.)' });

        // Save updated history
        _saveAgentHistory();

        // POST to /api/feedback (fire and forget)
        fetch('/api/feedback', {
            method: 'POST',
            headers: _agentHeaders(),
            body: JSON.stringify({
                action: 'edit',
                original: originalText,
                edited: editedText,
                persona: currentPersona,
                conversation_id: _agentActiveConvId || null
            })
        }).catch(function() {});

        if (typeof showToast === 'function') showToast('Correction saved', 'success');
    });
}

// ── RP Expansion: Feedback, Edit, Regenerate ──

function _rpThumbsFeedback(btn, type) {
    // Visual feedback
    const allBtns = btn.parentElement.querySelectorAll('.rp-feedback-btn');
    allBtns.forEach(b => {
        b.dataset.feedback = '';
        b.querySelector('[data-lucide]')?.setAttribute('style', 'color:var(--text-muted);');
    });
    btn.dataset.feedback = type;
    btn.querySelector('[data-lucide]')?.setAttribute('style',
        type === 'thumbs_up' ? 'color:#34d399;' : 'color:#f87171;');
    lucide.createIcons();

    // Send to backend (fire and forget)
    const msgId = btn.closest('.group\\/msg')?.dataset?.messageId || 0;
    fetch('/api/rp/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ..._agentHeaders() },
        body: JSON.stringify({ message_id: parseInt(msgId), feedback_type: type })
    }).catch(() => {});
}

function _rpRegenerateMessage(btn) {
    if (agentStreaming) return;
    // Find the last user message in history and resend it
    const lastUserIdx = agentHistory.findLastIndex(h => h.role === 'user');
    if (lastUserIdx < 0) return;
    const lastUserMsg = agentHistory[lastUserIdx].content;

    // Remove the last assistant message from history and DOM
    if (agentHistory.length > lastUserIdx + 1 && agentHistory[agentHistory.length - 1].role === 'assistant') {
        agentHistory.pop();
    }
    const msgWrapper = btn.closest('.group\\/msg');
    if (msgWrapper) msgWrapper.remove();

    // Re-send the last user message
    const input = document.getElementById('agent-input');
    if (input) {
        input.value = lastUserMsg;
        sendAgentMessage();
    }
}

function _rpEditMessage(btn) {
    const resp = btn.closest('.agent-msg-actions')?.parentElement?.querySelector('.agent-response');
    if (!resp) return;
    const original = resp.innerText || '';

    // Create inline edit UI
    const container = resp.parentElement;
    const editArea = document.createElement('div');
    editArea.className = 'rp-edit-area mt-2';
    editArea.innerHTML = `
        <textarea class="w-full rounded-lg p-3 text-xs" style="background:var(--bg-secondary); color:var(--text-body); border:1px solid var(--border-subtle); min-height:100px; resize:vertical;">${escAgent(original)}</textarea>
        <div class="flex items-center gap-2 mt-2">
            <select class="text-[10px] rounded px-2 py-1" style="background:var(--bg-tertiary); color:var(--text-body); border:1px solid var(--border-subtle);">
                <option value="">Reason (optional)</option>
                <option value="voice">Voice</option>
                <option value="length">Length</option>
                <option value="accuracy">Accuracy</option>
                <option value="tone">Tone</option>
                <option value="agency">Agency</option>
                <option value="other">Other</option>
            </select>
            <button onclick="_rpSaveEdit(this)" class="text-[10px] px-3 py-1 rounded" style="background:var(--accent-primary); color:#fff;">Save</button>
            <button onclick="this.closest('.rp-edit-area').remove()" class="text-[10px] px-3 py-1 rounded" style="background:var(--bg-tertiary); color:var(--text-muted);">Cancel</button>
        </div>`;
    container.appendChild(editArea);
    editArea.querySelector('textarea').focus();
}

function _rpSaveEdit(btn) {
    const editArea = btn.closest('.rp-edit-area');
    const textarea = editArea.querySelector('textarea');
    const reason = editArea.querySelector('select').value || undefined;
    const newContent = textarea.value.trim();
    const resp = editArea.parentElement.querySelector('.agent-response');
    const msgId = editArea.closest('.group\\/msg')?.dataset?.messageId || 0;

    if (!newContent) return;

    // Update display
    resp.innerHTML = formatAgentText(newContent);
    lucide.createIcons();
    editArea.remove();

    // Send to backend
    fetch('/api/rp/edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ..._agentHeaders() },
        body: JSON.stringify({ message_id: parseInt(msgId), edited_content: newContent, edit_reason: reason })
    }).catch(() => {});
}

// Director's Note panel (injected above chat input for RP personas)
let _rpDirectorNote = '';
let _rpLastUsedNote = '';

function _rpToggleDirector() {
    const panel = document.getElementById('rp-director-panel');
    if (panel) { panel.remove(); _rpUpdateSteerButton(); return; }

    const textarea = document.getElementById('agent-input');
    if (!textarea) return;
    const inputRow = textarea.closest('.px-3');
    if (!inputRow) return;

    const div = document.createElement('div');
    div.id = 'rp-director-panel';
    div.className = 'mx-3 mb-2 p-3 rounded-xl';
    div.style.cssText = 'background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.25);';
    div.innerHTML = `
        <div class="flex items-center gap-2 mb-2">
            <i data-lucide="sparkles" class="w-3.5 h-3.5" style="color:#fbbf24;"></i>
            <span class="text-[11px] font-semibold" style="color:#fbbf24;">Steer</span>
            <span class="text-[9px]" style="color:var(--text-muted);">Guide the AI's next response (one-shot)</span>
            <button onclick="_rpToggleDirector()" class="ml-auto p-1 rounded-md transition-all" style="color:var(--text-muted);" onmouseenter="this.style.color='#f87171'" onmouseleave="this.style.color='var(--text-muted)'">
                <i data-lucide="x" class="w-3 h-3"></i>
            </button>
        </div>
        <input type="text" id="rp-director-input" placeholder="e.g., Make the character more suspicious of the player..." maxlength="500"
            class="w-full text-[11px] rounded-lg px-3 py-2" style="background:var(--bg-tertiary); color:var(--text-body); border:1px solid rgba(251,191,36,0.3); outline:none;"
            onfocus="this.style.borderColor='#fbbf24';this.style.boxShadow='0 0 0 2px rgba(251,191,36,0.15)'"
            onblur="this.style.borderColor='rgba(251,191,36,0.3)';this.style.boxShadow='none'"
            value="${escAgent(_rpDirectorNote)}"
            oninput="_rpDirectorNote=this.value;_rpUpdateSteerButton();"
            onkeydown="if(event.key==='Enter'){event.preventDefault();document.getElementById('agent-input').focus();}">
        ${_rpLastUsedNote ? `<button onclick="_rpDirectorNote='${escAgent(_rpLastUsedNote)}';document.getElementById('rp-director-input').value=_rpDirectorNote;_rpUpdateSteerButton();" class="text-[9px] mt-1.5 px-2 py-0.5 rounded transition-all" style="color:#fbbf24;opacity:0.7;border:1px solid rgba(251,191,36,0.2);" onmouseenter="this.style.opacity='1'" onmouseleave="this.style.opacity='0.7'">&#x21a9; Reuse: ${escAgent(_rpLastUsedNote.slice(0, 40))}${_rpLastUsedNote.length > 40 ? '...' : ''}</button>` : ''}
        <div class="text-[9px] mt-1.5" style="color:var(--text-muted);opacity:0.6;">This note is consumed after your next message.</div>`;
    inputRow.parentElement.insertBefore(div, inputRow);
    lucide.createIcons();
    div.querySelector('#rp-director-input').focus();
    _rpUpdateSteerButton();
}

function _rpUpdateSteerButton() {
    const btn = document.getElementById('rp-director-btn');
    if (!btn) return;
    const active = !!_rpDirectorNote || !!document.getElementById('rp-director-panel');
    btn.style.borderColor = active ? 'rgba(251,191,36,0.5)' : 'var(--border-strong)';
    btn.style.color = active ? '#fbbf24' : 'var(--text-muted)';
    btn.style.background = active ? 'rgba(251,191,36,0.08)' : 'rgba(255,255,255,0.02)';
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

    // Extract URLs before escaping (markdown links + bare URLs)
    const urlPlaceholders = [];
    // Markdown links: [text](url)
    processed = processed.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => {
        const idx = urlPlaceholders.length;
        urlPlaceholders.push({ label, url });
        return `\x00URL_${idx}\x00`;
    });
    // Bare URLs: https://... or http://...
    processed = processed.replace(/(https?:\/\/[^\s<>\[\]()]+)/g, (_, url) => {
        // Clean trailing punctuation that's likely not part of URL
        let clean = url.replace(/[.,;:!?)]+$/, '');
        const trailing = url.slice(clean.length);
        const idx = urlPlaceholders.length;
        urlPlaceholders.push({ label: '', url: clean });
        return `\x00URL_${idx}\x00` + trailing;
    });

    // Extract blockquotes before escaping (> at line start = dialogue/emphasis)
    const blockquotes = [];
    processed = processed.replace(/^>\s*(.+)$/gm, (_, text) => {
        const idx = blockquotes.length;
        blockquotes.push(text);
        return `\x00BQUOTE_${idx}\x00`;
    });

    let html = processed
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong style="color:var(--text-heading,#f1f5f9);font-weight:600;">$1</strong>')
        // Italics (single * or _) — accent-tinted for narrative emphasis
        .replace(/(?<!\w)\*([^*\n]+?)\*(?!\w)/g, '<em style="color:var(--accent,#34d399);font-style:italic;opacity:0.85;">$1</em>')
        .replace(/(?<!\w)_([^_\n]+?)_(?!\w)/g, '<em style="color:var(--accent,#34d399);font-style:italic;opacity:0.85;">$1</em>')
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

    // Re-inject URL links
    urlPlaceholders.forEach((link, i) => {
        const display = link.label || link.url.replace(/^https?:\/\/(?:www\.)?/, '').slice(0, 50) + (link.url.length > 60 ? '...' : '');
        const safeUrl = link.url.replace(/"/g, '&quot;');
        html = html.replace(`\x00URL_${i}\x00`,
            `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-0.5" style="color:#60a5fa;text-decoration:underline;text-decoration-color:rgba(96,165,250,0.3);text-underline-offset:2px;" onmouseenter="this.style.color='#93bbfc';this.style.textDecorationColor='rgba(96,165,250,0.6)'" onmouseleave="this.style.color='#60a5fa';this.style.textDecorationColor='rgba(96,165,250,0.3)'">${escAgent(display)}</a>`);
    });

    // Re-inject fenced code blocks
    codeBlocks.forEach((block, i) => {
        const escaped = block.code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const langBadge = block.lang ? `<span class="absolute top-1.5 right-2 text-[9px] font-mono" style="color:var(--text-muted);opacity:0.5;">${block.lang}</span>` : '';
        html = html.replace(`\x00CODEBLOCK_${i}\x00`,
            `<div class="relative my-2 rounded-lg overflow-hidden" style="background:rgba(0,0,0,0.3);border:1px solid var(--border-strong);">${langBadge}<pre class="p-3 overflow-x-auto text-[11px] leading-relaxed font-mono" style="color:#e2e8f0;"><code>${escaped}</code></pre></div>`);
    });

    // Re-inject blockquotes (dialogue/emphasis)
    blockquotes.forEach((text, i) => {
        const formatted = text
            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong style="color:var(--text-heading);">$1</strong>')
            .replace(/(?<!\w)\*([^*\n]+?)\*(?!\w)/g, '<em style="color:var(--accent,#34d399);font-style:italic;">$1</em>');
        html = html.replace(`\x00BQUOTE_${i}\x00`,
            `<div class="pl-3 my-1" style="border-left:2px solid var(--accent,#34d399);color:var(--text-secondary,#94a3b8);font-style:italic;">${formatted}</div>`);
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

// ── UX-01: Hyperlink signal names in agent responses ──
// Post-render hook: scans assistant message HTML for signal titles present in newsData,
// wraps them with clickable links that navigate to the dashboard feed and scroll to the article.
function _hyperlinkSignals(containerEl) {
    if (!containerEl) return;
    if (typeof newsData === 'undefined' || !Array.isArray(newsData) || newsData.length === 0) return;
    // Build a list of titles long enough to be meaningful (>15 chars avoids false positives)
    const titles = newsData
        .map(n => (n.title || '').trim())
        .filter(t => t.length > 15);
    if (titles.length === 0) return;
    // Sort by length descending so longer titles match first
    titles.sort((a, b) => b.length - a.length);
    // Walk text nodes inside the container, skip nodes already inside <a> tags or code blocks
    const walker = document.createTreeWalker(containerEl, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            const parent = node.parentElement;
            if (!parent) return NodeFilter.FILTER_REJECT;
            const tag = parent.tagName;
            if (tag === 'A' || tag === 'CODE' || tag === 'PRE' || tag === 'TEXTAREA' || tag === 'SCRIPT')
                return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        }
    });
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    for (const node of textNodes) {
        let text = node.nodeValue;
        let matched = false;
        let fragments = null;
        for (const title of titles) {
            const idx = text.indexOf(title);
            if (idx === -1) continue;
            matched = true;
            fragments = document.createDocumentFragment();
            if (idx > 0) fragments.appendChild(document.createTextNode(text.slice(0, idx)));
            const link = document.createElement('a');
            link.textContent = title;
            link.href = '#';
            link.title = 'View in feed';
            link.style.cssText = 'color:#60a5fa;text-decoration:underline;text-decoration-color:rgba(96,165,250,0.3);text-underline-offset:2px;cursor:pointer;';
            link.addEventListener('click', function(e) {
                e.preventDefault();
                if (typeof setActive === 'function') setActive('dashboard');
                // Scroll to the article card by matching title text
                setTimeout(function() {
                    const cards = document.querySelectorAll('.news-card, .feed-card, [data-article-title]');
                    for (const card of cards) {
                        const cardTitle = card.getAttribute('data-article-title') || card.querySelector('.news-title, .feed-title, h3, h4')?.textContent || '';
                        if (cardTitle.trim() === title) {
                            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            card.style.outline = '2px solid #60a5fa';
                            card.style.outlineOffset = '2px';
                            setTimeout(function() { card.style.outline = ''; card.style.outlineOffset = ''; }, 3000);
                            break;
                        }
                    }
                }, 300);
            });
            fragments.appendChild(link);
            if (idx + title.length < text.length) fragments.appendChild(document.createTextNode(text.slice(idx + title.length)));
            break; // One match per text node to avoid complexity
        }
        if (matched && fragments) {
            node.parentNode.replaceChild(fragments, node);
        }
    }
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
var _agentShowMoreEnabled = localStorage.getItem('stratos_agent_showmore') !== 'false';
function toggleAgentShowMore() {
    _agentShowMoreEnabled = !_agentShowMoreEnabled;
    localStorage.setItem('stratos_agent_showmore', _agentShowMoreEnabled ? 'true' : 'false');
}
window.toggleAgentShowMore = toggleAgentShowMore;
function wrapWithShowMore(rawText, formattedHtml) {
    if (!_agentShowMoreEnabled || rawText.length <= 500) return formattedHtml;
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

// ── File upload for agent analysis ──
async function _agentFileSelected(fileInput) {
    const file = fileInput.files?.[0];
    if (!file) return;
    fileInput.value = ''; // Reset for re-upload

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        if (typeof showToast === 'function') showToast('File too large (max 10MB)', 'error');
        return;
    }

    const uploadBtn = document.getElementById('agent-upload-btn');
    if (uploadBtn) { uploadBtn.style.opacity = '0.5'; uploadBtn.style.pointerEvents = 'none'; }

    try {
        const token = typeof getAuthToken === 'function' ? getAuthToken() : '';
        const r = await fetch('/api/files/upload', {
            method: 'POST',
            headers: {
                'X-Auth-Token': token,
                'X-Filename': file.name,
                'X-Persona': currentPersona || 'intelligence',
                'Content-Type': 'application/octet-stream',
            },
            body: file,
        });
        const d = await r.json().catch(() => ({}));
        const fileId = d.file?.id || d.file_id;
        if (r.ok && fileId) {
            // Pre-fill agent input with analysis request
            const input = document.getElementById('agent-input');
            const isImage = /\.(png|jpg|jpeg|bmp|webp|gif)$/i.test(file.name);
            const isPdf = /\.pdf$/i.test(file.name);
            if (input) {
                if (isImage) {
                    input.value = `Analyze the image I just uploaded: "${file.name}" (file ID: ${fileId})`;
                } else if (isPdf) {
                    input.value = `Summarize the PDF I just uploaded: "${file.name}" (file ID: ${fileId})`;
                } else {
                    input.value = `Read the file I just uploaded: "${file.name}" (file ID: ${fileId})`;
                }
                sendAgentMessage();
            }
        } else {
            if (typeof showToast === 'function') showToast(d.error || 'Upload failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Upload failed', 'error');
    } finally {
        if (uploadBtn) { uploadBtn.style.opacity = '1'; uploadBtn.style.pointerEvents = 'auto'; }
    }
}
window._agentFileSelected = _agentFileSelected;

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
    
    // Auto-detect character name in gaming immersive mode
    if (typeof _gamesAutoDetectNpc === 'function') _gamesAutoDetectNpc(msg);

    agentHistory.push({ role: 'user', content: msg });
    appendAgentMessage('user', msg);

    // Show typing indicator — include steer badge if active
    const _dirBadge = _rpDirectorNote ? `<span class="text-[9px] ml-2 px-1.5 py-0.5 rounded" style="background:rgba(251,191,36,0.12);color:#fbbf24;border:1px solid rgba(251,191,36,0.25);">&#x2728; Steered</span>` : '';
    const typingEl = appendAgentMessage('assistant', `<div class="flex items-center gap-2.5"><div class="agent-thinking-dots flex gap-1"><span></span><span></span><span></span></div><span class="text-[10px]" style="color:var(--text-muted);">Thinking...</span>${_dirBadge}</div>`);

    // Consume director note (single-use)
    if (_rpDirectorNote) {
        _rpLastUsedNote = _rpDirectorNote;
        _rpDirectorNote = '';
        const _di = document.getElementById('rp-director-input');
        if (_di) _di.value = '';
        const _dp = document.getElementById('rp-director-panel');
        if (_dp) _dp.remove();
        if (typeof _rpUpdateSteerButton === 'function') _rpUpdateSteerButton();
    }

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
                ...(selectedPersonas.length > 1 ? { personas: selectedPersonas } : {}),
                ...(_agentFreeLength ? { free_length: true } : {}),
                ...(_agentAllScans ? { use_all_scans: true } : {}),
                ...(!_agentInjectSignals ? { inject_signals: false } : {}),
                ...(_rpDirectorNote ? { director_note: _rpDirectorNote } : {}),
                ...(currentPersona === 'gaming' && typeof _gamesGetState === 'function' ? {
                    rp_mode: _gamesGetState().rpMode,
                    active_npc: _gamesGetState().activeNpc,
                    active_scenario: _gamesGetState().activeScenario || ''
                } : {})
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

        // Throttle DOM updates during streaming for smooth rendering
        let _streamRenderPending = false;
        let _streamRenderTimer = null;
        const STREAM_RENDER_INTERVAL = 80; // ms between DOM updates

        function _scheduleStreamRender() {
            if (_streamRenderPending) return;
            _streamRenderPending = true;
            _streamRenderTimer = setTimeout(() => {
                _streamRenderPending = false;
                if (respDiv) {
                    respDiv.innerHTML = formatAgentText(fullResponse);
                    const msgs = document.getElementById('agent-messages');
                    if (msgs) msgs.scrollTop = msgs.scrollHeight;
                    if (typeof _onAgentStreamChunkHook === 'function') _onAgentStreamChunkHook();
                }
            }, STREAM_RENDER_INTERVAL);
        }

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
                            _scheduleStreamRender();
                        }
                        if (payload.suggestions && Array.isArray(payload.suggestions)) {
                            dynamicSuggestions = payload.suggestions;
                            _personaSuggestions[currentPersona] = payload.suggestions;
                            try { localStorage.setItem(_suggestionsKey(), JSON.stringify(_personaSuggestions)); } catch(e) {}
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

        // Cancel any pending throttled render — final render below replaces it
        if (_streamRenderTimer) clearTimeout(_streamRenderTimer);

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
            // UX-01: Hyperlink signal titles in the final rendered response
            _hyperlinkSignals(finalDiv);
            // Add suggestion chips — prefer LLM-generated, fallback to rule-based
            const chipSuggestions = dynamicSuggestions.length > 0
                ? dynamicSuggestions.map(s => {
                    // Support rich suggestions: {label, prompt} or plain strings
                    if (typeof s === 'object' && s.label && s.prompt) {
                        return { label: s.label, icon: 'sparkles', action: s.prompt, tip: s.prompt };
                    }
                    return { label: s, icon: 'sparkles', action: s, tip: s };
                })
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
        // Fire stream-end hook for mobile agent sync
        if (typeof _onAgentStreamEndHook === 'function') _onAgentStreamEndHook();
        // Refresh scenario list after gaming responses (catches tool-created scenarios)
        if (currentPersona === 'gaming' && typeof _loadScenarios === 'function') _loadScenarios();
    }
}

function cancelAgentStream() {
    if (_agentAbortController) {
        _agentAbortController.abort();
        _agentAbortController = null;
    }
}
window.cancelAgentStream = cancelAgentStream;

// Response chips moved to agent-suggestions.js


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
                            <span class="text-emerald-400 font-semibold">Context loaded: ${esc(fileName)}</span>
                        </div>
                        <div class="text-slate-400 whitespace-pre-wrap">${esc(preview)}</div>
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
        return `<button onclick="switchPersona('${key}')" class="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-lg text-left transition-all" style="background:${active ? t.bg : 'transparent'};color:${active ? t.color : 'var(--text-muted)'};" onmouseenter="if(!${active})this.style.background='rgba(255,255,255,0.03)'" onmouseleave="if(!${active})this.style.background='${active ? t.bg : 'transparent'}'">
            <i data-lucide="${t.icon}" class="w-4.5 h-4.5"></i>
            <span class="text-[14px] font-medium">${t.label}</span>
        </button>`;
    }).join('');

    return `
    <div class="agent-fs-sidebar" style="width:300px;min-width:300px;height:100%;display:flex;flex-direction:column;border-right:1px solid rgba(255,255,255,0.05);">
        <!-- New Chat button -->
        <div class="px-4 pt-4 pb-2">
            <button onclick="newAgentChat()" class="w-full flex items-center gap-2.5 px-4 py-3 rounded-xl text-[14px] font-medium transition-all" style="border:1px solid var(--border-strong);color:var(--text-heading);background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor='${theme.color}';this.style.background='${theme.bg}'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.background='rgba(255,255,255,0.02)'">
                <i data-lucide="plus" class="w-5 h-5"></i> New Chat
            </button>
        </div>
        <!-- Persona selector -->
        <div class="px-4 pb-2">
            <div class="text-[11px] font-bold uppercase tracking-wider mb-2 px-1" style="color:var(--text-muted);">Persona</div>
            <div class="space-y-0.5">${personaItems}</div>
        </div>
        <div class="mx-4 mb-2" style="height:1px;background:var(--border-strong);"></div>
        <!-- Conversation list -->
        <div class="flex-1 overflow-y-auto px-3" style="scrollbar-width:thin;">
            <div class="text-[11px] font-bold uppercase tracking-wider mb-2 px-1" style="color:var(--text-muted);">Chats</div>
            <div class="space-y-0.5">${convItems}</div>
        </div>
        <div class="mx-4 my-1" style="height:1px;background:var(--border-strong);"></div>
        <!-- Bottom actions -->
        <div class="px-4 pb-4 pt-2 space-y-1.5">
            <div class="flex items-center gap-1.5">
                <button onclick="importAgentChat()" class="flex-1 flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] transition-all" style="color:var(--text-muted);border:1px solid transparent;" title="Import chat" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="upload" class="w-4.5 h-4.5"></i> Import
                </button>
                <button onclick="exportAgentChat()" class="flex-1 flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] transition-all" style="color:var(--text-muted);border:1px solid transparent;" title="Export chat" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="download" class="w-4.5 h-4.5"></i> Export
                </button>
            </div>
            <div class="flex items-center gap-1.5">
                <button onclick="toggleContextEditor()" class="flex-1 flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all" style="color:var(--text-heading);border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);" title="Edit persona context — custom instructions" onmouseenter="this.style.background='rgba(16,185,129,0.08)';this.style.borderColor='rgba(52,211,153,0.3)';this.style.color='var(--accent)'" onmouseleave="this.style.background='rgba(255,255,255,0.03)';this.style.borderColor='rgba(255,255,255,0.1)';this.style.color='var(--text-heading)'">
                    <i data-lucide="file-cog" class="w-4.5 h-4.5"></i> Context
                </button>
                <button onclick="toggleFileBrowser()" class="flex-1 flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all" style="color:var(--text-heading);border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);" title="Browse persona files — uploads, notes" onmouseenter="this.style.background='rgba(16,185,129,0.08)';this.style.borderColor='rgba(52,211,153,0.3)';this.style.color='var(--accent)'" onmouseleave="this.style.background='rgba(255,255,255,0.03)';this.style.borderColor='rgba(255,255,255,0.1)';this.style.color='var(--text-heading)'">
                    <i data-lucide="folder-open" class="w-4.5 h-4.5"></i> Files
                </button>
            </div>
            <button onclick="clearAgentChat()" class="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] transition-all" style="color:var(--text-muted);" title="Clear current chat" onmouseenter="this.style.color='#f87171';this.style.background='rgba(239,68,68,0.06)'" onmouseleave="this.style.color='var(--text-muted)';this.style.background='transparent'">
                <i data-lucide="trash-2" class="w-4.5 h-4.5"></i> Clear Chat
            </button>
            <button onclick="_toggleFsCustomizer()" class="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-[13px] transition-all" style="color:var(--accent);" title="Customize fullscreen appearance" onmouseenter="this.style.background='rgba(255,255,255,0.04)'" onmouseleave="this.style.background='transparent'">
                <i data-lucide="palette" class="w-4.5 h-4.5"></i> Customize
            </button>
        </div>
    </div>`;
}

// Fullscreen Customizer moved to agent-customizer.js


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

// Ticker commands moved to agent-tickers.js


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

// ═══════════════════════════════════════════════
// COMPACT SIDEBAR (desktop agent panel)
// ═══════════════════════════════════════════════
let _compactSidebarOpen = false;

let _compactSidebarRemoveTimer = null;
function _toggleCompactSidebar() {
    _compactSidebarOpen = !_compactSidebarOpen;
    const existing = document.getElementById('agent-compact-sidebar');
    if (!_compactSidebarOpen && existing) {
        existing.style.transform = 'translateX(-100%)';
        existing.style.opacity = '0';
        if (_compactSidebarRemoveTimer) clearTimeout(_compactSidebarRemoveTimer);
        _compactSidebarRemoveTimer = setTimeout(() => { existing.remove(); _compactSidebarRemoveTimer = null; }, 200);
        return;
    }
    // Cancel pending removal if re-opening quickly
    if (_compactSidebarRemoveTimer) { clearTimeout(_compactSidebarRemoveTimer); _compactSidebarRemoveTimer = null; }
    if (_compactSidebarOpen) {
        // Ensure agent body is visible
        _openAgentPanel();
        if (!agentOpen) { agentOpen = true; }
        _renderCompactSidebar();
    }
}
window._toggleCompactSidebar = _toggleCompactSidebar;

function _renderCompactSidebar() {
    const body = document.getElementById('agent-body');
    if (!body) return;
    let sb = document.getElementById('agent-compact-sidebar');
    const theme = PERSONA_THEMES[currentPersona] || PERSONA_THEMES.intelligence;

    // Build conversation items
    const convItems = _agentConvList.map(c => {
        const active = c.id === _agentActiveConvId;
        const title = (c.title || 'New Chat').length > 22 ? (c.title || 'New Chat').slice(0, 20) + '…' : (c.title || 'New Chat');
        return `<div class="acs-conv-item${active ? ' acs-active' : ''}" onclick="_switchConversation(${c.id})" ondblclick="event.stopPropagation();_renameConversation(${c.id})" title="${escAgent(c.title || 'New Chat')}">
            <span class="acs-conv-title">${escAgent(title)}</span>
            ${_agentConvList.length > 1 ? `<span class="acs-conv-del" onclick="event.stopPropagation();_deleteConversation(${c.id})" title="Delete">&times;</span>` : ''}
        </div>`;
    }).join('');

    // Build persona items
    const personaItems = Object.entries(PERSONA_THEMES).map(([key, t]) => {
        const active = key === currentPersona;
        return `<button class="acs-persona-btn${active ? ' acs-persona-active' : ''}" onclick="switchPersona('${key}')" style="${active ? 'background:' + t.bg + ';color:' + t.color + ';border-color:' + t.color + '40' : ''}">
            ${t.label}
        </button>`;
    }).join('');

    const html = `
        <div class="acs-section">
            <div class="acs-heading" style="display:flex;align-items:center;justify-content:space-between;">Persona <button onclick="_showPersonaGuide()" title="Learn about personas" style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.25);color:var(--accent,#34d399);border-radius:4px;font-size:10px;cursor:pointer;display:flex;align-items:center;gap:3px;padding:2px 6px;line-height:1;" onmouseenter="this.style.borderColor='rgba(52,211,153,0.5)';this.style.boxShadow='0 0 8px rgba(52,211,153,0.2)'" onmouseleave="this.style.borderColor='rgba(52,211,153,0.25)';this.style.boxShadow='none'">Guide</button></div>
            <div class="acs-persona-list">${personaItems}</div>
        </div>
        <div class="acs-divider"></div>
        <div class="acs-section acs-flex-1">
            <div class="acs-heading">Chats</div>
            <div class="acs-conv-list">${convItems}</div>
        </div>
        <div class="acs-divider"></div>
        <div class="acs-actions">
            <button onclick="importAgentChat()" title="Import"><i data-lucide="upload" class="w-3.5 h-3.5"></i></button>
            <button onclick="exportAgentChat()" title="Export"><i data-lucide="download" class="w-3.5 h-3.5"></i></button>
            <button onclick="toggleContextEditor()" title="Context"><i data-lucide="file-cog" class="w-3.5 h-3.5"></i></button>
            <button onclick="toggleFileBrowser()" title="Files"><i data-lucide="folder-open" class="w-3.5 h-3.5"></i></button>
        </div>`;

    if (!sb) {
        sb = document.createElement('div');
        sb.id = 'agent-compact-sidebar';
        sb.className = 'agent-compact-sidebar';
        body.insertBefore(sb, body.firstChild);
        // Animate in
        sb.style.transform = 'translateX(-100%)';
        sb.style.opacity = '0';
        requestAnimationFrame(() => {
            sb.style.transition = 'transform 0.2s ease, opacity 0.2s ease';
            sb.style.transform = 'translateX(0)';
            sb.style.opacity = '1';
        });
    }
    sb.innerHTML = html;
    lucide.createIcons();
}

// Refresh compact sidebar when conversations change
function _refreshCompactSidebar() {
    if (_compactSidebarOpen) _renderCompactSidebar();
}

