// ═══════════════════════════════════════════════════════════
// STRAT AGENT — AI Chat over scraped intelligence data
// ═══════════════════════════════════════════════════════════

let agentHistory = [];   // {role:'user'|'assistant', content:string}
let agentOpen = false;
let agentStreaming = false;
let agentMode = 'structured'; // 'structured' or 'free'
let currentPersona = 'intelligence';
let availablePersonas = [];

// Persist agent chat to localStorage
function _saveAgentHistory() {
    try {
        // Keep last 40 messages to avoid localStorage bloat
        const toSave = agentHistory.slice(-40);
        localStorage.setItem('stratos_agent_history', JSON.stringify(toSave));
    } catch (e) { /* quota exceeded — silently skip */ }
}
function _restoreAgentHistory() {
    try {
        const saved = localStorage.getItem('stratos_agent_history');
        if (saved) {
            const parsed = JSON.parse(saved);
            if (Array.isArray(parsed) && parsed.length > 0) {
                agentHistory = parsed;
                _renderRestoredHistory();
            }
        }
    } catch (e) { /* corrupt data — start fresh */ }
}
function _renderRestoredHistory() {
    const msgs = document.getElementById('agent-messages');
    if (!msgs || agentHistory.length === 0) return;
    // Hide welcome, show messages
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

function toggleAgentMode() {
    agentMode = agentMode === 'structured' ? 'free' : 'structured';
    const btn = document.getElementById('agent-mode-btn');
    const input = document.getElementById('agent-input');
    if (agentMode === 'free') {
        if (btn) {
            btn.title = 'Free chat mode (no tools)';
            btn.style.background = 'rgba(99,102,241,0.1)';
            btn.style.color = '#818cf8';
            btn.style.borderColor = 'rgba(99,102,241,0.2)';
            btn.innerHTML = '<i data-lucide="message-circle" class="w-3.5 h-3.5"></i>';
        }
        if (input) input.placeholder = 'Free chat — no tools, just conversation...';
    } else {
        if (btn) {
            btn.title = 'Structured mode (tools enabled)';
            btn.style.background = 'rgba(16,185,129,0.1)';
            btn.style.color = 'var(--accent,#34d399)';
            btn.style.borderColor = 'rgba(16,185,129,0.2)';
            btn.innerHTML = '<i data-lucide="wrench" class="w-3.5 h-3.5"></i>';
        }
        if (input) input.placeholder = 'Ask anything, search the web, manage tickers...';
    }
    lucide.createIcons();
    if (typeof showToast === 'function') showToast(agentMode === 'free' ? 'Free chat mode' : 'Structured mode (tools)', 'info');
}

function switchPersona(name) {
    currentPersona = name;
    // Update subtitle text based on persona
    const subtitles = {
        intelligence: 'Search the web, manage your feed, analyze signals',
        market: 'Market data, price analysis, watchlist management',
        scholarly: 'History, language, philosophy, academic discussion',
        anime: 'Anime & manga tracking (coming soon)',
        tcg: 'Trading card games (coming soon)',
        gaming: 'Gaming news & deals (coming soon)',
    };
    const subtitle = document.querySelector('#agent-panel .text-\\[10px\\].mt-0\\.5');
    if (subtitle) subtitle.textContent = subtitles[name] || subtitles.intelligence;
    if (typeof showToast === 'function') showToast(`Switched to ${name} persona`, 'info');
}

async function loadPersonas() {
    try {
        const r = await fetch('/api/agent-personas', {
            headers: { 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' }
        });
        if (r.ok) {
            const d = await r.json();
            availablePersonas = d.personas || [];
            const select = document.getElementById('agent-persona-select');
            if (select && availablePersonas.length > 0) {
                select.innerHTML = '';
                for (const p of availablePersonas) {
                    const opt = document.createElement('option');
                    opt.value = p.name;
                    opt.textContent = p.name.charAt(0).toUpperCase() + p.name.slice(1);
                    if (p.name === currentPersona) opt.selected = true;
                    select.appendChild(opt);
                }
            }
        }
    } catch (e) { /* ignore — hardcoded options in HTML are fine as fallback */ }
}

// ── Clickable suggestion chips (dynamically generated from profile) ──
const _GENERIC_SUGGESTIONS = [
    "What are today's top stories?",
    "What's the most critical alert right now?",
    "Summarize today's news in 3 bullets",
    "What should I pay attention to today?",
    "How are the markets doing?",
    "Which assets are up and which are down?",
    "What's the single best signal for me?",
    "Anything I should be concerned about?",
    "What trends am I seeing this week?",
    "What are the top picks and why?",
];

function _buildDynamicSuggestions() {
    const suggestions = [..._GENERIC_SUGGESTIONS];
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
    const suggestions = _buildDynamicSuggestions();
    const shuffled = [...suggestions].sort(() => Math.random() - 0.5);
    const picks = shuffled.slice(0, 6);
    
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
        let icon = 'message-circle';
        for (const [key, val] of Object.entries(iconMap)) {
            if (lower.includes(key)) { icon = val; break; }
        }
        return `<button onclick="sendSuggestion(this)" class="text-[10px] px-2.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer whitespace-nowrap" style="border:1px solid var(--border-strong); color:var(--text-muted); background:rgba(255,255,255,0.02);" onmouseenter="this.style.borderColor='var(--accent,#34d399)';this.style.color='var(--accent,#34d399)';this.style.background='rgba(16,185,129,0.06)'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.color='var(--text-muted)';this.style.background='rgba(255,255,255,0.02)'"><i data-lucide="${icon}" class="w-3 h-3 opacity-60"></i>${s}</button>`;
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

function clearAgentChat() {
    agentHistory = [];
    _saveAgentHistory();
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
        wrapper.className = 'flex gap-3 mb-3';
        wrapper.innerHTML = `
            <div class="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5" style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.15);">
                <i data-lucide="bot" class="w-3.5 h-3.5 text-emerald-400"></i>
            </div>
            <div class="flex-1 min-w-0">
                <div class="agent-response text-xs leading-relaxed" style="color:var(--text-body,#cbd5e1);">${content}</div>
                <div class="text-[9px] mt-1" style="color:var(--text-muted); opacity:0.4;">${time}</div>
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

function escAgent(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Markdown-style formatting for agent responses with smart color highlighting
function formatAgentText(text) {
    let html = text
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
    return html;
}

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
    
    // Show typing indicator
    const typingEl = appendAgentMessage('assistant', `<div class="flex items-center gap-2"><div class="flex gap-1"><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:var(--accent,#34d399);animation-delay:0ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:var(--accent,#34d399);animation-delay:150ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:var(--accent,#34d399);animation-delay:300ms;"></span></div><span class="text-[10px]" style="color:var(--text-muted);">Thinking...</span></div>`);
    
    agentStreaming = true;
    sendBtn.disabled = true;
    input.disabled = true;
    sendBtn.style.opacity = '0.5';
    sendBtn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>';
    lucide.createIcons();
    
    try {
        const response = await fetch('/api/agent-chat', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
                'X-Device-Id': typeof getDeviceId === 'function' ? getDeviceId() : ''
            },
            body: JSON.stringify({
                message: msg,
                history: agentHistory.slice(-20),
                mode: agentMode,
                persona: currentPersona
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
                        if (payload.status) {
                            // Tool usage indicator — animated status bar
                            if (respDiv) {
                                const isSearch = payload.status.includes('web_search');
                                const icon = isSearch ? '🔍' : '⚙️';
                                const label = payload.status.replace(/^[🔍⚙️]\s*/, '');
                                respDiv.innerHTML = `<div class="flex items-center gap-2 py-1"><div class="flex gap-1"><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:var(--accent);animation-delay:0ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:var(--accent);animation-delay:150ms;"></span><span class="w-1.5 h-1.5 rounded-full animate-bounce" style="background:var(--accent);animation-delay:300ms;"></span></div><span class="text-[10px] font-mono" style="color:var(--accent,#34d399);">${icon} ${escAgent(label)}</span></div>`;
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
            // F2: Add suggestion chips based on response content
            const chips = _generateResponseChips(fullResponse, msg);
            if (chips.length) {
                const chipHtml = chips.map(c =>
                    `<button onclick="${escAgent(c.action)}" class="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all hover:scale-105" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);color:var(--accent,#34d399);" title="${escAgent(c.tip || '')}">
                        <i data-lucide="${c.icon}" class="w-3 h-3"></i> ${escAgent(c.label)}
                    </button>`
                ).join('');
                finalDiv.insertAdjacentHTML('afterend',
                    `<div class="flex flex-wrap gap-1.5 mt-2 agent-chips">${chipHtml}</div>`);
                lucide.createIcons();
            }
        }

        agentHistory.push({ role: 'assistant', content: fullResponse });
        
    } catch(e) {
        const respDiv = typingEl?.querySelector('.agent-response');
        const errMsg = e.message || 'Failed to reach agent';
        if (respDiv) {
            respDiv.innerHTML = `<span class="text-amber-400">⚠ ${escAgent(errMsg)}</span>`;
        }
        agentHistory.push({ role: 'assistant', content: errMsg });
    } finally {
        _saveAgentHistory();
        agentStreaming = false;
        sendBtn.disabled = false;
        input.disabled = false;
        sendBtn.style.opacity = '1';
        sendBtn.innerHTML = '<i data-lucide="arrow-up" class="w-4 h-4"></i>';
        lucide.createIcons();
        input.focus();
    }
}

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
function toggleAgentFullscreen() {
    const panel = document.getElementById('agent-panel');
    const msgs = document.getElementById('agent-messages');
    const btn = document.getElementById('agent-fs-btn');
    if (!panel) return;

    _agentFullscreen = !_agentFullscreen;

    if (_agentFullscreen) {
        // Open the chat body if collapsed
        _openAgentPanel();
        panel.classList.add('agent-fullscreen');
        if (msgs) { msgs.style.height = 'calc(100vh - 180px)'; msgs.style.maxHeight = 'none'; }
        if (btn) btn.title = 'Exit fullscreen';
        // Swap icon
        const icon = btn?.querySelector('[data-lucide]');
        if (icon) { icon.setAttribute('data-lucide', 'minimize-2'); lucide.createIcons(); }
    } else {
        panel.classList.remove('agent-fullscreen');
        if (msgs) { msgs.style.height = '280px'; msgs.style.maxHeight = '600px'; }
        if (btn) btn.title = 'Toggle fullscreen';
        const icon = btn?.querySelector('[data-lucide]');
        if (icon) { icon.setAttribute('data-lucide', 'maximize-2'); lucide.createIcons(); }
    }
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
