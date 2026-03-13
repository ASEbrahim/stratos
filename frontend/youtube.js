// ═══════════════════════════════════════════════════════════
// YOUTUBE CHANNEL MANAGEMENT & INSIGHTS VIEWER
// ═══════════════════════════════════════════════════════════

let _ytChannels = [];
let _ytVideos = {};
let _ytInsights = {};
let _ytInsightsAll = {};      // keyed by `${lens_name}_${language}`
let _ytAvailableLangs = [];
let _ytCurrentLang = localStorage.getItem('stratos_insight_lang') || 'en';

const _ytLangLabels = {
    'ar': 'العربية', 'ja': '日本語', 'ko': '한국어', 'zh': '中文',
    'fr': 'Français', 'de': 'Deutsch', 'es': 'Español', 'ru': 'Русский',
};

// ── SSE handler for realtime video processing updates ──
function _handleYouTubeSSE(event) {
    const { video_id, title, status, error, insights_count, transcript_method } = event;
    if (!video_id) return;

    // Find and update the video status in any visible video list
    const statusEl = document.querySelector(`[data-yt-video="${video_id}"]`);
    if (statusEl) {
        const statusColor = {
            'complete': 'text-emerald-400', 'transcribing': 'text-blue-400',
            'extracting': 'text-purple-400', 'failed': 'text-red-400',
            'started': 'text-amber-400'
        }[status] || 'text-slate-500';
        const statusIcon = {
            'complete': 'check-circle', 'transcribing': 'mic',
            'extracting': 'sparkles', 'failed': 'alert-circle',
            'started': 'loader-2'
        }[status] || 'clock';
        const iconEl = statusEl.querySelector('[data-lucide]');
        const textEl = statusEl.querySelector('.yt-video-status');
        if (iconEl) {
            iconEl.setAttribute('data-lucide', statusIcon);
            iconEl.className = `w-3 h-3 ${statusColor} flex-shrink-0${status === 'started' || status === 'transcribing' || status === 'extracting' ? ' animate-spin' : ''}`;
        }
        if (textEl) textEl.textContent = status;
        // Make clickable when complete
        if (status === 'complete') {
            const dbId = statusEl.dataset.ytDbId;
            if (dbId) {
                statusEl.style.cursor = 'pointer';
                statusEl.onclick = () => _ytShowInsights(parseInt(dbId));
            }
        }
        lucide.createIcons();
    }

    // Show toast for key transitions
    const shortTitle = (title || video_id).substring(0, 40);
    if (status === 'transcribing') {
        if (typeof showToast === 'function') showToast(`Transcribing: ${shortTitle}...`, 'info');
    } else if (status === 'complete') {
        if (typeof showToast === 'function') showToast(`✓ ${shortTitle} — ${insights_count || 0} insights extracted`, 'success');
    } else if (status === 'failed') {
        if (typeof showToast === 'function') showToast(`✗ ${shortTitle} failed: ${error || 'unknown'}`, 'error');
    }
}

function _ytHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

// ── Initialize YouTube panel in Settings ──
function initYouTubePanel() {
    const container = document.getElementById('youtube-settings-panel') || document.getElementById('youtube-panel-content');
    if (!container) return;
    _ytLoadChannels();
}
window.initYouTubePanel = initYouTubePanel;

// ── Load channels ──
async function _ytLoadChannels() {
    const container = document.getElementById('youtube-channel-list');
    if (!container) return;
    container.innerHTML = '<div class="text-[10px] text-center py-4" style="color:var(--text-muted)">Loading channels...</div>';

    try {
        const r = await fetch('/api/youtube/channels', { headers: _ytHeaders() });
        if (r.ok) {
            const d = await r.json();
            _ytChannels = d.channels || [];
            _ytRenderChannels();
        } else {
            container.innerHTML = '<div class="text-[10px] text-center py-4" style="color:var(--text-muted)">No channels tracked yet</div>';
        }
    } catch (e) {
        container.innerHTML = '<div class="text-[10px] text-center py-4 text-red-400">Failed to load channels</div>';
    }
}

function _ytRenderChannels() {
    const container = document.getElementById('youtube-channel-list');
    if (!container) return;

    if (_ytChannels.length === 0) {
        container.innerHTML = '<div class="text-[10px] text-center py-4" style="color:var(--text-muted)">No channels tracked yet. Add one above.</div>';
        return;
    }

    container.innerHTML = _ytChannels.map(ch => {
        const lensArr = typeof ch.lenses === 'string' ? (function(){ try { return JSON.parse(ch.lenses); } catch(e) { return []; } })() : (ch.lenses || []);
        const lenses = lensArr.join(', ') || 'all';
        const channelName = ch.channel_name || ch.name || ch.channel_id;
        const videoCount = ch.video_count || 0;
        return `<div class="yt-channel-card" data-channel-id="${ch.id}" onclick="_ytToggleVideos(${ch.id})" style="cursor:pointer">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-2 min-w-0">
                    <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);">
                        <i data-lucide="youtube" class="w-4 h-4 text-red-400"></i>
                    </div>
                    <div class="min-w-0">
                        <div class="text-[11px] font-semibold truncate" style="color:var(--text-heading)">${_escHtml(channelName)}</div>
                        <div class="text-[9px]" style="color:var(--text-muted)">${videoCount} videos · Lenses: ${lenses}</div>
                    </div>
                </div>
                <div class="flex items-center gap-1 flex-shrink-0">
                    <button onclick="event.stopPropagation();_ytProcessChannel(${ch.id})" class="fb-tool-btn" title="Process new videos">
                        <i data-lucide="play" class="w-3 h-3"></i>
                    </button>
                    <button onclick="event.stopPropagation();_ytToggleVideos(${ch.id})" class="fb-tool-btn" title="Show videos">
                        <i data-lucide="list" class="w-3 h-3"></i>
                    </button>
                    <button onclick="event.stopPropagation();_ytDeleteChannel(${ch.id}, '${_escAttr(channelName)}')" class="fb-tool-btn" title="Remove channel" onmouseenter="this.style.color='#f87171';this.style.borderColor='rgba(239,68,68,0.3)'" onmouseleave="this.style.color='var(--text-muted)';this.style.borderColor='var(--border-strong)'">
                        <i data-lucide="trash-2" class="w-3 h-3"></i>
                    </button>
                </div>
            </div>
            <div id="yt-videos-${ch.id}" class="hidden mt-2 space-y-1"></div>
        </div>`;
    }).join('');
    lucide.createIcons();
}

// ── Add channel ──
async function _ytAddChannel() {
    const input = document.getElementById('yt-channel-url');
    const url = input?.value?.trim();
    if (!url) return;

    // Get selected lenses
    const lenses = [];
    document.querySelectorAll('#yt-lens-checkboxes input:checked').forEach(cb => {
        lenses.push(cb.value);
    });

    try {
        const r = await fetch('/api/youtube/channels', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ channel_url: url, lenses: lenses.length ? lenses : undefined })
        });
        if (r.ok) {
            const d = await r.json();
            input.value = '';
            _ytLoadChannels();
            if (d.type === 'video') {
                if (typeof showToast === 'function') showToast(`Video queued: ${d.video?.title || 'Processing...'}`, 'success');
            } else {
                if (typeof showToast === 'function') showToast('Channel added', 'success');
            }
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Failed to add', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to add', 'error');
    }
}
window._ytAddChannel = _ytAddChannel;

// ── Delete channel ──
async function _ytDeleteChannel(id, name) {
    if (!(await stratosConfirm(`Remove channel "${name}"?`, { title: 'Remove Channel', okText: 'Remove', cancelText: 'Cancel' }))) return;
    try {
        await fetch(`/api/youtube/channels/${id}`, {
            method: 'DELETE',
            headers: _ytHeaders()
        });
        _ytLoadChannels();
        if (typeof showToast === 'function') showToast(`Removed ${name}`, 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to remove channel', 'error');
    }
}

// ── Process channel videos ──
async function _ytProcessChannel(channelId) {
    try {
        const r = await fetch(`/api/youtube/process/${channelId}`, {
            method: 'POST',
            headers: _ytHeaders()
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast('Processing queued', 'success');
        } else {
            if (typeof showToast === 'function') showToast('Failed to queue processing', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to queue processing', 'error');
    }
}

// ── Toggle video list for a channel ──
async function _ytToggleVideos(channelId) {
    const el = document.getElementById(`yt-videos-${channelId}`);
    if (!el) return;

    if (!el.classList.contains('hidden')) {
        el.classList.add('hidden');
        return;
    }

    el.classList.remove('hidden');
    el.innerHTML = '<div class="text-[9px] py-2" style="color:var(--text-muted)">Loading videos...</div>';

    try {
        const r = await fetch(`/api/youtube/videos/${channelId}`, { headers: _ytHeaders() });
        if (r.ok) {
            const d = await r.json();
            const videos = d.videos || [];
            if (videos.length === 0) {
                el.innerHTML = '<div class="text-[9px] py-2" style="color:var(--text-muted)">No videos yet. Click play to process.</div>';
                return;
            }
            el.innerHTML = videos.map(v => {
                const statusColor = {
                    'complete': 'text-emerald-400', 'processing': 'text-amber-400',
                    'pending': 'text-slate-500', 'failed': 'text-red-400',
                    'transcribing': 'text-blue-400', 'extracting': 'text-purple-400'
                }[v.status] || 'text-slate-500';
                const statusIcon = {
                    'complete': 'check-circle', 'processing': 'loader-2',
                    'pending': 'clock', 'failed': 'alert-circle',
                    'transcribing': 'mic', 'extracting': 'sparkles'
                }[v.status] || 'clock';
                const clickable = v.status === 'complete' ? `onclick="_ytShowInsights(${v.id})" class="cursor-pointer"` : '';

                return `<div data-yt-video="${v.video_id}" data-yt-db-id="${v.id}" ${clickable} class="flex items-center gap-2 px-2 py-1.5 rounded-md transition-colors" onmouseenter="this.style.background='var(--bg-hover)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="${statusIcon}" class="w-3 h-3 ${statusColor} flex-shrink-0${v.status === 'transcribing' || v.status === 'extracting' ? ' animate-spin' : ''}"></i>
                    <span class="text-[10px] flex-1 truncate" style="color:var(--text-secondary)">${_escHtml(v.title || v.video_id)}</span>
                    <span class="text-[8px] yt-video-status ${statusColor}">${v.status}</span>
                </div>`;
            }).join('');
            lucide.createIcons();
        }
    } catch (e) {
        el.innerHTML = '<div class="text-[9px] py-2 text-red-400">Failed to load videos</div>';
    }
}

// ═══════════════════════════════════════════════════════════
// YOUTUBE INSIGHTS VIEWER — Arcane Modal with Star Parallax
// ═══════════════════════════════════════════════════════════

let _ytCurrentVideoId = null;
let _ytStarRAF = null;
let _ytShowingGuide = false;
let _ytEloquenceFilter = 'all';

const _ytLensIcons = { summary: 'file-text', eloquence: 'pen-tool', narrations: 'book-open', history: 'landmark', spiritual: 'heart', politics: 'flag', transcript: 'scroll-text' };

async function _ytShowInsights(videoId) {
    _ytCurrentVideoId = videoId;
    _ytShowingGuide = false;
    let modal = document.getElementById('yt-insights-modal');
    if (modal) modal.remove();

    modal = document.createElement('div');
    modal.id = 'yt-insights-modal';
    modal.innerHTML = `
        <div class="yi-overlay" onclick="if(event.target===this)_ytCloseInsights()">
            <div class="yi-container">
                <canvas class="yi-stars" id="yi-star-canvas"></canvas>
                <div class="yi-titlebar">
                    <div style="width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0;background:rgba(192,132,252,0.1);border:1px solid rgba(192,132,252,0.2);">
                        <i data-lucide="sparkles" style="width:18px;height:18px;color:#c084fc;"></i>
                    </div>
                    <div style="flex:1;min-width:0;">
                        <div class="yi-title" id="yi-title">Video Insights</div>
                        <div class="yi-subtitle" id="yi-subtitle"></div>
                    </div>
                    <div class="yi-lang-area" id="yi-lang-area"></div>
                    <button class="yi-header-btn" onclick="_ytShowLensGuide()" title="Lens Guide"><i data-lucide="book-open" style="width:14px;height:14px;"></i> Guide</button>
                    <button class="yi-header-btn" onclick="_ytAskAgent()" title="Ask Strat Agent"><i data-lucide="bot" style="width:14px;height:14px;"></i> Ask Agent</button>
                    <button class="yi-close" onclick="_ytCloseInsights()" title="Close">&times;</button>
                </div>
                <div class="yi-tabs" id="yi-tabs"></div>
                <div class="yi-body" id="yi-body">
                    <div class="yi-loading"><div style="display:flex;gap:6px;"><span class="w-2.5 h-2.5 rounded-full animate-bounce" style="background:#c084fc;animation-delay:0ms;"></span><span class="w-2.5 h-2.5 rounded-full animate-bounce" style="background:#c084fc;animation-delay:150ms;"></span><span class="w-2.5 h-2.5 rounded-full animate-bounce" style="background:#c084fc;animation-delay:300ms;"></span></div></div>
                </div>
            </div>
        </div>`;
    document.body.appendChild(modal);
    lucide.createIcons();
    _ytInitStars();

    try {
        const r = await fetch(`/api/youtube/insights/${videoId}?language=all`, { headers: _ytHeaders() });
        if (!r.ok) throw new Error('Failed to fetch');
        const d = await r.json();
        const allInsights = d.insights || [];
        const videoTitle = d.video_title || d.title || '';
        _ytAvailableLangs = d.available_languages || ['en'];

        _ytInsightsAll = {};
        for (const ins of allInsights) {
            _ytInsightsAll[`${ins.lens_name}_${ins.language || 'en'}`] = ins;
        }

        if (!_ytAvailableLangs.includes(_ytCurrentLang)) _ytCurrentLang = 'en';

        const insights = allInsights.filter(ins => (ins.language || 'en') === _ytCurrentLang);
        const seenLenses = new Set();
        const uniqueInsights = [];
        for (const ins of insights) {
            if (!seenLenses.has(ins.lens_name)) { seenLenses.add(ins.lens_name); uniqueInsights.push(ins); }
        }

        const titleEl = document.getElementById('yi-title');
        const subEl = document.getElementById('yi-subtitle');
        if (titleEl && videoTitle) titleEl.textContent = videoTitle;
        if (subEl) subEl.textContent = `${uniqueInsights.length} lens${uniqueInsights.length !== 1 ? 'es' : ''} extracted`;

        // Language toggle
        _ytRenderLangToggle();

        if (uniqueInsights.length === 0) {
            const body = document.getElementById('yi-body');
            if (body) body.innerHTML = `<div class="yi-empty"><div class="yi-empty-icon"><i data-lucide="search-x" style="width:48px;height:48px;color:var(--text-muted);opacity:0.3;"></i></div><div style="font-size:13px;margin-bottom:4px;">No insights extracted yet</div><div style="font-size:11px;opacity:0.6;">Video may still be processing</div></div>`;
            lucide.createIcons();
            return;
        }

        const tabs = document.getElementById('yi-tabs');
        if (tabs) {
            tabs.innerHTML = uniqueInsights.map((ins, i) => {
                const lens = ins.lens_name || 'unknown';
                const lensName = lens.charAt(0).toUpperCase() + lens.slice(1);
                const icon = _ytLensIcons[lens] || 'sparkles';
                return `<button onclick="_ytShowLens(${i})" data-lens="${lens}" class="yi-tab${i === 0 ? ' yi-tab-active' : ''}"><i data-lucide="${icon}" style="width:14px;height:14px;"></i>${lensName}</button>`;
            }).join('');
            lucide.createIcons();
        }

        _ytInsights = uniqueInsights;
        _ytRenderLens(0);
    } catch (e) {
        const body = document.getElementById('yi-body');
        if (body) body.innerHTML = '<div class="yi-empty" style="color:#f87171;">Failed to load insights</div>';
    }
}

function _ytRenderLangToggle() {
    const area = document.getElementById('yi-lang-area');
    if (!area) return;
    if (_ytAvailableLangs.length <= 1) { area.innerHTML = ''; return; }
    area.innerHTML = `<div class="yt-lang-toggle">${_ytAvailableLangs.map(lang => {
        const label = lang === 'en' ? 'EN' : (_ytLangLabels[lang] || lang.toUpperCase());
        return `<button class="yt-lang-btn${_ytCurrentLang === lang ? ' active' : ''}" data-lang="${lang}" onclick="_ytSwitchLang('${lang}')">${_escHtml(label)}</button>`;
    }).join('')}</div>`;
}

function _ytAskAgent() {
    const titleEl = document.getElementById('yi-title');
    const title = titleEl?.textContent || 'this video';
    _ytCloseInsights();
    if (typeof _agentFullscreen !== 'undefined' && !_agentFullscreen && typeof toggleAgentFullscreen === 'function') {
        toggleAgentFullscreen();
    } else if (typeof _openAgentPanel === 'function') {
        _openAgentPanel();
    }
    const input = document.getElementById('agent-input');
    if (input) { input.value = `Tell me about the video "${title}" — summarize the key insights`; input.focus(); }
}
window._ytAskAgent = _ytAskAgent;

function _ytShowLens(index) {
    _ytShowingGuide = false;
    document.querySelectorAll('.yi-tab').forEach((tab, i) => {
        tab.classList.toggle('yi-tab-active', i === index);
    });
    _ytRenderLens(index);
}

function _ytRenderLens(index) {
    const body = document.getElementById('yi-body');
    if (!body || !_ytInsights[index]) return;

    const baseLens = _ytInsights[index].lens_name || 'unknown';
    const langKey = `${baseLens}_${_ytCurrentLang}`;
    const fallbackKey = `${baseLens}_en`;
    const ins = _ytInsightsAll[langKey] || _ytInsightsAll[fallbackKey] || _ytInsights[index];

    const lens = ins.lens_name || 'unknown';
    let data;
    const raw = ins.content || ins.data;
    try { data = typeof raw === 'string' ? JSON.parse(raw) : raw; } catch (e) { data = raw; }

    if (lens === 'summary') body.innerHTML = _ytRenderSummary(data);
    else if (lens === 'eloquence') body.innerHTML = _ytRenderEloquence(data);
    else if (lens === 'narrations') body.innerHTML = _ytRenderNarrations(data);
    else if (lens === 'transcript') body.innerHTML = _ytRenderTranscript(data);
    else if (lens === 'history') body.innerHTML = _ytRenderHistory(data);
    else if (lens === 'spiritual') body.innerHTML = _ytRenderSpiritual(data);
    else if (lens === 'politics') body.innerHTML = _ytRenderPolitics(data);
    else body.innerHTML = _ytRenderGeneric(data, lens);

    lucide.createIcons();
}

function _ytSwitchLang(lang) {
    _ytCurrentLang = lang;
    localStorage.setItem('stratos_insight_lang', lang);
    document.querySelectorAll('.yt-lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    if (_ytShowingGuide) return; // Don't re-render if on guide page
    const activeTab = document.querySelector('.yi-tab.yi-tab-active');
    if (activeTab) {
        const idx = Array.from(document.querySelectorAll('.yi-tab')).indexOf(activeTab);
        if (idx >= 0) _ytRenderLens(idx);
    }
}
window._ytSwitchLang = _ytSwitchLang;

// ── Renderers ──

function _ytRenderTranscript(data) {
    if (!data) return '<div class="yi-empty">No transcript data</div>';
    const text = typeof data === 'string' ? data : (data.transcript || JSON.stringify(data));
    const note = data.note ? `<div class="yi-note">${_escHtml(data.note)}</div>` : '';
    const sentences = text.split(/(?<=[.!?。！？])\s+/);
    const paragraphs = [];
    for (let i = 0; i < sentences.length; i += 4) paragraphs.push(sentences.slice(i, i + 4).join(' '));
    return `${note}<div class="yi-transcript">${paragraphs.map(p => `<p>${_escHtml(p)}</p>`).join('')}</div>`;
}

function _ytRenderSummary(data) {
    if (!data) return '<div class="yi-empty">No summary data</div>';
    // Handle {summary, key_takeaways} format
    if (data.summary || data.key_takeaways) {
        let html = '';
        if (data.summary) {
            html += `<div class="yi-card"><div class="yi-summary-text">${_escHtml(data.summary)}</div></div>`;
        }
        const takeaways = data.key_takeaways || data.takeaways || [];
        if (takeaways.length > 0) {
            html += `<div class="yi-takeaways-label"><i data-lucide="list-checks" style="width:14px;height:14px;"></i> Key Takeaways</div>`;
            html += takeaways.map((t, i) => `<div class="yi-takeaway"><div class="yi-takeaway-num">${i + 1}</div><div class="yi-takeaway-text">${_escHtml(typeof t === 'string' ? t : (t.text || t.point || JSON.stringify(t)))}</div></div>`).join('');
        }
        return html || '<div class="yi-empty">No summary content</div>';
    }
    // Fallback: array of items
    const items = Array.isArray(data) ? data : [data];
    return items.map(item => {
        const text = typeof item === 'string' ? item : (item.takeaway || item.point || item.text || Object.values(item).filter(v => typeof v === 'string').join(' — ') || 'No content');
        return `<div class="yi-card"><div class="yi-summary-text">${_escHtml(text)}</div></div>`;
    }).join('');
}

function _ytRenderEloquence(data) {
    if (!data) return '<div class="yi-empty">No eloquence data</div>';
    const terms = Array.isArray(data) ? data : (data.terms || []);
    if (terms.length === 0) return '<div class="yi-empty">No vocabulary extracted</div>';
    const filtered = _ytEloquenceFilter === 'all' ? terms : terms.filter(t => (t.rarity || 'uncommon') === _ytEloquenceFilter);
    const filterBar = `<div style="display:flex;gap:6px;margin-bottom:14px;">
        ${['all','uncommon','rare'].map(f => `<button onclick="_ytSetEloquenceFilter('${f}')" class="yi-header-btn" style="${_ytEloquenceFilter===f?'background:rgba(52,211,153,0.15);color:#34d399;border-color:rgba(52,211,153,0.3);':''}">${f.charAt(0).toUpperCase()+f.slice(1)}</button>`).join('')}
    </div>`;
    return filterBar + `<div class="yi-grid">${filtered.map(t => {
        const term = t.term || t.word || '';
        const def = t.definition || t.meaning || '';
        const ctx = t.context_quote || t.context || '';
        const rarity = t.rarity || 'uncommon';
        const rarityColor = rarity === 'rare' ? 'color:#f472b6;background:rgba(244,114,182,0.1);border:1px solid rgba(244,114,182,0.2)' : 'color:#34d399;background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.2)';
        return `<div class="yi-vocab-card">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <div class="yi-vocab-term" style="margin-bottom:0;">${_escHtml(term)}</div>
                <span style="font-size:9px;padding:1px 6px;border-radius:4px;flex-shrink:0;${rarityColor}">${rarity}</span>
            </div>
            <div class="yi-vocab-def">${_escHtml(def)}</div>
            ${ctx ? `<div class="yi-vocab-ctx">"${_escHtml(ctx)}"</div>` : ''}
        </div>`;
    }).join('')}</div>`;
}

function _ytSetEloquenceFilter(filter) {
    _ytEloquenceFilter = filter;
    const activeTab = document.querySelector('.yi-tab.yi-tab-active');
    if (activeTab) {
        const idx = Array.from(document.querySelectorAll('.yi-tab')).indexOf(activeTab);
        if (idx >= 0) _ytRenderLens(idx);
    }
}
window._ytSetEloquenceFilter = _ytSetEloquenceFilter;

function _ytRenderNarrations(data) {
    if (!data) return '<div class="yi-empty">No narrations data</div>';
    const narrations = Array.isArray(data) ? data : (data.narrations || []);
    if (narrations.length === 0) return '<div class="yi-empty">No narrations found</div>';
    return narrations.map(n => {
        const text = n.narration_text || n.text || n.narration || '';
        const attribution = n.speaker_attribution || n.attribution || '';
        const source = n.source_claimed || n.source || n.reference || '';
        const needsVerify = n.needs_verification !== false;
        const verified = n.verified || n.status === 'verified';
        let badgeClass = 'yi-badge-needs-check';
        let badgeText = 'Needs Verification';
        if (verified) { badgeClass = 'yi-badge-verified'; badgeText = 'Verified'; }
        else if (!needsVerify) { badgeClass = 'yi-badge-unverified'; badgeText = 'Unverified'; }
        return `<div class="yi-narration">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px;">
                <div class="yi-narration-text" style="margin-bottom:0;">${_escHtml(text)}</div>
                <span class="yi-badge ${badgeClass}">${badgeText}</span>
            </div>
            ${attribution ? `<div class="yi-field"><div class="yi-field-label">Attribution</div><div class="yi-field-value">${_escHtml(attribution)}</div></div>` : ''}
            ${source ? `<div class="yi-field"><div class="yi-field-label">Source</div><div class="yi-field-value">${_escHtml(source)}</div></div>` : ''}
        </div>`;
    }).join('');
}

function _ytRenderHistory(data) {
    if (!data) return '<div class="yi-empty">No historical data</div>';
    const events = Array.isArray(data) ? data : [data];
    if (events.length === 0) return '<div class="yi-empty">No historical events found</div>';
    return events.map(ev => {
        return `<div class="yi-card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.2);flex-shrink:0;">
                    <i data-lucide="landmark" style="width:16px;height:16px;color:#fbbf24;"></i>
                </div>
                <div>
                    <div style="font-size:14px;font-weight:700;color:#fff;">${_escHtml(ev.event || ev.name || 'Unknown Event')}</div>
                    ${ev.period ? `<div style="font-size:11px;color:var(--text-muted);">${_escHtml(ev.period)}</div>` : ''}
                </div>
            </div>
            ${ev.actors && ev.actors.length ? `<div class="yi-field"><div class="yi-field-label">Key Figures</div><div class="yi-field-value">${ev.actors.map(a => _escHtml(a)).join(', ')}</div></div>` : ''}
            ${ev.summary ? `<div class="yi-field"><div class="yi-field-label">Summary</div><div class="yi-field-value">${_escHtml(ev.summary)}</div></div>` : ''}
            ${ev.speaker_interpretation ? `<div class="yi-field"><div class="yi-field-label">Speaker's Interpretation</div><div class="yi-field-value" style="font-style:italic;">${_escHtml(ev.speaker_interpretation)}</div></div>` : ''}
        </div>`;
    }).join('');
}

function _ytRenderSpiritual(data) {
    if (!data) return '<div class="yi-empty">No spiritual data</div>';
    const lessons = Array.isArray(data) ? data : [data];
    if (lessons.length === 0) return '<div class="yi-empty">No spiritual insights found</div>';
    return lessons.map(l => {
        return `<div class="yi-card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:rgba(244,114,182,0.1);border:1px solid rgba(244,114,182,0.2);flex-shrink:0;">
                    <i data-lucide="heart" style="width:16px;height:16px;color:#f472b6;"></i>
                </div>
                <div style="font-size:14px;font-weight:600;color:#fff;">${_escHtml(l.lesson || l.teaching || 'Spiritual Insight')}</div>
            </div>
            ${l.supporting_evidence ? `<div class="yi-field"><div class="yi-field-label">Evidence Cited</div><div class="yi-field-value">${_escHtml(l.supporting_evidence)}</div></div>` : ''}
            ${l.speaker_framing ? `<div class="yi-field"><div class="yi-field-label">Speaker's Framing</div><div class="yi-field-value" style="font-style:italic;">${_escHtml(l.speaker_framing)}</div></div>` : ''}
        </div>`;
    }).join('');
}

function _ytRenderPolitics(data) {
    if (!data) return '<div class="yi-empty">No political data</div>';
    const topics = Array.isArray(data) ? data : [data];
    if (topics.length === 0) return '<div class="yi-empty">No political commentary found</div>';
    return topics.map(t => {
        return `<div class="yi-card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:rgba(96,165,250,0.1);border:1px solid rgba(96,165,250,0.2);flex-shrink:0;">
                    <i data-lucide="flag" style="width:16px;height:16px;color:#60a5fa;"></i>
                </div>
                <div style="font-size:14px;font-weight:600;color:#fff;">${_escHtml(t.topic || t.subject || 'Political Topic')}</div>
            </div>
            ${t.analysis ? `<div class="yi-field"><div class="yi-field-label">Analysis</div><div class="yi-field-value">${_escHtml(t.analysis)}</div></div>` : ''}
            ${t.speaker_position ? `<div class="yi-field"><div class="yi-field-label">Speaker's Position</div><div class="yi-field-value" style="font-style:italic;">${_escHtml(t.speaker_position)}</div></div>` : ''}
            ${t.cited_sources && t.cited_sources.length ? `<div class="yi-field"><div class="yi-field-label">Cited Sources</div><div class="yi-field-value">${t.cited_sources.map(s => _escHtml(s)).join(', ')}</div></div>` : ''}
        </div>`;
    }).join('');
}

function _ytRenderGeneric(data, lens) {
    if (!data) return `<div class="yi-empty">No data for ${_escHtml(lens)} lens</div>`;
    const lensLabel = lens.charAt(0).toUpperCase() + lens.slice(1);
    const items = Array.isArray(data) ? data : (typeof data === 'object' ? Object.entries(data) : [data]);
    let html = `<div class="yi-section-title"><i data-lucide="${_ytLensIcons[lens] || 'sparkles'}" style="width:18px;height:18px;color:#c084fc;"></i> ${_escHtml(lensLabel)} Analysis</div>`;
    for (const item of items) {
        if (typeof item === 'object' && item !== null && !Array.isArray(item)) {
            html += `<div class="yi-card">${Object.entries(item).map(([k, v]) => `<div class="yi-field"><div class="yi-field-label">${_escHtml(k)}</div><div class="yi-field-value">${_escHtml(String(v))}</div></div>`).join('')}</div>`;
        } else if (Array.isArray(item) && item.length === 2) {
            html += `<div class="yi-card"><div class="yi-field"><div class="yi-field-label">${_escHtml(String(item[0]))}</div><div class="yi-field-value">${_escHtml(String(item[1]))}</div></div></div>`;
        } else {
            html += `<div class="yi-card"><div style="font-size:13px;line-height:1.6;color:var(--text-secondary);">${_escHtml(String(item))}</div></div>`;
        }
    }
    return html;
}

// ── Lens Guide Page ──

function _ytShowLensGuide() {
    _ytShowingGuide = true;
    // Deactivate all tabs
    document.querySelectorAll('.yi-tab').forEach(tab => tab.classList.remove('yi-tab-active'));

    const body = document.getElementById('yi-body');
    if (!body) return;

    const lenses = [
        { name: 'Summary', icon: 'file-text', color: '#c084fc', rgb: '192,132,252',
          desc: 'Generates a concise 2-3 paragraph summary of the video content along with key takeaways.',
          example: '"2-3 paragraph summary with 3-7 key takeaway bullet points"' },
        { name: 'Transcript', icon: 'scroll-text', color: '#94a3b8', rgb: '148,163,184',
          desc: 'Stores the raw transcript text without any LLM processing. Useful for reference and full-text search.',
          example: '"Full verbatim transcript split into readable paragraphs"' },
        { name: 'Eloquence', icon: 'pen-tool', color: '#34d399', rgb: '52,211,153',
          desc: 'Extracts advanced vocabulary and phrases from any language with definitions, usage context, and rarity classification (uncommon/rare).',
          example: '"Term: فصاحة — Definition: Eloquence, clarity of speech — Context: \'the فصاحة of the Quran is unmatched\' — Rarity: rare"' },
        { name: 'Narrations', icon: 'book-open', color: '#fb923c', rgb: '251,146,60',
          desc: 'Detects hadith, historical narrations, and scholarly citations. Each narration includes attribution and source claims, flagged for verification.',
          example: '"Narration: \'Actions are by intentions\' — Attribution: Prophet Muhammad (PBUH) — Source: Sahih al-Bukhari — Needs Verification: Yes"' },
        { name: 'History', icon: 'landmark', color: '#fbbf24', rgb: '251,191,36',
          desc: 'Identifies historical events, battles, and figures discussed. Captures the time period, key actors, and the speaker\'s interpretation.',
          example: '"Event: Battle of Badr — Period: 624 CE — Actors: Prophet Muhammad, Quraysh — Speaker\'s Interpretation: turning point for early Islam"' },
        { name: 'Spiritual', icon: 'heart', color: '#f472b6', rgb: '244,114,182',
          desc: 'Extracts spiritual lessons, philosophical arguments, and moral teachings with supporting evidence and the speaker\'s framing.',
          example: '"Lesson: Patience in adversity — Evidence: Quran 2:153 — Framing: Speaker emphasizes sabr as active endurance, not passive waiting"' },
        { name: 'Politics', icon: 'flag', color: '#60a5fa', rgb: '96,165,250',
          desc: 'Identifies political commentary, policy analysis, and geopolitical observations including the speaker\'s position and cited sources.',
          example: '"Topic: Gulf cooperation — Analysis: Speaker argues for economic integration — Position: Pro-GCC unity — Sources: recent summit declarations"' },
    ];

    body.innerHTML = `
        <div class="yi-section-title" style="margin-bottom:18px;"><i data-lucide="book-open" style="width:20px;height:20px;color:#c084fc;"></i> Lens Guide</div>
        <div style="font-size:13px;color:var(--text-secondary);line-height:1.7;margin-bottom:20px;">
            Lenses are focused analysis prompts that extract specific types of insight from video transcripts. Each lens runs independently against the LLM, producing structured JSON output. Select lenses per-channel when adding a YouTube channel.
        </div>
        <div class="yi-guide-grid">
            ${lenses.map(l => `
                <div class="yi-guide-card" style="--gc-color:${l.color};--gc-rgb:${l.rgb};">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div class="yi-guide-icon" style="background:rgba(${l.rgb},0.1);border:1px solid rgba(${l.rgb},0.2);">
                            <i data-lucide="${l.icon}" style="width:22px;height:22px;color:${l.color};"></i>
                        </div>
                        <div>
                            <div class="yi-guide-title" style="color:${l.color};">${l.name}</div>
                            <div class="yi-guide-sub">Lens Mode</div>
                        </div>
                    </div>
                    <div class="yi-guide-desc">${l.desc}</div>
                    <div class="yi-guide-ex" style="background:rgba(${l.rgb},0.06);color:${l.color};border:1px solid rgba(${l.rgb},0.12);">${l.example}</div>
                </div>
            `).join('')}
        </div>`;
    lucide.createIcons();
}
window._ytShowLensGuide = _ytShowLensGuide;

// ── Star Parallax Engine (Purple theme) ──

function _ytInitStars() {
    const canvas = document.getElementById('yi-star-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H, stars = [], mouse = { x: 0.5, y: 0.5 };

    function resize() {
        const rect = canvas.parentElement.getBoundingClientRect();
        W = canvas.width = rect.width; H = canvas.height = rect.height;
    }
    resize();
    window.addEventListener('resize', resize);

    // Create stars
    for (let i = 0; i < 90; i++) {
        stars.push({
            x: Math.random() * W, y: Math.random() * H,
            r: Math.random() * 1.4 + 0.3,
            depth: Math.random() * 3 + 0.2,
            a: Math.random() * 0.5 + 0.15,
            dx: (Math.random() - 0.5) * 0.15,
            dy: (Math.random() - 0.5) * 0.1,
            phase: Math.random() * Math.PI * 2,
        });
    }

    canvas.parentElement.addEventListener('mousemove', e => {
        const rect = canvas.parentElement.getBoundingClientRect();
        mouse.x = (e.clientX - rect.left) / W;
        mouse.y = (e.clientY - rect.top) / H;
    });

    let shootingStar = null;
    let lastShoot = 0;

    function draw(t) {
        ctx.clearRect(0, 0, W, H);
        const px = (mouse.x - 0.5) * 12;
        const py = (mouse.y - 0.5) * 8;

        // Stars
        for (const s of stars) {
            s.x += s.dx; s.y += s.dy;
            if (s.x < 0) s.x = W; if (s.x > W) s.x = 0;
            if (s.y < 0) s.y = H; if (s.y > H) s.y = 0;
            const sx = s.x + px * s.depth;
            const sy = s.y + py * s.depth;
            const flicker = 0.7 + 0.3 * Math.sin(t * 0.002 + s.phase);
            ctx.beginPath();
            ctx.arc(sx, sy, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(192,132,252,${s.a * flicker})`;
            ctx.fill();
        }

        // Constellation lines (connect nearby stars)
        ctx.strokeStyle = 'rgba(192,132,252,0.04)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < stars.length; i++) {
            for (let j = i + 1; j < stars.length; j++) {
                const dx = (stars[i].x + px * stars[i].depth) - (stars[j].x + px * stars[j].depth);
                const dy = (stars[i].y + py * stars[i].depth) - (stars[j].y + py * stars[j].depth);
                const dist = dx * dx + dy * dy;
                if (dist < 8000) {
                    ctx.beginPath();
                    ctx.moveTo(stars[i].x + px * stars[i].depth, stars[i].y + py * stars[i].depth);
                    ctx.lineTo(stars[j].x + px * stars[j].depth, stars[j].y + py * stars[j].depth);
                    ctx.stroke();
                }
            }
        }

        // Shooting star
        if (t - lastShoot > 6000 && Math.random() < 0.008) {
            shootingStar = { x: Math.random() * W * 0.7, y: Math.random() * H * 0.3, len: 60 + Math.random() * 40, life: 1 };
            lastShoot = t;
        }
        if (shootingStar) {
            shootingStar.life -= 0.02;
            if (shootingStar.life <= 0) { shootingStar = null; }
            else {
                const ss = shootingStar;
                const grad = ctx.createLinearGradient(ss.x, ss.y, ss.x + ss.len, ss.y + ss.len * 0.4);
                grad.addColorStop(0, `rgba(192,132,252,${ss.life * 0.6})`);
                grad.addColorStop(1, 'rgba(192,132,252,0)');
                ctx.strokeStyle = grad; ctx.lineWidth = 1.5;
                ctx.beginPath(); ctx.moveTo(ss.x, ss.y); ctx.lineTo(ss.x + ss.len, ss.y + ss.len * 0.4); ctx.stroke();
                ss.x += 3; ss.y += 1.2;
            }
        }

        _ytStarRAF = requestAnimationFrame(draw);
    }
    _ytStarRAF = requestAnimationFrame(draw);
}

function _ytCloseInsights() {
    const modal = document.getElementById('yt-insights-modal');
    if (modal) modal.remove();
    if (_ytStarRAF) { cancelAnimationFrame(_ytStarRAF); _ytStarRAF = null; }
}
window._ytCloseInsights = _ytCloseInsights;

// Expose functions needed by onclick
window._ytProcessChannel = _ytProcessChannel;
window._ytToggleVideos = _ytToggleVideos;
window._ytShowInsights = _ytShowInsights;
window._ytShowLens = _ytShowLens;
window._ytDeleteChannel = _ytDeleteChannel;
