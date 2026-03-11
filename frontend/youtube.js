// ═══════════════════════════════════════════════════════════
// YOUTUBE CHANNEL MANAGEMENT & INSIGHTS VIEWER
// ═══════════════════════════════════════════════════════════

let _ytChannels = [];
let _ytVideos = {};
let _ytInsights = {};

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
        return `<div class="yt-channel-card" data-channel-id="${ch.id}">
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
                    <button onclick="_ytProcessChannel(${ch.id})" class="fb-tool-btn" title="Process new videos">
                        <i data-lucide="play" class="w-3 h-3"></i>
                    </button>
                    <button onclick="_ytToggleVideos(${ch.id})" class="fb-tool-btn" title="Show videos">
                        <i data-lucide="list" class="w-3 h-3"></i>
                    </button>
                    <button onclick="_ytDeleteChannel(${ch.id}, '${_escAttr(channelName)}')" class="fb-tool-btn" title="Remove channel" onmouseenter="this.style.color='#f87171';this.style.borderColor='rgba(239,68,68,0.3)'" onmouseleave="this.style.color='var(--text-muted)';this.style.borderColor='var(--border-strong)'">
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
            input.value = '';
            _ytLoadChannels();
            if (typeof showToast === 'function') showToast('Channel added', 'success');
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Failed to add channel', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to add channel', 'error');
    }
}
window._ytAddChannel = _ytAddChannel;

// ── Delete channel ──
async function _ytDeleteChannel(id, name) {
    if (!confirm(`Remove channel "${name}"?`)) return;
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
                const clickable = v.status === 'complete' ? `onclick="_ytShowInsights('${v.video_id}')" class="cursor-pointer"` : '';

                return `<div ${clickable} class="flex items-center gap-2 px-2 py-1.5 rounded-md transition-colors" onmouseenter="this.style.background='var(--bg-hover)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="${statusIcon}" class="w-3 h-3 ${statusColor} flex-shrink-0${v.status === 'processing' ? ' animate-spin' : ''}"></i>
                    <span class="text-[10px] flex-1 truncate" style="color:var(--text-secondary)">${_escHtml(v.title || v.video_id)}</span>
                    <span class="text-[8px] ${statusColor}">${v.status}</span>
                </div>`;
            }).join('');
            lucide.createIcons();
        }
    } catch (e) {
        el.innerHTML = '<div class="text-[9px] py-2 text-red-400">Failed to load videos</div>';
    }
}

// ── Show insights for a video ──
async function _ytShowInsights(videoId) {
    let modal = document.getElementById('yt-insights-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'yt-insights-modal';
        modal.className = 'hidden';
        modal.innerHTML = `
            <div class="ctx-editor-backdrop" onclick="_ytCloseInsights()"></div>
            <div class="ctx-editor-sidebar" style="width:min(520px,90vw);">
                <div class="ctx-editor-header">
                    <div class="flex items-center gap-2">
                        <i data-lucide="sparkles" class="w-4 h-4" style="color:var(--accent)"></i>
                        <span class="text-sm font-bold" style="color:var(--text-heading)">Video Insights</span>
                    </div>
                    <button onclick="_ytCloseInsights()" class="p-1 rounded-md transition-colors" style="color:var(--text-muted)" title="Close">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>
                <div id="yt-insights-tabs" class="flex items-center gap-1 px-4 py-2" style="border-bottom:1px solid var(--border-strong);overflow-x:auto;"></div>
                <div id="yt-insights-content" class="flex-1 overflow-y-auto px-4 py-3" style="min-height:0;"></div>
            </div>`;
        document.body.appendChild(modal);
        lucide.createIcons();
    }

    modal.classList.remove('hidden');
    modal.classList.add('ctx-slide-in');

    const tabs = document.getElementById('yt-insights-tabs');
    const content = document.getElementById('yt-insights-content');
    if (tabs) tabs.innerHTML = '<span class="text-[10px]" style="color:var(--text-muted)">Loading...</span>';
    if (content) content.innerHTML = '';

    try {
        const r = await fetch(`/api/youtube/insights/${videoId}`, { headers: _ytHeaders() });
        if (r.ok) {
            const d = await r.json();
            const insights = d.insights || [];
            if (insights.length === 0) {
                if (tabs) tabs.innerHTML = '';
                if (content) content.innerHTML = '<div class="text-[10px] text-center py-8" style="color:var(--text-muted)">No insights extracted yet</div>';
                return;
            }

            // Build tabs
            if (tabs) {
                tabs.innerHTML = insights.map((ins, i) => {
                    const lensName = (ins.lens_name || 'unknown').charAt(0).toUpperCase() + (ins.lens_name || 'unknown').slice(1);
                    return `<button onclick="_ytShowLens(${i})" class="yt-lens-tab text-[10px] px-2.5 py-1 rounded-md font-medium transition-all ${i === 0 ? 'yt-lens-active' : ''}" style="color:${i === 0 ? 'var(--accent)' : 'var(--text-muted)'};border:1px solid ${i === 0 ? 'var(--accent)' : 'var(--border-strong)'};">${lensName}</button>`;
                }).join('');
            }

            _ytInsights = insights;
            _ytRenderLens(0);
        }
    } catch (e) {
        if (content) content.innerHTML = '<div class="text-[10px] text-center py-8 text-red-400">Failed to load insights</div>';
    }
}

function _ytShowLens(index) {
    document.querySelectorAll('.yt-lens-tab').forEach((tab, i) => {
        const active = i === index;
        tab.style.color = active ? 'var(--accent)' : 'var(--text-muted)';
        tab.style.borderColor = active ? 'var(--accent)' : 'var(--border-strong)';
        tab.classList.toggle('yt-lens-active', active);
    });
    _ytRenderLens(index);
}

function _ytRenderLens(index) {
    const content = document.getElementById('yt-insights-content');
    if (!content || !_ytInsights[index]) return;

    const ins = _ytInsights[index];
    const lens = ins.lens_name || 'unknown';
    let data;
    try { data = typeof ins.data === 'string' ? JSON.parse(ins.data) : ins.data; }
    catch (e) { data = ins.data; }

    if (lens === 'summary') {
        content.innerHTML = _ytRenderSummary(data);
    } else if (lens === 'eloquence') {
        content.innerHTML = _ytRenderEloquence(data);
    } else if (lens === 'narrations') {
        content.innerHTML = _ytRenderNarrations(data);
    } else {
        // Generic structured render for unknown lens types
        content.innerHTML = _ytRenderGeneric(data, lens);
    }
}

function _ytRenderSummary(data) {
    if (!data) return '<div class="text-[10px]" style="color:var(--text-muted)">No summary data</div>';
    const items = Array.isArray(data) ? data : (data.takeaways || data.points || [data]);
    return items.map(item => {
        const text = typeof item === 'string' ? item : (item.takeaway || item.point || item.text || Object.values(item).filter(v => typeof v === 'string').join(' — ') || 'No content');
        return `<div class="mb-3 px-3 py-2 rounded-lg" style="background:var(--bg-hover);border:1px solid var(--border);">
            <div class="text-[11px] leading-relaxed" style="color:var(--text-secondary)">${_escHtml(text)}</div>
        </div>`;
    }).join('');
}

function _ytRenderGeneric(data, lens) {
    if (!data) return `<div class="text-[10px]" style="color:var(--text-muted)">No data for ${_escHtml(lens)} lens</div>`;
    const lensLabel = lens.charAt(0).toUpperCase() + lens.slice(1);
    const items = Array.isArray(data) ? data : (typeof data === 'object' ? Object.entries(data) : [data]);
    let html = `<div class="text-[10px] font-semibold mb-2" style="color:var(--text-heading);">${_escHtml(lensLabel)} Analysis</div>`;
    for (const item of items) {
        let text;
        if (typeof item === 'string') text = item;
        else if (Array.isArray(item) && item.length === 2) text = `<strong>${_escHtml(String(item[0]))}</strong>: ${_escHtml(String(item[1]))}`;
        else if (typeof item === 'object' && item !== null) text = Object.entries(item).map(([k,v]) => `<strong>${_escHtml(k)}</strong>: ${_escHtml(String(v))}`).join('<br>');
        else text = String(item);
        html += `<div class="mb-2 px-3 py-2 rounded-lg text-[11px] leading-relaxed" style="background:var(--bg-hover);border:1px solid var(--border);color:var(--text-secondary);">${text}</div>`;
    }
    return html;
}

function _ytRenderEloquence(data) {
    if (!data) return '<div class="text-[10px]" style="color:var(--text-muted)">No eloquence data</div>';
    const terms = Array.isArray(data) ? data : (data.terms || []);
    return `<div class="grid grid-cols-1 gap-2">${terms.map(t => {
        const term = t.term || t.word || '';
        const def = t.definition || t.meaning || '';
        return `<div class="px-3 py-2 rounded-lg" style="background:var(--bg-hover);border:1px solid var(--border);">
            <div class="text-[12px] font-bold mb-0.5" style="color:var(--accent)">${_escHtml(term)}</div>
            <div class="text-[10px]" style="color:var(--text-secondary)">${_escHtml(def)}</div>
        </div>`;
    }).join('')}</div>`;
}

function _ytRenderNarrations(data) {
    if (!data) return '<div class="text-[10px]" style="color:var(--text-muted)">No narrations data</div>';
    const narrations = Array.isArray(data) ? data : (data.narrations || []);
    return narrations.map(n => {
        const verified = n.verified || n.status === 'verified';
        const source = n.source || n.reference || '';
        const statusBadge = verified
            ? '<span class="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Verified</span>'
            : '<span class="text-[9px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">Unverified</span>';

        return `<div class="mb-3 px-3 py-2 rounded-lg" style="background:var(--bg-hover);border:1px solid var(--border);">
            <div class="flex items-center justify-between mb-1">
                <span class="text-[10px] font-semibold" style="color:var(--text-heading)">${_escHtml(n.text || n.narration || '')}</span>
                ${statusBadge}
            </div>
            ${source ? `<div class="text-[9px]" style="color:var(--text-muted)">Source: ${_escHtml(source)}</div>` : ''}
        </div>`;
    }).join('');
}

function _ytCloseInsights() {
    const modal = document.getElementById('yt-insights-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('ctx-slide-in');
    }
}
window._ytCloseInsights = _ytCloseInsights;

// Expose functions needed by onclick
window._ytProcessChannel = _ytProcessChannel;
window._ytToggleVideos = _ytToggleVideos;
window._ytShowInsights = _ytShowInsights;
window._ytShowLens = _ytShowLens;
window._ytDeleteChannel = _ytDeleteChannel;
