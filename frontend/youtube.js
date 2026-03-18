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
            'complete': 'text-emerald-400', 'transcribed': 'text-cyan-400',
            'transcribing': 'text-blue-400', 'extracting': 'text-purple-400',
            'failed': 'text-red-400', 'started': 'text-amber-400'
        }[status] || 'text-slate-500';
        const statusIcon = {
            'complete': 'check-circle', 'transcribed': 'file-check',
            'transcribing': 'mic', 'extracting': 'sparkles',
            'failed': 'alert-circle', 'started': 'loader-2'
        }[status] || 'clock';
        const isActive = status === 'started' || status === 'transcribing' || status === 'extracting';
        const iconEl = statusEl.querySelector('[data-lucide]');
        const textEl = statusEl.querySelector('.yt-video-status');
        if (iconEl) {
            iconEl.setAttribute('data-lucide', statusIcon);
            iconEl.className = `w-3 h-3 ${statusColor} flex-shrink-0${isActive ? ' animate-pulse' : ''}`;
        }
        if (textEl) { textEl.textContent = status; textEl.className = `text-[10px] yt-video-status ${statusColor}`; }
        // Make clickable when complete or transcribed
        if (status === 'complete' || status === 'transcribed') {
            const dbId = statusEl.dataset.ytDbId;
            if (dbId) {
                statusEl.style.cursor = 'pointer';
                statusEl.onclick = (e) => { e.stopPropagation(); _ytShowInsights(parseInt(dbId)); };
            }
        }
        lucide.createIcons();
    } else {
        // Video DOM not found — refresh any visible video lists so new videos appear
        document.querySelectorAll('[id^="yt-videos-"]:not(.hidden)').forEach(el => {
            const chId = el.id.replace('yt-videos-', '');
            if (chId) _ytRefreshVideos(parseInt(chId));
        });
    }

    // Update retranscription progress indicator if active
    if (_ytRetranscribeVideoTitle && typeof _ytUpdateRetranscribeProgress === 'function') {
        _ytUpdateRetranscribeProgress(status);
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
                    <div class="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);">
                        <i data-lucide="youtube" class="w-5 h-5 text-red-400"></i>
                    </div>
                    <div class="min-w-0">
                        <div class="text-[14px] font-semibold truncate" style="color:var(--text-heading)">${_escHtml(channelName)}</div>
                        <div class="text-[11px]" style="color:var(--text-muted)">${videoCount} videos · Lenses: ${lenses}</div>
                    </div>
                </div>
                <div class="flex items-center gap-2 flex-shrink-0">
                    <button onclick="event.stopPropagation();_ytProcessChannel(${ch.id})" class="yt-action-btn yt-action-process" title="Fetch & process new videos">
                        <i data-lucide="play" style="width:14px;height:14px;"></i> Process
                    </button>
                    <button onclick="event.stopPropagation();_ytExtractAllChannel(${ch.id})" class="yt-action-btn yt-action-extract" title="Extract all lenses for all videos" id="yt-extract-all-${ch.id}">
                        <i data-lucide="sparkles" style="width:14px;height:14px;"></i> Extract All
                    </button>
                    <button onclick="event.stopPropagation();_ytExportChannel(${ch.id})" class="yt-action-btn yt-action-export" title="Export all data as JSON or Markdown">
                        <i data-lucide="download" style="width:14px;height:14px;"></i> Export
                    </button>
                    <button onclick="event.stopPropagation();_ytToggleVideos(${ch.id})" class="yt-action-btn yt-action-videos" title="Show/hide video list">
                        <i data-lucide="list" style="width:14px;height:14px;"></i> Videos
                    </button>
                    <button onclick="event.stopPropagation();_ytDeleteChannel(${ch.id}, '${_escAttr(channelName)}')" class="yt-action-btn yt-action-remove" title="Remove channel">
                        <i data-lucide="trash-2" style="width:14px;height:14px;"></i> Remove
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
        // First try to discover new videos
        const r = await fetch(`/api/youtube/process/${channelId}`, {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ reprocess: true })
        });
        if (r.ok) {
            const data = await r.json();
            if (data.reset) {
                if (typeof showToast === 'function') showToast(`Reset ${data.reset} videos — re-transcribing...`, 'info');
            } else if (data.new_videos > 0) {
                if (typeof showToast === 'function') showToast(`Found ${data.new_videos} new videos`, 'success');
            } else {
                if (typeof showToast === 'function') showToast('Processing queued', 'success');
            }
            // Ensure video list is visible and start polling for status updates
            const vidEl = document.getElementById(`yt-videos-${channelId}`);
            if (vidEl && vidEl.classList.contains('hidden')) _ytToggleVideos(channelId);
            else if (vidEl) _ytRefreshVideos(channelId);
            _ytPollVideoStatus(channelId);
        } else {
            if (typeof showToast === 'function') showToast('Failed to queue processing', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to queue processing', 'error');
    }
}

let _ytVideoStatusPolls = {};
function _ytPollVideoStatus(channelId) {
    if (_ytVideoStatusPolls[channelId]) return; // already polling
    const startTime = Date.now();
    const maxWait = 300000; // 5 min max
    const poll = async () => {
        if (Date.now() - startTime > maxWait) { delete _ytVideoStatusPolls[channelId]; return; }
        try {
            const r = await fetch(`/api/youtube/videos/${channelId}`, { headers: _ytHeaders() });
            if (!r.ok) { delete _ytVideoStatusPolls[channelId]; return; }
            const d = await r.json();
            const videos = d.videos || [];
            // Update each video's status in-place if DOM exists
            let allDone = true;
            for (const v of videos) {
                const el = document.querySelector(`[data-yt-video="${v.video_id}"]`);
                if (el) {
                    const iconEl = el.querySelector('[data-lucide]');
                    const textEl = el.querySelector('.yt-video-status');
                    const statusIcon = { 'complete': 'check-circle', 'transcribed': 'file-check', 'processing': 'loader-2', 'pending': 'clock', 'failed': 'alert-circle', 'transcribing': 'mic', 'extracting': 'sparkles' }[v.status] || 'clock';
                    const statusColor = { 'complete': 'text-emerald-400', 'transcribed': 'text-cyan-400', 'processing': 'text-amber-400', 'pending': 'text-slate-500', 'failed': 'text-red-400', 'transcribing': 'text-blue-400', 'extracting': 'text-purple-400' }[v.status] || 'text-slate-500';
                    const isActive = v.status === 'processing' || v.status === 'transcribing' || v.status === 'extracting' || v.status === 'started';
                    if (iconEl) {
                        iconEl.setAttribute('data-lucide', statusIcon);
                        iconEl.className = `w-4 h-4 ${statusColor} flex-shrink-0${isActive ? ' animate-pulse' : ''}`;
                    }
                    if (textEl) { textEl.textContent = v.status; textEl.className = `text-[10px] yt-video-status ${statusColor}`; }
                    if (v.status === 'complete' || v.status === 'transcribed') {
                        el.style.cursor = 'pointer';
                        el.onclick = (e) => { e.stopPropagation(); _ytShowInsights(v.id); };
                    }
                }
                if (v.status !== 'complete' && v.status !== 'transcribed' && v.status !== 'failed') allDone = false;
            }
            lucide.createIcons();
            if (allDone) { delete _ytVideoStatusPolls[channelId]; return; }
        } catch(e) { /* ignore */ }
        _ytVideoStatusPolls[channelId] = setTimeout(poll, 3000);
    };
    _ytVideoStatusPolls[channelId] = setTimeout(poll, 2000);
}

// ── Extract all lenses for all videos in a channel ──
async function _ytExtractAllChannel(channelId) {
    const btn = document.getElementById(`yt-extract-all-${channelId}`);
    if (btn) { btn.style.opacity = '0.5'; btn.style.pointerEvents = 'none'; btn.innerHTML = '<i data-lucide="loader-2" style="width:14px;height:14px;" class="animate-pulse"></i> Extracting...'; lucide.createIcons(); }
    try {
        const r = await fetch(`/api/youtube/extract-all/${channelId}`, {
            method: 'POST',
            headers: _ytHeaders(),
        });
        const d = await r.json().catch(() => ({}));
        if (r.ok) {
            const count = d.queued || 0;
            if (typeof showToast === 'function') showToast(`Extracting lenses for ${count} video${count !== 1 ? 's' : ''}...`, 'success');
        } else {
            if (typeof showToast === 'function') showToast(d.error || 'Extract failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Extract failed', 'error');
    }
    if (btn) { btn.style.opacity = ''; btn.style.pointerEvents = ''; btn.innerHTML = '<i data-lucide="sparkles" style="width:14px;height:14px;"></i> Extract All'; lucide.createIcons(); }
}
window._ytExtractAllChannel = _ytExtractAllChannel;

// ── Export channel data ──
function _ytExportChannel(channelId) {
    // Show a small format picker dropdown
    const existing = document.getElementById('yt-export-picker');
    if (existing) { existing.remove(); return; }

    const btn = document.querySelector(`[onclick*="_ytExportChannel(${channelId})"]`);
    if (!btn) return;

    const picker = document.createElement('div');
    picker.id = 'yt-export-picker';
    picker.style.cssText = 'position:absolute;z-index:9999;background:var(--bg-card,#1a1a2e);border:1px solid var(--border-color,rgba(255,255,255,0.1));border-radius:8px;padding:6px;display:flex;flex-direction:column;gap:2px;box-shadow:0 8px 24px rgba(0,0,0,0.4);';

    const makeBtn = (label, icon, fmt) => {
        const b = document.createElement('button');
        b.className = 'yt-action-btn';
        b.style.cssText = 'justify-content:flex-start;width:100%;padding:8px 14px;font-size:12px;color:var(--text-secondary);';
        b.innerHTML = `<i data-lucide="${icon}" style="width:14px;height:14px;"></i> ${label}`;
        b.onclick = (e) => {
            e.stopPropagation();
            picker.remove();
            const token = typeof getAuthToken === 'function' ? getAuthToken() : '';
            const url = `/api/youtube/export/${channelId}?format=${fmt}&token=${encodeURIComponent(token)}`;
            // Use a hidden link to trigger download
            const a = document.createElement('a');
            a.href = url;
            a.download = '';
            document.body.appendChild(a);
            a.click();
            a.remove();
            if (typeof showToast === 'function') showToast(`Exporting as ${fmt.toUpperCase()}...`, 'info');
        };
        return b;
    };

    picker.appendChild(makeBtn('Export as JSON', 'file-json', 'json'));
    picker.appendChild(makeBtn('Export as Markdown', 'file-text', 'md'));

    // Position below the button
    const rect = btn.getBoundingClientRect();
    picker.style.top = (rect.bottom + 4) + 'px';
    picker.style.left = rect.left + 'px';
    document.body.appendChild(picker);
    lucide.createIcons();

    // Close on outside click
    const close = (e) => { if (!picker.contains(e.target) && e.target !== btn) { picker.remove(); document.removeEventListener('click', close); } };
    setTimeout(() => document.addEventListener('click', close), 0);
}
window._ytExportChannel = _ytExportChannel;

// ── Fetch and render video list into a container ──
async function _ytLoadVideoList(channelId, el) {
    try {
        const r = await fetch(`/api/youtube/videos/${channelId}`, { headers: _ytHeaders() });
        if (r.ok) {
            const d = await r.json();
            const videos = d.videos || [];
            if (videos.length === 0) {
                el.innerHTML = '<div class="text-[11px] py-2" style="color:var(--text-muted)">No videos yet. Click Process to fetch.</div>';
                return;
            }
            el.innerHTML = videos.map(v => {
                const statusColor = {
                    'complete': 'text-emerald-400', 'transcribed': 'text-cyan-400',
                    'processing': 'text-amber-400', 'pending': 'text-slate-500',
                    'failed': 'text-red-400', 'transcribing': 'text-blue-400',
                    'extracting': 'text-purple-400', 'low_quality': 'text-yellow-400'
                }[v.status] || 'text-slate-500';
                const statusIcon = {
                    'complete': 'check-circle', 'transcribed': 'file-check',
                    'processing': 'loader-2', 'pending': 'clock',
                    'failed': 'alert-circle', 'transcribing': 'mic',
                    'extracting': 'sparkles', 'low_quality': 'alert-triangle'
                }[v.status] || 'clock';
                const isActive = v.status === 'processing' || v.status === 'transcribing' || v.status === 'extracting';
                const viewable = v.status === 'complete' || v.status === 'transcribed' || v.status === 'low_quality';
                const clickable = viewable ? `onclick="event.stopPropagation();_ytShowInsights(${v.id})" class="cursor-pointer"` : '';
                const isStuck = v.status === 'transcribing' || v.status === 'processing' || v.status === 'extracting';
                const canRetranscribe = !isStuck && v.status !== 'pending';
                let actionBtn = '';
                if (isStuck) {
                    actionBtn = `<button onclick="event.stopPropagation();_ytCancelTranscribe(${v.id})" class="yi-header-btn" style="padding:2px 8px;font-size:10px;gap:3px;flex-shrink:0;color:#f87171;" title="Cancel and reset">
                        <i data-lucide="x-circle" style="width:10px;height:10px;"></i>
                    </button>`;
                } else if (canRetranscribe) {
                    actionBtn = `<button onclick="event.stopPropagation();_ytRetranscribeVideo(${v.id})" class="yi-header-btn" style="padding:2px 8px;font-size:10px;gap:3px;flex-shrink:0;" title="Re-transcribe this video">
                        <i data-lucide="refresh-cw" style="width:10px;height:10px;"></i>
                    </button>`;
                }

                return `<div data-yt-video="${v.video_id}" data-yt-db-id="${v.id}" ${clickable} class="flex items-center gap-3 px-3 py-2 rounded-md transition-colors" onmouseenter="this.style.background='var(--bg-hover)'" onmouseleave="this.style.background='transparent'">
                    <i data-lucide="${statusIcon}" class="w-4 h-4 ${statusColor} flex-shrink-0${isActive ? ' animate-pulse' : ''}"></i>
                    <span class="text-[13px] flex-1 truncate" style="color:var(--text-secondary)">${_escHtml(v.title || v.video_id)}</span>
                    ${actionBtn}
                    <span class="text-[10px] yt-video-status ${statusColor}" style="font-weight:600;">${v.status}</span>
                </div>`;
            }).join('');
            lucide.createIcons();
        }
    } catch (e) {
        el.innerHTML = '<div class="text-[9px] py-2 text-red-400">Failed to load videos</div>';
    }
}

// ── Refresh video list without toggling ──
async function _ytRefreshVideos(channelId) {
    const el = document.getElementById(`yt-videos-${channelId}`);
    if (!el || el.classList.contains('hidden')) return;
    return _ytLoadVideoList(channelId, el);
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
    el.innerHTML = '<div class="text-[11px] py-2" style="color:var(--text-muted)">Loading videos...</div>';
    return _ytLoadVideoList(channelId, el);
}

// ═══════════════════════════════════════════════════════════
// YOUTUBE INSIGHTS VIEWER — Arcane Modal with Star Parallax
// ═══════════════════════════════════════════════════════════

let _ytCurrentVideoId = null;
let _ytCurrentYtVideoId = null;
let _ytStarRAF = null;
let _ytShowingGuide = false;
let _ytEloquenceFilter = 'all';

const _ytLensIcons = { summary: 'file-text', eloquence: 'pen-tool', narrations: 'book-open', history: 'landmark', spiritual: 'heart', politics: 'flag', transcript: 'scroll-text' };
const _ytAllLenses = ['transcript', 'summary', 'eloquence', 'narrations', 'history', 'spiritual', 'politics'];
let _ytExtractingLenses = new Set();
let _ytTranscriptLang = 'en';
let _ytSizeMode = localStorage.getItem('stratos_yi_size') || 'normal'; // 'sm', 'normal', 'lg'
let _ytExtractLang = localStorage.getItem('stratos_yi_extract_lang') || ''; // '' = same as current view lang

function _ytLangSelect(id) {
    const langs = [['','Auto'],['en','English'],['ar','العربية'],['ja','日本語'],['ko','한국어'],['zh','中文'],['fr','Français'],['de','Deutsch'],['es','Español'],['ru','Русский']];
    const cur = _ytExtractLang;
    return `<select id="${id}" onchange="_ytSetExtractLang(this.value)" class="yi-header-btn" style="padding:4px 8px;font-size:11px;background:var(--bg-input,rgba(8,10,22,0.7));color:var(--text-secondary);cursor:pointer;appearance:auto;">
        ${langs.map(([v,l]) => `<option value="${v}"${cur===v?' selected':''}>${l}</option>`).join('')}
    </select>`;
}
function _ytSetExtractLang(v) {
    _ytExtractLang = v;
    localStorage.setItem('stratos_yi_extract_lang', v);
}
window._ytSetExtractLang = _ytSetExtractLang;

function _ytGetExtractLang() {
    return _ytExtractLang || _ytCurrentLang || _ytTranscriptLang || 'en';
}

async function _ytShowInsights(videoId) {
    _ytCurrentVideoId = videoId;
    _ytShowingGuide = false;
    _ytExtractingLenses = new Set();
    let modal = document.getElementById('yt-insights-modal');
    if (modal) modal.remove();

    modal = document.createElement('div');
    modal.id = 'yt-insights-modal';
    modal.innerHTML = `
        <div class="yi-overlay" onclick="if(event.target===this)_ytCloseInsights()">
            <div class="yi-container${_ytSizeMode !== 'normal' ? ` yi-size-${_ytSizeMode}` : ''}">
                <canvas class="yi-stars" id="yi-star-canvas"></canvas>
                <div class="yi-titlebar">
                    <div style="width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0;background:color-mix(in srgb, var(--accent) 12%, transparent);border:1px solid color-mix(in srgb, var(--accent) 25%, transparent);">
                        <i data-lucide="sparkles" style="width:18px;height:18px;color:var(--accent-light,var(--accent));"></i>
                    </div>
                    <div style="flex:1;min-width:0;">
                        <div class="yi-title" id="yi-title">Video Insights</div>
                        <div class="yi-subtitle" id="yi-subtitle"></div>
                    </div>
                    <div class="yi-lang-area" id="yi-lang-area"></div>
                    <a id="yi-yt-link" href="#" target="_blank" rel="noopener" class="yi-header-btn" style="display:none;text-decoration:none;color:var(--text-secondary);" title="Watch on YouTube"><i data-lucide="play" style="width:14px;height:14px;"></i> YouTube</a>
                    <button class="yi-header-btn" onclick="_ytToggleSize()" id="yi-size-btn" title="Toggle text size"><i data-lucide="${{sm:'a-large-small',normal:'type',lg:'move-diagonal'}[_ytSizeMode]||'type'}" style="width:14px;height:14px;"></i> ${{sm:'Small',normal:'Normal',lg:'Large'}[_ytSizeMode]||'Normal'}</button>
                    <button class="yi-header-btn" onclick="_ytShowLensGuide()" title="Lens Guide"><i data-lucide="book-open" style="width:14px;height:14px;"></i> Guide</button>
                    <button class="yi-header-btn" onclick="_ytAskAgent()" title="Ask Strat Agent"><i data-lucide="bot" style="width:14px;height:14px;"></i> Ask Agent</button>
                    <button class="yi-close" onclick="_ytCloseInsights()" title="Close">&times;</button>
                </div>
                <div class="yi-tabs" id="yi-tabs"></div>
                <div class="yi-body" id="yi-body">
                    <div class="yi-loading"><div style="display:flex;gap:6px;"><span class="w-2.5 h-2.5 rounded-full animate-bounce" style="background:var(--accent);animation-delay:0ms;"></span><span class="w-2.5 h-2.5 rounded-full animate-bounce" style="background:var(--accent);animation-delay:150ms;"></span><span class="w-2.5 h-2.5 rounded-full animate-bounce" style="background:var(--accent);animation-delay:300ms;"></span></div></div>
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
        _ytTranscriptLang = d.transcript_language || 'en';

        _ytInsightsAll = {};
        for (const ins of allInsights) {
            _ytInsightsAll[`${ins.lens_name}_${ins.language || 'en'}`] = ins;
        }

        // Fallback: inject transcript from video record if no transcript lens row exists
        if (d.transcript_text && !_ytInsightsAll[`transcript_${_ytTranscriptLang}`] && !_ytInsightsAll['transcript_en']) {
            const fallbackLang = _ytTranscriptLang || 'en';
            _ytInsightsAll[`transcript_${fallbackLang}`] = {
                lens_name: 'transcript', language: fallbackLang,
                content: JSON.stringify({ transcript: d.transcript_text }),
            };
            if (fallbackLang !== 'en') {
                _ytInsightsAll['transcript_en'] = {
                    lens_name: 'transcript', language: 'en',
                    content: JSON.stringify({ transcript: d.transcript_text, note: `Original language: ${fallbackLang}` }),
                };
            }
            // Add to allInsights for counting
            allInsights.push(_ytInsightsAll[`transcript_${fallbackLang}`]);
        }

        if (!_ytAvailableLangs.includes(_ytCurrentLang)) _ytCurrentLang = 'en';

        // Build set of lenses that have data
        const availableLensSet = new Set();
        for (const ins of allInsights) availableLensSet.add(ins.lens_name);

        const extractedCount = new Set(allInsights.filter(ins => (ins.language || 'en') === _ytCurrentLang).map(ins => ins.lens_name)).size;

        const titleEl = document.getElementById('yi-title');
        const subEl = document.getElementById('yi-subtitle');
        if (titleEl && videoTitle) titleEl.textContent = videoTitle;
        if (subEl) subEl.textContent = `${extractedCount} lens${extractedCount !== 1 ? 'es' : ''} extracted`;

        // Show YouTube link button if we have a video ID
        _ytCurrentYtVideoId = d.yt_video_id || '';
        const ytLinkEl = document.getElementById('yi-yt-link');
        if (ytLinkEl && _ytCurrentYtVideoId) {
            ytLinkEl.href = `https://www.youtube.com/watch?v=${encodeURIComponent(_ytCurrentYtVideoId)}`;
            ytLinkEl.style.display = '';
        }

        _ytRenderLangToggle();

        // Always show all lens tabs — dim those without data
        const tabs = document.getElementById('yi-tabs');
        if (tabs) {
            tabs.innerHTML = _ytAllLenses.map((lens, i) => {
                const lensName = lens.charAt(0).toUpperCase() + lens.slice(1);
                const icon = _ytLensIcons[lens] || 'sparkles';
                const hasData = availableLensSet.has(lens);
                const dimClass = hasData ? '' : ' yi-tab-dim';
                return `<button onclick="_ytShowLensTab('${lens}')" data-lens="${lens}" class="yi-tab${i === 0 ? ' yi-tab-active' : ''}${dimClass}"><i data-lucide="${icon}" style="width:14px;height:14px;"></i>${lensName}</button>`;
            }).join('');
            lucide.createIcons();
        }

        // Build _ytInsights array matching _ytAllLenses order
        _ytInsights = _ytAllLenses.map(lens => {
            const langKey = `${lens}_${_ytCurrentLang}`;
            const fallbackKey = `${lens}_en`;
            return _ytInsightsAll[langKey] || _ytInsightsAll[fallbackKey] || null;
        });

        _ytRenderLensByName(_ytAllLenses[0]);
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
    // Switch to scholarly persona
    if (typeof switchPersona === 'function') switchPersona('scholarly');
    if (typeof _agentFullscreen !== 'undefined' && !_agentFullscreen && typeof toggleAgentFullscreen === 'function') {
        toggleAgentFullscreen();
    } else if (typeof _openAgentPanel === 'function') {
        _openAgentPanel();
    }
    const input = document.getElementById('agent-input');
    if (input) { input.value = `Tell me about the video "${title}" — summarize the key insights`; input.focus(); }
}
window._ytAskAgent = _ytAskAgent;

function _ytShowLensTab(lensName) {
    _ytShowingGuide = false;
    document.querySelectorAll('.yi-tab').forEach(tab => {
        tab.classList.toggle('yi-tab-active', tab.dataset.lens === lensName);
    });
    _ytRenderLensByName(lensName);
}
window._ytShowLensTab = _ytShowLensTab;

function _ytShowLens(index) {
    _ytShowLensTab(_ytAllLenses[index] || _ytAllLenses[0]);
}

function _ytRenderLensByName(lensName) {
    const body = document.getElementById('yi-body');
    if (!body) return;

    const langKey = `${lensName}_${_ytCurrentLang}`;
    const fallbackKey = `${lensName}_en`;
    const ins = _ytInsightsAll[langKey] || _ytInsightsAll[fallbackKey] || null;

    if (!ins) {
        // No data — show extract button (not for transcript)
        if (lensName === 'transcript') {
            body.innerHTML = '<div class="yi-empty"><div style="font-size:13px;margin-bottom:4px;">No transcript available</div><div style="font-size:11px;opacity:0.6;">Video may still be processing</div></div>';
        } else {
            const isExtracting = _ytExtractingLenses.has(lensName);
            const lensLabel = lensName.charAt(0).toUpperCase() + lensName.slice(1);
            const icon = _ytLensIcons[lensName] || 'sparkles';
            body.innerHTML = `<div class="yi-empty">
                <div style="width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;margin:0 auto 16px;background:color-mix(in srgb, var(--accent) 12%, transparent);border:1px solid color-mix(in srgb, var(--accent) 18%, transparent);">
                    <i data-lucide="${icon}" style="width:24px;height:24px;color:var(--accent-light,var(--accent));opacity:0.5;"></i>
                </div>
                <div style="font-size:13px;margin-bottom:4px;">No ${lensLabel} data yet</div>
                <div style="font-size:11px;opacity:0.6;margin-bottom:16px;">Click below to extract this lens from the transcript</div>
                <div style="display:flex;align-items:center;gap:8px;justify-content:center;">
                    ${_ytLangSelect('yi-extract-lang-'+lensName)}
                    <button onclick="_ytExtractLens('${lensName}')" class="yi-header-btn" style="padding:8px 20px;font-size:12px;${isExtracting ? 'opacity:0.5;pointer-events:none;' : ''}" id="yi-extract-btn-${lensName}">
                        ${isExtracting ? '<i data-lucide="loader-2" style="width:14px;height:14px;" class="animate-pulse"></i> Extracting...' : `<i data-lucide="sparkles" style="width:14px;height:14px;"></i> Extract ${lensLabel}`}
                    </button>
                </div>
            </div>`;
        }
        lucide.createIcons();
        return;
    }

    const lens = ins.lens_name || 'unknown';
    let data;
    const raw = ins.content || ins.data;
    try { data = typeof raw === 'string' ? JSON.parse(raw) : raw; } catch (e) { data = raw; }

    let content = '';
    if (lens === 'summary') content = _ytRenderSummary(data);
    else if (lens === 'eloquence') content = _ytRenderEloquence(data);
    else if (lens === 'narrations') content = _ytRenderNarrations(data);
    else if (lens === 'transcript') content = _ytRenderTranscript(data);
    else if (lens === 'history') content = _ytRenderHistory(data);
    else if (lens === 'spiritual') content = _ytRenderSpiritual(data);
    else if (lens === 'politics') content = _ytRenderPolitics(data);
    else content = _ytRenderGeneric(data, lens);

    // Add refresh button + save button for non-transcript lenses
    if (lens !== 'transcript') {
        const isRefreshing = _ytExtractingLenses.has(lensName);
        const mode = lens === 'summary' ? 'replace' : 'merge';
        const tooltip = lens === 'summary' ? 'Re-summarize' : 'Find more';
        const saveBtn = typeof _ytBuildSaveButton === 'function' ? _ytBuildSaveButton(lensName) : '';
        body.innerHTML = `<div style="display:flex;justify-content:flex-end;align-items:center;gap:8px;margin-bottom:8px;">
            ${_ytLangSelect('yi-refresh-lang-'+lensName)}
            <button onclick="_ytRefreshLens('${lensName}','${mode}')" class="yi-header-btn" style="padding:4px 10px;font-size:11px;gap:4px;${isRefreshing ? 'opacity:0.5;pointer-events:none;' : ''}" title="${tooltip}">
                <i data-lucide="${isRefreshing ? 'loader-2' : 'refresh-cw'}" style="width:12px;height:12px;" ${isRefreshing ? 'class="animate-pulse"' : ''}></i> ${isRefreshing ? 'Refreshing...' : tooltip}
            </button>
            ${saveBtn}
        </div>` + content;
    } else {
        body.innerHTML = content;
    }

    lucide.createIcons();
}

function _ytRenderLens(index) {
    _ytRenderLensByName(_ytAllLenses[index] || _ytAllLenses[0]);
}

function _ytSwitchLang(lang) {
    _ytCurrentLang = lang;
    localStorage.setItem('stratos_insight_lang', lang);
    document.querySelectorAll('.yt-lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    if (_ytShowingGuide) return; // Don't re-render if on guide page
    const activeTab = document.querySelector('.yi-tab.yi-tab-active');
    if (activeTab && activeTab.dataset.lens) {
        _ytRenderLensByName(activeTab.dataset.lens);
    }
}
window._ytSwitchLang = _ytSwitchLang;

// ── Renderers ──

function _ytParagraphize(text, sentencesPerPara = 3) {
    // Split on existing double-newlines first
    const existing = text.split(/\n\s*\n/);
    if (existing.length > 1) return existing.map(p => p.trim()).filter(Boolean);
    // Fallback: split on sentence boundaries, group into paragraphs
    const sentences = text.split(/(?<=[.!?。！？؟])\s+/);
    const paragraphs = [];
    for (let i = 0; i < sentences.length; i += sentencesPerPara) {
        paragraphs.push(sentences.slice(i, i + sentencesPerPara).join(' '));
    }
    return paragraphs.filter(Boolean);
}

function _ytRenderTranscript(data) {
    if (!data) return '<div class="yi-empty">No transcript data</div>';
    const text = typeof data === 'string' ? data : (data.transcript || '');

    // If this is a placeholder (no real transcript, just a note)
    if (!text && data.translation_available === false) {
        const origLang = data.original_language || '?';
        return `<div class="yi-empty" style="text-align:center;padding:30px;">
            <p style="margin-bottom:8px;">Transcript is in <strong>${origLang.toUpperCase()}</strong></p>
            <p style="font-size:11px;color:var(--text-secondary);">Switch to the ${origLang.toUpperCase()} language tab to view the original transcript.</p>
        </div>`;
    }
    if (!text) return '<div class="yi-empty">No transcript data</div>';

    const note = data.note ? `<div class="yi-note">${_escHtml(data.note)}</div>` : '';
    const paragraphs = _ytParagraphize(text, 3);

    // Action bar: re-transcribe + translate
    const shortWarning = text.length < 500
        ? `<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;margin-bottom:12px;border-radius:8px;background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);">
            <i data-lucide="alert-triangle" style="width:16px;height:16px;color:#fbbf24;flex-shrink:0;"></i>
            <span style="font-size:12px;color:var(--text-secondary);flex:1;">This transcript looks incomplete (${text.length} chars).</span>
            <button onclick="_ytRetranscribe()" class="yi-header-btn" style="padding:6px 14px;font-size:11px;white-space:nowrap;" id="yi-retranscribe-btn">
                <i data-lucide="refresh-cw" style="width:12px;height:12px;"></i> Re-transcribe
            </button>
        </div>` : '';

    // Translate button bar
    const translateBar = `<div style="display:flex;justify-content:flex-end;align-items:center;gap:8px;margin-bottom:8px;">
        <button onclick="_ytRetranscribe()" class="yi-header-btn" style="padding:4px 10px;font-size:11px;gap:4px;" title="Re-fetch transcript from scratch" id="yi-retranscribe-btn2">
            <i data-lucide="refresh-cw" style="width:12px;height:12px;"></i> Re-transcribe
        </button>
        <select id="yi-translate-lang" style="background:var(--bg-input,rgba(255,255,255,0.06));border:1px solid var(--border-strong,rgba(255,255,255,0.1));border-radius:6px;padding:3px 8px;font-size:11px;color:var(--text-secondary,rgba(255,255,255,0.6));">
            <option value="en">English</option>
            <option value="ar">Arabic</option>
            <option value="ja">Japanese</option>
            <option value="zh">Chinese</option>
            <option value="ko">Korean</option>
            <option value="fr">French</option>
            <option value="de">German</option>
            <option value="es">Spanish</option>
        </select>
        <button onclick="_ytTranslateTranscript()" class="yi-header-btn" style="padding:4px 10px;font-size:11px;gap:4px;" id="yi-translate-btn">
            <i data-lucide="languages" style="width:12px;height:12px;"></i> Translate
        </button>
    </div>`;

    return `${shortWarning}${translateBar}${note}<div class="yi-transcript" id="yi-transcript-content">${paragraphs.map(p => `<p>${_escHtml(p)}</p>`).join('')}</div>`;
}

// ── Retranscription progress indicator ──
let _ytRetranscribeVideoTitle = null;

function _ytShowRetranscribeProgress(videoTitle) {
    _ytRetranscribeVideoTitle = videoTitle;
    // Insert progress bar below the titlebar in the insights modal
    const existing = document.getElementById('yi-retranscribe-progress');
    if (existing) existing.remove();
    const titlebar = document.querySelector('.yi-titlebar');
    if (!titlebar) return;
    const bar = document.createElement('div');
    bar.id = 'yi-retranscribe-progress';
    bar.style.cssText = 'display:flex;align-items:center;gap:10px;padding:8px 16px;background:rgba(96,165,250,0.08);border-bottom:1px solid rgba(96,165,250,0.15);';
    bar.innerHTML = `<div style="width:16px;height:16px;border:2px solid rgba(96,165,250,0.2);border-top-color:#60a5fa;border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0;"></div>
        <span style="font-size:12px;color:#60a5fa;font-weight:500;">Retranscribing: <strong style="color:#93c5fd;">${_escHtml(videoTitle.substring(0, 50))}</strong></span>
        <span id="yi-retranscribe-status" style="font-size:10px;color:rgba(96,165,250,0.6);margin-left:auto;">Starting...</span>`;
    titlebar.insertAdjacentElement('afterend', bar);
    // Add spin keyframe if not present
    if (!document.getElementById('yi-spin-style')) {
        const style = document.createElement('style');
        style.id = 'yi-spin-style';
        style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
        document.head.appendChild(style);
    }
}

function _ytUpdateRetranscribeProgress(status) {
    const el = document.getElementById('yi-retranscribe-status');
    if (el) {
        const labels = { started: 'Started...', transcribing: 'Transcribing audio...', transcribed: 'Transcript ready', extracting: 'Extracting insights...', complete: 'Complete!', failed: 'Failed' };
        el.textContent = labels[status] || status;
        if (status === 'complete' || status === 'failed') {
            setTimeout(_ytHideRetranscribeProgress, 2000);
        }
    }
}

function _ytHideRetranscribeProgress() {
    _ytRetranscribeVideoTitle = null;
    const el = document.getElementById('yi-retranscribe-progress');
    if (el) el.remove();
}

async function _ytRetranscribe() {
    if (!_ytCurrentVideoId) return;
    const videoTitle = document.getElementById('yi-title')?.textContent || 'Video';
    const btn = document.getElementById('yi-retranscribe-btn');
    const btn2 = document.getElementById('yi-retranscribe-btn2');
    [btn, btn2].forEach(b => {
        if (b) { b.style.opacity = '0.5'; b.style.pointerEvents = 'none'; b.innerHTML = '<i data-lucide="loader-2" style="width:12px;height:12px;" class="animate-pulse"></i> Re-transcribing...'; }
    });
    lucide.createIcons();
    // Show retranscription progress indicator
    _ytShowRetranscribeProgress(videoTitle);
    try {
        const r = await fetch('/api/youtube/retranscribe', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ video_id: _ytCurrentVideoId }),
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Retranscribing: ${videoTitle.substring(0, 40)}...`, 'info');
            // Poll for completion
            _ytPollForRetranscribe(_ytCurrentVideoId);
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Re-transcribe failed', 'error');
            _ytHideRetranscribeProgress();
            [btn, btn2].forEach(b => { if (b) { b.style.opacity = ''; b.style.pointerEvents = ''; } });
        }
    } catch(e) {
        if (typeof showToast === 'function') showToast('Re-transcribe failed', 'error');
        _ytHideRetranscribeProgress();
        [btn, btn2].forEach(b => { if (b) { b.style.opacity = ''; b.style.pointerEvents = ''; } });
    }
}
window._ytRetranscribe = _ytRetranscribe;

async function _ytTranslateTranscript() {
    if (!_ytCurrentVideoId) return;
    const targetLang = document.getElementById('yi-translate-lang')?.value || 'en';
    const btn = document.getElementById('yi-translate-btn');
    const contentEl = document.getElementById('yi-transcript-content');
    if (!contentEl) return;

    if (btn) { btn.style.opacity = '0.5'; btn.style.pointerEvents = 'none'; btn.innerHTML = '<i data-lucide="loader-2" style="width:12px;height:12px;" class="animate-pulse"></i> Translating...'; lucide.createIcons(); }

    try {
        const r = await fetch('/api/youtube/translate-transcript', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ video_id: _ytCurrentVideoId, target_language: targetLang }),
        });
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Translation failed', 'error');
            return;
        }
        const data = await r.json();
        if (data.translated_text) {
            const paragraphs = _ytParagraphize(data.translated_text, 3);
            contentEl.innerHTML = `<div class="yi-note" style="margin-bottom:8px;">Translated to ${targetLang.toUpperCase()} via LLM (${data.source_language || '?'} → ${targetLang})</div>` +
                paragraphs.map(p => `<p>${_escHtml(p)}</p>`).join('');
            if (typeof showToast === 'function') showToast(`Translated to ${targetLang.toUpperCase()}`, 'success');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Translation failed: ' + e.message, 'error');
    } finally {
        if (btn) { btn.style.opacity = ''; btn.style.pointerEvents = ''; btn.innerHTML = '<i data-lucide="languages" style="width:12px;height:12px;"></i> Translate'; lucide.createIcons(); }
    }
}
window._ytTranslateTranscript = _ytTranslateTranscript;

async function _ytRetranscribeVideo(dbId) {
    // Get video title from the DOM row
    const row = document.querySelector(`[data-yt-db-id="${dbId}"]`);
    const videoTitle = row ? (row.querySelector('.truncate')?.textContent || 'Video') : 'Video';
    try {
        const r = await fetch('/api/youtube/retranscribe', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ video_id: dbId }),
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Retranscribing: ${videoTitle.substring(0, 40)}...`, 'info');
            // Update status in UI immediately
            const el = document.querySelector(`[data-yt-db-id="${dbId}"] .yt-video-status`);
            if (el) { el.textContent = 'transcribing'; el.className = 'text-[10px] yt-video-status text-blue-400'; }
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Re-transcribe failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Re-transcribe failed', 'error');
    }
}
window._ytRetranscribeVideo = _ytRetranscribeVideo;

async function _ytCancelTranscribe(dbId) {
    try {
        const r = await fetch('/api/youtube/cancel-transcribe', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ video_id: dbId }),
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast('Transcription cancelled — video reset to pending', 'info');
            const el = document.querySelector(`[data-yt-db-id="${dbId}"] .yt-video-status`);
            if (el) { el.textContent = 'pending'; el.className = 'text-[10px] yt-video-status text-slate-500'; }
            // Re-render video list to update buttons
            const channelCard = document.querySelector(`[data-yt-db-id="${dbId}"]`)?.closest('[data-yt-channel]');
            if (channelCard) {
                const chId = channelCard.dataset.ytChannel;
                if (chId) _ytLoadVideos(parseInt(chId));
            }
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Cancel failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Cancel failed', 'error');
    }
}
window._ytCancelTranscribe = _ytCancelTranscribe;

function _ytPollForRetranscribe(videoId) {
    const startTime = Date.now();
    const poll = async () => {
        if (_ytCurrentVideoId != videoId) return;
        if (Date.now() - startTime > 600000) return; // 10 min max
        try {
            const r = await fetch(`/api/youtube/insights/${videoId}?language=all`, { headers: _ytHeaders() });
            if (!r.ok) return;
            const d = await r.json();
            const newTranscript = d.transcript_text || '';
            if (newTranscript.length > 500) {
                // Good transcript — refresh the modal
                _ytHideRetranscribeProgress();
                if (typeof showToast === 'function') showToast('Transcript updated successfully!', 'success');
                _ytShowInsights(videoId);
                return;
            }
        } catch(e) { /* ignore */ }
        setTimeout(poll, 5000);
    };
    setTimeout(poll, 5000);
}

function _ytRenderSummary(data) {
    if (!data) return '<div class="yi-empty">No summary data</div>';
    // Handle {summary, key_takeaways} format
    if (data.summary || data.key_takeaways) {
        let html = '';
        if (data.summary) {
            const paras = _ytParagraphize(data.summary, 3);
            html += `<div class="yi-card"><div class="yi-summary-text">${paras.map(p => `<p style="margin-bottom:12px;">${_escHtml(p)}</p>`).join('')}</div></div>`;
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

function _ytNarrationSourceUrl(source, sourceRef, text) {
    // Fallback: Google search from available info (Strategy C)
    const parts = [];
    if (source) parts.push(source);
    if (sourceRef) parts.push(sourceRef);
    if (parts.length === 0 && text) parts.push(text.slice(0, 60));
    if (parts.length === 0) return null;
    return `https://www.google.com/search?q=${encodeURIComponent(parts.join(' '))}`;
}

function _ytConfidenceBadge(confidence, method) {
    // Confidence-based badge for resolved sources
    if (confidence >= 0.85) return { cls: 'yi-badge-verified', text: 'High Confidence', icon: 'check-circle' };
    if (confidence >= 0.65) return { cls: 'yi-badge-unverified', text: 'Moderate', icon: 'circle-dot' };
    if (confidence >= 0.50) return { cls: 'yi-badge-needs-check', text: 'Low Confidence', icon: 'help-circle' };
    return { cls: 'yi-badge-needs-check', text: 'Needs Verification', icon: 'search' };
}

function _ytRenderNarrations(data) {
    if (!data) return '<div class="yi-empty">No narrations data</div>';
    const narrations = Array.isArray(data) ? data : (data.narrations || []);
    if (narrations.length === 0) return '<div class="yi-empty">No narrations found</div>';
    return narrations.map(n => {
        const text = n.narration_text || n.text || n.narration || '';
        const attribution = n.speaker_attribution || n.attribution || '';
        const source = n.source_claimed || n.source || n.reference || '';
        const sourceRef = n.source_reference || '';
        const resolved = n._resolved || null;

        // Determine source URL and confidence badge
        let sourceUrl, badge;
        if (resolved && resolved.url) {
            sourceUrl = resolved.url;
            badge = _ytConfidenceBadge(resolved.confidence || 0, resolved.method || '');
        } else {
            sourceUrl = _ytNarrationSourceUrl(source, sourceRef, text);
            badge = { cls: 'yi-badge-needs-check', text: 'Needs Verification', icon: 'search' };
        }

        const displayText = sourceRef ? `${source} (${sourceRef})` : source;
        const methodLabel = resolved?.method ? ` · ${resolved.method.replace('pattern:', '').replace('search_', 'web ')}` : '';

        let sourceHtml = '';
        if (source || sourceRef || resolved) {
            sourceHtml = `<div class="yi-field"><div class="yi-field-label">Source</div><div class="yi-field-value">
                <a href="${sourceUrl}" target="_blank" rel="noopener" style="color:var(--accent-light,var(--accent));text-decoration:underline;text-underline-offset:2px;">${_escHtml(displayText || 'Look up source')}</a>
                ${!resolved ? ' <span style="font-size:10px;opacity:0.5;margin-left:4px;">· Google Search</span>' : `<span style="font-size:10px;opacity:0.5;margin-left:4px;">${methodLabel}</span>`}
            </div></div>`;
        }

        return `<div class="yi-narration">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px;">
                <div class="yi-narration-text" style="margin-bottom:0;">${_escHtml(text)}</div>
                <span class="yi-badge ${badge.cls}" title="${badge.text}">${badge.text}</span>
            </div>
            ${attribution ? `<div class="yi-field"><div class="yi-field-label">Attribution</div><div class="yi-field-value">${_escHtml(attribution)}</div></div>` : ''}
            ${sourceHtml}
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
    let html = `<div class="yi-section-title"><i data-lucide="${_ytLensIcons[lens] || 'sparkles'}" style="width:18px;height:18px;color:var(--accent-light,var(--accent));"></i> ${_escHtml(lensLabel)} Analysis</div>`;
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
        <div class="yi-section-title" style="margin-bottom:18px;"><i data-lucide="book-open" style="width:20px;height:20px;color:var(--accent-light,var(--accent));"></i> Lens Guide</div>
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

// ── Star Parallax Engine (theme-aware) ──

function _ytGetAccentRGB() {
    const s = getComputedStyle(document.documentElement);
    const accent = s.getPropertyValue('--accent-light').trim() || s.getPropertyValue('--accent').trim() || '#c084fc';
    // Parse hex to RGB
    const el = document.createElement('div');
    el.style.color = accent; document.body.appendChild(el);
    const rgb = getComputedStyle(el).color.match(/\d+/g);
    el.remove();
    return rgb ? rgb.slice(0, 3).map(Number) : [192, 132, 252];
}

function _ytInitStars() {
    const canvas = document.getElementById('yi-star-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.style.pointerEvents = 'none';
    const [sr, sg, sb] = _ytGetAccentRGB();
    let W, H, stars = [];

    function resize() {
        const el = canvas.parentElement;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        W = canvas.width = rect.width;
        H = canvas.height = rect.height;
    }
    resize();
    // Re-measure when container resizes (content load, window resize)
    const _resizeObs = new ResizeObserver(resize);
    _resizeObs.observe(canvas.parentElement);

    // Create stars — depth controls visual size and brightness
    for (let i = 0; i < 100; i++) {
        const depth = Math.random() * 4 + 0.1;
        const depthNorm = depth / 4.1;
        stars.push({
            x: Math.random(), y: Math.random(), // normalized 0..1
            r: 0.3 + depthNorm * 1.8,
            depth: depth,
            a: 0.1 + depthNorm * 0.55,
            dx: (Math.random() - 0.5) * (0.05 + depthNorm * 0.2),
            dy: (Math.random() - 0.5) * (0.03 + depthNorm * 0.12),
            phase: Math.random() * Math.PI * 2,
        });
    }

    let shootingStar = null;
    let lastShoot = 0;

    function draw(t) {
        ctx.clearRect(0, 0, W, H);

        // Stars — autonomous drift, no mouse interaction
        for (const s of stars) {
            s.x += s.dx / W; s.y += s.dy / H;
            if (s.x < 0) s.x = 1; if (s.x > 1) s.x = 0;
            if (s.y < 0) s.y = 1; if (s.y > 1) s.y = 0;
            const sx = s.x * W;
            const sy = s.y * H;
            const flicker = 0.7 + 0.3 * Math.sin(t * 0.002 + s.phase);
            ctx.beginPath();
            ctx.arc(sx, sy, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${sr},${sg},${sb},${s.a * flicker})`;
            ctx.fill();
        }

        // Constellation lines (connect nearby stars)
        ctx.strokeStyle = `rgba(${sr},${sg},${sb},0.04)`;
        ctx.lineWidth = 0.5;
        for (let i = 0; i < stars.length; i++) {
            const sx1 = stars[i].x * W, sy1 = stars[i].y * H;
            for (let j = i + 1; j < stars.length; j++) {
                const sx2 = stars[j].x * W, sy2 = stars[j].y * H;
                const dx = sx1 - sx2, dy = sy1 - sy2;
                if (dx * dx + dy * dy < 8000) {
                    ctx.beginPath();
                    ctx.moveTo(sx1, sy1);
                    ctx.lineTo(sx2, sy2);
                    ctx.stroke();
                }
            }
        }

        // Shooting star — random spawn edge, random angle
        if (t - lastShoot > 6000 && Math.random() < 0.008) {
            const angle = (Math.random() * 0.8 + 0.3) * (Math.random() < 0.5 ? 1 : -1); // ±0.3..1.1 rad
            const len = 50 + Math.random() * 60;
            const spawnEdge = Math.random();
            let sx, sy;
            if (spawnEdge < 0.5) { sx = Math.random() * W; sy = Math.random() * H * 0.3; } // top
            else if (spawnEdge < 0.75) { sx = Math.random() * W * 0.3; sy = Math.random() * H; } // left
            else { sx = W * 0.7 + Math.random() * W * 0.3; sy = Math.random() * H; } // right
            const speed = 2.5 + Math.random() * 2;
            shootingStar = { x: sx, y: sy, len, life: 1, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed };
            lastShoot = t;
        }
        if (shootingStar) {
            shootingStar.life -= 0.018;
            if (shootingStar.life <= 0) { shootingStar = null; }
            else {
                const ss = shootingStar;
                const ex = ss.x + ss.vx * ss.len * 0.6, ey = ss.y + ss.vy * ss.len * 0.6;
                const grad = ctx.createLinearGradient(ss.x, ss.y, ex, ey);
                grad.addColorStop(0, `rgba(${sr},${sg},${sb},${ss.life * 0.6})`);
                grad.addColorStop(1, `rgba(${sr},${sg},${sb},0)`);
                ctx.strokeStyle = grad; ctx.lineWidth = 1.5;
                ctx.beginPath(); ctx.moveTo(ss.x, ss.y); ctx.lineTo(ex, ey); ctx.stroke();
                ss.x += ss.vx; ss.y += ss.vy;
            }
        }

        _ytStarRAF = requestAnimationFrame(draw);
    }
    _ytStarRAF = requestAnimationFrame(draw);
}

async function _ytExtractLens(lensName) {
    if (!_ytCurrentVideoId || _ytExtractingLenses.has(lensName)) return;
    const extractLang = _ytGetExtractLang();
    _ytExtractingLenses.add(lensName);
    _ytRenderLensByName(lensName);
    try {
        const r = await fetch('/api/youtube/extract-lens', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ video_id: _ytCurrentVideoId, lens: lensName, language: extractLang })
        });
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Extraction failed', 'error');
            _ytExtractingLenses.delete(lensName);
            _ytRenderLensByName(lensName);
            return;
        }
        // Poll for completion — SSE is unreliable, polling is the primary mechanism
        _ytPollForLens(_ytCurrentVideoId, lensName, extractLang);
    } catch (e) {
        const msg = e.message === 'Failed to fetch' ? 'Connection to server lost. Check if the server is running.' : `Extraction failed: ${e.message}`;
        if (typeof showToast === 'function') showToast(msg, 'error');
        _ytExtractingLenses.delete(lensName);
        _ytRenderLensByName(lensName);
    }
}
window._ytExtractLens = _ytExtractLens;

async function _ytRefreshLens(lensName, mode) {
    if (!_ytCurrentVideoId || _ytExtractingLenses.has(lensName)) return;
    const extractLang = _ytGetExtractLang();
    _ytExtractingLenses.add(lensName);
    _ytRenderLensByName(lensName);
    try {
        const r = await fetch('/api/youtube/extract-lens', {
            method: 'POST',
            headers: _ytHeaders(),
            body: JSON.stringify({ video_id: _ytCurrentVideoId, lens: lensName, language: extractLang, mode })
        });
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Refresh failed', 'error');
            _ytExtractingLenses.delete(lensName);
            _ytRenderLensByName(lensName);
            return;
        }
        _ytPollForLens(_ytCurrentVideoId, lensName, extractLang);
    } catch (e) {
        const msg = e.message === 'Failed to fetch' ? 'Connection to server lost. Check if the server is running.' : `Refresh failed: ${e.message}`;
        if (typeof showToast === 'function') showToast(msg, 'error');
        _ytExtractingLenses.delete(lensName);
        _ytRenderLensByName(lensName);
    }
}
window._ytRefreshLens = _ytRefreshLens;

// Poll for lens extraction completion — checks every 3s, up to 120s
function _ytPollForLens(videoId, lensName, extractLang) {
    const startTime = Date.now();
    const maxWait = 120000;
    // Snapshot current content to detect changes (for refresh/merge)
    const oldKey = `${lensName}_${extractLang}`;
    const oldContent = _ytInsightsAll[oldKey]?.content || null;

    const poll = async () => {
        if (_ytCurrentVideoId != videoId) return; // modal closed or different video
        if (Date.now() - startTime > maxWait) {
            _ytExtractingLenses.delete(lensName);
            _ytRenderLensByName(lensName);
            if (typeof showToast === 'function') showToast('Extraction timed out', 'error');
            return;
        }
        try {
            const r = await fetch(`/api/youtube/insights/${videoId}?language=all`, { headers: _ytHeaders() });
            if (!r.ok) { setTimeout(poll, 3000); return; }
            const d = await r.json();
            const allInsights = d.insights || [];
            const newRow = allInsights.find(ins => ins.lens_name === lensName && (ins.language || 'en') === extractLang);
            if (newRow && newRow.content !== oldContent) {
                _ytRefreshInsightsFromData(d, lensName, extractLang);
                return;
            }
        } catch (e) { /* retry */ }
        setTimeout(poll, 3000);
    };
    setTimeout(poll, 3000);
}

// Shared: rebuild insights cache from API data and re-render
function _ytRefreshInsightsFromData(d, lensName, extractLang) {
    const allInsights = d.insights || [];
    _ytAvailableLangs = d.available_languages || ['en'];
    _ytInsightsAll = {};
    for (const ins of allInsights) {
        _ytInsightsAll[`${ins.lens_name}_${ins.language || 'en'}`] = ins;
    }
    // Transcript fallback
    if (d.transcript_text && !_ytInsightsAll[`transcript_${_ytTranscriptLang}`] && !_ytInsightsAll['transcript_en']) {
        const fl = _ytTranscriptLang || 'en';
        _ytInsightsAll[`transcript_${fl}`] = { lens_name: 'transcript', language: fl, content: JSON.stringify({ transcript: d.transcript_text }) };
        allInsights.push(_ytInsightsAll[`transcript_${fl}`]);
    }
    // Ensure the extracted language is in the available list so the toggle appears
    if (extractLang && !_ytAvailableLangs.includes(extractLang)) {
        _ytAvailableLangs.push(extractLang);
    }
    // Only auto-switch if the current language has NO data for this lens
    // (i.e., first extraction ever — user hasn't built up data in current lang yet)
    const curKey = `${lensName}_${_ytCurrentLang}`;
    if (extractLang && extractLang !== _ytCurrentLang && !_ytInsightsAll[curKey]) {
        _ytCurrentLang = extractLang;
        localStorage.setItem('stratos_insight_lang', extractLang);
    }
    // Update tab dim state
    const availableLensSet = new Set();
    for (const ins of allInsights) availableLensSet.add(ins.lens_name);
    document.querySelectorAll('.yi-tab').forEach(tab => {
        if (availableLensSet.has(tab.dataset.lens)) tab.classList.remove('yi-tab-dim');
    });
    // Update subtitle count
    const extractedCount = new Set(allInsights.filter(ins => (ins.language || 'en') === _ytCurrentLang).map(ins => ins.lens_name)).size;
    const subEl = document.getElementById('yi-subtitle');
    if (subEl) subEl.textContent = `${extractedCount} lens${extractedCount !== 1 ? 'es' : ''} extracted`;
    // Clear spinner and re-render
    _ytExtractingLenses.delete(lensName);
    const activeTab = document.querySelector('.yi-tab.yi-tab-active');
    if (activeTab) _ytRenderLensByName(activeTab.dataset.lens);
    _ytRenderLangToggle();
    const langLabel = extractLang && extractLang !== _ytCurrentLang
        ? ` in ${_ytLangLabels[extractLang] || extractLang.toUpperCase()} — use language toggle to view`
        : '';
    if (typeof showToast === 'function') showToast(`${lensName.charAt(0).toUpperCase() + lensName.slice(1)} lens extracted${langLabel}`, 'success');
}

// SSE handler — fast-path, but polling is the reliable fallback
function _handleLensExtracted(event) {
    const { video_id, lens, language } = event;
    if (!_ytExtractingLenses.has(lens)) return; // polling already handled it
    if (_ytCurrentVideoId && _ytCurrentVideoId == video_id) {
        fetch(`/api/youtube/insights/${video_id}?language=all`, { headers: _ytHeaders() })
            .then(r => r.json())
            .then(d => _ytRefreshInsightsFromData(d, lens, language))
            .catch(() => {});
    }
}
window._handleLensExtracted = _handleLensExtracted;

function _handleNarrationResolved(event) {
    const { video_id } = event;
    // If narrations modal is open for this video, re-fetch to get resolved URLs
    if (_ytCurrentVideoId && _ytCurrentVideoId == video_id) {
        fetch(`/api/youtube/insights/${video_id}?language=all`, { headers: _ytHeaders() })
            .then(r => r.json())
            .then(d => _ytRefreshInsightsFromData(d, 'narrations', _ytCurrentLang))
            .catch(() => {});
    }
}
window._handleNarrationResolved = _handleNarrationResolved;

function _ytCloseInsights() {
    const modal = document.getElementById('yt-insights-modal');
    if (modal) modal.remove();
    if (_ytStarRAF) { cancelAnimationFrame(_ytStarRAF); _ytStarRAF = null; }
}
window._ytCloseInsights = _ytCloseInsights;

function _ytToggleSize() {
    const cycle = ['sm', 'normal', 'lg'];
    _ytSizeMode = cycle[(cycle.indexOf(_ytSizeMode) + 1) % 3];
    localStorage.setItem('stratos_yi_size', _ytSizeMode);
    const container = document.querySelector('.yi-container');
    if (container) {
        container.classList.remove('yi-size-sm', 'yi-size-lg');
        if (_ytSizeMode !== 'normal') container.classList.add(`yi-size-${_ytSizeMode}`);
    }
    const btn = document.getElementById('yi-size-btn');
    if (btn) {
        const icons = {sm:'a-large-small',normal:'type',lg:'move-diagonal'};
        const labels = {sm:'Small',normal:'Normal',lg:'Large'};
        btn.innerHTML = `<i data-lucide="${icons[_ytSizeMode]}" style="width:14px;height:14px;"></i> ${labels[_ytSizeMode]}`;
        lucide.createIcons();
    }
}
window._ytToggleSize = _ytToggleSize;

// ═══════════════════════════════════════════════════════════
// SAVED INSIGHTS — localStorage-based bookmarking
// ═══════════════════════════════════════════════════════════

function _ytGetSavedInsights() {
    try { return JSON.parse(localStorage.getItem('stratos_saved_insights') || '[]'); }
    catch (e) { return []; }
}

function _ytSetSavedInsights(arr) {
    localStorage.setItem('stratos_saved_insights', JSON.stringify(arr));
}

function _ytSaveInsight(videoId, videoTitle, lens, language, content) {
    const saved = _ytGetSavedInsights();
    // Avoid duplicates by video_id + lens + language
    const exists = saved.some(s => s.video_id === videoId && s.lens === lens && s.language === language);
    if (exists) {
        if (typeof showToast === 'function') showToast('Insight already saved', 'info');
        return;
    }
    saved.push({ video_id: videoId, video_title: videoTitle, lens, language, content, saved_at: new Date().toISOString() });
    _ytSetSavedInsights(saved);
    if (typeof showToast === 'function') showToast(`Saved ${lens} insight for later`, 'success');
    // Update save button state
    _ytUpdateSaveButton(lens, true);
}
window._ytSaveInsight = _ytSaveInsight;

function _ytRemoveSavedInsight(index) {
    const saved = _ytGetSavedInsights();
    if (index >= 0 && index < saved.length) {
        saved.splice(index, 1);
        _ytSetSavedInsights(saved);
    }
}
window._ytRemoveSavedInsight = _ytRemoveSavedInsight;

function _ytUpdateSaveButton(lens, isSaved) {
    const btn = document.getElementById(`yi-save-btn-${lens}`);
    if (btn) {
        if (isSaved) {
            btn.innerHTML = '<i data-lucide="bookmark-check" style="width:12px;height:12px;"></i> Saved';
            btn.style.color = 'var(--accent-light,#34d399)';
            btn.style.borderColor = 'var(--accent-border,rgba(16,185,129,0.3))';
        } else {
            btn.innerHTML = '<i data-lucide="bookmark" style="width:12px;height:12px;"></i> Save';
            btn.style.color = '';
            btn.style.borderColor = '';
        }
        lucide.createIcons();
    }
}

function _ytIsInsightSaved(videoId, lens, language) {
    const saved = _ytGetSavedInsights();
    return saved.some(s => s.video_id === videoId && s.lens === lens && s.language === language);
}

function _ytBuildSaveButton(lensName) {
    if (!_ytCurrentVideoId || lensName === 'transcript') return '';
    const videoTitle = document.getElementById('yi-title')?.textContent || '';
    const isSaved = _ytIsInsightSaved(_ytCurrentVideoId, lensName, _ytCurrentLang);
    const savedStyle = isSaved ? 'color:var(--accent-light,#34d399);border-color:var(--accent-border,rgba(16,185,129,0.3));' : '';
    const savedLabel = isSaved ? 'Saved' : 'Save';
    const savedIcon = isSaved ? 'bookmark-check' : 'bookmark';
    return `<button id="yi-save-btn-${lensName}" onclick="_ytDoSaveInsight('${lensName}')" class="yi-header-btn" style="padding:4px 10px;font-size:11px;gap:4px;${savedStyle}" title="Save this insight for later reference">
        <i data-lucide="${savedIcon}" style="width:12px;height:12px;"></i> ${savedLabel}
    </button>`;
}

function _ytDoSaveInsight(lensName) {
    if (!_ytCurrentVideoId) return;
    const videoTitle = document.getElementById('yi-title')?.textContent || 'Unknown Video';
    const langKey = `${lensName}_${_ytCurrentLang}`;
    const fallbackKey = `${lensName}_en`;
    const ins = _ytInsightsAll[langKey] || _ytInsightsAll[fallbackKey] || null;
    if (!ins) {
        if (typeof showToast === 'function') showToast('No insight data to save', 'error');
        return;
    }
    _ytSaveInsight(_ytCurrentVideoId, videoTitle, lensName, _ytCurrentLang, ins.content || '');
}
window._ytDoSaveInsight = _ytDoSaveInsight;

// Expose functions needed by onclick
window._ytProcessChannel = _ytProcessChannel;
window._ytToggleVideos = _ytToggleVideos;
window._ytShowInsights = _ytShowInsights;
window._ytShowLens = _ytShowLens;
window._ytDeleteChannel = _ytDeleteChannel;
