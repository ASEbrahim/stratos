// ═══════════════════════════════════════════════════════════
// YOUTUBE KNOWLEDGE BASE — Dedicated page view
// ═══════════════════════════════════════════════════════════

let _ykbChannels = [];
let _ykbVideos = {};       // keyed by channel db_id
let _ykbExpanded = null;   // currently expanded channel db_id
let _ykbActiveVideo = {};  // keyed by channel db_id → video object
let _ykbActiveLens = {};   // keyed by channel db_id → lens name
let _ykbInsights = {};     // legacy — kept for compat
let _ykbInsightsByLang = {}; // keyed by videoDbId → lensName → language → insight
let _ykbInsightsLoaded = {}; // keyed by videoDbId → boolean (all insights fetched)
let _ykbActiveLang = {};   // keyed by channel db_id → language code
let _ykbInitialized = false;

const _ykbAuthHeader = () => ({ 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '' });

// ── Styles injected once ──
const _ykbStyleId = 'ykb-styles';
function _ykbInjectStyles() {
    if (document.getElementById(_ykbStyleId)) return;
    const style = document.createElement('style');
    style.id = _ykbStyleId;
    style.textContent = `
        /* ═══ Theme-aware styles using CSS variables ═══ */
        .ykb-card { background: var(--bg-panel-solid, #0e1026); border: 1px solid var(--accent-border, rgba(16,185,129,0.2)); border-radius: 12px; overflow: hidden; transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 12px; }
        .ykb-card:hover { border-color: var(--accent-border, rgba(16,185,129,0.3)); box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 20px var(--accent-bg, rgba(16,185,129,0.06)); }
        .ykb-card.expanded { border-color: var(--accent, #10b981); box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px var(--accent-bg, rgba(16,185,129,0.1)); }
        .ykb-header { display: flex; align-items: center; gap: 12px; padding: 14px 18px; cursor: pointer; user-select: none; transition: background 0.2s; }
        .ykb-header:hover { background: var(--bg-hover, rgba(22,26,55,0.5)); }
        .ykb-card.expanded .ykb-header { box-shadow: inset 0 3px 0 0 var(--accent, #10b981), inset 0 3px 20px -6px var(--accent-bg, rgba(16,185,129,0.15)); }
        .ykb-avatar { width: 40px; height: 40px; border-radius: 50%; object-fit: cover; border: 2px solid var(--accent-border, rgba(16,185,129,0.2)); flex-shrink: 0; }
        .ykb-avatar-placeholder { width: 40px; height: 40px; border-radius: 50%; background: var(--bg-panel, rgba(14,16,38,0.82)); display: flex; align-items: center; justify-content: center; flex-shrink: 0; border: 2px solid var(--accent-border, rgba(16,185,129,0.2)); backdrop-filter: blur(8px); }
        .ykb-channel-name { font-size: 14px; font-weight: 600; color: var(--text-primary, #e2e8f0); }
        .ykb-channel-stats { font-size: 10px; color: var(--text-muted, #64748b); display: flex; gap: 12px; margin-top: 2px; }
        .ykb-progress-bar { height: 3px; background: var(--border, rgba(255,255,255,0.05)); border-radius: 2px; overflow: hidden; margin-top: 4px; }
        .ykb-progress-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }
        .ykb-collapsed-strip { display: flex; gap: 8px; padding: 0 18px 14px; overflow-x: auto; scrollbar-width: none; }
        .ykb-collapsed-strip::-webkit-scrollbar { display: none; }
        .ykb-thumb { width: 120px; height: 68px; border-radius: 6px; object-fit: cover; cursor: pointer; border: 2px solid var(--border, rgba(255,255,255,0.05)); transition: all 0.2s; flex-shrink: 0; opacity: 0.7; }
        .ykb-thumb:hover { opacity: 1; border-color: var(--accent-border, rgba(16,185,129,0.3)); transform: scale(1.05); box-shadow: 0 4px 16px rgba(0,0,0,0.4), 0 0 12px var(--accent-bg, rgba(16,185,129,0.1)); }
        .ykb-expanded-body { max-height: 0; overflow: hidden; transition: max-height 0.4s ease; position: relative; background: var(--bg-primary, #050810); }
        .ykb-expanded-body.open { max-height: 2000px; }
        .ykb-stars { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; }
        .ykb-player-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 18px; position: relative; z-index: 1; }
        @media (max-width: 900px) { .ykb-player-grid { grid-template-columns: 1fr; } }
        .ykb-player-wrap { background: #000; border-radius: 10px; overflow: hidden; aspect-ratio: 16/9; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
        .ykb-player-wrap iframe { width: 100%; height: 100%; border: 0; }
        .ykb-player-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: var(--text-faint, #475569); font-size: 13px; background: var(--bg-primary, #050810); }
        .ykb-insights-panel { background: var(--bg-panel, rgba(14,16,38,0.82)); border: 1px solid var(--border, rgba(255,255,255,0.05)); border-radius: 10px; display: flex; flex-direction: column; min-height: 280px; max-height: 450px; backdrop-filter: blur(var(--te-blur, 12px)); }
        .ykb-lens-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border, rgba(255,255,255,0.05)); overflow-x: auto; scrollbar-width: none; flex-shrink: 0; background: var(--bg-hover, rgba(22,26,55,0.3)); }
        .ykb-lens-tabs::-webkit-scrollbar { display: none; }
        .ykb-lens-tab { padding: 8px 14px; font-size: 11px; font-weight: 500; color: var(--text-faint, #475569); cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap; transition: all 0.2s; }
        .ykb-lens-tab:hover { color: var(--text-secondary, #94a3b8); background: var(--bg-hover, rgba(22,26,55,0.3)); }
        .ykb-lens-tab.active { color: var(--accent-light, #34d399); border-bottom-color: var(--accent, #10b981); text-shadow: 0 0 8px var(--accent-bg, rgba(16,185,129,0.3)); }
        .ykb-lens-content { padding: 14px; overflow-y: auto; flex: 1; font-size: 12px; line-height: 1.7; color: var(--text-primary, #e2e8f0); white-space: pre-wrap; word-break: break-word; }
        .ykb-lens-content::-webkit-scrollbar { width: 4px; }
        .ykb-lens-content::-webkit-scrollbar-thumb { background: var(--border-strong, rgba(255,255,255,0.1)); border-radius: 2px; }
        .ykb-video-strip { display: flex; gap: 6px; padding: 10px 18px; overflow-x: auto; scrollbar-width: none; border-top: 1px solid var(--border, rgba(255,255,255,0.05)); position: relative; z-index: 1; }
        .ykb-video-strip::-webkit-scrollbar { display: none; }
        .ykb-strip-item { flex-shrink: 0; width: 100px; cursor: pointer; text-align: center; }
        .ykb-strip-thumb { width: 100px; height: 56px; border-radius: 4px; object-fit: cover; border: 2px solid var(--border, rgba(255,255,255,0.05)); transition: all 0.2s; }
        .ykb-strip-item.active .ykb-strip-thumb { border-color: var(--accent, #10b981); box-shadow: 0 0 12px var(--accent-bg, rgba(16,185,129,0.2)); }
        .ykb-strip-item:hover .ykb-strip-thumb { border-color: var(--accent-border, rgba(16,185,129,0.3)); }
        .ykb-strip-title { font-size: 9px; color: var(--text-muted, #64748b); margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .ykb-status { display: inline-flex; align-items: center; gap: 4px; font-size: 9px; padding: 2px 6px; border-radius: 4px; font-weight: 500; }
        .ykb-status-complete { background: var(--accent-bg, rgba(16,185,129,0.1)); color: var(--accent-light, #34d399); }
        .ykb-status-pending { background: rgba(251,191,36,0.12); color: #fbbf24; }
        .ykb-status-failed { background: rgba(239,68,68,0.12); color: #ef4444; }
        .ykb-status-transcribed { background: rgba(96,165,250,0.12); color: #60a5fa; }
        .ykb-status-transcribing { background: rgba(96,165,250,0.12); color: #60a5fa; animation: ykb-pulse 1.5s ease infinite; }
        @keyframes ykb-pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        .ykb-actions { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; }
        .ykb-btn { padding: 5px 12px; border-radius: 6px; font-size: 11px; font-weight: 500; cursor: pointer; border: 1px solid var(--border, rgba(255,255,255,0.05)); transition: all 0.2s; display: inline-flex; align-items: center; gap: 5px; background: var(--bg-hover, rgba(22,26,55,0.5)); color: var(--text-secondary, #94a3b8); backdrop-filter: blur(8px); }
        .ykb-btn:hover { border-color: var(--accent-border, rgba(16,185,129,0.3)); color: var(--accent-light, #34d399); background: var(--accent-bg, rgba(16,185,129,0.1)); box-shadow: 0 0 12px var(--accent-bg, rgba(16,185,129,0.08)); }
        .ykb-btn-primary { background: var(--accent-bg, rgba(16,185,129,0.1)); color: var(--accent-light, #34d399); border-color: var(--accent-border, rgba(16,185,129,0.25)); }
        .ykb-btn-primary:hover { background: var(--accent-bg, rgba(16,185,129,0.2)); box-shadow: 0 0 20px var(--accent-bg, rgba(16,185,129,0.15)), 0 0 0 1px var(--accent-border, rgba(16,185,129,0.3)); }
        .ykb-btn-sm { padding: 3px 8px; font-size: 10px; }
        .ykb-vid-actions { display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; position: relative; z-index: 1; padding: 0 18px 10px; }
        .ykb-btn-secondary { background: rgba(30,41,59,0.8); color: #94a3b8; border: 1px solid rgba(51,65,85,0.5); }
        .ykb-btn-secondary:hover { border-color: rgba(52,211,153,0.4); color: #e2e8f0; }
        .ykb-btn-add { background: rgba(52,211,153,0.1); color: #34d399; border: 1px dashed rgba(52,211,153,0.3); }
        .ykb-btn-add:hover { background: rgba(52,211,153,0.2); border-style: solid; }
        .ykb-empty { text-align: center; padding: 60px 20px; color: #475569; }
        .ykb-empty-icon { font-size: 48px; margin-bottom: 12px; opacity: 0.3; }
        .ykb-toast { position: fixed; bottom: 24px; right: 24px; padding: 10px 18px; border-radius: 8px; font-size: 12px; font-weight: 500; z-index: 9999; animation: ykbSlideUp 0.3s ease; }
        @keyframes ykbSlideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .ykb-toolbar { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .ykb-title { font-size: 20px; font-weight: 700; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
        .ykb-subtitle { font-size: 11px; color: #64748b; }
    `;
    document.head.appendChild(style);
}

// ── Entry point ──
function initYouTubeKB() {
    _ykbInjectStyles();
    _ykbLoadChannels();
}

async function _ykbLoadChannels() {
    const panel = document.getElementById('youtube-kb-panel');
    if (!panel) return;

    panel.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-muted,#64748b);"><div style="width:20px;height:20px;border:2px solid var(--border-strong,rgba(255,255,255,0.1));border-top-color:var(--accent,#10b981);border-radius:50%;animation:ykb-pulse 0.8s linear infinite;margin:0 auto 12px;"></div>Loading channels...</div>';

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000);
        const res = await fetch('/api/youtube/channels', { headers: _ykbAuthHeader(), signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok) throw new Error(`HTTP ${res.status} — ${res.statusText}`);
        const data = await res.json();
        _ykbChannels = data.channels || [];
    } catch (err) {
        const msg = err.name === 'AbortError' ? 'Request timed out' : err.message;
        panel.innerHTML = `<div style="text-align:center;padding:40px;">
            <div style="color:#ef4444;font-size:13px;margin-bottom:12px;">Failed to load channels: ${msg}</div>
            <button onclick="initYouTubeKB()" style="padding:8px 18px;border-radius:8px;border:1px solid var(--accent-border,rgba(16,185,129,0.2));background:var(--accent-bg,rgba(16,185,129,0.1));color:var(--accent-light,#34d399);font-size:12px;font-weight:600;cursor:pointer;">Retry</button>
        </div>`;
        return;
    }

    _ykbRender();
}

function _ykbRender() {
    const panel = document.getElementById('youtube-kb-panel');
    if (!panel) return;

    let html = '';

    // Toolbar
    html += `<div class="ykb-toolbar">
        <div>
            <div class="ykb-title"><i data-lucide="play-circle" class="w-5 h-5" style="color:var(--accent,#10b981);"></i> YouTube Knowledge Base</div>
            <div class="ykb-subtitle">${_ykbChannels.length} channel${_ykbChannels.length !== 1 ? 's' : ''} tracked</div>
        </div>
        <div style="flex:1;"></div>
        <button class="ykb-btn" onclick="_ykbOpenLensGuide()" title="Lens Guide — what each analysis mode does">
            <i data-lucide="book-open" class="w-3.5 h-3.5"></i> Guide
        </button>
        <button onclick="_ykbAddChannel()" style="padding:9px 20px;border-radius:10px;border:none;background:var(--accent,#10b981);color:var(--bg-primary,#050810);font-size:12px;font-weight:700;cursor:pointer;display:inline-flex;align-items:center;gap:6px;box-shadow:0 4px 16px var(--accent-bg,rgba(16,185,129,0.2));transition:all 0.2s;" onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 6px 24px var(--accent-bg)'" onmouseout="this.style.transform='';this.style.boxShadow='0 4px 16px var(--accent-bg)'">
            <i data-lucide="plus" class="w-4 h-4"></i> Add Channel
        </button>
    </div>`;

    if (_ykbChannels.length === 0) {
        html += `<div class="ykb-empty">
            <div class="ykb-empty-icon">📺</div>
            <div style="font-size:14px;font-weight:600;color:#94a3b8;margin-bottom:6px;">No channels yet</div>
            <div style="font-size:12px;color:#64748b;margin-bottom:16px;">Add a YouTube channel to start building your knowledge base.</div>
            <button class="ykb-btn ykb-btn-primary" onclick="_ykbAddChannel()"><i data-lucide="plus" class="w-3.5 h-3.5"></i> Add Channel</button>
        </div>`;
    } else {
        _ykbChannels.forEach(ch => {
            const isExpanded = _ykbExpanded === ch.id;
            const videos = _ykbVideos[ch.id] || [];
            const totalVids = videos.length || ch.video_count || 0;
            const readyVids = videos.filter(v => v.status === 'complete' || v.status === 'transcribed').length;
            const pct = totalVids > 0 ? Math.round((readyVids / totalVids) * 100) : 0;

            html += `<div class="ykb-card ${isExpanded ? 'expanded' : ''}" data-ykb-ch="${ch.id}">`;

            // Header
            html += `<div class="ykb-header" onclick="_ykbToggleChannel(${ch.id})">`;
            if (ch.avatar_url) {
                html += `<img src="${_ykbEsc(ch.avatar_url)}" class="ykb-avatar" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
                         <div class="ykb-avatar-placeholder" style="display:none;"><i data-lucide="user" class="w-5 h-5 text-slate-500"></i></div>`;
            } else {
                html += `<div class="ykb-avatar-placeholder"><i data-lucide="user" class="w-5 h-5 text-slate-500"></i></div>`;
            }
            html += `<div style="flex:1;min-width:0;">
                        <div class="ykb-channel-name">${_ykbEsc(ch.channel_name || ch.channel_id)}</div>
                        <div class="ykb-channel-stats">
                            <span>${totalVids} video${totalVids !== 1 ? 's' : ''}</span>
                            <span>${readyVids} ready</span>${videos.some(v => v.status === "transcribing" || v.status === "extracting") ? '<span style="color:var(--accent-light,#34d399);animation:ykb-pulse 1.5s ease infinite;display:inline-flex;align-items:center;gap:3px;"><span style="width:6px;height:6px;border-radius:50%;background:var(--accent,#10b981);display:inline-block;"></span> Transcribing...</span>' : ""}
                            <span>${pct}% processed</span>
                        </div>
                        <div class="ykb-progress-bar">
                            <div class="ykb-progress-fill" style="width:${pct}%;background:linear-gradient(90deg,#34d399,#6ee7b7);"></div>
                        </div>
                    </div>`;
            // Action buttons (stop propagation)
            html += `<div class="ykb-actions" onclick="event.stopPropagation()">
                        <button class="ykb-btn ykb-btn-secondary" onclick="_ykbTranscribeAll(${ch.id})" title="Transcribe all videos & lyrics for all videos">
                            <i data-lucide="file-text" class="w-3 h-3"></i> Transcribe All
                        </button>
                        <button class="ykb-btn ykb-btn-secondary" onclick="_ykbProcessAll(${ch.id})" title="Run AI processing on all videos">
                            <i data-lucide="sparkles" class="w-3 h-3"></i> Process
                        </button>
                        <button class="ykb-btn ykb-btn-secondary" onclick="_ykbTranslateAll(${ch.id})" title="Translate all videos in this channel">
                            <i data-lucide="languages" class="w-3 h-3"></i> Translate All
                        </button>
                        <button class="ykb-btn ykb-btn-secondary" onclick="_ykbExportChannel(${ch.id})" title="Export channel data as JSON">
                            <i data-lucide="download" class="w-3 h-3"></i> Export
                        </button>
                        <button class="ykb-btn" onclick="_ykbDeleteChannel(${ch.id},'${_ykbEsc(ch.channel_name)}')" title="Delete channel" style="color:#ef4444;border-color:rgba(239,68,68,0.1);">
                            <i data-lucide="trash-2" class="w-3 h-3"></i>
                        </button>
                    </div>`;
            html += `<i data-lucide="${isExpanded ? 'chevron-up' : 'chevron-down'}" class="w-4 h-4 text-slate-500 flex-shrink-0"></i>`;
            html += `</div>`; // end header

            // Collapsed thumbnail strip
            if (!isExpanded && videos.length > 0) {
                html += `<div class="ykb-collapsed-strip">`;
                videos.slice(0, 12).forEach(v => {
                    const thumbUrl = `https://img.youtube.com/vi/${_ykbEsc(v.video_id)}/mqdefault.jpg`;
                    html += `<img src="${thumbUrl}" class="ykb-thumb" alt="${_ykbEsc(v.title)}" title="${_ykbEsc(v.title)}" onclick="event.stopPropagation();_ykbExpandAndPlay(${ch.id},${v.id})">`;
                });
                html += `</div>`;
            }

            // Expanded body
            html += `<div class="ykb-expanded-body ${isExpanded ? 'open' : ''}" id="ykb-body-${ch.id}">`;
            if (isExpanded) {
                html += _ykbRenderExpanded(ch.id);
            }
            html += `</div>`;

            html += `</div>`; // end card
        });
    }

    panel.innerHTML = html;
    if (typeof lucide !== 'undefined') lucide.createIcons();

    // Initialize stars for expanded channel
    if (_ykbExpanded !== null) {
        _ykbInitStars(_ykbExpanded);
    }
}

function _ykbRenderExpanded(chId) {
    const videos = _ykbVideos[chId] || [];
    const activeVid = _ykbActiveVideo[chId];
    const activeLens = _ykbActiveLens[chId] || 'transcript';

    let html = '';

    // Stars canvas background
    html += `<canvas class="ykb-stars" id="ykb-stars-${chId}"></canvas>`;

    // Player + Insights grid
    html += `<div class="ykb-player-grid">`;

    // Player
    html += `<div>`;
    html += `<div class="ykb-player-wrap" id="ykb-player-${chId}">`;
    if (activeVid) {
        html += `<iframe src="https://www.youtube.com/embed/${_ykbEsc(activeVid.video_id)}?rel=0&modestbranding=1" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>`;
    } else {
        html += `<div class="ykb-player-placeholder"><span>Select a video to play</span></div>`;
    }
    html += `</div>`;
    // Video title + status below player
    if (activeVid) {
        const statusClass = _ykbStatusClass(activeVid.status);
        const method = activeVid.transcript_method || '';
        html += `<div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
            <div style="flex:1;min-width:0;">
                <div style="font-size:13px;font-weight:600;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-shadow:0 0 6px rgba(52,211,153,0.1);">${_ykbEsc(activeVid.title)}</div>
            </div>
            <span class="ykb-status ${statusClass}" ${activeVid.error_message ? `title="${_ykbEsc(activeVid.error_message)}" style="cursor:help;"` : ''}>${_ykbEsc(activeVid.status || 'pending')}</span>
            ${method ? `<span style="font-size:9px;padding:2px 6px;border-radius:4px;background:rgba(167,139,250,0.1);color:#a78bfa;">${_ykbEsc(method)}</span>` : ''}
        </div>`;
        // Video action toolbar — prominent, full width
        html += `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--bg-panel,rgba(14,16,38,0.82));border:1px solid var(--border,rgba(255,255,255,0.05));backdrop-filter:blur(8px);">
            <button class="ykb-btn ykb-btn-primary" onclick="event.stopPropagation();_ykbRetranscribe(${activeVid.id})" title="Re-transcribe this video"><i data-lucide="refresh-cw" style="width:12px;height:12px;"></i> Re-transcribe</button>
            <button class="ykb-btn" onclick="event.stopPropagation();_ykbExtractLens(${activeVid.id},'summary')" title="Generate AI summary"><i data-lucide="file-text" style="width:12px;height:12px;"></i> Extract Summary</button>
            <button class="ykb-btn" onclick="event.stopPropagation();_ykbExtractLens(${activeVid.id},'eloquence')" title="Extract vocabulary & eloquence"><i data-lucide="sparkles" style="width:12px;height:12px;"></i> Eloquence</button>
            <button class="ykb-btn" onclick="event.stopPropagation();_ykbExtractLens(${activeVid.id},'narrations')" title="Extract narrations & sources"><i data-lucide="message-circle" style="width:12px;height:12px;"></i> Narrations</button>
            <button class="ykb-btn" onclick="event.stopPropagation();_ykbTranslate(${activeVid.id})" title="Translate transcript to another language"><i data-lucide="languages" style="width:12px;height:12px;"></i> Translate</button>
            <button class="ykb-btn" onclick="event.stopPropagation();_ykbShowCaptions(${activeVid.id},'${_ykbEsc(activeVid.video_id)}','${_ykbEsc(activeVid.title)}')" title="View timed YouTube captions"><i data-lucide="subtitles" style="width:12px;height:12px;"></i> Captions</button>
            <div style="flex:1;"></div>
            <button class="ykb-btn" onclick="event.stopPropagation();_ykbCancelTranscribe(${activeVid.id})" title="Cancel and reset to pending" style="color:#ef4444;border-color:rgba(239,68,68,0.1);"><i data-lucide="x-circle" style="width:12px;height:12px;"></i> Reset</button>
        </div>`;
    }
    html += `</div>`;

    // Insights panel
    html += `<div class="ykb-insights-panel" id="ykb-insights-${chId}">`;
    if (activeVid) {
        const lenses = ['transcript', 'summary', 'eloquence', 'narrations', 'history', 'spiritual', 'politics'];
        html += `<div class="ykb-lens-tabs">`;
        lenses.forEach(lens => {
            const label = lens === 'transcript' ? 'Transcript' : lens.charAt(0).toUpperCase() + lens.slice(1);
            html += `<div class="ykb-lens-tab ${activeLens === lens ? 'active' : ''}" data-lens="${lens}" onclick="_ykbSetLens(${chId},'${lens}')">${label}</div>`;
        });
        html += `</div>`;
        html += `<div class="ykb-lens-content" id="ykb-lens-content-${chId}">Loading...</div>`;
    } else {
        html += `<div style="display:flex;align-items:center;justify-content:center;flex:1;color:#475569;font-size:12px;">Select a video to view insights</div>`;
    }
    html += `</div>`;

    html += `</div>`; // end grid

    // Video strip
    if (videos.length > 0) {
        html += `<div class="ykb-video-strip">`;
        videos.forEach(v => {
            const isActive = activeVid && activeVid.id === v.id;
            const thumbUrl = `https://img.youtube.com/vi/${_ykbEsc(v.video_id)}/mqdefault.jpg`;
            const statusClass = _ykbStatusClass(v.status);
            const errTip = v.error_message ? ` title="${_ykbEsc(v.error_message)}"` : '';
            const _isTranscribing = v.status === 'transcribing' || v.status === 'extracting';
            html += `<div class="ykb-strip-item ${isActive ? 'active' : ''}" onclick="_ykbPlayVideo(${chId},${v.id})" style="position:relative;">
                <img src="${thumbUrl}" class="ykb-strip-thumb" alt="" style="${_isTranscribing ? 'opacity:0.4;' : ''}">
                ${_isTranscribing ? `<div style="position:absolute;top:8px;left:50%;transform:translateX(-50%);width:22px;height:22px;border:2.5px solid rgba(255,255,255,0.15);border-top-color:var(--accent,#10b981);border-radius:50%;animation:ykb-pulse 0.8s linear infinite;"></div>` : ''}
                <div class="ykb-strip-title" style="${_isTranscribing ? 'color:var(--accent-light,#34d399);font-weight:600;' : ''}">${_ykbEsc(v.title)}</div>
                <span class="ykb-status ${statusClass}" style="margin-top:2px;cursor:${v.error_message ? 'help' : 'default'};"${errTip}>${_ykbEsc(v.status || 'pending')}</span>
            </div>`;
        });
        html += `</div>`;
    }

    return html;
}

// ── Channel toggle ──
async function _ykbToggleChannel(chId) {
    if (_ykbExpanded === chId) {
        _ykbExpanded = null;
        _ykbRender();
        return;
    }

    _ykbExpanded = chId;

    // Load videos if not cached
    if (!_ykbVideos[chId]) {
        await _ykbLoadVideos(chId);
    }

    _ykbRender();

    // If there's an active video, load its insights
    if (_ykbActiveVideo[chId]) {
        _ykbLoadInsights(chId, _ykbActiveVideo[chId].id);
    }
}

async function _ykbExpandAndPlay(chId, videoDbId) {
    _ykbExpanded = chId;
    if (!_ykbVideos[chId]) {
        await _ykbLoadVideos(chId);
    }
    const vid = (_ykbVideos[chId] || []).find(v => v.id === videoDbId);
    if (vid) _ykbActiveVideo[chId] = vid;
    _ykbRender();
    if (vid) _ykbLoadInsights(chId, vid.id);
}

// ── Load videos for a channel ──
async function _ykbLoadVideos(chId) {
    try {
        const res = await fetch(`/api/youtube/videos/${chId}`, { headers: _ykbAuthHeader() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        _ykbVideos[chId] = data.videos || [];
    } catch (err) {
        console.error('[YKB] Failed to load videos for channel', chId, err);
        _ykbVideos[chId] = [];
    }
}

// ── Play a video ──
function _ykbPlayVideo(chId, videoDbId) {
    const videos = _ykbVideos[chId] || [];
    const vid = videos.find(v => v.id === videoDbId);
    if (!vid) return;

    _ykbActiveVideo[chId] = vid;
    _ykbActiveLens[chId] = _ykbActiveLens[chId] || 'transcript';

    // Update player iframe
    const playerWrap = document.getElementById(`ykb-player-${chId}`);
    if (playerWrap) {
        playerWrap.innerHTML = `<iframe src="https://www.youtube.com/embed/${_ykbEsc(vid.video_id)}?rel=0&modestbranding=1&autoplay=1" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>`;
    }

    // Re-render expanded body for updated strip/title
    const body = document.getElementById(`ykb-body-${chId}`);
    if (body) {
        body.innerHTML = _ykbRenderExpanded(chId);
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }

    // Load insights
    _ykbLoadInsights(chId, vid.id);
}

// ── Set active lens ──
function _ykbSetLens(chId, lens) {
    _ykbActiveLens[chId] = lens;
    const activeVid = _ykbActiveVideo[chId];
    if (!activeVid) return;

    // Update tab highlights
    const body = document.getElementById(`ykb-body-${chId}`);
    if (body) {
        body.querySelectorAll('.ykb-lens-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.lens === lens);
        });
    }

    // Load content for this lens
    _ykbLoadInsights(chId, activeVid.id);
}

// ── Load insights ──
async function _ykbLoadInsights(chId, videoDbId) {
    const contentEl = document.getElementById(`ykb-lens-content-${chId}`);
    if (!contentEl) return;

    const lens = _ykbActiveLens[chId] || 'transcript';
    contentEl.innerHTML = '<div style="color:#64748b;text-align:center;padding:20px;">Loading...</div>';

    const activeLang = _ykbActiveLang[chId] || null; // null = auto/default

    // Check cache — keyed by video_lens_language
    const allCached = _ykbInsightsLoaded[videoDbId];
    if (allCached) {
        _ykbRenderLensContent(contentEl, videoDbId, lens, activeLang, chId);
        return;
    }

    try {
        const res = await fetch(`/api/youtube/insights/${videoDbId}?language=all`, { headers: _ykbAuthHeader() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const insights = data.insights || [];

        // Cache all insights keyed by video_lens_lang
        if (!_ykbInsightsByLang[videoDbId]) _ykbInsightsByLang[videoDbId] = {};
        insights.forEach(ins => {
            const lensName = ins.lens_name || ins.lens || 'unknown';
            const lang = ins.language || 'en';
            if (!_ykbInsightsByLang[videoDbId][lensName]) _ykbInsightsByLang[videoDbId][lensName] = {};
            _ykbInsightsByLang[videoDbId][lensName][lang] = ins;
        });

        // Always inject transcript_text from the video row as fallback
        if (data.transcript_text) {
            if (!_ykbInsightsByLang[videoDbId]['transcript']) {
                _ykbInsightsByLang[videoDbId]['transcript'] = {};
            }
            const tLang = data.transcript_language || 'en';
            // Use it if no transcript insight exists for this language, or if the existing one is empty
            const existing = _ykbInsightsByLang[videoDbId]['transcript'][tLang];
            if (!existing || !existing.content || existing.content.length < 20) {
                _ykbInsightsByLang[videoDbId]['transcript'][tLang] = {
                    content: JSON.stringify({ transcript: data.transcript_text }),
                    language: tLang
                };
            }
        }

        _ykbInsightsLoaded[videoDbId] = true;
        _ykbRenderLensContent(contentEl, videoDbId, lens, activeLang, chId);
    } catch (err) {
        contentEl.innerHTML = `<div style="color:var(--text-faint);text-align:center;padding:20px;">Failed to load insights: ${err.message}</div>`;
    }
}

function _ykbRenderLensContent(contentEl, videoDbId, lens, forceLang, chId) {
    const lensData = (_ykbInsightsByLang[videoDbId] || {})[lens] || {};
    const availLangs = Object.keys(lensData);

    if (!availLangs.length) {
        // Check if the video is currently being transcribed
        const activeVid = _ykbActiveVideo[chId];
        const isProcessing = activeVid && (activeVid.status === 'transcribing' || activeVid.status === 'extracting' || activeVid.status === 'pending');
        if (isProcessing) {
            contentEl.innerHTML = `<div style="text-align:center;padding:30px;"><div style="width:24px;height:24px;border:2px solid var(--border-strong,rgba(255,255,255,0.1));border-top-color:var(--accent,#10b981);border-radius:50%;animation:ykb-pulse 0.8s linear infinite;margin:0 auto 12px;"></div><div style="color:var(--text-muted,#64748b);font-size:12px;">${activeVid.status === 'pending' ? 'Queued for processing...' : 'Transcribing...'}</div><div style="color:var(--text-faint,#475569);font-size:10px;margin-top:4px;">This may take a minute</div></div>`;
        } else {
            contentEl.innerHTML = `<div style="color:var(--text-faint,#475569);text-align:center;padding:20px;">No ${lens} data yet.<br><span style="font-size:10px;">Click "Re-transcribe" or "Extract Summary" above.</span></div>`;
        }
        return;
    }

    // Pick language: forced > active > first available
    let lang = forceLang && lensData[forceLang] ? forceLang : availLangs[0];

    // Build language switcher pills if multiple languages
    let langHtml = '';
    if (availLangs.length > 1) {
        const langLabels = { en:'EN', ar:'AR', ja:'JA', ko:'KO', zh:'ZH', fr:'FR', de:'DE', es:'ES', ru:'RU' };
        const langFull = { en:'English', ar:'Arabic', ja:'Japanese', ko:'Korean', zh:'Chinese', fr:'French', de:'German', es:'Spanish', ru:'Russian' };
        langHtml = `<div style="display:flex;gap:4px;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--border,rgba(255,255,255,0.05));flex-wrap:wrap;align-items:center;"><span style="font-size:9px;color:var(--text-faint,#475569);margin-right:4px;text-transform:uppercase;letter-spacing:0.05em;">Lang:</span>`;
        availLangs.forEach(l => {
            const isActive = l === lang;
            const label = langLabels[l] || l.toUpperCase();
            const title = langFull[l] || l;
            langHtml += `<button onclick="_ykbSwitchLang(${chId},'${l}')" title="${title}" style="padding:3px 10px;border-radius:10px;font-size:10px;font-weight:700;cursor:pointer;transition:all 0.2s;letter-spacing:0.03em;border:1px solid ${isActive ? 'var(--accent-border,rgba(16,185,129,0.3))' : 'var(--border,rgba(255,255,255,0.05))'};background:${isActive ? 'var(--accent-bg,rgba(16,185,129,0.1))' : 'transparent'};color:${isActive ? 'var(--accent-light,#34d399)' : 'var(--text-muted,#64748b)'};">${label}</button>`;
        });
        langHtml += '</div>';
    }

    const insight = lensData[lang];
    contentEl.innerHTML = langHtml + _ykbFormatInsight(insight, lens);
}

function _ykbSwitchLang(chId, lang) {
    _ykbActiveLang[chId] = lang;
    const activeVid = _ykbActiveVideo[chId];
    if (!activeVid) return;
    const contentEl = document.getElementById(`ykb-lens-content-${chId}`);
    if (!contentEl) return;
    const lens = _ykbActiveLens[chId] || 'transcript';
    _ykbRenderLensContent(contentEl, activeVid.id, lens, lang, chId);
}

function _ykbFormatInsight(insight, lens) {
    const raw = insight.content || insight.text || '';
    if (!raw) return `<div style="color:#475569;text-align:center;padding:20px;">No content available.</div>`;

    // Parse JSON content string
    let parsed;
    try { parsed = typeof raw === 'string' ? JSON.parse(raw) : raw; } catch(e) { parsed = raw; }

    if (lens === 'summary') {
        const summary = parsed.summary || (typeof parsed === 'string' ? parsed : '');
        const takeaways = parsed.key_takeaways || [];
        let html = '';

        // Key takeaways — prominent separate section
        if (takeaways.length) {
            html += `<div style="margin-bottom:16px;">
                <div style="color:var(--accent-light,#34d399);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Key Takeaways</div>`;
            takeaways.forEach((t, i) => {
                html += `<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;">
                    <div style="width:22px;height:22px;border-radius:50%;background:var(--accent-bg,rgba(16,185,129,0.1));border:1px solid var(--accent-border,rgba(16,185,129,0.2));display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:10px;font-weight:700;color:var(--accent-light,#34d399);">${i+1}</div>
                    <div style="color:var(--text-primary,#e2e8f0);font-size:12px;line-height:1.6;padding-top:2px;">${_ykbEsc(t)}</div>
                </div>`;
            });
            html += '</div>';
            html += '<div style="height:1px;background:var(--border,rgba(255,255,255,0.05));margin:12px 0;"></div>';
        }

        // Summary text — well-paragraphed
        if (summary) {
            // Split on double newlines for real paragraphs, or sentences for long blocks
            let paragraphs = summary.split(/\n\n+/).filter(p => p.trim());
            if (paragraphs.length === 1 && paragraphs[0].length > 300) {
                // Single long block — split into ~2 sentence chunks
                const sentences = paragraphs[0].match(/[^.!?]+[.!?]+/g) || [paragraphs[0]];
                paragraphs = [];
                for (let i = 0; i < sentences.length; i += 2) {
                    paragraphs.push(sentences.slice(i, i + 2).join(' ').trim());
                }
            }
            html += '<div style="color:var(--text-secondary,#94a3b8);font-size:13px;line-height:2;">';
            paragraphs.forEach(p => {
                html += `<p style="margin-bottom:12px;">${_ykbEsc(p.trim())}</p>`;
            });
            html += '</div>';
        }
        return html || `<div style="color:var(--text-faint,#475569);text-align:center;padding:20px;">No summary content.</div>`;
    }

    if (lens === 'transcript') {
        const transcript = parsed.transcript || (typeof parsed === 'string' ? parsed : '');
        if (!transcript) return `<div style="color:#475569;text-align:center;padding:20px;">No transcript available.</div>`;
        // Format lyrics/transcript with line breaks
        const lines = transcript.split('\n').filter(l => l.trim());
        return lines.map(l => `<div style="color:#a78bfa;font-size:13px;line-height:2;padding:1px 6px;border-radius:4px;transition:background 0.15s;cursor:default;" onmouseover="this.style.background='rgba(167,139,250,0.06)'" onmouseout="this.style.background=''">${_ykbEsc(l)}</div>`).join('');
    }

    if (lens === 'eloquence') {
        const items = Array.isArray(parsed) ? parsed : [];
        if (!items.length) return `<div style="color:#475569;text-align:center;padding:20px;">No eloquence data.</div>`;
        return items.map(e => `<div style="margin-bottom:10px;padding:8px 10px;border-radius:8px;background:rgba(30,41,59,0.3);border:1px solid rgba(51,65,85,0.2);">
            <span style="color:#fbbf24;font-weight:600;font-size:13px;">${_ykbEsc(e.term || '')}</span>
            <span style="color:#475569;font-size:10px;margin-left:6px;">${_ykbEsc(e.rarity || '')}</span>
            <div style="color:#94a3b8;font-size:11px;margin-top:3px;">${_ykbEsc(e.definition || '')}</div>
            ${e.context_quote ? `<div style="color:#64748b;font-size:10px;margin-top:3px;font-style:italic;">"${_ykbEsc(e.context_quote)}"</div>` : ''}
        </div>`).join('');
    }

    if (lens === 'narrations') {
        const items = Array.isArray(parsed) ? parsed : [];
        if (!items.length) return `<div style="color:#475569;text-align:center;padding:20px;">No narrations data.</div>`;
        return items.map(n => `<div style="margin-bottom:10px;padding:8px 10px;border-radius:8px;background:rgba(30,41,59,0.3);border:1px solid rgba(51,65,85,0.2);">
            <div style="color:#cbd5e1;font-size:12px;line-height:1.6;">${_ykbEsc(n.narration_text || '')}</div>
            ${n.speaker_attribution ? `<div style="color:#34d399;font-size:10px;margin-top:3px;">— ${_ykbEsc(n.speaker_attribution)}</div>` : ''}
            ${n.source_claimed ? `<div style="color:#64748b;font-size:10px;">Source: ${_ykbEsc(n.source_claimed)}</div>` : ''}
        </div>`).join('');
    }

    // Default: show raw text
    const text = typeof parsed === 'string' ? parsed : JSON.stringify(parsed, null, 2);
    return `<div style="color:#94a3b8;font-size:12px;line-height:1.8;white-space:pre-wrap;">${_ykbEsc(text)}</div>`;
}

// ── Actions ──
async function _ykbTranscribeAll(chId) {
    _ykbShowToast('Extracting transcripts...', '#fbbf24');
    try {
        const res = await fetch(`/api/youtube/extract-all/${chId}`, { method: 'POST', headers: _ykbAuthHeader() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        _ykbShowToast(data.message || 'Extraction started', '#34d399');
        // Refresh videos after a delay
        setTimeout(() => { _ykbLoadVideos(chId).then(() => _ykbRender()); }, 2000);
    } catch (err) {
        _ykbShowToast('Extraction failed: ' + err.message, '#ef4444');
    }
}

async function _ykbProcessAll(chId) {
    _ykbShowToast('Discovering & processing videos...', '#fbbf24');
    try {
        const res = await fetch(`/api/youtube/process/${chId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
            body: JSON.stringify({})
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const msg = data.new_videos ? `Found ${data.new_videos} new videos` : (data.reset ? `Reset ${data.reset} videos` : 'Re-processing started');
        _ykbShowToast(msg, '#34d399');
        // Reload videos immediately and again after delay
        await _ykbLoadVideos(chId);
        _ykbRender();
        setTimeout(() => { _ykbLoadVideos(chId).then(() => _ykbRender()); }, 5000);
    } catch (err) {
        _ykbShowToast('Re-processing failed: ' + err.message, '#ef4444');
    }
}

function _ykbAddChannel() {
    const existing = document.getElementById('ykb-add-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'ykb-add-modal';
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);';
    modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

    const allLenses = ['transcript', 'summary', 'eloquence', 'narrations'];

    modal.innerHTML = `<div style="background:var(--bg-panel-solid,#0e1026);border:1px solid var(--accent-border,rgba(16,185,129,0.2));border-radius:16px;padding:28px;max-width:440px;width:92%;box-shadow:0 20px 60px rgba(0,0,0,0.5),0 0 30px var(--accent-bg,rgba(16,185,129,0.08));">
        <div style="font-size:16px;font-weight:700;color:var(--text-primary,#e2e8f0);margin-bottom:4px;">Add YouTube Channel</div>
        <div style="font-size:11px;color:var(--text-muted,#64748b);margin-bottom:16px;">Enter a channel URL, @handle, or video URL</div>

        <input id="ykb-add-url" type="text" placeholder="https://www.youtube.com/@channelname" style="width:100%;padding:10px 14px;border-radius:10px;border:1px solid var(--border-strong,rgba(255,255,255,0.1));background:var(--bg-input,rgba(8,10,22,0.7));color:var(--text-primary,#e2e8f0);font-size:13px;outline:none;transition:border-color 0.2s;" onfocus="this.style.borderColor='var(--accent-border)'" onblur="this.style.borderColor='var(--border-strong)'">

        <div style="margin-top:14px;">
            <div style="font-size:11px;font-weight:600;color:var(--text-secondary,#94a3b8);margin-bottom:8px;">Lenses to extract</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;">
                ${allLenses.map(l => `<label style="display:flex;align-items:center;gap:5px;padding:5px 12px;border-radius:8px;cursor:pointer;border:1px solid var(--border,rgba(255,255,255,0.05));background:var(--bg-hover,rgba(22,26,55,0.5));font-size:11px;color:var(--text-secondary,#94a3b8);transition:all 0.15s;" onmouseover="this.style.borderColor='var(--accent-border)'" onmouseout="this.style.borderColor='var(--border)'"><input type="checkbox" class="ykb-add-lens" value="${l}" ${l === 'transcript' || l === 'summary' ? 'checked' : ''} style="accent-color:var(--accent,#10b981);width:13px;height:13px;">${l.charAt(0).toUpperCase() + l.slice(1)}</label>`).join('')}
            </div>
        </div>

        <div style="display:flex;gap:8px;margin-top:18px;justify-content:flex-end;">
            <button onclick="document.getElementById('ykb-add-modal').remove()" style="padding:8px 18px;border-radius:8px;border:1px solid var(--border,rgba(255,255,255,0.05));background:none;color:var(--text-muted,#64748b);font-size:12px;cursor:pointer;">Cancel</button>
            <button onclick="_ykbDoAddChannel()" style="padding:8px 22px;border-radius:8px;border:none;background:var(--accent,#10b981);color:var(--bg-primary,#050810);font-size:12px;font-weight:700;cursor:pointer;box-shadow:0 4px 16px var(--accent-bg,rgba(16,185,129,0.2));transition:all 0.2s;" onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 6px 24px var(--accent-bg)'" onmouseout="this.style.transform='';this.style.boxShadow='0 4px 16px var(--accent-bg)'">Add Channel</button>
        </div>
    </div>`;

    document.body.appendChild(modal);
    setTimeout(() => document.getElementById('ykb-add-url')?.focus(), 100);

    // Allow Enter key to submit
    document.getElementById('ykb-add-url').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') _ykbDoAddChannel();
    });
}

async function _ykbDoAddChannel() {
    const urlInput = document.getElementById('ykb-add-url');
    const url = urlInput?.value?.trim();
    if (!url) { urlInput?.focus(); return; }

    const lenses = [...document.querySelectorAll('.ykb-add-lens:checked')].map(cb => cb.value);

    document.getElementById('ykb-add-modal')?.remove();
    _ykbShowToast('Adding channel...', '#fbbf24');
    try {
        const res = await fetch('/api/youtube/channels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
            body: JSON.stringify({ channel_url: url, lenses: lenses.length ? lenses : undefined })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
        }
        const addData = await res.json();
        _ykbShowToast('Channel added! Discovering videos...', '#34d399');
        await _ykbLoadChannels();

        // Auto-process: discover videos for the new channel
        const newChId = addData.channel?.id;
        if (newChId) {
            try {
                const procRes = await fetch(`/api/youtube/process/${newChId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
                    body: JSON.stringify({})
                });
                if (procRes.ok) {
                    const procData = await procRes.json();
                    _ykbShowToast(`Found ${procData.new_videos || 0} videos — transcription starting`, '#34d399');
                    await _ykbLoadVideos(newChId);
                    _ykbExpanded = newChId;
                    _ykbRender();
                }
            } catch(e) { /* process failed silently, user can click Re-process */ }
        }
    } catch (err) {
        _ykbShowToast('Failed: ' + err.message, '#ef4444');
    }
}

// ── Individual video actions ──
async function _ykbRetranscribe(videoDbId) {
    _ykbShowToast('Queuing re-transcription...', '#fbbf24');
    try {
        const res = await fetch('/api/youtube/retranscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
            body: JSON.stringify({ video_id: videoDbId })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        _ykbShowToast('Re-transcription queued', '#34d399');
        _ykbClearVideoCache(videoDbId);
    } catch (err) { _ykbShowToast('Failed: ' + err.message, '#ef4444'); }
}

async function _ykbExtractLens(videoDbId, lens) {
    _ykbShowToast(`Extracting ${lens}...`, '#fbbf24');
    try {
        const res = await fetch('/api/youtube/extract-lens', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
            body: JSON.stringify({ video_id: videoDbId, lens: lens, language: 'en', mode: 'replace' })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        _ykbShowToast(`${lens} extraction started`, '#34d399');
        // Clear cache and reload after delay
        _ykbClearVideoCache(videoDbId);
        setTimeout(() => {
            const chId = _ykbExpanded;
            if (chId) _ykbLoadInsights(chId, videoDbId);
        }, 3000);
    } catch (err) { _ykbShowToast('Failed: ' + err.message, '#ef4444'); }
}

async function _ykbTranslate(videoDbId) {
    // Show language picker popup
    const existing = document.getElementById('ykb-lang-picker');
    if (existing) existing.remove();

    const picker = document.createElement('div');
    picker.id = 'ykb-lang-picker';
    picker.style.cssText = 'position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);';
    picker.onclick = (e) => { if (e.target === picker) picker.remove(); };

    const langs = [
        { code: 'ar', label: 'Arabic' },
        { code: 'ja', label: 'Japanese' },
        { code: 'ko', label: 'Korean' },
        { code: 'zh', label: 'Chinese' },
        { code: 'fr', label: 'French' },
        { code: 'de', label: 'German' },
        { code: 'es', label: 'Spanish' },
        { code: 'ru', label: 'Russian' },
        { code: 'en', label: 'English' },
    ];

    picker.innerHTML = `<div style="background:var(--bg-panel-solid,#0e1026);border:1px solid var(--accent-border,rgba(16,185,129,0.2));border-radius:16px;padding:24px;max-width:360px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,0.5),0 0 30px var(--accent-bg,rgba(16,185,129,0.08));">
        <div style="font-size:14px;font-weight:700;color:var(--text-primary,#e2e8f0);margin-bottom:4px;">Translate Transcript</div>
        <div style="font-size:11px;color:var(--text-muted,#64748b);margin-bottom:16px;">Select target language</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
            ${langs.map(l => `<button onclick="_ykbDoTranslate(${videoDbId},'${l.code}');document.getElementById('ykb-lang-picker').remove();" style="padding:8px 4px;border-radius:8px;border:1px solid var(--border,rgba(255,255,255,0.05));background:var(--bg-hover,rgba(22,26,55,0.5));color:var(--text-secondary,#94a3b8);font-size:11px;font-weight:600;cursor:pointer;transition:all 0.2s;text-align:center;" onmouseover="this.style.borderColor='var(--accent-border)';this.style.color='var(--accent-light,#34d399)';this.style.background='var(--accent-bg,rgba(16,185,129,0.1))';this.style.boxShadow='0 0 12px var(--accent-bg,rgba(16,185,129,0.08))'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--text-secondary)';this.style.background='var(--bg-hover)';this.style.boxShadow=''">${l.label}</button>`).join('')}
        </div>
        <div style="margin-top:14px;text-align:right;"><button onclick="document.getElementById('ykb-lang-picker').remove()" style="padding:6px 14px;border-radius:8px;border:1px solid var(--border,rgba(255,255,255,0.05));background:none;color:var(--text-muted,#64748b);font-size:11px;cursor:pointer;">Cancel</button></div>
    </div>`;

    document.body.appendChild(picker);
}

function _ykbTranslateAll(chId) {
    const videos = (_ykbVideos[chId] || []).filter(v => v.transcript_text || v.status === 'complete' || v.status === 'transcribed');
    if (!videos.length) {
        _ykbShowToast('No transcribed videos to translate', '#ef4444');
        return;
    }

    // Reuse the same picker UI but batch translate
    const existing = document.getElementById('ykb-lang-picker');
    if (existing) existing.remove();

    const picker = document.createElement('div');
    picker.id = 'ykb-lang-picker';
    picker.style.cssText = 'position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);';
    picker.onclick = (e) => { if (e.target === picker) picker.remove(); };

    const langs = [
        { code: 'ar', label: 'Arabic' }, { code: 'ja', label: 'Japanese' }, { code: 'ko', label: 'Korean' },
        { code: 'zh', label: 'Chinese' }, { code: 'fr', label: 'French' }, { code: 'de', label: 'German' },
        { code: 'es', label: 'Spanish' }, { code: 'ru', label: 'Russian' }, { code: 'en', label: 'English' },
    ];

    picker.innerHTML = `<div style="background:var(--bg-panel-solid,#0e1026);border:1px solid var(--accent-border,rgba(16,185,129,0.2));border-radius:16px;padding:24px;max-width:360px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,0.5),0 0 30px var(--accent-bg,rgba(16,185,129,0.08));">
        <div style="font-size:14px;font-weight:700;color:var(--text-primary,#e2e8f0);margin-bottom:4px;">Translate All Videos</div>
        <div style="font-size:11px;color:var(--text-muted,#64748b);margin-bottom:16px;">${videos.length} videos will be translated</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
            ${langs.map(l => `<button onclick="_ykbDoTranslateAll(${chId},'${l.code}');document.getElementById('ykb-lang-picker').remove();" style="padding:8px 4px;border-radius:8px;border:1px solid var(--border,rgba(255,255,255,0.05));background:var(--bg-hover,rgba(22,26,55,0.5));color:var(--text-secondary,#94a3b8);font-size:11px;font-weight:600;cursor:pointer;transition:all 0.2s;text-align:center;" onmouseover="this.style.borderColor='var(--accent-border)';this.style.color='var(--accent-light,#34d399)';this.style.background='var(--accent-bg,rgba(16,185,129,0.1))'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--text-secondary)';this.style.background='var(--bg-hover)'">${l.label}</button>`).join('')}
        </div>
        <div style="margin-top:14px;text-align:right;"><button onclick="document.getElementById('ykb-lang-picker').remove()" style="padding:6px 14px;border-radius:8px;border:1px solid var(--border,rgba(255,255,255,0.05));background:none;color:var(--text-muted,#64748b);font-size:11px;cursor:pointer;">Cancel</button></div>
    </div>`;

    document.body.appendChild(picker);
}

async function _ykbDoTranslateAll(chId, lang) {
    const videos = (_ykbVideos[chId] || []).filter(v => v.transcript_text || v.status === 'complete' || v.status === 'transcribed');
    _ykbShowToast(`Translating ${videos.length} videos to ${lang}...`, '#fbbf24');
    let done = 0, failed = 0;
    for (const vid of videos) {
        try {
            const res = await fetch('/api/youtube/translate-transcript', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
                body: JSON.stringify({ video_id: vid.id, target_language: lang })
            });
            if (res.ok) { done++; _ykbClearVideoCache(vid.id); }
            else failed++;
        } catch (e) { failed++; }
    }
    _ykbShowToast(`Translated ${done}/${videos.length} videos${failed ? ` (${failed} failed)` : ''}`, done > 0 ? '#34d399' : '#ef4444');
}

async function _ykbDoTranslate(videoDbId, lang) {
    _ykbShowToast(`Translating to ${lang}...`, '#fbbf24');
    try {
        const res = await fetch('/api/youtube/translate-transcript', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
            body: JSON.stringify({ video_id: videoDbId, target_language: lang })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        _ykbShowToast(`Translated ${data.chars_translated || 0} chars to ${lang}`, '#34d399');
        // Clear cache so next load fetches all languages including new translation
        delete _ykbInsightsLoaded[videoDbId];
        delete _ykbInsightsByLang[videoDbId];
        // Switch to the translated language and reload
        if (_ykbExpanded) {
            _ykbActiveLang[_ykbExpanded] = lang;
            _ykbActiveLens[_ykbExpanded] = 'transcript';
            _ykbLoadInsights(_ykbExpanded, videoDbId);
        }
    } catch (err) { _ykbShowToast('Failed: ' + err.message, '#ef4444'); }
}

async function _ykbCancelTranscribe(videoDbId) {
    if (!confirm('Cancel transcription and reset this video to pending?')) return;
    try {
        const res = await fetch('/api/youtube/cancel-transcribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._ykbAuthHeader() },
            body: JSON.stringify({ video_id: videoDbId })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        _ykbShowToast('Video reset to pending', '#34d399');
        Object.keys(_ykbInsights).forEach(k => { if (k.startsWith(videoDbId + '_')) delete _ykbInsights[k]; });
        if (_ykbExpanded) {
            await _ykbLoadVideos(_ykbExpanded);
            _ykbRender();
        }
    } catch (err) { _ykbShowToast('Failed: ' + err.message, '#ef4444'); }
}

async function _ykbDeleteChannel(chId, name) {
    if (!confirm(`Delete "${name}" and all its videos/insights? This cannot be undone.`)) return;
    _ykbShowToast('Deleting channel...', '#fbbf24');
    try {
        const res = await fetch(`/api/youtube/channels/${chId}`, { method: 'DELETE', headers: _ykbAuthHeader() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        _ykbShowToast('Channel deleted', '#34d399');
        if (_ykbExpanded === chId) _ykbExpanded = null;
        delete _ykbVideos[chId];
        await _ykbLoadChannels();
    } catch (err) { _ykbShowToast('Failed: ' + err.message, '#ef4444'); }
}

function _ykbExportChannel(chId) {
    const token = typeof getAuthToken === 'function' ? getAuthToken() : '';
    window.open(`/api/youtube/export/${chId}?format=json&token=${encodeURIComponent(token)}`, '_blank');
}

function _ykbOpenLensGuide() {
    // Use the existing lens guide from youtube.js insights modal
    if (typeof _ytShowLensGuide === 'function') {
        // Need to open the insights modal first, then show the guide
        // Create a minimal dummy modal if none exists
        let modal = document.getElementById('yt-insights-modal');
        if (!modal) {
            // Open a dummy insights view to get the modal created, then show guide
            if (typeof _ytShowInsights === 'function') {
                // Find any video to open insights for
                const anyVid = Object.values(_ykbVideos).flat().find(v => v.status === 'complete' || v.status === 'transcribed');
                if (anyVid) {
                    _ytShowInsights(anyVid.id);
                    setTimeout(() => _ytShowLensGuide(), 300);
                    return;
                }
            }
        }
        _ytShowLensGuide();
    } else {
        _ykbShowToast('Guide not available — open a video first', '#fbbf24');
    }
}

async function _ykbShowCaptions(videoDbId, ytVideoId, title) {
    // Remove existing modal
    document.getElementById('ykb-captions-modal')?.remove();

    const modal = document.createElement('div');
    modal.id = 'ykb-captions-modal';
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.7);backdrop-filter:blur(6px);';
    modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

    // Arcane-themed container
    modal.innerHTML = `<div id="ykb-cap-container" style="position:relative;width:92%;max-width:680px;max-height:85vh;background:var(--bg-panel-solid,#0e1026);border:1px solid var(--accent-border,rgba(16,185,129,0.2));border-radius:16px;overflow:hidden;box-shadow:0 32px 80px rgba(0,0,0,0.6),0 0 40px var(--accent-bg,rgba(16,185,129,0.06));display:flex;flex-direction:column;">
        <canvas id="ykb-cap-stars" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:0;"></canvas>
        <div style="position:relative;z-index:1;padding:18px 20px;border-bottom:1px solid var(--border,rgba(255,255,255,0.05));box-shadow:inset 0 3px 0 0 var(--accent,#10b981),inset 0 3px 20px -6px var(--accent-bg,rgba(16,185,129,0.15));">
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="width:32px;height:32px;border-radius:8px;background:var(--accent-bg,rgba(16,185,129,0.1));border:1px solid var(--accent-border,rgba(16,185,129,0.2));display:flex;align-items:center;justify-content:center;">
                        <i data-lucide="subtitles" style="width:16px;height:16px;color:var(--accent-light,#34d399);"></i>
                    </div>
                    <div>
                        <div style="font-size:14px;font-weight:700;color:var(--text-primary,#e2e8f0);">Captions</div>
                        <div style="font-size:10px;color:var(--text-muted,#64748b);max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${_ykbEsc(title)}</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span id="ykb-cap-track-info" style="font-size:10px;color:var(--text-faint,#475569);"></span>
                    <button onclick="document.getElementById('ykb-captions-modal').remove()" style="background:none;border:none;color:var(--text-muted,#64748b);cursor:pointer;font-size:20px;line-height:1;padding:4px;">&times;</button>
                </div>
            </div>
        </div>
        <div id="ykb-cap-body" style="position:relative;z-index:1;flex:1;overflow-y:auto;padding:16px 20px;min-height:200px;">
            <div style="text-align:center;padding:30px;"><div style="width:20px;height:20px;border:2px solid var(--border-strong,rgba(255,255,255,0.1));border-top-color:var(--accent,#10b981);border-radius:50%;animation:ykb-pulse 0.8s linear infinite;margin:0 auto 10px;"></div><span style="font-size:11px;color:var(--text-muted,#64748b);">Loading captions...</span></div>
        </div>
    </div>`;

    document.body.appendChild(modal);
    if (typeof lucide !== 'undefined') lucide.createIcons();

    // Init stars for the modal
    _ykbInitCaptionStars();

    // Fetch captions
    try {
        const res = await fetch(`/api/youtube/captions/${videoDbId}`, { headers: _ykbAuthHeader() });
        const data = await res.json();
        const body = document.getElementById('ykb-cap-body');
        const trackInfo = document.getElementById('ykb-cap-track-info');
        if (!body) return;

        if (data.error || !data.captions || data.captions.length === 0) {
            // Show available tracks info
            let trackHtml = '';
            if (data.tracks && data.tracks.length > 0) {
                trackHtml = `<div style="margin-top:10px;font-size:10px;color:var(--text-faint);">Available tracks: ${data.tracks.map(t => `${t.language_name} (${t.language}${t.is_generated ? ', auto' : ''})`).join(', ')}</div>`;
            }
            body.innerHTML = `<div style="text-align:center;padding:30px;color:var(--text-muted,#64748b);">
                <i data-lucide="subtitles" style="width:32px;height:32px;opacity:0.3;margin:0 auto 10px;display:block;"></i>
                <div style="font-size:13px;">No captions available</div>
                <div style="font-size:10px;margin-top:4px;color:var(--text-faint);">${data.error || 'This video may not have subtitles'}</div>
                ${trackHtml}
            </div>`;
            if (typeof lucide !== 'undefined') lucide.createIcons();
            return;
        }

        if (trackInfo) trackInfo.textContent = `${data.language?.toUpperCase()} · ${data.count} segments`;

        // Render timed captions
        let html = '<div style="display:flex;flex-direction:column;gap:2px;">';
        data.captions.forEach(cap => {
            const mins = Math.floor(cap.start / 60);
            const secs = Math.floor(cap.start % 60);
            const ts = `${mins}:${secs.toString().padStart(2, '0')}`;
            html += `<div style="display:flex;gap:12px;padding:6px 8px;border-radius:6px;transition:background 0.15s;cursor:default;" onmouseover="this.style.background='var(--bg-hover,rgba(22,26,55,0.5))'" onmouseout="this.style.background=''">
                <span style="flex-shrink:0;width:42px;font-size:10px;font-weight:600;color:var(--accent-light,#34d399);font-variant-numeric:tabular-nums;padding-top:2px;">${ts}</span>
                <span style="font-size:12px;color:var(--text-primary,#e2e8f0);line-height:1.6;">${_ykbEsc(cap.text)}</span>
            </div>`;
        });
        html += '</div>';
        body.innerHTML = html;
    } catch (err) {
        const body = document.getElementById('ykb-cap-body');
        if (body) body.innerHTML = `<div style="text-align:center;padding:30px;color:#ef4444;">Failed to load captions: ${err.message}</div>`;
    }
}

function _ykbInitCaptionStars() {
    const canvas = document.getElementById('ykb-cap-stars');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const _accentStr = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#10b981';
    const _hm = _accentStr.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    const sr = _hm ? parseInt(_hm[1],16) : 16, sg = _hm ? parseInt(_hm[2],16) : 185, sb = _hm ? parseInt(_hm[3],16) : 129;

    function resize() {
        const el = canvas.parentElement;
        if (!el) return;
        canvas.width = el.offsetWidth;
        canvas.height = el.offsetHeight;
    }
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(canvas.parentElement);

    const stars = [];
    for (let i = 0; i < 60; i++) {
        const d = Math.random() * 4 + 0.1, dn = d / 4.1;
        stars.push({ x: Math.random(), y: Math.random(), r: 0.4 + dn * 1.8, a: 0.12 + dn * 0.5, dx: (Math.random()-0.5)*0.03, dy: (Math.random()-0.5)*0.02, phase: Math.random()*Math.PI*2 });
    }

    let _anim;
    function draw(t) {
        if (!document.getElementById('ykb-cap-stars')) { obs.disconnect(); return; }
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        for (const s of stars) {
            s.x += s.dx/W; s.y += s.dy/H;
            if (s.x<0) s.x=1; if (s.x>1) s.x=0; if (s.y<0) s.y=1; if (s.y>1) s.y=0;
            const f = 0.7 + 0.3*Math.sin(t*0.002+s.phase);
            ctx.beginPath(); ctx.arc(s.x*W, s.y*H, s.r, 0, Math.PI*2);
            ctx.fillStyle = `rgba(${sr},${sg},${sb},${s.a*f})`; ctx.fill();
        }
        // Constellation lines
        ctx.strokeStyle = `rgba(${sr},${sg},${sb},0.05)`; ctx.lineWidth = 0.5;
        for (let i=0;i<stars.length;i++) for (let j=i+1;j<stars.length;j++) {
            const dx=(stars[i].x-stars[j].x)*W, dy=(stars[i].y-stars[j].y)*H;
            if (dx*dx+dy*dy<6000) { ctx.beginPath(); ctx.moveTo(stars[i].x*W,stars[i].y*H); ctx.lineTo(stars[j].x*W,stars[j].y*H); ctx.stroke(); }
        }
        _anim = requestAnimationFrame(draw);
    }
    _anim = requestAnimationFrame(draw);
}

function _ykbClearVideoCache(videoDbId) {
    delete _ykbInsightsLoaded[videoDbId];
    delete _ykbInsightsByLang[videoDbId];
    Object.keys(_ykbInsights).forEach(k => { if (k.startsWith(videoDbId + '_')) delete _ykbInsights[k]; });
}

// ── Helpers ──
function _ykbStatusClass(status) {
    if (!status) return 'ykb-status-pending';
    if (status === 'complete') return 'ykb-status-complete';
    if (status === 'transcribed') return 'ykb-status-transcribed';
    if (status === 'transcribing' || status === 'extracting') return 'ykb-status-transcribing';
    if (status === 'failed') return 'ykb-status-failed';
    if (status === 'low_quality') return 'ykb-status-failed';
    return 'ykb-status-pending';
}

function _ykbEsc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function _ykbShowToast(msg, color) {
    // Remove existing toast
    document.querySelectorAll('.ykb-toast').forEach(t => t.remove());
    const toast = document.createElement('div');
    toast.className = 'ykb-toast';
    toast.style.background = color || '#34d399';
    toast.style.color = color === '#fbbf24' || color === '#34d399' ? '#0f172a' : '#fff';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── SSE integration: listen for video processing updates ──
(function _ykbListenSSE() {
    const origHandler = window._handleYouTubeSSE;
    window._handleYouTubeSSE = function(event) {
        // Call original handler for settings view
        if (typeof origHandler === 'function') origHandler(event);

        // Update KB view if visible
        const panel = document.getElementById('youtube-kb-panel');
        if (!panel || panel.classList.contains('hidden')) return;

        const { video_id, status } = event;
        if (!video_id || !status) return;

        // Update cached video status
        for (const chId of Object.keys(_ykbVideos)) {
            const vids = _ykbVideos[chId];
            const vid = vids.find(v => v.video_id === video_id);
            if (vid) {
                vid.status = status;
                // Clear insight cache for this video if status changed to complete
                if (status === 'complete') {
                    Object.keys(_ykbInsights).forEach(k => {
                        if (k.startsWith(vid.id + '_')) delete _ykbInsights[k];
                    });
                }
                // Re-render if this channel is expanded
                if (_ykbExpanded === parseInt(chId)) {
                    _ykbRender();
                    if (_ykbActiveVideo[chId] && _ykbActiveVideo[chId].id === vid.id) {
                        _ykbLoadInsights(parseInt(chId), vid.id);
                    }
                }
                break;
            }
        }
    };
})();

// ═══ SSE: Refresh insights when lens extraction completes ═══
(function() {
    const _origHandler = window._handleLensExtracted;
    window._handleLensExtracted = function(event) {
        // Call the original youtube.js handler first
        if (typeof _origHandler === 'function') _origHandler(event);

        // Invalidate cache for this video and re-render in the KB tab
        const videoDbId = event.video_id;
        if (videoDbId && _ykbInsightsLoaded[videoDbId]) {
            delete _ykbInsightsLoaded[videoDbId];
            // Find which channel has this video active and reload
            for (const chId of Object.keys(_ykbActiveVideo)) {
                const av = _ykbActiveVideo[chId];
                if (av && av.id == videoDbId) {
                    _ykbLoadInsights(parseInt(chId), videoDbId);
                    break;
                }
            }
        }
    };
})();

// ═══ Stars Parallax ═══
let _ykbStarsAnim = null;
function _ykbInitStars(chId) {
    if (_ykbStarsAnim) { cancelAnimationFrame(_ykbStarsAnim); _ykbStarsAnim = null; }
    const canvas = document.getElementById(`ykb-stars-${chId}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    // Read accent color from CSS variables for theme awareness
    const _accentStr = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#10b981';
    const _hm = _accentStr.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    const sr = _hm ? parseInt(_hm[1],16) : 16, sg = _hm ? parseInt(_hm[2],16) : 185, sb = _hm ? parseInt(_hm[3],16) : 129;

    let W, H, stars = [];

    function resize() {
        const el = canvas.parentElement;
        if (!el) return;
        W = canvas.width = el.offsetWidth;
        H = canvas.height = el.offsetHeight;
    }
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(canvas.parentElement);

    // More stars with higher brightness for visibility
    for (let i = 0; i < 140; i++) {
        const depth = Math.random() * 4 + 0.1;
        const dn = depth / 4.1;
        stars.push({
            x: Math.random(), y: Math.random(),
            r: 0.4 + dn * 2.2,
            a: 0.15 + dn * 0.6,
            dx: (Math.random() - 0.5) * (0.04 + dn * 0.15),
            dy: (Math.random() - 0.5) * (0.02 + dn * 0.1),
            phase: Math.random() * Math.PI * 2,
        });
    }

    let shootingStar = null, lastShoot = 0;

    function draw(t) {
        if (!document.getElementById(`ykb-stars-${chId}`)) { obs.disconnect(); return; }
        ctx.clearRect(0, 0, W, H);

        for (const s of stars) {
            s.x += s.dx / W; s.y += s.dy / H;
            if (s.x < 0) s.x = 1; if (s.x > 1) s.x = 0;
            if (s.y < 0) s.y = 1; if (s.y > 1) s.y = 0;
            const flicker = 0.7 + 0.3 * Math.sin(t * 0.002 + s.phase);
            ctx.beginPath();
            ctx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${sr},${sg},${sb},${s.a * flicker})`;
            ctx.fill();
        }

        // Constellation lines — more visible
        ctx.strokeStyle = `rgba(${sr},${sg},${sb},0.07)`;
        ctx.lineWidth = 0.7;
        for (let i = 0; i < stars.length; i++) {
            for (let j = i + 1; j < stars.length; j++) {
                const dx = (stars[i].x - stars[j].x) * W;
                const dy = (stars[i].y - stars[j].y) * H;
                const dist = dx * dx + dy * dy;
                if (dist < 8000) {
                    ctx.beginPath();
                    ctx.moveTo(stars[i].x * W, stars[i].y * H);
                    ctx.lineTo(stars[j].x * W, stars[j].y * H);
                    ctx.stroke();
                }
            }
        }

        // Shooting star
        if (!shootingStar && t - lastShoot > 4000 + Math.random() * 6000) {
            lastShoot = t;
            shootingStar = { x: Math.random() * W * 0.6, y: Math.random() * H * 0.3, vx: 3 + Math.random() * 4, vy: 1.5 + Math.random() * 2, life: 1 };
        }
        if (shootingStar) {
            shootingStar.x += shootingStar.vx;
            shootingStar.y += shootingStar.vy;
            shootingStar.life -= 0.015;
            if (shootingStar.life <= 0) { shootingStar = null; }
            else {
                const grad = ctx.createLinearGradient(shootingStar.x, shootingStar.y, shootingStar.x - shootingStar.vx * 8, shootingStar.y - shootingStar.vy * 8);
                grad.addColorStop(0, `rgba(${sr},${sg},${sb},${shootingStar.life * 0.6})`);
                grad.addColorStop(1, `rgba(${sr},${sg},${sb},0)`);
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(shootingStar.x, shootingStar.y);
                ctx.lineTo(shootingStar.x - shootingStar.vx * 8, shootingStar.y - shootingStar.vy * 8);
                ctx.stroke();
            }
        }

        _ykbStarsAnim = requestAnimationFrame(draw);
    }
    _ykbStarsAnim = requestAnimationFrame(draw);
}
