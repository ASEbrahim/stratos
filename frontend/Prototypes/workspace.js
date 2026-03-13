// ═══════════════════════════════════════════════════════════
// WORKSPACE — Profile export/import and preference signals
// ═══════════════════════════════════════════════════════════

function _wsHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

// ── Initialize workspace panel ──
function initWorkspacePanel() {
    _wsLoadStats();
    _wsLoadSignals();
}
window.initWorkspacePanel = initWorkspacePanel;

// ── Load workspace stats ──
async function _wsLoadStats() {
    const el = document.getElementById('ws-stats-content');
    if (!el) return;
    try {
        const r = await fetch('/api/profile/workspace-stats', { headers: _wsHeaders() });
        if (r.ok) {
            const d = await r.json();
            el.innerHTML = `
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    ${_wsStat('News Items', d.news_items || 0, 'newspaper')}
                    ${_wsStat('Feedback', d.feedback_entries || 0, 'thumbs-up')}
                    ${_wsStat('Files', d.uploaded_files || 0, 'file')}
                    ${_wsStat('Contexts', d.persona_contexts || 0, 'file-text')}
                    ${_wsStat('Channels', d.youtube_channels || 0, 'youtube')}
                    ${_wsStat('Videos', d.youtube_videos || 0, 'play')}
                    ${_wsStat('Insights', d.video_insights || 0, 'sparkles')}
                    ${_wsStat('Signals', d.preference_signals || 0, 'zap')}
                </div>
                <div class="text-[10px] mt-2" style="color:var(--text-muted)">Disk usage: ${d.disk_human || '0 B'}</div>`;
            lucide.createIcons();
        }
    } catch (e) {
        el.innerHTML = '<div class="text-[10px] text-red-400">Failed to load stats</div>';
    }
}

function _wsStat(label, count, icon) {
    return `<div class="px-2 py-1.5 rounded-lg text-center" style="background:var(--bg-hover);border:1px solid var(--border);">
        <i data-lucide="${icon}" class="w-3 h-3 mx-auto mb-0.5" style="color:var(--accent)"></i>
        <div class="text-[12px] font-bold" style="color:var(--text-heading)">${count}</div>
        <div class="text-[8px]" style="color:var(--text-muted)">${label}</div>
    </div>`;
}

// ── Export profile ──
async function _wsExport() {
    try {
        const r = await fetch('/api/profile/export', {
            method: 'POST',
            headers: _wsHeaders(),
            body: JSON.stringify({ include_files: true, include_insights: true })
        });
        if (r.ok) {
            const blob = await r.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `stratos-profile-${new Date().toISOString().slice(0,10)}.zip`;
            a.click();
            URL.revokeObjectURL(url);
            if (typeof showToast === 'function') showToast('Profile exported', 'success');
        } else {
            if (typeof showToast === 'function') showToast('Export failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Export failed', 'error');
    }
}
window._wsExport = _wsExport;

// ── Import profile ──
async function _wsImport(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const strategy = document.getElementById('ws-import-strategy')?.value || 'merge';
    try {
        const r = await fetch(`/api/profile/import?strategy=${strategy}`, {
            method: 'POST',
            headers: {
                'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
            },
            body: file
        });
        if (r.ok) {
            const d = await r.json();
            const stats = d.stats || {};
            if (typeof showToast === 'function') showToast(`Imported: ${stats.contexts || 0} contexts, ${stats.channels || 0} channels, ${stats.signals || 0} signals`, 'success');
            _wsLoadStats();
        } else {
            if (typeof showToast === 'function') showToast('Import failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Import failed', 'error');
    }
    event.target.value = '';
}
window._wsImport = _wsImport;

// ── Load preference signals ──
async function _wsLoadSignals() {
    const el = document.getElementById('ws-signals-list');
    if (!el) return;
    try {
        const r = await fetch('/api/preference-signals', { headers: _wsHeaders() });
        if (r.ok) {
            const d = await r.json();
            const signals = d.signals || [];
            if (signals.length === 0) {
                el.innerHTML = '<div class="text-[10px]" style="color:var(--text-muted)">No preference signals yet. These are learned automatically as you use personas.</div>';
                return;
            }
            el.innerHTML = signals.map(s => `
                <div class="flex items-center justify-between px-2 py-1.5 rounded-md" style="background:var(--bg-hover);border:1px solid var(--border);">
                    <div class="flex-1 min-w-0">
                        <div class="text-[10px] font-semibold" style="color:var(--text-heading)">${_escHtml(s.signal_key)}</div>
                        <div class="text-[9px]" style="color:var(--text-muted)">${_escHtml(s.persona_source)} · ${_escHtml(s.signal_type)} · weight: ${s.signal_weight}${s.auto_generated ? ' · auto' : ''}</div>
                    </div>
                    <button onclick="_wsDeleteSignal(${s.id})" class="p-1 rounded text-slate-600 hover:text-red-400 transition-colors" title="Remove signal">
                        <i data-lucide="x" class="w-3 h-3"></i>
                    </button>
                </div>
            `).join('');
            lucide.createIcons();
        }
    } catch (e) {
        el.innerHTML = '<div class="text-[10px] text-red-400">Failed to load signals</div>';
    }
}

async function _wsDeleteSignal(id) {
    try {
        await fetch(`/api/preference-signals/${id}`, { method: 'DELETE', headers: _wsHeaders() });
        _wsLoadSignals();
    } catch (e) {}
}
window._wsDeleteSignal = _wsDeleteSignal;

// ── Add manual signal ──
async function _wsAddSignal() {
    const persona = document.getElementById('ws-signal-persona')?.value || 'intelligence';
    const type = document.getElementById('ws-signal-type')?.value || 'topic_interest';
    const key = document.getElementById('ws-signal-key')?.value?.trim();
    if (!key) { if (typeof showToast === 'function') showToast('Enter a signal key', 'warning'); return; }

    try {
        const r = await fetch('/api/preference-signals', {
            method: 'POST',
            headers: _wsHeaders(),
            body: JSON.stringify({ persona_source: persona, signal_type: type, signal_key: key, signal_weight: 1.0 })
        });
        if (r.ok) {
            document.getElementById('ws-signal-key').value = '';
            _wsLoadSignals();
            if (typeof showToast === 'function') showToast('Signal added', 'success');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to add signal', 'error');
    }
}
window._wsAddSignal = _wsAddSignal;
