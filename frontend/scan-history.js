/**
 * STRAT_OS — Scan History Panel
 * Floating panel showing recent scan results, accessible via the history button.
 */

let _scanHistoryOpen = false;

function toggleScanHistory() {
    const existing = document.getElementById('scan-history-panel');
    if (existing) { closeScanHistory(); return; }
    openScanHistory();
}

async function openScanHistory() {
    if (document.getElementById('scan-history-panel')) return;
    _scanHistoryOpen = true;

    // Fetch log
    let scans = [];
    try {
        const r = await fetch('/api/scan-log');
        if (r.ok) scans = await r.json();
    } catch (e) { console.error('Failed to fetch scan log:', e); }

    const panel = document.createElement('div');
    panel.id = 'scan-history-panel';

    const rows = scans.length === 0
        ? '<div class="sh-empty">No scans recorded yet. Run your first scan!</div>'
        : scans.slice(0, 30).map(s => {
            const d = new Date(s.started_at || s.timestamp);
            const date = d.toLocaleDateString('en-GB', { day:'numeric', month:'short' });
            const time = d.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit' });
            const elapsed = s.elapsed_secs ? `${s.elapsed_secs}s` : '—';
            const total = s.items_scored || s.items_fetched || 0;

            if (s.error) {
                return `<div class="sh-row sh-row-err">
                    <div class="sh-date">
                        <span class="sh-date-d">${date}</span>
                        <span class="sh-date-t">${time}</span>
                    </div>
                    <div class="sh-stats">
                        <span class="sh-total" style="color:#f87171;">Failed</span>
                        <span class="sh-elapsed">${elapsed}</span>
                    </div>
                    <div class="sh-breakdown"><span class="sh-tag" style="background:rgba(248,113,113,.12);color:#f87171;">${s.error.slice(0,50)}</span></div>
                    <div class="sh-engine"></div>
                </div>`;
            }

            const cr = s.critical || 0;
            const hi = s.high || 0;
            const med = s.medium || 0;
            const no = s.noise || 0;
            const ruleN = s.rule_scored || 0;
            const llmN = s.llm_scored || 0;
            const ret = s.retained || 0;

            return `<div class="sh-row">
                <div class="sh-date">
                    <span class="sh-date-d">${date}</span>
                    <span class="sh-date-t">${time}</span>
                </div>
                <div class="sh-stats">
                    <span class="sh-total">${total} items</span>
                    <span class="sh-elapsed">${elapsed}</span>
                </div>
                <div class="sh-breakdown">
                    ${cr ? `<span class="sh-tag sh-critical">${cr} critical</span>` : ''}
                    ${hi ? `<span class="sh-tag sh-high">${hi} high</span>` : ''}
                    ${med ? `<span class="sh-tag sh-medium">${med} mid</span>` : ''}
                    <span class="sh-tag sh-noise">${no} noise</span>
                    ${ret ? `<span class="sh-tag sh-retained">${ret} kept</span>` : ''}
                </div>
                <div class="sh-engine">
                    <span title="Rule-based scoring">R:${ruleN}</span>
                    <span title="LLM scoring">L:${llmN}</span>
                </div>
            </div>`;
        }).join('');

    panel.innerHTML = `
        <div class="sh-backdrop" onclick="closeScanHistory()"></div>
        <div class="sh-card">
            <div class="sh-header">
                <div class="sh-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    Scan History
                </div>
                <button class="sh-close" onclick="closeScanHistory()">&times;</button>
            </div>
            <div class="sh-body">${rows}</div>
            <div class="sh-footer">
                <div class="sh-footer-row">
                    <span>${scans.length} scan${scans.length !== 1 ? 's' : ''} recorded</span>
                    <div class="sh-export-btns">
                        <button class="sh-export-btn" onclick="exportDashboard('csv')" title="Export signals as CSV (for spreadsheets)">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                            CSV
                        </button>
                        <button class="sh-export-btn" onclick="exportDashboard('json')" title="Export full diagnostic JSON (for sharing with AI)">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                            JSON
                        </button>
                    </div>
                </div>
            </div>
        </div>`;

    document.body.appendChild(panel);
    requestAnimationFrame(() => panel.classList.add('sh-open'));
}

function closeScanHistory() {
    const panel = document.getElementById('scan-history-panel');
    if (!panel) return;
    _scanHistoryOpen = false;
    panel.classList.remove('sh-open');
    setTimeout(() => panel.remove(), 200);
}

async function exportDashboard(format) {
    const btn = event?.target?.closest('.sh-export-btn');
    if (btn) { btn.disabled = true; btn.style.opacity = '.5'; }
    try {
        const r = await fetch(`/api/export?format=${format}`);
        if (!r.ok) throw new Error(`Export failed: ${r.status}`);
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const ts = new Date().toISOString().slice(0,16).replace(/[T:]/g, '-');
        a.href = url;
        a.download = `stratos_export_${ts}.${format}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (e) {
        console.error('Export failed:', e);
        alert('Export failed: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.style.opacity = ''; }
    }
}

/* ═══ STYLES ═══ */
const _shCss = document.createElement('style');
_shCss.textContent = `
#scan-history-panel { position:fixed; inset:0; z-index:8000; pointer-events:none; opacity:0; transition:opacity .2s ease; }
#scan-history-panel.sh-open { opacity:1; pointer-events:auto; }
.sh-backdrop { position:absolute; inset:0; background:rgba(0,0,0,.4); }
.sh-card {
    position:absolute; top:60px; right:20px; width:420px; max-height:70vh;
    background:var(--bg-panel-solid, #0f172a); border:1px solid var(--border-strong, rgba(100,116,139,.2));
    border-radius:14px; display:flex; flex-direction:column;
    box-shadow:0 20px 60px rgba(0,0,0,.5); overflow:hidden;
}
.sh-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:14px 18px; border-bottom:1px solid var(--border-strong, rgba(100,116,139,.15));
}
.sh-title { display:flex; align-items:center; gap:8px; font-size:14px; font-weight:600; color:rgba(226,232,240,.9); }
.sh-title svg { opacity:.6; }
.sh-close { background:none; border:none; color:rgba(100,116,139,.6); font-size:20px; cursor:pointer; padding:2px 6px; border-radius:4px; transition:color .15s; }
.sh-close:hover { color:rgba(226,232,240,.8); }
.sh-body { overflow-y:auto; padding:8px 12px; flex:1; }
.sh-footer { padding:10px 18px; border-top:1px solid var(--border-strong, rgba(100,116,139,.1)); font-size:10px; color:rgba(100,116,139,.5); }
.sh-footer-row { display:flex; align-items:center; justify-content:space-between; }
.sh-export-btns { display:flex; gap:6px; }
.sh-export-btn {
    display:flex; align-items:center; gap:4px; padding:4px 10px; border-radius:6px;
    background:rgba(100,116,139,.1); border:1px solid rgba(100,116,139,.15);
    color:rgba(148,163,184,.7); font-size:10px; font-weight:600; cursor:pointer;
    transition:background .15s, color .15s, border-color .15s; font-family:inherit;
}
.sh-export-btn:hover { background:rgba(100,116,139,.2); color:rgba(226,232,240,.9); border-color:rgba(100,116,139,.3); }

.sh-empty { text-align:center; padding:32px 16px; color:rgba(100,116,139,.5); font-size:13px; }

.sh-row {
    display:grid; grid-template-columns:72px 1fr auto auto; gap:8px; align-items:center;
    padding:10px 8px; border-radius:8px; transition:background .15s;
}
.sh-row:hover { background:rgba(255,255,255,.03); }
.sh-row + .sh-row { border-top:1px solid rgba(100,116,139,.08); }

.sh-date { display:flex; flex-direction:column; }
.sh-date-d { font-size:11px; font-weight:600; color:rgba(226,232,240,.8); }
.sh-date-t { font-size:10px; color:rgba(100,116,139,.5); font-family:monospace; }

.sh-stats { display:flex; flex-direction:column; }
.sh-total { font-size:12px; font-weight:500; color:rgba(226,232,240,.75); }
.sh-elapsed { font-size:10px; color:rgba(100,116,139,.45); font-family:monospace; }

.sh-breakdown { display:flex; flex-wrap:wrap; gap:4px; }
.sh-tag { font-size:9px; padding:2px 6px; border-radius:4px; font-weight:600; white-space:nowrap; }
.sh-critical { background:rgba(52,211,153,.12); color:#34d399; }
.sh-high { background:rgba(59,130,246,.12); color:#60a5fa; }
.sh-medium { background:rgba(251,191,36,.10); color:#fbbf24; }
.sh-noise { background:rgba(100,116,139,.10); color:rgba(148,163,184,.6); }
.sh-retained { background:rgba(245,158,11,.10); color:#f59e0b; }

.sh-engine { display:flex; gap:6px; font-size:9px; font-family:monospace; color:rgba(100,116,139,.45); }

@media (max-width:500px) {
    .sh-card { right:8px; left:8px; width:auto; top:50px; max-height:80vh; }
    .sh-row { grid-template-columns:64px 1fr; }
    .sh-breakdown, .sh-engine { grid-column:1/-1; }
}
`;
document.head.appendChild(_shCss);
