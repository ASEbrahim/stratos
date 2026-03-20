// ═══════════════════════════════════════════════════════════
// GAMES UI — Scenario selector for Games/Roleplay persona
// ═══════════════════════════════════════════════════════════

let _gamesScenarios = [];
let _gamesActiveScenario = null;
let _gamesRpMode = 'gm';  // 'gm' or 'immersive'
let _gamesActiveNpc = '';   // Currently talking to NPC name

// Restore RP mode from localStorage
try { _gamesRpMode = localStorage.getItem('stratos_games_rp_mode') || 'gm'; } catch(e) {}

function _refreshFileBrowserIfOpen() {
    if (typeof window._fbLoadDir === 'function' && window._fbOpen) {
        window._fbLoadDir(window._fbPath || '/');
    }
}

function _gamesHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

// ── Show/hide scenario bar based on active persona ──
function updateScenarioBar() {
    const bar = document.getElementById('games-scenario-bar');
    if (!bar) return;
    if (currentPersona === 'gaming' || currentPersona === 'games') {
        bar.classList.remove('hidden');
        _loadScenarios();
    } else {
        bar.classList.add('hidden');
    }
}
window.updateScenarioBar = updateScenarioBar;

// ── Load scenarios ──
async function _loadScenarios() {
    try {
        const r = await fetch('/api/scenarios', { headers: _gamesHeaders() });
        if (r.ok) {
            const d = await r.json();
            _gamesScenarios = d.scenarios || [];
        }

        const r2 = await fetch('/api/scenarios/active', { headers: _gamesHeaders() });
        if (r2.ok) {
            const d2 = await r2.json();
            _gamesActiveScenario = d2.active || null;
        }

        _renderScenarioBar();
    } catch (e) {}
}

function _renderScenarioBar() {
    const bar = document.getElementById('games-scenario-content');
    if (!bar) return;
    const theme = (typeof PERSONA_THEMES !== 'undefined') ? (PERSONA_THEMES.gaming || { color: '#f472b6', bg: 'rgba(244,114,182,0.1)' }) : { color: '#f472b6', bg: 'rgba(244,114,182,0.1)' };

    if (_gamesScenarios.length === 0) {
        bar.innerHTML = `
            <div class="flex flex-col items-center py-3 gap-2">
                <div class="w-10 h-10 rounded-xl flex items-center justify-center" style="background:${theme.bg};border:1px solid ${theme.color}30;">
                    <i data-lucide="swords" class="w-5 h-5" style="color:${theme.color}"></i>
                </div>
                <div class="text-[11px] font-semibold" style="color:var(--text-heading)">Create Your First World</div>
                <div class="text-[9px]" style="color:var(--text-muted)">Start an RPG, adventure, or roleplay scenario</div>
                <div class="flex flex-wrap gap-1 justify-center mt-1">
                    ${['Fantasy RPG', 'Sci-Fi', 'Mystery', 'Horror', 'Slice of Life'].map(genre =>
                        `<button onclick="_gamesCreateScenarioWithGenre('${genre}')" class="text-[9px] px-2 py-1 rounded-md transition-all cursor-pointer" style="border:1px solid var(--border-strong);color:var(--text-muted);background:transparent;" onmouseenter="this.style.borderColor='${theme.color}';this.style.color='${theme.color}'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.color='var(--text-muted)'">${genre}</button>`
                    ).join('')}
                </div>
                <button onclick="_gamesCreateScenario()" class="text-[10px] px-3 py-1.5 rounded-lg font-medium transition-all mt-1" style="background:${theme.bg};color:${theme.color};border:1px solid ${theme.color}40;" onmouseenter="this.style.background='${theme.color}20'" onmouseleave="this.style.background='${theme.bg}'">
                    <i data-lucide="plus" class="w-3 h-3 inline"></i> Custom Scenario
                </button>
            </div>`;
        lucide.createIcons();
        return;
    }

    // Scenario cards — horizontal scroll
    const cards = _gamesScenarios.map(s => {
        const active = s.name === _gamesActiveScenario;
        const world = s.world_description || s.world || '';
        const preview = world.length > 60 ? world.slice(0, 58) + '…' : (world || 'No description');
        return `<button onclick="_gamesActivateScenario('${_escForAttr(s.name)}')" class="flex-shrink-0 px-3 py-2 rounded-lg text-left transition-all" style="min-width:140px;max-width:200px;background:${active ? theme.bg : 'rgba(255,255,255,0.02)'};border:1px solid ${active ? theme.color + '60' : 'var(--border-strong)'};" onmouseenter="if(!${active})this.style.borderColor='${theme.color}40'" onmouseleave="if(!${active})this.style.borderColor='${active ? theme.color + '60' : 'var(--border-strong)'}'" >
            <div class="flex items-center justify-between gap-1">
                <span class="text-[11px] font-semibold truncate" style="color:${active ? theme.color : 'var(--text-heading)'}">${_escHtmlG(s.name)}</span>
                ${active ? `<span class="w-2 h-2 rounded-full flex-shrink-0" style="background:${theme.color};box-shadow:0 0 6px ${theme.color}60;"></span>` : ''}
            </div>
            <div class="text-[9px] mt-0.5 line-clamp-2" style="color:var(--text-muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">${_escHtmlG(preview)}</div>
        </button>`;
    }).join('');

    bar.innerHTML = `
        <div class="flex items-center gap-2 overflow-x-auto pb-1" style="scrollbar-width:none;">
            ${cards}
            <button onclick="_gamesCreateScenario()" class="flex-shrink-0 w-9 h-full min-h-[52px] rounded-lg flex items-center justify-center transition-all" style="border:1px dashed var(--border-strong);color:var(--text-muted);" title="New scenario" onmouseenter="this.style.borderColor='${theme.color}';this.style.color='${theme.color}'" onmouseleave="this.style.borderColor='var(--border-strong)';this.style.color='var(--text-muted)'">
                <i data-lucide="plus" class="w-4 h-4"></i>
            </button>
        </div>
        ${_gamesActiveScenario ? `<div class="flex items-center gap-1.5 mt-1.5 flex-wrap">
            <div class="games-mode-toggle flex rounded-md overflow-hidden" style="border:1px solid var(--border-strong);">
                <button onclick="_gamesSetMode('gm')" class="text-[9px] px-2.5 py-1 flex items-center gap-1 transition-all" style="background:${_gamesRpMode === 'gm' ? theme.bg : 'transparent'};color:${_gamesRpMode === 'gm' ? theme.color : 'var(--text-muted)'};border-right:1px solid var(--border-strong);" title="Game Master mode — third-person narration, stat boxes, choices"><i data-lucide="gamepad-2" class="w-3 h-3"></i> GM</button>
                <button onclick="_gamesSetMode('immersive')" class="text-[9px] px-2.5 py-1 flex items-center gap-1 transition-all" style="background:${_gamesRpMode === 'immersive' ? 'rgba(168,85,247,0.1)' : 'transparent'};color:${_gamesRpMode === 'immersive' ? '#c084fc' : 'var(--text-muted)'};" title="Immersive RP mode — AI responds AS characters"><i data-lucide="message-circle" class="w-3 h-3"></i> RP</button>
            </div>
            ${_gamesRpMode === 'immersive' ? `<div class="flex items-center gap-1">
                <span class="text-[9px]" style="color:var(--text-muted)">Talking to:</span>
                <select id="games-npc-input" class="text-[9px] px-2 py-1 rounded-md" style="background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);color:#c084fc;min-width:120px;outline:none;cursor:pointer;" onchange="_gamesSetNpc(this.value)">
                    <option value="" style="background:var(--bg-panel-solid);color:var(--text-muted)">— select —</option>
                    ${_gamesEntities.map(e => `<option value="${_escHtmlG(e.display_name)}" style="background:var(--bg-panel-solid);color:var(--text-primary)"${_gamesActiveNpc === e.display_name ? ' selected' : ''}>${_escHtmlG(e.display_name)}</option>`).join('')}
                </select>
            </div>` : ''}
            <button onclick="toggleFileBrowser('gaming')" class="text-[9px] px-2 py-1 rounded-md flex items-center gap-1 transition-all" style="color:var(--text-muted);border:1px solid var(--border-strong);" onmouseenter="this.style.color='${theme.color}';this.style.borderColor='${theme.color}40'" onmouseleave="this.style.color='var(--text-muted)';this.style.borderColor='var(--border-strong)'"><i data-lucide="folder-open" class="w-3 h-3"></i> Files</button>
            <button onclick="_gamesDeleteScenario('${_escForAttr(_gamesActiveScenario)}')" class="text-[9px] px-2 py-1 rounded-md flex items-center gap-1 transition-all" style="color:var(--text-muted);border:1px solid var(--border-strong);" onmouseenter="this.style.color='#f87171';this.style.borderColor='rgba(239,68,68,0.3)'" onmouseleave="this.style.color='var(--text-muted)';this.style.borderColor='var(--border-strong)'"><i data-lucide="trash-2" class="w-3 h-3"></i> Delete</button>
        </div>
        <div id="games-entity-content"></div>` : ''}`;
    lucide.createIcons();
    // Only load character list in immersive/RP mode, not GM
    if (_gamesActiveScenario && _gamesRpMode === 'immersive') _gamesLoadEntities();
    else { _gamesEntities = []; const ec = document.getElementById('games-entity-content'); if (ec) ec.innerHTML = ''; }
}

function _escHtmlG(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function _escForAttr(s) { return s.replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

function _gamesSetMode(mode) {
    _gamesRpMode = mode;
    try { localStorage.setItem('stratos_games_rp_mode', mode); } catch(e) {}
    _renderScenarioBar();
    if (typeof showToast === 'function') showToast(mode === 'immersive' ? 'Immersive RP mode — AI speaks as characters' : 'Game Master mode — narration & choices', 'info');
}
window._gamesSetMode = _gamesSetMode;

function _gamesSetNpc(name) {
    _gamesActiveNpc = name.trim();
    _renderEntityBar();
}
window._gamesSetNpc = _gamesSetNpc;

// Auto-detect character name in message text and switch active NPC
function _gamesAutoDetectNpc(msg) {
    if (_gamesRpMode !== 'immersive' || _gamesEntities.length === 0) return;
    const lower = msg.toLowerCase();
    // Check each entity — match display_name (case-insensitive)
    // Prefer longest match first to avoid partial matches (e.g., "Gojo Satoru" before "Gojo")
    const sorted = [..._gamesEntities].sort((a, b) => b.display_name.length - a.display_name.length);
    for (const e of sorted) {
        if (lower.includes(e.display_name.toLowerCase())) {
            if (_gamesActiveNpc !== e.display_name) {
                _gamesActiveNpc = e.display_name;
                const sel = document.getElementById('games-npc-input');
                if (sel) sel.value = e.display_name;
                _renderEntityBar();
            }
            return;
        }
    }
}
window._gamesAutoDetectNpc = _gamesAutoDetectNpc;

// Expose state for agent.js to read when sending messages
window._gamesGetState = function() {
    return { rpMode: _gamesRpMode, activeNpc: _gamesActiveNpc, activeScenario: _gamesActiveScenario };
};

// ── Create scenario (enhanced with genre + description) ──
async function _gamesCreateScenario() {
    const result = await stratosPrompt({ title: 'Create New Scenario', fields: [
        { key: 'name', label: 'Scenario name', placeholder: 'e.g., SAO_Aincrad' },
        { key: 'genre', label: 'Genre', placeholder: 'Fantasy RPG', optional: true },
        { key: 'description', label: 'Describe your world', placeholder: 'A dark fantasy setting where...\n\nThe more detail you provide, the richer the generated world.', optional: true, multiline: true }
    ]});
    if (!result) return;
    const name = result.name;
    const genre = result.genre || 'Fantasy RPG';
    const description = result.description || '';
    await _gamesDoCreate(name.trim(), genre, description);
}
window._gamesCreateScenario = _gamesCreateScenario;

// ── Create with genre preset ──
async function _gamesCreateScenarioWithGenre(genre) {
    const result = await stratosPrompt({ title: `New ${genre} Scenario`, fields: [
        { key: 'name', label: 'Scenario name', placeholder: genre.replace(/\s/g, '_'), defaultValue: genre.replace(/\s/g, '_') },
        { key: 'description', label: 'Describe your world (optional)', placeholder: `A ${genre.toLowerCase()} setting...`, optional: true, multiline: true }
    ]});
    if (!result) return;
    const name = result.name;
    const description = result.description || `A ${genre.toLowerCase()} setting.`;
    await _gamesDoCreate(name.trim(), genre, description);
}
window._gamesCreateScenarioWithGenre = _gamesCreateScenarioWithGenre;

// ── Shared create + generation progress ──
async function _gamesDoCreate(name, genre, description) {
    if (!name) return;
    try {
        const r = await fetch('/api/scenarios/create', {
            method: 'POST',
            headers: _gamesHeaders(),
            body: JSON.stringify({ name, genre, description })
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
            if (typeof showToast === 'function') showToast(d.error || 'Failed to create scenario', 'error');
            return;
        }
        const safeName = d.name || name;
        if (typeof showToast === 'function') showToast(`Created "${safeName}"`, 'success');
        _loadScenarios();
        _refreshFileBrowserIfOpen();

        // Poll generation status if LLM generation was triggered
        if (d.status === 'generating') {
            _pollGenerationStatus(safeName);
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create scenario', 'error');
    }
}

// ── Poll generation progress and show in scenario bar ──
let _genPollTimer = null;
function _pollGenerationStatus(scenarioName) {
    // Show progress indicator in the scenario bar
    const bar = document.getElementById('games-scenario-content');
    if (bar) {
        const theme = (typeof PERSONA_THEMES !== 'undefined') ? (PERSONA_THEMES.gaming || { color: '#f472b6' }) : { color: '#f472b6' };
        let progressEl = document.getElementById('games-gen-progress');
        if (!progressEl) {
            progressEl = document.createElement('div');
            progressEl.id = 'games-gen-progress';
            progressEl.className = 'mt-1.5 text-[10px] flex flex-col gap-0.5';
            progressEl.style.color = theme.color;
            bar.appendChild(progressEl);
        }
        progressEl.innerHTML = '<div class="flex items-center gap-1"><span class="animate-pulse">Generating scenario...</span></div>';
        progressEl._scenarioName = scenarioName;
    }

    let pollCount = 0;
    const maxPolls = 150; // 5 minutes max (canon imports take longer)
    if (_genPollTimer) clearInterval(_genPollTimer);
    _genPollTimer = setInterval(async () => {
        pollCount++;
        if (pollCount > maxPolls) {
            clearInterval(_genPollTimer);
            _genPollTimer = null;
            const el = document.getElementById('games-gen-progress');
            if (el) el.remove();
            return;
        }
        try {
            const r = await fetch('/api/scenarios/generate-status', {
                method: 'POST',
                headers: _gamesHeaders(),
                body: JSON.stringify({ name: scenarioName })
            });
            if (!r.ok) return;
            const status = await r.json();
            const el = document.getElementById('games-gen-progress');
            if (!el) { clearInterval(_genPollTimer); _genPollTimer = null; return; }

            if (status.passes) {
                const isCanon = status.source === 'canon_import';
                const header = isCanon
                    ? `<div class="flex items-center gap-1 mb-1 font-semibold"><span>&#127759;</span> Importing ${status.franchise || 'world'} from wiki...</div>`
                    : '';
                const lines = Object.entries(status.passes).map(([num, p]) => {
                    const icon = p.status === 'done' ? '&#10003;' : p.status === 'failed' ? '&#10007;' : '<span class="animate-pulse">&#9679;</span>';
                    return `<div class="flex items-center gap-1"><span>${icon}</span> ${p.name}</div>`;
                }).join('');
                el.innerHTML = header + lines;
            }

            if (status.status === 'complete' || status.status === 'failed') {
                clearInterval(_genPollTimer);
                _genPollTimer = null;
                setTimeout(() => {
                    const el2 = document.getElementById('games-gen-progress');
                    if (el2) el2.remove();
                    _loadScenarios();
                    _refreshFileBrowserIfOpen();
                }, 2000);
            }
        } catch (e) {}
    }, 2000);
}
window._pollGenerationStatus = _pollGenerationStatus;

// ── Activate scenario ──
async function _gamesActivateScenario(name) {
    if (!name) return;
    try {
        await fetch('/api/scenarios/activate', {
            method: 'POST',
            headers: _gamesHeaders(),
            body: JSON.stringify({ name })
        });
        _gamesActiveScenario = name;
        _renderScenarioBar();
        _refreshFileBrowserIfOpen();
        if (typeof showToast === 'function') showToast(`Switched to "${name}"`, 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to activate scenario', 'error');
    }
}
window._gamesActivateScenario = _gamesActivateScenario;

// ── Delete scenario ──
async function _gamesDeleteScenario(name) {
    if (!(await stratosConfirm(`Delete scenario "${name}"? This cannot be undone.`, { title: 'Delete Scenario', okText: 'Delete', cancelText: 'Cancel' }))) return;
    try {
        await fetch(`/api/scenarios?name=${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: _gamesHeaders()
        });
        if (_gamesActiveScenario === name) _gamesActiveScenario = null;
        _loadScenarios();
        _refreshFileBrowserIfOpen();
        if (typeof showToast === 'function') showToast(`Deleted "${name}"`, 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to delete scenario', 'error');
    }
}
window._gamesDeleteScenario = _gamesDeleteScenario;

// ═══════════════════════════════════════════════════════════
// ENTITY PERSISTENCE (Characters for gaming, Figures for scholarly)
// ═══════════════════════════════════════════════════════════

let _gamesEntities = [];

async function _gamesLoadEntities() {
    if (!_gamesActiveScenario) { _gamesEntities = []; return; }
    try {
        const persona = (typeof currentPersona !== 'undefined') ? currentPersona : 'gaming';
        const r = await fetch(`/api/personas/${persona}/entities?scenario=${encodeURIComponent(_gamesActiveScenario)}`, {
            headers: _gamesHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            _gamesEntities = d.entities || [];
        }
    } catch (e) { _gamesEntities = []; }
    _renderEntityBar();
}

function _renderEntityBar() {
    const bar = document.getElementById('games-entity-content');
    if (!bar) return;
    if (!_gamesActiveScenario) { bar.innerHTML = ''; return; }
    const theme = (typeof PERSONA_THEMES !== 'undefined') ? (PERSONA_THEMES.gaming || { color: '#f472b6', bg: 'rgba(244,114,182,0.1)' }) : { color: '#f472b6', bg: 'rgba(244,114,182,0.1)' };

    const entityLabel = (typeof currentPersona !== 'undefined' && currentPersona === 'scholarly') ? 'Figure' : 'Character';

    if (_gamesEntities.length === 0) {
        bar.innerHTML = `<div class="flex items-center gap-2 py-1">
            <span class="text-[9px]" style="color:var(--text-muted)">No ${entityLabel.toLowerCase()}s yet</span>
            <button onclick="_gamesCreateEntity()" class="text-[9px] px-2 py-0.5 rounded-md transition-all" style="color:${theme.color};border:1px solid ${theme.color}30;" onmouseenter="this.style.background='${theme.bg}'" onmouseleave="this.style.background='transparent'">+ Add ${entityLabel}</button>
        </div>`;
        return;
    }

    // Sort: last spoken to first (updated_at DESC), then alphabetical
    const sorted = [..._gamesEntities].sort((a, b) => {
        const aTime = a.updated_at || a.created_at || '';
        const bTime = b.updated_at || b.created_at || '';
        if (bTime > aTime) return 1;
        if (aTime > bTime) return -1;
        return (a.display_name || '').localeCompare(b.display_name || '');
    });

    const chips = sorted.map(e => {
        const active = _gamesActiveNpc === e.display_name;
        const preview = (e.personality_md || '').slice(0, 40) || e.entity_type;
        return `<div class="flex-shrink-0 flex items-center gap-0.5 rounded-md transition-all" style="background:${active ? 'rgba(168,85,247,0.15)' : 'transparent'};border:1px solid ${active ? 'rgba(168,85,247,0.4)' : 'var(--border-strong)'};">
            <button onclick="_gamesSelectEntity('${_escForAttr(e.display_name)}')" class="px-2 py-1 text-left" title="${active ? 'Click to deselect' : _escHtmlG(preview)}">
                <span class="text-[9px] font-medium" style="color:${active ? '#c084fc' : 'var(--text-heading)'}">${_escHtmlG(e.display_name)}</span>
            </button>
            <button onclick="event.stopPropagation();_gamesDeleteEntity('${_escForAttr(e.name)}')" class="px-1 py-1 transition-all" style="color:var(--text-muted);opacity:0.4;" title="Delete ${_escHtmlG(e.display_name)}" onmouseenter="this.style.color='#f87171';this.style.opacity='1'" onmouseleave="this.style.color='var(--text-muted)';this.style.opacity='0.4'">
                <i data-lucide="x" class="w-2.5 h-2.5"></i>
            </button>
        </div>`;
    }).join('');

    bar.innerHTML = `<div class="flex items-center gap-1.5 overflow-x-auto pb-0.5" style="scrollbar-width:thin;scrollbar-color:rgba(168,85,247,0.3) transparent;">
        <span class="text-[9px] flex-shrink-0" style="color:var(--text-muted)">${entityLabel}s:</span>
        ${chips}
        <button onclick="_gamesCreateEntity()" class="flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded-md transition-all" style="color:var(--text-muted);border:1px dashed var(--border-strong);" title="Add ${entityLabel}" onmouseenter="this.style.color='${theme.color}'" onmouseleave="this.style.color='var(--text-muted)'">+</button>
    </div>`;
}

function _gamesSelectEntity(displayName) {
    // Toggle: clicking the active character deselects them
    if (_gamesActiveNpc === displayName) {
        _gamesActiveNpc = '';
    } else {
        _gamesActiveNpc = displayName;
    }
    const sel = document.getElementById('games-npc-input');
    if (sel) sel.value = _gamesActiveNpc;
    _renderEntityBar();
}
window._gamesSelectEntity = _gamesSelectEntity;

async function _gamesCreateEntity() {
    const persona = (typeof currentPersona !== 'undefined') ? currentPersona : 'gaming';
    const entityLabel = persona === 'scholarly' ? 'Figure' : 'Character';
    const result = await stratosPrompt({ title: `New ${entityLabel}`, fields: [
        { key: 'name', label: `${entityLabel} name`, placeholder: 'e.g., Tanjiro' },
        { key: 'personality', label: 'Personality', placeholder: 'Brave, kind-hearted...', optional: true }
    ]});
    if (!result) return;
    const displayName = result.name;
    const personality = result.personality || '';
    try {
        const r = await fetch(`/api/personas/${persona}/entities`, {
            method: 'POST',
            headers: _gamesHeaders(),
            body: JSON.stringify({
                scenario: _gamesActiveScenario || '',
                name: displayName.trim().toLowerCase().replace(/\s+/g, '_'),
                display_name: displayName.trim(),
                personality_md: personality,
                entity_type: persona === 'scholarly' ? 'figure' : 'character'
            })
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Created "${displayName.trim()}"`, 'success');
            _gamesLoadEntities();
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Failed to create', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create entity', 'error');
    }
}
window._gamesCreateEntity = _gamesCreateEntity;

async function _gamesDeleteEntity(name) {
    const persona = (typeof currentPersona !== 'undefined') ? currentPersona : 'gaming';
    if (!(await stratosConfirm(`Delete "${name}"?`, { title: 'Delete Character', okText: 'Delete', cancelText: 'Cancel' }))) return;
    try {
        await fetch(`/api/personas/${persona}/entities/${encodeURIComponent(name)}?scenario=${encodeURIComponent(_gamesActiveScenario || '')}`, {
            method: 'DELETE',
            headers: _gamesHeaders()
        });
        if (_gamesActiveNpc.toLowerCase().replace(/ /g, '_') === name) _gamesActiveNpc = '';
        _gamesLoadEntities();
        if (typeof showToast === 'function') showToast('Deleted', 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to delete', 'error');
    }
}
window._gamesDeleteEntity = _gamesDeleteEntity;

// ═══════════════════════════════════════════════════════════
// GAME-04: Interactive Stat Display
// ═══════════════════════════════════════════════════════════

let _gsStylesInjected = false;

function _injectGameStatStyles() {
    if (_gsStylesInjected) return;
    _gsStylesInjected = true;
    const style = document.createElement('style');
    style.textContent = `
        .gs-block{background:var(--bg-panel,rgba(255,255,255,0.03));border:1px solid var(--border-strong,#333);border-radius:12px;padding:12px 14px;margin:8px 0;font-family:inherit}
        .gs-row{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
        .gs-row:last-child{margin-bottom:0}
        .gs-hp-wrap{flex:1;min-width:160px}
        .gs-hp-label{font-size:11px;font-weight:600;color:var(--text-heading,#fff);margin-bottom:3px;display:flex;justify-content:space-between}
        .gs-hp-bar{height:8px;border-radius:4px;background:rgba(255,255,255,0.08);overflow:hidden}
        .gs-hp-fill{height:100%;border-radius:4px;transition:width .3s ease}
        .gs-stat-card{display:inline-flex;flex-direction:column;align-items:center;padding:6px 12px;border-radius:8px;border:1px solid var(--border-strong,#333);background:rgba(255,255,255,0.02);min-width:48px}
        .gs-stat-icon{font-size:14px;line-height:1}
        .gs-stat-label{font-size:9px;color:var(--text-muted,#888);margin-top:2px}
        .gs-stat-value{font-size:13px;font-weight:700;color:var(--text-heading,#fff)}
        .gs-inv-label{font-size:11px;font-weight:600;color:var(--text-heading,#fff);margin-bottom:4px}
        .gs-inv-pills{display:flex;flex-wrap:wrap;gap:4px}
        .gs-inv-pill{padding:3px 10px;border-radius:6px;font-size:10px;background:rgba(244,114,182,0.08);border:1px solid rgba(244,114,182,0.2);color:var(--accent,#f472b6)}
    `;
    document.head.appendChild(style);
}

/**
 * Parse emoji stat blocks from agent responses and replace with styled HTML cards.
 * Detects patterns like: ❤️ HP: 50/50 | ⚔ ATK: 4 | 🛡 DEF: 2
 * and: 📦 Inventory: Wooden Sword, Leather Armor
 *
 * @param {string} html — The agent response HTML
 * @returns {string} — HTML with stat blocks replaced by styled cards
 */
function formatGameStats(html) {
    if (!html || typeof html !== 'string') return html;

    // Pattern: line containing emoji stat blocks separated by |
    // Matches lines like: ❤️ HP: 50/50 | ⚔ ATK: 4 | 🛡 DEF: 2
    const statLineRegex = /^([^\n]*(?:❤️?|❤)\s*HP\s*:\s*(\d+)\s*\/\s*(\d+)\s*\|.+)$/gm;
    // Inventory line: 📦 Inventory: item1, item2, ...
    const invLineRegex = /^([^\n]*📦\s*Inventory\s*:\s*(.+))$/gm;

    let hasStats = statLineRegex.test(html) || invLineRegex.test(html);
    if (!hasStats) return html;

    _injectGameStatStyles();

    // Reset regex lastIndex after test
    statLineRegex.lastIndex = 0;
    invLineRegex.lastIndex = 0;

    // Replace stat lines
    html = html.replace(statLineRegex, function (match, fullLine, hpCur, hpMax) {
        const cur = parseInt(hpCur, 10);
        const max = parseInt(hpMax, 10);
        const pct = max > 0 ? Math.min(100, Math.round((cur / max) * 100)) : 0;

        // HP bar color: green > 60%, yellow > 30%, red <= 30%
        let hpColor = '#4ade80';
        if (pct <= 30) hpColor = '#f87171';
        else if (pct <= 60) hpColor = '#fbbf24';

        // Extract other stats from the line (ATK, DEF, etc.)
        const statParts = fullLine.split('|').slice(1); // skip HP part
        const statCards = statParts.map(function (part) {
            const trimmed = part.trim();
            // Match: emoji LABEL: VALUE
            const m = trimmed.match(/^([^\w]*)\s*(\w+)\s*:\s*(.+)$/);
            if (!m) return '';
            const icon = m[1].trim() || '';
            const label = m[2].trim();
            const value = m[3].trim();
            return `<div class="gs-stat-card">
                <span class="gs-stat-icon">${_escHtmlG(icon)}</span>
                <span class="gs-stat-value">${_escHtmlG(value)}</span>
                <span class="gs-stat-label">${_escHtmlG(label)}</span>
            </div>`;
        }).filter(Boolean).join('');

        return `<div class="gs-block">
            <div class="gs-row">
                <div class="gs-hp-wrap">
                    <div class="gs-hp-label"><span>HP</span><span>${cur}/${max}</span></div>
                    <div class="gs-hp-bar"><div class="gs-hp-fill" style="width:${pct}%;background:${hpColor}"></div></div>
                </div>
            </div>
            ${statCards ? `<div class="gs-row">${statCards}</div>` : ''}
        </div>`;
    });

    // Replace inventory lines
    html = html.replace(invLineRegex, function (match, fullLine, items) {
        const itemList = items.split(',').map(function (item) {
            const trimmed = item.trim();
            if (!trimmed) return '';
            return `<span class="gs-inv-pill">${_escHtmlG(trimmed)}</span>`;
        }).filter(Boolean).join('');

        return `<div class="gs-block">
            <div class="gs-inv-label">Inventory</div>
            <div class="gs-inv-pills">${itemList}</div>
        </div>`;
    });

    return html;
}

window.formatGameStats = formatGameStats;
