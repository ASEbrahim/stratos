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
    if (_gamesActiveScenario) _gamesLoadEntities();
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

// ── Create scenario ──
async function _gamesCreateScenario() {
    const result = await stratosPrompt({ title: 'New Scenario', fields: [
        { key: 'name', label: 'Scenario name', placeholder: 'e.g., Dragon_Quest' },
        { key: 'world', label: 'World description', placeholder: 'A dark fantasy setting...', optional: true }
    ]});
    if (!result) return;
    const name = result.name;
    const world = result.world || '';
    try {
        const r = await fetch('/api/scenarios/create', {
            method: 'POST',
            headers: _gamesHeaders(),
            body: JSON.stringify({ name: name.trim(), world })
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Created scenario "${name.trim()}"`, 'success');
            _loadScenarios();
            _refreshFileBrowserIfOpen();
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Failed to create scenario', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create scenario', 'error');
    }
}
window._gamesCreateScenario = _gamesCreateScenario;

// ── Create with genre preset ──
async function _gamesCreateScenarioWithGenre(genre) {
    const name = await stratosPrompt({ title: `New ${genre} Scenario`, label: 'Scenario name', placeholder: genre.replace(/\s/g, '_'), defaultValue: genre.replace(/\s/g, '_') });
    if (!name || !name.trim()) return;
    try {
        const r = await fetch('/api/scenarios/create', {
            method: 'POST',
            headers: _gamesHeaders(),
            body: JSON.stringify({ name: name.trim(), world: `A ${genre.toLowerCase()} setting.` })
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Created "${name.trim()}"`, 'success');
            _loadScenarios();
            _refreshFileBrowserIfOpen();
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Failed to create', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create scenario', 'error');
    }
}
window._gamesCreateScenarioWithGenre = _gamesCreateScenarioWithGenre;

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

    const chips = _gamesEntities.map(e => {
        const active = _gamesActiveNpc.toLowerCase().replace(/ /g, '_') === e.name;
        const preview = (e.personality_md || '').slice(0, 40) || e.entity_type;
        return `<button onclick="_gamesSelectEntity('${_escForAttr(e.display_name)}')" class="flex-shrink-0 px-2 py-1 rounded-md text-left transition-all" style="background:${active ? 'rgba(168,85,247,0.15)' : 'transparent'};border:1px solid ${active ? 'rgba(168,85,247,0.4)' : 'var(--border-strong)'};" title="${_escHtmlG(preview)}">
            <span class="text-[9px] font-medium" style="color:${active ? '#c084fc' : 'var(--text-heading)'}">${_escHtmlG(e.display_name)}</span>
        </button>`;
    }).join('');

    bar.innerHTML = `<div class="flex items-center gap-1.5 overflow-x-auto" style="scrollbar-width:none;">
        <span class="text-[9px] flex-shrink-0" style="color:var(--text-muted)">${entityLabel}s:</span>
        ${chips}
        <button onclick="_gamesCreateEntity()" class="flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded-md transition-all" style="color:var(--text-muted);border:1px dashed var(--border-strong);" title="Add ${entityLabel}" onmouseenter="this.style.color='${theme.color}'" onmouseleave="this.style.color='var(--text-muted)'">+</button>
    </div>`;
}

function _gamesSelectEntity(displayName) {
    _gamesActiveNpc = displayName;
    const sel = document.getElementById('games-npc-input');
    if (sel) sel.value = displayName;
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
