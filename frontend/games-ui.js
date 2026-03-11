// ═══════════════════════════════════════════════════════════
// GAMES UI — Scenario selector for Games/Roleplay persona
// ═══════════════════════════════════════════════════════════

let _gamesScenarios = [];
let _gamesActiveScenario = null;

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

    const select = _gamesScenarios.map(s => {
        const active = s.name === _gamesActiveScenario;
        return `<option value="${s.name}" ${active ? 'selected' : ''}>${s.name}${active ? ' (active)' : ''}</option>`;
    }).join('');

    bar.innerHTML = `
        <div class="flex items-center gap-2">
            <i data-lucide="swords" class="w-3 h-3" style="color:var(--accent)"></i>
            <select onchange="_gamesActivateScenario(this.value)" class="text-[10px] font-mono px-2 py-1 rounded cursor-pointer" style="background:var(--bg-panel-solid);color:var(--text-secondary);border:1px solid var(--border-strong);outline:none;">
                ${_gamesScenarios.length ? select : '<option value="">No scenarios</option>'}
            </select>
            <button onclick="_gamesCreateScenario()" class="fb-tool-btn" title="New scenario">
                <i data-lucide="plus" class="w-3 h-3"></i>
            </button>
            ${_gamesActiveScenario ? `<button onclick="_gamesDeleteScenario('${_gamesActiveScenario}')" class="fb-tool-btn" title="Delete active scenario" onmouseenter="this.style.color='#f87171'" onmouseleave="this.style.color='var(--text-muted)'"><i data-lucide="trash-2" class="w-3 h-3"></i></button>` : ''}
            ${_gamesActiveScenario ? `<button onclick="toggleFileBrowser('gaming')" class="fb-tool-btn" title="Browse world files"><i data-lucide="folder-open" class="w-3 h-3"></i></button>` : ''}
        </div>`;
    lucide.createIcons();
}

// ── Create scenario ──
async function _gamesCreateScenario() {
    const name = prompt('Scenario name (e.g., Dragon_Quest):');
    if (!name || !name.trim()) return;
    const world = prompt('World description (optional):') || '';
    try {
        const r = await fetch('/api/scenarios/create', {
            method: 'POST',
            headers: _gamesHeaders(),
            body: JSON.stringify({ name: name.trim(), world })
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Created scenario "${name.trim()}"`, 'success');
            _loadScenarios();
        } else {
            const d = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(d.error || 'Failed to create scenario', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create scenario', 'error');
    }
}
window._gamesCreateScenario = _gamesCreateScenario;

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
        if (typeof showToast === 'function') showToast(`Switched to "${name}"`, 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to activate scenario', 'error');
    }
}
window._gamesActivateScenario = _gamesActivateScenario;

// ── Delete scenario ──
async function _gamesDeleteScenario(name) {
    if (!confirm(`Delete scenario "${name}"? This cannot be undone.`)) return;
    try {
        await fetch(`/api/scenarios?name=${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: _gamesHeaders()
        });
        if (_gamesActiveScenario === name) _gamesActiveScenario = null;
        _loadScenarios();
        if (typeof showToast === 'function') showToast(`Deleted "${name}"`, 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to delete scenario', 'error');
    }
}
window._gamesDeleteScenario = _gamesDeleteScenario;
