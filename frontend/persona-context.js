// ═══════════════════════════════════════════════════════════
// PERSONA CONTEXT EDITOR — View/edit per-persona system context
// ═══════════════════════════════════════════════════════════

let _ctxEditorOpen = false;
let _ctxEditorPersona = 'intelligence';
let _ctxOriginalContent = '';
let _ctxVersions = [];

function _ctxHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

// ── Toggle context editor panel ──
function toggleContextEditor() {
    _ctxEditorOpen = !_ctxEditorOpen;
    let panel = document.getElementById('context-editor-panel');
    if (!panel) {
        _createContextEditorPanel();
        panel = document.getElementById('context-editor-panel');
    }
    if (_ctxEditorOpen) {
        // Close file browser if open
        if (typeof _fbOpen !== 'undefined' && _fbOpen) toggleFileBrowser();
        _ctxEditorPersona = currentPersona || 'intelligence';
        panel.classList.remove('hidden');
        panel.classList.add('ctx-slide-in');
        _loadContextForPersona(_ctxEditorPersona);
    } else {
        panel.classList.add('hidden');
        panel.classList.remove('ctx-slide-in');
    }
}
window.toggleContextEditor = toggleContextEditor;

// ── Create the context editor panel DOM ──
function _createContextEditorPanel() {
    const panel = document.createElement('div');
    panel.id = 'context-editor-panel';
    panel.className = 'hidden';
    panel.style.cssText = 'position:fixed;inset:0;z-index:10000;';
    panel.innerHTML = `
        <div class="ctx-editor-backdrop" onclick="toggleContextEditor()"></div>
        <div class="ctx-editor-sidebar">
            <div class="ctx-editor-header">
                <div class="flex items-center gap-2">
                    <i data-lucide="file-text" class="w-4 h-4" style="color:var(--accent)"></i>
                    <span class="text-sm font-bold" style="color:var(--text-heading)">Persona Context</span>
                </div>
                <div class="flex items-center gap-2">
                    <select id="ctx-persona-select" onchange="_ctxSwitchPersona(this.value)" class="text-[10px] font-mono px-2 py-1 rounded cursor-pointer" style="background:var(--bg-panel-solid); color:var(--accent); border:1px solid var(--border-strong); outline:none;">
                    </select>
                    <button onclick="toggleContextEditor()" class="p-1 rounded-md transition-colors" style="color:var(--text-muted)" onmouseenter="this.style.color='var(--text-heading)'" onmouseleave="this.style.color='var(--text-muted)'" title="Close">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>
            </div>
            <div class="ctx-editor-body">
                <div class="ctx-editor-info">
                    <p class="text-[10px]" style="color:var(--text-muted)">
                        This context is included in every conversation with this persona. Use it to set preferences, instructions, or background info.
                    </p>
                </div>
                <textarea id="ctx-editor-textarea" class="ctx-editor-textarea" placeholder="Enter persona context here...

Example: I'm interested in semiconductor industry trends. Always cite sources when discussing historical events."></textarea>
                <div class="ctx-editor-actions">
                    <div class="flex items-center gap-2">
                        <button onclick="_ctxSave()" class="ctx-btn ctx-btn-save">
                            <i data-lucide="save" class="w-3 h-3"></i> Save
                        </button>
                        <button onclick="_ctxReset()" class="ctx-btn ctx-btn-reset" title="Reset to default context">
                            <i data-lucide="rotate-ccw" class="w-3 h-3"></i> Reset
                        </button>
                        <span id="ctx-save-status" class="text-[10px]" style="color:var(--accent); opacity:0;transition:opacity 0.3s;"></span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span id="ctx-modified-badge" class="hidden text-[9px] px-1.5 py-0.5 rounded-full" style="background:rgba(251,191,36,0.1);color:#fbbf24;border:1px solid rgba(251,191,36,0.2);">Modified</span>
                        <select id="ctx-version-select" onchange="_ctxRevert(this.value)" class="text-[9px] font-mono px-1.5 py-0.5 rounded cursor-pointer" style="background:var(--bg-panel-solid); color:var(--text-muted); border:1px solid var(--border-strong); outline:none;">
                            <option value="">Version History</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>`;
    document.body.appendChild(panel);
    lucide.createIcons();
    _populateCtxPersonaSelect();
}

// ── Populate persona dropdown in context editor ──
function _populateCtxPersonaSelect() {
    const select = document.getElementById('ctx-persona-select');
    if (!select) return;
    const personas = availablePersonas.length ? availablePersonas : [
        {name:'intelligence'},{name:'market'},{name:'scholarly'},{name:'gaming'},{name:'anime'},{name:'tcg'}
    ];
    select.innerHTML = '';
    for (const p of personas) {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name.charAt(0).toUpperCase() + p.name.slice(1);
        if (p.name === _ctxEditorPersona) opt.selected = true;
        select.appendChild(opt);
    }
}

// ── Switch persona in context editor ──
function _ctxSwitchPersona(name) {
    _ctxEditorPersona = name;
    _loadContextForPersona(name);
}

// ── Load context content for a persona ──
async function _loadContextForPersona(persona) {
    const textarea = document.getElementById('ctx-editor-textarea');
    const versionSelect = document.getElementById('ctx-version-select');
    const badge = document.getElementById('ctx-modified-badge');
    if (!textarea) return;

    textarea.value = 'Loading...';
    textarea.disabled = true;

    try {
        const r = await fetch(`/api/persona-context?persona=${encodeURIComponent(persona)}&key=system_context`, {
            headers: _ctxHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            _ctxOriginalContent = d.content || '';
            textarea.value = _ctxOriginalContent;
            if (badge) badge.classList.toggle('hidden', !_ctxOriginalContent);
        } else {
            _ctxOriginalContent = '';
            textarea.value = '';
            if (badge) badge.classList.add('hidden');
        }
    } catch (e) {
        textarea.value = '';
        _ctxOriginalContent = '';
    }
    textarea.disabled = false;

    // Load version history
    _loadVersionHistory(persona);
}

// ── Load version history ──
async function _loadVersionHistory(persona) {
    const versionSelect = document.getElementById('ctx-version-select');
    if (!versionSelect) return;
    versionSelect.innerHTML = '<option value="">Version History</option>';

    try {
        const r = await fetch(`/api/persona-context/versions?persona=${encodeURIComponent(persona)}&key=system_context`, {
            headers: _ctxHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            _ctxVersions = d.versions || [];
            for (const v of _ctxVersions) {
                const opt = document.createElement('option');
                opt.value = v.filename;
                const date = v.timestamp ? v.timestamp.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3 $4:$5') : v.filename;
                opt.textContent = `${date} (${_formatBytes(v.size)})`;
                versionSelect.appendChild(opt);
            }
        }
    } catch (e) {}
}

function _formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    return (bytes / 1024).toFixed(1) + ' KB';
}

// ── Save context ──
async function _ctxSave() {
    const textarea = document.getElementById('ctx-editor-textarea');
    const status = document.getElementById('ctx-save-status');
    if (!textarea) return;

    try {
        const r = await fetch('/api/persona-context', {
            method: 'POST',
            headers: _ctxHeaders(),
            body: JSON.stringify({
                persona: _ctxEditorPersona,
                key: 'system_context',
                content: textarea.value
            })
        });
        if (r.ok) {
            _ctxOriginalContent = textarea.value;
            if (status) {
                status.textContent = 'Saved!';
                status.style.opacity = '1';
                setTimeout(() => { status.style.opacity = '0'; }, 2000);
            }
            const badge = document.getElementById('ctx-modified-badge');
            if (badge) badge.classList.toggle('hidden', !textarea.value);
            // Update context button indicator
            _updateCtxIndicator(_ctxEditorPersona, !!textarea.value);
            // Refresh version history
            _loadVersionHistory(_ctxEditorPersona);
            if (typeof showToast === 'function') showToast('Context saved', 'success');
        } else {
            if (typeof showToast === 'function') showToast('Failed to save context', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to save context', 'error');
    }
}

// ── Reset context to default ──
async function _ctxReset() {
    if (!confirm(`Reset ${_ctxEditorPersona} context to default? This cannot be undone.`)) return;
    try {
        const r = await fetch(`/api/persona-context?persona=${encodeURIComponent(_ctxEditorPersona)}&key=system_context`, {
            method: 'DELETE',
            headers: _ctxHeaders()
        });
        if (r.ok) {
            const textarea = document.getElementById('ctx-editor-textarea');
            if (textarea) textarea.value = '';
            _ctxOriginalContent = '';
            const badge = document.getElementById('ctx-modified-badge');
            if (badge) badge.classList.add('hidden');
            _updateCtxIndicator(_ctxEditorPersona, false);
            if (typeof showToast === 'function') showToast('Context reset to default', 'success');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to reset context', 'error');
    }
}

// ── Revert to a previous version ──
async function _ctxRevert(filename) {
    if (!filename) return;
    if (!confirm(`Revert to this version? Current content will be saved as a new version.`)) {
        document.getElementById('ctx-version-select').value = '';
        return;
    }
    try {
        const r = await fetch('/api/persona-context/revert', {
            method: 'POST',
            headers: _ctxHeaders(),
            body: JSON.stringify({
                persona: _ctxEditorPersona,
                version: filename
            })
        });
        if (r.ok) {
            _loadContextForPersona(_ctxEditorPersona);
            if (typeof showToast === 'function') showToast('Reverted to previous version', 'success');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to revert', 'error');
    }
    document.getElementById('ctx-version-select').value = '';
}

// ── Update indicator dot on context button ──
function _updateCtxIndicator(persona, hasContext) {
    const dot = document.getElementById('ctx-indicator-dot');
    if (dot) {
        dot.style.display = hasContext ? 'block' : 'none';
    }
}

// ── Sync context editor when persona changes in agent panel ──
function _onPersonaChanged(newPersona) {
    if (_ctxEditorOpen) {
        _ctxEditorPersona = newPersona;
        const select = document.getElementById('ctx-persona-select');
        if (select) select.value = newPersona;
        _loadContextForPersona(newPersona);
    }
    // Check if this persona has custom context (for indicator dot)
    _checkPersonaContext(newPersona);
}

async function _checkPersonaContext(persona) {
    try {
        const r = await fetch(`/api/persona-context?persona=${encodeURIComponent(persona)}&key=system_context`, {
            headers: _ctxHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            _updateCtxIndicator(persona, !!(d.content));
        }
    } catch (e) {}
}
