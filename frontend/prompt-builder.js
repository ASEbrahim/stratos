// === SPRINT PROMPT BUILDER ===
// Form UI that assembles sprint prompts from templates + live filesystem state.

(function() {
    'use strict';

    let _pbContext = null;
    let _pbPhases = [{ name: '', description: '' }];
    let _pbGenerated = '';
    let _pbView = 'build'; // 'build' | 'history' | 'templates'
    let _pbModal = null;

    // ── Open / Close ──
    window.openPromptBuilder = function() {
        if (_pbModal) { _pbModal.style.display = 'flex'; _loadContext(); return; }
        _createModal();
        _loadContext();
    };

    function _close() {
        if (_pbModal) _pbModal.style.display = 'none';
    }

    // ── Load live context ──
    async function _loadContext() {
        try {
            const token = localStorage.getItem('stratos_session_token') || '';
            const r = await fetch('/api/dev/context', { headers: { 'X-Auth-Token': token } });
            if (r.ok) {
                _pbContext = await r.json();
                _renderContextPreview();
            }
        } catch (e) { console.warn('PB: context load failed', e); }
    }

    // ── Create the modal DOM ──
    function _createModal() {
        _pbModal = document.createElement('div');
        _pbModal.className = 'pb-overlay';
        _pbModal.innerHTML = `
        <div class="pb-modal">
            <div class="pb-header">
                <div class="pb-title">Sprint Prompt Builder</div>
                <div class="pb-tabs">
                    <button class="pb-tab pb-tab-active" data-view="build" onclick="_pbSwitchView('build')">Build</button>
                    <button class="pb-tab" data-view="history" onclick="_pbSwitchView('history')">History</button>
                    <button class="pb-tab" data-view="templates" onclick="_pbSwitchView('templates')">Templates</button>
                </div>
                <button class="pb-close" onclick="document.querySelector('.pb-overlay').style.display='none'">&times;</button>
            </div>

            <div class="pb-body" id="pb-body">
                <!-- Build view (default) -->
                <div class="pb-view" id="pb-view-build">
                    <div class="pb-form">
                        <div class="pb-row">
                            <div class="pb-field pb-field-sm">
                                <label>Sprint #</label>
                                <input type="number" id="pb-sprint-num" class="pb-input" value="1">
                            </div>
                            <div class="pb-field pb-field-lg">
                                <label>Sprint Name</label>
                                <input type="text" id="pb-sprint-name" class="pb-input" placeholder="e.g. Narration Source Pipeline">
                            </div>
                        </div>

                        <div class="pb-row">
                            <div class="pb-field">
                                <label>Owner</label>
                                <div class="pb-radio-group" id="pb-owner-group">
                                    <label class="pb-radio"><input type="radio" name="pb-owner" value="kirissie" checked> Kirissie</label>
                                    <label class="pb-radio"><input type="radio" name="pb-owner" value="ahmad"> Ahmad</label>
                                    <label class="pb-radio"><input type="radio" name="pb-owner" value="either"> Either</label>
                                    <label class="pb-radio"><input type="radio" name="pb-owner" value="parallel"> Parallel</label>
                                </div>
                            </div>
                        </div>

                        <div class="pb-field">
                            <label>Feature Spec</label>
                            <textarea id="pb-feature-spec" class="pb-textarea" rows="5" placeholder="Describe what to build (markdown supported)..."></textarea>
                        </div>

                        <div class="pb-field">
                            <label>Phases <button class="pb-btn-sm" onclick="_pbAddPhase()">+ Add Phase</button></label>
                            <div id="pb-phases-list"></div>
                        </div>

                        <div class="pb-field">
                            <label>Include Sections</label>
                            <div class="pb-checkbox-grid">
                                <label class="pb-check"><input type="checkbox" data-section="safety_branch" checked> Safety branch + rollback</label>
                                <label class="pb-check"><input type="checkbox" data-section="do_not_touch" checked> Do-not-touch list</label>
                                <label class="pb-check"><input type="checkbox" data-section="commit_discipline" checked> Commit discipline</label>
                                <label class="pb-check"><input type="checkbox" data-section="handoff" checked> Handoff protocol</label>
                                <label class="pb-check"><input type="checkbox" data-section="smoke_tests" checked> Smoke test gate</label>
                                <label class="pb-check"><input type="checkbox" data-section="read_before_starting" checked> Read-before-starting</label>
                                <label class="pb-check"><input type="checkbox" data-section="priority_order" checked> Priority order</label>
                                <label class="pb-check"><input type="checkbox" data-section="parallel_rules"> Parallel session rules</label>
                            </div>
                        </div>

                        <div class="pb-field">
                            <label>Custom Rules <span class="pb-hint">(one per line)</span></label>
                            <textarea id="pb-custom-rules" class="pb-textarea pb-textarea-sm" rows="3" placeholder="e.g. Arabic patterns mandatory"></textarea>
                        </div>

                        <!-- Parallel mode fields -->
                        <div class="pb-parallel-fields" id="pb-parallel-fields" style="display:none;">
                            <div class="pb-field">
                                <label>Files Owned <span class="pb-hint">(comma-separated)</span></label>
                                <input type="text" id="pb-files-owned" class="pb-input" placeholder="backend/routes/my_route.py, frontend/my-module.js">
                            </div>
                            <div class="pb-field">
                                <label>Files Forbidden <span class="pb-hint">(comma-separated)</span></label>
                                <input type="text" id="pb-files-forbidden" class="pb-input" placeholder="frontend/*, agent.js">
                            </div>
                        </div>

                        <!-- Auto-populated context preview -->
                        <div class="pb-field">
                            <label>Auto-Populated Context <span class="pb-hint">(read-only, from filesystem)</span></label>
                            <div class="pb-context-preview" id="pb-context-preview">Loading...</div>
                        </div>

                        <div class="pb-actions">
                            <button class="pb-btn pb-btn-primary" onclick="_pbGenerate()">Generate Prompt</button>
                            <button class="pb-btn" onclick="_pbSaveTemplate()">Save Template</button>
                            <select class="pb-select" id="pb-load-template" onchange="_pbLoadTemplate(this.value)">
                                <option value="">Load Template...</option>
                            </select>
                        </div>
                    </div>

                    <!-- Generated preview -->
                    <div class="pb-preview-section" id="pb-preview-section" style="display:none;">
                        <div class="pb-preview-header">
                            <span>Generated Prompt</span>
                            <div class="pb-preview-actions">
                                <button class="pb-btn-sm" onclick="_pbCopy()">Copy</button>
                                <button class="pb-btn-sm" onclick="_pbDownload()">Download .md</button>
                                <button class="pb-btn-sm" onclick="_pbSaveToLog()">Save to Log</button>
                            </div>
                        </div>
                        <pre class="pb-preview" id="pb-preview"></pre>
                    </div>
                </div>

                <!-- History view -->
                <div class="pb-view" id="pb-view-history" style="display:none;"></div>

                <!-- Templates view -->
                <div class="pb-view" id="pb-view-templates" style="display:none;"></div>
            </div>
        </div>`;

        document.body.appendChild(_pbModal);
        _renderPhases();
        _setupOwnerToggle();
        _loadTemplateList();

        // Close on overlay click
        _pbModal.addEventListener('click', (e) => {
            if (e.target === _pbModal) _close();
        });
        // Esc to close
        _pbModal.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') _close();
        });
    }

    // ── Owner toggle: show parallel fields ──
    function _setupOwnerToggle() {
        const group = document.getElementById('pb-owner-group');
        if (!group) return;
        group.addEventListener('change', () => {
            const val = document.querySelector('input[name="pb-owner"]:checked')?.value;
            const pf = document.getElementById('pb-parallel-fields');
            if (pf) pf.style.display = val === 'parallel' ? 'block' : 'none';
        });
    }

    // ── Phase management ──
    function _renderPhases() {
        const list = document.getElementById('pb-phases-list');
        if (!list) return;
        list.innerHTML = _pbPhases.map((p, i) => `
            <div class="pb-phase" data-idx="${i}">
                <span class="pb-phase-handle">&#9776;</span>
                <span class="pb-phase-num">${i + 1}.</span>
                <input type="text" class="pb-input pb-phase-name" value="${_esc(p.name)}" placeholder="Phase name" oninput="_pbUpdatePhase(${i},'name',this.value)">
                <input type="text" class="pb-input pb-phase-desc" value="${_esc(p.description)}" placeholder="Description (optional)" oninput="_pbUpdatePhase(${i},'description',this.value)">
                <button class="pb-btn-x" onclick="_pbRemovePhase(${i})" title="Remove">&times;</button>
            </div>
        `).join('');

        // Simple drag-to-reorder
        _initPhaseDrag(list);
    }

    window._pbAddPhase = function() {
        _pbPhases.push({ name: '', description: '' });
        _renderPhases();
    };

    window._pbRemovePhase = function(idx) {
        if (_pbPhases.length <= 1) return;
        _pbPhases.splice(idx, 1);
        _renderPhases();
    };

    window._pbUpdatePhase = function(idx, key, val) {
        if (_pbPhases[idx]) _pbPhases[idx][key] = val;
    };

    function _initPhaseDrag(container) {
        let dragIdx = null;
        container.querySelectorAll('.pb-phase').forEach(el => {
            const handle = el.querySelector('.pb-phase-handle');
            handle.addEventListener('mousedown', (e) => {
                dragIdx = parseInt(el.dataset.idx);
                el.classList.add('pb-dragging');
                e.preventDefault();
            });
        });
        document.addEventListener('mouseup', () => {
            if (dragIdx !== null) {
                container.querySelectorAll('.pb-phase').forEach(el => el.classList.remove('pb-dragging'));
                dragIdx = null;
            }
        });
        container.addEventListener('mouseover', (e) => {
            if (dragIdx === null) return;
            const target = e.target.closest('.pb-phase');
            if (!target) return;
            const targetIdx = parseInt(target.dataset.idx);
            if (targetIdx !== dragIdx) {
                const moved = _pbPhases.splice(dragIdx, 1)[0];
                _pbPhases.splice(targetIdx, 0, moved);
                dragIdx = targetIdx;
                _renderPhases();
            }
        });
    }

    // ── Context preview ──
    function _renderContextPreview() {
        const el = document.getElementById('pb-context-preview');
        if (!el || !_pbContext) return;

        const c = _pbContext;
        const commits = (c.git_log || '').split('\n').slice(0, 5).join('\n');
        const tests = (c.test_files || []).length;
        const branches = (c.safety_branches || []).join(', ');
        const tables = (c.db_tables || []).length;

        el.innerHTML = `<div class="pb-ctx-section"><strong>Last commits:</strong><pre>${_esc(commits)}</pre></div>
<div class="pb-ctx-section"><strong>Tests:</strong> ${tests} spec files</div>
<div class="pb-ctx-section"><strong>Safety branches:</strong> ${_esc(branches || 'none')}</div>
<div class="pb-ctx-section"><strong>DB tables:</strong> ${tables}</div>
<div class="pb-ctx-section"><strong>Sprint #:</strong> ${c.sprint_number || '?'}</div>`;

        // Auto-fill sprint number
        const numEl = document.getElementById('pb-sprint-num');
        if (numEl && c.sprint_number) numEl.value = c.sprint_number;
    }

    // ── Generate prompt ──
    window._pbGenerate = async function() {
        const formData = _collectFormData();
        const token = localStorage.getItem('stratos_session_token') || '';

        try {
            const r = await fetch('/api/prompt-builder/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
                body: JSON.stringify(formData)
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const data = await r.json();
            _pbGenerated = data.prompt || '';
            const preview = document.getElementById('pb-preview');
            const section = document.getElementById('pb-preview-section');
            if (preview) preview.textContent = _pbGenerated;
            if (section) section.style.display = 'block';
            section?.scrollIntoView({ behavior: 'smooth' });
        } catch (e) {
            alert('Generate failed: ' + e.message);
        }
    };

    function _collectFormData() {
        const owner = document.querySelector('input[name="pb-owner"]:checked')?.value || 'kirissie';
        const customRulesText = document.getElementById('pb-custom-rules')?.value || '';
        const customRules = customRulesText.split('\n').map(s => s.trim()).filter(Boolean);
        const filesOwned = (document.getElementById('pb-files-owned')?.value || '').split(',').map(s => s.trim()).filter(Boolean);
        const filesForbidden = (document.getElementById('pb-files-forbidden')?.value || '').split(',').map(s => s.trim()).filter(Boolean);

        const include = {};
        document.querySelectorAll('.pb-checkbox-grid input[type="checkbox"]').forEach(cb => {
            include[cb.dataset.section] = cb.checked;
        });

        return {
            sprint_number: parseInt(document.getElementById('pb-sprint-num')?.value) || 1,
            sprint_name: document.getElementById('pb-sprint-name')?.value || '',
            owner,
            feature_spec: document.getElementById('pb-feature-spec')?.value || '',
            phases: _pbPhases.filter(p => p.name),
            include_sections: include,
            custom_rules: customRules,
            files_owned: filesOwned,
            files_forbidden: filesForbidden
        };
    }

    // ── Copy / Download / Save ──
    window._pbCopy = function() {
        if (!_pbGenerated) return;
        navigator.clipboard.writeText(_pbGenerated).then(() => {
            const btn = document.querySelector('.pb-preview-actions .pb-btn-sm');
            if (btn) { const orig = btn.textContent; btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = orig, 1500); }
        });
    };

    window._pbDownload = function() {
        if (!_pbGenerated) return;
        const num = document.getElementById('pb-sprint-num')?.value || '0';
        const name = (document.getElementById('pb-sprint-name')?.value || 'sprint').replace(/\s+/g, '_');
        const blob = new Blob([_pbGenerated], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `Sprint${num}_${name}.md`;
        a.click();
        URL.revokeObjectURL(a.href);
    };

    window._pbSaveToLog = async function() {
        if (!_pbGenerated) return;
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            await fetch('/api/dev/sprint-log', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
                body: JSON.stringify({
                    sprint_number: parseInt(document.getElementById('pb-sprint-num')?.value) || 0,
                    sprint_name: document.getElementById('pb-sprint-name')?.value || '',
                    owner: document.querySelector('input[name="pb-owner"]:checked')?.value || '',
                    prompt: _pbGenerated
                })
            });
            alert('Saved to sprint log');
        } catch (e) {
            alert('Save failed: ' + e.message);
        }
    };

    // ── Templates ──
    window._pbSaveTemplate = async function() {
        const name = prompt('Template name:');
        if (!name) return;
        const formData = _collectFormData();
        delete formData.feature_spec; // Templates exclude feature spec
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            await fetch('/api/dev/templates', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
                body: JSON.stringify({ name, template: formData })
            });
            _loadTemplateList();
        } catch (e) { console.error('Save template failed', e); }
    };

    async function _loadTemplateList() {
        const sel = document.getElementById('pb-load-template');
        if (!sel) return;
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            const r = await fetch('/api/dev/templates', { headers: { 'X-Auth-Token': token } });
            if (!r.ok) return;
            const data = await r.json();
            sel.innerHTML = '<option value="">Load Template...</option>';
            (data.templates || []).forEach(t => {
                sel.innerHTML += `<option value='${_esc(JSON.stringify(t.template))}'>${_esc(t.name)}</option>`;
            });
        } catch (e) { /* ignore */ }
    }

    window._pbLoadTemplate = function(jsonStr) {
        if (!jsonStr) return;
        try {
            const t = JSON.parse(jsonStr);
            if (t.sprint_name) document.getElementById('pb-sprint-name').value = t.sprint_name;
            if (t.owner) {
                const radio = document.querySelector(`input[name="pb-owner"][value="${t.owner}"]`);
                if (radio) radio.checked = true;
            }
            if (t.phases && t.phases.length) {
                _pbPhases = t.phases;
                _renderPhases();
            }
            if (t.include_sections) {
                document.querySelectorAll('.pb-checkbox-grid input[type="checkbox"]').forEach(cb => {
                    if (cb.dataset.section in t.include_sections) cb.checked = t.include_sections[cb.dataset.section];
                });
            }
            if (t.custom_rules) document.getElementById('pb-custom-rules').value = t.custom_rules.join('\n');
            if (t.files_owned) document.getElementById('pb-files-owned').value = t.files_owned.join(', ');
            if (t.files_forbidden) document.getElementById('pb-files-forbidden').value = t.files_forbidden.join(', ');
        } catch (e) { console.error('Load template failed', e); }
    };

    // ── View switching ──
    window._pbSwitchView = function(view) {
        _pbView = view;
        document.querySelectorAll('.pb-tab').forEach(t => t.classList.toggle('pb-tab-active', t.dataset.view === view));
        document.querySelectorAll('.pb-view').forEach(v => v.style.display = 'none');
        const target = document.getElementById('pb-view-' + view);
        if (target) target.style.display = 'block';

        if (view === 'history') _loadHistory();
        if (view === 'templates') _loadTemplatesView();
    };

    // ── History view ──
    async function _loadHistory() {
        const container = document.getElementById('pb-view-history');
        if (!container) return;
        container.innerHTML = '<div class="pb-loading">Loading...</div>';
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            const r = await fetch('/api/dev/sprint-log', { headers: { 'X-Auth-Token': token } });
            if (!r.ok) throw new Error('Failed');
            const data = await r.json();
            const entries = data.entries || [];
            if (!entries.length) {
                container.innerHTML = '<div class="pb-empty">No sprint history yet. Generate and save a prompt to start.</div>';
                return;
            }
            container.innerHTML = entries.map(e => `
                <div class="pb-history-item">
                    <div class="pb-history-header">
                        <strong>Sprint ${e.sprint_number} — ${_esc(e.sprint_name)}</strong>
                        <span class="pb-history-meta">${_esc(e.created_at || '')} &middot; ${_esc(e.owner || '')}</span>
                    </div>
                    <div class="pb-history-actions">
                        <button class="pb-btn-sm" onclick="_pbViewPrompt(${e.id})">View</button>
                        <button class="pb-btn-sm" onclick="_pbCopyPrompt(${e.id})">Copy</button>
                        <select class="pb-select-sm" onchange="_pbUpdateStatus(${e.id}, this.value)">
                            <option value="generated" ${e.status === 'generated' ? 'selected' : ''}>Generated</option>
                            <option value="in_progress" ${e.status === 'in_progress' ? 'selected' : ''}>In Progress</option>
                            <option value="completed" ${e.status === 'completed' ? 'selected' : ''}>Completed</option>
                        </select>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            container.innerHTML = '<div class="pb-empty">Failed to load history.</div>';
        }
    }

    window._pbViewPrompt = async function(id) {
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            const r = await fetch(`/api/dev/sprint-log/prompt?id=${id}`, { headers: { 'X-Auth-Token': token } });
            if (!r.ok) return;
            const data = await r.json();
            // Show in a simple overlay
            const overlay = document.createElement('div');
            overlay.className = 'pb-prompt-viewer';
            overlay.innerHTML = `<div class="pb-prompt-viewer-inner">
                <div class="pb-prompt-viewer-header"><span>Sprint Prompt</span><button onclick="this.closest('.pb-prompt-viewer').remove()">&times;</button></div>
                <pre class="pb-preview">${_esc(data.prompt || '(empty)')}</pre>
            </div>`;
            document.body.appendChild(overlay);
            overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
        } catch (e) { alert('Failed to load prompt'); }
    };

    window._pbCopyPrompt = async function(id) {
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            const r = await fetch(`/api/dev/sprint-log/prompt?id=${id}`, { headers: { 'X-Auth-Token': token } });
            if (!r.ok) return;
            const data = await r.json();
            await navigator.clipboard.writeText(data.prompt || '');
        } catch (e) { alert('Failed to copy'); }
    };

    window._pbUpdateStatus = async function(id, status) {
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            await fetch('/api/dev/sprint-log', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
                body: JSON.stringify({ action: 'update_status', id, status })
            });
        } catch (e) { console.error('Status update failed', e); }
    };

    // ── Templates view ──
    async function _loadTemplatesView() {
        const container = document.getElementById('pb-view-templates');
        if (!container) return;
        container.innerHTML = '<div class="pb-loading">Loading...</div>';
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            const r = await fetch('/api/dev/templates', { headers: { 'X-Auth-Token': token } });
            if (!r.ok) throw new Error('Failed');
            const data = await r.json();
            const templates = data.templates || [];
            if (!templates.length) {
                container.innerHTML = '<div class="pb-empty">No templates saved. Build a prompt and click "Save Template".</div>';
                return;
            }
            container.innerHTML = templates.map(t => `
                <div class="pb-template-item">
                    <strong>${_esc(t.name)}</strong>
                    <span class="pb-history-meta">${_esc(t.created_at || '')}</span>
                    <div class="pb-history-actions">
                        <button class="pb-btn-sm" onclick="_pbApplyTemplate('${_esc(JSON.stringify(t.template).replace(/'/g, "\\'"))}')">Use</button>
                        <button class="pb-btn-sm pb-btn-danger" onclick="_pbDeleteTemplate(${t.id})">Delete</button>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            container.innerHTML = '<div class="pb-empty">Failed to load templates.</div>';
        }
    }

    window._pbApplyTemplate = function(jsonStr) {
        _pbSwitchView('build');
        _pbLoadTemplate(jsonStr);
    };

    window._pbDeleteTemplate = async function(id) {
        if (!confirm('Delete this template?')) return;
        const token = localStorage.getItem('stratos_session_token') || '';
        try {
            await fetch('/api/dev/templates', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
                body: JSON.stringify({ action: 'delete', id })
            });
            _loadTemplatesView();
            _loadTemplateList();
        } catch (e) { console.error('Delete failed', e); }
    };

    // ── Helpers ──
    function _esc(s) {
        if (!s) return '';
        const div = document.createElement('div');
        div.textContent = String(s);
        return div.innerHTML;
    }

})();
