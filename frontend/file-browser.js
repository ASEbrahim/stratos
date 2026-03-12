// ═══════════════════════════════════════════════════════════
// FILE EXPLORER — Ubuntu-style modal file browser + editor
// ═══════════════════════════════════════════════════════════

let _fbOpen = false;
let _fbPersona = 'intelligence';
let _fbPath = '/';
let _fbEntries = [];
let _fbEditorDirty = false;
let _fbEditorPath = '';

const _FB_ICONS = {
    folder: 'folder', md: 'file-text', txt: 'file-text', text: 'file-text',
    json: 'file-json', yaml: 'file-json', yml: 'file-json', csv: 'table',
    pdf: 'file', png: 'image', jpg: 'image', jpeg: 'image', gif: 'image',
    webp: 'image', svg: 'image', wav: 'volume-2', mp3: 'volume-2', default: 'file'
};

const _FB_TOOLBAR = [
    { key: 'B', prefix: '**', suffix: '**', icon: 'bold' },
    { key: 'I', prefix: '_', suffix: '_', icon: 'italic' },
    { key: 'H2', prefix: '## ', suffix: '', icon: 'heading-2' },
    { key: '•', prefix: '- ', suffix: '', icon: 'list' },
    { key: '1.', prefix: '1. ', suffix: '', icon: 'list-ordered' },
    { key: '```', prefix: '```\n', suffix: '\n```', icon: 'code' },
    { key: '—', prefix: '\n---\n', suffix: '', icon: 'minus' },
];

function _fbHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

// ── Toggle file explorer ──
function toggleFileBrowser(persona) {
    if (_fbOpen) {
        _fbClose();
        return;
    }
    _fbOpen = true;
    _fbPersona = persona || (typeof currentPersona !== 'undefined' ? currentPersona : 'intelligence');
    _fbPath = '/';
    _fbEditorDirty = false;
    _fbEditorPath = '';

    let modal = document.getElementById('file-explorer-modal');
    if (!modal) {
        _fbCreateModal();
        modal = document.getElementById('file-explorer-modal');
    }
    modal.classList.remove('hidden');
    _fbPopulatePersonaSelect();
    _fbLoadDir('/');
    _fbShowList();
}
window.toggleFileBrowser = toggleFileBrowser;

function _fbClose() {
    if (_fbEditorDirty && !confirm('Unsaved changes will be lost. Close anyway?')) return;
    _fbOpen = false;
    _fbEditorDirty = false;
    const modal = document.getElementById('file-explorer-modal');
    if (modal) modal.classList.add('hidden');
}

// ── Create Modal DOM ──
function _fbCreateModal() {
    const modal = document.createElement('div');
    modal.id = 'file-explorer-modal';
    modal.className = 'hidden';
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;';
    modal.onclick = (e) => { if (e.target === modal) _fbClose(); };

    modal.innerHTML = `
        <div class="fe-container" onclick="event.stopPropagation()">
            <!-- Header / breadcrumb -->
            <div class="fe-header">
                <div class="flex items-center gap-2 flex-1 min-w-0 overflow-x-auto" style="scrollbar-width:none;">
                    <button onclick="_fbNavigateUp()" class="fe-btn flex-shrink-0" title="Go up">
                        <i data-lucide="arrow-left" class="w-3.5 h-3.5"></i>
                    </button>
                    <div id="fe-breadcrumb" class="flex items-center gap-1 text-[11px] flex-shrink-0"></div>
                </div>
                <div class="flex items-center gap-2 flex-shrink-0">
                    <select id="fb-persona-select" onchange="_fbSwitchPersona(this.value)" class="text-[10px] px-2 py-1 rounded cursor-pointer" style="background:var(--bg-panel-solid);color:var(--accent);border:1px solid var(--border-strong);outline:none;"></select>
                    <button onclick="_fbClose()" class="fe-btn" title="Close"><i data-lucide="x" class="w-4 h-4"></i></button>
                </div>
            </div>

            <!-- File list view -->
            <div id="fe-list-view" style="flex:1;display:flex;flex-direction:column;min-height:0;">
                <div id="fe-file-list" class="flex-1 overflow-y-auto" style="min-height:0;"></div>
                <div class="fe-action-bar">
                    <button onclick="_fbCreateFile()" class="fe-action-btn"><i data-lucide="file-plus" class="w-3.5 h-3.5"></i> New File</button>
                    <button onclick="_fbCreateFolder()" class="fe-action-btn"><i data-lucide="folder-plus" class="w-3.5 h-3.5"></i> New Folder</button>
                    <label class="fe-action-btn cursor-pointer"><i data-lucide="upload" class="w-3.5 h-3.5"></i> Upload<input type="file" class="hidden" onchange="_fbUpload(event)"></label>
                    <div class="flex-1"></div>
                    <button onclick="_fbLoadDir(_fbPath)" class="fe-btn" title="Refresh"><i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i></button>
                </div>
            </div>

            <!-- Editor view (hidden by default) -->
            <div id="fe-editor-view" style="flex:1;flex-direction:column;min-height:0;display:none;">
                <!-- Editor toolbar -->
                <div class="flex items-center gap-1 px-3 py-1.5" style="border-bottom:1px solid var(--border-strong);">
                    <button onclick="_fbCloseEditor()" class="fe-btn" title="Back to files"><i data-lucide="arrow-left" class="w-3.5 h-3.5"></i></button>
                    <span id="fe-editor-name" class="text-[11px] font-mono flex-1 truncate" style="color:var(--text-secondary)"></span>
                    <span id="fe-editor-dirty" class="hidden text-[9px] px-1.5 py-0.5 rounded" style="background:rgba(251,191,36,0.15);color:#fbbf24;">unsaved</span>
                    <div class="fe-toolbar-divider"></div>
                    ${_FB_TOOLBAR.map(t => `<button onclick="_fbInsertFormat('${t.prefix.replace(/\n/g,'\\n')}','${t.suffix.replace(/\n/g,'\\n')}')" class="fe-fmt-btn" title="${t.key}"><i data-lucide="${t.icon}" class="w-3 h-3"></i></button>`).join('')}
                    <div class="fe-toolbar-divider"></div>
                    <button onclick="_fbTogglePreview()" id="fe-preview-btn" class="fe-fmt-btn" title="Preview (Ctrl+Shift+P)"><i data-lucide="eye" class="w-3 h-3"></i></button>
                    <button onclick="_fbAiAssistMenu(event)" class="fe-ai-btn" title="AI Assist"><i data-lucide="sparkles" class="w-3 h-3"></i> AI</button>
                </div>
                <!-- Editor + preview -->
                <textarea id="fe-editor-textarea" class="fe-textarea" spellcheck="false"></textarea>
                <div id="fe-preview-pane" class="hidden fe-preview"></div>
                <!-- Status bar -->
                <div class="fe-status-bar">
                    <button onclick="_fbSaveFile()" class="fe-save-btn"><i data-lucide="save" class="w-3 h-3"></i> Save</button>
                    <button onclick="_fbTogglePreview()" class="fe-action-btn text-[10px]"><i data-lucide="eye" class="w-3 h-3"></i> Preview</button>
                    <div class="flex-1"></div>
                    <span id="fe-word-count" class="text-[10px]" style="color:var(--text-faint)"></span>
                </div>
            </div>
        </div>`;

    document.body.appendChild(modal);
    lucide.createIcons();

    // Keyboard shortcuts
    const textarea = document.getElementById('fe-editor-textarea');
    if (textarea) {
        textarea.addEventListener('input', () => {
            _fbEditorDirty = true;
            const dirty = document.getElementById('fe-editor-dirty');
            if (dirty) dirty.classList.remove('hidden');
            _fbUpdateWordCount();
        });
        textarea.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's') { e.preventDefault(); _fbSaveFile(); }
            if (e.ctrlKey && e.shiftKey && e.key === 'P') { e.preventDefault(); _fbTogglePreview(); }
        });
    }
}

function _fbPopulatePersonaSelect() {
    const select = document.getElementById('fb-persona-select');
    if (!select) return;
    const personas = typeof availablePersonas !== 'undefined' && availablePersonas.length ? availablePersonas : [
        {name:'intelligence'},{name:'market'},{name:'scholarly'},{name:'gaming'},{name:'anime'},{name:'tcg'}
    ];
    select.innerHTML = '';
    for (const p of personas) {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name.charAt(0).toUpperCase() + p.name.slice(1);
        if (p.name === _fbPersona) opt.selected = true;
        select.appendChild(opt);
    }
}

function _fbSwitchPersona(name) {
    _fbPersona = name;
    _fbPath = '/';
    _fbShowList();
    _fbLoadDir('/');
}

// ── View switching ──
function _fbShowList() {
    const listView = document.getElementById('fe-list-view');
    const editorView = document.getElementById('fe-editor-view');
    if (listView) { listView.style.display = 'flex'; }
    if (editorView) { editorView.style.display = 'none'; }
}

function _fbShowEditor() {
    const listView = document.getElementById('fe-list-view');
    const editorView = document.getElementById('fe-editor-view');
    if (listView) { listView.style.display = 'none'; }
    if (editorView) { editorView.style.display = 'flex'; }
    // Hide preview, show textarea
    const ta = document.getElementById('fe-editor-textarea');
    const prev = document.getElementById('fe-preview-pane');
    if (ta) { ta.style.display = ''; ta.classList.remove('hidden'); }
    if (prev) { prev.style.display = 'none'; prev.classList.add('hidden'); }
}

// ── Breadcrumb ──
function _fbRenderBreadcrumb() {
    const el = document.getElementById('fe-breadcrumb');
    if (!el) return;
    const personaIcon = { gaming: '🎮', scholarly: '📚', market: '📊', intelligence: '🔍', anime: '🎌', tcg: '🃏' };
    const icon = personaIcon[_fbPersona] || '📁';
    const parts = _fbPath.split('/').filter(Boolean);

    let html = `<button onclick="_fbLoadDir('/')" class="fe-crumb">${icon} ${_fbPersona.charAt(0).toUpperCase() + _fbPersona.slice(1)}</button>`;
    let accumulated = '';
    for (const part of parts) {
        accumulated += '/' + part;
        const p = accumulated;
        html += `<span style="color:var(--text-faint)">›</span>`;
        html += `<button onclick="_fbLoadDir('${_escAttr(p)}')" class="fe-crumb">${_escHtml(part)}</button>`;
    }
    el.innerHTML = html;
}

// ── Load directory ──
async function _fbLoadDir(path) {
    _fbPath = path;
    _fbRenderBreadcrumb();
    const list = document.getElementById('fe-file-list');
    if (!list) return;
    list.innerHTML = '<div class="text-[10px] text-center py-12" style="color:var(--text-muted)">Loading...</div>';

    try {
        const r = await fetch(`/api/persona-files?persona=${encodeURIComponent(_fbPersona)}&path=${encodeURIComponent(path)}`, {
            headers: _fbHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            _fbEntries = d.entries || [];
            _fbRenderEntries();
        } else {
            list.innerHTML = '<div class="text-[10px] text-center py-12" style="color:var(--text-muted)">No files yet</div>';
        }
    } catch (e) {
        list.innerHTML = '<div class="text-[10px] text-center py-12 text-red-400">Failed to load</div>';
    }
}

function _fbRenderEntries() {
    const list = document.getElementById('fe-file-list');
    if (!list) return;

    if (_fbEntries.length === 0) {
        list.innerHTML = '<div class="text-[10px] text-center py-12" style="color:var(--text-muted)">Empty folder — create a file or upload</div>';
        return;
    }

    const sorted = [..._fbEntries].sort((a, b) => {
        if (a.type === 'directory' && b.type !== 'directory') return -1;
        if (a.type !== 'directory' && b.type === 'directory') return 1;
        return a.name.localeCompare(b.name);
    });

    list.innerHTML = sorted.map(entry => {
        const isDir = entry.type === 'directory';
        const ext = entry.name.split('.').pop().toLowerCase();
        const icon = isDir ? 'folder' : (_FB_ICONS[ext] || _FB_ICONS.default);
        const size = isDir ? '—' : _formatBytes(entry.size || 0);
        const date = entry.modified ? new Date(entry.modified).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '';
        const clickAction = isDir
            ? `_fbLoadDir('${_escAttr(entry.path)}')`
            : `_fbOpenFile('${_escAttr(entry.path)}', '${_escAttr(entry.name)}')`;

        return `<div class="fe-row" onclick="${clickAction}">
            <div class="fe-icon"><i data-lucide="${icon}" class="w-5 h-5" style="color:${isDir ? 'var(--accent)' : 'var(--text-muted)'}"></i></div>
            <div class="fe-name">${_escHtml(entry.name)}</div>
            <div class="fe-size">${size}</div>
            <div class="fe-date">${date}</div>
            <button onclick="event.stopPropagation();_fbDelete('${_escAttr(entry.path)}','${_escAttr(entry.name)}')" class="fe-del-btn" title="Delete">
                <i data-lucide="trash-2" class="w-3 h-3"></i>
            </button>
        </div>`;
    }).join('');
    lucide.createIcons();
}

// ── Navigate up ──
function _fbNavigateUp() {
    if (_fbPath === '/' || _fbPath === '') return;
    const parts = _fbPath.split('/').filter(Boolean);
    parts.pop();
    _fbLoadDir('/' + parts.join('/'));
}

// ── Open file in editor ──
async function _fbOpenFile(path, name) {
    _fbShowEditor();
    const textarea = document.getElementById('fe-editor-textarea');
    const nameEl = document.getElementById('fe-editor-name');
    const dirty = document.getElementById('fe-editor-dirty');
    if (!textarea) return;

    if (nameEl) nameEl.textContent = name;
    if (dirty) dirty.classList.add('hidden');
    textarea.value = 'Loading...';
    textarea.disabled = true;
    _fbEditorPath = path;
    _fbEditorDirty = false;

    try {
        const r = await fetch(`/api/persona-files/read?persona=${encodeURIComponent(_fbPersona)}&path=${encodeURIComponent(path)}`, {
            headers: _fbHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            textarea.value = d.content || '';
        } else {
            textarea.value = '(Failed to read file)';
        }
    } catch (e) {
        textarea.value = '(Error reading file)';
    }
    textarea.disabled = false;
    textarea.focus();
    _fbUpdateWordCount();
    lucide.createIcons();
}

function _fbCloseEditor() {
    if (_fbEditorDirty && !confirm('Unsaved changes. Discard?')) return;
    _fbEditorDirty = false;
    _fbShowList();
    _fbLoadDir(_fbPath);
}

// ── Save file ──
async function _fbSaveFile() {
    const textarea = document.getElementById('fe-editor-textarea');
    if (!textarea || !_fbEditorPath) return;

    try {
        const r = await fetch('/api/persona-files/write', {
            method: 'POST',
            headers: _fbHeaders(),
            body: JSON.stringify({ persona: _fbPersona, path: _fbEditorPath, content: textarea.value })
        });
        if (r.ok) {
            _fbEditorDirty = false;
            const dirty = document.getElementById('fe-editor-dirty');
            if (dirty) dirty.classList.add('hidden');
            if (typeof showToast === 'function') showToast('Saved', 'success');
        } else {
            if (typeof showToast === 'function') showToast('Save failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Save failed', 'error');
    }
}

// ── Formatting ──
function _fbInsertFormat(prefix, suffix) {
    const ta = document.getElementById('fe-editor-textarea');
    if (!ta) return;
    prefix = prefix.replace(/\\n/g, '\n');
    suffix = suffix.replace(/\\n/g, '\n');
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const selected = ta.value.substring(start, end) || 'text';
    const replacement = prefix + selected + suffix;
    ta.value = ta.value.substring(0, start) + replacement + ta.value.substring(end);
    ta.focus();
    ta.selectionStart = start + prefix.length;
    ta.selectionEnd = start + prefix.length + selected.length;
    _fbEditorDirty = true;
    const dirty = document.getElementById('fe-editor-dirty');
    if (dirty) dirty.classList.remove('hidden');
}
window._fbInsertFormat = _fbInsertFormat;

// ── Preview toggle ──
function _fbTogglePreview() {
    const ta = document.getElementById('fe-editor-textarea');
    const preview = document.getElementById('fe-preview-pane');
    if (!ta || !preview) return;
    const showing = preview.style.display !== 'none';
    if (showing) {
        preview.style.display = 'none';
        ta.style.display = '';
    } else {
        ta.style.display = 'none';
        preview.style.display = '';
        // Render markdown — use marked if available, otherwise basic
        const md = ta.value;
        if (typeof marked !== 'undefined') {
            preview.innerHTML = marked.parse(md);
        } else {
            // Basic markdown rendering
            preview.innerHTML = md
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                .replace(/^### (.+)$/gm, '<h3>$1</h3>')
                .replace(/^## (.+)$/gm, '<h2>$1</h2>')
                .replace(/^# (.+)$/gm, '<h1>$1</h1>')
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/_(.+?)_/g, '<em>$1</em>')
                .replace(/^- (.+)$/gm, '<li>$1</li>')
                .replace(/\n/g, '<br>');
        }
    }
}
window._fbTogglePreview = _fbTogglePreview;

// ── AI Assist ──
function _fbAiAssistMenu(event) {
    // Remove existing menu
    document.querySelectorAll('.fe-ai-menu').forEach(el => el.remove());
    const menu = document.createElement('div');
    menu.className = 'fe-ai-menu';
    const actions = [
        { label: 'Revise & improve', action: 'revise' },
        { label: 'Continue writing', action: 'continue' },
        { label: 'Summarize', action: 'summarize' },
        { label: 'Fix grammar', action: 'grammar' },
        { label: 'Custom instruction...', action: 'custom' },
    ];
    menu.innerHTML = actions.map(a =>
        `<button class="fe-ai-menu-item" onclick="_fbAiAssist('${a.action}')">${a.label}</button>`
    ).join('');
    event.target.closest('.fe-ai-btn')?.appendChild(menu);
    setTimeout(() => document.addEventListener('click', function close() { menu.remove(); document.removeEventListener('click', close); }), 10);
}
window._fbAiAssistMenu = _fbAiAssistMenu;

async function _fbAiAssist(action) {
    const ta = document.getElementById('fe-editor-textarea');
    if (!ta) return;
    const content = ta.value;
    if (!content.trim()) { if (typeof showToast === 'function') showToast('Nothing to process', 'info'); return; }

    let instruction = '';
    if (action === 'custom') {
        instruction = prompt('What should AI do with this text?');
        if (!instruction) return;
    }

    const prompts = {
        revise: `Revise and improve this text. Keep the same format and structure. Return ONLY the improved text:\n\n${content}`,
        continue: `Continue writing from where this text leaves off. Match the style and format. Return ONLY the continuation:\n\n${content}`,
        summarize: `Summarize this text in 2-3 sentences:\n\n${content}`,
        grammar: `Fix any grammar, spelling, or punctuation errors. Return ONLY the corrected text:\n\n${content}`,
        custom: `${instruction}\n\nText:\n${content}`
    };

    if (typeof showToast === 'function') showToast('AI processing...', 'info');
    try {
        const r = await fetch('/api/ask', {
            method: 'POST',
            headers: _fbHeaders(),
            body: JSON.stringify({
                question: prompts[action],
                persona: _fbPersona,
                system_override: 'You are a writing assistant. Follow the instruction exactly. Return only the requested output, no explanations.'
            })
        });
        if (r.ok) {
            const d = await r.json();
            const result = (d.answer || d.response || '').trim();
            if (result) {
                if (action === 'continue') {
                    ta.value = content + '\n\n' + result;
                } else if (action === 'summarize') {
                    if (typeof showToast === 'function') showToast(result, 'info');
                    return;
                } else {
                    // Show confirm before replacing
                    if (confirm('Replace content with AI result?\n\nPreview:\n' + result.substring(0, 200) + '...')) {
                        ta.value = result;
                    }
                }
                _fbEditorDirty = true;
                const dirty = document.getElementById('fe-editor-dirty');
                if (dirty) dirty.classList.remove('hidden');
                _fbUpdateWordCount();
                if (typeof showToast === 'function') showToast('AI assist complete', 'success');
            }
        } else {
            if (typeof showToast === 'function') showToast('AI assist failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('AI assist failed', 'error');
    }
}
window._fbAiAssist = _fbAiAssist;

// ── Word count ──
function _fbUpdateWordCount() {
    const ta = document.getElementById('fe-editor-textarea');
    const el = document.getElementById('fe-word-count');
    if (!ta || !el) return;
    const words = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
    el.textContent = `${words} words`;
}

// ── Create file/folder ──
async function _fbCreateFile() {
    const name = prompt('File name (e.g., notes.md):');
    if (!name || !name.trim()) return;
    const path = _fbPath === '/' ? '/' + name.trim() : _fbPath + '/' + name.trim();
    try {
        await fetch('/api/persona-files/write', {
            method: 'POST', headers: _fbHeaders(),
            body: JSON.stringify({ persona: _fbPersona, path, content: '' })
        });
        _fbLoadDir(_fbPath);
    } catch (e) { if (typeof showToast === 'function') showToast('Failed', 'error'); }
}
window._fbCreateFile = _fbCreateFile;

async function _fbCreateFolder() {
    const name = prompt('Folder name:');
    if (!name || !name.trim()) return;
    const path = _fbPath === '/' ? '/' + name.trim() : _fbPath + '/' + name.trim();
    try {
        await fetch('/api/persona-files/mkdir', {
            method: 'POST', headers: _fbHeaders(),
            body: JSON.stringify({ persona: _fbPersona, path })
        });
        _fbLoadDir(_fbPath);
    } catch (e) { if (typeof showToast === 'function') showToast('Failed', 'error'); }
}
window._fbCreateFolder = _fbCreateFolder;

async function _fbDelete(path, name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
        await fetch(`/api/persona-files?persona=${encodeURIComponent(_fbPersona)}&path=${encodeURIComponent(path)}`, {
            method: 'DELETE', headers: _fbHeaders()
        });
        _fbLoadDir(_fbPath);
        if (typeof showToast === 'function') showToast(`Deleted ${name}`, 'success');
    } catch (e) { if (typeof showToast === 'function') showToast('Failed', 'error'); }
}
window._fbDelete = _fbDelete;

async function _fbUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
        const r = await fetch('/api/files/upload', {
            method: 'POST',
            headers: { 'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '', 'X-Filename': file.name, 'X-Persona': _fbPersona },
            body: file
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Uploaded ${file.name}`, 'success');
            _fbLoadDir(_fbPath);
        } else {
            if (typeof showToast === 'function') showToast('Upload failed', 'error');
        }
    } catch (e) { if (typeof showToast === 'function') showToast('Upload failed', 'error'); }
    event.target.value = '';
}
window._fbUpload = _fbUpload;

// ── Helpers ──
function _escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function _escAttr(s) { return s.replace(/'/g, "\\'").replace(/\\/g, '\\\\'); }
function _formatBytes(b) {
    if (b === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(b) / Math.log(k));
    return parseFloat((b / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
