// ═══════════════════════════════════════════════════════════
// FILE EXPLORER — Nautilus-style modal file browser + editor
// ═══════════════════════════════════════════════════════════

let _fbOpen = false;
let _fbPersona = 'intelligence';
let _fbPath = '/';
let _fbEntries = [];
let _fbEditorDirty = false;
let _fbEditorPath = '';
let _fbInteracting = false; // drag/resize in progress — suppress backdrop close

const _FB_ICONS = {
    folder: 'folder', md: 'file-text', txt: 'file-text', text: 'file-text',
    json: 'file-json', yaml: 'file-json', yml: 'file-json', csv: 'table',
    pdf: 'file', png: 'image', jpg: 'image', jpeg: 'image', gif: 'image',
    webp: 'image', svg: 'image', wav: 'volume-2', mp3: 'volume-2', default: 'file'
};

const _FB_PERSONAS = [
    { name: 'intelligence', icon: '🔍', label: 'Intelligence' },
    { name: 'market', icon: '📊', label: 'Market' },
    { name: 'scholarly', icon: '📚', label: 'Scholarly' },
    { name: 'gaming', icon: '🎮', label: 'Gaming' },
    { name: 'anime', icon: '🎌', label: 'Anime' },
    { name: 'tcg', icon: '🃏', label: 'TCG' },
];

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
    if (_fbOpen) { _fbClose(); return; }
    _fbOpen = true;
    window._fbOpen = true;
    _fbPersona = persona || (typeof currentPersona !== 'undefined' ? currentPersona : 'intelligence');
    _fbPath = '/';
    window._fbPath = '/';
    _fbEditorDirty = false;
    _fbEditorPath = '';

    let modal = document.getElementById('file-explorer-modal');
    if (!modal) {
        _fbCreateModal();
        modal = document.getElementById('file-explorer-modal');
    }
    modal.style.display = 'flex';
    _fbUpdateSidebar();
    _fbLoadDir('/');
    _fbShowList();
}
window.toggleFileBrowser = toggleFileBrowser;

async function _fbClose() {
    if (_fbEditorDirty && !(await stratosConfirm('Unsaved changes will be lost. Close anyway?', { title: 'Unsaved Changes', okText: 'Discard', cancelText: 'Keep Editing' }))) return;
    _fbOpen = false;
    window._fbOpen = false;
    _fbEditorDirty = false;
    _fbStopStars();
    const modal = document.getElementById('file-explorer-modal');
    if (modal) modal.style.display = 'none';
}

// ── Create Modal DOM ──
function _fbCreateModal() {
    const modal = document.createElement('div');
    modal.id = 'file-explorer-modal';
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.6);display:none;align-items:center;justify-content:center;';
    modal.onclick = (e) => { if (e.target === modal && !_fbInteracting) _fbClose(); };

    modal.innerHTML = `
        <div class="fe-container" onclick="event.stopPropagation()">
            <canvas id="fe-stars-canvas" class="fe-stars-canvas"></canvas>
            <!-- Title bar (Nautilus style) -->
            <div class="fe-titlebar" id="fe-titlebar">
                <div class="fe-titlebar-left">
                    <i data-lucide="search" class="w-4 h-4" style="color:var(--text-muted)"></i>
                    <span class="fe-titlebar-title">Files</span>
                </div>
                <div class="fe-titlebar-center">
                    <button onclick="_fbNavigateUp()" class="fe-nav-btn" data-tip="Navigate up" data-tip-pos="bottom">
                        <i data-lucide="chevron-left" class="w-4 h-4"></i>
                    </button>
                    <button class="fe-nav-btn" disabled style="opacity:0.3" data-tip="Forward" data-tip-pos="bottom">
                        <i data-lucide="chevron-right" class="w-4 h-4"></i>
                    </button>
                    <div id="fe-breadcrumb" class="fe-breadcrumb-bar"></div>
                </div>
                <div class="fe-titlebar-right">
                    <div class="fe-atmos-group" id="fe-atmos-group"></div>
                    <button onclick="_fbClose()" class="fe-wnd-btn fe-wnd-close" data-tip="Close file browser" data-tip-pos="left">
                        <i data-lucide="x" class="w-3.5 h-3.5"></i>
                    </button>
                </div>
            </div>

            <!-- Body: sidebar + main -->
            <div class="fe-body">
                <!-- Left sidebar -->
                <div class="fe-sidebar" id="fe-sidebar"></div>

                <!-- Main content area -->
                <div class="fe-main">
                    <!-- Column header -->
                    <div class="fe-col-header">
                        <span class="fe-col-name">Name</span>
                        <span class="fe-col-size">Size</span>
                        <span class="fe-col-date">Modified</span>
                        <span class="fe-col-actions"></span>
                    </div>

                    <!-- File list view -->
                    <div id="fe-list-view" style="flex:1;display:flex;flex-direction:column;min-height:0;">
                        <div id="fe-file-list" class="flex-1 overflow-y-auto" style="min-height:0;"></div>
                        <div class="fe-action-bar">
                            <button onclick="_fbCreateFile()" class="fe-action-btn" data-tip="Create a new text file" data-tip-pos="bottom"><i data-lucide="file-plus" class="w-4 h-4"></i> New File</button>
                            <button onclick="_fbCreateFolder()" class="fe-action-btn" data-tip="Create a new folder" data-tip-pos="bottom"><i data-lucide="folder-plus" class="w-4 h-4"></i> New Folder</button>
                            <label class="fe-action-btn cursor-pointer" data-tip="Upload a file from your device" data-tip-pos="bottom"><i data-lucide="upload" class="w-4 h-4"></i> Upload<input type="file" class="hidden" onchange="_fbUpload(event)"></label>
                            <div class="flex-1"></div>
                            <button onclick="_fbLoadDir(_fbPath)" class="fe-nav-btn" data-tip="Refresh file list" data-tip-pos="bottom"><i data-lucide="refresh-cw" class="w-4 h-4"></i></button>
                        </div>
                    </div>

                    <!-- Editor view (hidden by default) -->
                    <div id="fe-editor-view" style="flex:1;flex-direction:column;min-height:0;display:none;">
                        <div class="fe-editor-toolbar">
                            ${_FB_TOOLBAR.map(t => `<button onclick="_fbInsertFormat('${t.prefix.replace(/\n/g,'\\n')}','${t.suffix.replace(/\n/g,'\\n')}')" class="fe-fmt-btn" data-tip="${t.key}"><i data-lucide="${t.icon}" class="w-3.5 h-3.5"></i></button>`).join('')}
                            <div class="fe-toolbar-divider"></div>
                            <button onclick="_fbTogglePreview()" class="fe-fmt-btn" id="fe-preview-toggle-btn" data-tip="Toggle preview (Ctrl+Shift+P)"><i data-lucide="eye" class="w-3.5 h-3.5"></i></button>
                            <div class="flex-1"></div>
                            <span id="fe-editor-dirty" class="fe-dirty-badge" style="display:none">●</span>
                            <button onclick="_fbAiAssistMenu(event)" class="fe-ai-btn" data-tip="AI writing assistant"><i data-lucide="sparkles" class="w-3.5 h-3.5"></i> AI</button>
                        </div>
                        <textarea id="fe-editor-textarea" class="fe-textarea" spellcheck="false" placeholder="Start typing..."></textarea>
                        <div id="fe-preview-pane" style="display:none" class="fe-preview"></div>
                        <div class="fe-action-bar">
                            <button onclick="_fbSaveFile()" class="fe-save-btn" data-tip="Save file (Ctrl+S)"><i data-lucide="save" class="w-3.5 h-3.5"></i> Save</button>
                            <button onclick="_fbTogglePreview()" class="fe-action-btn" data-tip="Toggle markdown preview"><i data-lucide="eye" class="w-3.5 h-3.5"></i> Preview</button>
                            <div class="flex-1"></div>
                            <span id="fe-word-count" class="text-[11px]" style="color:var(--text-faint)"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;

    document.body.appendChild(modal);
    _fbInitDrag(modal);
    _fbInitResize(modal);
    lucide.createIcons();

    // Keyboard shortcuts
    const textarea = document.getElementById('fe-editor-textarea');
    if (textarea) {
        textarea.addEventListener('input', () => {
            _fbEditorDirty = true;
            const dirty = document.getElementById('fe-editor-dirty');
            if (dirty) dirty.style.display = '';
            _fbUpdateWordCount();
        });
        textarea.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's') { e.preventDefault(); _fbSaveFile(); }
            if (e.ctrlKey && e.shiftKey && e.key === 'P') { e.preventDefault(); _fbTogglePreview(); }
        });
    }
}

// ── Drag to move ──
function _fbInitDrag(modal) {
    const titlebar = modal.querySelector('#fe-titlebar');
    const container = modal.querySelector('.fe-container');
    if (!titlebar || !container) return;
    let dragging = false, startX, startY, origLeft, origTop;

    titlebar.addEventListener('mousedown', (e) => {
        if (e.target.closest('button')) return;
        dragging = true;
        _fbInteracting = true;
        const rect = container.getBoundingClientRect();
        startX = e.clientX; startY = e.clientY;
        origLeft = rect.left; origTop = rect.top;
        container.style.position = 'fixed';
        container.style.margin = '0';
        container.style.left = origLeft + 'px';
        container.style.top = origTop + 'px';
        e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        container.style.left = (origLeft + e.clientX - startX) + 'px';
        container.style.top = (origTop + e.clientY - startY) + 'px';
    });
    document.addEventListener('mouseup', () => { dragging = false; setTimeout(() => { _fbInteracting = false; }, 0); });
}

// ── Resize from edges ──
function _fbInitResize(modal) {
    const container = modal.querySelector('.fe-container');
    if (!container) return;
    const handle = document.createElement('div');
    handle.className = 'fe-resize-handle';
    container.appendChild(handle);

    let resizing = false, startX, startY, startW, startH;
    handle.addEventListener('mousedown', (e) => {
        resizing = true;
        _fbInteracting = true;
        startX = e.clientX; startY = e.clientY;
        startW = container.offsetWidth; startH = container.offsetHeight;
        e.preventDefault(); e.stopPropagation();
    });
    document.addEventListener('mousemove', (e) => {
        if (!resizing) return;
        const newW = Math.max(500, startW + (e.clientX - startX));
        const newH = Math.max(300, startH + (e.clientY - startY));
        container.style.width = newW + 'px';
        container.style.height = newH + 'px';
    });
    document.addEventListener('mouseup', () => { resizing = false; setTimeout(() => { _fbInteracting = false; }, 0); });
}

// ── Sidebar ──
function _fbUpdateSidebar() {
    const sidebar = document.getElementById('fe-sidebar');
    if (!sidebar) return;
    const personas = typeof availablePersonas !== 'undefined' && availablePersonas.length
        ? availablePersonas.map(p => _FB_PERSONAS.find(fp => fp.name === p.name) || { name: p.name, icon: '📁', label: p.name.charAt(0).toUpperCase() + p.name.slice(1) })
        : _FB_PERSONAS;

    sidebar.innerHTML = `
        <div class="fe-sidebar-section">
            ${personas.map(p => `
                <button onclick="_fbSwitchPersona('${p.name}')" class="fe-sidebar-item${p.name === _fbPersona ? ' fe-sidebar-active' : ''}" data-tip="Browse ${p.label} files" data-tip-pos="bottom">
                    <span class="fe-sidebar-icon">${p.icon}</span>
                    <span>${p.label}</span>
                </button>
            `).join('')}
        </div>
    `;
    // Render atmosphere buttons in titlebar
    _fbRenderAtmosButtons();
}

function _fbSwitchPersona(name) {
    _fbPersona = name;
    _fbPath = '/';
    window._fbPath = '/';
    _fbUpdateSidebar();
    _fbShowList();
    _fbLoadDir('/');
}

// ── View switching ──
function _fbShowList() {
    const listView = document.getElementById('fe-list-view');
    const editorView = document.getElementById('fe-editor-view');
    const colHeader = document.querySelector('.fe-col-header');
    if (listView) listView.style.display = 'flex';
    if (editorView) editorView.style.display = 'none';
    if (colHeader) colHeader.style.display = 'flex';
    // Restore titlebar to explorer mode
    _fbSetTitlebarMode('explorer');
    _fbRenderBreadcrumb();
}

function _fbShowEditor() {
    const listView = document.getElementById('fe-list-view');
    const editorView = document.getElementById('fe-editor-view');
    const colHeader = document.querySelector('.fe-col-header');
    if (listView) listView.style.display = 'none';
    if (editorView) editorView.style.display = 'flex';
    if (colHeader) colHeader.style.display = 'none';
    const ta = document.getElementById('fe-editor-textarea');
    const prev = document.getElementById('fe-preview-pane');
    if (ta) { ta.style.display = ''; }
    if (prev) { prev.style.display = 'none'; }
    // Switch titlebar to editor mode (back button + file breadcrumb)
    _fbSetTitlebarMode('editor');
}

function _fbSetTitlebarMode(mode) {
    const left = document.querySelector('.fe-titlebar-left');
    if (!left) return;
    if (mode === 'editor') {
        left.innerHTML = `
            <button onclick="_fbCloseEditor()" class="fe-nav-btn" data-tip="Back to file list" data-tip-pos="bottom">
                <i data-lucide="arrow-left" class="w-4 h-4"></i>
            </button>
            <span class="fe-titlebar-title">Edit</span>`;
    } else {
        left.innerHTML = `
            <i data-lucide="search" class="w-4 h-4" style="color:var(--text-muted)"></i>
            <span class="fe-titlebar-title">Files</span>`;
    }
    lucide.createIcons();
}

function _fbRenderEditorBreadcrumb(filePath) {
    const el = document.getElementById('fe-breadcrumb');
    if (!el) return;
    const p = _FB_PERSONAS.find(p => p.name === _fbPersona);
    const icon = p ? p.icon : '📁';
    const label = p ? p.label : _fbPersona;
    // Build breadcrumb: Persona / path / to / file.md
    const parts = filePath.replace(/^\//, '').split('/').filter(Boolean);
    let html = `<button onclick="_fbCloseEditor()" class="fe-crumb"><span class="fe-crumb-icon">${icon}</span> ${label}</button>`;
    for (let i = 0; i < parts.length; i++) {
        const isLast = i === parts.length - 1;
        html += `<span class="fe-crumb-sep">/</span>`;
        if (isLast) {
            html += `<span class="fe-crumb fe-crumb-file"><b>${_escHtml(parts[i])}</b></span>`;
        } else {
            const pathSoFar = '/' + parts.slice(0, i + 1).join('/');
            html += `<button onclick="_fbCloseEditor();_fbLoadDir('${_escAttr(pathSoFar)}')" class="fe-crumb"><b>${_escHtml(parts[i])}</b></button>`;
        }
    }
    el.innerHTML = html;
}

// ── Breadcrumb ──
function _fbRenderBreadcrumb() {
    const el = document.getElementById('fe-breadcrumb');
    if (!el) return;
    const p = _FB_PERSONAS.find(p => p.name === _fbPersona);
    const icon = p ? p.icon : '📁';
    const label = p ? p.label : _fbPersona;
    const parts = _fbPath.split('/').filter(Boolean);

    let html = `<button onclick="_fbLoadDir('/')" class="fe-crumb"><span class="fe-crumb-icon">${icon}</span> ${label}</button>`;
    let accumulated = '';
    for (const part of parts) {
        accumulated += '/' + part;
        const pathStr = accumulated;
        html += `<span class="fe-crumb-sep">/</span>`;
        html += `<button onclick="_fbLoadDir('${_escAttr(pathStr)}')" class="fe-crumb"><b>${_escHtml(part)}</b></button>`;
    }
    el.innerHTML = html;
}

// ── Load directory ──
async function _fbLoadDir(path) {
    _fbPath = path;
    window._fbPath = path;
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

        return `<div class="fe-row" onclick="${clickAction}" data-tip="${isDir ? 'Open folder' : 'Edit file'}" data-tip-pos="bottom">
            <div class="fe-icon"><i data-lucide="${icon}" class="w-5 h-5" style="color:${isDir ? 'var(--accent)' : 'var(--text-muted)'}"></i></div>
            <div class="fe-name">${_escHtml(entry.name)}</div>
            <div class="fe-size">${size}</div>
            <div class="fe-date">${date}</div>
            <button onclick="event.stopPropagation();_fbDelete('${_escAttr(entry.path)}','${_escAttr(entry.name)}')" class="fe-del-btn" data-tip="Delete ${_escHtml(entry.name)}">
                <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
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
    const dirty = document.getElementById('fe-editor-dirty');
    if (!textarea) return;

    // Update breadcrumb to show file path
    _fbRenderEditorBreadcrumb(path);
    if (dirty) dirty.style.display = 'none';
    const aiBadge = document.getElementById('fe-ai-badge');
    if (aiBadge) aiBadge.remove();
    _fbAiUndoContent = null;
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

async function _fbCloseEditor() {
    if (_fbEditorDirty && !(await stratosConfirm('Unsaved changes will be lost.', { title: 'Discard Changes?', okText: 'Discard', cancelText: 'Keep Editing' }))) return;
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
            method: 'POST', headers: _fbHeaders(),
            body: JSON.stringify({ persona: _fbPersona, path: _fbEditorPath, content: textarea.value })
        });
        if (r.ok) {
            _fbEditorDirty = false;
            const dirty = document.getElementById('fe-editor-dirty');
            if (dirty) dirty.style.display = 'none';
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
    ta.focus();
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const selected = ta.value.substring(start, end) || 'text';
    const replacement = prefix + selected + suffix;
    // Use execCommand to preserve undo stack
    ta.setSelectionRange(start, end);
    document.execCommand('insertText', false, replacement);
    // Position cursor around the selected text
    ta.selectionStart = start + prefix.length;
    ta.selectionEnd = start + prefix.length + selected.length;
    _fbEditorDirty = true;
    const dirty = document.getElementById('fe-editor-dirty');
    if (dirty) dirty.style.display = '';
}

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
        const md = ta.value;
        if (typeof marked !== 'undefined') {
            preview.innerHTML = marked.parse(md);
        } else {
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

// ── AI Assist ──
function _fbAiAssistMenu(event) {
    event.stopPropagation();
    const existing = document.querySelector('.fe-ai-menu');
    if (existing) { existing.remove(); return; }
    const btn = event.target.closest('.fe-ai-btn');
    if (!btn) return;
    const menu = document.createElement('div');
    menu.className = 'fe-ai-menu';
    const actions = [
        { label: 'Revise & improve', action: 'revise' },
        { label: 'Continue writing', action: 'continue' },
        { label: 'Summarize', action: 'summarize' },
        { label: 'Fix grammar', action: 'grammar' },
        { label: 'Custom instruction...', action: 'custom' },
    ];
    actions.forEach(a => {
        const item = document.createElement('button');
        item.className = 'fe-ai-menu-item';
        item.textContent = a.label;
        item.addEventListener('click', (e) => { e.stopPropagation(); menu.remove(); _fbAiAssist(a.action); });
        menu.appendChild(item);
    });
    document.body.appendChild(menu);
    const rect = btn.getBoundingClientRect();
    menu.style.top = (rect.bottom + 4) + 'px';
    menu.style.left = Math.max(8, rect.right - menu.offsetWidth) + 'px';
    function dismiss(e) { if (!menu.contains(e.target) && e.target !== btn) { menu.remove(); document.removeEventListener('click', dismiss); } }
    setTimeout(() => document.addEventListener('click', dismiss), 0);
}

async function _fbAiAssist(action) {
    const ta = document.getElementById('fe-editor-textarea');
    if (!ta) return;
    const content = ta.value;
    if (!content.trim()) { if (typeof showToast === 'function') showToast('Nothing to process — file is empty', 'info'); return; }

    let instruction = '';
    if (action === 'custom') {
        instruction = await stratosPrompt({ title: 'Custom AI Instruction', label: 'What should AI do with this file?', placeholder: 'e.g., Rewrite in a formal tone' });
        if (!instruction) return;
    }

    const fileName = _fbEditorPath ? _fbEditorPath.split('/').pop() : 'file';

    // Show loading overlay on the editor
    const editorView = document.getElementById('fe-editor-view');
    let overlay = null;
    if (editorView) {
        overlay = document.createElement('div');
        overlay.className = 'fe-ai-loading';
        overlay.innerHTML = '<div class="fe-ai-spinner"></div><div class="fe-ai-loading-text">AI processing ' + fileName + '...</div>';
        editorView.style.position = 'relative';
        editorView.appendChild(overlay);
    }
    ta.disabled = true;

    try {
        const r = await fetch('/api/file-assist', {
            method: 'POST', headers: _fbHeaders(),
            body: JSON.stringify({ action, content, filename: fileName, instruction })
        });
        if (r.ok) {
            const d = await r.json();
            const result = (d.result || '').trim();
            if (result) {
                if (action === 'continue') {
                    // Insert AI continuation with a visible marker
                    _fbAiUndoContent = content;
                    const marker = '\n\n── AI generated ──\n';
                    ta.value = content + marker + result;
                    _fbEditorDirty = true;
                    _fbShowAiBadge();
                } else if (action === 'summarize') {
                    if (typeof showToast === 'function') showToast(result, 'info', 8000);
                } else {
                    // Show preview — let user accept or reject
                    if (await stratosConfirm(result.substring(0, 400) + (result.length > 400 ? '...' : ''), { title: 'Replace with AI result?', okText: 'Apply', cancelText: 'Keep Original' })) {
                        _fbAiUndoContent = content;
                        ta.value = result;
                        _fbEditorDirty = true;
                        _fbShowAiBadge();
                    }
                }
                if (_fbEditorDirty) {
                    const dirty = document.getElementById('fe-editor-dirty');
                    if (dirty) dirty.style.display = '';
                    _fbUpdateWordCount();
                }
                if (typeof showToast === 'function') showToast('AI assist complete', 'success');
            } else {
                if (typeof showToast === 'function') showToast('AI returned empty result', 'warning');
            }
        } else {
            const err = await r.json().catch(() => ({}));
            if (typeof showToast === 'function') showToast(err.error || 'AI assist failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('AI assist failed: ' + e.message, 'error');
    } finally {
        ta.disabled = false;
        if (overlay) overlay.remove();
    }
}

let _fbAiUndoContent = null;  // stores content before AI edit

function _fbShowAiBadge() {
    let existing = document.getElementById('fe-ai-badge');
    if (existing) existing.remove();
    const statusBar = document.querySelector('#fe-editor-view .fe-action-bar');
    if (!statusBar) return;
    const badge = document.createElement('span');
    badge.id = 'fe-ai-badge';
    badge.className = 'fe-ai-badge';
    badge.innerHTML = '✦ AI edited';
    // Add undo button
    const undoBtn = document.createElement('button');
    undoBtn.className = 'fe-ai-undo';
    undoBtn.textContent = 'Undo';
    undoBtn.onclick = _fbAiUndo;
    badge.appendChild(undoBtn);
    statusBar.prepend(badge);
}

function _fbAiUndo() {
    if (_fbAiUndoContent === null) return;
    const ta = document.getElementById('fe-editor-textarea');
    if (!ta) return;
    ta.value = _fbAiUndoContent;
    _fbAiUndoContent = null;
    _fbEditorDirty = true;
    const dirty = document.getElementById('fe-editor-dirty');
    if (dirty) dirty.style.display = '';
    const badge = document.getElementById('fe-ai-badge');
    if (badge) badge.remove();
    _fbUpdateWordCount();
    if (typeof showToast === 'function') showToast('AI changes reverted', 'info');
}
window._fbAiUndo = _fbAiUndo;

// ── Word count ──
function _fbUpdateWordCount() {
    const ta = document.getElementById('fe-editor-textarea');
    const el = document.getElementById('fe-word-count');
    if (!ta || !el) return;
    el.textContent = `${ta.value.length} chars`;
}

// ── Create file/folder ──
async function _fbCreateFile() {
    const name = await stratosPrompt({ title: 'New File', label: 'File name', placeholder: 'e.g., notes.md' });
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

async function _fbCreateFolder() {
    const name = await stratosPrompt({ title: 'New Folder', label: 'Folder name', placeholder: 'e.g., characters' });
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

async function _fbDelete(path, name) {
    if (!(await stratosConfirm(`Delete "${name}"? This cannot be undone.`, { title: 'Delete File', okText: 'Delete', cancelText: 'Cancel' }))) return;
    try {
        const r = await fetch(`/api/persona-files?persona=${encodeURIComponent(_fbPersona)}&path=${encodeURIComponent(path)}`, {
            method: 'DELETE', headers: _fbHeaders()
        });
        const d = await r.json().catch(() => ({}));
        if (r.ok && d.ok) {
            _fbLoadDir(_fbPath);
            if (typeof showToast === 'function') showToast(`Deleted ${name}`, 'success');
        } else {
            if (typeof showToast === 'function') showToast(d.error || 'Delete failed', 'error');
        }
    } catch (e) { if (typeof showToast === 'function') showToast('Failed', 'error'); }
}

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

// ── Atmosphere system ──
function _fbRenderAtmosButtons() {
    const group = document.getElementById('fe-atmos-group');
    if (!group) return;
    const curAtmos = localStorage.getItem('fe-atmosphere') || 'clean';
    group.innerHTML = [
        { id: 'arcane', label: '✦ Arcane', tip: 'Starfield with accent glow' },
        { id: 'clean', label: '○ Clean', tip: 'Frosted glass transparency' },
        { id: 'deep', label: '● Deep', tip: 'Pure black minimal' },
    ].map(a => `<button onclick="_fbSetAtmosphere('${a.id}')" class="fe-atmos-btn${curAtmos === a.id ? ' active' : ''}" data-atmos="${a.id}" data-tip="${a.tip}">${a.label}</button>`).join('');
    _fbApplyAtmosphere(curAtmos);
}

function _fbSetAtmosphere(atmos) {
    _fbApplyAtmosphere(atmos);
    try { localStorage.setItem('fe-atmosphere', atmos); } catch(e) {}
    document.querySelectorAll('.fe-atmos-btn').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-atmos') === atmos);
    });
}

function _fbApplyAtmosphere(atmos) {
    const container = document.querySelector('.fe-container');
    if (!container) return;
    if (atmos) {
        container.setAttribute('data-fe-atmos', atmos);
    } else {
        container.removeAttribute('data-fe-atmos');
    }
    if (atmos === 'arcane') {
        _fbStartStars();
    } else {
        _fbStopStars();
    }
}

// ── Star parallax engine (matches wizard Arcane) ──
let _fbStarEngine = null;

function _fbStopStars() {
    if (!_fbStarEngine) return;
    cancelAnimationFrame(_fbStarEngine.raf);
    if (_fbStarEngine.onResize) window.removeEventListener('resize', _fbStarEngine.onResize);
    _fbStarEngine = null;
    const c = document.getElementById('fe-stars-canvas');
    if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); }
}

function _fbStartStars() {
    _fbStopStars();
    const canvas = document.getElementById('fe-stars-canvas');
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    }
    resize();

    const isMobile = window.innerWidth <= 768;
    const COUNT = isMobile ? 30 : 120;
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#34d399';

    const temp = document.createElement('div');
    temp.style.color = accent;
    document.body.appendChild(temp);
    const rgb = getComputedStyle(temp).color.match(/\d+/g) || [52, 211, 153];
    temp.remove();
    const ar = +rgb[0], ag = +rgb[1], ab = +rgb[2];

    const stars = [];
    for (let i = 0; i < COUNT; i++) {
        stars.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.4 + 0.3,
            a: Math.random() * 0.4 + 0.08,
            speed: Math.random() * 0.08 + 0.02,
            phase: Math.random() * Math.PI * 2,
            cr: Math.random() < 0.3 ? ar : 255,
            cg: Math.random() < 0.3 ? ag : 255,
            cb: Math.random() < 0.3 ? ab : 255
        });
    }

    const shooters = [];
    let lastShoot = Date.now();

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const t = Date.now() * 0.001;

        for (const s of stars) {
            s.y -= s.speed;
            if (s.y < -2) { s.y = canvas.height + 2; s.x = Math.random() * canvas.width; }
            const flicker = 0.6 + 0.4 * Math.sin(t * 2 + s.phase);
            const alpha = s.a * flicker;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${alpha})`;
            ctx.fill();
            if (s.r > 1) {
                ctx.beginPath();
                ctx.arc(s.x, s.y, s.r * 3, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${s.cr},${s.cg},${s.cb},${alpha * 0.15})`;
                ctx.fill();
            }
        }

        // Constellation lines
        for (let i = 0; i < stars.length; i++) {
            for (let j = i + 1; j < stars.length; j++) {
                const dx = stars[i].x - stars[j].x;
                const dy = stars[i].y - stars[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(stars[i].x, stars[i].y);
                    ctx.lineTo(stars[j].x, stars[j].y);
                    ctx.strokeStyle = `rgba(${ar},${ag},${ab},${0.06 * (1 - dist / 100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        // Shooting stars
        if (Date.now() - lastShoot > 5000 + Math.random() * 4000) {
            lastShoot = Date.now();
            const angle = Math.random() * 0.5 + 0.2;
            const spd = Math.random() * 5 + 3;
            shooters.push({
                x: Math.random() * canvas.width * 0.6,
                y: Math.random() * canvas.height * 0.3,
                vx: Math.cos(angle) * spd, vy: Math.sin(angle) * spd,
                life: 1, len: Math.random() * 35 + 20
            });
        }
        for (let i = shooters.length - 1; i >= 0; i--) {
            const sh = shooters[i];
            sh.x += sh.vx; sh.y += sh.vy; sh.life -= 0.015;
            if (sh.life <= 0) { shooters.splice(i, 1); continue; }
            const grad = ctx.createLinearGradient(sh.x, sh.y, sh.x - sh.vx * sh.len / 5, sh.y - sh.vy * sh.len / 5);
            grad.addColorStop(0, `rgba(255,255,255,${sh.life * 0.7})`);
            grad.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.beginPath();
            ctx.moveTo(sh.x, sh.y);
            ctx.lineTo(sh.x - sh.vx * sh.len / 5, sh.y - sh.vy * sh.len / 5);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        _fbStarEngine.raf = requestAnimationFrame(draw);
    }

    _fbStarEngine = { raf: requestAnimationFrame(draw) };
    const onResize = () => { if (_fbStarEngine) resize(); };
    window.addEventListener('resize', onResize);
    _fbStarEngine.onResize = onResize;
}

// ── Tooltip engine — appends to body so it's always on top ──
let _fbTipEl = null;
let _fbTipTarget = null;

function _fbInitTooltips() {
    if (_fbTipEl) return;
    _fbTipEl = document.createElement('div');
    _fbTipEl.className = 'fe-tooltip';
    document.body.appendChild(_fbTipEl);

    document.addEventListener('mouseover', (e) => {
        const target = e.target.closest('[data-tip]');
        // Only handle tips inside the file explorer
        if (!target || !target.closest('#file-explorer-modal')) {
            if (_fbTipTarget) { _fbTipEl.classList.remove('visible'); _fbTipTarget = null; }
            return;
        }
        if (target === _fbTipTarget) return;
        _fbTipTarget = target;
        _fbTipEl.textContent = target.getAttribute('data-tip');
        const rect = target.getBoundingClientRect();
        // Position below the element, centered
        const tipW = _fbTipEl.offsetWidth || 100;
        let left = rect.left + rect.width / 2 - tipW / 2;
        let top = rect.bottom + 6;
        // If tip goes off right edge
        if (left + tipW > window.innerWidth - 8) left = window.innerWidth - tipW - 8;
        if (left < 8) left = 8;
        // If tip goes below viewport, show above instead
        if (top + 30 > window.innerHeight) {
            top = rect.top - 30;
        }
        _fbTipEl.style.left = left + 'px';
        _fbTipEl.style.top = top + 'px';
        // Re-measure after content set
        requestAnimationFrame(() => {
            const actualW = _fbTipEl.offsetWidth;
            let adjLeft = rect.left + rect.width / 2 - actualW / 2;
            if (adjLeft + actualW > window.innerWidth - 8) adjLeft = window.innerWidth - actualW - 8;
            if (adjLeft < 8) adjLeft = 8;
            _fbTipEl.style.left = adjLeft + 'px';
            _fbTipEl.classList.add('visible');
        });
    });

    document.addEventListener('mouseout', (e) => {
        const target = e.target.closest('[data-tip]');
        if (target && target === _fbTipTarget) {
            // Check if we're leaving to a child — if so, don't hide
            const related = e.relatedTarget;
            if (related && target.contains(related)) return;
            _fbTipEl.classList.remove('visible');
            _fbTipTarget = null;
        } else if (!target && _fbTipTarget) {
            _fbTipEl.classList.remove('visible');
            _fbTipTarget = null;
        }
    });
}

// Init tooltips on first load
_fbInitTooltips();

// ── Expose all functions to window for onclick handlers + cross-module access ──
window._fbOpen = _fbOpen;
window._fbPath = _fbPath;
window._fbClose = _fbClose;
window._fbNavigateUp = _fbNavigateUp;
window._fbCloseEditor = _fbCloseEditor;
window._fbSwitchPersona = _fbSwitchPersona;
window._fbLoadDir = _fbLoadDir;
window._fbOpenFile = _fbOpenFile;
window._fbSaveFile = _fbSaveFile;
window._fbDelete = _fbDelete;
window._fbCreateFile = _fbCreateFile;
window._fbCreateFolder = _fbCreateFolder;
window._fbUpload = _fbUpload;
window._fbInsertFormat = _fbInsertFormat;
window._fbTogglePreview = _fbTogglePreview;
window._fbAiAssistMenu = _fbAiAssistMenu;
window._fbAiAssist = _fbAiAssist;
window._fbSetAtmosphere = _fbSetAtmosphere;
window._fbRenderAtmosButtons = _fbRenderAtmosButtons;
