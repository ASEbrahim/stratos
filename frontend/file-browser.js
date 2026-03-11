// ═══════════════════════════════════════════════════════════
// PERSONA FILE BROWSER — Browse, view, and edit persona files
// ═══════════════════════════════════════════════════════════

let _fbOpen = false;
let _fbPersona = 'intelligence';
let _fbPath = '/';
let _fbEntries = [];

function _fbHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : ''
    };
}

// ── Toggle file browser ──
function toggleFileBrowser(persona) {
    _fbOpen = !_fbOpen;
    let panel = document.getElementById('file-browser-panel');
    if (!panel) {
        _createFileBrowserPanel();
        panel = document.getElementById('file-browser-panel');
    }
    if (_fbOpen) {
        // Close context editor if open
        if (typeof _ctxEditorOpen !== 'undefined' && _ctxEditorOpen) toggleContextEditor();
        _fbPersona = persona || currentPersona || 'intelligence';
        _fbPath = '/';
        panel.classList.remove('hidden');
        panel.classList.add('ctx-slide-in');
        _fbLoadDir(_fbPath);
        // Update persona selector
        const sel = document.getElementById('fb-persona-select');
        if (sel) sel.value = _fbPersona;
    } else {
        panel.classList.add('hidden');
        panel.classList.remove('ctx-slide-in');
    }
}
window.toggleFileBrowser = toggleFileBrowser;

// ── Create DOM ──
function _createFileBrowserPanel() {
    const panel = document.createElement('div');
    panel.id = 'file-browser-panel';
    panel.className = 'hidden';
    panel.style.cssText = 'position:fixed;inset:0;z-index:10000;';
    panel.innerHTML = `
        <div class="ctx-editor-backdrop" onclick="toggleFileBrowser()"></div>
        <div class="ctx-editor-sidebar">
            <div class="ctx-editor-header">
                <div class="flex items-center gap-2">
                    <i data-lucide="folder-open" class="w-4 h-4" style="color:var(--accent)"></i>
                    <span class="text-sm font-bold" style="color:var(--text-heading)">Files</span>
                    <span id="fb-path-display" class="text-[10px] font-mono" style="color:var(--text-muted)">/</span>
                </div>
                <div class="flex items-center gap-2">
                    <select id="fb-persona-select" onchange="_fbSwitchPersona(this.value)" class="text-[10px] font-mono px-2 py-1 rounded cursor-pointer" style="background:var(--bg-panel-solid); color:var(--accent); border:1px solid var(--border-strong); outline:none;"></select>
                    <button onclick="toggleFileBrowser()" class="p-1 rounded-md transition-colors" style="color:var(--text-muted)" onmouseenter="this.style.color='var(--text-heading)'" onmouseleave="this.style.color='var(--text-muted)'" title="Close">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>
            </div>
            <!-- Toolbar -->
            <div class="flex items-center gap-2 px-4 py-2" style="border-bottom:1px solid var(--border-strong);">
                <button onclick="_fbNavigateUp()" class="fb-tool-btn" title="Go up">
                    <i data-lucide="arrow-up" class="w-3 h-3"></i>
                </button>
                <button onclick="_fbLoadDir(_fbPath)" class="fb-tool-btn" title="Refresh">
                    <i data-lucide="refresh-cw" class="w-3 h-3"></i>
                </button>
                <div class="flex-1"></div>
                <button onclick="_fbCreateFile()" class="fb-tool-btn" title="New file">
                    <i data-lucide="file-plus" class="w-3 h-3"></i>
                </button>
                <button onclick="_fbCreateFolder()" class="fb-tool-btn" title="New folder">
                    <i data-lucide="folder-plus" class="w-3 h-3"></i>
                </button>
                <label class="fb-tool-btn cursor-pointer" title="Upload file">
                    <i data-lucide="upload" class="w-3 h-3"></i>
                    <input type="file" class="hidden" onchange="_fbUpload(event)">
                </label>
            </div>
            <!-- File list -->
            <div id="fb-file-list" class="flex-1 overflow-y-auto px-2 py-2 space-y-0.5" style="min-height:0;"></div>
            <!-- File editor (hidden by default) -->
            <div id="fb-editor" class="hidden" style="flex:1;display:flex;flex-direction:column;min-height:0;">
                <div class="flex items-center justify-between px-4 py-2" style="border-top:1px solid var(--border-strong);border-bottom:1px solid var(--border-strong);">
                    <div class="flex items-center gap-2">
                        <button onclick="_fbCloseEditor()" class="fb-tool-btn" title="Back to list">
                            <i data-lucide="arrow-left" class="w-3 h-3"></i>
                        </button>
                        <span id="fb-editor-name" class="text-[11px] font-mono" style="color:var(--text-secondary)"></span>
                    </div>
                    <button onclick="_fbSaveFile()" class="ctx-btn ctx-btn-save text-[10px]">
                        <i data-lucide="save" class="w-3 h-3"></i> Save
                    </button>
                </div>
                <textarea id="fb-editor-textarea" class="ctx-editor-textarea" style="flex:1;border-radius:0;border:none;"></textarea>
            </div>
        </div>`;
    document.body.appendChild(panel);
    lucide.createIcons();
    _fbPopulatePersonaSelect();
}

function _fbPopulatePersonaSelect() {
    const select = document.getElementById('fb-persona-select');
    if (!select) return;
    const personas = availablePersonas.length ? availablePersonas : [
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
    _fbCloseEditor();
    _fbLoadDir('/');
}

// ── Load directory listing ──
async function _fbLoadDir(path) {
    _fbPath = path;
    const pathDisplay = document.getElementById('fb-path-display');
    if (pathDisplay) pathDisplay.textContent = path;

    const list = document.getElementById('fb-file-list');
    if (!list) return;
    list.innerHTML = '<div class="text-[10px] text-center py-8" style="color:var(--text-muted)">Loading...</div>';

    // Hide editor, show list
    const editor = document.getElementById('fb-editor');
    const fileList = document.getElementById('fb-file-list');
    if (editor) editor.classList.add('hidden');
    if (fileList) fileList.style.display = '';

    try {
        const r = await fetch(`/api/persona-files?persona=${encodeURIComponent(_fbPersona)}&path=${encodeURIComponent(path)}`, {
            headers: _fbHeaders()
        });
        if (r.ok) {
            const d = await r.json();
            _fbEntries = d.entries || [];
            _fbRenderEntries();
        } else {
            list.innerHTML = '<div class="text-[10px] text-center py-8" style="color:var(--text-muted)">No files yet</div>';
        }
    } catch (e) {
        list.innerHTML = '<div class="text-[10px] text-center py-8 text-red-400">Failed to load files</div>';
    }
}

function _fbRenderEntries() {
    const list = document.getElementById('fb-file-list');
    if (!list) return;

    if (_fbEntries.length === 0) {
        list.innerHTML = '<div class="text-[10px] text-center py-8" style="color:var(--text-muted)">Empty — create a file or folder</div>';
        return;
    }

    // Sort: directories first, then files alphabetically
    const sorted = [..._fbEntries].sort((a, b) => {
        if (a.type === 'directory' && b.type !== 'directory') return -1;
        if (a.type !== 'directory' && b.type === 'directory') return 1;
        return a.name.localeCompare(b.name);
    });

    list.innerHTML = sorted.map(entry => {
        const isDir = entry.type === 'directory';
        const icon = isDir ? 'folder' : _fbFileIcon(entry.name);
        const size = isDir ? '' : `<span class="text-[9px]" style="color:var(--text-faint)">${_formatBytes(entry.size || 0)}</span>`;
        const clickAction = isDir
            ? `_fbLoadDir('${_escAttr(entry.path)}')`
            : `_fbOpenFile('${_escAttr(entry.path)}', '${_escAttr(entry.name)}')`;

        return `<div class="flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-colors group" style="color:var(--text-secondary)" onmouseenter="this.style.background='var(--bg-hover)'" onmouseleave="this.style.background='transparent'" onclick="${clickAction}">
            <i data-lucide="${icon}" class="w-3.5 h-3.5 flex-shrink-0" style="color:${isDir ? 'var(--accent)' : 'var(--text-muted)'}"></i>
            <span class="text-[11px] flex-1 truncate">${_escHtml(entry.name)}</span>
            ${size}
            <button onclick="event.stopPropagation();_fbDelete('${_escAttr(entry.path)}','${_escAttr(entry.name)}')" class="p-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity" style="color:var(--text-muted)" onmouseenter="this.style.color='#f87171'" onmouseleave="this.style.color='var(--text-muted)'" title="Delete">
                <i data-lucide="trash-2" class="w-3 h-3"></i>
            </button>
        </div>`;
    }).join('');
    lucide.createIcons();
}

function _fbFileIcon(name) {
    const ext = name.split('.').pop().toLowerCase();
    if (['md','txt','text'].includes(ext)) return 'file-text';
    if (['json','yaml','yml'].includes(ext)) return 'file-json';
    if (['png','jpg','jpeg','gif','webp','svg'].includes(ext)) return 'image';
    if (['pdf'].includes(ext)) return 'file';
    return 'file';
}

function _escHtml(s) {
    const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}
function _escAttr(s) {
    return s.replace(/'/g, "\\'").replace(/\\/g, '\\\\');
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
    const editor = document.getElementById('fb-editor');
    const fileList = document.getElementById('fb-file-list');
    const textarea = document.getElementById('fb-editor-textarea');
    const nameEl = document.getElementById('fb-editor-name');
    if (!editor || !textarea) return;

    fileList.style.display = 'none';
    editor.classList.remove('hidden');
    editor.style.display = 'flex';
    if (nameEl) nameEl.textContent = name;
    textarea.value = 'Loading...';
    textarea.disabled = true;
    textarea.dataset.path = path;

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
    lucide.createIcons();
}

function _fbCloseEditor() {
    const editor = document.getElementById('fb-editor');
    const fileList = document.getElementById('fb-file-list');
    if (editor) { editor.classList.add('hidden'); editor.style.display = ''; }
    if (fileList) fileList.style.display = '';
}

// ── Save file ──
async function _fbSaveFile() {
    const textarea = document.getElementById('fb-editor-textarea');
    if (!textarea || !textarea.dataset.path) return;

    try {
        const r = await fetch('/api/persona-files/write', {
            method: 'POST',
            headers: _fbHeaders(),
            body: JSON.stringify({
                persona: _fbPersona,
                path: textarea.dataset.path,
                content: textarea.value
            })
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast('File saved', 'success');
        } else {
            if (typeof showToast === 'function') showToast('Failed to save file', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to save file', 'error');
    }
}

// ── Create new file ──
async function _fbCreateFile() {
    const name = prompt('File name (e.g., notes.md):');
    if (!name || !name.trim()) return;
    const path = _fbPath === '/' ? '/' + name.trim() : _fbPath + '/' + name.trim();
    try {
        await fetch('/api/persona-files/write', {
            method: 'POST',
            headers: _fbHeaders(),
            body: JSON.stringify({ persona: _fbPersona, path, content: '' })
        });
        _fbLoadDir(_fbPath);
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create file', 'error');
    }
}

// ── Create new folder ──
async function _fbCreateFolder() {
    const name = prompt('Folder name:');
    if (!name || !name.trim()) return;
    const path = _fbPath === '/' ? '/' + name.trim() : _fbPath + '/' + name.trim();
    try {
        await fetch('/api/persona-files/mkdir', {
            method: 'POST',
            headers: _fbHeaders(),
            body: JSON.stringify({ persona: _fbPersona, path })
        });
        _fbLoadDir(_fbPath);
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to create folder', 'error');
    }
}

// ── Delete file/folder ──
async function _fbDelete(path, name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
        await fetch(`/api/persona-files?persona=${encodeURIComponent(_fbPersona)}&path=${encodeURIComponent(path)}`, {
            method: 'DELETE',
            headers: _fbHeaders()
        });
        _fbLoadDir(_fbPath);
        if (typeof showToast === 'function') showToast(`Deleted ${name}`, 'success');
    } catch (e) {
        if (typeof showToast === 'function') showToast('Failed to delete', 'error');
    }
}

// ── Upload file ──
async function _fbUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
        const r = await fetch('/api/files/upload', {
            method: 'POST',
            headers: {
                'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
                'X-Filename': file.name,
                'X-Persona': _fbPersona
            },
            body: file
        });
        if (r.ok) {
            if (typeof showToast === 'function') showToast(`Uploaded ${file.name}`, 'success');
            _fbLoadDir(_fbPath);
        } else {
            if (typeof showToast === 'function') showToast('Upload failed', 'error');
        }
    } catch (e) {
        if (typeof showToast === 'function') showToast('Upload failed', 'error');
    }
    event.target.value = '';
}
