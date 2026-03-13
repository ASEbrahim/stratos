// ═══════════════════════════════════════════════════════════
// CODEX BROWSER — Searchable codebase glossary modal
// CSS prefix: cx-   (no collisions with other modules)
// ═══════════════════════════════════════════════════════════

let _cxOpen = false;
let _cxData = null;
let _cxView = 'categories';  // 'categories' | 'terms' | 'detail'
let _cxCategory = null;
let _cxTerm = null;

function openCodex() {
    if (_cxOpen) return;
    _cxOpen = true;
    _cxView = 'categories';
    _cxCategory = null;
    _cxTerm = null;

    if (_cxData) {
        _cxRender();
    } else {
        _cxLoadData();
    }
}

function closeCodex() {
    _cxOpen = false;
    const el = document.getElementById('cx-overlay');
    if (el) el.remove();
}

async function _cxLoadData() {
    try {
        const resp = await fetch('/codex.json');
        _cxData = await resp.json();
        _cxRender();
    } catch (e) {
        console.error('Codex load failed:', e);
    }
}

function _cxRender() {
    let overlay = document.getElementById('cx-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'cx-overlay';
        overlay.className = 'cx-overlay';
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) closeCodex();
        });
        document.body.appendChild(overlay);
    }

    const d = _cxData;
    if (!d) return;

    let html = '<div class="cx-modal">';
    // Header
    html += '<div class="cx-header">';
    html += '<div class="cx-header-left">';
    // Breadcrumb: Codex > Category > Term
    html += '<nav class="cx-breadcrumb">';
    if (_cxView === 'categories') {
        html += '<span class="cx-breadcrumb-active">Codex</span>';
    } else {
        html += '<button class="cx-breadcrumb-link" onclick="_cxView=\'categories\';_cxCategory=null;_cxTerm=null;_cxRender()">Codex</button>';
        html += '<span class="cx-breadcrumb-sep">&rsaquo;</span>';
        if (_cxView === 'terms') {
            html += '<span class="cx-breadcrumb-active">' + _esc(_cxCategory) + '</span>';
        } else if (_cxView === 'detail') {
            html += '<button class="cx-breadcrumb-link" onclick="_cxOpenCategory(\'' + _esc(_cxCategory) + '\')">' + _esc(_cxCategory) + '</button>';
            html += '<span class="cx-breadcrumb-sep">&rsaquo;</span>';
            html += '<span class="cx-breadcrumb-active">' + _esc(_cxTerm) + '</span>';
        }
    }
    html += '</nav>';
    html += '<span class="cx-badge">' + d.categories.reduce((s, c) => s + c.terms.length, 0) + ' terms</span>';
    html += '</div>';
    html += '<div class="cx-header-right">';
    html += '<input type="text" class="cx-search" placeholder="Search terms..." oninput="_cxSearch(this.value)" id="cx-search-input">';
    html += '<button class="cx-close" onclick="closeCodex()">&times;</button>';
    html += '</div>';
    html += '</div>';

    // Content
    html += '<div class="cx-content">';

    if (_cxView === 'categories') {
        html += _cxRenderCategories(d);
    } else if (_cxView === 'terms') {
        html += _cxRenderTermList(d);
    } else if (_cxView === 'detail') {
        html += _cxRenderDetail(d);
    }

    html += '</div>';
    html += '</div>';

    overlay.innerHTML = html;

    // Focus search
    const si = document.getElementById('cx-search-input');
    if (si) si.focus();
}

function _cxRenderCategories(d) {
    let html = '<div class="cx-grid">';
    for (const cat of d.categories) {
        html += '<div class="cx-card" onclick="_cxOpenCategory(\'' + _esc(cat.name) + '\')">';
        html += '<div class="cx-card-icon">' + (cat.icon || '') + '</div>';
        html += '<div class="cx-card-name">' + _esc(cat.name) + '</div>';
        html += '<div class="cx-card-count">' + cat.terms.length + ' terms</div>';
        html += '</div>';
    }
    html += '</div>';
    return html;
}

function _cxRenderTermList(d) {
    const cat = d.categories.find(c => c.name === _cxCategory);
    if (!cat) return '<p>Category not found.</p>';

    let html = '<div class="cx-section-header">';
    html += '<span class="cx-section-icon">' + (cat.icon || '') + '</span>';
    html += '<span class="cx-section-name">' + _esc(cat.name) + '</span>';
    html += '</div>';
    html += '<div class="cx-term-list">';
    for (const t of cat.terms) {
        const typeTag = t.type ? '<span class="cx-type-tag">' + _esc(t.type) + '</span>' : '';
        html += '<div class="cx-term-row" onclick="_cxOpenTerm(\'' + _esc(cat.name) + '\',\'' + _esc(t.term) + '\')">';
        html += '<span class="cx-term-name">' + _esc(t.term) + '</span>';
        html += typeTag;
        html += '<span class="cx-term-preview">' + _esc(t.definition.slice(0, 80)) + '...</span>';
        html += '</div>';
    }
    html += '</div>';
    return html;
}

function _cxRenderDetail(d) {
    const cat = d.categories.find(c => c.name === _cxCategory);
    if (!cat) return '';
    const t = cat.terms.find(x => x.term === _cxTerm);
    if (!t) return '<p>Term not found.</p>';

    let html = '<div class="cx-detail">';
    html += '<h2 class="cx-detail-title">' + _esc(t.term) + '</h2>';
    if (t.type) {
        html += '<span class="cx-type-tag cx-type-lg">' + _esc(t.type) + '</span>';
    }
    if (t.added_in) {
        html += '<span class="cx-sprint-tag">' + _esc(t.added_in) + '</span>';
    }
    html += '<div class="cx-detail-def">' + _esc(t.definition) + '</div>';

    if (t.files && t.files.length) {
        html += '<div class="cx-detail-section"><h4>Files</h4><ul>';
        for (const f of t.files) {
            html += '<li><code>' + _esc(f) + '</code></li>';
        }
        html += '</ul></div>';
    }

    if (t.related && t.related.length) {
        html += '<div class="cx-detail-section"><h4>Related</h4><div class="cx-related-chips">';
        for (const r of t.related) {
            html += '<button class="cx-chip" onclick="_cxJumpToTerm(\'' + _esc(r) + '\')">' + _esc(r) + '</button>';
        }
        html += '</div></div>';
    }

    html += '<div class="cx-detail-meta">';
    html += 'Category: <button class="cx-chip" onclick="_cxOpenCategory(\'' + _esc(cat.name) + '\')">' +
            (cat.icon || '') + ' ' + _esc(cat.name) + '</button>';
    html += '</div>';
    html += '</div>';
    return html;
}

function _cxOpenCategory(name) {
    _cxCategory = name;
    _cxView = 'terms';
    _cxRender();
}

function _cxOpenTerm(catName, termName) {
    _cxCategory = catName;
    _cxTerm = termName;
    _cxView = 'detail';
    _cxRender();
}

function _cxJumpToTerm(termName) {
    if (!_cxData) return;
    for (const cat of _cxData.categories) {
        const t = cat.terms.find(x => x.term === termName);
        if (t) {
            _cxOpenTerm(cat.name, t.term);
            return;
        }
    }
}

function _cxBack() {
    if (_cxView === 'detail') {
        _cxView = 'terms';
        _cxTerm = null;
    } else if (_cxView === 'terms') {
        _cxView = 'categories';
        _cxCategory = null;
    }
    _cxRender();
}

function _cxSearch(query) {
    if (!_cxData || !query.trim()) {
        // Reset to current view
        _cxRender();
        // Restore search input value
        const si = document.getElementById('cx-search-input');
        if (si) { si.value = query; si.focus(); }
        return;
    }

    const q = query.toLowerCase();
    const results = [];
    for (const cat of _cxData.categories) {
        for (const t of cat.terms) {
            const score =
                (t.term.toLowerCase().includes(q) ? 10 : 0) +
                (t.definition.toLowerCase().includes(q) ? 3 : 0) +
                ((t.files || []).some(f => f.toLowerCase().includes(q)) ? 2 : 0);
            if (score > 0) {
                results.push({ term: t, cat: cat, score });
            }
        }
    }
    results.sort((a, b) => b.score - a.score);

    const overlay = document.getElementById('cx-overlay');
    if (!overlay) return;

    // Re-render with search results
    const content = overlay.querySelector('.cx-content');
    if (!content) return;

    let html = '<div class="cx-search-results">';
    html += '<div class="cx-section-header"><span class="cx-section-name">' + results.length + ' results for "' + _esc(query) + '"</span></div>';
    html += '<div class="cx-term-list">';
    for (const r of results.slice(0, 50)) {
        const typeTag = r.term.type ? '<span class="cx-type-tag">' + _esc(r.term.type) + '</span>' : '';
        const catTag = '<span class="cx-cat-tag">' + (r.cat.icon || '') + ' ' + _esc(r.cat.name) + '</span>';
        html += '<div class="cx-term-row" onclick="_cxOpenTerm(\'' + _esc(r.cat.name) + '\',\'' + _esc(r.term.term) + '\')">';
        html += '<span class="cx-term-name">' + _esc(r.term.term) + '</span>';
        html += typeTag + catTag;
        html += '<span class="cx-term-preview">' + _esc(r.term.definition.slice(0, 80)) + '...</span>';
        html += '</div>';
    }
    html += '</div></div>';
    content.innerHTML = html;

    // Keep search input focused and valued
    const si = document.getElementById('cx-search-input');
    if (si) { si.value = query; si.focus(); }
}

function _esc(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// Esc to close
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _cxOpen) { closeCodex(); }
});

// Make openCodex globally available
window.openCodex = openCodex;
window.openCodexBrowser = openCodex;
