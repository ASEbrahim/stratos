'use strict';
// settings-categories.js — Dynamic category rendering and manipulation
// Depends on: simpleCategories, CATEGORY_ICONS, esc(), showToast(), rebuildNavFromConfig(), saveSimpleState(), configData, renderCategoryPresets(), lucide

// === DYNAMIC CATEGORY RENDERING ===
function renderDynamicCategories() {
    const container = document.getElementById('simple-categories-container');
    if (!container) return;

    if (simpleCategories.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-slate-500 border border-dashed border-slate-700 rounded-xl">
                <i data-lucide="sparkles" class="w-8 h-8 mx-auto mb-2 opacity-50"></i>
                <p class="text-sm">Enter your role above and click <strong>Generate</strong></p>
                <p class="text-xs mt-1">AI will create tracking categories tailored to you</p>
            </div>`;
        lucide.createIcons();
        renderAdvancedCategories();
        if (typeof renderCategoryPresets === 'function') renderCategoryPresets();
        return;
    }

    container.innerHTML = simpleCategories.map((cat, ci) => {
        const isEnabled = cat.enabled !== false;  // default true
        // Normalize: handle old {name,keywords} and new {label,items} formats
        const catLabel = cat.label || cat.name || 'Untitled';
        const catItems = cat.items || cat.keywords || [];
        return `
        <div class="glass-panel rounded-xl p-4 relative group/cat ${isEnabled ? '' : 'opacity-40'}" style="transition: opacity 0.2s">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                    <i data-lucide="${CATEGORY_ICONS[cat.icon] || 'tag'}" class="w-4 h-4" style="color:var(--accent)"></i>
                    <span class="text-sm font-bold text-slate-200">${esc(catLabel)}</span>
                    <span class="text-[10px] text-slate-600">${catItems.length}</span>
                    <button onclick="saveSingleCategory(${ci})" class="opacity-0 group-hover/cat:opacity-100 text-slate-500 hover:text-amber-400 transition-all p-1 rounded hover:bg-amber-900/20" title="Save to library">
                        <i data-lucide="bookmark" class="w-3.5 h-3.5"></i>
                    </button>
                    <button onclick="removeCategory(${ci})" class="opacity-0 group-hover/cat:opacity-100 text-slate-500 hover:text-red-400 transition-all p-1 rounded hover:bg-red-900/20" title="Delete category">
                        <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                    </button>
                </div>
                <div class="flex items-center gap-2">
                    <label class="relative inline-flex items-center cursor-pointer" title="${isEnabled ? 'Disable category (won\'t be scraped)' : 'Enable category'}">
                        <input type="checkbox" class="sr-only peer" ${isEnabled ? 'checked' : ''}
                               onchange="toggleCategory(${ci}, this.checked)">
                        <div class="w-8 h-4 bg-slate-700 rounded-full peer peer-checked:bg-emerald-600
                             after:content-[''] after:absolute after:top-[2px] after:left-[2px]
                             after:bg-slate-300 after:rounded-full after:h-3 after:w-3
                             after:transition-all peer-checked:after:translate-x-4 peer-checked:after:bg-white"></div>
                    </label>
                </div>
            </div>
            <div class="flex flex-wrap gap-1.5" id="cat-bubbles-${ci}">
                ${catItems.map(item => `
                    <span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium cursor-pointer transition-all hover:opacity-75"
                         style="background:var(--accent-bg); border:1px solid var(--accent-border); color:var(--accent-light)"
                         onclick="removeCategoryItem(${ci}, '${esc(item).replace(/'/g, "\\'")}')">${esc(item)} <i data-lucide="x" class="w-2.5 h-2.5 opacity-40 hover:opacity-100"></i></span>
                `).join('')}
            </div>
            <input type="text" class="mt-2 w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-[11px] text-slate-300 focus:border-emerald-500/50 focus:outline-none placeholder-slate-600"
                   placeholder="+ Add to ${esc(catLabel)}..."
                   onkeydown="if(event.key==='Enter'){addCategoryItem(${ci}, this.value); this.value='';}">
        </div>`;
    }).join('');

    lucide.createIcons();
    renderAdvancedCategories();
    if (typeof renderCategoryPresets === 'function') renderCategoryPresets();
}

// Also render in Advanced panel
function renderAdvancedCategories() {
    const container = document.getElementById('adv-categories-list');
    const countEl = document.getElementById('adv-cat-count');
    if (!container) return;

    if (countEl) countEl.textContent = simpleCategories.length ? `${simpleCategories.length}` : '';

    if (simpleCategories.length === 0) {
        container.innerHTML = `<p class="text-xs text-slate-500 italic">No categories yet — use Generate in Simple mode or add manually below.</p>`;
        return;
    }

    // Human-readable labels for scorer types
    const SCORER_LABELS = { career: 'Job postings', banks: 'Bank offers', tech: 'Tech & science', regional: 'Regional GCC', auto: 'Auto-detect' };
    const ROOT_LABELS = { kuwait: 'Kuwait Radar', regional: 'Regional News', global: 'Global News', ai: 'AI & Tech' };

    container.innerHTML = simpleCategories.map((cat, ci) => {
        const isEnabled = cat.enabled !== false;
        const catLabel = cat.label || cat.name || 'Untitled';
        const catItems = cat.items || cat.keywords || [];
        const scorerType = cat.scorer_type || 'auto';
        const root = cat.root || 'kuwait';
        const iconName = CATEGORY_ICONS[cat.icon] || cat.icon || 'tag';

        return `
        <div class="rounded-lg ${isEnabled ? '' : 'opacity-40'}" style="background: var(--bg-panel-solid); border: 1px solid var(--border-strong); transition: opacity 0.2s">
            <div class="flex items-center justify-between px-3 py-2" style="border-bottom: 1px solid var(--border-strong)">
                <div class="flex items-center gap-2 min-w-0">
                    <i data-lucide="${iconName}" class="w-3.5 h-3.5 flex-shrink-0" style="color:var(--accent)"></i>
                    <span class="text-xs font-bold text-slate-200 truncate">${esc(catLabel)}</span>
                    <span class="text-[9px] text-slate-600">${catItems.length} items</span>
                </div>
                <div class="flex items-center gap-1.5 flex-shrink-0">
                    <label class="relative inline-flex items-center cursor-pointer" title="${isEnabled ? 'Disable' : 'Enable'}">
                        <input type="checkbox" class="sr-only peer" ${isEnabled ? 'checked' : ''}
                               onchange="toggleCategory(${ci}, this.checked)">
                        <div class="w-7 h-3.5 bg-slate-700 rounded-full peer peer-checked:bg-emerald-600
                             after:content-[''] after:absolute after:top-[2px] after:left-[2px]
                             after:bg-slate-300 after:rounded-full after:h-2.5 after:w-2.5
                             after:transition-all peer-checked:after:translate-x-3.5 peer-checked:after:bg-white"></div>
                    </label>
                    <button onclick="removeCategory(${ci})" class="text-red-400/40 hover:text-red-400 transition-all p-0.5 rounded hover:bg-red-900/20" title="Delete">
                        <i data-lucide="trash-2" class="w-3 h-3"></i>
                    </button>
                </div>
            </div>
            <div class="px-3 py-2">
                <div class="flex flex-wrap gap-1 mb-1.5">
                    ${catItems.map(item => `
                        <span class="inline-flex items-center gap-0.5 px-2 py-0.5 rounded-full text-[10px] font-medium cursor-pointer transition-all hover:opacity-70"
                             style="background:var(--accent-bg); border:1px solid var(--accent-border); color:var(--accent-light)"
                             onclick="removeCategoryItem(${ci}, '${esc(item).replace(/'/g, "\\\\'")}')" title="Click to remove">${esc(item)} ×</span>
                    `).join('')}
                </div>
                <input type="text" class="w-full bg-transparent border-0 border-b border-slate-700/30 px-0 py-1 text-[10px] text-slate-400 focus:border-emerald-500/50 focus:outline-none placeholder-slate-600"
                       placeholder="+ Add item..."
                       onkeydown="if(event.key==='Enter'){addCategoryItem(${ci}, this.value); this.value='';}">
            </div>
            <div class="px-3 py-1.5 text-[9px] text-slate-600" style="border-top: 1px solid var(--border-strong)">
                <span>Scored as: <strong class="text-slate-500">${SCORER_LABELS[scorerType] || scorerType}</strong></span>
            </div>
        </div>`;
    }).join('');

    lucide.createIcons();
}

function toggleAdvAddCategory() {
    const form = document.getElementById('adv-add-cat-form');
    const toggle = document.getElementById('adv-add-cat-toggle');
    if (!form) return;
    const showing = !form.classList.contains('hidden');
    form.classList.toggle('hidden');
    if (toggle) toggle.classList.toggle('hidden');
    if (!showing) {
        document.getElementById('adv-new-cat-label')?.focus();
    }
}

function addAdvancedCategory() {
    const labelEl = document.getElementById('adv-new-cat-label');
    const scorerEl = document.getElementById('adv-new-cat-scorer');
    const rootEl = document.getElementById('adv-new-cat-root');
    const iconEl = document.getElementById('adv-new-cat-icon');
    const itemsEl = document.getElementById('adv-new-cat-items');

    const label = labelEl?.value.trim();
    if (!label) {
        if (typeof showToast === 'function') showToast('Enter a category label', 'warning');
        return;
    }

    const items = (itemsEl?.value || '')
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0);

    if (items.length === 0) {
        if (typeof showToast === 'function') showToast('Add at least one item', 'warning');
        return;
    }

    // Generate a short ID from label
    const id = label.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 12);

    // Check for duplicate ID
    if (simpleCategories.some(c => (c.id || '').toLowerCase() === id)) {
        if (typeof showToast === 'function') showToast('A category with a similar name already exists', 'warning');
        return;
    }

    const newCat = {
        id: id,
        label: label,
        icon: iconEl?.value || 'globe',
        items: items,
        enabled: true,
        scorer_type: scorerEl?.value || 'auto',
        root: rootEl?.value || 'kuwait',
        pinned: true,  // Manual categories survive Generate
    };

    simpleCategories.push(newCat);
    _syncCategoriesState();
    renderAdvancedCategories();

    // Clear inputs and collapse form
    if (labelEl) labelEl.value = '';
    if (itemsEl) itemsEl.value = '';
    toggleAdvAddCategory();

    window._settingsDirty = true;
    if (typeof showToast === 'function') showToast(`Added "${label}"`, 'success');
}

// === CATEGORY MANIPULATION ===
function _syncCategoriesState() {
    // Keep configData and nav in sync with simpleCategories
    saveSimpleState();
    if (typeof configData !== 'undefined') {
        configData.dynamic_categories = JSON.parse(JSON.stringify(simpleCategories));
    }
    renderDynamicCategories();
    if (typeof rebuildNavFromConfig === 'function') {
        rebuildNavFromConfig();
    }
    window._settingsDirty = true;
}

function toggleCategory(catIdx, enabled) {
    if (catIdx >= simpleCategories.length) return;
    simpleCategories[catIdx].enabled = enabled;
    _syncCategoriesState();
}

function addCategoryItem(catIdx, value) {
    if (!value || !value.trim()) return;
    value = value.trim();
    if (catIdx >= simpleCategories.length) return;
    if (simpleCategories[catIdx].items.includes(value)) return;
    simpleCategories[catIdx].items.push(value);
    _syncCategoriesState();
}

function removeCategoryItem(catIdx, value) {
    if (catIdx >= simpleCategories.length) return;
    simpleCategories[catIdx].items = simpleCategories[catIdx].items.filter(v => v !== value);
    _syncCategoriesState();
}

function removeCategory(catIdx) {
    simpleCategories.splice(catIdx, 1);
    _syncCategoriesState();
}
