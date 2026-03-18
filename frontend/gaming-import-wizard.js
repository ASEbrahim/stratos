// ═══════════════════════════════════════════════════════════
// GAMING IMPORT WIZARD — 5-step world import modal
// ═══════════════════════════════════════════════════════════

(function () {
    'use strict';

    let _wizardStep = 1;
    const _wizardData = {
        worldName: '',
        storyPosition: 'Beginning',
        startingLevel: 1,
        difficulty: 'Normal',
        startingClass: '',
        stats: { STR: 10, DEX: 10, INT: 10 },
        checkboxItems: [],
        canonCharacters: 'canon',
        nameStyle: 'real',
        locations: [],
        loreDepth: 'standard',
    };

    const POPULAR_WORLDS = ['SAO', 'Witcher', 'Naruto', 'Skyrim', 'Dark Souls', 'Elden Ring', 'Pokemon', 'One Piece', 'Zelda', 'Final Fantasy'];
    const STORY_POSITIONS = ['Beginning', 'Middle', 'End'];
    const DIFFICULTIES = ['Casual', 'Normal', 'Hard'];
    const CLASSES = ['Warrior', 'Mage', 'Rogue', 'Ranger', 'Cleric', 'Paladin', 'Bard', 'Monk'];
    const CHECKBOX_ITEMS = ['Companions', 'Pets', 'Housing', 'Mounts', 'Crafting', 'Romance'];
    const LORE_DEPTHS = ['Light', 'Standard', 'Deep'];

    let _stylesInjected = false;

    function _injectStyles() {
        if (_stylesInjected) return;
        _stylesInjected = true;
        const style = document.createElement('style');
        style.textContent = `
            .giw-overlay{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);animation:giw-fadein .2s ease}
            @keyframes giw-fadein{from{opacity:0}to{opacity:1}}
            .giw-modal{background:var(--bg-panel-solid,#1a1a2e);border:1px solid var(--border-strong,#333);border-radius:16px;width:90%;max-width:520px;max-height:85vh;overflow-y:auto;padding:28px 24px;box-shadow:0 20px 60px rgba(0,0,0,0.5);color:var(--text-primary,#e0e0e0)}
            .giw-title{font-size:18px;font-weight:700;color:var(--text-heading,#fff);margin-bottom:4px}
            .giw-subtitle{font-size:11px;color:var(--text-muted,#888);margin-bottom:20px}
            .giw-steps{display:flex;gap:6px;margin-bottom:20px}
            .giw-step-dot{width:32px;height:4px;border-radius:2px;background:var(--border-strong,#333);transition:background .2s}
            .giw-step-dot.active{background:var(--accent,#f472b6)}
            .giw-step-dot.done{background:var(--accent,#f472b6);opacity:.5}
            .giw-label{font-size:12px;font-weight:600;color:var(--text-heading,#fff);margin:14px 0 6px}
            .giw-input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid var(--border-strong,#333);background:rgba(255,255,255,0.04);color:var(--text-primary,#e0e0e0);font-size:13px;outline:none;box-sizing:border-box}
            .giw-input:focus{border-color:var(--accent,#f472b6)}
            .giw-pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
            .giw-pill{padding:6px 14px;border-radius:8px;border:1px solid var(--border-strong,#333);background:transparent;color:var(--text-muted,#888);font-size:11px;cursor:pointer;transition:all .15s}
            .giw-pill:hover{border-color:var(--accent,#f472b6);color:var(--accent,#f472b6)}
            .giw-pill.selected{background:rgba(244,114,182,0.12);border-color:var(--accent,#f472b6);color:var(--accent,#f472b6);font-weight:600}
            .giw-slider-wrap{display:flex;align-items:center;gap:10px;margin-top:4px}
            .giw-slider{flex:1;appearance:none;height:4px;border-radius:2px;background:var(--border-strong,#333);outline:none}
            .giw-slider::-webkit-slider-thumb{appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent,#f472b6);cursor:pointer}
            .giw-slider-val{font-size:13px;font-weight:600;color:var(--accent,#f472b6);min-width:28px;text-align:right}
            .giw-checkbox{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:12px;color:var(--text-muted,#888);padding:4px 0}
            .giw-checkbox input{accent-color:var(--accent,#f472b6)}
            .giw-checkbox.checked{color:var(--accent,#f472b6)}
            .giw-radio-group{display:flex;flex-direction:column;gap:4px;margin-top:4px}
            .giw-radio{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-muted,#888);cursor:pointer;padding:4px 0}
            .giw-radio input{accent-color:var(--accent,#f472b6)}
            .giw-radio.selected{color:var(--accent,#f472b6)}
            .giw-actions{display:flex;justify-content:space-between;margin-top:24px;gap:10px}
            .giw-btn{padding:10px 22px;border-radius:10px;font-size:12px;font-weight:600;cursor:pointer;border:none;transition:all .15s}
            .giw-btn-secondary{background:transparent;border:1px solid var(--border-strong,#333);color:var(--text-muted,#888)}
            .giw-btn-secondary:hover{border-color:var(--accent,#f472b6);color:var(--accent,#f472b6)}
            .giw-btn-primary{background:var(--accent,#f472b6);color:#fff}
            .giw-btn-primary:hover{filter:brightness(1.1)}
            .giw-btn-primary:disabled{opacity:0.4;cursor:not-allowed}
            .giw-summary{font-size:12px;line-height:1.7;color:var(--text-muted,#888)}
            .giw-summary b{color:var(--text-heading,#fff)}
        `;
        document.head.appendChild(style);
    }

    function _pill(label, selected, onclick) {
        return `<button class="giw-pill${selected ? ' selected' : ''}" onclick="${onclick}">${label}</button>`;
    }

    function _slider(key, min, max, val, onchange) {
        return `<div class="giw-slider-wrap">
            <input type="range" class="giw-slider" min="${min}" max="${max}" value="${val}" oninput="${onchange}">
            <span class="giw-slider-val" id="giw-sv-${key}">${val}</span>
        </div>`;
    }

    function _renderStep() {
        const body = document.getElementById('giw-body');
        if (!body) return;

        if (_wizardStep === 1) {
            body.innerHTML = `
                <div class="giw-label">World Name</div>
                <input class="giw-input" id="giw-world-input" placeholder="e.g., Sword Art Online" value="${_escH(_wizardData.worldName)}" oninput="_giwSetWorld(this.value)">
                <div class="giw-label">Popular Worlds</div>
                <div class="giw-pills">
                    ${POPULAR_WORLDS.map(w => _pill(w, _wizardData.worldName === w, `_giwPickWorld('${w}')`)).join('')}
                </div>`;
        } else if (_wizardStep === 2) {
            body.innerHTML = `
                <div class="giw-label">Story Position</div>
                <div class="giw-pills">
                    ${STORY_POSITIONS.map(p => _pill(p, _wizardData.storyPosition === p, `_giwSet('storyPosition','${p}')`)).join('')}
                </div>
                <div class="giw-label">Starting Level</div>
                ${_slider('level', 1, 100, _wizardData.startingLevel, "_giwSetLevel(this.value)")}
                <div class="giw-label">Difficulty</div>
                <div class="giw-pills">
                    ${DIFFICULTIES.map(d => _pill(d, _wizardData.difficulty === d, `_giwSet('difficulty','${d}')`)).join('')}
                </div>`;
        } else if (_wizardStep === 3) {
            body.innerHTML = `
                <div class="giw-label">Starting Class</div>
                <div class="giw-pills">
                    ${CLASSES.map(c => _pill(c, _wizardData.startingClass === c, `_giwSet('startingClass','${c}')`)).join('')}
                </div>
                <div class="giw-label">STR</div>
                ${_slider('str', 1, 20, _wizardData.stats.STR, "_giwSetStat('STR',this.value)")}
                <div class="giw-label">DEX</div>
                ${_slider('dex', 1, 20, _wizardData.stats.DEX, "_giwSetStat('DEX',this.value)")}
                <div class="giw-label">INT</div>
                ${_slider('int', 1, 20, _wizardData.stats.INT, "_giwSetStat('INT',this.value)")}
                <div class="giw-label">Extras</div>
                ${CHECKBOX_ITEMS.map(item => {
                    const checked = _wizardData.checkboxItems.includes(item);
                    return `<label class="giw-checkbox${checked ? ' checked' : ''}">
                        <input type="checkbox" ${checked ? 'checked' : ''} onchange="_giwToggleItem('${item}',this.checked)"> ${item}
                    </label>`;
                }).join('')}`;
        } else if (_wizardStep === 4) {
            body.innerHTML = `
                <div class="giw-label">Character Source</div>
                <div class="giw-radio-group">
                    ${['canon', 'generated'].map(v => `<label class="giw-radio${_wizardData.canonCharacters === v ? ' selected' : ''}">
                        <input type="radio" name="giw-chars" value="${v}" ${_wizardData.canonCharacters === v ? 'checked' : ''} onchange="_giwSet('canonCharacters','${v}')"> ${v === 'canon' ? 'Canon Characters' : 'AI-Generated Characters'}
                    </label>`).join('')}
                </div>
                <div class="giw-label">Name Style</div>
                <div class="giw-radio-group">
                    ${['real', 'changed'].map(v => `<label class="giw-radio${_wizardData.nameStyle === v ? ' selected' : ''}">
                        <input type="radio" name="giw-names" value="${v}" ${_wizardData.nameStyle === v ? 'checked' : ''} onchange="_giwSet('nameStyle','${v}')"> ${v === 'real' ? 'Real Names' : 'Changed Names'}
                    </label>`).join('')}
                </div>
                <div class="giw-label">Lore Depth</div>
                <div class="giw-pills">
                    ${LORE_DEPTHS.map(d => _pill(d, _wizardData.loreDepth === d.toLowerCase(), `_giwSet('loreDepth','${d.toLowerCase()}')`)).join('')}
                </div>`;
        } else if (_wizardStep === 5) {
            const extras = _wizardData.checkboxItems.length ? _wizardData.checkboxItems.join(', ') : 'None';
            body.innerHTML = `
                <div class="giw-summary">
                    <b>World:</b> ${_escH(_wizardData.worldName || 'Custom')}<br>
                    <b>Story Position:</b> ${_wizardData.storyPosition} &middot; <b>Level:</b> ${_wizardData.startingLevel} &middot; <b>Difficulty:</b> ${_wizardData.difficulty}<br>
                    <b>Class:</b> ${_wizardData.startingClass || 'None'} &middot; <b>STR:</b> ${_wizardData.stats.STR} / <b>DEX:</b> ${_wizardData.stats.DEX} / <b>INT:</b> ${_wizardData.stats.INT}<br>
                    <b>Extras:</b> ${_escH(extras)}<br>
                    <b>Characters:</b> ${_wizardData.canonCharacters === 'canon' ? 'Canon' : 'Generated'} &middot; <b>Names:</b> ${_wizardData.nameStyle === 'real' ? 'Real' : 'Changed'} &middot; <b>Lore:</b> ${_wizardData.loreDepth}<br>
                </div>`;
        }

        // Update step dots
        document.querySelectorAll('.giw-step-dot').forEach((dot, i) => {
            dot.classList.toggle('active', i + 1 === _wizardStep);
            dot.classList.toggle('done', i + 1 < _wizardStep);
        });

        // Update button labels
        const nextBtn = document.getElementById('giw-next-btn');
        if (nextBtn) {
            nextBtn.textContent = _wizardStep === 5 ? 'Generate World' : 'Next';
            nextBtn.disabled = _wizardStep === 1 && !_wizardData.worldName.trim();
        }
        const backBtn = document.getElementById('giw-back-btn');
        if (backBtn) backBtn.style.visibility = _wizardStep === 1 ? 'hidden' : 'visible';
    }

    function _escH(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function openImportWorldWizard() {
        _injectStyles();
        _wizardStep = 1;
        _wizardData.worldName = '';
        _wizardData.storyPosition = 'Beginning';
        _wizardData.startingLevel = 1;
        _wizardData.difficulty = 'Normal';
        _wizardData.startingClass = '';
        _wizardData.stats = { STR: 10, DEX: 10, INT: 10 };
        _wizardData.checkboxItems = [];
        _wizardData.canonCharacters = 'canon';
        _wizardData.nameStyle = 'real';
        _wizardData.loreDepth = 'standard';

        const overlay = document.createElement('div');
        overlay.className = 'giw-overlay';
        overlay.id = 'giw-overlay';
        overlay.onclick = function (e) { if (e.target === overlay) closeImportWorldWizard(); };

        const stepLabels = ['World', 'Conditions', 'Character', 'Customization', 'Confirm'];
        overlay.innerHTML = `
            <div class="giw-modal">
                <div class="giw-title">Import Game World</div>
                <div class="giw-subtitle">Step <span id="giw-step-num">1</span> of 5 &mdash; <span id="giw-step-label">${stepLabels[0]}</span></div>
                <div class="giw-steps">${[1,2,3,4,5].map(() => '<div class="giw-step-dot"></div>').join('')}</div>
                <div id="giw-body"></div>
                <div class="giw-actions">
                    <button class="giw-btn giw-btn-secondary" id="giw-back-btn" onclick="_giwBack()" style="visibility:hidden">Back</button>
                    <div style="display:flex;gap:8px">
                        <button class="giw-btn giw-btn-secondary" onclick="closeImportWorldWizard()">Cancel</button>
                        <button class="giw-btn giw-btn-primary" id="giw-next-btn" onclick="_giwNext()">Next</button>
                    </div>
                </div>
            </div>`;

        document.body.appendChild(overlay);
        _renderStep();
    }

    function closeImportWorldWizard() {
        const overlay = document.getElementById('giw-overlay');
        if (overlay) overlay.remove();
    }

    // ── Wizard navigation ──
    window._giwNext = function () {
        if (_wizardStep < 5) {
            _wizardStep++;
            _updateStepHeader();
            _renderStep();
        } else {
            _submitWizard();
        }
    };

    window._giwBack = function () {
        if (_wizardStep > 1) {
            _wizardStep--;
            _updateStepHeader();
            _renderStep();
        }
    };

    function _updateStepHeader() {
        const labels = ['World', 'Conditions', 'Character', 'Customization', 'Confirm'];
        const numEl = document.getElementById('giw-step-num');
        const labelEl = document.getElementById('giw-step-label');
        if (numEl) numEl.textContent = _wizardStep;
        if (labelEl) labelEl.textContent = labels[_wizardStep - 1];
    }

    // ── Data setters ──
    window._giwSetWorld = function (v) {
        _wizardData.worldName = v;
        const btn = document.getElementById('giw-next-btn');
        if (btn) btn.disabled = !v.trim();
    };

    window._giwPickWorld = function (w) {
        _wizardData.worldName = w;
        const input = document.getElementById('giw-world-input');
        if (input) input.value = w;
        _renderStep();
    };

    window._giwSet = function (key, val) {
        _wizardData[key] = val;
        _renderStep();
    };

    window._giwSetLevel = function (val) {
        _wizardData.startingLevel = parseInt(val, 10);
        const sv = document.getElementById('giw-sv-level');
        if (sv) sv.textContent = val;
    };

    window._giwSetStat = function (stat, val) {
        _wizardData.stats[stat] = parseInt(val, 10);
        const sv = document.getElementById('giw-sv-' + stat.toLowerCase());
        if (sv) sv.textContent = val;
    };

    window._giwToggleItem = function (item, checked) {
        if (checked && !_wizardData.checkboxItems.includes(item)) {
            _wizardData.checkboxItems.push(item);
        } else if (!checked) {
            _wizardData.checkboxItems = _wizardData.checkboxItems.filter(i => i !== item);
        }
        _renderStep();
    };

    // ── Submit ──
    async function _submitWizard() {
        const btn = document.getElementById('giw-next-btn');
        if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }

        const payload = {
            name: _wizardData.worldName.trim() || 'Custom_World',
            genre: 'Imported',
            description: [
                `World: ${_wizardData.worldName}`,
                `Story position: ${_wizardData.storyPosition}`,
                `Starting level: ${_wizardData.startingLevel}`,
                `Difficulty: ${_wizardData.difficulty}`,
                _wizardData.startingClass ? `Class: ${_wizardData.startingClass}` : '',
                `Stats: STR ${_wizardData.stats.STR}, DEX ${_wizardData.stats.DEX}, INT ${_wizardData.stats.INT}`,
                _wizardData.checkboxItems.length ? `Features: ${_wizardData.checkboxItems.join(', ')}` : '',
                `Characters: ${_wizardData.canonCharacters}`,
                `Names: ${_wizardData.nameStyle}`,
                `Lore depth: ${_wizardData.loreDepth}`,
            ].filter(Boolean).join('\n'),
            wizard_config: {
                story_position: _wizardData.storyPosition,
                starting_level: _wizardData.startingLevel,
                difficulty: _wizardData.difficulty,
                starting_class: _wizardData.startingClass,
                stats: _wizardData.stats,
                extras: _wizardData.checkboxItems,
                canon_characters: _wizardData.canonCharacters === 'canon',
                real_names: _wizardData.nameStyle === 'real',
                lore_depth: _wizardData.loreDepth,
            },
        };

        try {
            const headers = {
                'Content-Type': 'application/json',
                'X-Auth-Token': typeof getAuthToken === 'function' ? getAuthToken() : '',
            };
            const r = await fetch('/api/scenarios/create', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(payload),
            });
            const d = await r.json().catch(() => ({}));
            if (!r.ok) {
                if (typeof showToast === 'function') showToast(d.error || 'Failed to create world', 'error');
                if (btn) { btn.disabled = false; btn.textContent = 'Generate World'; }
                return;
            }
            if (typeof showToast === 'function') showToast(`World "${d.name || payload.name}" created!`, 'success');
            closeImportWorldWizard();

            // Trigger scenario reload and generation polling
            if (typeof _loadScenarios === 'function') _loadScenarios();
            if (d.status === 'generating' && typeof _pollGenerationStatus === 'function') {
                _pollGenerationStatus(d.name || payload.name);
            }
        } catch (e) {
            if (typeof showToast === 'function') showToast('Failed to create world', 'error');
            if (btn) { btn.disabled = false; btn.textContent = 'Generate World'; }
        }
    }

    // ── Exports ──
    window.openImportWorldWizard = openImportWorldWizard;
    window.closeImportWorldWizard = closeImportWorldWizard;
})();
