/**
 * wizard.js — StratOS Onboarding Wizard (dashboard-integrated)
 * Extracted from wizard_v4.html with theme-adaptive CSS custom properties.
 * Stages: 1-2 = CSS + modal shell, 3-7 = AI + settings (added incrementally)
 */
(function() {
'use strict';

/* ═══════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════ */

const CK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
const STEP_NAMES = ['Priorities','Details','Your Feed'];

/* ═══════════════════════════════════════
   DATA  (loaded from wizard-data.js)
   ═══════════════════════════════════════ */

if (!window._wizData) { console.error('wizard-data.js not loaded — wizard disabled'); return; }
const { CATS, INTEREST_SUGGESTIONS, DEF_CATS, DEF_SUBS, PANELS, WIZ_TICKER_MAP,
        ROLE_KEYWORDS, DOMAIN_PILLS, DOMAIN_CAT_MAP, DOMAIN_SUB_MAP } = window._wizData;

/* PANELS, WIZ_TICKER_MAP, ROLE_KEYWORDS, DOMAIN_PILLS — now in wizard-data.js */

/* ═══════════════════════════════════════
   ROLE CLASSIFICATION & ADAPTIVE PILLS
   ═══════════════════════════════════════ */

function classifyRole(role) {
  if (!role) return 'default';
  const r = role.toLowerCase();
  let best = 'default', bestScore = 0;
  for (const [domain, keywords] of Object.entries(ROLE_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (new RegExp(kw, 'i').test(r)) score++;
    }
    if (score > bestScore) { bestScore = score; best = domain; }
  }
  return best;
}

function inferStage(role) {
  const r = role.toLowerCase();
  if (/student|undergrad|freshman|sophomore|junior(?! dev)|senior year|phd candidate|master.*student/.test(r)) return 'Student';
  if (/fresh.*grad|new grad|entry.?level|junior|associate|trainee|apprentice|graduate/.test(r)) return 'Fresh Graduate';
  if (/senior|lead|principal|staff|director|vp|vice.?president|chief|head of|c-level|executive|partner|managing/.test(r)) return 'Senior';
  return 'Mid-Career';
}

let _currentDomain = 'default';

function getEffectivePills(qId, originalPills) {
  const override = DOMAIN_PILLS[qId]?.[_currentDomain];
  return override || DOMAIN_PILLS[qId]?.['default'] || originalPills;
}

function getAllPills(qId, originalPills) {
  // Merge domain pills + default pills + original pills (deduped, for "View all")
  const seen = new Set();
  const result = [];
  const domain = DOMAIN_PILLS[qId]?.[_currentDomain];
  const dflt = DOMAIN_PILLS[qId]?.['default'];
  for (const src of [domain, dflt, originalPills]) {
    if (!src) continue;
    for (const p of src) { if (!seen.has(p)) { seen.add(p); result.push(p); } }
  }
  return result;
}

function getEffectiveDefs(qId, originalDefs, type) {
  // For single-select, keep original default. For multi-select, clear defaults so user/AI chooses.
  if (type === 's') return originalDefs;
  const override = DOMAIN_PILLS[qId]?.[_currentDomain];
  if (override) return []; // When pills change, don't auto-select stale defaults
  return originalDefs;
}

// Deterministic question IDs — changing these should invalidate suggestion cache
const DETERMINISTIC_QS = new Set(['stage','opptype','itype','clevel','rtype','sdepth','cdepth','eregion']);
// Panels that share the 'stage' question — changing in one propagates to all
const STAGE_SHARED_PANELS = ['career_opps', 'jobhunt'];

/* ═══════════════════════════════════════
   STATE
   ═══════════════════════════════════════ */

let step = 0;
let selCats, selSubs, customSubs, interestTopics;
let panelSel, panelCustom, activeTab;
let rvItems, rvCollapsed, discoverAdded;
let _wizMode = null;       // 'suggest' | 'generate'
let _wizGenerateData = null; // response from /api/generate-profile (Stage 5)
let _tabSuggestCache = {};  // cache per tab: {tabId: {suggestions:[], loading:false}}
let _tabSuggestHistory = {};  // rotation history per tab: {tabId: [[batch1], [batch2], [batch3]]}
let _rvItemsCache = null;   // AI-generated role-aware items for Step 3: {sections:{}, discover:[]}
let _rvLoading = false;     // true while fetching rv items from backend
let _viewAllPills = new Set();      // Issue 4: qIds where "View all" is active
let _collapsedSections = new Set(); // Issue 6: collapsed section IDs in Step 2
let _s2CollapseAll = false;         // Issue 6: collapse-all toggle state
let _suggestDebounceTimer = null;   // Step 3: debounce timer for suggestion refresh
let _suggestAbortCtrl = null;       // Step 3: AbortController for in-flight suggest requests
let _s2BannerDismissed = false;     // Step 3: banner dismissed flag

function _wizKey() {
  const p = (typeof getActiveProfile === 'function') ? getActiveProfile() : '';
  return p ? `stratos_wizard_state_${p}` : 'stratos_wizard_state';
}

function _wizSaveState() {
  try {
    // Serialize Sets to arrays for JSON storage
    const serSubs = {};
    for (const [k, v] of Object.entries(selSubs)) serSubs[k] = [...v];
    const serPanelSel = {};
    for (const [sid, qs] of Object.entries(panelSel)) {
      serPanelSel[sid] = {};
      for (const [qid, set] of Object.entries(qs)) serPanelSel[sid][qid] = [...set];
    }
    localStorage.setItem(_wizKey(), JSON.stringify({
      selCats: [...selCats],
      selSubs: serSubs,
      customSubs,
      interestTopics,
      panelSel: serPanelSel,
      panelCustom,
      ts: Date.now()
    }));
  } catch(e) { /* storage full or unavailable */ }
}

function _wizLoadState() {
  try {
    const raw = localStorage.getItem(_wizKey());
    if (!raw) return false;
    const saved = JSON.parse(raw);
    // Expire after 24 hours
    if (Date.now() - (saved.ts || 0) > 86400000) { localStorage.removeItem(_wizKey()); return false; }
    if (!saved.selCats?.length) return false;

    selCats = new Set(saved.selCats);
    for (const [k, v] of Object.entries(saved.selSubs || {})) {
      if (selSubs.hasOwnProperty(k)) selSubs[k] = new Set(v);
    }
    if (saved.customSubs) {
      for (const [k, v] of Object.entries(saved.customSubs)) {
        if (customSubs.hasOwnProperty(k)) customSubs[k] = v;
      }
    }
    if (saved.interestTopics) interestTopics = saved.interestTopics;
    if (saved.panelSel) {
      for (const [sid, qs] of Object.entries(saved.panelSel)) {
        if (!panelSel[sid]) continue;
        for (const [qid, arr] of Object.entries(qs)) {
          panelSel[sid][qid] = new Set(arr);
        }
      }
    }
    if (saved.panelCustom) {
      for (const [sid, qs] of Object.entries(saved.panelCustom)) {
        if (!panelCustom[sid]) continue;
        for (const [qid, arr] of Object.entries(qs)) {
          panelCustom[sid][qid] = arr;
        }
      }
    }
    return true;
  } catch(e) { return false; }
}

function _wizClearState() {
  try { localStorage.removeItem(_wizKey()); } catch(e) {}
}

function clearAll() {
  initState();
  _wizClearState();
  renderAll();
}

/* ═══════════════════════════════════════
   PROFILE PRESETS
   ═══════════════════════════════════════ */

function _presetsKey() {
  const p = (typeof getActiveProfile === 'function') ? getActiveProfile() : '';
  return p ? `stratos_wizard_presets_${p}` : 'stratos_wizard_presets';
}
let _presetMenuOpen = false;

function _getPresets() {
  try {
    const raw = localStorage.getItem(_presetsKey());
    return raw ? JSON.parse(raw) : {};
  } catch(e) { return {}; }
}

function _setPresets(presets) {
  try {
    localStorage.setItem(_presetsKey(), JSON.stringify(presets));
  } catch(e) {
    if (typeof showToast === 'function') showToast('Storage full \u2014 delete some presets to save new ones', 'warning');
  }
}

function _serializeWizardState() {
  const serSubs = {};
  for (const [k, v] of Object.entries(selSubs)) serSubs[k] = [...v];
  const serPanelSel = {};
  for (const [sid, qs] of Object.entries(panelSel)) {
    serPanelSel[sid] = {};
    for (const [qid, set] of Object.entries(qs)) serPanelSel[sid][qid] = [...set];
  }
  return {
    selCats: [...selCats],
    selSubs: serSubs,
    customSubs: JSON.parse(JSON.stringify(customSubs)),
    interestTopics: [...interestTopics],
    panelSel: serPanelSel,
    panelCustom: JSON.parse(JSON.stringify(panelCustom)),
    rvItems: JSON.parse(JSON.stringify(rvItems || {})),
    rvItemsCache: _rvItemsCache ? JSON.parse(JSON.stringify(_rvItemsCache)) : null,
  };
}

function _loadPresetState(preset) {
  // Fully replace all wizard state from preset — no blending
  selCats = new Set(preset.selCats || []);
  for (const c of CATS) {
    selSubs[c.id] = new Set(preset.selSubs?.[c.id] || []);
    customSubs[c.id] = preset.customSubs?.[c.id] || [];
  }
  interestTopics = preset.interestTopics ? [...preset.interestTopics] : [];
  // Restore panelSel: re-init defaults first, then overlay preset
  for (const [sid, cfg] of Object.entries(PANELS)) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of cfg.qs) panelCustom[sid][q.id] = [];
  }
  if (preset.panelSel) {
    for (const [sid, qs] of Object.entries(preset.panelSel)) {
      if (!panelSel[sid]) panelSel[sid] = {};
      for (const [qid, arr] of Object.entries(qs)) panelSel[sid][qid] = new Set(arr);
    }
  }
  if (preset.panelCustom) {
    for (const [sid, qs] of Object.entries(preset.panelCustom)) {
      if (!panelCustom[sid]) panelCustom[sid] = {};
      for (const [qid, arr] of Object.entries(qs)) panelCustom[sid][qid] = arr;
    }
  }
  rvItems = preset.rvItems ? JSON.parse(JSON.stringify(preset.rvItems)) : {};
  _rvItemsCache = preset.rvItemsCache ? JSON.parse(JSON.stringify(preset.rvItemsCache)) : null;
  // Clear transient caches
  _tabSuggestCache = {}; _tabSuggestHistory = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  for (const c of Object.values(_tabSuggestCache)) { if (c._abort) c._abort.abort(); }
  _s2BannerDismissed = false;
  _viewAllPills = new Set();
  _collapsedSections = new Set();
  _s2CollapseAll = false;
  rvCollapsed = new Set();
  discoverAdded = new Set();
  _rvLoading = false;
}

async function savePreset() {
  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const suggested = role + (location ? ' \u00B7 ' + location : '');
  const name = await stratosPrompt({ title: 'Save Profile', label: 'Profile name', defaultValue: suggested });
  if (!name || !name.trim()) return;
  const key = name.trim();
  const presets = _getPresets();
  if (presets[key]) {
    if (!(await stratosConfirm(`"${key}" already exists. Overwrite?`, { title: 'Overwrite Profile', okText: 'Overwrite', cancelText: 'Cancel' }))) return;
  }
  presets[key] = {
    name: key,
    role, location,
    savedAt: new Date().toISOString(),
    ..._serializeWizardState()
  };
  _setPresets(presets);
  renderPresetBar();
  if (typeof showToast === 'function') showToast(`Profile "${key}" saved`, 'success');
}

function loadPreset(name) {
  const presets = _getPresets();
  const preset = presets[name];
  if (!preset) return;
  _loadPresetState(preset);
  // Update role/location inputs
  if (preset.role) {
    const roleInput = document.getElementById('simple-role');
    if (roleInput) roleInput.value = preset.role;
    _currentDomain = classifyRole(preset.role);
  }
  if (preset.location) {
    const locInput = document.getElementById('simple-location');
    if (locInput) locInput.value = preset.location;
  }
  _presetMenuOpen = false;
  renderPresetBar();
  // Re-render with loaded state
  renderAll();
  _wizSaveState();
  if (typeof showToast === 'function') showToast(`Loaded "${name}"`, 'success');
}

async function deletePreset(name, evt) {
  if (evt) { evt.stopPropagation(); evt.preventDefault(); }
  if (!(await stratosConfirm(`Delete "${name}"?`, { title: 'Delete Profile', okText: 'Delete', cancelText: 'Cancel' }))) return;
  const presets = _getPresets();
  delete presets[name];
  _setPresets(presets);
  renderPresetBar();
}

function togglePresetMenu() {
  _presetMenuOpen = !_presetMenuOpen;
  renderPresetBar();
}

function renderPresetBar() {
  const el = document.getElementById('wiz-preset-dd');
  if (!el) return;
  const presets = _getPresets();
  const names = Object.keys(presets).sort((a, b) => {
    const ta = presets[a].savedAt || '';
    const tb = presets[b].savedAt || '';
    return tb.localeCompare(ta); // newest first
  });
  const hasPresets = names.length > 0;

  let menuHtml = '';
  if (hasPresets) {
    menuHtml = names.map(n => {
      const p = presets[n];
      const date = p.savedAt ? new Date(p.savedAt).toLocaleDateString(undefined, {month:'short', day:'numeric'}) : '';
      return `<div class="preset-item" onclick="_wiz.loadPreset('${escAttr(n)}')">
        <span class="preset-item-name">${esc(n)}</span>
        <span class="preset-item-date">${esc(date)}</span>
        <button class="preset-del" onclick="_wiz.deletePreset('${escAttr(n)}', event)" title="Delete">&times;</button>
      </div>`;
    }).join('');
  } else {
    menuHtml = '<div class="preset-empty">No saved profiles</div>';
  }

  el.innerHTML = `
    <button class="preset-save-btn-hdr" onclick="_wiz.savePreset()">Save Profile</button>
    <button class="preset-btn ${_presetMenuOpen ? 'open' : ''}" onclick="_wiz.togglePresetMenu()">
      <span class="preset-name">${hasPresets ? `${names.length} saved` : 'Profiles'}</span>
      <span class="preset-chev">\u25BC</span>
    </button>
    <div class="preset-menu ${_presetMenuOpen ? 'open' : ''}">${menuHtml}</div>`;
}

function initState() {
  // Classify role for adaptive pills
  const roleVal = document.getElementById('simple-role')?.value?.trim() || '';
  _currentDomain = classifyRole(roleVal);
  const inferredStage = inferStage(roleVal);

  selCats = new Set();
  selSubs = {}; customSubs = {};
  for (const c of CATS) { selSubs[c.id] = new Set(); customSubs[c.id] = []; }
  interestTopics = [];
  panelSel = {}; panelCustom = {};
  for (const [sid, cfg] of Object.entries(PANELS)) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of cfg.qs) {
      panelCustom[sid][q.id] = [];
      if (q.type === 's' && q.id === 'stage') {
        panelSel[sid][q.id] = new Set([inferredStage]);
      } else if (q.type === 's' && q.def) {
        panelSel[sid][q.id] = new Set([q.def]);
      } else {
        const defs = getEffectiveDefs(q.id, q.defs || [], q.type);
        panelSel[sid][q.id] = new Set(defs);
      }
    }
  }
  activeTab = null;
  rvItems = {}; rvCollapsed = new Set();
  discoverAdded = new Set();
  _tabSuggestCache = {}; _tabSuggestHistory = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  for (const c of Object.values(_tabSuggestCache)) { if (c._abort) c._abort.abort(); }
  _s2BannerDismissed = false;
  _rvItemsCache = null;
  _rvLoading = false;
  _viewAllPills = new Set();
  _collapsedSections = new Set();
  _s2CollapseAll = false;
}

/* ═══════════════════════════════════════
   CSS INJECTION  (Stage 1: Theme Integration)
   Maps wizard colors → dashboard CSS custom properties.
   All selectors scoped under .wiz-scope.
   ═══════════════════════════════════════ */


/* CSS extracted to wizard-styles.css — loaded via <link> in injectCSS() */

function injectCSS() {
  if (document.getElementById('wiz-styles')) return;
  const link = document.createElement('link');
  link.id = 'wiz-styles';
  link.rel = 'stylesheet';
  link.href = 'wizard-styles.css';
  document.head.appendChild(link);
}

function removeCSS() {
  const el = document.getElementById('wiz-styles');
  if (el) el.remove();
}

/* ═══════════════════════════════════════
   DOM INJECTION
   ═══════════════════════════════════════ */

function injectDOM(role, location) {
  if (document.getElementById('wiz-root')) return;
  const wrapper = document.createElement('div');
  wrapper.id = 'wiz-root';
  wrapper.className = 'wiz-scope';
  const circumference = 2 * Math.PI * 16; // r=16 for ring
  wrapper.innerHTML = `
    <div class="backdrop" id="wiz-bk" onclick="_wiz.closeWizard()"></div>
    <div class="modal" id="wiz-modal">
      <canvas class="wiz-stars-canvas" id="wiz-stars-canvas"></canvas>
      <div class="grad-bar"></div>
      <div class="hdr">
        <div class="hdr-logo" data-tip="StratOS News Intelligence Platform">StratOS</div>
        <div class="hdr-sub">Intelligence Wizard</div>
        <div class="hdr-badges">
          ${role ? `<span class="badge" data-tip="Your professional role">${esc(role)}</span>` : ''}
          ${location ? `<span class="badge blue" data-tip="Your location">${esc(location)}</span>` : ''}
        </div>
        <div class="preset-dd" id="wiz-preset-dd"></div>
        <button class="wiz-clear-all-btn" onclick="_wiz.clearAll()" data-tip="Reset all selections">Clear All</button>
        <div class="atmos-selector">
          <button class="atmos-btn active" data-atmos="arcane" onclick="_wiz.setAtmosphere(this)">Arcane</button>
          <button class="atmos-btn" data-atmos="clean" onclick="_wiz.setAtmosphere(this)">Clean</button>
          <button class="atmos-btn" data-atmos="deep" onclick="_wiz.setAtmosphere(this)">Deep</button>
        </div>
        <div class="ring-wrap" id="wiz-ring-wrap" data-tip="Profile completion progress">
          <svg width="40" height="40" viewBox="0 0 42 42">
            <defs><linearGradient id="wizRingGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="var(--accent)"/><stop offset="100%" stop-color="var(--accent2)"/></linearGradient></defs>
            <circle class="ring-bg" cx="21" cy="21" r="16"/>
            <circle class="ring-fg" id="wiz-ring-fg" cx="21" cy="21" r="16" stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"/>
          </svg>
          <span class="ring-label" id="wiz-ring-pct">0%</span>
        </div>
        <button class="close-btn" onclick="_wiz.closeWizard()" data-tip="Close wizard (Esc)">&times;</button>
      </div>
      <div class="body">
        <div class="rail" id="wiz-rail">
          <div class="rail-handle" id="wiz-rail-handle" onclick="_wiz.toggleRail()">
            <div class="rail-handle-bar"></div>
          </div>
          <div class="rail-inner" id="wiz-rail-scroll"></div>
          <div id="wiz-feed-summary" style="padding:4px 0;"></div>
          <div class="rail-bottom" id="wiz-rail-bottom">
            <button class="build-btn" id="wiz-build-btn" onclick="_wiz.doBuild()" disabled data-tip="Generate your personalized intelligence feed">&#x2728; BUILD FEED &#x2728;</button>
          </div>
        </div>
        <div class="main" id="wiz-main">
          <div id="wiz-step-bar"></div>
          <div id="wiz-priorities"></div>
          <div id="wiz-details"></div>
          <div id="wiz-loading" class="wiz-hidden"></div>
        </div>
      </div>
    </div>`;
  document.body.appendChild(wrapper);
  _initWizTooltip();
}

function removeDOM() {
  _destroyWizTooltip();
  const el = document.getElementById('wiz-root');
  if (el) el.remove();
}


/* ═══════════════════════════════════════
   NAVIGATION HELPERS
   ═══════════════════════════════════════ */

function getActiveTabs() {
  const tabs = [];
  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    if (c.id === 'interests') { tabs.push({id:c.id,name:c.name,icon:c.icon,type:'interests'}); continue; }
    const allSubs = [...c.subs, ...customSubs[c.id]];
    const active = allSubs.filter(s => selSubs[c.id].has(s.id));
    if (active.length) tabs.push({id:c.id,name:c.name,icon:c.icon,type:'category',subs:active});
  }
  return tabs;
}

function getTabSections(tab) {
  const secs = [];
  if (tab.id === 'career') {
    const hasJ = tab.subs.some(s => s.id === 'jobhunt'), hasI = tab.subs.some(s => s.id === 'intern');
    if (hasJ && hasI) secs.push({id:'career_opps',name:'Career Opportunities',icon:'🎯'});
    else if (hasJ) secs.push({id:'jobhunt',name:'Job Hunting',icon:'🔍'});
    else if (hasI) secs.push({id:'intern',name:'Internships',icon:'🧑‍💻'});
    for (const s of tab.subs) if (s.id !== 'jobhunt' && s.id !== 'intern') secs.push({id:s.id,name:s.name,icon:getSubIcon(s.id)});
  } else {
    for (const s of tab.subs) secs.push({id:s.id,name:s.name,icon:getSubIcon(s.id)});
  }
  return secs;
}

function getS3Sections(cat) {
  const allSubs = [...cat.subs, ...customSubs[cat.id]];
  const active = allSubs.filter(s => selSubs[cat.id].has(s.id));
  if (cat.id === 'career') {
    const secs = [];
    const hasJ = active.some(s => s.id === 'jobhunt'), hasI = active.some(s => s.id === 'intern');
    if (hasJ && hasI) secs.push({id:'career_opps',name:'Career Opportunities',icon:'🎯'});
    else if (hasJ) secs.push({id:'jobhunt',name:'Job Hunting',icon:'🔍'});
    else if (hasI) secs.push({id:'intern',name:'Internships',icon:'🧑‍💻'});
    for (const s of active) if (s.id !== 'jobhunt' && s.id !== 'intern') secs.push({id:s.id,name:s.name,icon:getSubIcon(s.id)});
    return secs;
  }
  return active.map(s => ({id:s.id,name:s.name,icon:getSubIcon(s.id)}));
}

function getSubIcon(sid) { return PANELS[sid]?.icon || '\uD83D\uDCCC'; }

function tabHasSelections(tab) {
  if (tab.type === 'interests') return interestTopics.length > 0;
  if (!tab.subs) return false;
  const sections = getTabSections(tab);
  for (const sec of sections) {
    const panel = PANELS[sec.id]; if (!panel) continue;
    const sel = panelSel[sec.id]; if (!sel) continue;
    for (const q of panel.qs) { if (sel[q.id] && sel[q.id].size > 0) return true; }
  }
  return false;
}


/* ═══════════════════════════════════════
   PROGRESS RING
   ═══════════════════════════════════════ */

function updateRing() {
  const fg = document.getElementById('wiz-ring-fg');
  const pct = document.getElementById('wiz-ring-pct');
  if (!fg || !pct) return;
  // Calculate: how many selected categories have at least one detail configured?
  const total = selCats.size;
  if (total === 0) { pct.textContent = '0%'; fg.style.strokeDashoffset = 2 * Math.PI * 16; return; }
  let done = 0;
  for (const cid of selCats) {
    if (cid === 'interests') { if (interestTopics.length > 0) done++; continue; }
    const subs = selSubs[cid];
    if (!subs || subs.size === 0) continue;
    // Check if any sub has panel selections
    let hasSel = false;
    for (const sid of subs) {
      const ps = panelSel[sid];
      if (ps) { for (const qid of Object.keys(ps)) { if (ps[qid] && ps[qid].size > 0) { hasSel = true; break; } } }
      if (hasSel) break;
    }
    if (hasSel) done++;
  }
  const ratio = done / total;
  const circumference = 2 * Math.PI * 16;
  fg.style.strokeDashoffset = circumference * (1 - ratio);
  pct.textContent = Math.round(ratio * 100) + '%';
}

function updateBuildButton() {
  const btn = document.getElementById('wiz-build-btn');
  if (!btn) return;
  if (selCats.size === 0) { btn.disabled = true; return; }
  const hasActiveSubs = [...selCats].some(id => {
    if (id === 'interests') return interestTopics.length > 0;
    return selSubs[id] && selSubs[id].size > 0;
  });
  btn.disabled = !hasActiveSubs;
}

/* ═══════════════════════════════════════
   PRIORITIES — Card grid
   ═══════════════════════════════════════ */

function renderPriorities() {
  const el = document.getElementById('wiz-priorities');
  if (!el) return;
  el.innerHTML = `
    <h2 class="sec-title">Choose Your Focus Areas</h2>
    <p class="sec-sub">Select the categories that matter most to you</p>
    <div class="cards-grid">${CATS.map(c => renderCard(c)).join('')}</div>
    <div class="quick-setup-wrap">
      <button class="quick-setup-btn" onclick="_wiz.skipToQuick()">&#x26A1; Quick Setup <span class="quick-setup-sub">Auto-configure based on your role</span></button>
    </div>`;
}

function renderCard(c) {
  const sel = selCats.has(c.id);
  let pillsHTML = '';
  if (c.dynamic) {
    pillsHTML = `<div class="card-pills"><span class="card-pill-hint">You'll customize these in details \u2192</span></div>`;
  } else {
    const allSubs = [...c.subs, ...customSubs[c.id]];
    const subsHTML = allSubs.map(s => {
      const on = selSubs[c.id].has(s.id);
      const isC = customSubs[c.id].some(cs => cs.id === s.id);
      return `<span class="card-pill ${on ? '' : ''}" onclick="event.stopPropagation();_wiz.togSub('${c.id}','${s.id}',this)" style="cursor:pointer${on ? '' : ';opacity:.5'}">
        ${s.name}${isC ? `<span class="card-pill-x" onclick="event.stopPropagation();_wiz.rmCustomSub('${c.id}','${s.id}')">&times;</span>` : ''}
      </span>`;
    }).join('');
    pillsHTML = `<div class="card-pills">${subsHTML}
      <span id="wiz-saw-${c.id}" style="display:none" onclick="event.stopPropagation()"><input class="add-inp-card" id="wiz-sai-${c.id}" placeholder="Type & Enter" onkeydown="_wiz.addSubKey(event,'${c.id}')"></span>
      <span class="card-pill sp-add" id="wiz-sab-${c.id}" onclick="event.stopPropagation();_wiz.showAddSub('${c.id}')" style="cursor:pointer">+ Add</span>
    </div>`;
  }
  const tapHint = sel ? '' : `<div class="card-tap">Tap to add</div>`;
  return `<div class="p-card ${sel ? 'sel' : ''}" onclick="_wiz.togCat('${c.id}')" data-tip="${esc(c.desc)}">
    <div class="card-check">${CK}</div>
    <div class="card-icon ci-${c.id}">${c.icon}</div>
    <div class="card-name">${c.name}</div>
    <div class="card-desc">${c.desc}</div>
    ${pillsHTML}${tapHint}
  </div>`;
}

/* ═══════════════════════════════════════
   DETAILS — Accordion sections
   ═══════════════════════════════════════ */

function renderDetails() {
  const el = document.getElementById('wiz-details');
  if (!el) return;
  const tabs = getActiveTabs();
  if (!tabs.length) { el.innerHTML = ''; return; }

  const bannerH = _s2BannerDismissed ? '' : `<div class="s2-banner"><span class="s2-banner-icon">&#x2728;</span><span><strong>Bold selections steer the AI</strong> &mdash; pick your career stage and preferences to get personalized suggestions below.</span><span class="s2-banner-x" onclick="_wiz.dismissS2Banner()">&times;</span></div>`;

  let sectionsHTML = '';
  for (const tab of tabs) {
    if (tab.type === 'interests') {
      const isOpen = !_collapsedSections.has('interests_det');
      const itemsH = interestTopics.map(t => `<span class="acc-pill on">${esc(t)}<span class="pill-x" onclick="event.stopPropagation();_wiz.rmIntS2('${escAttr(t)}')">&times;</span></span>`).join('');
      const sugH = INTEREST_SUGGESTIONS.map(s => {
        const on = interestTopics.includes(s);
        return `<span class="acc-pill pill-sug ${on ? 'on' : ''}" onclick="_wiz.togIntS2('${escAttr(s)}')">${esc(s)}</span>`;
      }).join('');
      sectionsHTML += `<div class="accordion ${isOpen ? 'open' : ''}">
        <div class="acc-header" onclick="_wiz.togDetSection('interests_det')" data-tip="Customize your personal interests and hobbies">
          <span class="acc-icon ai-${tab.id}">${tab.icon}</span>
          <span class="acc-title">${tab.name}</span>
          ${interestTopics.length ? `<span class="acc-tag">${interestTopics.length} topics</span>` : ''}
          <span class="acc-chevron">&#9660;</span>
        </div>
        <div class="acc-body"><div class="acc-content">
          <div><div class="acc-group-label">What do you follow? <span class="s2-hint">\u00B7 Type a topic and press Enter</span></div>
          <div class="int-wrap"><input class="int-inp" id="wiz-int-inp" placeholder="e.g. Quantum Computing, Gaming..." onkeydown="_wiz.intKeyS2(event)"></div></div>
          ${interestTopics.length ? `<div><div class="acc-group-label">Your topics</div><div class="acc-pills">${itemsH}</div></div>` : ''}
          <div><div class="acc-group-label">Suggested for your role <span class="s2-hint">\u00B7 tap to add</span></div>
          <div class="acc-pills">${sugH}</div></div>
        </div></div>
      </div>`;
      continue;
    }

    const sections = getTabSections(tab);
    // Category header above all its subcategory accordions
    sectionsHTML += `<div class="wiz-cat-hdr"><span class="wiz-cat-hdr-icon">${tab.icon}</span> ${esc(tab.name)}</div>`;
    for (const sec of sections) {
      const panel = PANELS[sec.id];
      if (!panel) {
        sectionsHTML += renderGenericDetSection(sec);
        continue;
      }
      const sel = panelSel[sec.id] || {};
      const custom = panelCustom[sec.id] || {};
      const isOpen = !_collapsedSections.has(sec.id);
      let selCount = 0;
      for (const q of panel.qs) { selCount += (sel[q.id]?.size || 0); }

      let bodyH = panel.qs.map(q => {
        const isViewAll = _viewAllPills.has(q.id);
        const basePills = isViewAll ? getAllPills(q.id, q.pills) : getEffectivePills(q.id, q.pills);
        const fullCount = getAllPills(q.id, q.pills).length;
        const shortCount = getEffectivePills(q.id, q.pills).length;
        const hasMore = fullCount > shortCount;
        const picked = sel[q.id] || new Set();
        const all = [...basePills, ...(custom[q.id] || [])];
        const hint = q.hint ? ` <span class="s2-hint">\u00B7 ${q.hint}</span>` : '';
        const isDecisive = DETERMINISTIC_QS.has(q.id);

        let h = `<div><div class="acc-group-label">${q.label}${hint}</div><div class="acc-pills">`;
        for (const p of all) {
          const isC = (custom[q.id] || []).includes(p);
          h += `<span class="acc-pill ${isDecisive ? 'pill-decisive' : ''} ${picked.has(p) ? 'on' : ''}" onclick="_wiz.togPanel('${sec.id}','${q.id}','${escAttr(p)}','${q.type}')">
            ${esc(p)}${isC ? `<span class="pill-x" onclick="event.stopPropagation();_wiz.rmPanelCustom('${sec.id}','${q.id}','${escAttr(p)}')">&times;</span>` : ''}
          </span>`;
        }
        if (q.canAdd) {
          h += `<span id="wiz-aw-${sec.id}-${q.id}" style="display:none"><input class="add-inp" id="wiz-ai-${sec.id}-${q.id}" placeholder="Type & Enter" onkeydown="_wiz.addPanelKey(event,'${sec.id}','${q.id}')"></span>`;
          h += `<span class="add-pill" id="wiz-ab-${sec.id}-${q.id}" onclick="_wiz.showPanelAdd('${sec.id}','${q.id}')">+ Add</span>`;
        }
        h += `</div>`;
        if (hasMore && q.type !== 's') {
          h += `<button class="va-tog" onclick="_wiz.togViewAll('${q.id}')">${isViewAll ? 'Show less' : `View all (${fullCount})`}</button>`;
        }
        h += `</div>`;
        return h;
      }).join('');

      sectionsHTML += `<div class="accordion ${isOpen ? 'open' : ''}">
        <div class="acc-header" onclick="_wiz.togDetSection('${sec.id}')">
          <span class="acc-icon ai-${tab.id}">${sec.icon}</span>
          <span class="acc-title">${sec.name}</span>
          ${selCount ? `<span class="acc-tag">${selCount} selected</span>` : ''}
          <span class="acc-chevron">&#9660;</span>
        </div>
        <div class="acc-body"><div class="acc-content">${bodyH}</div></div>
      </div>`;

    }

    // Suggestions + discover rendered once per tab (after all sections)
    const sugH = renderTabSuggestions(tab.id);
    let discoverH = '';
    const discoverItems = getRailDiscover(tab.id);
    if (discoverItems.length) {
      discoverH = `<div style="padding:0 12px 12px"><div class="acc-group-label" style="display:flex;align-items:center;gap:6px">&#128269; Discover <span class="s2-hint">\u00B7 related topics to explore</span></div>
        <div class="acc-pills">${discoverItems.map(d =>
          `<span class="acc-pill pill-sug ${discoverAdded.has(d.name) ? 'on' : ''}" onclick="_wiz.discoverAdd('${escAttr(d.name)}','${escAttr(d.target || '')}')">${esc(d.name)}${discoverAdded.has(d.name) ? '<span class="pill-x" onclick="event.stopPropagation();_wiz.discoverRm(\''+escAttr(d.name)+'\',\''+escAttr(d.target || '')+'\')">&times;</span>' : ''}</span>`
        ).join('')}</div></div>`;
    }
    if (sugH || discoverH) sectionsHTML += sugH + discoverH;
  }
  // Kick off suggestion fetch for the first uncached tab only if none are in-flight
  // (subsequent tabs are chained from fetchTabSuggestion completion — NOT from here)
  const anyInFlight = tabs.some(t => _tabSuggestCache[t.id]?.loading);
  const anyCached = tabs.some(t => _tabSuggestCache[t.id]);
  if (!anyInFlight && !anyCached) {
    // First-time trigger only — chain handles the rest
    for (const tab of tabs) {
      if (tab.id !== 'interests') { fetchTabSuggestion(tab.id); break; }
    }
  }

  el.innerHTML = `
    <div class="section-divider"></div>
    <h2 class="sec-title2">Fine-tune Your Selections</h2>
    <p class="sec-sub2">Customize each category to get exactly the intelligence you need.</p>
    ${bannerH}${sectionsHTML}`;
}

function renderGenericDetSection(sec) {
  const custom = panelCustom[sec.id]?.kw || [];
  const isOpen = !_collapsedSections.has(sec.id);
  let html = `<div class="accordion ${isOpen ? 'open' : ''}">
    <div class="acc-header" onclick="_wiz.togDetSection('${sec.id}')">
      <span class="acc-icon">\uD83D\uDCCC</span>
      <span class="acc-title">${sec.name}</span>
      ${custom.length ? `<span class="acc-tag">${custom.length} keywords</span>` : ''}
      <span class="acc-chevron">&#9660;</span>
    </div>
    <div class="acc-body"><div class="acc-content">
      <div><div class="acc-group-label">Keywords to track <span class="s2-hint">\u00B7 Type and press Enter</span></div><div class="acc-pills">`;
  for (const p of custom) html += `<span class="acc-pill on">${esc(p)}<span class="pill-x" onclick="event.stopPropagation();_wiz.rmPanelCustom('${sec.id}','kw','${escAttr(p)}')">&times;</span></span>`;
  html += `<span id="wiz-aw-${sec.id}-kw" style="display:none"><input class="add-inp" id="wiz-ai-${sec.id}-kw" placeholder="Type & Enter" onkeydown="_wiz.addPanelKey(event,'${sec.id}','kw')"></span>`;
  html += `<span class="add-pill" id="wiz-ab-${sec.id}-kw" onclick="_wiz.showPanelAdd('${sec.id}','kw')">+ Add</span></div></div>
    </div></div>
  </div>`;
  return html;
}

function togDetSection(secId) {
  // Toggle: if in collapsed set, remove (= open). If not, add (= collapse).
  _collapsedSections.has(secId) ? _collapsedSections.delete(secId) : _collapsedSections.add(secId);
  renderDetails();
}

/* ═══════════════════════════════════════
   LEFT RAIL — Live preview + per-category discover
   ═══════════════════════════════════════ */

function renderRail() {
  const scroll = document.getElementById('wiz-rail-scroll');
  if (!scroll) return;

  let html = '';
  let totalItems = 0;
  let catCount = 0;

  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    catCount++;

    if (catCount > 1) html += `<div class="rail-divider"></div>`;

    if (c.id === 'interests') {
      if (!interestTopics.length) continue;
      totalItems += interestTopics.length;
      const col = rvCollapsed.has('interests');
      html += `<div class="rail-section ${col ? 'collapsed' : ''}">
        <div class="rail-cat-hdr cat-interests" onclick="_wiz.togRvCollapse('interests')" data-tip="Your personal interest topics">
          <span class="rail-cat-icon">${c.icon}</span>
          <span>${c.name}</span>
          <span class="rail-cat-count">${interestTopics.length}</span>
          <span class="rail-sec-chev">\u25BC</span>
        </div>
        <div class="rail-sec-body">
          <div class="rail-items">
            ${interestTopics.map(t => `<div class="rail-item"><span class="bullet"></span>${esc(t)}<span class="rp-x" onclick="event.stopPropagation();_wiz.rvRmInt('${escAttr(t)}')">&times;</span></div>`).join('')}
          </div>
        </div>
      </div>`;
      continue;
    }

    const sections = getS3Sections(c);
    if (!sections.length) continue;

    let catItems = [];
    for (const sec of sections) {
      const items = rvItems[sec.id] || [];
      catItems.push(...items.map(it => ({name: it, sec: sec.id})));
    }
    const subs = selSubs[c.id];
    let subCount = subs ? subs.size : 0;

    totalItems += catItems.length + subCount;
    const col = rvCollapsed.has(c.id);

    html += `<div class="rail-section ${col ? 'collapsed' : ''}">
      <div class="rail-cat-hdr cat-${c.id}" onclick="_wiz.togRvCollapse('${c.id}')" data-tip="${esc(c.desc)}">
        <span class="rail-cat-icon">${c.icon}</span>
        <span>${c.name}</span>
        <span class="rail-cat-count">${catItems.length || subCount}</span>
        <span class="rail-sec-chev">\u25BC</span>
      </div>
      <div class="rail-sec-body">`;

    if (catItems.length) {
      html += `<div class="rail-items">`;
      for (const it of catItems) {
        html += `<div class="rail-item"><span class="bullet"></span>${esc(it.name)}<span class="rp-x" onclick="event.stopPropagation();_wiz.rvRm('${escAttr(it.sec)}','${escAttr(it.name)}')">&times;</span></div>`;
      }
      html += `</div>`;
    } else if (subCount) {
      const subNames = [...subs].map(sid => {
        const sub = [...c.subs, ...customSubs[c.id]].find(s => s.id === sid);
        return sub ? sub.name : sid;
      });
      html += `<div class="rail-items">${subNames.map(n => `<div class="rail-item"><span class="bullet"></span>${esc(n)}</div>`).join('')}</div>`;
    } else {
      html += `<div class="rail-empty">No items yet</div>`;
    }

    // Suggestions and discover moved to main panel (renderDetails)

    html += `</div></div>`;
  }

  if (!html) {
    html = `<div class="rail-empty" style="padding:20px;text-align:center">Select categories to see your preview here</div>`;
  }

  scroll.innerHTML = html;

  renderFeedSummary(totalItems);
  updateBuildButton();
  updateRing();
}

function getRailDiscover(catId) {
  if (!_rvItemsCache?.discover?.length) return [];
  // Match discover items whose target belongs to a sub of this category
  const catSubs = selSubs[catId] || new Set();
  const catSubIds = new Set(catSubs);
  // Also include S3 section IDs (career_opps etc.)
  const cat = CATS.find(c => c.id === catId);
  if (cat) {
    const s3 = getS3Sections(cat);
    for (const sec of s3) catSubIds.add(sec.id);
  }
  return _rvItemsCache.discover.filter(d => catSubIds.has(d.target));
}

function renderFeedSummary(totalItems) {
  const el = document.getElementById('wiz-feed-summary');
  if (!el) return;
  const catNum = selCats.size;
  if (catNum === 0) { el.innerHTML = ''; return; }
  el.innerHTML = `<div class="feed-summary">
    <div class="feed-summary-title">Feed Summary</div>
    <div class="feed-stat"><span class="feed-stat-label">Categories</span><span class="feed-stat-val accent">${catNum} selected</span></div>
    <div class="feed-stat"><span class="feed-stat-label">Tracked topics</span><span class="feed-stat-val">${totalItems} items</span></div>
  </div>`;
}

function renderStepBar() {
  const el = document.getElementById('wiz-step-bar');
  if (!el) return;
  const hasCats = selCats.size > 0;
  const hasDetails = hasCats && [...selCats].some(id => {
    if (id === 'interests') return interestTopics.length > 0;
    return selSubs[id] && selSubs[id].size > 0;
  });
  el.innerHTML = `<div class="steps-bar">
    <div class="step-num-circle ${hasCats ? 'done' : 'active'}" data-tip="Step 1: Choose your priority categories" data-tip-pos="bottom">1</div>
    <span class="step-bar-label ${hasCats ? 'active' : ''}">Priorities</span>
    <div class="step-line ${hasCats ? 'done' : ''}"></div>
    <div class="step-num-circle ${hasDetails ? 'done' : hasCats ? 'active' : ''}" data-tip="Step 2: Fine-tune your selections" data-tip-pos="bottom">2</div>
    <span class="step-bar-label ${hasCats ? 'active' : ''}">Details</span>
    <div class="step-line ${hasDetails ? 'done' : ''}"></div>
    <div class="step-num-circle ${hasDetails ? 'active' : ''}" data-tip="Step 3: Preview and build your feed" data-tip-pos="bottom">3</div>
    <span class="step-bar-label ${hasDetails ? 'active' : ''}">Build</span>
  </div>`;
}

function toggleRail() {
  const rail = document.getElementById('wiz-rail');
  if (rail) rail.classList.toggle('expanded');
}

/* ── Atmosphere system ── */
let _wizStarEngine = null;

function setAtmosphere(btnEl) {
  const atmos = btnEl.getAttribute('data-atmos');
  const root = document.getElementById('wiz-root');
  if (!root) return;
  root.querySelectorAll('.atmos-btn').forEach(b => b.classList.remove('active'));
  btnEl.classList.add('active');
  if (atmos === 'clean') {
    root.removeAttribute('data-wiz-atmos');
  } else {
    root.setAttribute('data-wiz-atmos', atmos);
  }
  // Start/stop star canvas
  if (atmos === 'arcane') {
    _startWizStars();
  } else {
    _stopWizStars();
  }
  try { localStorage.setItem('wiz-atmosphere', atmos); } catch(e) {}
}

function restoreAtmosphere() {
  try {
    const a = localStorage.getItem('wiz-atmosphere') || 'arcane';
    const root = document.getElementById('wiz-root');
    if (root) {
      if (a === 'clean') root.removeAttribute('data-wiz-atmos');
      else root.setAttribute('data-wiz-atmos', a);
      root.querySelectorAll('.atmos-btn').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-atmos') === a);
      });
      if (a === 'arcane') _startWizStars();
    }
  } catch(e) {}
}

function _stopWizStars() {
  if (!_wizStarEngine) return;
  cancelAnimationFrame(_wizStarEngine.raf);
  if (_wizStarEngine.onResize) window.removeEventListener('resize', _wizStarEngine.onResize);
  _wizStarEngine = null;
  const c = document.getElementById('wiz-stars-canvas');
  if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); }
}

function _startWizStars() {
  _stopWizStars();
  const canvas = document.getElementById('wiz-stars-canvas');
  if (!canvas) return;
  const modal = canvas.parentElement;
  if (!modal) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = modal.offsetWidth;
    canvas.height = modal.offsetHeight;
  }
  resize();

  const isMobile = window.innerWidth <= 768;
  const COUNT = isMobile ? 30 : 120;
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#34d399';

  // Parse accent for glow
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

  // Shooting stars
  const shooters = [];
  let lastShoot = Date.now();

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const t = Date.now() * 0.001;

    // Stars
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

    // Constellation lines between close stars
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

    _wizStarEngine.raf = requestAnimationFrame(draw);
  }

  _wizStarEngine = { raf: requestAnimationFrame(draw) };

  // Resize on window change
  const onResize = () => { if (_wizStarEngine) resize(); };
  window.addEventListener('resize', onResize);
  _wizStarEngine.onResize = onResize;
}

function _cleanupWizStars() {
  if (_wizStarEngine) {
    cancelAnimationFrame(_wizStarEngine.raf);
    if (_wizStarEngine.onResize) window.removeEventListener('resize', _wizStarEngine.onResize);
    _wizStarEngine = null;
  }
}

/* ═══════════════════════════════════════
   JS FLOATING TOOLTIP (escapes overflow:clip)
   ═══════════════════════════════════════ */
var _wizTipEl = null;
var _wizTipTimer = null;

function _initWizTooltip() {
  if (_wizTipEl) return;
  _wizTipEl = document.createElement('div');
  _wizTipEl.className = 'wiz-tip-float';
  document.body.appendChild(_wizTipEl);

  var root = document.getElementById('wiz-root');
  if (!root) return;

  root.addEventListener('mouseover', function(e) {
    var target = e.target.closest('[data-tip]');
    if (!target) return;
    var text = target.getAttribute('data-tip');
    if (!text) return;
    clearTimeout(_wizTipTimer);
    _wizTipEl.textContent = text;
    _wizTipEl.classList.remove('visible');

    // Position: prefer above, fall back to below
    var rect = target.getBoundingClientRect();
    _wizTipEl.style.left = '0'; _wizTipEl.style.top = '0';
    _wizTipEl.classList.add('visible'); // needed to measure
    _wizTipEl.style.opacity = '0';

    var tw = _wizTipEl.offsetWidth, th = _wizTipEl.offsetHeight;
    var posBottom = target.getAttribute('data-tip-pos') === 'bottom';
    var left = rect.left + rect.width / 2 - tw / 2;
    var top = posBottom ? rect.bottom + 10 : rect.top - th - 10;

    // Fallback if tooltip goes above viewport
    if (top < 4) { top = rect.bottom + 10; }
    // Keep within horizontal viewport
    if (left < 4) left = 4;
    if (left + tw > window.innerWidth - 4) left = window.innerWidth - tw - 4;

    _wizTipEl.style.left = Math.round(left) + 'px';
    _wizTipEl.style.top = Math.round(top) + 'px';
    _wizTipEl.style.opacity = '';
  });

  root.addEventListener('mouseout', function(e) {
    var target = e.target.closest('[data-tip]');
    if (!target) return;
    _wizTipTimer = setTimeout(function() {
      if (_wizTipEl) _wizTipEl.classList.remove('visible');
    }, 80);
  });
}

function _destroyWizTooltip() {
  clearTimeout(_wizTipTimer);
  if (_wizTipEl) { _wizTipEl.remove(); _wizTipEl = null; }
}

/* ═══════════════════════════════════════
   CATEGORY / SUB TOGGLES
   ═══════════════════════════════════════ */

function togCat(id) {
  if (selCats.has(id)) { selCats.delete(id); if (id !== 'interests') selSubs[id].clear(); else interestTopics = []; }
  else { selCats.add(id); selSubs[id] = selSubs[id] && selSubs[id].size ? selSubs[id] : new Set(); }
  _rvItemsCache = null;
  renderAll(); _wizSaveState();
  // Re-fetch discover items in background
  if (selCats.size > 0) initRvWithAI();
}

function togSub(cid, sid, el) {
  selSubs[cid].has(sid) ? selSubs[cid].delete(sid) : selSubs[cid].add(sid);
  if (selSubs[cid].has(sid) && PANELS[sid] && !panelSel[sid]) {
    panelSel[sid] = {}; panelCustom[sid] = {};
    for (const q of PANELS[sid].qs) { panelCustom[sid][q.id] = []; panelSel[sid][q.id] = q.type === 's' && q.def ? new Set([q.def]) : new Set(q.defs || []); }
  }
  _rvItemsCache = null;
  if (el) el.classList.toggle('on', selSubs[cid].has(sid));
  renderAll(); _wizSaveState();
  // Re-fetch discover items in background
  if (selCats.size > 0) initRvWithAI();
}

function showAddSub(cid) {
  const w = document.getElementById('wiz-saw-' + cid);
  if (w) w.style.display = 'inline';
  const b = document.getElementById('wiz-sab-' + cid);
  if (b) b.style.display = 'none';
  const i = document.getElementById('wiz-sai-' + cid);
  if (i) i.focus();
}

function addSubKey(e, cid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const name = e.target.value.trim();
    const id = 'custom_' + name.toLowerCase().replace(/\W+/g, '_');
    if (!customSubs[cid].some(s => s.id === id)) { customSubs[cid].push({id, name}); selSubs[cid].add(id); panelSel[id] = {kw: new Set()}; panelCustom[id] = {kw: []}; }
    e.target.value = '';
    renderPriorities(); _wizSaveState();
    setTimeout(() => {
      const w = document.getElementById('wiz-saw-' + cid), b = document.getElementById('wiz-sab-' + cid), i = document.getElementById('wiz-sai-' + cid);
      if (w) w.style.display = 'inline'; if (b) b.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const b = document.getElementById('wiz-sab-' + cid); if (b) b.style.display = '';
  }
}

function rmCustomSub(cid, sid) {
  customSubs[cid] = customSubs[cid].filter(s => s.id !== sid);
  selSubs[cid].delete(sid); delete panelSel[sid]; delete panelCustom[sid];
  renderAll(); _wizSaveState();
}

function clearAll() {
  initState();
  _wizClearState();
  _tabSuggestCache = {}; _tabSuggestHistory = {};
  clearTimeout(_suggestDebounceTimer); _suggestDebounceTimer = null;
  for (const c of Object.values(_tabSuggestCache)) { if (c._abort) c._abort.abort(); }
  _s2BannerDismissed = false;
  renderAll();
  if (typeof showToast === 'function') showToast('All selections cleared', 'info');
}

/** Render all sections — called after any toggle */
function renderAll() {
  renderStepBar();
  renderPriorities();
  renderDetails();
  renderRail();
  updateBuildButton();
  updateRing();
}

/* ═══════════════════════════════════════
   DETAIL PANEL INTERACTIONS
   ═══════════════════════════════════════ */

function dismissS2Banner() { _s2BannerDismissed = true; const b = document.querySelector('.s2-banner'); if (b) b.remove(); }

function togIntS2(val) { const i = interestTopics.indexOf(val); i >= 0 ? interestTopics.splice(i, 1) : interestTopics.push(val); renderDetails(); renderRail(); _wizSaveState(); }
function rmIntS2(val) { interestTopics = interestTopics.filter(v => v !== val); renderDetails(); renderRail(); _wizSaveState(); }
function intKeyS2(e) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim(); if (!interestTopics.includes(v)) interestTopics.push(v);
    e.target.value = ''; renderDetails(); renderRail(); _wizSaveState();
    setTimeout(() => { const i = document.getElementById('wiz-int-inp'); if (i) i.focus(); }, 60);
  }
}

function togPanel(sid, qid, val, type) {
  if (!panelSel[sid]) panelSel[sid] = {}; if (!panelSel[sid][qid]) panelSel[sid][qid] = new Set();
  const s = panelSel[sid][qid];
  const oldVal = type === 's' ? [...s][0] : null;
  if (type === 's') { s.clear(); s.add(val); } else { s.has(val) ? s.delete(val) : s.add(val); }
  // Find the parent category tab for this section
  const parentTab = getActiveTabs().find(t => {
    if (t.type === 'interests') return false;
    const secs = getTabSections(t);
    return secs.some(sec => sec.id === sid);
  });
  const tabId = parentTab?.id;
  const changed = DETERMINISTIC_QS.has(qid) && (type !== 's' || oldVal !== val);
  if (changed && tabId) {
    if (qid === 'stage') {
      for (const pid of STAGE_SHARED_PANELS) {
        if (pid !== sid) {
          if (!panelSel[pid]) panelSel[pid] = {};
          panelSel[pid]['stage'] = new Set([val]);
        }
      }
      for (const tid of Object.keys(_tabSuggestCache)) {
        if (tid !== tabId) delete _tabSuggestCache[tid];
      }
    }
    clearTimeout(_suggestDebounceTimer);
    _suggestDebounceTimer = setTimeout(() => refreshSuggestions(tabId), 800);
  }
  renderDetails(); renderRail(); updateRing(); _wizSaveState();
}

function showPanelAdd(sid, qid) {
  const w = document.getElementById('wiz-aw-' + sid + '-' + qid); if (w) w.style.display = 'inline';
  const b = document.getElementById('wiz-ab-' + sid + '-' + qid); if (b) b.style.display = 'none';
  const i = document.getElementById('wiz-ai-' + sid + '-' + qid); if (i) i.focus();
}

function addPanelKey(e, sid, qid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim();
    if (!panelCustom[sid]) panelCustom[sid] = {}; if (!panelCustom[sid][qid]) panelCustom[sid][qid] = [];
    if (!panelSel[sid]) panelSel[sid] = {}; if (!panelSel[sid][qid]) panelSel[sid][qid] = new Set();
    if (!panelCustom[sid][qid].includes(v)) { panelCustom[sid][qid].push(v); panelSel[sid][qid].add(v); }
    e.target.value = ''; renderDetails(); _wizSaveState();
    setTimeout(() => {
      const w = document.getElementById('wiz-aw-' + sid + '-' + qid), b = document.getElementById('wiz-ab-' + sid + '-' + qid), i = document.getElementById('wiz-ai-' + sid + '-' + qid);
      if (w) w.style.display = 'inline'; if (b) b.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const b = document.getElementById('wiz-ab-' + sid + '-' + qid); if (b) b.style.display = '';
  }
}

function rmPanelCustom(sid, qid, val) {
  panelCustom[sid][qid] = (panelCustom[sid][qid] || []).filter(v => v !== val);
  if (panelSel[sid]?.[qid]) panelSel[sid][qid].delete(val);
  renderDetails(); _wizSaveState();
}

function togViewAll(qId) {
  _viewAllPills.has(qId) ? _viewAllPills.delete(qId) : _viewAllPills.add(qId);
  renderDetails();
}

/* ═══════════════════════════════════════
   REVIEW ITEMS (AI entities)
   ═══════════════════════════════════════ */

async function fetchRvItems() {
  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  if (!role) return;
  // Build sections list from current selections
  const sections = [];
  for (const c of CATS) {
    if (!selCats.has(c.id) || c.id === 'interests') continue;
    for (const sec of getS3Sections(c)) {
      sections.push({id: sec.id, name: sec.name, category: c.name});
    }
  }
  if (!sections.length) return;
  _rvLoading = true;
  try {
    const resp = await fetch('/api/wizard-rv-items', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role, location, sections})
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data.sections && Object.keys(data.sections).length) {
        _rvItemsCache = data;
        console.log('[Wizard] rv-items: got AI entities for', Object.keys(data.sections).length, 'sections');
      }
    }
  } catch(e) { console.warn('[Wizard] rv-items fetch failed:', e); }
  _rvLoading = false;
}

function initRv() {
  rvItems = {};
  for (const c of CATS) {
    if (!selCats.has(c.id) || c.id === 'interests') continue;
    const sections = getS3Sections(c);
    for (const sec of sections) {
      // Use AI-generated items only — no hardcoded fallback
      const aiSec = _rvItemsCache?.sections?.[sec.id];
      if (aiSec && aiSec.items?.length) {
        rvItems[sec.id] = [...aiSec.items.slice(0, 8)];
      } else {
        // Empty — renderRail will show "No items yet" placeholder
        rvItems[sec.id] = [];
      }
    }
  }
}

async function initRvWithAI() {
  if (_rvItemsCache) {
    initRv();
    renderRail();
    return;
  }
  // Show loading in main area
  const loading = document.getElementById('wiz-loading');
  if (loading) {
    loading.classList.remove('wiz-hidden');
    const role = document.getElementById('simple-role')?.value?.trim() || '';
    loading.innerHTML = '<div class="ld"><div class="wiz-ring"></div><div class="ld-t">Personalizing your review...</div><div class="ld-s">Finding relevant entities for <strong>' + esc(role) + '</strong></div></div>';
  }
  await fetchRvItems();
  initRv();
  if (loading) { loading.classList.add('wiz-hidden'); loading.innerHTML = ''; }
  renderRail();
}

function togRvCollapse(id) { rvCollapsed.has(id) ? rvCollapsed.delete(id) : rvCollapsed.add(id); renderRail(); }
function collapseAllRv() { for (const c of CATS) if (selCats.has(c.id)) rvCollapsed.add(c.id); renderRail(); }
function expandAllRv() { rvCollapsed.clear(); renderRail(); }

function rvRm(sid, val) { rvItems[sid] = (rvItems[sid] || []).filter(v => v !== val); discoverAdded.delete(val); renderRail(); }
function rvRmInt(val) { interestTopics = interestTopics.filter(v => v !== val); renderRail(); renderDetails(); }
function rvShowAdd(sid) {
  const w = document.getElementById('wiz-rw-' + sid); if (w) w.style.display = 'inline';
  const a = document.getElementById('wiz-ra-' + sid); if (a) a.style.display = 'none';
  const i = document.getElementById('wiz-ri-' + sid); if (i) i.focus();
}
function rvAddKey(e, sid) {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const v = e.target.value.trim(); if (!rvItems[sid]) rvItems[sid] = []; if (!rvItems[sid].includes(v)) rvItems[sid].push(v);
    e.target.value = ''; renderRail();
    setTimeout(() => {
      const w = document.getElementById('wiz-rw-' + sid), a = document.getElementById('wiz-ra-' + sid), i = document.getElementById('wiz-ri-' + sid);
      if (w) w.style.display = 'inline'; if (a) a.style.display = 'none'; if (i) i.focus();
    }, 60);
  } else if (e.key === 'Escape') {
    e.target.value = ''; e.target.parentElement.style.display = 'none';
    const a = document.getElementById('wiz-ra-' + sid); if (a) a.style.display = '';
  }
}

/* ═══════════════════════════════════════
   BUILD / DONE
   ═══════════════════════════════════════ */

function buildWizardContext() {
  // Build a CLEAN context string from wizard selections for the generate API.
  //
  // CRITICAL: Only include role-relevant signals. Default pill text like
  // "5G Rollout", "Cloud & SaaS", "AI & Automation" are generic tech defaults
  // that contaminate every profile regardless of role.
  //
  // INCLUDE: category/sub names, deterministic answers (stage, opptype),
  //          user-typed custom items, interest topics
  // EXCLUDE: default pill selections (non-deterministic, tech-biased defaults)
  const parts = [];
  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    if (c.id === 'interests' && interestTopics.length) {
      parts.push(`Interests: ${interestTopics.join(', ')}`);
      continue;
    }
    const allSubs = [...c.subs, ...(customSubs[c.id] || [])];
    const activeSubs = allSubs.filter(s => selSubs[c.id]?.has(s.id));
    if (!activeSubs.length) continue;
    const subDetails = [];
    for (const sub of activeSubs) {
      const panel = PANELS[sub.id];
      if (panel && panelSel[sub.id]) {
        const contextParts = [];
        for (const q of panel.qs) {
          const sel = panelSel[sub.id]?.[q.id];
          if (!sel || !sel.size) continue;
          if (DETERMINISTIC_QS.has(q.id)) {
            // Stage, opptype, clevel, etc. — always include (role-relevant context)
            contextParts.push([...sel].join(', '));
          } else {
            // Non-deterministic: ONLY include user-added custom items, not defaults
            const customs = panelCustom[sub.id]?.[q.id] || [];
            if (customs.length) contextParts.push(customs.join(', '));
          }
        }
        if (contextParts.length) subDetails.push(`${sub.name} (${contextParts.join('; ')})`);
        else subDetails.push(sub.name);
      } else {
        subDetails.push(sub.name);
      }
    }
    parts.push(`${c.name}: ${subDetails.join(', ')}`);
  }
  const result = parts.join('. ');
  console.debug('[Wizard] buildWizardContext:', result);
  return result;
}

function doBuild() {
  const role = document.getElementById('simple-role')?.value?.trim() || 'your profile';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  // Show loading in main area
  const main = document.getElementById('wiz-main');
  if (!main) return;
  main.innerHTML = `<div class="ld" id="wiz-build-ld">
    <div class="wiz-ring"></div>
    <div class="ld-t">Building your feed...</div>
    <div class="ld-s">Generating for <strong>${esc(role)}</strong></div>
    <div class="ld-bar"><div class="ld-bar-fill" id="wiz-bar"></div></div>
    <div class="ld-list">
      <div class="ls on" id="wiz-l0"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Generating context from your choices</span></div>
      <div class="ls" id="wiz-l1"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Building tracking categories</span></div>
      <div class="ls" id="wiz-l2"><div class="ls-d"><div class="ls-sp"></div>${CK}</div><span>Selecting news sources & scoring model</span></div>
    </div>
  </div>`;
  main.scrollTo(0, 0);
  // Disable build button
  const btn = document.getElementById('wiz-build-btn');
  if (btn) btn.disabled = true;
  const wizContext = buildWizardContext();
  callGenerateProfile(role, location, wizContext);
}

async function callGenerateProfile(role, location, context) {
  console.debug('[Wizard] callGenerateProfile:', {role, location, context});
  let stepIdx = 0;
  const bar = document.getElementById('wiz-bar');
  const setBar = (pct) => { if (bar) bar.style.width = pct + '%'; };
  const advanceStep = () => {
    if (stepIdx > 0) { const prev = document.getElementById('wiz-l' + (stepIdx - 1)); if (prev) { prev.classList.remove('on'); prev.classList.add('ok'); } }
    if (stepIdx < 3) { const cur = document.getElementById('wiz-l' + stepIdx); if (cur) cur.classList.add('on'); stepIdx++; }
  };
  advanceStep(); setBar(5);
  try {
    await new Promise(r => setTimeout(r, 600));
    advanceStep(); setBar(15);
    let progressPct = 15;
    const startTime = Date.now();
    const progressTimer = setInterval(() => {
      if (progressPct < 75) { progressPct += (75 - progressPct) * 0.06; setBar(Math.round(progressPct)); }
      const subtitle = document.querySelector('.ld-s');
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      if (subtitle && elapsed > 3) {
        subtitle.innerHTML = 'Generating for <strong>' + esc(role) + '</strong> (' + elapsed + 's)';
      }
    }, 400);
    const resp = await fetch('/api/generate-profile', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: AbortSignal.timeout(180000),
      body: JSON.stringify({role, location, context})
    });
    clearInterval(progressTimer);
    if (!resp.ok) throw new Error('Server error: ' + resp.status);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    _wizGenerateData = data;
    advanceStep(); setBar(80);
    await new Promise(r => setTimeout(r, 500));
    advanceStep(); setBar(100);
    await new Promise(r => setTimeout(r, 400));
    showDone();
  } catch (e) {
    console.error('Wizard generate failed:', e);
    setBar(100);
    _wizGenerateData = null;
    showDone(e.message);
  }
}

function showDone(errorMsg) {
  const el = document.getElementById('wiz-main');
  if (!el) return;
  if (errorMsg) {
    el.innerHTML = '<div class="done">' +
      '<div class="done-c err">' + CK.replace('viewBox', 'width="30" height="30" fill="none" viewBox') + '</div>' +
      '<div class="done-t">Generation had issues</div>' +
      '<div class="done-s">We couldn\'t fully generate your profile: ' + esc(errorMsg) + '.<br>You can still use the wizard selections or try again.</div>' +
      '<div style="display:flex;gap:12px">' +
        '<button class="btn bo" onclick="_wiz.restoreMainView()">Try Again</button>' +
        '<button class="btn bp" onclick="_wiz.finishWizard()" style="padding:14px 36px;font-size:16px">Use Selections Anyway \u2192</button>' +
      '</div></div>';
  } else {
    const catCount = _wizGenerateData?.categories?.length || 0;
    const itemCount = (_wizGenerateData?.categories || []).reduce((a, c) => a + (c.items?.length || 0), 0);
    el.innerHTML = '<div class="done">' +
      '<div class="done-c">' + CK.replace('viewBox', 'width="30" height="30" fill="none" viewBox') + '</div>' +
      '<div class="done-t">Your feed is ready!</div>' +
      '<div class="done-s">Generated ' + catCount + ' categories with ' + itemCount + ' tracking items. Your dashboard will now show signals tailored to your profile.</div>' +
      '<button class="btn bp" onclick="_wiz.finishWizard()" style="padding:14px 36px;font-size:16px">Apply & Close \u2192</button>' +
      '</div>';
  }
}

function restoreMainView() {
  const main = document.getElementById('wiz-main');
  if (!main) return;
  main.innerHTML = '<div id="wiz-step-bar"></div><div id="wiz-priorities"></div><div id="wiz-details"></div><div id="wiz-loading" class="wiz-hidden"></div>';
  renderAll();
}

/* ═══════════════════════════════════════
   FINISH — Connect wizard output to settings
   (Stage 6: will be expanded with real settings integration)
   ═══════════════════════════════════════ */

function _wizExtractTickers() {
  const tickers = [];
  const seen = new Set();
  const panelQMap = { crypto: 'ccoins', commodities: 'comms', forex: 'fpairs', stocks: 'smkt' };
  for (const [panelId, qId] of Object.entries(panelQMap)) {
    const sel = panelSel[panelId]?.[qId];
    if (!sel) continue;
    for (const item of sel) {
      const ticker = WIZ_TICKER_MAP[item];
      if (ticker && !seen.has(ticker)) {
        seen.add(ticker);
        tickers.push(ticker);
      }
    }
  }
  return tickers;
}

async function finishWizard() {
  _wizClearState(); // Clear saved wizard state — user has committed to this config

  // Clear stale category library — wizard generates a complete new profile
  try {
    const libKey = typeof _categoryLibraryKey === 'function' ? _categoryLibraryKey() : 'categoryLibrary';
    localStorage.removeItem(libKey);
  } catch(e) {}
  // Clear previous simpleCategories to prevent load-priority contamination
  try { localStorage.removeItem('simpleCategories'); } catch(e) {}

  if (_wizGenerateData) {
    // Apply AI-generated categories, preserving any pinned (manually-added) ones
    if (_wizGenerateData.categories && Array.isArray(_wizGenerateData.categories)) {
      const pinned = (typeof simpleCategories !== 'undefined' ? simpleCategories : []).filter(c => c.pinned);
      const generated = _wizGenerateData.categories;
      const pinnedLabels = new Set(pinned.map(c => (c.label || c.name || '').toLowerCase()));
      const filtered = generated.filter(c => !pinnedLabels.has((c.label || c.name || '').toLowerCase()));
      if (typeof simpleCategories !== 'undefined') {
        simpleCategories.length = 0;
        [...filtered, ...pinned].forEach(c => simpleCategories.push(c));
      }
    }
    // Apply tickers — merge LLM-generated with wizard panel selections
    if (typeof simpleTickers !== 'undefined') {
      const llmTickers = (_wizGenerateData.tickers && Array.isArray(_wizGenerateData.tickers))
        ? _wizGenerateData.tickers : [];
      const wizTickers = _wizExtractTickers();
      const seen = new Set();
      simpleTickers.length = 0;
      for (const t of [...llmTickers, ...wizTickers]) {
        const upper = t.toUpperCase();
        if (!seen.has(upper)) { seen.add(upper); simpleTickers.push(t); }
      }
    }
    // Apply timelimit
    if (_wizGenerateData.timelimit && typeof simpleTimelimit !== 'undefined') {
      simpleTimelimit = _wizGenerateData.timelimit;
    }
    // Apply context
    if (_wizGenerateData.context && typeof simpleContext !== 'undefined') {
      simpleContext = _wizGenerateData.context;
      const ctxEl = document.getElementById('simple-context');
      if (ctxEl) ctxEl.value = simpleContext;
    }

    // Call settings.js render/sync functions
    if (typeof renderDynamicCategories === 'function') renderDynamicCategories();
    if (typeof renderSimpleTickers === 'function') renderSimpleTickers();
    if (typeof updateTimelimitButtons === 'function') updateTimelimitButtons();
    if (typeof syncToAdvanced === 'function') syncToAdvanced();
    if (typeof saveSimpleState === 'function') saveSimpleState();

    const catCount = _wizGenerateData.categories?.length || 0;

    // Auto-save to backend BEFORE closing wizard — closeWizard removes DOM
    // elements that syncToAdvanced reads, causing a race condition if done after
    if (typeof saveConfig === 'function') {
      window._pendingDynamicCategories = typeof simpleCategories !== 'undefined' ? simpleCategories : null;
      try {
        await saveConfig();
        window._pendingDynamicCategories = null;
        closeWizard();
        if (typeof showToast === 'function') showToast(`Profile saved! ${catCount} categories configured. Run your first scan!`, 'success');
        // Navigate to feed and highlight scan/refresh button
        _postWizardScanGuide();
      } catch(e) {
        window._pendingDynamicCategories = null;
        closeWizard();
        if (typeof showToast === 'function') showToast(`Wizard applied ${catCount} categories but save failed. Click Save to retry.`, 'warning');
      }
    } else {
      closeWizard();
      if (typeof showToast === 'function') showToast(`Wizard applied ${catCount} categories. Click Save to persist.`, 'success');
    }
  } else {
    // No generate data — just close
    closeWizard();
    if (typeof showToast === 'function') showToast('Wizard closed. No categories were generated.', 'info');
  }
}

/* ═══════════════════════════════════════
   SKIP TO QUICK SETUP
   ═══════════════════════════════════════ */

/* DOMAIN_CAT_MAP, DOMAIN_SUB_MAP — now in wizard-data.js */

function applyDomainDefaults() {
  const cats = DOMAIN_CAT_MAP[_currentDomain] || DOMAIN_CAT_MAP.default;
  const subs = DOMAIN_SUB_MAP[_currentDomain] || DOMAIN_SUB_MAP.default;
  selCats = new Set(cats);
  for (const c of CATS) {
    if (selCats.has(c.id) && subs[c.id]) {
      selSubs[c.id] = new Set(subs[c.id].filter(s => c.subs.some(cs => cs.id === s)));
    } else if (!selCats.has(c.id)) {
      selSubs[c.id] = new Set();
    }
  }
}

function applySmartPanelDefaults() {
  const inferredStage = inferStage(document.getElementById('simple-role')?.value?.trim() || '');
  for (const c of CATS) {
    if (!selCats.has(c.id)) continue;
    const allSubs = [...c.subs, ...(customSubs[c.id] || [])];
    for (const sub of allSubs) {
      if (!selSubs[c.id].has(sub.id)) continue;
      const panel = PANELS[sub.id];
      if (!panel) continue;
      if (!panelSel[sub.id]) panelSel[sub.id] = {};
      if (!panelCustom[sub.id]) panelCustom[sub.id] = {};
      for (const q of panel.qs) {
        if (!panelCustom[sub.id][q.id]) panelCustom[sub.id][q.id] = [];
        if (q.id === 'stage') {
          panelSel[sub.id][q.id] = new Set([inferredStage]);
        } else if (q.id === 'opptype') {
          // Cascade: stage affects opportunity type
          if (inferredStage === 'Student') panelSel[sub.id][q.id] = new Set(['Internships']);
          else if (inferredStage === 'Fresh Graduate') panelSel[sub.id][q.id] = new Set(['Full-time Jobs','Internships']);
          else panelSel[sub.id][q.id] = new Set(['Full-time Jobs']);
        } else if (q.type === 's' && q.def) {
          panelSel[sub.id][q.id] = new Set([q.def]);
        } else {
          const pills = getEffectivePills(q.id, q.pills);
          panelSel[sub.id][q.id] = new Set(pills.slice(0, Math.min(3, pills.length)));
        }
      }
    }
  }
}

async function skipToQuick() {
  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';

  // Show loading in main area
  const main = document.getElementById('wiz-main');
  if (!main) return;
  main.innerHTML = '<div class="ld">' +
    '<div class="wiz-ring"></div>' +
    '<div class="ld-t">Analyzing your profile...</div>' +
    '<div class="ld-s">Finding the best setup for <strong>' + esc(role) + '</strong>' + (location ? ' in <strong>' + esc(location) + '</strong>' : '') + '</div>' +
    '<div class="ld-bar"><div class="ld-bar-fill" id="wiz-qbar"></div></div>' +
    '<div class="ld-list">' +
      '<div class="ls on" id="wiz-q0"><div class="ls-d"><div class="ls-sp"></div>' + CK + '</div><span>Analyzing role &amp; location</span></div>' +
      '<div class="ls" id="wiz-q1"><div class="ls-d"><div class="ls-sp"></div>' + CK + '</div><span>Selecting relevant categories</span></div>' +
      '<div class="ls" id="wiz-q2"><div class="ls-d"><div class="ls-sp"></div>' + CK + '</div><span>Preparing your review</span></div>' +
    '</div></div>';
  main.scrollTo(0, 0);

  let qIdx = 0;
  const qbar = document.getElementById('wiz-qbar');
  const setQBar = (pct) => { if (qbar) qbar.style.width = pct + '%'; };
  const advQ = () => {
    if (qIdx > 0) { const prev = document.getElementById('wiz-q' + (qIdx - 1)); if (prev) { prev.classList.remove('on'); prev.classList.add('ok'); } }
    if (qIdx < 3) { const cur = document.getElementById('wiz-q' + qIdx); if (cur) cur.classList.add('on'); qIdx++; }
  };
  advQ(); setQBar(10);

  try {
    await new Promise(r => setTimeout(r, 400));
    advQ(); setQBar(40);
    _currentDomain = classifyRole(role);
    let aiSuccess = false;
    try {
      const available = CATS.map(c => ({ id: c.id, label: c.name, subs: (c.subs || []).map(s => ({id: s.id, label: s.name})) }));
      const resp = await fetch('/api/wizard-preselect', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({role, location, available_categories: available})
      });
      if (resp.ok) {
        const data = await resp.json();
        if (!data.error && data.selected_categories?.length) {
          selCats = new Set(data.selected_categories.filter(id => CATS.some(c => c.id === id)));
          const aiSubs = data.selected_subs || {};
          for (const c of CATS) {
            if (selCats.has(c.id) && aiSubs[c.id]) {
              const validSubIds = new Set(c.subs.map(s => s.id));
              selSubs[c.id] = new Set(aiSubs[c.id].filter(s => validSubIds.has(s)));
            } else if (!selCats.has(c.id)) { selSubs[c.id] = new Set(); }
          }
          aiSuccess = true;
        }
      }
    } catch(e) { /* AI unavailable */ }
    if (!aiSuccess) applyDomainDefaults();
    applySmartPanelDefaults();

    await new Promise(r => setTimeout(r, 300));
    advQ(); setQBar(50);
    await fetchRvItems();
    setQBar(80);
    initRv();

    advQ(); setQBar(100);
    await new Promise(r => setTimeout(r, 400));

    // Restore main view with selections applied
    rvCollapsed.clear();
    restoreMainView();
  } catch (e) {
    console.error('Quick setup failed:', e);
    restoreMainView();
    if (typeof showToast === 'function') showToast('Quick setup had an issue. Please select manually.', 'warning');
  }
}

function discoverAdd(name, target) {
  if (discoverAdded.has(name)) {
    // Toggle off
    discoverAdded.delete(name);
    rvItems[target] = (rvItems[target] || []).filter(v => v !== name);
  } else {
    discoverAdded.add(name);
    if (!rvItems[target]) rvItems[target] = [];
    if (!rvItems[target].includes(name)) rvItems[target].push(name);
  }
  renderRail();
  renderDetails();
}
function discoverRm(name, target) {
  discoverAdded.delete(name);
  rvItems[target] = (rvItems[target] || []).filter(v => v !== name);
  renderRail();
  renderDetails();
}

/* ═══════════════════════════════════════
   UTILITY
   ═══════════════════════════════════════ */

function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function escAttr(s) { return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"'); }


/* ═══════════════════════════════════════
   KEYBOARD SHORTCUTS
   ═══════════════════════════════════════ */

function setupKeyboard() {
  document.addEventListener('keydown', function wizKeyHandler(e) {
    const root = document.getElementById('wiz-root');
    const modal = root?.querySelector('.modal');
    if (!modal || !modal.classList.contains('open')) return;
    const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
    if (inInput) {
      if (e.key === 'Escape') { e.preventDefault(); e.target.blur(); }
      return;
    }
    if (e.key === 'Escape') { e.preventDefault(); if (_presetMenuOpen) { _presetMenuOpen = false; renderPresetBar(); } else closeWizard(); }
    else if (e.key >= '1' && e.key <= '6') {
      e.preventDefault(); const idx = parseInt(e.key) - 1;
      if (idx < CATS.length) togCat(CATS[idx].id);
    }
  });
  document.addEventListener('click', function wizClickOutside(e) {
    if (!_presetMenuOpen) return;
    const dd = document.querySelector('.wiz-scope .preset-dd');
    if (dd && !dd.contains(e.target)) { _presetMenuOpen = false; renderPresetBar(); }
  });
}

/* ═══════════════════════════════════════
   OPEN / CLOSE
   ═══════════════════════════════════════ */

function openWizard(opts) {
  opts = opts || {};
  const role = document.getElementById('simple-role')?.value?.trim();
  const location = document.getElementById('simple-location')?.value?.trim();
  if (!role) {
    if (typeof showToast === 'function') showToast('Enter your role first', 'warning');
    return;
  }
  _wizMode = opts.mode || 'suggest';
  initState();
  const restored = _wizLoadState();
  if (restored) console.log('[Wizard] Restored saved state from localStorage');
  injectCSS();
  injectDOM(role, location);
  _presetMenuOpen = false;
  renderPresetBar();

  requestAnimationFrame(() => {
    const root = document.getElementById('wiz-root');
    if (root) {
      const bk = root.querySelector('.backdrop');
      const modal = root.querySelector('.modal');
      if (bk) bk.classList.add('open');
      if (modal) modal.classList.add('open');
    }
  });
  document.body.style.overflow = 'hidden';

  // Restore saved atmosphere
  restoreAtmosphere();

  // Flash save button on first open
  if (!localStorage.getItem('wiz_opened_before')) {
    localStorage.setItem('wiz_opened_before', '1');
    setTimeout(() => {
      const saveBtn = document.querySelector('.wiz-scope .preset-save-btn-hdr');
      if (saveBtn) saveBtn.classList.add('flash');
    }, 800);
  }

  // Render initial state
  renderAll();

  // Fetch AI pre-selection in background
  if (!restored && role) {
    _currentDomain = classifyRole(role);
    fetchPreselection(role, location);
  }

  // Kick off AI entity fetch in background for rail
  if (!_rvItemsCache) initRvWithAI();
}


/* ═══════════════════════════════════════
   AI PRE-SELECTION
   ═══════════════════════════════════════ */

async function fetchPreselection(role, location) {
  try {
    const available = CATS.map(c => ({
      id: c.id, label: c.name,
      subs: (c.subs || []).map(s => ({id: s.id, label: s.name}))
    }));
    const resp = await fetch('/api/wizard-preselect', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role, location, available_categories: available})
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.error) return;
    const aiCats = data.selected_categories || [];
    const aiSubs = data.selected_subs || {};
    selCats = new Set(aiCats.filter(id => CATS.some(c => c.id === id)));
    for (const c of CATS) {
      if (selCats.has(c.id) && aiSubs[c.id]) {
        const validSubIds = new Set([...c.subs.map(s => s.id)]);
        selSubs[c.id] = new Set(aiSubs[c.id].filter(s => validSubIds.has(s)));
      } else if (!selCats.has(c.id)) {
        selSubs[c.id] = new Set();
      }
    }
    renderAll();
  } catch (e) {
    console.log('Wizard: AI pre-selection unavailable, using defaults');
  }
}

/* ═══════════════════════════════════════
   AI TAB SUGGESTIONS
   ═══════════════════════════════════════ */

async function fetchTabSuggestion(tabId, extraExclude, isRefresh) {
  if (_tabSuggestCache[tabId]) return;
  _tabSuggestCache[tabId] = {suggestions: [], loading: true, added: new Set(), isRefresh: !!isRefresh};

  const role = document.getElementById('simple-role')?.value?.trim() || '';
  const location = document.getElementById('simple-location')?.value?.trim() || '';
  const cat = CATS.find(c => c.id === tabId);
  if (!cat) { _tabSuggestCache[tabId].loading = false; return; }

  const existingItems = [];
  const deterministicParts = [];
  const tab = getActiveTabs().find(t => t.id === tabId);
  const sections = tab ? getTabSections(tab) : [];
  for (const sec of sections) {
    const panel = PANELS[sec.id];
    if (panel && panelSel[sec.id]) {
      for (const q of panel.qs) {
        const picked = panelSel[sec.id]?.[q.id];
        if (picked && picked.size) {
          if (DETERMINISTIC_QS.has(q.id)) { deterministicParts.push(q.label + ': ' + [...picked].join(', ')); }
          else { existingItems.push(...picked); }
        }
      }
    }
  }
  if (extraExclude && extraExclude.size) existingItems.push(...extraExclude);
  const selectionsContext = deterministicParts.join('; ');
  const selections = {};
  for (const sec of sections) {
    const panel = PANELS[sec.id];
    if (panel && panelSel[sec.id]) {
      for (const q of panel.qs) {
        const picked = panelSel[sec.id]?.[q.id];
        if (picked && picked.size) selections[q.label] = [...picked];
      }
    }
  }

  try {
    const abortCtrl = new AbortController();
    _tabSuggestCache[tabId]._abort = abortCtrl;
    const resp = await fetch('/api/wizard-tab-suggest', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      signal: AbortSignal.any([abortCtrl.signal, AbortSignal.timeout(300000)]),
      body: JSON.stringify({ role, location, category_id: tabId, category_label: cat.name,
        existing_items: existingItems, selections_context: selectionsContext, selections,
        exclude_selected: extraExclude ? [...extraExclude] : [], is_refresh: !!isRefresh })
    });
    if (!resp.ok) throw new Error('Request failed');
    const data = await resp.json();
    _tabSuggestCache[tabId] = { suggestions: data.suggestions || [], loading: false, added: new Set() };
  } catch (e) {
    if (e.name === 'AbortError') return;
    _tabSuggestCache[tabId] = {suggestions: [], loading: false, added: new Set()};
    console.log('Wizard: Tab suggestion unavailable for', tabId);
  }
  renderDetails();
  // Chain: fetch next uncached tab (one at a time, no parallel LLM calls)
  if (!isRefresh) {
    const tabs = getActiveTabs();
    for (const tab of tabs) {
      if (tab.id !== 'interests' && !_tabSuggestCache[tab.id]) { fetchTabSuggestion(tab.id); break; }
    }
  }
}

function addTabSuggestion(tabId, keyword) {
  const cache = _tabSuggestCache[tabId];
  if (!cache) return;
  if (!cache.added) cache.added = new Set();
  // Toggle: if already added, remove it
  if (cache.added.has(keyword)) {
    cache.added.delete(keyword);
    if (tabId === 'interests') {
      interestTopics = interestTopics.filter(t => t !== keyword);
    } else {
      if (selSubs[tabId]) selSubs[tabId].delete(keyword);
    }
  } else {
    cache.added.add(keyword);
    if (tabId === 'interests') {
      if (!interestTopics.includes(keyword)) interestTopics.push(keyword);
    } else {
      if (!selSubs[tabId]) selSubs[tabId] = new Set();
      selSubs[tabId].add(keyword);
    }
  }
  renderDetails();
  renderRail();
}

function renderTabSuggestions(tabId) {
  const cache = _tabSuggestCache[tabId];
  if (!cache) return '';
  if (cache.loading) {
    // Shimmer placeholders instead of just a spinner
    const shimmerPills = '<div class="ai-sug-pill shimmer"></div>'.repeat(5);
    return `<div class="ai-sug"><div class="ai-sug-hdr ${cache.isRefresh ? 'flash' : ''}"><div class="ai-sug-spin"></div> Suggesting...<button class="ai-ref spin" disabled style="opacity:.3">&#x21bb;</button></div><div class="acc-pills">${shimmerPills}</div></div>`;
  }
  const added = cache.added || new Set();
  const kept = cache.keptFromPrev || new Set();
  const allSugs = cache.suggestions || [];
  // No suggestions at all and nothing kept? Hide section
  if (!allSugs.length && !kept.size) return '';

  // Build pills: first show kept (previously selected, always "added"), then current suggestions
  let pillsHtml = '';
  for (const s of kept) {
    pillsHtml += `<div class="ai-sug-pill added">\u2713 ${esc(s)}</div>`;
  }
  for (const s of allSugs) {
    const isAdded = added.has(s);
    pillsHtml += `<div class="ai-sug-pill ${isAdded ? 'added' : ''}" onclick="_wiz.addTabSuggestion('${escAttr(tabId)}','${escAttr(s)}')">${isAdded ? '\u2713 ' : '+ '}${esc(s)}${isAdded ? '<span class="sug-x">\u00D7</span>' : ''}</div>`;
  }
  return `<div class="ai-sug"><div class="ai-sug-hdr">\u2728 Suggestions<button class="ai-ref" onclick="_wiz.refreshSuggestions('${escAttr(tabId)}')" title="Get new suggestions">&#x21bb;</button></div><div class="acc-pills">${pillsHtml}</div></div>`;
}

function refreshSuggestions(tabId) {
  const cache = _tabSuggestCache[tabId];
  // Abort any in-flight fetch for this tab
  if (cache?._abort) cache._abort.abort();
  // Collect previously added items to exclude from new suggestions
  const allPreviouslyAdded = new Set();
  if (cache?.added) for (const item of cache.added) allPreviouslyAdded.add(item);
  if (cache?.keptFromPrev) for (const item of cache.keptFromPrev) allPreviouslyAdded.add(item);
  // Add current suggestions to rotation history (keep last 3 batches)
  if (cache?.suggestions?.length) {
    if (!_tabSuggestHistory[tabId]) _tabSuggestHistory[tabId] = [];
    _tabSuggestHistory[tabId].push([...cache.suggestions]);
    if (_tabSuggestHistory[tabId].length > 3) _tabSuggestHistory[tabId].shift();
  }
  // Build full exclude set: user-added items + all items from last 3 batches
  const excludeSet = new Set(allPreviouslyAdded);
  if (_tabSuggestHistory[tabId]) {
    for (const batch of _tabSuggestHistory[tabId]) {
      for (const item of batch) excludeSet.add(item);
    }
  }
  // Clear cache so fetchTabSuggestion proceeds (it early-returns if cache exists)
  delete _tabSuggestCache[tabId];
  // Set loading state immediately and re-render to show spinner
  _tabSuggestCache[tabId] = { suggestions: [], loading: true, added: new Set(), isRefresh: true };
  renderDetails();
  // Clear the temporary loading entry so fetchTabSuggestion can create its own
  delete _tabSuggestCache[tabId];
  fetchTabSuggestion(tabId, excludeSet, true).then(() => {
    const newCache = _tabSuggestCache[tabId];
    if (newCache && allPreviouslyAdded.size) {
      newCache.keptFromPrev = allPreviouslyAdded;
    }
    renderDetails();
  });
}

function closeWizard() {
  _wizSaveState();
  _cleanupWizStars();
  const root = document.getElementById('wiz-root');
  if (root) {
    const bk = root.querySelector('.backdrop');
    const modal = root.querySelector('.modal');
    if (bk) bk.classList.remove('open');
    if (modal) modal.classList.remove('open');
  }
  document.body.style.overflow = '';
  setTimeout(() => { removeDOM(); removeCSS(); }, 400);
}

/* ═══════════════════════════════════════
   GLOBAL EXPORTS (for inline onclick handlers)
   ═══════════════════════════════════════ */

// Post-wizard scan guidance: navigate to feed, pulse the refresh button
function _postWizardScanGuide() {
  setTimeout(() => {
    // Switch to dashboard tab
    if (typeof setActive === 'function') setActive('dashboard');
    // Find and pulse the refresh/scan button
    const refreshBtn = document.querySelector('[onclick*="triggerRefresh"]') ||
                       document.querySelector('[onclick*="refreshAll"]') ||
                       document.querySelector('#refresh-btn') ||
                       document.querySelector('.refresh-trigger');
    if (refreshBtn) {
      refreshBtn.style.animation = 'wizPulseGuide 1.5s ease-in-out 4';
      refreshBtn.style.boxShadow = '0 0 20px var(--accent, #10b981)';
      refreshBtn.title = 'Click to run your first scan!';
      setTimeout(() => {
        refreshBtn.style.animation = '';
        refreshBtn.style.boxShadow = '';
      }, 6000);
    }
    // Inject pulse animation if not present
    if (!document.getElementById('wiz-pulse-guide-style')) {
      const s = document.createElement('style');
      s.id = 'wiz-pulse-guide-style';
      s.textContent = `@keyframes wizPulseGuide { 0%,100% { transform:scale(1); box-shadow:0 0 8px var(--accent,#10b981); } 50% { transform:scale(1.15); box-shadow:0 0 24px var(--accent,#10b981); } }`;
      document.head.appendChild(s);
    }
  }, 600);
}

window._wiz = {
  // Priorities
  togCat, togSub, showAddSub, addSubKey, rmCustomSub,
  // Details
  togPanel, showPanelAdd, addPanelKey, rmPanelCustom,
  togIntS2, rmIntS2, intKeyS2,
  togViewAll,
  togDetSection,
  refreshSuggestions,
  dismissS2Banner,
  // Rail / Review
  togRvCollapse, collapseAllRv, expandAllRv,
  rvRm, rvRmInt, rvShowAdd, rvAddKey,
  discoverAdd, discoverRm, addTabSuggestion,
  toggleRail,
  // Build
  doBuild, finishWizard, restoreMainView,
  // Presets
  savePreset, loadPreset, deletePreset, togglePresetMenu,
  // Other
  skipToQuick, closeWizard, clearAll, setAtmosphere,
};

// Public API
window.openWizard = openWizard;
window.closeWizard = closeWizard;

// Set up keyboard once
setupKeyboard();

/* Button wiring removed — Generate/Suggest use their original
   settings.js implementations directly. Setup Wizard button
   calls openWizard() via onclick in HTML. */

})();
