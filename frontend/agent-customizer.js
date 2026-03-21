// ═══════════════════════════════════════════════════════════
// AGENT FULLSCREEN CUSTOMIZER — Appearance settings panel
// Split from agent.js
// ═══════════════════════════════════════════════════════════

var _fsCustomizerOpen = false;
var _fsCustomDefaults = {
    chatOpacity: 0.7, chatBlur: 16, sidebarOpacity: 0.75, sidebarBlur: 20,
    fontSize: 15, lineHeight: 1.65, bubblePadding: 14, bubbleRadius: 16,
    chatWidth: 1400, inputHeight: 52, sendSize: 44, uiScale: 1, gradBar: true
};
var _fsPresets = {
    Default:   { chatOpacity:0.7, chatBlur:16, sidebarOpacity:0.75, sidebarBlur:20, fontSize:15, lineHeight:1.65, bubblePadding:14, bubbleRadius:16, chatWidth:1400, inputHeight:52, sendSize:44, uiScale:1, gradBar:true },
    Minimal:   { chatOpacity:0.55, chatBlur:12, sidebarOpacity:0.6, sidebarBlur:14, fontSize:13, lineHeight:1.5, bubblePadding:10, bubbleRadius:8, chatWidth:700, inputHeight:42, sendSize:36, uiScale:0.92, gradBar:false },
    Cozy:      { chatOpacity:0.82, chatBlur:22, sidebarOpacity:0.85, sidebarBlur:24, fontSize:17, lineHeight:1.8, bubblePadding:18, bubbleRadius:20, chatWidth:900, inputHeight:58, sendSize:48, uiScale:1.05, gradBar:true },
    Glass:     { chatOpacity:0.3, chatBlur:32, sidebarOpacity:0.25, sidebarBlur:36, fontSize:15, lineHeight:1.65, bubblePadding:14, bubbleRadius:16, chatWidth:800, inputHeight:52, sendSize:44, uiScale:1, gradBar:true },
    Wide:      { chatOpacity:0.7, chatBlur:16, sidebarOpacity:0.75, sidebarBlur:20, fontSize:14, lineHeight:1.55, bubblePadding:12, bubbleRadius:12, chatWidth:1200, inputHeight:48, sendSize:42, uiScale:1, gradBar:true },
    Compact:   { chatOpacity:0.8, chatBlur:10, sidebarOpacity:0.85, sidebarBlur:12, fontSize:13, lineHeight:1.4, bubblePadding:8, bubbleRadius:6, chatWidth:680, inputHeight:38, sendSize:32, uiScale:0.9, gradBar:false },
};

var _fsPresetTips = {
    Default: 'Balanced defaults — good for most setups',
    Minimal: 'Clean and compact — less chrome, more content',
    Cozy: 'Larger text, extra padding — comfortable reading',
    Glass: 'Translucent panels — lets the theme shine through',
    Wide: 'Extra-wide chat area for long conversations',
    Compact: 'Maximum density — fit more on screen',
};

function _fsLoadUserPresets() {
    try { return JSON.parse(localStorage.getItem('stratos-fs-user-presets') || '{}'); } catch(e) { return {}; }
}
function _fsSaveUserPresets(p) { localStorage.setItem('stratos-fs-user-presets', JSON.stringify(p)); }
function _fsAllPresets() { return Object.assign({}, _fsPresets, _fsLoadUserPresets()); }
function _fsPresetActiveCheck(c, name) {
    var all = _fsAllPresets();
    var p = all[name];
    if (!p) return false;
    for (var k of Object.keys(p)) { if (c[k] !== p[k]) return false; }
    return true;
}
function _fsCustApplyPreset(name) {
    var all = _fsAllPresets();
    var p = all[name];
    if (!p) return;
    var c = Object.assign({}, _fsCustomDefaults, p);
    _saveFsCustom(c);
    _applyFsCustom(c);
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; _toggleFsCustomizer(); }
}
async function _fsCustSavePreset() {
    var name = await stratosPrompt({ title: 'Save Preset', message: 'Name your preset:', placeholder: 'My Preset' });
    if (!name || !name.trim()) return;
    var trimmed = name.trim();
    if (_fsPresets[trimmed]) {
        if (typeof showToast === 'function') showToast('Cannot overwrite a built-in preset', 'error');
        return;
    }
    var userPresets = _fsLoadUserPresets();
    if (userPresets[trimmed]) {
        var ok = await stratosConfirm('Overwrite existing preset "' + trimmed + '"?');
        if (!ok) return;
    }
    var c = _loadFsCustom();
    var saved = {};
    for (var k of Object.keys(_fsCustomDefaults)) saved[k] = c[k];
    userPresets[trimmed] = saved;
    _fsSaveUserPresets(userPresets);
    if (typeof showToast === 'function') showToast('Preset "' + trimmed + '" saved');
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; _toggleFsCustomizer(); }
}
async function _fsCustDeletePreset(name) {
    if (_fsPresets[name]) return; // can't delete built-ins
    var ok = await stratosConfirm('Delete preset "' + name + '"?', { danger: true });
    if (!ok) return;
    var userPresets = _fsLoadUserPresets();
    delete userPresets[name];
    _fsSaveUserPresets(userPresets);
    if (typeof showToast === 'function') showToast('Preset "' + name + '" deleted');
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; _toggleFsCustomizer(); }
}

function _loadFsCustom() {
    try { return Object.assign({}, _fsCustomDefaults, JSON.parse(localStorage.getItem('stratos-fs-custom') || '{}')); }
    catch(e) { return Object.assign({}, _fsCustomDefaults); }
}
function _saveFsCustom(c) { localStorage.setItem('stratos-fs-custom', JSON.stringify(c)); }

function _applyFsCustom(c) {
    var panel = document.querySelector('.agent-fullscreen');
    if (!panel) return;
    var inner = panel.querySelector('[data-agent-inner]');
    var sidebar = panel.querySelector('.agent-fs-sidebar');
    if (inner) {
        inner.style.background = 'rgba(8,8,26,' + c.chatOpacity + ')';
        inner.style.backdropFilter = 'blur(' + c.chatBlur + 'px)';
        inner.style.webkitBackdropFilter = 'blur(' + c.chatBlur + 'px)';
        inner.style.maxWidth = c.chatWidth + 'px';
    }
    if (sidebar) {
        sidebar.style.setProperty('background', 'rgba(8,8,26,' + c.sidebarOpacity + ')', 'important');
        sidebar.style.setProperty('backdrop-filter', 'blur(' + c.sidebarBlur + 'px)', 'important');
        sidebar.style.setProperty('-webkit-backdrop-filter', 'blur(' + c.sidebarBlur + 'px)', 'important');
    }
    // Use CSS custom properties so they apply to all current + future elements
    panel.style.setProperty('--fs-font-size', c.fontSize + 'px');
    panel.style.setProperty('--fs-line-height', String(c.lineHeight));
    panel.style.setProperty('--fs-bubble-padding', c.bubblePadding + 'px ' + Math.round(c.bubblePadding * 1.3) + 'px');
    panel.style.setProperty('--fs-bubble-radius', c.bubbleRadius + 'px');
    panel.style.setProperty('--fs-input-height', c.inputHeight + 'px');
    panel.style.setProperty('--fs-send-size', c.sendSize + 'px');
    panel.style.setProperty('--fs-ui-scale', String(c.uiScale));
    panel.style.setProperty('--fs-grad-display', c.gradBar ? 'block' : 'none');
}

function _toggleFsCustomizer() {
    _fsCustomizerOpen = !_fsCustomizerOpen;
    var p = document.getElementById('agent-fs-customizer');
    if (_fsCustomizerOpen) {
        if (p) p.remove();
        var c = _loadFsCustom();
        var themes = ['midnight','coffee','rose','noir','aurora','cosmos','sakura','nebula'];
        var curTheme = document.documentElement.getAttribute('data-theme') || 'midnight';
        var themeColors = {midnight:'#34d399',coffee:'#fbbf24',rose:'#fb7185',noir:'#a78bfa',aurora:'#34d399',cosmos:'#e8b931',sakura:'#f0a0b8',nebula:'#38bdf8'};
        var themeBtns = themes.map(function(t) {
            return '<button onclick="_fsCustSetTheme(\'' + t + '\')" class="px-2 py-1 rounded text-[10px] font-bold transition-all" style="color:' + (t===curTheme?themeColors[t]:'var(--text-muted)') + ';border:1px solid ' + (t===curTheme?themeColors[t]+'60':'var(--border-strong)') + ';background:' + (t===curTheme?themeColors[t]+'15':'transparent') + ';" onmouseenter="this.style.background=\'rgba(255,255,255,0.05)\'" onmouseleave="this.style.background=\'' + (t===curTheme?themeColors[t]+'15':'transparent') + '\'">' + t.charAt(0).toUpperCase()+t.slice(1) + '</button>';
        }).join('');

        // Build presets HTML
        var allP = _fsAllPresets();
        var userP = _fsLoadUserPresets();
        var presetBtns = Object.keys(allP).map(function(name) {
            var active = _fsPresetActiveCheck(c, name);
            var isUser = !!userP[name];
            var tip = _fsPresetTips[name] || (isUser ? 'Your saved preset — right-click to delete' : '');
            return '<button onclick="_fsCustApplyPreset(\'' + name.replace(/'/g, "\\'") + '\')" ' + (isUser ? 'oncontextmenu="event.preventDefault();_fsCustDeletePreset(\'' + name.replace(/'/g, "\\'") + '\')"' : '') + ' class="px-2 py-1 rounded text-[10px] font-bold transition-all strat-tip" data-tip="' + tip + '" style="color:' + (active?'var(--accent)':'var(--text-muted)') + ';border:1px solid ' + (active?'rgba(var(--accent-rgb,52,211,153),0.4)':'var(--border-strong)') + ';background:' + (active?'rgba(var(--accent-rgb,52,211,153),0.12)':'transparent') + ';' + (isUser?'font-style:italic;':'') + '" onmouseenter="this.style.background=\'rgba(255,255,255,0.05)\'" onmouseleave="this.style.background=\'' + (active?'rgba(var(--accent-rgb,52,211,153),0.12)':'transparent') + '\'">' + (isUser ? '&#9670; ' : '') + name + '</button>';
        }).join('');

        var html = '\
        <div id="agent-fs-customizer" class="agent-fs-customizer" style="position:absolute;top:0;left:280px;width:300px;height:100%;background:rgba(8,8,26,0.92);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border-left:1px solid rgba(255,255,255,0.06);z-index:20;display:flex;flex-direction:column;animation:slideInLeft 0.2s ease;">\
            <div style="padding:16px 16px 8px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.06);flex-shrink:0;">\
                <div style="display:flex;align-items:center;gap:8px;">\
                    <i data-lucide="palette" class="w-4 h-4" style="color:var(--accent)"></i>\
                    <span class="text-[13px] font-bold" style="color:var(--text-heading);">Customize</span>\
                </div>\
                <div style="display:flex;align-items:center;gap:6px;">\
                    <button onclick="_fsCustReset()" class="strat-tip text-[10px] px-2 py-1 rounded transition-all" style="color:var(--text-muted);border:1px solid var(--border-strong);" data-tip="Restore all settings to their default values" onmouseenter="this.style.color=\'var(--accent)\'" onmouseleave="this.style.color=\'var(--text-muted)\'">Reset</button>\
                    <button onclick="_toggleFsCustomizer()" class="p-1 rounded transition-all strat-tip" style="color:var(--text-muted);" data-tip="Close customizer" onmouseenter="this.style.color=\'var(--text-heading)\'" onmouseleave="this.style.color=\'var(--text-muted)\'">&times;</button>\
                </div>\
            </div>\
            <div style="padding:12px 16px;flex:1;overflow-y:auto;" class="fs-cust-body">\
                <!-- Theme -->\
                <div class="fs-cust-group">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Switch the base color theme for the entire app">Theme</div>\
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">' + themeBtns + '</div>\
                </div>\
                <!-- Presets -->\
                <div class="fs-cust-group" style="margin-top:14px;">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);display:flex;align-items:center;justify-content:space-between;" data-tip="Quick presets that configure all settings at once. Right-click a saved preset to delete it">\
                        Presets\
                        <button onclick="_fsCustSavePreset()" class="strat-tip" data-tip="Save current settings as a new preset" style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;cursor:pointer;border:1px solid var(--border-strong);color:var(--text-muted);background:transparent;text-transform:none;letter-spacing:normal;" onmouseenter="this.style.color=\'var(--accent)\';this.style.borderColor=\'rgba(var(--accent-rgb,52,211,153),0.4)\'" onmouseleave="this.style.color=\'var(--text-muted)\';this.style.borderColor=\'var(--border-strong)\'">+ Save</button>\
                    </div>\
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">' + presetBtns + '</div>\
                </div>\
                <!-- Chat Area -->\
                <div class="fs-cust-group" style="margin-top:14px;">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Control the chat panel background transparency and blur">Chat Area</div>\
                    <label class="fs-cust-label strat-tip" data-tip="How transparent the chat background is. Lower values let more of the theme show through">Opacity</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-chat-opacity" min="0" max="1" step="0.05" value="' + c.chatOpacity + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-chat-opacity-val">' + Math.round(c.chatOpacity*100) + '%</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Frosted glass blur intensity behind the chat panel">Glass Blur</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-chat-blur" min="0" max="40" step="1" value="' + c.chatBlur + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-chat-blur-val">' + c.chatBlur + 'px</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Maximum width of the conversation area. Wider = more horizontal space for messages">Max Width</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-chat-width" min="600" max="1400" step="20" value="' + c.chatWidth + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-chat-width-val">' + c.chatWidth + 'px</span>\
                    </div>\
                </div>\
                <!-- Sidebar -->\
                <div class="fs-cust-group" style="margin-top:14px;">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Control the sidebar panel transparency and blur">Sidebar</div>\
                    <label class="fs-cust-label strat-tip" data-tip="How transparent the sidebar background is">Opacity</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-sidebar-opacity" min="0" max="1" step="0.05" value="' + c.sidebarOpacity + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-sidebar-opacity-val">' + Math.round(c.sidebarOpacity*100) + '%</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Frosted glass blur intensity behind the sidebar">Glass Blur</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-sidebar-blur" min="0" max="40" step="1" value="' + c.sidebarBlur + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-sidebar-blur-val">' + c.sidebarBlur + 'px</span>\
                    </div>\
                </div>\
                <!-- Typography -->\
                <div class="fs-cust-group" style="margin-top:14px;">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Adjust text sizing for messages, input, and all UI elements">Typography</div>\
                    <label class="fs-cust-label strat-tip" data-tip="Scale everything — sidebar, messages, buttons, input, all text and UI. This zooms the entire fullscreen view">UI Scale</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-ui-scale" min="0.85" max="1.3" step="0.01" value="' + c.uiScale + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-ui-scale-val">' + Math.round(c.uiScale*100) + '%</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Base font size for message bubbles, input field, welcome text, and option buttons">Message Font</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-font-size" min="12" max="24" step="1" value="' + c.fontSize + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-font-size-val">' + c.fontSize + 'px</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Space between lines of text in messages. Higher values make text more spread out and easier to read">Line Height</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-line-height" min="1.2" max="2.2" step="0.05" value="' + c.lineHeight + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-line-height-val">' + c.lineHeight.toFixed(2) + '</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Height of the text input box at the bottom of the chat">Input Height</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-input-height" min="36" max="80" step="2" value="' + c.inputHeight + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-input-height-val">' + c.inputHeight + 'px</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Size of the send button next to the input field">Send Button</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-send-size" min="30" max="60" step="2" value="' + c.sendSize + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-send-size-val">' + c.sendSize + 'px</span>\
                    </div>\
                </div>\
                <!-- Bubbles -->\
                <div class="fs-cust-group" style="margin-top:14px;">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Adjust the shape and spacing of chat message bubbles">Bubbles</div>\
                    <label class="fs-cust-label strat-tip" data-tip="How rounded the corners of message bubbles are. 0 = sharp squares, 28 = pill-shaped">Corner Radius</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-bubble-radius" min="0" max="28" step="1" value="' + c.bubbleRadius + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-bubble-radius-val">' + c.bubbleRadius + 'px</span>\
                    </div>\
                    <label class="fs-cust-label strat-tip" data-tip="Internal spacing inside each message bubble. More padding = more breathing room around text">Padding</label>\
                    <div class="fs-cust-row">\
                        <input type="range" class="fs-cust-slider" id="fsc-bubble-padding" min="6" max="28" step="1" value="' + c.bubblePadding + '" oninput="_fsCustUpdate()">\
                        <span class="fs-cust-val" id="fsc-bubble-padding-val">' + c.bubblePadding + 'px</span>\
                    </div>\
                </div>\
                <!-- Effects -->\
                <div class="fs-cust-group" style="margin-top:14px;">\
                    <div class="text-[10px] font-bold uppercase tracking-wider mb-2 strat-tip" style="color:var(--text-muted);" data-tip="Visual effects and decorative elements">Effects</div>\
                    <label class="fs-cust-label strat-tip" style="display:flex;align-items:center;gap:8px;cursor:pointer;" data-tip="Animated accent-colored gradient line at the top of the fullscreen view">\
                        <input type="checkbox" id="fsc-grad-bar" ' + (c.gradBar?'checked':'') + ' onchange="_fsCustUpdate()" style="accent-color:var(--accent);">\
                        Gradient accent bar\
                    </label>\
                </div>\
            </div>\
        </div>';
        var wrapper = document.querySelector('.agent-fs-wrapper');
        if (wrapper) {
            wrapper.insertAdjacentHTML('afterbegin', html);
            lucide.createIcons();
            _initFsCustTooltips();
        }
    } else {
        if (p) p.remove();
        var tt = document.getElementById('fsc-tooltip');
        if (tt) tt.remove();
    }
}

// Floating tooltip for customizer (avoids overflow clipping)
function _initFsCustTooltips() {
    var tt = document.getElementById('fsc-tooltip');
    if (!tt) {
        tt = document.createElement('div');
        tt.id = 'fsc-tooltip';
        tt.style.cssText = 'position:fixed;z-index:999999;pointer-events:none;opacity:0;transition:opacity 0.15s;background:rgba(2,6,18,0.97);border:1px solid rgba(var(--accent-rgb,52,211,153),0.25);box-shadow:0 4px 20px rgba(0,0,0,0.6);font-size:11px;padding:8px 12px;border-radius:8px;color:#e2e8f0;line-height:1.45;max-width:230px;white-space:normal;';
        document.body.appendChild(tt);
    }
    var cust = document.getElementById('agent-fs-customizer');
    if (!cust) return;
    cust.addEventListener('mouseover', function(e) {
        var tip = e.target.closest('[data-tip]');
        if (!tip) { tt.style.opacity = '0'; return; }
        tt.textContent = tip.dataset.tip;
        tt.style.opacity = '1';
        var r = tip.getBoundingClientRect();
        // Position to the right of the customizer panel
        var custRect = cust.getBoundingClientRect();
        var left = custRect.right + 8;
        var top = r.top + r.height / 2;
        // If it would overflow the viewport right, position to the left instead
        if (left + 240 > window.innerWidth) left = custRect.left - 240;
        tt.style.left = left + 'px';
        tt.style.top = top + 'px';
        tt.style.transform = 'translateY(-50%)';
    });
    cust.addEventListener('mouseout', function(e) {
        if (!e.relatedTarget || !cust.contains(e.relatedTarget)) tt.style.opacity = '0';
    });
    cust.addEventListener('mousemove', function(e) {
        var tip = e.target.closest('[data-tip]');
        if (!tip) { tt.style.opacity = '0'; return; }
        if (tt.textContent !== tip.dataset.tip) {
            tt.textContent = tip.dataset.tip;
            var r = tip.getBoundingClientRect();
            var custRect = cust.getBoundingClientRect();
            var left = custRect.right + 8;
            if (left + 240 > window.innerWidth) left = custRect.left - 240;
            tt.style.left = left + 'px';
            tt.style.top = (r.top + r.height / 2) + 'px';
        }
    });
}

function _fsCustUpdate() {
    var c = _loadFsCustom();
    var get = function(id) { return document.getElementById(id); };
    c.chatOpacity = parseFloat(get('fsc-chat-opacity')?.value ?? c.chatOpacity);
    c.chatBlur = parseInt(get('fsc-chat-blur')?.value ?? c.chatBlur);
    c.chatWidth = parseInt(get('fsc-chat-width')?.value ?? c.chatWidth);
    c.sidebarOpacity = parseFloat(get('fsc-sidebar-opacity')?.value ?? c.sidebarOpacity);
    c.sidebarBlur = parseInt(get('fsc-sidebar-blur')?.value ?? c.sidebarBlur);
    c.fontSize = parseInt(get('fsc-font-size')?.value ?? c.fontSize);
    c.lineHeight = parseFloat(get('fsc-line-height')?.value ?? c.lineHeight);
    c.bubblePadding = parseInt(get('fsc-bubble-padding')?.value ?? c.bubblePadding);
    c.bubbleRadius = parseInt(get('fsc-bubble-radius')?.value ?? c.bubbleRadius);
    c.inputHeight = parseInt(get('fsc-input-height')?.value ?? c.inputHeight);
    c.sendSize = parseInt(get('fsc-send-size')?.value ?? c.sendSize);
    c.uiScale = parseFloat(get('fsc-ui-scale')?.value ?? c.uiScale);
    c.gradBar = get('fsc-grad-bar')?.checked ?? c.gradBar;
    // Update labels
    var setVal = function(id, v) { var el = get(id); if (el) el.textContent = v; };
    setVal('fsc-chat-opacity-val', Math.round(c.chatOpacity * 100) + '%');
    setVal('fsc-chat-blur-val', c.chatBlur + 'px');
    setVal('fsc-chat-width-val', c.chatWidth + 'px');
    setVal('fsc-sidebar-opacity-val', Math.round(c.sidebarOpacity * 100) + '%');
    setVal('fsc-sidebar-blur-val', c.sidebarBlur + 'px');
    setVal('fsc-font-size-val', c.fontSize + 'px');
    setVal('fsc-line-height-val', c.lineHeight.toFixed(2));
    setVal('fsc-bubble-padding-val', c.bubblePadding + 'px');
    setVal('fsc-bubble-radius-val', c.bubbleRadius + 'px');
    setVal('fsc-input-height-val', c.inputHeight + 'px');
    setVal('fsc-send-size-val', c.sendSize + 'px');
    setVal('fsc-ui-scale-val', Math.round(c.uiScale * 100) + '%');
    _saveFsCustom(c);
    _applyFsCustom(c);
}

function _fsCustReset() {
    localStorage.removeItem('stratos-fs-custom');
    _applyFsCustom(_fsCustomDefaults);
    // Refresh customizer panel if open
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; _toggleFsCustomizer(); }
}

function _fsCustSetTheme(t) {
    if (typeof setTheme === 'function') setTheme(t);
    // Refresh customizer to update active theme highlight
    if (_fsCustomizerOpen) { _fsCustomizerOpen = false; setTimeout(function() { _toggleFsCustomizer(); }, 100); }
}
