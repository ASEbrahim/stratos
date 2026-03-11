/**
 * mobile.js — StratOS Mobile Gestures, Touch Interactions & Fullscreen Charts
 * Touch features (swipe, bottom nav, pull-to-refresh) only activate on touch devices.
 * Fullscreen chart mode works on ALL devices (desktop + mobile).
 */
(function() {
'use strict';

/* ── Utility guards ── */
const isTouchDevice = () => 'ontouchstart' in window || navigator.maxTouchPoints > 0;
const isMobile = () => window.innerWidth <= 1024;
const isSmall  = () => window.innerWidth <= 768;

/* ── Init: fullscreen charts for ALL devices, touch features for touch only ── */
document.addEventListener('DOMContentLoaded', () => {
    /* initChartFullscreen() moved to fullscreen-chart.js */
    if (!isTouchDevice()) return;
    initSidebarSwipe();
    initCardSwipe();
    initPullToRefresh();
    initBottomNav();
    initPWAInstall();
});

/* Re-evaluate bottom nav on orientation change / resize (touch only) */
let _lastW = window.innerWidth;
if (isTouchDevice()) {
    window.addEventListener('resize', () => {
        if (Math.abs(window.innerWidth - _lastW) > 50) {
            _lastW = window.innerWidth;
            _updateBottomNavVis();
            _updateBottomNav();
        }
    });
}

/* ═══════════════════════════════════════════════
   A.  SWIPE SIDEBAR DRAWER
   ═══════════════════════════════════════════════ */

function initSidebarSwipe() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    const EDGE  = 28;       // px from left edge to start open gesture
    const THRESHOLD = 60;   // px to commit
    const VELO  = 0.35;     // px/ms for flick

    let tracking = false, startX = 0, startY = 0, curX = 0, startT = 0;
    let sideW = 224;

    /* Backdrop — created once */
    let bd = document.getElementById('sidebar-backdrop');
    if (!bd) {
        bd = document.createElement('div');
        bd.id = 'sidebar-backdrop';
        bd.className = 'sidebar-backdrop';
        sidebar.parentElement.insertBefore(bd, sidebar.nextSibling);
        bd.addEventListener('click', () => {
            if (!sidebarCollapsed) toggleSidebar();
            _hideBd();
        });
    }

    /* Show backdrop + hide bottom nav when sidebar opens on mobile */
    const _origApply = window.applySidebarState;
    window.applySidebarState = function() {
        _origApply.apply(this, arguments);
        if (isSmall()) {
            var bnav = document.getElementById('bottom-nav');
            if (!sidebarCollapsed) {
                _showBd(0.5);
                if (bnav) bnav.style.transform = 'translateY(100%)';
            } else {
                _hideBd();
                if (bnav) bnav.style.transform = '';
            }
        }
    };

    document.addEventListener('touchstart', e => {
        if (!isMobile()) return;
        const t = e.touches[0];
        startX = t.clientX; startY = t.clientY; curX = startX;
        startT = Date.now();
        sideW = parseInt(localStorage.getItem('sidebarWidth')) || 224;

        if (sidebarCollapsed && startX <= EDGE) {
            tracking = true;
            /* Prepare sidebar for drag — make it visible but zero-width */
            sidebar.classList.remove('sidebar-collapsed');
            sidebar.style.transition = 'none';
            sidebar.style.width = '0px';
            sidebar.style.minWidth = '0';
            sidebar.style.overflow = 'hidden';
        } else if (!sidebarCollapsed) {
            const r = sidebar.getBoundingClientRect();
            if (startX > r.right || startX >= r.left) tracking = true;
        }
    }, { passive: true });

    document.addEventListener('touchmove', e => {
        if (!tracking) return;
        const t = e.touches[0];
        curX = t.clientX;
        const dx = curX - startX;
        const dy = t.clientY - startY;

        /* Vertical dominant → abort */
        if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 12) {
            tracking = false;
            _resetSidebar(); return;
        }
        if (Math.abs(dx) > 10) e.preventDefault();

        if (sidebarCollapsed) {
            const p = Math.max(0, Math.min(1, dx / sideW));
            sidebar.style.width = (p * sideW) + 'px';
            _showBd(p * 0.5);
        } else {
            const p = Math.max(0, Math.min(1, -dx / sideW));
            sidebar.style.width = ((1 - p) * sideW) + 'px';
            _showBd((1 - p) * 0.5);
        }
    }, { passive: false });

    document.addEventListener('touchend', () => {
        if (!tracking) return;
        tracking = false;
        const dx = curX - startX;
        const v  = Math.abs(dx) / (Date.now() - startT);
        sidebar.style.transition = '';

        if (sidebarCollapsed) {
            if (dx > THRESHOLD || (dx > 20 && v > VELO)) {
                /* Commit open */
                sidebarCollapsed = false;
                localStorage.setItem('sidebarCollapsed', false);
                applySidebarState();
            } else {
                sidebarCollapsed = true;
                applySidebarState();
                _hideBd();
            }
        } else {
            if (-dx > THRESHOLD || (-dx > 20 && v > VELO)) {
                toggleSidebar();
                _hideBd();
            } else {
                applySidebarState();
            }
        }
    }, { passive: true });

    function _resetSidebar() {
        sidebar.style.transition = '';
        applySidebarState();
        if (sidebarCollapsed) _hideBd();
    }
    function _showBd(op) {
        if (!isSmall()) return;
        bd.style.display = 'block';
        bd.style.opacity  = op;
    }
    function _hideBd() {
        bd.style.opacity = '0';
        setTimeout(() => bd.style.display = 'none', 200);
    }
}

/* ═══════════════════════════════════════════════
   B.  CARD SWIPE ACTIONS (Save / Dismiss)
   ═══════════════════════════════════════════════ */

function initCardSwipe() {
    const feed = document.getElementById('news-feed');
    if (!feed) return;

    const THRESHOLD = 85;
    let card = null, startX = 0, startY = 0, dx = 0, tracking = false;

    feed.addEventListener('touchstart', e => {
        const c = e.target.closest('[data-card-idx]');
        if (!c) return;
        if (e.target.closest('button, a, input, select, textarea')) return;
        card = c; startX = e.touches[0].clientX; startY = e.touches[0].clientY;
        dx = 0; tracking = true;
        card.style.transition = 'none';
    }, { passive: true });

    feed.addEventListener('touchmove', e => {
        if (!tracking || !card) return;
        const cx = e.touches[0].clientX;
        const dy = e.touches[0].clientY - startY;
        dx = cx - startX;

        if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 15) {
            tracking = false; _snap(card); return;
        }
        if (Math.abs(dx) > 10) e.preventDefault();

        const resist = 0.55;
        const cap = Math.sign(dx) * Math.min(Math.abs(dx) * resist, 160);
        card.style.transform = `translateX(${cap}px)`;
        _indicator(card, cap);
    }, { passive: false });

    feed.addEventListener('touchend', () => {
        if (!tracking || !card) return;
        tracking = false;
        const c = card, idx = parseInt(c.dataset.cardIdx);

        if (dx > THRESHOLD) {
            c.style.transition = 'transform .3s ease, opacity .3s ease';
            c.style.transform = 'translateX(100%)'; c.style.opacity = '0.3';
            setTimeout(() => {
                if (typeof toggleSaveSignal === 'function') toggleSaveSignal(idx);
                _snap(c); _clearInd(c);
                if (typeof showToast === 'function') showToast('Signal saved', 'success');
            }, 280);
        } else if (dx < -THRESHOLD) {
            c.style.transition = 'transform .3s ease, opacity .3s ease';
            c.style.transform = 'translateX(-100%)'; c.style.opacity = '0';
            setTimeout(() => {
                if (typeof dismissSignal === 'function') dismissSignal(idx);
                _clearInd(c);
            }, 280);
        } else {
            _snap(c); _clearInd(c);
        }
        card = null;
    }, { passive: true });

    function _snap(c) {
        c.style.transition = 'transform .2s ease';
        c.style.transform = '';  c.style.opacity = '';
        setTimeout(() => c.style.transition = '', 200);
    }

    function _indicator(c, dx) {
        let ind = c.querySelector('.card-swipe-ind');
        if (!ind) {
            ind = document.createElement('div');
            ind.className = 'card-swipe-ind';
            ind.innerHTML = '<span class="csi-icon"></span>';
            c.style.position = 'relative';
            c.appendChild(ind);
        }
        const icon = ind.querySelector('.csi-icon');
        if (dx > 30) { ind.className = 'card-swipe-ind csi-save'; icon.textContent = '★ Save'; }
        else if (dx < -30) { ind.className = 'card-swipe-ind csi-dismiss'; icon.textContent = '✕ Dismiss'; }
        else ind.className = 'card-swipe-ind';
    }
    function _clearInd(c) { const i = c.querySelector('.card-swipe-ind'); if (i) i.remove(); }
}

/* ═══════════════════════════════════════════════
   C.  PULL-TO-REFRESH
   ═══════════════════════════════════════════════ */

function initPullToRefresh() {
    const main = document.querySelector('main');
    if (!main) return;

    const THRESHOLD = 70;
    let startY = 0, pulling = false, dist = 0;

    let ind = document.getElementById('ptr-indicator');
    if (!ind) {
        ind = document.createElement('div');
        ind.id = 'ptr-indicator';
        ind.className = 'ptr-indicator';
        ind.innerHTML = '<div class="ptr-spinner"></div><span class="ptr-text">Pull to refresh</span>';
        main.insertBefore(ind, main.firstChild);
    }

    main.addEventListener('touchstart', e => {
        if (main.scrollTop > 5) return;
        startY = e.touches[0].clientY;
        pulling = true; dist = 0;
    }, { passive: true });

    main.addEventListener('touchmove', e => {
        if (!pulling) return;
        const dy = e.touches[0].clientY - startY;
        if (dy < 0 || main.scrollTop > 0) { pulling = false; return; }

        dist = Math.min(dy * 0.45, 110);
        if (dist > 10) {
            e.preventDefault();
            ind.style.height = dist + 'px';
            ind.style.opacity = Math.min(1, dist / THRESHOLD);
            ind.querySelector('.ptr-spinner').style.transform = `rotate(${dist * 3}deg)`;
            ind.querySelector('.ptr-text').textContent = dist >= THRESHOLD ? 'Release to refresh' : 'Pull to refresh';
        }
    }, { passive: false });

    main.addEventListener('touchend', () => {
        if (!pulling) return;
        pulling = false;

        if (dist >= THRESHOLD) {
            ind.style.height = '44px';
            ind.querySelector('.ptr-text').textContent = 'Refreshing...';
            ind.querySelector('.ptr-spinner').classList.add('spinning');
            const fn = typeof toggleScan === 'function' ? toggleScan : null;
            if (fn) {
                Promise.resolve(fn()).finally(() => _collapse());
            } else _collapse();
        } else _collapse();
    }, { passive: true });

    function _collapse() {
        ind.style.transition = 'height .3s ease, opacity .3s ease';
        ind.style.height = '0'; ind.style.opacity = '0';
        ind.querySelector('.ptr-spinner').classList.remove('spinning');
        setTimeout(() => ind.style.transition = '', 300);
    }
}

/* ═══════════════════════════════════════════════
   D.  BOTTOM NAVIGATION BAR
   ═══════════════════════════════════════════════ */

const _btabs = [
    { id: 'dashboard',    icon: 'layout-dashboard', label: 'Home' },
    { id: 'markets_view', icon: 'trending-up',      label: 'Markets' },
    { id: '__agent__',    icon: 'bot',               label: 'Agent' },
    { id: 'saved',        icon: 'bookmark',          label: 'Saved' },
    { id: 'settings',     icon: 'settings',          label: 'Settings' },
];

let _mobileAgentOpen = false;

let _bnavBuilt = false;

function initBottomNav() {
    if (document.getElementById('bottom-nav')) return;
    const nav = document.createElement('nav');
    nav.id = 'bottom-nav';
    nav.className = 'bottom-nav';
    document.body.appendChild(nav);
    _buildBottomNav();
    _syncBottomNavHighlight();
    _updateBottomNavVis();

    /* Hook into setActive so bottom nav stays in sync + close mobile agent */
    const _origSet = window.setActive;
    window.setActive = function(id) {
        if (_mobileAgentOpen) {
            _mobileAgentOpen = false;
            var v = document.getElementById('mobile-agent-view');
            if (v) v.remove();
            window.removeEventListener('popstate', _onAgentPop);
        }
        _origSet.apply(this, arguments);
        /* Synchronous class toggle — no setTimeout needed */
        _syncBottomNavHighlight();
    };
}

function _getActiveTab() {
    return typeof activeRoot !== 'undefined' ? activeRoot : 'dashboard';
}

function _isHomeTab(id) {
    /* Only 'dashboard' (or empty) counts as Home — dynamic categories don't */
    return id === 'dashboard' || !id;
}

/**
 * Build bottom nav HTML once. Call lucide.createIcons() once.
 * Subsequent updates only toggle CSS classes.
 */
function _buildBottomNav() {
    const nav = document.getElementById('bottom-nav');
    if (!nav) return;

    nav.innerHTML = _btabs.map(t => {
        let oc;
        if (t.id === '__agent__') {
            oc = '_toggleMobileAgent()';
        } else {
            oc = "setActive('" + t.id + "')";
        }
        return '<button onclick="' + oc + '" class="bnav-item" data-tab="' + t.id + '">' +
            '<i data-lucide="' + t.icon + '"></i><span>' + t.label + '</span></button>';
    }).join('');

    if (typeof lucide !== 'undefined') lucide.createIcons();
    _bnavBuilt = true;
}

/**
 * Toggle .active class on existing buttons — no HTML rebuild, no icon clobber.
 */
function _syncBottomNavHighlight() {
    const nav = document.getElementById('bottom-nav');
    if (!nav) return;
    /* If nav was never built (shouldn't happen), build it */
    if (!_bnavBuilt) { _buildBottomNav(); }

    const cur = _getActiveTab();
    const buttons = nav.querySelectorAll('.bnav-item');
    buttons.forEach(function(btn) {
        const tabId = btn.getAttribute('data-tab');
        let isActive = false;
        if (tabId === '__agent__') {
            isActive = _mobileAgentOpen;
        } else if (tabId === 'dashboard') {
            isActive = !_mobileAgentOpen && _isHomeTab(cur);
        } else {
            isActive = !_mobileAgentOpen && tabId === cur;
        }
        if (isActive) btn.classList.add('active');
        else btn.classList.remove('active');
    });
}

/* Keep backward compat: _updateBottomNav now just syncs highlight */
function _updateBottomNav() { _syncBottomNavHighlight(); }

function _updateBottomNavVis() {
    const nav = document.getElementById('bottom-nav');
    if (nav) nav.style.display = isSmall() ? 'flex' : 'none';
}

/* ═══════════════════════════════════════════════
   D2. MOBILE AGENT FULL-PAGE VIEW
   ═══════════════════════════════════════════════ */

/* Expose globally so bottom nav onclick can call it */
window._toggleMobileAgent = function() {
    if (_mobileAgentOpen) _closeMobileAgent();
    else _openMobileAgent();
};

function _openMobileAgent() {
    if (document.getElementById('mobile-agent-view')) return;
    _mobileAgentOpen = true;

    /* Make sure the real agent panel exists and agent body is open */
    if (typeof showAgentPanel === 'function') showAgentPanel(true);
    if (typeof agentOpen !== 'undefined' && !agentOpen && typeof toggleAgentChat === 'function') {
        toggleAgentChat();
    }

    const view = document.createElement('div');
    view.id = 'mobile-agent-view';
    const personaRow = typeof _mavBuildPersonaRow === 'function' ? _mavBuildPersonaRow() : '';
    view.innerHTML = `
        <div class="mav-header">
            <div class="mav-title">
                <div class="mav-dot" id="mav-status-dot"></div>
                STRAT AGENT
                <span id="mav-model-badge" style="font-size:10px;font-weight:500;color:var(--text-muted);"></span>
            </div>
            <div class="mav-actions">
                <button onclick="_mavNewChat()" title="New chat"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg></button>
                <button onclick="_mavToggleFiles()" title="Files"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg></button>
                <button onclick="_mavToggleContext()" title="Context"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></button>
                <button onclick="_mavExport()" title="Export"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>
                <button onclick="_mavClear()" title="Clear" style="color:#f87171;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg></button>
                <button class="mav-close" onclick="_closeMobileAgent()" title="Close"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
            </div>
        </div>
        ${personaRow}
        <div class="mav-body" id="mav-messages"></div>
        <div class="mav-input-wrap">
            <div class="mav-input-row">
                <input class="mav-input" id="mav-input" type="text" placeholder="Ask anything..."
                    onkeydown="if(event.key==='Enter'){event.preventDefault();_mavSend();}">
                <button class="mav-send" onclick="_mavSend()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
                </button>
            </div>
        </div>`;
    document.body.appendChild(view);

    /* Sync messages from existing agent panel */
    _mavSyncMessages();

    /* Sync status dot */
    _mavSyncStatus();

    /* Back button support */
    history.pushState({ mobileAgent: true }, '');
    window.addEventListener('popstate', _onAgentPop);

    _updateBottomNav();
    setTimeout(() => document.getElementById('mav-input')?.focus(), 100);
}

function _closeMobileAgent() {
    const view = document.getElementById('mobile-agent-view');
    if (view) view.remove();
    _mobileAgentOpen = false;
    window.removeEventListener('popstate', _onAgentPop);
    _updateBottomNav();
}

function _onAgentPop() {
    _closeMobileAgent();
}

/* Sync messages from #agent-messages to #mav-messages */
function _mavSyncMessages() {
    const src = document.getElementById('agent-messages');
    const dst = document.getElementById('mav-messages');
    if (!src || !dst) return;
    dst.innerHTML = src.innerHTML;
    dst.scrollTop = dst.scrollHeight;
    /* Also re-init lucide icons in cloned content */
    if (typeof lucide !== 'undefined') lucide.createIcons();
}

/* Sync status dot color */
function _mavSyncStatus() {
    const origDot = document.getElementById('agent-status-dot');
    const mavDot = document.getElementById('mav-status-dot');
    const origBadge = document.getElementById('agent-model-badge');
    const mavBadge = document.getElementById('mav-model-badge');
    if (origDot && mavDot) mavDot.style.background = getComputedStyle(origDot).backgroundColor;
    if (origBadge && mavBadge) mavBadge.textContent = origBadge.textContent;
}

/* Send message through existing agent system */
window._mavSend = function() {
    const input = document.getElementById('mav-input');
    if (!input || !input.value.trim()) return;

    /* Copy value to real agent input and send */
    const realInput = document.getElementById('agent-input');
    if (realInput) {
        realInput.value = input.value;
        input.value = '';
        if (typeof sendAgentMessage === 'function') sendAgentMessage();

        /* Poll for response updates */
        let polls = 0;
        const poller = setInterval(() => {
            _mavSyncMessages();
            polls++;
            if (polls > 120) clearInterval(poller); /* 60 seconds max */
        }, 500);
    }
};

window._mavImport = function() { if (typeof importAgentChat === 'function') importAgentChat(); };
window._mavExport = function() { if (typeof exportAgentChat === 'function') exportAgentChat(); };
window._mavNewChat = function() {
    if (typeof newAgentChat === 'function') newAgentChat();
    setTimeout(_mavSyncMessages, 200);
};
window._mavToggleFiles = function() { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); };
window._mavToggleContext = function() { if (typeof toggleContextEditor === 'function') toggleContextEditor(); };
window._mavClear = function() {
    if (typeof clearAgentChat === 'function') clearAgentChat();
    setTimeout(_mavSyncMessages, 100);
};

/* Also expose _closeMobileAgent globally for onclick */
window._closeMobileAgent = _closeMobileAgent;

/* ═══════════════════════════════════════════════
   D3. MOBILE BOTTOM SHEET SWIPE-TO-DISMISS
   (Context editor & file browser)
   ═══════════════════════════════════════════════ */

function initBottomSheetSwipe() {
    if (!isSmall()) return;
    /* Observe DOM for bottom sheet panels appearing */
    const observer = new MutationObserver(mutations => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                if (node.id === 'context-editor-panel' || node.id === 'file-browser-panel') {
                    _attachSheetSwipe(node);
                }
            }
        }
    });
    observer.observe(document.body, { childList: true });
    /* Also attach to already-existing panels */
    const existing = ['context-editor-panel', 'file-browser-panel'];
    existing.forEach(id => {
        const el = document.getElementById(id);
        if (el) _attachSheetSwipe(el);
    });
}

function _attachSheetSwipe(panel) {
    const sidebar = panel.querySelector('.ctx-editor-sidebar');
    if (!sidebar || sidebar._sheetSwipe) return;
    sidebar._sheetSwipe = true;

    let startY = 0, curY = 0, tracking = false;

    sidebar.addEventListener('touchstart', e => {
        /* Only start swipe from header area (top 60px) */
        const rect = sidebar.getBoundingClientRect();
        const ty = e.touches[0].clientY - rect.top;
        if (ty > 60) return;
        startY = e.touches[0].clientY;
        curY = startY;
        tracking = true;
        sidebar.style.transition = 'none';
    }, { passive: true });

    sidebar.addEventListener('touchmove', e => {
        if (!tracking) return;
        curY = e.touches[0].clientY;
        const dy = curY - startY;
        if (dy > 0) {
            sidebar.style.transform = `translateY(${dy}px)`;
            e.preventDefault();
        }
    }, { passive: false });

    sidebar.addEventListener('touchend', () => {
        if (!tracking) return;
        tracking = false;
        sidebar.style.transition = '';
        const dy = curY - startY;
        if (dy > 80) {
            /* Dismiss */
            if (panel.id === 'context-editor-panel' && typeof toggleContextEditor === 'function') {
                toggleContextEditor();
            } else if (panel.id === 'file-browser-panel' && typeof toggleFileBrowser === 'function') {
                toggleFileBrowser();
            }
        }
        sidebar.style.transform = '';
    }, { passive: true });
}

/* Init bottom sheet swipe on DOMContentLoaded (touch only) */
document.addEventListener('DOMContentLoaded', () => {
    if (isTouchDevice() && isSmall()) initBottomSheetSwipe();
});

/* ═══════════════════════════════════════════════
   D4. MOBILE AGENT — Persona & Conversation Tabs
   ═══════════════════════════════════════════════ */

function _mavBuildPersonaRow() {
    const personas = (typeof availablePersonas !== 'undefined' && availablePersonas.length)
        ? availablePersonas
        : [{name:'intelligence'},{name:'market'},{name:'scholarly'},{name:'gaming'}];
    const cur = typeof currentPersona !== 'undefined' ? currentPersona : 'intelligence';

    return '<div class="mav-persona-row">' + personas.map(p => {
        const active = p.name === cur ? ' mav-persona-active' : '';
        const label = p.name.charAt(0).toUpperCase() + p.name.slice(1);
        return `<button class="mav-persona-btn${active}" onclick="_mavSwitchPersona('${p.name}')">${label}</button>`;
    }).join('') + '</div>';
}

window._mavSwitchPersona = function(name) {
    if (typeof switchPersona === 'function') switchPersona(name);
    /* Update active state */
    const row = document.querySelector('.mav-persona-row');
    if (row) {
        row.querySelectorAll('.mav-persona-btn').forEach(btn => {
            btn.classList.toggle('mav-persona-active', btn.textContent.toLowerCase() === name);
        });
    }
    /* Sync messages after switch */
    setTimeout(_mavSyncMessages, 200);
};


/* ═══════════════════════════════════════════════
   E.  FULLSCREEN CHART MODE — moved to fullscreen-chart.js
   ═══════════════════════════════════════════════ */


/* ═══════════════════════════════════════════════
   F.  SWIPE GESTURE HINTS (first-time users)
   ═══════════════════════════════════════════════ */

function initSwipeHints() {
    if (!isTouchDevice() || !isSmall()) return;
    if (localStorage.getItem('swipe-hints-seen')) return;

    /* Show hint after 3 seconds on first mobile visit */
    setTimeout(() => {
        const hint = document.createElement('div');
        hint.id = 'swipe-hint';
        hint.innerHTML = `
            <div style="position:fixed;bottom:70px;left:50%;transform:translateX(-50%);z-index:300;
                background:var(--bg-panel-solid);border:1px solid var(--border-strong);
                border-radius:12px;padding:10px 16px;display:flex;align-items:center;gap:10px;
                box-shadow:0 4px 20px rgba(0,0,0,0.4);animation:swipeHintIn 0.3s ease;">
                <span style="font-size:18px;">👆</span>
                <div>
                    <div style="font-size:11px;font-weight:600;color:var(--text-primary);">Swipe gestures</div>
                    <div style="font-size:10px;color:var(--text-muted);">Swipe cards right to save, left to dismiss. Swipe from left edge to open sidebar.</div>
                </div>
                <button onclick="this.closest('#swipe-hint').remove();localStorage.setItem('swipe-hints-seen','1')"
                    style="background:none;border:none;color:var(--text-muted);font-size:16px;cursor:pointer;padding:4px;">✕</button>
            </div>`;
        document.body.appendChild(hint);
        /* Auto-dismiss after 8 seconds */
        setTimeout(() => {
            const h = document.getElementById('swipe-hint');
            if (h) { h.style.opacity = '0'; h.style.transition = 'opacity 0.3s'; setTimeout(() => h.remove(), 300); }
            localStorage.setItem('swipe-hints-seen', '1');
        }, 8000);
    }, 3000);
}

document.addEventListener('DOMContentLoaded', initSwipeHints);

/* ── PWA Install Prompt ─────────────────────────────────── */
function initPWAInstall() {
    var _deferredPrompt = null;

    /* Capture the browser install prompt */
    window.addEventListener('beforeinstallprompt', function(e) {
        e.preventDefault();
        _deferredPrompt = e;
        _showInstallBanner();
    });

    function _showInstallBanner() {
        if (document.getElementById('pwa-install-banner')) return;
        var banner = document.createElement('div');
        banner.id = 'pwa-install-banner';
        banner.innerHTML =
            '<span>Install STRAT_OS as an app</span>' +
            '<button id="pwa-install-btn">Install</button>' +
            '<button id="pwa-dismiss-btn" aria-label="Dismiss">&times;</button>';
        document.body.appendChild(banner);

        document.getElementById('pwa-install-btn').addEventListener('click', function() {
            if (!_deferredPrompt) return;
            _deferredPrompt.prompt();
            _deferredPrompt.userChoice.then(function(result) {
                _deferredPrompt = null;
                banner.remove();
            });
        });

        document.getElementById('pwa-dismiss-btn').addEventListener('click', function() {
            banner.remove();
            localStorage.setItem('pwa-dismiss', Date.now());
        });
    }

    /* Don't show if user dismissed within the last 7 days */
    var dismissed = parseInt(localStorage.getItem('pwa-dismiss') || '0');
    if (dismissed && (Date.now() - dismissed) < 7 * 86400000) return;

    /* iOS Safari: no beforeinstallprompt — show manual instructions */
    var isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    var isStandalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone;
    if (isIOS && !isStandalone) {
        var tip = document.createElement('div');
        tip.id = 'pwa-install-banner';
        tip.innerHTML =
            '<span>Install: tap <strong>Share</strong> then <strong>Add to Home Screen</strong></span>' +
            '<button id="pwa-dismiss-btn" aria-label="Dismiss">&times;</button>';
        document.body.appendChild(tip);
        document.getElementById('pwa-dismiss-btn').addEventListener('click', function() {
            tip.remove();
            localStorage.setItem('pwa-dismiss', Date.now());
        });
    }
}

})();
