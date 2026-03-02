/**
 * tour.js — StratOS Guided Tour System
 * Spotlight overlay with SVG mask cutout + positioned tooltip cards.
 * Works across all 8 themes x 3 modes (24 combinations) using CSS variables.
 *
 * Two tours:
 *   BASIC_TOUR  — first-time onboarding (welcome → profile → sources → scan)
 *   EXPLORE_TOUR — deeper feature walkthrough (markets, feed, settings, agent)
 */
(function () {
'use strict';

/* ═══════════════════════════════════════
   STEP DEFINITIONS
   ═══════════════════════════════════════ */

const BASIC_TOUR = [
  // Phase 1 — Welcome
  { id: 'welcome', type: 'modal', target: null,
    title: 'Welcome to StratOS',
    body: 'StratOS builds a personalized intelligence feed based on who you are. Let\u2019s set up your profile \u2014 takes about 60 seconds.',
    cta: 'Let\u2019s go', skip: 'I\u2019ll explore on my own',
    nav: { page: 'settings', tab: 'profile' } },

  // Phase 2 — Profile Identity
  { id: 'role', type: 'spotlight', target: '[data-tour="role-field"]',
    title: 'Your Role', position: 'bottom',
    body: 'What do you do? This shapes everything \u2014 the news StratOS hunts for, how it scores relevance, what jobs it surfaces.',
    example: 'e.g. "Computer Engineering Student" or "Marketing Manager"',
    interact: 'input',
    nav: { page: 'settings', tab: 'profile' } },

  { id: 'location', type: 'spotlight', target: '[data-tour="location-field"]',
    title: 'Your Location', position: 'bottom',
    body: 'Where are you based? This scopes local job markets, regional news, and nearby deals.',
    example: 'e.g. "Kuwait City, Kuwait" or "Austin, Texas"',
    interact: 'input',
    nav: { page: 'settings', tab: 'profile' } },

  { id: 'context', type: 'spotlight', target: '[data-tour="context-field"]',
    title: 'Tracking Context', position: 'bottom',
    body: 'Tell StratOS what to watch for. Companies, certifications, industries \u2014 the AI reads this to prioritize your feed.',
    example: 'e.g. "Track Equate, KOC, SLB hiring. AWS certs. Oil & gas trends."',
    interact: 'textarea',
    nav: { page: 'settings', tab: 'profile' } },

  { id: 'wizard-btn', type: 'spotlight', target: '[data-tour="wizard-btn"]',
    title: 'Setup Wizard', position: 'bottom',
    body: 'This guided wizard builds your intelligence categories. Pick priorities, refine details \u2014 all AI-assisted.',
    hint: 'Click to open, or press Next to continue.',
    nav: { page: 'settings', tab: 'profile' } },

  // Phase 3 — Sources & Market Config
  { id: 'sources-tab', type: 'spotlight', target: '[data-tour="sources-tab"]',
    title: 'News Sources', position: 'bottom',
    body: 'Control WHERE your news comes from. Toggle RSS feeds, add custom sources, configure search providers.',
    nav: { page: 'settings', tab: 'sources' } },

  { id: 'market-tickers', type: 'spotlight', target: '[data-tour="market-tickers"]',
    title: 'Market Tickers', position: 'top',
    body: 'Add stocks (NVDA), crypto (BTC-USD), or commodities (CL=F). Use "Top 10" for blue chips or "Movers" for today\u2019s biggest swings.',
    nav: { page: 'settings', tab: 'market' } },

  // Phase 4 — Dashboard
  { id: 'score-filters', type: 'spotlight', target: '[data-tour="score-filters"]',
    title: 'Score Filters', position: 'bottom',
    body: 'Every article is scored by AI relevance. 9+ is critical, 7+ high, 5+ moderate. Filter to focus on what matters most to you.',
    nav: { page: 'dashboard' } },

  { id: 'agent-chat', type: 'spotlight', target: '#agent-panel',
    title: 'Strat Agent', position: 'top',
    body: 'Your AI assistant. Search the web, manage your watchlist, analyze signals. Type $NVDA for quick price lookups.',
    nav: { page: 'dashboard' } },

  { id: 'markets-widget', type: 'spotlight', target: '[data-tour="markets-widget"]',
    title: 'Markets Widget', position: 'left',
    body: 'Live price charts for your watchlist tickers. Switch between line and candlestick views, draw trend lines, go fullscreen.',
    nav: { page: 'dashboard' } },

  { id: 'theme-picker', type: 'spotlight', target: '[data-tour="theme-picker"]',
    title: 'Theme System', position: 'right',
    body: '8 themes with Brighter, Normal, and Deeper modes. Brighter lifts backgrounds for OLED readability \u2014 not a light theme.',
    nav: { page: 'dashboard', sidebar: 'open' } },

  // Phase 5 — Scan
  { id: 'scan-btn', type: 'spotlight', target: '[data-tour="scan-btn"]',
    title: 'Run Your First Scan', position: 'bottom',
    body: 'Hit this to launch your first intelligence scan. StratOS scrapes, scores, and surfaces your personalized feed.',
    hint: 'You\u2019re all set!',
    nav: { page: 'dashboard', sidebar: 'close' } },
];

const EXPLORE_TOUR = [
  // — Markets Panel — navigate first, then spotlight individual sections
  { id: 'mp-overview', type: 'spotlight', target: '[data-tour="mp-overview"]',
    title: 'Market Overview', position: 'bottom',
    body: 'Welcome to the Markets tab! This heatmap shows your watchlist at a glance. Green means up, red means down. Click any ticker to jump to its chart.',
    nav: { page: 'markets_view' } },

  { id: 'mp-shortcuts', type: 'spotlight', target: '[data-tour="mp-shortcuts"]',
    title: 'Charts & Shortcuts', position: 'bottom',
    body: 'Up to 6 charts side-by-side with drag-to-swap. Hover this button for keyboard shortcuts \u2014 keys 1\u20135 change timeframes, J/K cycle tickers, C swaps chart type, D draws.',
    nav: { page: 'markets_view' } },

  { id: 'mp-intel', type: 'spotlight', target: '[data-tour="mp-intel"]',
    title: 'Ticker Intel', position: 'top',
    body: 'News and analysis for the selected ticker. Articles are pulled from your scan and scored for relevance. Click any article to read more.',
    nav: { page: 'markets_view' } },

  { id: 'mp-agent', type: 'spotlight', target: '[data-tour="mp-agent"]',
    title: 'Market Agent', position: 'top',
    body: 'A dedicated market AI. Ask it to compare tickers, explain price moves, or add symbols to your watchlist. Import context or export chat history.',
    nav: { page: 'markets_view' } },

  // — Dashboard deep-dives (unique to Explore — no overlap with Basic) —
  { id: 'chart-tools', type: 'spotlight', target: '[data-tour="chart-tools"]',
    title: 'Chart Toolbar', position: 'bottom',
    body: 'Toggle candlestick view, enable crosshair for precise readings, draw trend lines, auto-detect swing points, or go fullscreen with Focus mode.',
    hint: 'Keyboard: C = chart type, X = crosshair, D = draw, F = focus',
    nav: { page: 'dashboard' } },

  { id: 'feed-cards', type: 'spotlight', target: '#news-feed',
    mobileTarget: '#news-feed > div:first-child',
    title: 'Feed Card Actions', position: 'top',
    body: 'Each card has actions: bookmark to save, thumbs up/down to train the scorer, star to rate 1\u201310, and sparkles to ask the AI about that specific article.',
    nav: { page: 'dashboard' } },

  // — Settings —
  { id: 'display-settings', type: 'spotlight', target: '[data-tour="display-settings"]',
    title: 'Display Settings', position: 'top',
    body: 'Adjust feed density, font size, default chart type, and auto-refresh interval. These are per-device preferences.',
    nav: { page: 'settings', tab: 'system' } },
];

/* ═══════════════════════════════════════
   GUIDED TOUR CLASS
   ═══════════════════════════════════════ */

class GuidedTour {
  constructor(steps, tourId) {
    this._steps = steps;
    this._tourId = tourId || 'basic';
    this._current = 0;
    this._active = false;
    this._overlay = null;
    this._tooltip = null;
    this._welcome = null;
    this._svg = null;
    this._resizeTimer = null;
    this._interactionCleanup = null;
    this._transitioning = false;
    this._animFrame = null;
    this._boundResize = this._onResize.bind(this);
    this._boundKeydown = this._onKeydown.bind(this);
    this._boundScroll = this._onScroll.bind(this);
  }

  /* ── Public API ── */

  start(fromStep) {
    if (window._stratosTourActive) {
      window._stratosTour?.destroy();
    }
    window._stratosTourActive = true;
    window._stratosTour = this;
    this._active = true;

    if (fromStep !== undefined) {
      this._current = fromStep;
    } else {
      const saved = this._loadStep();
      this._current = saved >= 0 ? saved : 0;
    }

    this._injectOverlay();
    this._bindListeners();

    // Let overlay fade in, then show first step
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        this._overlay.classList.add('tour-visible');
        setTimeout(() => this._show(), 250);
      });
    });
  }

  advance() {
    if (!this._active || this._transitioning) return;
    this._cleanupInteraction();
    if (this._current < this._steps.length - 1) {
      this._current++;
      this._persist();
      this._transitionToStep();
    } else {
      this._complete();
    }
  }

  skip() {
    if (this._transitioning) return;
    this._cleanupInteraction();
    this._complete();
  }

  destroy() {
    this._active = false;
    this._transitioning = false;
    window._stratosTourActive = false;
    this._cleanupInteraction();
    this._unbindListeners();
    if (this._animFrame) cancelAnimationFrame(this._animFrame);
    if (this._overlay && this._overlay.parentNode) {
      this._overlay.parentNode.removeChild(this._overlay);
    }
    this._overlay = null;
    this._tooltip = null;
    this._welcome = null;
    this._svg = null;
  }

  /* ── DOM Injection ── */

  _injectOverlay() {
    const old = document.getElementById('stratos-tour-overlay');
    if (old) old.remove();

    const overlay = document.createElement('div');
    overlay.id = 'stratos-tour-overlay';
    overlay.innerHTML = `
      <svg id="stratos-tour-backdrop" class="tour-backdrop">
        <defs>
          <mask id="tour-mask">
            <rect width="100%" height="100%" fill="white"/>
            <rect id="tour-cutout" rx="12" ry="12" fill="black" x="-20" y="-20" width="0" height="0"/>
          </mask>
        </defs>
        <rect class="tour-backdrop-fill" width="100%" height="100%" mask="url(#tour-mask)"/>
        <rect id="tour-highlight-ring" class="tour-highlight-ring" x="-20" y="-20" width="0" height="0" rx="12" ry="12"/>
      </svg>
      <div id="stratos-tour-tooltip" class="tour-tooltip tour-tooltip-hidden">
        <div class="tour-tag-bar">
          <span class="tour-tag-icon"></span>
          <span class="tour-tag-text">GUIDED TOUR</span>
          <span class="tour-progress-text"></span>
        </div>
        <div class="tour-progress-track">
          <div class="tour-progress-fill"></div>
        </div>
        <div class="tour-content">
          <h4 class="tour-title"></h4>
          <p class="tour-body"></p>
          <p class="tour-example"></p>
          <p class="tour-hint"></p>
        </div>
        <div class="tour-actions">
          <button class="tour-skip">Skip tour</button>
          <button class="tour-next">Next <span class="tour-next-arrow">\u2192</span></button>
        </div>
        <button class="tour-never">Don't show again</button>
      </div>
      <div id="stratos-tour-welcome" class="tour-welcome-modal tour-welcome-hidden">
        <div class="tour-welcome-card">
          <div class="tour-welcome-glow"></div>
          <div class="tour-welcome-icon">\uD83D\uDCE1</div>
          <h3 class="tour-welcome-title"></h3>
          <p class="tour-welcome-body"></p>
          <div class="tour-welcome-actions">
            <button class="tour-welcome-skip"></button>
            <button class="tour-welcome-cta"></button>
          </div>
          <button class="tour-never tour-welcome-never">Don't show again</button>
        </div>
      </div>`;

    document.body.appendChild(overlay);
    this._overlay = overlay;
    this._svg = overlay.querySelector('#stratos-tour-backdrop');
    this._tooltip = overlay.querySelector('#stratos-tour-tooltip');
    this._welcome = overlay.querySelector('#stratos-tour-welcome');

    // Wire buttons
    this._tooltip.querySelector('.tour-next').onclick = () => this.advance();
    this._tooltip.querySelector('.tour-skip').onclick = () => this.skip();
    this._tooltip.querySelector('.tour-never').onclick = () => { localStorage.setItem('stratos_tour_never', 'true'); this.skip(); };
    this._welcome.querySelector('.tour-welcome-cta').onclick = () => this.advance();
    this._welcome.querySelector('.tour-welcome-skip').onclick = () => this.skip();
    this._welcome.querySelector('.tour-welcome-never').onclick = () => { localStorage.setItem('stratos_tour_never', 'true'); this.skip(); };

    // Click on backdrop (dimmed area) advances the tour
    this._svg.onclick = (e) => {
      if (e.target.closest('.tour-tooltip') || e.target.closest('.tour-welcome-modal')) return;
      this.advance();
    };
  }

  /* ── Step Transitions ── */

  _transitionToStep() {
    this._transitioning = true;
    const tt = this._tooltip;

    // Quick fade out
    tt.classList.remove('tour-tooltip-visible');
    tt.classList.add('tour-tooltip-exit');

    setTimeout(() => {
      tt.classList.remove('tour-tooltip-exit');
      tt.classList.add('tour-tooltip-hidden');
      this._transitioning = false;
      this._show();
    }, 200);
  }

  _show() {
    const step = this._steps[this._current];
    if (!step) { this._complete(); return; }

    this._navigate(step, () => {
      if (step.type === 'modal') {
        this._showWelcome(step);
      } else {
        this._showSpotlight(step);
      }
    });
  }

  /* ── Welcome Modal ── */

  _showWelcome(step) {
    this._tooltip.classList.add('tour-tooltip-hidden');
    this._hideCutout();

    const w = this._welcome;
    w.querySelector('.tour-welcome-title').textContent = step.title;
    w.querySelector('.tour-welcome-body').textContent = step.body;
    w.querySelector('.tour-welcome-cta').textContent = step.cta || 'Next';
    w.querySelector('.tour-welcome-skip').textContent = step.skip || 'Skip tour';

    w.classList.remove('tour-welcome-hidden', 'tour-welcome-exiting');
    w.classList.add('tour-welcome-entering');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        w.classList.remove('tour-welcome-entering');
        w.classList.add('tour-welcome-visible');
      });
    });
  }

  _hideWelcome(cb) {
    const w = this._welcome;
    w.classList.remove('tour-welcome-visible');
    w.classList.add('tour-welcome-exiting');
    setTimeout(() => {
      w.classList.remove('tour-welcome-exiting');
      w.classList.add('tour-welcome-hidden');
      if (cb) cb();
    }, 220);
  }

  /* ── Spotlight ── */

  _showSpotlight(step) {
    if (!this._welcome.classList.contains('tour-welcome-hidden')) {
      this._hideWelcome(() => this._doSpotlight(step));
      return;
    }
    this._doSpotlight(step);
  }

  _doSpotlight(step) {
    // On mobile, use mobileTarget if available
    const _isMob = window.innerWidth <= 768;
    const selector = (_isMob && step.mobileTarget) ? step.mobileTarget : step.target;
    const target = document.querySelector(selector);
    // Skip if target missing or truly invisible
    const isVisible = target && (target.offsetParent !== null ||
      target.offsetWidth > 0 || target.offsetHeight > 0 ||
      getComputedStyle(target).position === 'fixed');
    if (!isVisible) {
      // Retry once after 500ms for async-rendered elements (e.g. markets panel)
      if (!step._retryCount) step._retryCount = 0;
      if (step._retryCount < 2) {
        step._retryCount++;
        setTimeout(() => this._doSpotlight(step), 500);
        return;
      }
      console.warn(`[Tour] Target not found or hidden: ${selector}, skipping "${step.id}"`);
      if (this._current < this._steps.length - 1) {
        this._current++;
        this._persist();
        this._show();
      } else {
        this._complete();
      }
      return;
    }

    // Scroll into view
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });

    setTimeout(() => {
      if (!this._active) return;
      const rect = target.getBoundingClientRect();
      const pad = 12;

      this._animateCutout(
        rect.left - pad, rect.top - pad,
        rect.width + pad * 2, rect.height + pad * 2
      );

      // Show tooltip after spotlight settles
      setTimeout(() => {
        if (!this._active) return;
        const freshRect = target.getBoundingClientRect();
        this._renderTooltip(step, freshRect);
        if (step.interact) this._setupInteraction(step, target);
      }, 150);
    }, 350);
  }

  /* ── Cutout ── */

  _animateCutout(x, y, w, h) {
    const cutout = this._svg.querySelector('#tour-cutout');
    const ring = this._svg.querySelector('#tour-highlight-ring');
    ring.classList.add('tour-ring-pulse');

    cutout.setAttribute('x', x);
    cutout.setAttribute('y', y);
    cutout.setAttribute('width', Math.max(0, w));
    cutout.setAttribute('height', Math.max(0, h));
    ring.setAttribute('x', x);
    ring.setAttribute('y', y);
    ring.setAttribute('width', Math.max(0, w));
    ring.setAttribute('height', Math.max(0, h));
  }

  _hideCutout() {
    const cutout = this._svg.querySelector('#tour-cutout');
    const ring = this._svg.querySelector('#tour-highlight-ring');
    ring.classList.remove('tour-ring-pulse');
    cutout.setAttribute('width', '0');
    cutout.setAttribute('height', '0');
    ring.setAttribute('width', '0');
    ring.setAttribute('height', '0');
  }

  /* ── Tooltip ── */

  _renderTooltip(step, targetRect) {
    const tt = this._tooltip;
    const total = this._steps.length;
    const cur = this._current + 1;
    const pct = Math.round((cur / total) * 100);

    // Progress
    tt.querySelector('.tour-progress-text').textContent = `${cur} of ${total}`;
    tt.querySelector('.tour-progress-fill').style.width = `${pct}%`;

    // Icon based on phase
    const icon = tt.querySelector('.tour-tag-icon');
    if (step.id.startsWith('market') || step.id.startsWith('mp-') || step.id.startsWith('chart')) icon.textContent = '\uD83D\uDCC8';
    else if (step.id.startsWith('agent') || step.id === 'feed-cards') icon.textContent = '\u2728';
    else if (step.id.startsWith('display') || step.id.startsWith('theme')) icon.textContent = '\uD83C\uDFA8';
    else if (step.id === 'scan-btn') icon.textContent = '\uD83D\uDE80';
    else if (step.id === 'score-filters') icon.textContent = '\uD83C\uDFAF';
    else icon.textContent = '\uD83D\uDCA1';

    // Content
    tt.querySelector('.tour-title').textContent = step.title;
    tt.querySelector('.tour-body').textContent = step.body;

    const exEl = tt.querySelector('.tour-example');
    exEl.textContent = step.example || '';
    exEl.classList.toggle('tour-el-hidden', !step.example);

    const hintEl = tt.querySelector('.tour-hint');
    hintEl.textContent = step.hint || '';
    hintEl.classList.toggle('tour-el-hidden', !step.hint);

    // Button text
    const nextBtn = tt.querySelector('.tour-next');
    if (this._current === this._steps.length - 1) {
      nextBtn.innerHTML = 'Done \u2713';
    } else {
      nextBtn.innerHTML = 'Next <span class="tour-next-arrow">\u2192</span>';
    }

    // Position & animate in
    this._positionTooltip(step.position || 'bottom', targetRect);
    tt.classList.remove('tour-tooltip-hidden', 'tour-tooltip-exit');
    tt.classList.add('tour-tooltip-entering');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        tt.classList.remove('tour-tooltip-entering');
        tt.classList.add('tour-tooltip-visible');
      });
    });
  }

  _positionTooltip(preferred, targetRect) {
    const tt = this._tooltip;
    const ttW = Math.min(330, window.innerWidth - 32);
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const gap = 16;
    const isMob = vw < 768;
    const pos = isMob ? 'bottom' : preferred;
    const pad = 16;

    let left, top;

    if (pos === 'bottom') {
      top = targetRect.bottom + gap;
      left = targetRect.left + targetRect.width / 2 - ttW / 2;
      if (top + 220 > vh) { top = targetRect.top - gap - 220; } // flip
    } else if (pos === 'top') {
      top = targetRect.top - gap - 220;
      if (top < pad) { top = targetRect.bottom + gap; }
      left = targetRect.left + targetRect.width / 2 - ttW / 2;
    } else if (pos === 'right') {
      top = targetRect.top + targetRect.height / 2 - 110;
      left = targetRect.right + gap;
      if (left + ttW > vw - pad) { left = targetRect.left - gap - ttW; }
    } else if (pos === 'left') {
      top = targetRect.top + targetRect.height / 2 - 110;
      left = targetRect.left - gap - ttW;
      if (left < pad) { left = targetRect.right + gap; }
    }

    left = Math.max(pad, Math.min(left, vw - pad - ttW));
    top = Math.max(pad, Math.min(top, vh - 240));

    if (isMob) {
      tt.style.width = `calc(100vw - ${pad * 2}px)`;
      tt.style.maxWidth = '380px';
      left = pad;
    } else {
      tt.style.width = `${ttW}px`;
      tt.style.maxWidth = '';
    }

    tt.style.left = `${left}px`;
    tt.style.top = `${top}px`;
  }

  /* ── Interaction Gates ── */

  _setupInteraction(step, target) {
    this._cleanupInteraction();
    const handlers = [];
    const self = this;

    target.classList.add('tour-interactive-target');
    // Don't auto-focus inputs on touch devices — it triggers the virtual keyboard
    const _isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (!_isTouch) {
      setTimeout(() => { try { target.focus(); } catch(e) {} }, 120);
    }

    function onKeydown(e) {
      if (step.interact === 'input' && e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        if (target.value.trim()) self.advance();
      }
      if (step.interact === 'textarea' && e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        e.stopPropagation();
        if (target.value.trim()) self.advance();
      }
    }
    target.addEventListener('keydown', onKeydown);
    handlers.push(() => target.removeEventListener('keydown', onKeydown));

    this._interactionCleanup = () => {
      handlers.forEach(fn => fn());
      target.classList.remove('tour-interactive-target');
    };
  }

  _cleanupInteraction() {
    if (this._interactionCleanup) {
      this._interactionCleanup();
      this._interactionCleanup = null;
    }
  }

  /* ── Navigation ── */

  _navigate(step, cb) {
    const nav = step.nav;
    if (!nav) { cb(); return; }
    let changed = false;

    if (nav.page === 'markets_view' && typeof setActive === 'function') {
      const mp = document.getElementById('markets-panel');
      if (!mp || mp.classList.contains('hidden')) {
        setActive('markets_view');
        changed = true;
      }
    } else if (nav.page === 'settings' && typeof setActive === 'function') {
      const sp = document.getElementById('settings-panel');
      if (sp && sp.classList.contains('hidden')) {
        setActive('settings');
        changed = true;
      }
    } else if (nav.page === 'dashboard' && typeof setActive === 'function') {
      const sp = document.getElementById('settings-panel');
      const mc = document.getElementById('main-content');
      if (!mc || mc.classList.contains('hidden') || (sp && !sp.classList.contains('hidden'))) {
        setActive('dashboard');
        changed = true;
      }
    }

    if (nav.tab && typeof switchSettingsTab === 'function') {
      const cur = typeof _currentSettingsTab !== 'undefined' ? _currentSettingsTab : '';
      if (cur !== nav.tab) {
        switchSettingsTab(nav.tab);
        changed = true;
      }
    }

    if (nav.wizard === false && typeof closeWizard === 'function') {
      const wb = document.getElementById('wiz-bk');
      if (wb && wb.classList.contains('open')) {
        closeWizard();
        changed = true;
      }
    }

    // Sidebar control for mobile tour steps
    const _isMob = window.innerWidth <= 768;
    if (_isMob && nav.sidebar && typeof sidebarCollapsed !== 'undefined') {
      if (nav.sidebar === 'open' && sidebarCollapsed) {
        sidebarCollapsed = false;
        localStorage.setItem('sidebarCollapsed', 'false');
        if (typeof applySidebarState === 'function') applySidebarState();
        changed = true;
      } else if (nav.sidebar === 'close' && !sidebarCollapsed) {
        sidebarCollapsed = true;
        localStorage.setItem('sidebarCollapsed', 'true');
        if (typeof applySidebarState === 'function') applySidebarState();
        changed = true;
      }
    }

    // Longer delay for page changes so DOM settles
    if (changed) {
      // Markets panel needs extra time for initMarketsPanel() to render
      const delay = nav.page === 'markets_view' ? 700 : 400;
      setTimeout(cb, delay);
    } else {
      requestAnimationFrame(cb);
    }
  }

  /* ── Events ── */

  _bindListeners() {
    window.addEventListener('resize', this._boundResize);
    window.addEventListener('keydown', this._boundKeydown, true);
    window.addEventListener('scroll', this._boundScroll, { capture: true, passive: true });
  }

  _unbindListeners() {
    window.removeEventListener('resize', this._boundResize);
    window.removeEventListener('keydown', this._boundKeydown, true);
    window.removeEventListener('scroll', this._boundScroll, { capture: true, passive: true });
  }

  _onResize() {
    clearTimeout(this._resizeTimer);
    this._resizeTimer = setTimeout(() => {
      if (!this._active) return;
      const step = this._steps[this._current];
      if (step && step.type === 'spotlight' && step.target) {
        const target = document.querySelector(step.target);
        if (target) {
          const rect = target.getBoundingClientRect();
          const pad = 12;
          this._animateCutout(rect.left - pad, rect.top - pad, rect.width + pad * 2, rect.height + pad * 2);
          this._positionTooltip(step.position || 'bottom', rect);
        }
      }
    }, 80);
  }

  _onScroll() { this._onResize(); }

  _onKeydown(e) {
    if (!this._active || this._transitioning) return;
    const step = this._steps[this._current];
    if (step && step.interact) {
      const target = document.querySelector(step.target);
      if (target && document.activeElement === target) {
        if (e.key === 'Escape') { e.preventDefault(); this.skip(); }
        return; // let interaction handler deal with Enter
      }
    }
    if (e.key === 'Enter') { e.preventDefault(); this.advance(); }
    else if (e.key === 'Escape') { e.preventDefault(); this.skip(); }
  }

  /* ── State ── */

  _persist() {
    try {
      const p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';
      localStorage.setItem(`stratos_${p}_tour_${this._tourId}_step`, String(this._current));
    } catch (e) {}
  }

  _loadStep() {
    try {
      const p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';
      const v = localStorage.getItem(`stratos_${p}_tour_${this._tourId}_step`);
      return v !== null ? parseInt(v, 10) : -1;
    } catch (e) { return -1; }
  }

  _complete() {
    try {
      const p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';
      localStorage.setItem(`stratos_${p}_tour_${this._tourId}_done`, 'true');
      localStorage.removeItem(`stratos_${p}_tour_${this._tourId}_step`);
    } catch (e) {}

    const tourId = this._tourId;

    if (this._overlay) {
      this._tooltip.classList.add('tour-tooltip-exit');
      this._overlay.classList.remove('tour-visible');
      this._overlay.classList.add('tour-fading-out');
      setTimeout(() => {
        this.destroy();
        _onTourDone(tourId);
      }, 450);
    } else {
      this.destroy();
      _onTourDone(tourId);
    }
  }
}

/* ═══════════════════════════════════════
   PUBLIC API
   ═══════════════════════════════════════ */

function maybeStartTour() {
  try {
    // Global "don't show again" — survives localStorage wipes
    if (localStorage.getItem('stratos_tour_never') === 'true') return;
    const p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';
    if (localStorage.getItem(`stratos_${p}_tour_basic_done`) === 'true') return;
    if (localStorage.getItem(`strat_tutorial_dismissed_${p}`)) return;
  } catch (e) { return; }

  // Dismiss old wizard
  _dismissOldWizard();

  new GuidedTour(BASIC_TOUR, 'basic').start();
}

function restartTour() {
  _dismissOldWizard();
  new GuidedTour(BASIC_TOUR, 'basic').start(0);
}

function startExploreTour() {
  _dismissOldWizard();
  new GuidedTour(EXPLORE_TOUR, 'explore').start(0);
}

function _onTourDone(tourId) {
  if (tourId === 'basic') {
    if (typeof showToast === 'function') {
      showToast('Tour complete \u2014 you\u2019re all set!', 'success');
    }
    // After basic tour, offer the deeper explore tour
    setTimeout(() => _offerExploreTour(), 1200);
  } else {
    if (typeof showToast === 'function') {
      showToast('Exploration complete!', 'success');
    }
  }
}

function _offerExploreTour() {
  try {
    const p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';
    if (localStorage.getItem(`stratos_${p}_tour_explore_done`) === 'true') return;
  } catch (e) { return; }

  // Create a minimal prompt banner
  const banner = document.createElement('div');
  banner.className = 'tour-explore-prompt';
  banner.innerHTML = `
    <span class="tour-explore-prompt-text">\u2728 Want a deeper walkthrough of markets, charts, and scoring?</span>
    <button class="tour-explore-prompt-yes">Show me</button>
    <button class="tour-explore-prompt-no">\u2715</button>`;
  document.body.appendChild(banner);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => banner.classList.add('tour-explore-prompt-visible'));
  });

  banner.querySelector('.tour-explore-prompt-yes').onclick = () => {
    banner.classList.remove('tour-explore-prompt-visible');
    setTimeout(() => { banner.remove(); startExploreTour(); }, 300);
  };
  banner.querySelector('.tour-explore-prompt-no').onclick = () => {
    banner.classList.remove('tour-explore-prompt-visible');
    setTimeout(() => banner.remove(), 300);
  };

  // Auto-dismiss after 12 seconds
  setTimeout(() => {
    if (banner.parentNode) {
      banner.classList.remove('tour-explore-prompt-visible');
      setTimeout(() => { if (banner.parentNode) banner.remove(); }, 300);
    }
  }, 12000);
}

function _dismissOldWizard() {
  // Legacy — old tutorial overlay has been removed.
  // Kept as no-op so callers don't break.
}

// Close help menu on outside click
document.addEventListener('click', function(e) {
  const menu = document.getElementById('tour-help-menu');
  if (!menu || menu.classList.contains('tour-el-hidden')) return;
  if (!e.target.closest('#tour-help-menu') && !e.target.closest('#help-btn')) {
    menu.classList.add('tour-el-hidden');
  }
});

window.GuidedTour = GuidedTour;
window.maybeStartTour = maybeStartTour;
window.restartTour = restartTour;
window.startExploreTour = startExploreTour;

})();
