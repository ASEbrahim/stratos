/**
 * UI Settings Sync — Persists localStorage settings to DB via /api/ui-state.
 *
 * Problem: _clearProfileLocalStorage() wipes all settings on login/profile switch.
 * Solution: Save all UI settings to DB, reload them after login clears localStorage.
 *
 * Flow: login → clear localStorage → load from DB → populate localStorage → init()
 * On setting change: save to localStorage (instant) + debounced POST to DB (2s).
 */

// ── Settings key registry ──
// Maps localStorage key → ui_state blob key (shorter names for DB storage)
var _UI_SETTINGS_MAP = {
    // Tour & onboarding
    'stratos_tour_never': 'tour_never',

    // Display settings
    'stratos_density': 'density',
    'stratos_fontsize': 'fontsize',
    'stratos_auto_refresh': 'auto_refresh',
    'stratos_chart_type': 'chart_type',

    // Audio
    'stratos_tts_enabled': 'tts_enabled',
    'stratos_tts_speed': 'tts_speed',
    'stratos_tts_voice': 'tts_voice',
    'stratos_stt_enabled': 'stt_enabled',

    // Agent customizer
    'stratos-fs-custom': 'fs_custom',
    'stratos-fs-user-presets': 'fs_presets',

    // Theme particles & positions
    'stratos-cosmos-scale': 'cosmos_scale',
    'stratos-cosmos-blur': 'cosmos_blur',
    'stratos-cosmos-opacity': 'cosmos_opacity',
    'stratos-cosmos-density': 'cosmos_density',
    'stratos-cosmos-cx': 'cosmos_cx',
    'stratos-cosmos-cy': 'cosmos_cy',
    'stratos-cosmos-preset': 'cosmos_preset',

    'stratos-sakura-size': 'sakura_size',
    'stratos-sakura-density': 'sakura_density',
    'stratos-sakura-fall': 'sakura_fall',
    'stratos-sakura-wind': 'sakura_wind',
    'stratos-sakura-opacity': 'sakura_opacity',
    'stratos-sakura-tree': 'sakura_tree',
    'stratos-sakura-tree-blur': 'sakura_tree_blur',
    'stratos-sakura-tree-cx': 'sakura_tree_cx',
    'stratos-sakura-tree-cy': 'sakura_tree_cy',
    'stratos-sakura-tree-opacity': 'sakura_tree_opacity',
    'stratos-sakura-tree-scale': 'sakura_tree_scale',

    'stratos-stars-density': 'stars_density',
    'stratos-stars-drift': 'stars_drift',
    'stratos-stars-brightness': 'stars_brightness',

    // Perf mode
    'stratos-perf-mode': 'perf_mode',

    // Sidebar
    'sidebarCollapsed': 'sidebar_collapsed',
    'sidebarWidth': 'sidebar_width',
    'navIntelCollapsed': 'nav_intel_collapsed',
    'navOverviewCollapsed': 'nav_overview_collapsed',
    'navFeedsCollapsed': 'nav_feeds_collapsed',

    // Markets panel
    'mp_layout': 'mp_layout',
    'mp_sec2': 'mp_sec2',
    'mp_bottom_layout': 'mp_bottom_layout',
    'mp_mobile_compact': 'mp_mobile_compact',

    // Feed
    'savedViewGrid': 'feed_grid',

    // File browser
    'fe-atmosphere': 'fe_atmosphere',

    // Games
    'stratos_games_rp_mode': 'games_rp_mode',

    // Notifications
    'notif_muted': 'notif_muted',

    // YouTube
    'stratos_insight_lang': 'insight_lang',

    // Settings mode
    'settingsMode': 'settings_mode',
    'advPanelLayout': 'adv_panel_layout',

    // Mobile
    'pwa-dismiss': 'pwa_dismiss',
    'swipe-hints-seen': 'swipe_hints_seen'
};

// Reverse map for DB → localStorage
var _UI_SETTINGS_REVERSE = {};
(function() {
    for (var lsKey in _UI_SETTINGS_MAP) {
        _UI_SETTINGS_REVERSE[_UI_SETTINGS_MAP[lsKey]] = lsKey;
    }
})();

// ── Dynamic key patterns (profile-scoped) ──
// These keys contain the profile name and need special handling
var _UI_DYNAMIC_PREFIXES = [
    'stratos_tts_voice_',       // per-persona TTS voices
    'stratos-',                 // theme overrides & presets (e.g., stratos-midnight-overrides)
    'stratos_source_order_'     // feed source order
];

// Theme override/preset keys to sync (pattern: stratos-{theme}-overrides, stratos-{theme}-presets)
var _THEME_NAMES = ['midnight', 'nebula', 'aurora', 'noir', 'cosmos', 'sakura', 'rose', 'coffee'];

// ── Debounce timer ──
var _uiSyncTimer = null;
var _UI_SYNC_DEBOUNCE = 2000; // 2 seconds

/**
 * Collect all syncable settings from localStorage into a flat object.
 */
function _collectUiSettings() {
    var settings = {};

    // Static keys
    for (var lsKey in _UI_SETTINGS_MAP) {
        var val = localStorage.getItem(lsKey);
        if (val !== null) {
            settings[_UI_SETTINGS_MAP[lsKey]] = val;
        }
    }

    // Tour completion keys (profile-scoped)
    var p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';
    var tourBasic = localStorage.getItem('stratos_' + p + '_tour_basic_done');
    if (tourBasic) settings['tour_basic_done'] = tourBasic;
    var tourExplore = localStorage.getItem('stratos_' + p + '_tour_explore_done');
    if (tourExplore) settings['tour_explore_done'] = tourExplore;
    var tutDismissed = localStorage.getItem('strat_tutorial_dismissed_' + p);
    if (tutDismissed) settings['tutorial_dismissed'] = tutDismissed;

    // Per-persona TTS voices
    var ttsVoices = {};
    ['intelligence', 'market', 'scholarly', 'gaming', 'anime', 'tcg'].forEach(function(persona) {
        var v = localStorage.getItem('stratos_tts_voice_' + persona);
        if (v) ttsVoices[persona] = v;
    });
    if (Object.keys(ttsVoices).length) settings['tts_persona_voices'] = JSON.stringify(ttsVoices);

    // Theme overrides and presets
    var themeCustom = {};
    _THEME_NAMES.forEach(function(t) {
        var ov = localStorage.getItem('stratos-' + t + '-overrides');
        if (ov) themeCustom[t + '_overrides'] = ov;
        var pr = localStorage.getItem('stratos-' + t + '-presets');
        if (pr) themeCustom[t + '_presets'] = pr;
    });
    if (Object.keys(themeCustom).length) settings['theme_custom'] = JSON.stringify(themeCustom);

    // Theme element positions (pendulum, rose, coffee, etc.)
    var themePositions = {};
    _THEME_NAMES.forEach(function(t) {
        ['cx', 'cy', 'scale', 'blur', 'opacity'].forEach(function(prop) {
            var v = localStorage.getItem('stratos-' + t + '-' + prop);
            if (v) {
                if (!themePositions[t]) themePositions[t] = {};
                themePositions[t][prop] = v;
            }
        });
    });
    if (Object.keys(themePositions).length) settings['theme_positions'] = JSON.stringify(themePositions);

    // Feed source order
    var sourceOrders = {};
    for (var i = 0; i < localStorage.length; i++) {
        var k = localStorage.key(i);
        if (k && k.indexOf('stratos_source_order_') === 0) {
            var root = k.replace('stratos_source_order_', '');
            sourceOrders[root] = localStorage.getItem(k);
        }
    }
    if (Object.keys(sourceOrders).length) settings['source_orders'] = JSON.stringify(sourceOrders);

    return settings;
}

/**
 * Debounced save of all UI settings to the server.
 * Called whenever a setting changes.
 */
function syncUiSettingsToServer() {
    if (_uiSyncTimer) clearTimeout(_uiSyncTimer);
    _uiSyncTimer = setTimeout(function() {
        var settings = _collectUiSettings();
        if (!Object.keys(settings).length) return;
        var token = typeof getAuthToken === 'function' ? getAuthToken() : localStorage.getItem('stratos_auth_token');
        if (!token) return;
        // Wrap in ui_settings key to distinguish from existing ui_state fields (theme, stars, avatar)
        fetch('/api/ui-state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Auth-Token': token },
            body: JSON.stringify({ ui_settings: settings })
        }).catch(function() {});
    }, _UI_SYNC_DEBOUNCE);
}

/**
 * Load UI settings from server and populate localStorage.
 * Returns a Promise. Should be called after _clearProfileLocalStorage() and before init().
 */
function loadUiSettingsFromServer() {
    var token = typeof getAuthToken === 'function' ? getAuthToken() : localStorage.getItem('stratos_auth_token');
    if (!token) return Promise.resolve();

    return fetch('/api/ui-state', {
        headers: { 'X-Auth-Token': token }
    }).then(function(r) {
        if (!r.ok) throw new Error('Failed to load ui settings');
        return r.json();
    }).then(function(uiState) {
        if (!uiState || !uiState.ui_settings) return;
        var settings = uiState.ui_settings;
        var p = typeof getActiveProfile === 'function' ? getActiveProfile() : 'default';

        // Restore static keys
        for (var dbKey in settings) {
            var lsKey = _UI_SETTINGS_REVERSE[dbKey];
            if (lsKey) {
                localStorage.setItem(lsKey, settings[dbKey]);
                continue;
            }

            // Handle special keys
            if (dbKey === 'tour_basic_done' && settings[dbKey]) {
                localStorage.setItem('stratos_' + p + '_tour_basic_done', settings[dbKey]);
            } else if (dbKey === 'tour_explore_done' && settings[dbKey]) {
                localStorage.setItem('stratos_' + p + '_tour_explore_done', settings[dbKey]);
            } else if (dbKey === 'tutorial_dismissed' && settings[dbKey]) {
                localStorage.setItem('strat_tutorial_dismissed_' + p, settings[dbKey]);
            } else if (dbKey === 'tts_persona_voices') {
                try {
                    var voices = JSON.parse(settings[dbKey]);
                    for (var persona in voices) {
                        localStorage.setItem('stratos_tts_voice_' + persona, voices[persona]);
                    }
                } catch (e) {}
            } else if (dbKey === 'theme_custom') {
                try {
                    var custom = JSON.parse(settings[dbKey]);
                    for (var ck in custom) {
                        // ck is like "midnight_overrides" → "stratos-midnight-overrides"
                        var parts = ck.split('_');
                        var suffix = parts.pop(); // 'overrides' or 'presets'
                        var theme = parts.join('_');
                        localStorage.setItem('stratos-' + theme + '-' + suffix, custom[ck]);
                    }
                } catch (e) {}
            } else if (dbKey === 'theme_positions') {
                try {
                    var positions = JSON.parse(settings[dbKey]);
                    for (var theme in positions) {
                        for (var prop in positions[theme]) {
                            localStorage.setItem('stratos-' + theme + '-' + prop, positions[theme][prop]);
                        }
                    }
                } catch (e) {}
            } else if (dbKey === 'source_orders') {
                try {
                    var orders = JSON.parse(settings[dbKey]);
                    for (var root in orders) {
                        localStorage.setItem('stratos_source_order_' + root, orders[root]);
                    }
                } catch (e) {}
            }
        }
    }).catch(function(e) {
        console.warn('UI settings load failed:', e);
    });
}

/**
 * Hook into localStorage.setItem to auto-sync on changes.
 * This patches the native method so all existing code triggers DB sync automatically.
 */
(function() {
    var _origSetItem = localStorage.setItem.bind(localStorage);
    var _origRemoveItem = localStorage.removeItem.bind(localStorage);

    localStorage.setItem = function(key, value) {
        _origSetItem(key, value);
        // Check if this key is one we sync
        if (_UI_SETTINGS_MAP[key] || _shouldSyncDynamic(key)) {
            syncUiSettingsToServer();
        }
    };

    localStorage.removeItem = function(key) {
        _origRemoveItem(key);
        if (_UI_SETTINGS_MAP[key] || _shouldSyncDynamic(key)) {
            syncUiSettingsToServer();
        }
    };

    function _shouldSyncDynamic(key) {
        if (key.indexOf('stratos_tts_voice_') === 0) return true;
        if (key.indexOf('stratos_source_order_') === 0) return true;
        if (key.indexOf('strat_tutorial_dismissed_') === 0) return true;
        // Tour completion keys
        if (key.indexOf('stratos_') === 0 && (key.indexOf('_tour_') > 0)) return true;
        // Theme overrides/presets/positions
        for (var i = 0; i < _THEME_NAMES.length; i++) {
            if (key === 'stratos-' + _THEME_NAMES[i] + '-overrides') return true;
            if (key === 'stratos-' + _THEME_NAMES[i] + '-presets') return true;
            if (key.indexOf('stratos-' + _THEME_NAMES[i] + '-') === 0) return true;
        }
        return false;
    }
})();
