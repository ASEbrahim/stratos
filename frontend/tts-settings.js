/**
 * TTS Settings — Voice picker, speed, persona voice overrides, preview.
 * Extracted from app.js for maintainability.
 * All functions are global (called from index.html inline handlers and app.js init).
 */

function _toggleTTS(enabled) {
    if (enabled) {
        localStorage.removeItem('stratos_tts_enabled');
        document.body.classList.remove('tts-disabled');
    } else {
        localStorage.setItem('stratos_tts_enabled', '0');
        document.body.classList.add('tts-disabled');
    }
    if (typeof showToast === 'function') showToast(enabled ? 'Text-to-speech enabled' : 'Text-to-speech disabled', 'success');
}

// ── TTS Voice Picker ──
var _ttsVoicesCache = null;

function _loadTTSVoices() {
    var headers = {};
    if (typeof getAuthToken === 'function') headers['X-Auth-Token'] = getAuthToken();
    // Load engine status
    fetch('/api/tts/status', { headers: headers }).then(function(r) { return r.json(); }).then(function(status) {
        var el = document.getElementById('tts-engine-status');
        if (!el) return;
        var parts = [];
        if (status.kokoro && status.kokoro.available) parts.push((status.kokoro.gpu ? 'GPU' : 'CPU') + ' Kokoro — 54 voices, 8 languages');
        if (status.edge && status.edge.available) parts.push('Edge-TTS — Arabic (26 dialect voices)');
        if (!parts.length) parts.push('No TTS engine available');
        el.innerHTML = parts.map(function(p) { return '<div>' + p + '</div>'; }).join('');
    }).catch(function() {});

    // Load voice list
    fetch('/api/tts/voices', { headers: headers }).then(function(r) { return r.json(); }).then(function(voices) {
        _ttsVoicesCache = voices;
        var select = document.getElementById('cfg-tts-voice');
        if (!select) return;
        // Clear options except first
        while (select.options.length > 1) select.remove(1);

        var currentVoice = localStorage.getItem('stratos_tts_voice') || '';
        var personaNames = ['intelligence', 'market', 'scholarly', 'gaming', 'anime', 'tcg'];

        // Add Kokoro voices grouped by language
        var langNames = { en: 'English', ja: 'Japanese', zh: 'Chinese', fr: 'French', ko: 'Korean', hi: 'Hindi', it: 'Italian', pt: 'Portuguese' };
        if (voices.kokoro) {
            Object.keys(voices.kokoro).sort().forEach(function(lang) {
                var group = document.createElement('optgroup');
                group.label = (langNames[lang] || lang) + ' (Kokoro)';
                voices.kokoro[lang].forEach(function(v) {
                    var opt = document.createElement('option');
                    opt.value = v.id;
                    opt.textContent = v.name + ' (' + v.gender + ', ' + v.accent + ')';
                    if (v.id === currentVoice) opt.selected = true;
                    group.appendChild(opt);
                });
                select.appendChild(group);
            });
        }

        // Edge-TTS voices (Arabic)
        if (voices.edge) {
            Object.keys(voices.edge).forEach(function(lang) {
                var group = document.createElement('optgroup');
                group.label = 'Arabic (Edge-TTS)';
                voices.edge[lang].forEach(function(v) {
                    var opt = document.createElement('option');
                    opt.value = v.id;
                    opt.textContent = v.name + ' (' + v.gender + ')';
                    if (v.id === currentVoice) opt.selected = true;
                    group.appendChild(opt);
                });
                select.appendChild(group);
            });
        }


        // Build persona voice dropdowns
        var container = document.getElementById('tts-persona-voices');
        if (container) {
            container.innerHTML = '';
            personaNames.forEach(function(pname) {
                var saved = localStorage.getItem('stratos_tts_voice_' + pname) || '';
                var row = document.createElement('div');
                row.className = 'flex items-center gap-2';
                row.innerHTML = '<span class="text-[10px] text-slate-500 w-20 capitalize">' + pname + ':</span>' +
                    '<select onchange="_setPersonaTTSVoice(\'' + pname + '\', this.value)" class="flex-1 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-[10px] text-slate-300 focus:border-emerald-500 focus:outline-none">' +
                    '<option value="">Default</option></select>';
                var sel = row.querySelector('select');
                // Copy options from main select
                for (var i = 1; i < select.options.length; i++) {
                    var o = select.options[i].cloneNode(true);
                    o.selected = o.value === saved;
                    sel.appendChild(o);
                }
                // Copy optgroups
                var groups = select.querySelectorAll('optgroup');
                sel.innerHTML = '<option value="">Default</option>';
                groups.forEach(function(g) {
                    var gc = g.cloneNode(true);
                    gc.querySelectorAll('option').forEach(function(o) { o.selected = o.value === saved; });
                    sel.appendChild(gc);
                });
                container.appendChild(row);
            });
        }

        // Restore speed
        var speed = localStorage.getItem('stratos_tts_speed') || '1.0';
        var speedEl = document.getElementById('cfg-tts-speed');
        var speedVal = document.getElementById('cfg-tts-speed-val');
        if (speedEl) speedEl.value = speed;
        if (speedVal) speedVal.textContent = parseFloat(speed).toFixed(1) + 'x';
    }).catch(function(e) { console.warn('Failed to load TTS voices:', e); });
}

function _setTTSVoice(voiceId) {
    if (voiceId) {
        localStorage.setItem('stratos_tts_voice', voiceId);
    } else {
        localStorage.removeItem('stratos_tts_voice');
    }
}

function _setTTSSpeed(val) {
    localStorage.setItem('stratos_tts_speed', val);
    var el = document.getElementById('cfg-tts-speed-val');
    if (el) el.textContent = parseFloat(val).toFixed(1) + 'x';
}

function _previewTTSVoice() {
    var select = document.getElementById('cfg-tts-voice');
    var voiceId = select ? select.value : '';
    if (!voiceId) voiceId = 'af_heart';

    var headers = { 'Content-Type': 'application/json' };
    if (typeof getAuthToken === 'function') headers['X-Auth-Token'] = getAuthToken();

    fetch('/api/tts/preview', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ voice: voiceId })
    }).then(function(r) {
        if (!r.ok) throw new Error('Preview failed');
        return r.blob();
    }).then(function(blob) {
        var url = URL.createObjectURL(blob);
        var audio = new Audio(url);
        audio.onended = function() { URL.revokeObjectURL(url); };
        audio.preload = 'auto';
        audio.oncanplaythrough = function() { audio.play(); };
        audio.load();
    }).catch(function(e) {
        if (typeof showToast === 'function') showToast('Voice preview failed: ' + e.message, 'error');
    });
}

function _setPersonaTTSVoice(persona, voiceId) {
    // Save to localStorage for immediate use
    if (voiceId) {
        localStorage.setItem('stratos_tts_voice_' + persona, voiceId);
    } else {
        localStorage.removeItem('stratos_tts_voice_' + persona);
    }
    // Also persist to config.yaml so it survives cache clears
    var payload = { tts: { persona_voices: {} } };
    payload.tts.persona_voices[persona] = voiceId || '';
    var headers = { 'Content-Type': 'application/json' };
    if (typeof getAuthToken === 'function') headers['X-Auth-Token'] = getAuthToken();
    fetch('/api/config', { method: 'POST', headers: headers, body: JSON.stringify(payload) }).catch(function() {});
}
