/* ================================================================
   Voice Input — Speech-to-Text via faster-whisper
   Standalone module: injects mic button into agent input area.
   Does NOT modify agent.js.
   ================================================================ */

let _sttMediaRecorder = null;
let _sttAudioChunks = [];
let _sttRecording = false;
let _sttStream = null;
let _sttRecordingStartTime = null;
let _sttDurationInterval = null;
let _sttMaxDurationTimer = null;
let _sttHoldTimer = null;
let _sttIsHoldMode = false;

const STT_MAX_DURATION_MS = 120000; // 2 minutes

// ── Availability check ──────────────────────────────────────

async function _sttCheckAvailability() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        return { available: false, reason: 'Browser does not support audio recording' };
    }
    if (!window.MediaRecorder) {
        return { available: false, reason: 'MediaRecorder API not available' };
    }
    try {
        const resp = await fetch('/api/agent-status');
        const data = await resp.json();
        if (data.stt && !data.stt.available) {
            return { available: false, reason: data.stt.message || 'STT not available on server' };
        }
    } catch (e) { /* show button anyway, handle error on use */ }
    return { available: true };
}

// ── Recording control ───────────────────────────────────────

async function _sttStartRecording(micBtn) {
    try {
        _sttStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });
    } catch (err) {
        if (err.name === 'NotAllowedError') {
            showToast('Microphone access denied. Please allow microphone in browser settings.', 'error');
        } else if (err.name === 'NotFoundError') {
            showToast('No microphone found. Please connect a microphone.', 'error');
        } else {
            showToast('Microphone error: ' + err.message, 'error');
        }
        return;
    }

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
            ? 'audio/ogg;codecs=opus'
            : 'audio/webm';

    _sttAudioChunks = [];
    _sttMediaRecorder = new MediaRecorder(_sttStream, { mimeType });

    _sttMediaRecorder.ondataavailable = function(e) {
        if (e.data.size > 0) _sttAudioChunks.push(e.data);
    };

    _sttMediaRecorder.onstop = async function() {
        var blob = new Blob(_sttAudioChunks, { type: mimeType });
        await _sttProcessAudio(blob, micBtn);
        if (_sttStream) {
            _sttStream.getTracks().forEach(function(t) { t.stop(); });
            _sttStream = null;
        }
    };

    _sttMediaRecorder.onerror = function() {
        _sttSetState(micBtn, 'error', 'Recording failed');
        _sttCleanup();
    };

    _sttMediaRecorder.start(250);
    _sttRecording = true;
    _sttRecordingStartTime = Date.now();
    _sttSetState(micBtn, 'recording');
    _sttStartDuration(micBtn);

    _sttMaxDurationTimer = setTimeout(function() {
        if (_sttRecording) {
            showToast('Maximum recording time reached (2 minutes)', 'info');
            _sttStopRecording(micBtn);
        }
    }, STT_MAX_DURATION_MS);
}

function _sttStopRecording(micBtn) {
    _sttStopDuration();
    if (_sttMaxDurationTimer) {
        clearTimeout(_sttMaxDurationTimer);
        _sttMaxDurationTimer = null;
    }
    if (_sttMediaRecorder && _sttMediaRecorder.state === 'recording') {
        _sttRecording = false;
        _sttMediaRecorder.stop();
        _sttSetState(micBtn, 'processing');
    }
}

function _sttToggle(micBtn) {
    if (_sttRecording) {
        _sttStopRecording(micBtn);
    } else {
        _sttStartRecording(micBtn);
    }
}

// ── Audio processing ────────────────────────────────────────

async function _sttProcessAudio(blob, micBtn) {
    var durationMs = Date.now() - _sttRecordingStartTime;
    if (durationMs < 500) {
        _sttSetState(micBtn, 'idle');
        return;
    }
    _sttSetState(micBtn, 'processing');
    try {
        var resp = await fetch('/api/stt', {
            method: 'POST',
            headers: { 'Content-Type': blob.type || 'audio/webm' },
            body: blob,
        });
        if (!resp.ok) {
            var errData = {};
            try { errData = await resp.json(); } catch (e) {}
            throw new Error(errData.error || 'Server error: ' + resp.status);
        }
        var result = await resp.json();
        if (!result.text || result.text.trim().length === 0) {
            showToast('No speech detected. Try speaking louder or closer to the microphone.', 'info');
            _sttSetState(micBtn, 'idle');
            return;
        }
        _sttInsertText(result.text);
        if (result.language && result.language !== 'en') {
            var langNames = { ar: 'Arabic', ja: 'Japanese', ko: 'Korean', zh: 'Chinese', fr: 'French', de: 'German', es: 'Spanish' };
            showToast('Detected: ' + (langNames[result.language] || result.language) +
                ' (' + result.duration_seconds + 's audio)', 'info', 3000);
        }
        _sttSetState(micBtn, 'idle');
    } catch (err) {
        console.error('STT error:', err);
        showToast('Transcription failed: ' + err.message, 'error');
        _sttSetState(micBtn, 'error', err.message);
    }
}

function _sttInsertText(text) {
    var ta = document.getElementById('agent-input');
    if (!ta) return;
    if (ta.value.trim().length > 0) {
        ta.value = ta.value.trimEnd() + ' ' + text;
    } else {
        ta.value = text;
    }
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
    ta.selectionStart = ta.selectionEnd = ta.value.length;
}

// ── Duration display ────────────────────────────────────────

function _sttStartDuration(micBtn) {
    var indicator = micBtn.parentElement.querySelector('.stt-duration');
    if (!indicator) {
        indicator = document.createElement('span');
        indicator.className = 'stt-duration';
        micBtn.parentElement.style.position = 'relative';
        micBtn.parentElement.appendChild(indicator);
    }
    indicator.textContent = '0:00';
    indicator.style.opacity = '1';
    var start = Date.now();
    _sttDurationInterval = setInterval(function() {
        var elapsed = Math.floor((Date.now() - start) / 1000);
        var m = Math.floor(elapsed / 60);
        var s = elapsed % 60;
        indicator.textContent = m + ':' + (s < 10 ? '0' : '') + s;
        if (elapsed >= 100) indicator.style.color = '#f59e0b';
    }, 1000);
}

function _sttStopDuration() {
    if (_sttDurationInterval) {
        clearInterval(_sttDurationInterval);
        _sttDurationInterval = null;
    }
    document.querySelectorAll('.stt-duration').forEach(function(el) {
        el.style.opacity = '0';
    });
}

// ── UI state ────────────────────────────────────────────────

function _sttSetState(btn, state, errorMsg) {
    btn.classList.remove('stt-recording', 'stt-processing', 'stt-error');
    btn.disabled = false;
    var ico = btn.querySelector('svg') ? null : null; // using innerHTML
    switch (state) {
        case 'idle':
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>';
            btn.title = 'Voice input (click to record)';
            break;
        case 'recording':
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
            btn.classList.add('stt-recording');
            btn.title = 'Recording… click to stop';
            break;
        case 'processing':
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>';
            btn.classList.add('stt-processing');
            btn.disabled = true;
            btn.title = 'Transcribing…';
            break;
        case 'error':
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>';
            btn.classList.add('stt-error');
            btn.title = errorMsg || 'Recording failed';
            setTimeout(function() {
                btn.classList.remove('stt-error');
                btn.title = 'Voice input (click to record)';
            }, 2000);
            break;
    }
}

// ── Cleanup ─────────────────────────────────────────────────

function _sttCleanup() {
    _sttRecording = false;
    _sttStopDuration();
    if (_sttMaxDurationTimer) {
        clearTimeout(_sttMaxDurationTimer);
        _sttMaxDurationTimer = null;
    }
    if (_sttStream) {
        _sttStream.getTracks().forEach(function(t) { t.stop(); });
        _sttStream = null;
    }
    _sttMediaRecorder = null;
    _sttAudioChunks = [];
}

// ── Hold-to-record support ──────────────────────────────────

function _sttBindHold(micBtn) {
    micBtn.addEventListener('pointerdown', function(e) {
        e.preventDefault();
        _sttHoldTimer = setTimeout(async function() {
            _sttIsHoldMode = true;
            if (!_sttRecording) await _sttStartRecording(micBtn);
        }, 300);
    });

    micBtn.addEventListener('pointerup', function(e) {
        e.preventDefault();
        if (_sttHoldTimer) { clearTimeout(_sttHoldTimer); _sttHoldTimer = null; }
        if (_sttIsHoldMode && _sttRecording) {
            _sttIsHoldMode = false;
            _sttStopRecording(micBtn);
        } else if (!_sttIsHoldMode) {
            _sttToggle(micBtn);
        }
    });

    micBtn.addEventListener('pointerleave', function() {
        if (_sttHoldTimer) { clearTimeout(_sttHoldTimer); _sttHoldTimer = null; }
    });

    micBtn.addEventListener('contextmenu', function(e) { e.preventDefault(); });
}

// ── Settings toggle ─────────────────────────────────────────

function _toggleSTT(enabled) {
    if (enabled) {
        localStorage.removeItem('stratos_stt_enabled');
        document.body.classList.remove('stt-disabled');
    } else {
        localStorage.setItem('stratos_stt_enabled', '0');
        document.body.classList.add('stt-disabled');
    }
}

// ── Init: inject mic button into agent input ────────────────

(function _sttInit() {
    // Restore disabled state
    if (localStorage.getItem('stratos_stt_enabled') === '0') {
        document.body.classList.add('stt-disabled');
    }

    // Wait for DOM — agent input is built by agent.js (deferred)
    function inject() {
        var sendBtn = document.getElementById('agent-send-btn');
        if (!sendBtn) return setTimeout(inject, 300);

        // Don't double-inject
        if (document.getElementById('agent-mic-btn')) return;

        var micBtn = document.createElement('button');
        micBtn.id = 'agent-mic-btn';
        micBtn.className = 'agent-mic-btn';
        micBtn.type = 'button';
        micBtn.title = 'Voice input (click to record)';
        micBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>';

        // Insert before send button
        sendBtn.parentElement.insertBefore(micBtn, sendBtn);

        // Bind hold-to-record + click-to-toggle
        _sttBindHold(micBtn);

        // Check availability — hide if unavailable
        _sttCheckAvailability().then(function(r) {
            if (!r.available) {
                micBtn.style.display = 'none';
                console.log('STT unavailable: ' + r.reason);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { setTimeout(inject, 200); });
    } else {
        setTimeout(inject, 200);
    }
})();
