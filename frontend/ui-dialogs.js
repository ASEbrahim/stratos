// ═══════════════════════════════════════════════════════════
// TOAST NOTIFICATION SYSTEM
// ═══════════════════════════════════════════════════════════

function showToast(message, type = 'info', duration = 3000) {
    // type: 'success' | 'error' | 'warning' | 'info'
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed bottom-6 right-6 z-[999] flex flex-col gap-2 pointer-events-none';
        document.body.appendChild(container);
    }

    const colors = {
        success: { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)', text: '#34d399', icon: 'check-circle' },
        error:   { bg: 'rgba(239,68,68,0.12)',   border: 'rgba(239,68,68,0.3)',   text: '#f87171', icon: 'x-circle' },
        warning: { bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.3)',  text: '#fbbf24', icon: 'alert-triangle' },
        info:    { bg: 'rgba(96,165,250,0.12)',   border: 'rgba(96,165,250,0.3)',  text: '#60a5fa', icon: 'info' }
    };
    const c = colors[type] || colors.info;

    const toast = document.createElement('div');
    toast.className = 'pointer-events-auto flex items-center gap-2.5 px-4 py-2.5 rounded-lg text-xs font-medium backdrop-blur-md shadow-lg transition-all duration-300';
    toast.style.cssText = `background:${c.bg}; border:1px solid ${c.border}; color:${c.text}; transform:translateX(120%); opacity:0;`;
    const _e = document.createElement('span'); _e.textContent = message;
    toast.innerHTML = `<i data-lucide="${c.icon}" class="w-3.5 h-3.5 flex-shrink-0"></i> <span>${_e.innerHTML}</span>`;

    container.appendChild(toast);
    if (typeof lucide !== 'undefined') lucide.createIcons();

    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
    });

    setTimeout(() => {
        toast.style.transform = 'translateX(120%)';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ═══════════════════════════════════════════════════════════
// MODAL PROMPT / CONFIRM — replaces browser prompt() & confirm()
// ═══════════════════════════════════════════════════════════

/**
 * stratosPrompt — styled replacement for window.prompt()
 * @param {Object} opts
 *   title       — modal title
 *   fields      — array of { key, label, placeholder?, defaultValue?, optional? }
 *                  OR omit for a single input and use label/placeholder/defaultValue instead
 *   label       — single-field label (shorthand)
 *   placeholder — single-field placeholder
 *   defaultValue — single-field default
 * @returns {Promise<string|object|null>}  null if cancelled, string for single field, object for multi
 */
function stratosPrompt(opts) {
    if (typeof opts === 'string') opts = { label: opts };
    return new Promise(resolve => {
        // Normalise fields
        const fields = opts.fields || [{ key: '_single', label: opts.label || 'Input', placeholder: opts.placeholder || '', defaultValue: opts.defaultValue || '', optional: false }];
        const single = !opts.fields;

        // Backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'stratos-prompt-backdrop';
        backdrop.onclick = (e) => { if (e.target === backdrop) close(null); };

        // Dialog
        const dialog = document.createElement('div');
        dialog.className = 'stratos-prompt-dialog';

        const title = opts.title || (single ? fields[0].label : 'Input');
        dialog.innerHTML = `<div class="sp-title">${title}</div><div class="sp-fields"></div>
            <div class="sp-buttons"><button class="sp-btn sp-cancel">Cancel</button><button class="sp-btn sp-ok">OK</button></div>`;

        const fieldsContainer = dialog.querySelector('.sp-fields');
        fields.forEach((f, i) => {
            const row = document.createElement('div');
            row.className = 'sp-field';
            if (!single || fields.length > 1) {
                const lbl = document.createElement('label');
                lbl.className = 'sp-label';
                lbl.textContent = f.label + (f.optional ? ' (optional)' : '');
                row.appendChild(lbl);
            }
            const input = f.multiline
                ? document.createElement('textarea')
                : document.createElement('input');
            if (!f.multiline) input.type = 'text';
            input.className = 'sp-input';
            if (f.multiline) { input.rows = 4; input.style.resize = 'vertical'; }
            input.placeholder = f.placeholder || '';
            input.value = f.defaultValue || '';
            input.dataset.key = f.key;
            if (i === 0) setTimeout(() => { input.focus(); if (!f.multiline) input.select(); }, 50);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && (!f.multiline || e.ctrlKey)) submit();
                if (e.key === 'Escape') close(null);
            });
            row.appendChild(input);
            fieldsContainer.appendChild(row);
        });

        dialog.querySelector('.sp-cancel').onclick = () => close(null);
        dialog.querySelector('.sp-ok').onclick = submit;

        function submit() {
            const inputs = dialog.querySelectorAll('.sp-input');
            if (single) {
                const val = inputs[0].value;
                if (!val.trim() && !fields[0].optional) { inputs[0].classList.add('sp-shake'); setTimeout(() => inputs[0].classList.remove('sp-shake'), 400); return; }
                close(val);
            } else {
                const result = {};
                let valid = true;
                inputs.forEach((inp, i) => {
                    result[inp.dataset.key] = inp.value;
                    if (!inp.value.trim() && !fields[i].optional) {
                        inp.classList.add('sp-shake'); setTimeout(() => inp.classList.remove('sp-shake'), 400);
                        valid = false;
                    }
                });
                if (valid) close(result);
            }
        }

        function close(val) {
            dialog.style.transform = 'scale(0.95)';
            dialog.style.opacity = '0';
            backdrop.style.opacity = '0';
            setTimeout(() => { backdrop.remove(); resolve(val); }, 150);
        }

        backdrop.appendChild(dialog);
        document.body.appendChild(backdrop);
        requestAnimationFrame(() => { backdrop.style.opacity = '1'; dialog.style.transform = 'scale(1)'; dialog.style.opacity = '1'; });
    });
}
window.stratosPrompt = stratosPrompt;

function stratosConfirm(message, opts = {}) {
    return new Promise(resolve => {
        const backdrop = document.createElement('div');
        backdrop.className = 'stratos-prompt-backdrop';
        backdrop.onclick = (e) => { if (e.target === backdrop) close(false); };

        const dialog = document.createElement('div');
        dialog.className = 'stratos-prompt-dialog';
        dialog.innerHTML = `<div class="sp-title">${opts.title || 'Confirm'}</div>
            <div class="sp-message">${message}</div>
            <div class="sp-buttons"><button class="sp-btn sp-cancel">${opts.cancelText || 'Cancel'}</button><button class="sp-btn sp-ok sp-danger">${opts.okText || 'Confirm'}</button></div>`;

        dialog.querySelector('.sp-cancel').onclick = () => close(false);
        dialog.querySelector('.sp-ok').onclick = () => close(true);
        dialog.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(false); });

        function close(val) {
            dialog.style.transform = 'scale(0.95)';
            dialog.style.opacity = '0';
            backdrop.style.opacity = '0';
            setTimeout(() => { backdrop.remove(); resolve(val); }, 150);
        }

        backdrop.appendChild(dialog);
        document.body.appendChild(backdrop);
        requestAnimationFrame(() => { backdrop.style.opacity = '1'; dialog.style.transform = 'scale(1)'; dialog.style.opacity = '1'; });
        dialog.querySelector('.sp-ok').focus();
    });
}
window.stratosConfirm = stratosConfirm;
