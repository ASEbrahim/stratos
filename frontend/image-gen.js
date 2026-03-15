/**
 * image-gen.js — StratOS Image Generation Panel
 *
 * Provides text-to-image generation via ComfyUI backend.
 * Accessible from nav sidebar or character card detail.
 * Two models: FLUX (SFW) and Pony V7 (NSFW).
 */

// ── State ──
let _igGenerating = false;
let _igGallery = [];

function _igHeaders() {
    const token = localStorage.getItem('stratos_auth_token') || '';
    return { 'Content-Type': 'application/json', 'X-Auth-Token': token };
}

// ── Panel Toggle ──
function toggleImageGenPanel() {
    let panel = document.getElementById('image-gen-panel');
    if (panel) {
        panel.classList.toggle('hidden');
        return;
    }
    // Create panel
    panel = document.createElement('div');
    panel.id = 'image-gen-panel';
    panel.className = 'fixed inset-0 z-50 flex items-center justify-center';
    panel.style.background = 'rgba(0,0,0,0.6)';
    panel.onclick = (e) => { if (e.target === panel) panel.classList.add('hidden'); };

    panel.innerHTML = `
    <div class="rounded-2xl shadow-2xl w-[500px] max-h-[90vh] overflow-y-auto" style="background:var(--bg-primary); border:1px solid var(--border-subtle);">
        <div class="flex items-center justify-between px-5 py-4" style="border-bottom:1px solid var(--border-subtle);">
            <div class="flex items-center gap-2">
                <i data-lucide="wand-2" class="w-4 h-4" style="color:var(--accent-primary);"></i>
                <span class="text-sm font-semibold" style="color:var(--text-heading);">Image Generation</span>
            </div>
            <button onclick="document.getElementById('image-gen-panel').classList.add('hidden')" class="p-1 rounded-lg hover:opacity-70">
                <i data-lucide="x" class="w-4 h-4" style="color:var(--text-muted);"></i>
            </button>
        </div>
        <div class="p-5 space-y-4">
            <div>
                <label class="text-[10px] font-semibold uppercase tracking-wide mb-1 block" style="color:var(--text-muted);">Prompt</label>
                <textarea id="ig-prompt" rows="3" class="w-full rounded-lg px-3 py-2 text-xs resize-none"
                    style="background:var(--bg-secondary); color:var(--text-body); border:1px solid var(--border-subtle);"
                    placeholder="Describe what you want to generate..."></textarea>
            </div>
            <div class="flex gap-3">
                <div class="flex-1">
                    <label class="text-[10px] font-semibold uppercase tracking-wide mb-1 block" style="color:var(--text-muted);">Style</label>
                    <select id="ig-style" class="w-full rounded-lg px-3 py-1.5 text-xs" style="background:var(--bg-secondary); color:var(--text-body); border:1px solid var(--border-subtle);">
                        <option value="anime">Anime</option>
                        <option value="realistic">Realistic</option>
                        <option value="illustration">Illustration</option>
                    </select>
                </div>
                <div class="flex-1">
                    <label class="text-[10px] font-semibold uppercase tracking-wide mb-1 block" style="color:var(--text-muted);">Size</label>
                    <select id="ig-size" class="w-full rounded-lg px-3 py-1.5 text-xs" style="background:var(--bg-secondary); color:var(--text-body); border:1px solid var(--border-subtle);">
                        <option value="768x1024">Portrait (768×1024)</option>
                        <option value="1024x1024">Square (1024×1024)</option>
                        <option value="1024x768">Landscape (1024×768)</option>
                    </select>
                </div>
            </div>
            <button id="ig-generate-btn" onclick="_igGenerate()" class="w-full rounded-lg py-2.5 text-xs font-semibold flex items-center justify-center gap-2" style="background:var(--accent-primary); color:#fff;">
                <i data-lucide="wand-2" class="w-3.5 h-3.5"></i>
                Generate
            </button>
            <div id="ig-status" class="text-center text-[10px] hidden" style="color:var(--text-muted);"></div>
            <div id="ig-result" class="hidden">
                <img id="ig-result-img" class="w-full rounded-lg" style="border:1px solid var(--border-subtle);">
            </div>
            <div id="ig-gallery-section">
                <label class="text-[10px] font-semibold uppercase tracking-wide mb-2 block" style="color:var(--text-muted);">Recent</label>
                <div id="ig-gallery" class="grid grid-cols-3 gap-2"></div>
            </div>
        </div>
    </div>`;

    document.body.appendChild(panel);
    lucide.createIcons();
    _igLoadGallery();
}

async function _igGenerate() {
    if (_igGenerating) return;
    const prompt = document.getElementById('ig-prompt')?.value?.trim();
    if (!prompt) return;

    _igGenerating = true;
    const btn = document.getElementById('ig-generate-btn');
    const status = document.getElementById('ig-status');
    const result = document.getElementById('ig-result');

    btn.innerHTML = '<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i> Generating... ~5-30s';
    btn.style.opacity = '0.6';
    status.textContent = '';
    status.classList.add('hidden');
    result.classList.add('hidden');
    lucide.createIcons();

    const size = document.getElementById('ig-size')?.value || '768x1024';
    const [w, h] = size.split('x').map(Number);

    try {
        const resp = await fetch('/api/image/generate', {
            method: 'POST',
            headers: _igHeaders(),
            body: JSON.stringify({ prompt, model: 'flux', width: w, height: h }),
        });
        const data = await resp.json();

        if (data.success && data.image_id) {
            const img = document.getElementById('ig-result-img');
            img.src = `/api/image/${data.image_id}`;
            result.classList.remove('hidden');
            status.textContent = `Generated: ${data.image_id} (${data.model}, ${data.size})`;
            status.classList.remove('hidden');
            _igLoadGallery();
        } else {
            status.textContent = data.error || 'Generation failed. Is ComfyUI running?';
            status.style.color = '#f87171';
            status.classList.remove('hidden');
        }
    } catch (e) {
        status.textContent = 'Connection failed — is the server running?';
        status.style.color = '#f87171';
        status.classList.remove('hidden');
    } finally {
        _igGenerating = false;
        btn.innerHTML = '<i data-lucide="wand-2" class="w-3.5 h-3.5"></i> Generate';
        btn.style.opacity = '1';
        lucide.createIcons();
    }
}

async function _igLoadGallery() {
    try {
        const resp = await fetch('/api/image/gallery', { headers: _igHeaders() });
        const data = await resp.json();
        const gallery = document.getElementById('ig-gallery');
        if (!gallery) return;

        const images = data.images || [];
        if (images.length === 0) {
            gallery.innerHTML = '<div class="col-span-3 text-center text-[10px] py-4" style="color:var(--text-muted);">No images yet</div>';
            return;
        }

        gallery.innerHTML = images.slice(0, 9).map(img => `
            <div class="aspect-square rounded-lg overflow-hidden cursor-pointer hover:ring-2 hover:ring-white/20 transition-all" style="border:1px solid var(--border-subtle);">
                <img src="/api/image/${img.id}" class="w-full h-full object-cover" loading="lazy">
            </div>
        `).join('');
    } catch {
        // Silent — gallery is non-critical
    }
}

// ── Character Portrait shortcut ──
async function generateCharacterPortrait(name, description, cardId) {
    toggleImageGenPanel();
    const promptEl = document.getElementById('ig-prompt');
    if (promptEl) {
        promptEl.value = `${name}: ${description}`;
    }
    // Auto-generate
    setTimeout(() => _igGenerate(), 300);
}
