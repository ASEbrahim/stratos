// STRAT_OS Service Worker — network-first with offline shell caching
const CACHE_NAME = 'stratos-v11';

// App shell — pre-cache these for instant offline loading
const SHELL_ASSETS = [
    '/',
    '/index.html',
    '/styles.css',
    '/tailwind-built.css',
    '/app.js',
    '/auth.js',
    '/nav.js',
    '/ui.js',
    '/feed.js',
    '/market.js',
    '/markets-panel.js',
    '/agent.js',
    '/settings.js',
    '/wizard.js',
    '/scan-history.js',
    '/theme-editor.js',
    '/mobile.js',
    '/icon-192.png',
    '/manifest.json',
];

// Install: pre-cache app shell for offline support
self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(SHELL_ASSETS))
            .then(() => self.skipWaiting())
    );
});

// Activate: claim clients and clean old caches
self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        ).then(() => self.clients.claim())
    );
});

// Fetch: network-first for API calls, cache-first for shell assets
self.addEventListener('fetch', (e) => {
    const url = new URL(e.request.url);

    // Skip SSE, non-GET, and cross-origin
    if (url.pathname === '/api/events' || e.request.method !== 'GET') return;
    if (url.origin !== self.location.origin) return;

    // API calls: network-first, cache fallback
    if (url.pathname.startsWith('/api/')) {
        e.respondWith(
            fetch(e.request)
                .then(resp => {
                    if (resp.ok) {
                        const clone = resp.clone();
                        caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
                    }
                    return resp;
                })
                .catch(() => caches.match(e.request))
        );
        return;
    }

    // Static assets: network-first, cache fallback (ensures fresh code on reload)
    e.respondWith(
        fetch(e.request)
            .then(resp => {
                if (resp.ok) {
                    const clone = resp.clone();
                    caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
                }
                return resp;
            })
            .catch(() => caches.match(e.request))
    );
});
