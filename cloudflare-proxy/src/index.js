/**
 * StratOS Media Proxy — Cloudflare Worker
 * 
 * Proxies requests to ISP-blocked domains and serves media/RSS content with CORS.
 * Runs on Cloudflare edge (outside Kuwait ISP) → bypasses TLS-level blocks.
 * 
 * Routes:
 *   /proxy?url=<encoded>   — generic media/image proxy
 *   /feed?url=<encoded>    — RSS feed fetch proxy
 *   /health                — health check
 * 
 * Deploy: npx wrangler deploy
 * Free tier: 100K requests/day
 */

// ── Domain allowlists ──────────────────────────────────────────────
// Only these domains can be proxied (prevents open relay abuse)

const PROXY_ALLOWED_DOMAINS = new Set([
    // Image boards (ISP-blocked in Kuwait)
    'yande.re',
    'danbooru.donmai.us',
    'gelbooru.com',
    'konachan.com',
    'safebooru.org',
    'f95zone.to',
    'chan.sankakucomplex.com',
    // CDNs used by the above
    'files.yande.re',
    'assets.yande.re',
    'raikou1.donmai.us',
    'raikou2.donmai.us',
    'cdn.donmai.us',
    'img1.gelbooru.com',
    'img2.gelbooru.com',
    'img3.gelbooru.com',
    'konachan.com',
    'safebooru.org',
    // Manga
    'mangadex.org',
    'uploads.mangadex.org',
    'api.mangadex.org',
    // Anime tracking
    'myanimelist.net',
    'cdn.myanimelist.net',
    'anilist.co',
    // Twitch thumbnails
    'static-cdn.jtvnw.net',
    'clips-media-assets2.twitch.tv',
]);

export default {
    async fetch(request, env) {
        const url = new URL(request.url);
        const origin = request.headers.get('Origin') || '';
        const allowedOrigins = (env.ALLOWED_ORIGINS || '').split(',').map(s => s.trim());

        // CORS preflight
        if (request.method === 'OPTIONS') {
            return new Response(null, {
                status: 204,
                headers: corsHeaders(origin, allowedOrigins),
            });
        }

        // Only allow GET
        if (request.method !== 'GET') {
            return jsonResp({ error: 'Method not allowed' }, 405, origin, allowedOrigins);
        }

        // ── /health ──
        if (url.pathname === '/health') {
            return jsonResp({ status: 'ok', ts: Date.now() }, 200, origin, allowedOrigins);
        }

        // ── /captions — proxy a timedtext URL through CF edge (bypasses IP blocks) ──
        if (url.pathname === '/captions') {
            const timedtextUrl = url.searchParams.get('url');
            if (!timedtextUrl) {
                return jsonResp({ error: 'Missing url parameter (timedtext base URL)' }, 400, origin, allowedOrigins);
            }

            try {
                const target = new URL(timedtextUrl);
                // Only allow youtube.com timedtext URLs
                if (!target.hostname.includes('youtube.com') && !target.hostname.includes('google.com')) {
                    return jsonResp({ error: 'Only YouTube timedtext URLs allowed' }, 403, origin, allowedOrigins);
                }

                const capResp = await fetch(timedtextUrl, {
                    headers: {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                        'Accept-Language': 'en-US,en;q=0.9',
                    },
                });
                const capText = await capResp.text();

                // Parse XML: <text start="1.23" dur="4.56">caption text</text>
                const captions = [];
                const regex = /<text start="([^"]*)" dur="([^"]*)"[^>]*>([\s\S]*?)<\/text>/g;
                let m;
                while ((m = regex.exec(capText)) !== null) {
                    const text = m[3].replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&#39;/g,"'").replace(/&quot;/g,'"').replace(/<[^>]+>/g,'').trim();
                    if (text) {
                        captions.push({
                            start: Math.round(parseFloat(m[1]) * 100) / 100,
                            duration: Math.round(parseFloat(m[2]) * 100) / 100,
                            text,
                        });
                    }
                }

                const responseHeaders = new Headers();
                Object.entries(corsHeaders(origin, allowedOrigins)).forEach(([k, v]) => responseHeaders.set(k, v));
                responseHeaders.set('Content-Type', 'application/json');
                responseHeaders.set('Cache-Control', 'public, max-age=3600');

                return new Response(JSON.stringify({ captions, count: captions.length }), { status: 200, headers: responseHeaders });
            } catch (err) {
                return jsonResp({ error: 'Caption proxy failed: ' + err.message }, 502, origin, allowedOrigins);
            }
        }

        // ── /proxy and /feed ──
        if (url.pathname === '/proxy' || url.pathname === '/feed') {
            const targetUrl = url.searchParams.get('url');
            if (!targetUrl) {
                return jsonResp({ error: 'Missing url parameter' }, 400, origin, allowedOrigins);
            }

            let target;
            try {
                target = new URL(targetUrl);
            } catch {
                return jsonResp({ error: 'Invalid url parameter' }, 400, origin, allowedOrigins);
            }

            const domain = target.hostname.replace(/^www\./, '');

            // Security: only proxy allowed domains
            if (!isAllowedDomain(domain)) {
                return jsonResp(
                    { error: 'Domain not in allowlist', domain },
                    403, origin, allowedOrigins
                );
            }

            try {
                const isFeed = url.pathname === '/feed';
                const resp = await fetch(targetUrl, {
                    headers: {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                        'Accept': isFeed
                            ? 'application/rss+xml, application/atom+xml, application/xml, text/xml, */*'
                            : 'image/avif, image/webp, image/apng, image/*, */*;q=0.8',
                        'Referer': target.origin + '/',
                        'Accept-Language': 'en-US,en;q=0.9',
                    },
                    cf: {
                        cacheTtl: isFeed ? 300 : 3600,     // 5min feeds, 1hr images
                        cacheEverything: true,
                    },
                });

                // Build response with CORS headers
                const responseHeaders = new Headers();
                // Copy content headers
                const ct = resp.headers.get('Content-Type');
                if (ct) responseHeaders.set('Content-Type', ct);
                const cl = resp.headers.get('Content-Length');
                if (cl) responseHeaders.set('Content-Length', cl);

                // CORS
                Object.entries(corsHeaders(origin, allowedOrigins)).forEach(([k, v]) => {
                    responseHeaders.set(k, v);
                });

                // Cache on browser side
                if (isImageContent(ct)) {
                    responseHeaders.set('Cache-Control', 'public, max-age=86400'); // 24hr for images
                } else if (isFeed) {
                    responseHeaders.set('Cache-Control', 'public, max-age=300');
                }

                return new Response(resp.body, {
                    status: resp.status,
                    headers: responseHeaders,
                });
            } catch (err) {
                return jsonResp(
                    { error: 'Upstream fetch failed', detail: err.message },
                    502, origin, allowedOrigins
                );
            }
        }

        // ── Fallback ──
        return new Response('StratOS Media Proxy\n\nRoutes:\n  /proxy?url=<url>\n  /feed?url=<url>\n  /health', {
            status: 200,
            headers: { 'Content-Type': 'text/plain' },
        });
    },
};

// ── Helpers ────────────────────────────────────────────────────────

function isAllowedDomain(domain) {
    for (const allowed of PROXY_ALLOWED_DOMAINS) {
        if (domain === allowed || domain.endsWith('.' + allowed)) return true;
    }
    return false;
}

function isImageContent(ct) {
    if (!ct) return false;
    return ct.startsWith('image/') || ct.includes('jpeg') || ct.includes('png') ||
           ct.includes('webp') || ct.includes('gif') || ct.includes('avif');
}

function corsHeaders(origin, allowed) {
    const headers = {
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Accept',
        'Access-Control-Max-Age': '86400',
    };
    if (allowed.some(a => a && origin.startsWith(a)) ||
        origin.includes('localhost') || origin.includes('127.0.0.1')) {
        headers['Access-Control-Allow-Origin'] = origin;
    } else if (allowed.length === 0 || (allowed.length === 1 && !allowed[0])) {
        headers['Access-Control-Allow-Origin'] = '*';
    }
    return headers;
}

function jsonResp(obj, status, origin, allowed) {
    return new Response(JSON.stringify(obj), {
        status,
        headers: {
            'Content-Type': 'application/json',
            ...corsHeaders(origin, allowed),
        },
    });
}
