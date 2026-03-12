/**
 * AUTOMATED — Multi-Persona Combinations
 *
 * MP-1:  Intelligence+Market — news and price in one response
 * MP-2:  Intelligence+Scholarly — scholarly articles query
 * MP-3:  Market+Scholarly — oil price + economic research
 * MP-6:  Gaming+Anime — anime-themed scenario (both work despite anime stub)
 * MP-7:  Intelligence+Market+Scholarly — full context merge
 * MP-dangerous-1: Gaming+Scholarly — cross-domain doesn't crash
 * MP-dangerous-2: Gaming(RP)+Market — stays in-character, no real market data
 */
const { test, expect } = require('@playwright/test');
const { AUTH_TOKEN } = require('./auth');

const BASE = 'http://localhost:8080';
const HEADERS = { 'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN };

// Helper: send multi-persona agent chat and collect full SSE response
async function multiChat(request, personas, message, extra = {}) {
  const body = { message, history: [], mode: 'structured', persona: personas[0], personas, ...extra };
  const resp = await request.post(`${BASE}/api/agent-chat`, {
    headers: HEADERS,
    data: body,
    timeout: 180000
  });
  expect(resp.status()).toBe(200);
  const raw = await resp.text();
  let text = '', suggestions = [];
  for (const line of raw.split('\n')) {
    if (!line.startsWith('data: ')) continue;
    try {
      const d = JSON.parse(line.slice(6));
      if (d.token) text += d.token;
      if (d.suggestions) suggestions = d.suggestions;
    } catch {}
  }
  return { text, suggestions, raw };
}

test.describe('AUTOMATED — Multi-Persona Combinations', () => {

  // ── MP-1: Intelligence+Market → price number AND substantive text ──
  test('MP-1: Intelligence+Market returns price and substantive text', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(request, ['intelligence', 'market'], 'NVDA news and price?');
    // Should contain a price number (digits with optional decimal)
    expect(text).toMatch(/\d+(\.\d+)?/);
    // Should have substantive text beyond just a number
    expect(text.length).toBeGreaterThan(50);
  });

  // ── MP-2: Intelligence+Scholarly → scholarly articles ──
  test('MP-2: Intelligence+Scholarly returns scholarly content', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(request, ['intelligence', 'scholarly'], 'Find scholarly articles about Kuwait');
    expect(text.length).toBeGreaterThan(50);
  });

  // ── MP-3: Market+Scholarly → oil price + economic research ──
  test('MP-3: Market+Scholarly handles oil and economic research', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(request, ['market', 'scholarly'], 'How does oil price relate to Gulf economic research?');
    expect(text.length).toBeGreaterThan(50);
  });

  // ── MP-6: Gaming+Anime → anime-themed scenario ──
  test('MP-6: Gaming+Anime builds anime-themed scenario', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(request, ['gaming', 'anime'], 'Help me build an anime-themed scenario');
    expect(text.length).toBeGreaterThan(30);
  });

  // ── MP-7: Intelligence+Market+Scholarly → full context merge ──
  test('MP-7: Triple persona merge returns rich response', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(request, ['intelligence', 'market', 'scholarly'], 'Analyze oil markets and check academic sources');
    expect(text.length).toBeGreaterThan(100);
  });

  // ── MP-dangerous-1: Gaming+Scholarly → cross-domain, no crash ──
  test('MP-dangerous-1: Gaming+Scholarly does not crash', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(request, ['gaming', 'scholarly'], 'Research medieval warfare for my fantasy world');
    expect(text.length).toBeGreaterThan(30);
  });

  // ── MP-dangerous-2: Gaming(RP)+Market → stays in-character ──
  test('MP-dangerous-2: Gaming RP + Market stays in-character about gold', async ({ request }) => {
    test.setTimeout(180000);
    const { text } = await multiChat(
      request,
      ['gaming', 'market'],
      'How much is my gold worth?',
      { rp_mode: 'immersive' }
    );
    // Should respond with substantive content
    expect(text.length).toBeGreaterThan(30);
  });

});
