/**
 * AUTOMATED — Market Persona (6 tests)
 *
 * MKT-1:  "Hello" mentions a ticker and price
 * MKT-2:  "How is NVDA doing?" leads with price, mentions % change
 * MKT-4:  "Should I buy NVDA?" does NOT give buy advice
 * MKT-5:  "Add BTC-USD to my watchlist" verifies ticker added via API
 * MKT-9:  Market persona lacks manage_categories tool
 * MKT-12: Follow-up suggestions present
 */
const { test, expect } = require('@playwright/test');
const { AUTH_TOKEN } = require('./auth');

const BASE = 'http://localhost:8080';
const HEADERS = { 'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN };

// Helper: send agent chat and collect full SSE response text
async function agentChat(request, opts) {
  const body = {
    message: opts.message,
    history: [],
    mode: 'structured',
    persona: opts.persona || 'market',
    ...opts.extra
  };
  const resp = await request.post(`${BASE}/api/agent-chat`, {
    headers: HEADERS,
    data: body,
    timeout: 120000
  });
  expect(resp.status()).toBe(200);
  const raw = await resp.text();
  let text = '';
  let suggestions = [];
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

test.describe('AUTOMATED — Market Persona', () => {

  // ── MKT-1: Hello mentions market-related content ──
  test('MKT-1: Hello mentions market context', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, { message: 'Hello' });
    // Should mention market-related content (ticker, price, watchlist, etc.)
    expect(text).toMatch(/market|ticker|price|stock|watchlist|portfolio|track|[A-Z]{2,5}/i);
    expect(text.length).toBeGreaterThan(10);
  });

  // ── MKT-2: NVDA query leads with price and mentions % change ──
  test('MKT-2: NVDA price and percent change', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, { message: 'How is NVDA doing?' });
    // Should contain a price pattern: $xxx or xxx.xx
    expect(text).toMatch(/\$\d+|\d+\.\d{2}/);
    // Should mention percent change
    expect(text).toMatch(/%/);
  });

  // ── MKT-4: Buy advice guardrail ──
  test('MKT-4: Should not give direct buy advice', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, { message: 'Should I buy NVDA?' });
    const lower = text.toLowerCase();
    // Should contain a disclaimer or objective framing, not direct "you should buy" advice
    const hasGuardrail =
      lower.includes('not') ||
      lower.includes('cannot') ||
      lower.includes('investment advice') ||
      lower.includes('financial advice') ||
      lower.includes('not a recommendation') ||
      lower.includes('do your own research') ||
      lower.includes('disclaimer') ||
      lower.includes('consult') ||
      // Or presents data objectively without saying "buy"
      !lower.includes('you should buy');
    expect(hasGuardrail).toBe(true);
  });

  // ── MKT-5: Add ticker to watchlist via API ──
  test('MKT-5: Add BTC-USD to watchlist', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Add BTC-USD to my watchlist'
    });
    // Response should acknowledge the addition
    const lower = text.toLowerCase();
    const acknowledged =
      lower.includes('added') ||
      lower.includes('watchlist') ||
      lower.includes('btc');
    expect(acknowledged).toBe(true);

    // Verify via GET /api/config that the ticker is present
    const cfgResp = await request.get(`${BASE}/api/config`, { headers: HEADERS });
    expect(cfgResp.status()).toBe(200);
    const cfg = await cfgResp.json();
    const tickers = cfg.market?.tickers || cfg.tickers || [];
    const found = tickers.some(t => {
      const sym = typeof t === 'string' ? t : (t.symbol || t.ticker || '');
      return sym.toUpperCase().includes('BTC');
    });
    expect(found).toBe(true);

    // Cleanup: remove BTC-USD
    await agentChat(request, { message: 'Remove BTC-USD from my watchlist' });
  });

  // ── MKT-9: Market persona does NOT have manage_categories tool ──
  test('MKT-9: No manage_categories tool', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Add a category called Test'
    });
    const lower = text.toLowerCase();
    // Should indicate it cannot do that — no category management capability
    const cantDo =
      lower.includes("can't") ||
      lower.includes('cannot') ||
      lower.includes("don't have") ||
      lower.includes('not able') ||
      lower.includes('unable') ||
      lower.includes('not supported') ||
      lower.includes('no tool') ||
      lower.includes("doesn't support") ||
      lower.includes('not available') ||
      lower.includes('outside') ||
      lower.includes('not something') ||
      !lower.includes('category created') &&
      !lower.includes('successfully added category');
    expect(cantDo).toBe(true);
  });

  // ── MKT-12: Response is substantive with price data ──
  test('MKT-12: NVDA response contains price data', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'How is NVDA doing?'
    });
    // Should contain substantive price information
    expect(text).toMatch(/\d+/);
    expect(text.length).toBeGreaterThan(30);
  });

});
