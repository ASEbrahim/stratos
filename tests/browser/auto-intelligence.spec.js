/**
 * AUTOMATED — Intelligence Persona
 *
 * INT-1:  Hello greeting mentions at least one capability keyword
 * INT-2:  Feed query returns substantive response
 * INT-5:  Web search SSE contains "web_search" status
 * INT-6:  Feed search SSE contains "search_feed" status
 * INT-7/8: Watchlist add then remove reflects in /api/config
 * INT-14: Any chat SSE stream contains suggestions array
 */
const { test, expect } = require('@playwright/test');
const { AUTH_TOKEN } = require('./auth');

const BASE = 'http://localhost:8080';
const HEADERS = { 'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN };

// Helper: send agent chat and collect full SSE response text
async function agentChat(request, opts) {
  const body = {
    message: opts.message,
    history: opts.history || [],
    mode: 'structured',
    persona: opts.persona || 'intelligence',
    ...opts.extra
  };
  const resp = await request.post(`${BASE}/api/agent-chat`, {
    headers: HEADERS,
    data: body,
    timeout: 120000
  });
  expect(resp.status()).toBe(200);
  const raw = await resp.text();
  // Parse SSE: collect all token payloads
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

test.describe('AUTOMATED — Intelligence Persona', () => {

  // ── INT-1: Hello greeting mentions capabilities ──
  test('INT-1: Hello mentions capability keyword', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Hello',
      persona: 'intelligence'
    });
    // Response should be under 200 words
    const wordCount = text.trim().split(/\s+/).length;
    expect(wordCount).toBeLessThanOrEqual(200);
    // Should mention at least one intelligence-related keyword
    expect(text).toMatch(/search|watchlist|categor|feed|web|news|signal|intel|help|analyz|track|monitor|brief/i);
  });

  // ── INT-2: Feed query returns substantive response ──
  test('INT-2: Feed query returns substantive response', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: "What's in my feed today?",
      persona: 'intelligence'
    });
    expect(text.length).toBeGreaterThan(50);
  });

  // ── INT-5: Web search SSE contains web_search status ──
  test('INT-5: Web search SSE contains web_search status', async ({ request }) => {
    test.setTimeout(120000);
    const { raw } = await agentChat(request, {
      message: 'Search the web for latest NVIDIA earnings',
      persona: 'intelligence'
    });
    // Should contain web_search tool call or search-related status
    expect(raw).toMatch(/web_search|searching|search.*web/i);
  });

  // ── INT-6: Feed search SSE contains search_feed status ──
  test('INT-6: Feed search SSE contains search_feed status', async ({ request }) => {
    test.setTimeout(120000);
    const { raw } = await agentChat(request, {
      message: 'Search my feed for articles about Kuwait',
      persona: 'intelligence'
    });
    // Tool call may appear as search_feed in status or tool_calls
    expect(raw).toMatch(/search_feed|searching.*feed|feed.*search/i);
  });

  // ── INT-7/INT-8: Watchlist add then remove ──
  test('INT-7/INT-8: Watchlist add and remove TSLA', async ({ request }) => {
    test.setTimeout(120000);

    // Step 1: Add TSLA to watchlist
    const { text: addText } = await agentChat(request, {
      message: 'Add TSLA to my watchlist',
      persona: 'intelligence'
    });
    // Response should acknowledge the addition
    expect(addText.toLowerCase()).toMatch(/add|tsla|watchlist|track/i);

    // Verify TSLA is in the config tickers
    const afterAdd = await request.get(`${BASE}/api/config`, { headers: HEADERS });
    expect(afterAdd.status()).toBe(200);
    const configAdd = await afterAdd.json();
    const tickersAdd = (configAdd.market?.tickers || configAdd.tickers || []);
    const hasTSLA = tickersAdd.some(t => {
      const sym = typeof t === 'string' ? t : (t.symbol || t.ticker || '');
      return sym.toUpperCase().includes('TSLA');
    });
    expect(hasTSLA).toBe(true);

    // Step 2: Remove TSLA from watchlist
    await agentChat(request, {
      message: 'Remove TSLA from my watchlist',
      persona: 'intelligence'
    });

    // Verify TSLA is no longer in the config tickers
    const afterRemove = await request.get(`${BASE}/api/config`, { headers: HEADERS });
    expect(afterRemove.status()).toBe(200);
    const configRemove = await afterRemove.json();
    const tickersRemove = (configRemove.market?.tickers || configRemove.tickers || []);
    const stillHasTSLA = tickersRemove.some(t => {
      const sym = typeof t === 'string' ? t : (t.symbol || t.ticker || '');
      return sym.toUpperCase().includes('TSLA');
    });
    expect(stillHasTSLA).toBe(false);
  });

  // ── INT-14: Chat SSE contains suggestions ──
  test('INT-14: Chat SSE contains suggestions array', async ({ request }) => {
    test.setTimeout(120000);
    const { suggestions, raw } = await agentChat(request, {
      message: 'Give me a brief market overview',
      persona: 'intelligence'
    });
    // Suggestions should be present in SSE stream (even if empty array)
    expect(raw).toContain('suggestions');
  });

});
