/**
 * AUTOMATED — Scholarly Persona
 *
 * SCH-1:  Hello greeting mentions academic/research/scholarly topic
 * SCH-2:  Battle of Badr returns substantive response (>100 chars)
 * SCH-9:  Web search SSE contains "web_search" status
 * SCH-15: Scholarly does NOT have manage_watchlist — watchlist request refused
 * SCH-narration-rigor: Narration search returns verified/unverified or meaningful response
 * SCH-suggestions: Follow-up suggestions present
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
    persona: opts.persona || 'scholarly',
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

test.describe('AUTOMATED — Scholarly Persona', () => {

  // ── SCH-1: Hello greeting mentions academic/research/scholarly topic ──
  test('SCH-1: Hello mentions academic/research/scholarly topic', async ({ request }) => {
    test.setTimeout(60000);
    const { text } = await agentChat(request, {
      message: 'Hello',
      persona: 'scholarly'
    });
    expect(text.length).toBeGreaterThan(0);
    expect(text).toMatch(/academic|research|scholar|knowledge|study|learn|question|help|literature|source/i);
  });

  // ── SCH-2: Battle of Badr returns substantive response ──
  test('SCH-2: Battle of Badr returns substantive response', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Tell me about the Battle of Badr',
      persona: 'scholarly'
    });
    expect(text.length).toBeGreaterThan(100);
  });

  // ── SCH-9: Web search SSE contains web_search status ──
  test('SCH-9: Web search SSE contains web_search status', async ({ request }) => {
    test.setTimeout(60000);
    const { raw } = await agentChat(request, {
      message: 'Search the web for recent papers on quantum computing',
      persona: 'scholarly'
    });
    expect(raw).toContain('web_search');
  });

  // ── SCH-15: Scholarly does NOT have manage_watchlist ──
  test('SCH-15: Watchlist request is refused', async ({ request }) => {
    test.setTimeout(60000);
    const { text } = await agentChat(request, {
      message: 'Add TSLA to watchlist',
      persona: 'scholarly'
    });
    // Response should indicate inability — cannot, don't have, not available, etc.
    expect(text).toMatch(/can'?t|cannot|don'?t|unable|not available|not supported|doesn'?t|outside|beyond|no.+ability|not.+capable/i);
  });

  // ── SCH-narration-rigor: Narration search returns meaningful scholarly response ──
  test('SCH-narration-rigor: Narration search rigor check', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Find narrations about patience',
      persona: 'scholarly'
    });
    // Should return a substantive scholarly response about narrations/patience
    expect(text.length).toBeGreaterThan(50);
    // Should reference narrations, patience, or related scholarly concepts
    expect(text).toMatch(/narration|hadith|patience|sabr|found|search|database/i);
  });

  // ── SCH-suggestions: Follow-up suggestions present in SSE ──
  test('SCH-suggestions: Follow-up suggestions present', async ({ request }) => {
    test.setTimeout(120000);
    const { raw } = await agentChat(request, {
      message: 'Tell me about the Battle of Badr',
      persona: 'scholarly'
    });
    // Suggestions field should appear in SSE stream
    expect(raw).toContain('suggestions');
  });

});
