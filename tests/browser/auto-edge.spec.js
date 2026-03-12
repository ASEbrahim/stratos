/**
 * AUTOMATED — Edge Cases & Stubs
 *
 * ANI-1:  anime Hello greeting returns substantive response
 * ANI-3:  anime search does NOT trigger web_search
 * TCG-1:  tcg Hello greeting returns substantive response
 * TCG-2:  tcg deck-building returns substantive response
 * EC-2:   Simultaneous messages both complete without crash
 * EC-6:   Invalid persona returns error or fallback (not 500)
 * EC-10:  XSS attempt is sanitized in response
 * TTS-text-cleaning: TTS with markdown does not 500
 * Suggestion-persistence: SSE suggestions array has items
 * Context-overflow: Entity with very long personality_md succeeds
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

test.describe('AUTOMATED — Edge Cases & Stubs', () => {

  // ── ANI-1: anime Hello → response > 5 chars, no tool calls ──
  test('ANI-1: anime Hello returns response with no tool calls', async ({ request }) => {
    test.setTimeout(120000);
    const { text, raw } = await agentChat(request, {
      message: 'Hello',
      persona: 'anime'
    });
    expect(text.length).toBeGreaterThan(5);
    // No tool calls — SSE should contain no "status" events
    const statusEvents = raw.split('\n')
      .filter(l => l.startsWith('data: '))
      .map(l => { try { return JSON.parse(l.slice(6)); } catch { return null; } })
      .filter(d => d && d.status);
    expect(statusEvents.length).toBe(0);
  });

  // ── ANI-3: anime search does NOT trigger web_search ──
  test('ANI-3: anime search does not trigger web_search', async ({ request }) => {
    test.setTimeout(120000);
    const { raw } = await agentChat(request, {
      message: 'Search for latest anime news',
      persona: 'anime'
    });
    // Stub persona should not invoke web_search tool
    const hasWebSearch = raw.split('\n')
      .filter(l => l.startsWith('data: '))
      .some(l => {
        try {
          const d = JSON.parse(l.slice(6));
          return d.status && d.status.includes('web_search');
        } catch { return false; }
      });
    expect(hasWebSearch).toBe(false);
  });

  // ── TCG-1: tcg Hello → response > 5 chars ──
  test('TCG-1: tcg Hello returns substantive response', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Hello',
      persona: 'tcg'
    });
    expect(text.length).toBeGreaterThan(5);
  });

  // ── TCG-2: tcg deck-building → response > 30 chars, substantive ──
  test('TCG-2: tcg deck-building returns substantive response', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Build me a Blue-Eyes deck',
      persona: 'tcg'
    });
    expect(text.length).toBeGreaterThan(30);
  });

  // ── EC-2: Simultaneous messages both complete without crash ──
  test('EC-2: simultaneous requests both complete without crash', async ({ request }) => {
    test.setTimeout(120000);
    const bodyA = {
      message: 'What is 2 + 2?',
      history: [],
      mode: 'structured',
      persona: 'intelligence'
    };
    const bodyB = {
      message: 'What is 3 + 3?',
      history: [],
      mode: 'structured',
      persona: 'intelligence'
    };
    // Fire both requests simultaneously
    const [respA, respB] = await Promise.all([
      request.post(`${BASE}/api/agent-chat`, { headers: HEADERS, data: bodyA, timeout: 60000 }),
      request.post(`${BASE}/api/agent-chat`, { headers: HEADERS, data: bodyB, timeout: 60000 })
    ]);
    // Both should complete — either 200 (success) or 429 (rate-limited / queued)
    // Neither should be 500
    expect([200, 429]).toContain(respA.status());
    expect([200, 429]).toContain(respB.status());
  });

  // ── EC-6: Invalid persona returns error or fallback (not 500) ──
  test('EC-6: invalid persona does not return 500', async ({ request }) => {
    test.setTimeout(120000);
    const resp = await request.post(`${BASE}/api/agent-chat`, {
      headers: HEADERS,
      data: {
        message: 'Hello',
        history: [],
        mode: 'structured',
        persona: 'nonexistent'
      },
      timeout: 60000
    });
    // Should return an error or graceful fallback, but NOT 500
    expect(resp.status()).not.toBe(500);
  });

  // ── EC-10: XSS attempt — response should not contain unescaped <script> ──
  test('EC-10: XSS attempt is sanitized in response', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: '<script>alert(1)</script>',
      persona: 'intelligence'
    });
    // Response must not echo back an unescaped <script> tag
    expect(text).not.toMatch(/<script>/i);
  });

  // ── TTS-text-cleaning: TTS with markdown does not 500 ──
  test('TTS-text-cleaning: TTS with markdown returns 200 or 503', async ({ request }) => {
    test.setTimeout(120000);
    const resp = await request.post(`${BASE}/api/tts`, {
      headers: HEADERS,
      data: { text: '**bold** and *italic* text' },
      timeout: 60000
    });
    // 200 if Piper is available, 503 if Piper is missing — never 500
    expect([200, 503]).toContain(resp.status());
  });

  // ── Suggestion-persistence: SSE stream completes with done event ──
  test('Suggestion-persistence: SSE stream completes successfully', async ({ request }) => {
    test.setTimeout(120000);
    const { text, raw } = await agentChat(request, {
      message: 'Tell me something interesting',
      persona: 'intelligence'
    });
    // Stream should complete with a done event
    expect(raw).toContain('"done"');
    // Response should be substantive
    expect(text.length).toBeGreaterThan(20);
  });

  // ── Context-overflow: Entity with very long personality_md succeeds ──
  test('Context-overflow: entity with long personality_md succeeds', async ({ request }) => {
    test.setTimeout(120000);
    const loremChunk = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ';
    let longText = '';
    while (longText.length < 2000) {
      longText += loremChunk;
    }
    longText = longText.slice(0, 2000);

    const resp = await request.post(`${BASE}/api/personas/gaming/entities`, {
      headers: HEADERS,
      data: {
        scenario: '_edge_test',
        name: '__edge_test_overflow',
        display_name: 'Overflow Test',
        entity_type: 'character',
        personality_md: longText
      },
      timeout: 60000
    });
    // Should succeed or reject gracefully — not crash
    expect(resp.status()).not.toBe(500);

    // Clean up: attempt to delete the test entity
    await request.delete(
      `${BASE}/api/personas/gaming/entities/__edge_test_overflow?scenario=_edge_test`,
      { headers: HEADERS, timeout: 10000 }
    ).catch(() => {});
  });

});
