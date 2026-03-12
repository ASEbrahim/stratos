/**
 * AUTOMATED — Gaming Persona
 *
 * GM-4:  Starting area narration with numbered options
 * GM-5:  Continue narrative after choosing option
 * GM-8:  OOC request breaks out of narrative voice
 * RP-4:  Immersive first-person, no stat blocks or emoji headers
 * RP-8:  Negative assertions on emoji/stat patterns
 * RP-11: No numbered choice blocks in immersive mode
 * Entity CRUD: create → read → verify personality_md → delete
 * Scenario CRUD: create → list → verify → delete
 */
const { test, expect } = require('@playwright/test');
const { AUTH_TOKEN } = require('./auth');

const BASE = 'http://localhost:8080';
const HEADERS = { 'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN };

// Helper: send agent chat and collect full SSE response text
// Supports extra fields (e.g. rp_mode) via opts.extra
async function agentChat(request, opts) {
  const body = {
    message: opts.message,
    history: opts.history || [],
    mode: 'structured',
    persona: opts.persona || 'gaming',
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

test.describe('AUTOMATED — Gaming Persona', () => {

  // ════════════════════════════════════════════════
  //  GM Mode Tests
  // ════════════════════════════════════════════════

  // ── GM-4: Starting area narration with numbered options ──
  test('GM-4: Describe starting area — third-person narration with numbered options', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Describe the starting area',
      persona: 'gaming',
      extra: { rp_mode: 'gm' }
    });
    expect(text.length).toBeGreaterThan(50);
    // Should contain numbered options (1) or 1. or 2) or 2. or 3) or 3.)
    expect(text).toMatch(/[1-3][.)]\s/);
  });

  // ── GM-5: Continue narrative after choosing an option ──
  test('GM-5: Choose option 1 — continues narrative', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'I choose option 1',
      persona: 'gaming',
      extra: { rp_mode: 'gm' },
      history: [
        { role: 'user', content: 'Describe the starting area' },
        { role: 'assistant', content: 'You stand at the edge of a misty forest clearing. Three paths diverge before you:\n1) A narrow trail leading into the dense woods.\n2) A stone bridge crossing a rushing river.\n3) A worn road heading toward distant towers.' }
      ]
    });
    expect(text.length).toBeGreaterThan(30);
  });

  // ── GM-8: OOC request breaks out of narrative voice ──
  test('GM-8: OOC request responds out-of-character', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'OOC: Can you explain the rules?',
      persona: 'gaming',
      extra: { rp_mode: 'gm' }
    });
    expect(text.length).toBeGreaterThan(20);
    // OOC response should not read like in-game narration; it should reference
    // rules, mechanics, or acknowledge the out-of-character request
    expect(text).toMatch(/rule|mechanic|out.of.character|OOC|system|explain/i);
  });

  // ════════════════════════════════════════════════
  //  Immersive RP Tests
  // ════════════════════════════════════════════════

  // ── RP-4: First-person response, no stat blocks, no emoji headers ──
  test('RP-4: Immersive greeting — first-person, no stat blocks, no emoji headers', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Hello, how are you?',
      persona: 'gaming',
      extra: { rp_mode: 'immersive' }
    });
    expect(text.length).toBeGreaterThan(10);
    // Should NOT contain stat block patterns
    expect(text).not.toMatch(/HP:|ATK:|DEF:/);
    // Should NOT start with an emoji
    const firstChar = text.trimStart().codePointAt(0);
    // Emoji range: most emoji are above U+1F000 or in misc symbol blocks
    const startsWithEmoji = firstChar > 0x1F000 ||
      (firstChar >= 0x2600 && firstChar <= 0x27BF) ||
      (firstChar >= 0x1F300 && firstChar <= 0x1FAFF);
    expect(startsWithEmoji).toBe(false);
  });

  // ── RP-8: Negative assertions on forbidden patterns ──
  test('RP-8: Immersive mode — no forbidden emoji or stat patterns', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'Tell me about yourself.',
      persona: 'gaming',
      extra: { rp_mode: 'immersive' }
    });
    // Must NOT contain these specific emoji
    expect(text).not.toContain('\u{1F5E3}');  // 🗣️
    expect(text).not.toContain('\u{1F4DC}');  // 📜
    expect(text).not.toContain('\u{1F3AF}');  // 🎯
    // Must NOT contain stat table patterns
    expect(text).not.toMatch(/HP:|ATK:|DEF:/);
    // Must NOT contain emoji section headers (emoji followed by text as a heading)
    // Pattern: line starting with an emoji then a space and capitalized word
    expect(text).not.toMatch(/^[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}]\s+[A-Z]/mu);
  });

  // ── RP-11: No numbered choice blocks in immersive mode ──
  test('RP-11: Immersive mode — no numbered choice blocks', async ({ request }) => {
    test.setTimeout(120000);
    const { text } = await agentChat(request, {
      message: 'What do you think we should do next?',
      persona: 'gaming',
      extra: { rp_mode: 'immersive' }
    });
    expect(text.length).toBeGreaterThan(10);
    // Must NOT have a "1) ... 2) ... 3)" pattern (numbered choice list)
    expect(text).not.toMatch(/1\)\s+.*\n.*2\)\s+.*\n.*3\)\s+/s);
  });

  // ════════════════════════════════════════════════
  //  Cross-mode: Entity CRUD
  // ════════════════════════════════════════════════

  test('Entity CRUD: create → read → verify personality_md → delete', async ({ request }) => {
    test.setTimeout(120000);
    const entityName = 'test_warrior';
    const scenario = '_auto_test';

    // Create entity
    const create = await request.post(`${BASE}/api/personas/gaming/entities`, {
      headers: HEADERS,
      data: {
        scenario: scenario,
        name: entityName,
        display_name: 'Test Warrior',
        entity_type: 'character',
        personality_md: 'A brave warrior with a heart of gold'
      }
    });
    expect(create.status()).toBe(200);
    const createBody = await create.json();
    expect(createBody.ok).toBe(true);

    // Read entity
    const read = await request.get(
      `${BASE}/api/personas/gaming/entities/${entityName}?scenario=${scenario}`,
      { headers: HEADERS }
    );
    expect(read.status()).toBe(200);
    const entity = await read.json();
    expect(entity.display_name).toBe('Test Warrior');
    expect(entity.personality_md).toBe('A brave warrior with a heart of gold');

    // Delete entity
    const del = await request.delete(
      `${BASE}/api/personas/gaming/entities/${entityName}?scenario=${scenario}`,
      { headers: HEADERS }
    );
    expect(del.status()).toBe(200);
    const delBody = await del.json();
    expect(delBody.ok).toBe(true);
  });

  // ════════════════════════════════════════════════
  //  Cross-mode: Scenario CRUD
  // ════════════════════════════════════════════════

  test('Scenario CRUD: create → list → verify → delete', async ({ request }) => {
    test.setTimeout(120000);
    const scenarioName = 'AutoTest_Scenario';

    // Create scenario
    const create = await request.post(`${BASE}/api/scenarios/create`, {
      headers: HEADERS,
      data: { name: scenarioName }
    });
    expect(create.status()).toBe(200);

    // List scenarios and verify it appears
    const list = await request.get(`${BASE}/api/scenarios`, { headers: HEADERS });
    expect(list.status()).toBe(200);
    const listBody = await list.json();
    const names = (listBody.scenarios || []).map(s => s.name || s);
    expect(names).toContain(scenarioName);

    // Delete scenario
    const del = await request.delete(
      `${BASE}/api/scenarios?name=${encodeURIComponent(scenarioName)}`,
      { headers: HEADERS }
    );
    expect(del.status()).toBe(200);
  });

});
