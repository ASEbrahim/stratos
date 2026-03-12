/**
 * SEMI-AUTO TIER — Capture agent outputs to /tmp/agent-test-review/
 *
 * These tests run the agent with specific prompts and save the full response
 * to timestamped files for human review. They PASS if the agent responds
 * without crashing — quality is judged by the reviewer.
 *
 * Run:  npx playwright test tests/browser/semi-auto-capture.spec.js --workers=1
 * Then: ls /tmp/agent-test-review/
 */
const { test, expect } = require('@playwright/test');
const { AUTH_TOKEN } = require('./auth');
const fs = require('fs');
const path = require('path');

const BASE = 'http://localhost:8080';
const HEADERS = { 'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN };
const OUT_DIR = '/tmp/agent-test-review';

// Ensure output directory exists
test.beforeAll(() => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
});

// Helper: send agent chat and collect full SSE response text
async function agentChat(request, opts) {
  const body = {
    message: opts.message,
    history: opts.history || [],
    mode: 'structured',
    persona: opts.persona || 'intelligence',
    ...opts.extra
  };
  if (opts.personas) body.personas = opts.personas;
  const resp = await request.post(`${BASE}/api/agent-chat`, {
    headers: HEADERS,
    data: body,
    timeout: 180000
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

// Helper: save captured output
function saveCapture(filename, data) {
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const file = path.join(OUT_DIR, `${ts}_${filename}.md`);
  const content = [
    `# ${filename}`,
    `Captured: ${new Date().toISOString()}`,
    '',
    '## Prompt',
    data.prompt,
    '',
    '## Response',
    data.text,
    '',
    '## Suggestions',
    (data.suggestions || []).map(s => `- ${s}`).join('\n') || '(none)',
    '',
    '## Review Checklist',
    ...data.checks.map(c => `- [ ] ${c}`),
    ''
  ].join('\n');
  fs.writeFileSync(file, content);
  return file;
}

test.describe('SEMI-AUTO — Capture for Human Review', () => {

  // ══════════════════════════════════════════
  //  Intelligence Quality Captures
  // ══════════════════════════════════════════

  test('SA-INT-1: Intelligence briefing quality', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Give me a full intelligence briefing on the current state of the world.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'intelligence'
    });
    saveCapture('INT-briefing', {
      prompt, text, suggestions,
      checks: [
        'References actual feed/web data (not generic)',
        'Organized with clear sections or bullet points',
        'Mentions specific events, names, or dates',
        'No hallucinated statistics',
        'Professional tone appropriate for intelligence briefing'
      ]
    });
    expect(text.length).toBeGreaterThan(100);
  });

  test('SA-INT-2: Intelligence web search depth', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Search the web for the latest developments in AI regulation worldwide.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'intelligence'
    });
    saveCapture('INT-web-search', {
      prompt, text, suggestions,
      checks: [
        'Cites specific sources or articles',
        'Covers multiple countries/regions',
        'Information is current (not outdated)',
        'Balanced coverage (not just US-centric)',
        'Suggestions are relevant follow-ups'
      ]
    });
    expect(text.length).toBeGreaterThan(100);
  });

  // ══════════════════════════════════════════
  //  Market Quality Captures
  // ══════════════════════════════════════════

  test('SA-MKT-1: Market portfolio analysis', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Analyze my entire watchlist. Give me a full breakdown of each position.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'market'
    });
    saveCapture('MKT-portfolio', {
      prompt, text, suggestions,
      checks: [
        'Lists actual tickers from watchlist',
        'Includes real price data (not made up)',
        'Mentions percentage changes',
        'Does NOT give direct buy/sell advice',
        'Organized per-ticker or in a table format'
      ]
    });
    expect(text.length).toBeGreaterThan(50);
  });

  test('SA-MKT-2: Market sector comparison', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Compare the tech sector vs energy sector performance this week.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'market'
    });
    saveCapture('MKT-sector-compare', {
      prompt, text, suggestions,
      checks: [
        'References actual sector data or tickers',
        'Compares at least 2 metrics (price, volume, % change)',
        'Does NOT give investment advice',
        'Mentions specific companies or ETFs',
        'Acknowledges limitations of available data'
      ]
    });
    expect(text.length).toBeGreaterThan(50);
  });

  // ══════════════════════════════════════════
  //  Scholarly Quality Captures
  // ══════════════════════════════════════════

  test('SA-SCH-1: Scholarly deep-dive topic', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Explain the theological significance of Surah Al-Kahf and its four main stories.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'scholarly'
    });
    saveCapture('SCH-alkahf', {
      prompt, text, suggestions,
      checks: [
        'Covers all four stories (Sleepers, Garden Owner, Musa & Khidr, Dhul-Qarnayn)',
        'Theological analysis is substantive (not superficial)',
        'Uses appropriate scholarly terminology',
        'References sources (Quran verses, scholars)',
        'Respectful and academically rigorous tone'
      ]
    });
    expect(text.length).toBeGreaterThan(200);
  });

  test('SA-SCH-2: Scholarly narration search', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Find narrations about the importance of seeking knowledge.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'scholarly'
    });
    saveCapture('SCH-narrations-knowledge', {
      prompt, text, suggestions,
      checks: [
        'Cites specific narrations or hadith references',
        'Includes chain of transmission or source if available',
        'Distinguishes verified vs unverified if DB has data',
        'Does NOT fabricate hadith text',
        'Academic rigor in presentation'
      ]
    });
    expect(text.length).toBeGreaterThan(50);
  });

  // ══════════════════════════════════════════
  //  Gaming Quality Captures
  // ══════════════════════════════════════════

  test('SA-GM-1: GM opening narration', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'I am a warrior entering a haunted castle for the first time. Set the scene.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'gaming', extra: { rp_mode: 'gm' }
    });
    saveCapture('GM-opening-narration', {
      prompt, text, suggestions,
      checks: [
        'Third-person narration style',
        'Atmospheric and immersive description',
        'Includes numbered options (1, 2, 3)',
        'Options are meaningful choices (not trivial)',
        'No stat blocks or emoji headers'
      ]
    });
    expect(text.length).toBeGreaterThan(100);
  });

  test('SA-RP-1: Immersive RP conversation', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'I approach the old merchant and ask about the curse on this town.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'gaming', extra: { rp_mode: 'immersive' }
    });
    saveCapture('RP-merchant-dialogue', {
      prompt, text, suggestions,
      checks: [
        'In-character dialogue (first or third person)',
        'NO stat blocks (HP, ATK, DEF)',
        'NO emoji section headers',
        'NO numbered option lists',
        'Engaging narrative with character personality'
      ]
    });
    expect(text.length).toBeGreaterThan(50);
  });

  // ══════════════════════════════════════════
  //  Anime & TCG Quality Captures
  // ══════════════════════════════════════════

  test('SA-ANI-1: Anime recommendation', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'I liked Attack on Titan and Death Note. What should I watch next?';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'anime'
    });
    saveCapture('ANI-recommendation', {
      prompt, text, suggestions,
      checks: [
        'Recommends actual anime titles (not made up)',
        'Explains why each recommendation fits',
        'References genres or themes from the mentioned shows',
        'At least 3 recommendations',
        'Enthusiastic but informative tone'
      ]
    });
    expect(text.length).toBeGreaterThan(50);
  });

  test('SA-TCG-1: TCG deck strategy', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'How do I build a competitive Blue-Eyes White Dragon deck in modern Yu-Gi-Oh?';
    const { text, suggestions } = await agentChat(request, {
      message: prompt, persona: 'tcg'
    });
    saveCapture('TCG-blue-eyes-deck', {
      prompt, text, suggestions,
      checks: [
        'Mentions specific card names',
        'Explains strategy/win condition',
        'Lists core cards vs tech choices',
        'References current meta or format',
        'Organized deck-building advice'
      ]
    });
    expect(text.length).toBeGreaterThan(50);
  });

  // ══════════════════════════════════════════
  //  Multi-Persona Quality Captures
  // ══════════════════════════════════════════

  test('SA-MP-1: Intelligence+Market cross-domain', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'What geopolitical events are affecting oil prices right now?';
    const { text, suggestions } = await agentChat(request, {
      message: prompt,
      persona: 'intelligence',
      personas: ['intelligence', 'market']
    });
    saveCapture('MP-geopolitics-oil', {
      prompt, text, suggestions,
      checks: [
        'References specific geopolitical events',
        'Includes actual oil price data',
        'Connects events to price movements',
        'Uses both intelligence and market data',
        'Balanced analysis without speculation'
      ]
    });
    expect(text.length).toBeGreaterThan(100);
  });

  test('SA-MP-2: Scholarly+Intelligence research', async ({ request }) => {
    test.setTimeout(180000);
    const prompt = 'Research the history of Islamic finance and its modern applications.';
    const { text, suggestions } = await agentChat(request, {
      message: prompt,
      persona: 'scholarly',
      personas: ['scholarly', 'intelligence']
    });
    saveCapture('MP-islamic-finance', {
      prompt, text, suggestions,
      checks: [
        'Covers historical foundations',
        'Mentions modern applications (sukuk, takaful, etc.)',
        'Uses scholarly sources where available',
        'Web search supplements academic knowledge',
        'Substantive and well-organized response'
      ]
    });
    expect(text.length).toBeGreaterThan(100);
  });

});
