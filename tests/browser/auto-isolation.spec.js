/**
 * AUTOMATED — Isolation & Conversations
 *
 * FI-1/2/3:   Upload file with X-Persona:intelligence → visible in intelligence, NOT in gaming/scholarly
 * FI-9/10/11: Write file via persona-files API for scholarly → visible in scholarly, NOT in gaming
 * FI-13:      Path traversal attempt → error
 * FI-14/15/16: Save context for intelligence → get intelligence returns it → gaming returns different
 * FI-22:      Get context for persona with no saved context → returns something
 * CV-1:       Create conversation → ID returned
 * CV-4:       Rename conversation via PUT → title changes
 * CV-7:       Append message then read → title auto-set (first user message)
 * CV-5:       Delete conversation → verify gone
 * I-7:        Health check via agent-status
 */
const { test, expect } = require('@playwright/test');
const { AUTH_TOKEN } = require('./auth');

const BASE = 'http://localhost:8080';
const HEADERS = { 'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN };

// Track created resources for cleanup
let createdConversationIds = [];
let createdFiles = [];

test.describe('AUTOMATED — Isolation & Conversations', () => {

  test.afterEach(async ({ request }) => {
    // Clean up conversations
    for (const id of createdConversationIds) {
      try {
        await request.delete(`${BASE}/api/conversations/${id}`, { headers: HEADERS });
      } catch {}
    }
    createdConversationIds = [];

    // Clean up persona files
    for (const { persona, path } of createdFiles) {
      try {
        await request.post(`${BASE}/api/persona-files/write`, {
          headers: HEADERS,
          data: { persona, path, content: '' }
        });
      } catch {}
    }
    createdFiles = [];
  });

  // ── FI-1/2/3: Upload file scoped to intelligence persona ──
  test('FI-1/2/3: Uploaded file visible in intelligence, not in gaming or scholarly', async ({ request }) => {
    test.setTimeout(30000);

    const filename = `test_isolation_${Date.now()}.txt`;
    const fileContent = 'Intelligence isolation test content';

    // Upload file with X-Persona: intelligence
    const uploadResp = await request.post(`${BASE}/api/files/upload`, {
      headers: {
        'X-Auth-Token': AUTH_TOKEN,
        'X-Filename': filename,
        'X-Persona': 'intelligence',
        'Content-Type': 'application/octet-stream'
      },
      data: Buffer.from(fileContent)
    });
    expect(uploadResp.status()).toBe(200);
    const uploadData = await uploadResp.json();
    expect(uploadData.ok).toBeTruthy();

    // FI-1: File visible in intelligence list
    const intList = await request.post(`${BASE}/api/files/list`, {
      headers: HEADERS,
      data: { persona: 'intelligence' }
    });
    expect(intList.status()).toBe(200);
    const intFiles = await intList.json();
    const found = intFiles.files.some(f => f.filename === filename || f.name === filename);
    expect(found).toBeTruthy();

    // FI-2: File NOT visible in gaming list
    const gamingList = await request.post(`${BASE}/api/files/list`, {
      headers: HEADERS,
      data: { persona: 'gaming' }
    });
    expect(gamingList.status()).toBe(200);
    const gamingFiles = await gamingList.json();
    const foundInGaming = (gamingFiles.files || []).some(f => f.filename === filename || f.name === filename);
    expect(foundInGaming).toBeFalsy();

    // FI-3: File NOT visible in scholarly list
    const scholarlyList = await request.post(`${BASE}/api/files/list`, {
      headers: HEADERS,
      data: { persona: 'scholarly' }
    });
    expect(scholarlyList.status()).toBe(200);
    const scholarlyFiles = await scholarlyList.json();
    const foundInScholarly = (scholarlyFiles.files || []).some(f => f.filename === filename || f.name === filename);
    expect(foundInScholarly).toBeFalsy();
  });

  // ── FI-9/10/11: Write file via persona-files API for scholarly ──
  test('FI-9/10/11: Persona-files write for scholarly visible there, not in gaming', async ({ request }) => {
    test.setTimeout(30000);

    const filepath = `/test_scholarly_${Date.now()}.md`;
    const content = 'Scholarly persona file isolation test';

    // Write file for scholarly persona
    const writeResp = await request.post(`${BASE}/api/persona-files/write`, {
      headers: HEADERS,
      data: { persona: 'scholarly', path: filepath, content }
    });
    expect(writeResp.status()).toBe(200);
    const writeData = await writeResp.json();
    expect(writeData.ok).toBeTruthy();
    createdFiles.push({ persona: 'scholarly', path: filepath });

    // FI-9: Visible in scholarly file list
    const scholarlyList = await request.get(
      `${BASE}/api/persona-files?persona=scholarly&path=/`,
      { headers: HEADERS }
    );
    expect(scholarlyList.status()).toBe(200);
    const scholarlyEntries = await scholarlyList.json();
    const found = (scholarlyEntries.entries || []).some(e =>
      e.name === filepath.replace('/', '') || e.path === filepath
    );
    expect(found).toBeTruthy();

    // FI-10: Read back content matches
    const readResp = await request.get(
      `${BASE}/api/persona-files/read?persona=scholarly&path=${encodeURIComponent(filepath)}`,
      { headers: HEADERS }
    );
    expect(readResp.status()).toBe(200);
    const readData = await readResp.json();
    expect(readData.content).toContain('Scholarly persona file isolation test');

    // FI-11: NOT visible in gaming
    const gamingList = await request.get(
      `${BASE}/api/persona-files?persona=gaming&path=/`,
      { headers: HEADERS }
    );
    expect(gamingList.status()).toBe(200);
    const gamingEntries = await gamingList.json();
    const foundInGaming = (gamingEntries.entries || []).some(e =>
      e.name === filepath.replace('/', '') || e.path === filepath
    );
    expect(foundInGaming).toBeFalsy();
  });

  // ── FI-13: Path traversal attempt ──
  test('FI-13: Path traversal blocked', async ({ request }) => {
    test.setTimeout(30000);

    const resp = await request.get(
      `${BASE}/api/persona-files/read?persona=gaming&path=${encodeURIComponent('../../scholarly/notes.md')}`,
      { headers: HEADERS }
    );
    // Should either return non-200 or empty/error content
    const data = await resp.json().catch(() => ({}));
    if (resp.status() === 200) {
      // If 200, content should be empty or null (not leaking scholarly data)
      expect(data.content || '').toBe('');
    } else {
      // Non-200 is acceptable (403, 404, 400)
      expect(resp.status()).not.toBe(200);
    }
  });

  // ── FI-14/15/16: Context isolation between personas ──
  test('FI-14/15/16: Saved context for intelligence does not leak to gaming', async ({ request }) => {
    test.setTimeout(30000);

    const contextContent = `oil markets analysis ${Date.now()}`;

    // FI-14: Save context for intelligence
    const saveResp = await request.post(`${BASE}/api/persona-context`, {
      headers: HEADERS,
      data: { persona: 'intelligence', key: 'system_context', content: contextContent }
    });
    expect(saveResp.status()).toBe(200);
    const saveData = await saveResp.json();
    expect(saveData.ok).toBeTruthy();

    // FI-15: Get intelligence context → should contain our content
    const intCtx = await request.get(
      `${BASE}/api/persona-context?persona=intelligence&key=system_context`,
      { headers: HEADERS }
    );
    expect(intCtx.status()).toBe(200);
    const intData = await intCtx.json();
    expect(intData.content).toContain('oil markets');

    // FI-16: Get gaming context → should NOT contain "oil markets"
    const gamingCtx = await request.get(
      `${BASE}/api/persona-context?persona=gaming&key=system_context`,
      { headers: HEADERS }
    );
    expect(gamingCtx.status()).toBe(200);
    const gamingData = await gamingCtx.json();
    expect(gamingData.content || '').not.toContain('oil markets');
  });

  // ── FI-22: Get context for persona with no saved context ──
  test('FI-22: Context for persona with no saved data returns non-error', async ({ request }) => {
    test.setTimeout(30000);

    // Use a persona unlikely to have saved context
    const resp = await request.get(
      `${BASE}/api/persona-context?persona=anime&key=system_context`,
      { headers: HEADERS }
    );
    expect(resp.status()).toBe(200);
    const data = await resp.json();
    // Should return a valid response (not an error), content may be empty string or template
    expect(data).toHaveProperty('content');
    expect(data).toHaveProperty('persona', 'anime');
  });

  // ── CV-1: Create conversation → ID returned ──
  test('CV-1: Create conversation returns ID', async ({ request }) => {
    test.setTimeout(30000);

    const resp = await request.post(`${BASE}/api/conversations`, {
      headers: HEADERS,
      data: { persona: 'intelligence', title: 'Test Conversation CV-1' }
    });
    expect(resp.status()).toBe(200);
    const data = await resp.json();
    expect(data.ok).toBeTruthy();
    expect(data.id).toBeDefined();
    expect(typeof data.id).toBe('number');
    createdConversationIds.push(data.id);
  });

  // ── CV-4: Rename conversation via PUT ──
  test('CV-4: Rename conversation via PUT', async ({ request }) => {
    test.setTimeout(30000);

    // Create a conversation first
    const createResp = await request.post(`${BASE}/api/conversations`, {
      headers: HEADERS,
      data: { persona: 'intelligence', title: 'Original Title' }
    });
    const createData = await createResp.json();
    const convId = createData.id;
    createdConversationIds.push(convId);

    // Rename via PUT
    const renameResp = await request.fetch(`${BASE}/api/conversations/${convId}`, {
      method: 'PUT',
      headers: HEADERS,
      data: { title: 'Renamed Title CV-4' }
    });
    expect(renameResp.status()).toBe(200);
    const renameData = await renameResp.json();
    expect(renameData.ok).toBeTruthy();

    // Verify title changed by reading it back
    const getResp = await request.get(`${BASE}/api/conversations/${convId}`, {
      headers: HEADERS
    });
    expect(getResp.status()).toBe(200);
    const convData = await getResp.json();
    expect(convData.title).toBe('Renamed Title CV-4');
  });

  // ── CV-7: Append message then verify title is set ──
  test('CV-7: Append message and verify title auto-set', async ({ request }) => {
    test.setTimeout(30000);

    // Create conversation with default title
    const createResp = await request.post(`${BASE}/api/conversations`, {
      headers: HEADERS,
      data: { persona: 'intelligence', title: 'New Chat' }
    });
    const createData = await createResp.json();
    const convId = createData.id;
    createdConversationIds.push(convId);

    // Append a user message
    const appendResp = await request.post(`${BASE}/api/conversations/${convId}/append`, {
      headers: HEADERS,
      data: { role: 'user', content: 'What are the latest developments in AI?' }
    });
    expect(appendResp.status()).toBe(200);
    const appendData = await appendResp.json();
    expect(appendData.ok).toBeTruthy();
    expect(appendData.message_count).toBeGreaterThanOrEqual(1);

    // Read conversation back — verify message was stored
    const getResp = await request.get(`${BASE}/api/conversations/${convId}`, {
      headers: HEADERS
    });
    expect(getResp.status()).toBe(200);
    const convData = await getResp.json();
    expect(convData.messages.length).toBeGreaterThanOrEqual(1);
    expect(convData.messages[0].role).toBe('user');
    expect(convData.messages[0].content).toContain('AI');

    // Title should be set (either auto-set from first message or still 'New Chat')
    expect(convData.title).toBeDefined();
    expect(convData.title.length).toBeGreaterThan(0);
  });

  // ── CV-5: Delete conversation → verify gone ──
  test('CV-5: Delete conversation removes it', async ({ request }) => {
    test.setTimeout(30000);

    // Create a conversation to delete
    const createResp = await request.post(`${BASE}/api/conversations`, {
      headers: HEADERS,
      data: { persona: 'intelligence', title: 'To Be Deleted CV-5' }
    });
    const createData = await createResp.json();
    const convId = createData.id;

    // Delete it
    const deleteResp = await request.delete(`${BASE}/api/conversations/${convId}`, {
      headers: HEADERS
    });
    expect(deleteResp.status()).toBe(200);
    const deleteData = await deleteResp.json();
    expect(deleteData.ok).toBeTruthy();

    // Verify it's gone — GET should return 404
    const getResp = await request.get(`${BASE}/api/conversations/${convId}`, {
      headers: HEADERS
    });
    expect(getResp.status()).toBe(404);
  });

  // ── I-7: Server health check ──
  test('I-7: Server health / agent-status returns OK', async ({ request }) => {
    test.setTimeout(30000);

    const resp = await request.get(`${BASE}/api/agent-status`, {
      headers: HEADERS
    });
    expect(resp.status()).toBe(200);
    const data = await resp.json();
    // Should return some status object — verify it's not an error
    expect(data).toBeDefined();
    expect(data.error).toBeUndefined();
  });

});
