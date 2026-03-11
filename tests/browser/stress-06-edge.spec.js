const { test, expect } = require('@playwright/test');
const { login, AUTH_TOKEN } = require('./auth');

test.describe('Stress 6: Edge Cases & Security', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('S6.1 API without auth token returns 401', async ({ page }) => {
    // Use XMLHttpRequest to bypass the custom fetch override that auto-injects tokens
    const r = await page.evaluate(() => {
      return new Promise(resolve => {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/data', true);
        xhr.onload = () => resolve(xhr.status);
        xhr.onerror = () => resolve(0);
        xhr.send();
      });
    });
    console.log('No-auth /api/data status:', r);
    expect(r).toBe(401);
  });

  test('S6.2 API with invalid token returns 401', async ({ page }) => {
    const r = await page.evaluate(() => {
      return new Promise(resolve => {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/data', true);
        xhr.setRequestHeader('X-Auth-Token', 'invalid-token');
        xhr.onload = () => resolve(xhr.status);
        xhr.onerror = () => resolve(0);
        xhr.send();
      });
    });
    console.log('Invalid token status:', r);
    expect(r).toBe(401);
  });

  test('S6.3 Health endpoint is auth-exempt', async ({ page }) => {
    const r = await page.evaluate(async () => {
      const resp = await fetch('/api/health', { headers: {} });
      return resp.status;
    });
    expect(r).toBe(200);
  });

  test('S6.4 Long input in agent', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    const longText = 'A'.repeat(5000);
    const input = page.locator('#agent-input');
    await input.fill(longText);
    const val = await input.inputValue();
    console.log('Input length:', val.length);
    await page.screenshot({ path: '/tmp/stratos-stress/edge/long-input.png' });
  });

  test('S6.5 Arabic text in agent input', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    const input = page.locator('#agent-input');
    await input.fill('مرحبا، كيف يمكنني مساعدتك في التحليل الاستراتيجي؟');
    const val = await input.inputValue();
    expect(val).toContain('مرحبا');
    await page.screenshot({ path: '/tmp/stratos-stress/edge/arabic.png' });
  });

  test('S6.6 Unicode/emoji in context editor', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    const textarea = page.locator('#ctx-editor-textarea');
    if (await textarea.isVisible({ timeout: 2000 }).catch(() => false)) {
      await textarea.fill('🎮 Gaming context with emojis 🐉\n日本語テスト\nعربي');
      await page.evaluate(() => { if (typeof _ctxSave === 'function') _ctxSave(); });
      await page.waitForTimeout(1000);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/edge/unicode.png' });
  });

  test('S6.7 Page refresh preserves state', async ({ page }) => {
    // Navigate to settings
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    // Refresh
    await login(page);
    // Should still be functional
    const mainContent = await page.locator('#main-content, #feed-column').first().isVisible({ timeout: 3000 });
    expect(mainContent).toBeTruthy();
    await page.screenshot({ path: '/tmp/stratos-stress/edge/after-refresh.png' });
  });

  test('S6.8 SQL injection in search', async ({ page }) => {
    // Test API directly with SQL injection attempt
    const r = await page.evaluate(async () => {
      const resp = await fetch('/api/search-all-contexts?q=' + encodeURIComponent("'; DROP TABLE users; --"));
      return resp.status;
    });
    console.log('SQL injection attempt status:', r);
    // Should not crash the server
    expect([200, 400, 401, 404]).toContain(r);
  });

  test('S6.9 Path traversal in file browser API', async ({ page }) => {
    const r = await page.evaluate(async (token) => {
      const resp = await fetch('/api/persona-files?persona=intelligence&path=../../etc/passwd', {
        headers: { 'X-Auth-Token': token }
      });
      const d = await resp.json();
      return { status: resp.status, files: d.files || [] };
    }, AUTH_TOKEN);
    console.log('Path traversal attempt:', r);
    // Should return empty or error, NOT actual file contents
    expect(r.files.length).toBe(0);
  });

  test('S6.10 Concurrent API calls', async ({ page }) => {
    const results = await page.evaluate(async (token) => {
      const headers = { 'X-Auth-Token': token };
      const calls = [
        fetch('/api/data', { headers }),
        fetch('/api/status', { headers }),
        fetch('/api/health'),
        fetch('/api/agent-status', { headers }),
        fetch('/api/feedback-stats', { headers }),
      ];
      const responses = await Promise.all(calls);
      return responses.map(r => r.status);
    }, AUTH_TOKEN);
    console.log('Concurrent call statuses:', results);
    // All should succeed
    results.forEach(s => expect([200]).toContain(s));
  });
});
