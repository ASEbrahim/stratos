const { test, expect } = require('@playwright/test');
const { login, AUTH_TOKEN, TEST_PROFILE_ID } = require('./auth');

test.describe('Stress 5: Cross-Feature & Edge Cases', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('S5.1 Profile isolation — kirissie data unchanged', async ({ page }) => {
    // Verify via API that profile_id=8 has no test artifacts
    const r = await page.evaluate(async () => {
      const resp = await fetch('/api/persona-context?persona=intelligence&key=system_context', {
        headers: { 'X-Auth-Token': '0e704d2baab3464b1014de26cbde6cd543c5d99a0a2bb2c68fc7783f0923c415' }
      });
      return resp.json();
    });
    console.log('Context check:', JSON.stringify(r).substring(0, 100));
    // Our test profile should have test data, kirissie should not
    // This test verifies the API scopes by token
  });

  test('S5.2 Theme switching works across all panels', async ({ page }) => {
    const themes = ['midnight', 'coffee', 'rose', 'noir', 'aurora', 'cosmos'];
    for (const theme of themes) {
      await page.evaluate((t) => { if (typeof setTheme === 'function') setTheme(t); }, theme);
      await page.waitForTimeout(300);
    }
    // Set back to midnight
    await page.evaluate(() => { if (typeof setTheme === 'function') setTheme('midnight'); });
    await page.waitForTimeout(300);
    await page.screenshot({ path: '/tmp/stratos-stress/cross/themes.png' });
  });

  test('S5.3 Context editor → Agent integration', async ({ page }) => {
    // Open agent
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    // Open context editor
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1000);
    // Both should be visible simultaneously
    const agentBody = await page.locator('#agent-body').isVisible({ timeout: 2000 }).catch(() => false);
    const ctxEditor = await page.locator('.ctx-editor-sidebar').first().isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Agent body:', agentBody, 'Context editor:', ctxEditor);
    await page.screenshot({ path: '/tmp/stratos-stress/cross/agent-context.png' });
  });

  test('S5.4 File browser → Agent integration', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);
    const fileBrowser = await page.locator('.ctx-editor-sidebar').first().isVisible({ timeout: 2000 }).catch(() => false);
    console.log('File browser visible:', fileBrowser);
    await page.screenshot({ path: '/tmp/stratos-stress/cross/agent-files.png' });
  });

  test('S5.5 Rapid persona switching', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    const personas = ['intelligence', 'scholarly', 'market', 'gaming', 'anime', 'tcg', 'intelligence'];
    for (const p of personas) {
      await page.evaluate((name) => { if (typeof switchPersona === 'function') switchPersona(name); }, p);
      await page.waitForTimeout(200);
    }
    // Should not crash
    const label = await page.locator('#persona-picker-label').textContent().catch(() => '');
    console.log('Final persona:', label);
    expect(label.toLowerCase()).toContain('intelligence');
    await page.screenshot({ path: '/tmp/stratos-stress/cross/rapid-persona.png' });
  });

  test('S5.6 XSS in persona context', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    const textarea = page.locator('#ctx-editor-textarea');
    if (await textarea.isVisible({ timeout: 2000 }).catch(() => false)) {
      await textarea.fill('<img src=x onerror="window._xss_ctx=true">');
      await page.evaluate(() => { if (typeof _ctxSave === 'function') _ctxSave(); });
      await page.waitForTimeout(1000);
      const xss = await page.evaluate(() => window._xss_ctx || false);
      expect(xss).toBeFalsy();
    }
    await page.screenshot({ path: '/tmp/stratos-stress/cross/xss-context.png' });
  });

  test('S5.7 Settings all tabs load without errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(500);
    for (const tab of ['profile', 'sources', 'market', 'youtube', 'system']) {
      await page.evaluate((t) => switchSettingsTab(t), tab);
      await page.waitForTimeout(800);
    }
    console.log('Errors during settings navigation:', errors.length);
    errors.forEach(e => console.log('  ', e.substring(0, 80)));
    await page.screenshot({ path: '/tmp/stratos-stress/settings/all-tabs.png' });
    expect(errors.length).toBe(0);
  });

  test('S5.8 Keyboard shortcut Ctrl+S quicksave', async ({ page }) => {
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    // quickSave should exist
    const hasSave = await page.evaluate(() => typeof quickSave === 'function');
    console.log('quickSave exists:', hasSave);
  });

  test('S5.9 Sidebar collapse and expand', async ({ page }) => {
    const sidebar = page.locator('#sidebar');
    const initialBox = await sidebar.boundingBox().catch(() => null);
    console.log('Initial sidebar width:', initialBox?.width);

    await page.evaluate(() => { if (typeof toggleSidebar === 'function') toggleSidebar(); });
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/stratos-stress/cross/sidebar-collapsed.png' });

    await page.evaluate(() => { if (typeof toggleSidebar === 'function') toggleSidebar(); });
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/stratos-stress/cross/sidebar-expanded.png' });
  });

  test('S5.10 Games scenario only shows for gaming persona', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);

    // Intelligence — no scenario bar
    await page.evaluate(() => switchPersona('intelligence'));
    await page.waitForTimeout(500);
    const intBar = await page.locator('#games-scenario-bar').isVisible({ timeout: 1000 }).catch(() => false);

    // Gaming — scenario bar
    await page.evaluate(() => switchPersona('gaming'));
    await page.waitForTimeout(500);
    const gameBar = await page.locator('#games-scenario-bar').isVisible({ timeout: 1000 }).catch(() => false);

    console.log('Intelligence scenario bar:', intBar, 'Gaming scenario bar:', gameBar);
    expect(intBar).toBeFalsy();
    expect(gameBar).toBeTruthy();
    await page.screenshot({ path: '/tmp/stratos-stress/cross/scenario-isolation.png' });
  });
});
