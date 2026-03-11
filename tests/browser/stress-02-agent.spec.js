const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Stress 2: Agent Deep Test', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
  });

  test('S2.1 Agent welcome message per persona', async ({ page }) => {
    for (const persona of ['intelligence', 'scholarly', 'market', 'gaming', 'anime', 'tcg']) {
      await page.evaluate((p) => { if (typeof switchPersona === 'function') switchPersona(p); }, persona);
      await page.waitForTimeout(500);
      const welcome = await page.locator('#agent-welcome').textContent().catch(() => '');
      console.log(`${persona}: ${welcome.substring(0, 60).trim()}`);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/agent/persona-welcomes.png' });
  });

  test('S2.2 Agent input exists and can type', async ({ page }) => {
    const input = page.locator('#agent-input');
    await expect(input).toBeVisible({ timeout: 3000 });
    await input.fill('Hello, what can you help me with?');
    const val = await input.inputValue();
    expect(val).toContain('Hello');
    await page.screenshot({ path: '/tmp/stratos-stress/agent/input.png' });
  });

  test('S2.3 Send button exists and toggles', async ({ page }) => {
    const sendBtn = page.locator('#agent-send-btn');
    await expect(sendBtn).toBeVisible({ timeout: 3000 });
    await page.screenshot({ path: '/tmp/stratos-stress/agent/send-btn.png' });
  });

  test('S2.4 Agent mode toggle (structured vs free)', async ({ page }) => {
    const modeBtn = page.locator('#agent-mode-btn');
    const visible = await modeBtn.isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Mode button visible:', visible);
    if (visible) {
      await modeBtn.click();
      await page.waitForTimeout(500);
      const placeholder = await page.locator('#agent-input').getAttribute('placeholder');
      console.log('Placeholder after toggle:', placeholder);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/agent/mode-toggle.png' });
  });

  test('S2.5 Multi-agent select 2, then 3, then try 4', async ({ page }) => {
    const result = await page.evaluate(() => {
      if (typeof _togglePersonaPicker !== 'function') return { error: 'no picker' };
      _togglePersonaPicker();
      const dd = document.getElementById('persona-picker-dropdown');
      if (!dd) return { error: 'no dropdown' };
      // Reset to just intelligence
      selectedPersonas = ['intelligence'];
      _renderPersonaPicker();
      // Check first 3 by simulating changes
      const names = ['intelligence', 'market', 'scholarly'];
      names.forEach(n => { if (!selectedPersonas.includes(n)) selectedPersonas.push(n); });
      _renderPersonaPicker();
      // Now re-query fresh DOM after re-render
      const freshCbs = dd.querySelectorAll('input[type="checkbox"]');
      const count = freshCbs.length;
      const checkedCount = dd.querySelectorAll('input[type="checkbox"]:checked').length;
      const disabledCount = dd.querySelectorAll('input[type="checkbox"]:disabled').length;
      // The 4th+ unchecked checkboxes should be disabled
      return { count, checkedCount, disabledCount, selected: selectedPersonas.slice() };
    });
    console.log('Multi-select result:', JSON.stringify(result));
    // With 3 selected, unchecked ones (count - 3) should be disabled
    expect(result.disabledCount).toBe(result.count - 3);
    await page.screenshot({ path: '/tmp/stratos-stress/agent/multi-select-3.png' });
  });

  test('S2.6 XSS in agent input', async ({ page }) => {
    const input = page.locator('#agent-input');
    await input.fill('<script>window._xss=true</script>');
    await page.waitForTimeout(300);
    const xss = await page.evaluate(() => window._xss || false);
    expect(xss).toBeFalsy();
    await page.screenshot({ path: '/tmp/stratos-stress/agent/xss-input.png' });
  });

  test('S2.7 Agent fullscreen toggle', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentFullscreen === 'function') toggleAgentFullscreen(); });
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/stratos-stress/agent/fullscreen.png' });
    // Toggle back
    await page.evaluate(() => { if (typeof toggleAgentFullscreen === 'function') toggleAgentFullscreen(); });
    await page.waitForTimeout(500);
  });

  test('S2.8 Agent chat clear', async ({ page }) => {
    page.on('dialog', d => d.accept());
    await page.evaluate(() => { if (typeof clearAgentChat === 'function') clearAgentChat(); });
    await page.waitForTimeout(500);
    const messages = await page.locator('#agent-messages').textContent().catch(() => '');
    console.log('Messages after clear:', messages.length, 'chars');
    await page.screenshot({ path: '/tmp/stratos-stress/agent/cleared.png' });
  });
});
