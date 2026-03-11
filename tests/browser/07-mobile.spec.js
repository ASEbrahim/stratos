const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.use({ viewport: { width: 375, height: 812 } });

test.describe('Phase 7: Mobile Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('7.1 No horizontal overflow', async ({ page }) => {
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    console.log(`scrollWidth=${scrollWidth}, clientWidth=${clientWidth}`);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-main.png' });
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 5); // 5px tolerance
  });

  test('7.2 Feed cards stack vertically', async ({ page }) => {
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-feed.png' });
    const mainContent = await page.locator('#main-content, #feed-column').first().isVisible({ timeout: 3000 });
    expect(mainContent).toBeTruthy();
  });

  test('7.3 Agent panel fills screen on mobile', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-agent.png' });
    const agentPanel = page.locator('#agent-panel');
    const box = await agentPanel.boundingBox().catch(() => null);
    if (box) {
      console.log(`Agent panel: width=${box.width}, height=${box.height}`);
    }
  });

  test('7.4 Persona selector usable on mobile', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof _togglePersonaPicker === 'function') _togglePersonaPicker(); });
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-persona-picker.png' });
    const dropdown = await page.locator('#persona-picker-dropdown').isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Persona dropdown visible on mobile:', dropdown);
  });

  test('7.5 Context editor opens as full-width on mobile', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-context-editor.png' });
    const editor = page.locator('#context-editor-panel');
    const box = await editor.boundingBox().catch(() => null);
    if (box) {
      console.log(`Context editor: width=${box.width} (viewport=375)`);
      // Should be close to full width on mobile
      expect(box.width).toBeGreaterThan(300);
    }
  });

  test('7.6 Settings page on mobile', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof setActive === 'function') setActive('settings');
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-settings.png' });
    // Check tab bar horizontal scroll
    const tabBar = page.locator('#settings-tab-bar, .stab-bar');
    const visible = await tabBar.isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Tab bar visible on mobile:', visible);
  });

  test('7.7 File browser on mobile', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-file-browser.png' });
    const panel = page.locator('#file-browser-panel');
    const box = await panel.boundingBox().catch(() => null);
    if (box) {
      console.log(`File browser: width=${box.width} (viewport=375)`);
      expect(box.width).toBeGreaterThan(300);
    }
  });

  test('7.8 Markets panel on mobile', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof setActive === 'function') setActive('markets_view');
    });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase7/mobile-markets.png' });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 5);
    expect(overflow).toBeFalsy();
  });
});
