const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.use({ viewport: { width: 375, height: 812 } });

test.describe('Stress 7: Mobile Deep Test', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('S7.1 No horizontal overflow on feed', async ({ page }) => {
    await page.waitForTimeout(2000);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 5);
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/feed.png' });
    expect(overflow).toBeFalsy();
  });

  test('S7.2 Agent panel on mobile', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    const panel = page.locator('#agent-panel');
    const box = await panel.boundingBox().catch(() => null);
    console.log('Agent panel mobile:', box ? `w=${box.width} h=${box.height}` : 'not visible');
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/agent.png' });
  });

  test('S7.3 Mobile agent (bottom sheet)', async ({ page }) => {
    // Check if mobile agent toggle exists
    const hasToggle = await page.evaluate(() => typeof _toggleMobileAgent === 'function');
    console.log('Mobile agent toggle exists:', hasToggle);
    if (hasToggle) {
      await page.evaluate(() => _toggleMobileAgent());
      await page.waitForTimeout(1000);
      await page.screenshot({ path: '/tmp/stratos-stress/mobile/mobile-agent.png' });
    }
  });

  test('S7.4 Settings scrollable on mobile', async ({ page }) => {
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    // Scroll down
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 5);
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/settings-scroll.png' });
    expect(overflow).toBeFalsy();
  });

  test('S7.5 Markets on mobile — no overflow', async ({ page }) => {
    await page.evaluate(() => setActive('markets_view'));
    await page.waitForTimeout(2000);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 5);
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/markets.png' });
    expect(overflow).toBeFalsy();
  });

  test('S7.6 YouTube tab on mobile', async ({ page }) => {
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(500);
    await page.evaluate(() => switchSettingsTab('youtube'));
    await page.waitForTimeout(1000);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 5);
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/youtube.png' });
    expect(overflow).toBeFalsy();
  });

  test('S7.7 Sidebar behavior on mobile', async ({ page }) => {
    const sidebar = page.locator('#sidebar');
    const box = await sidebar.boundingBox().catch(() => null);
    console.log('Sidebar on mobile:', box ? `w=${box.width} h=${box.height}` : 'hidden');
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/sidebar.png' });
  });

  test('S7.8 Context editor full-width on mobile', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1000);
    const editor = page.locator('.ctx-editor-sidebar').first();
    const box = await editor.boundingBox().catch(() => null);
    if (box) {
      console.log(`Context editor mobile: w=${box.width} (viewport=375)`);
      expect(box.width).toBeGreaterThan(300);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/mobile/context-editor.png' });
  });
});
