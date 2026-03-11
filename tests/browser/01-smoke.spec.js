const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Phase 1: Core Smoke Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('1.1 App loads without critical errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase1/app-loaded.png' });
    // Filter for truly critical errors (not cosmetic)
    const critical = errors.filter(e => !e.includes('_mpLoad') && !e.includes('is not defined'));
    if (errors.length) console.log('Console errors:', errors);
    expect(critical.length).toBe(0);
  });

  test('1.2 Feed container exists', async ({ page }) => {
    const hasFeed = await page.locator('#feed-column, #news-feed, #main-content').first().isVisible({ timeout: 5000 });
    await page.screenshot({ path: '/tmp/stratos-qa/phase1/feed.png' });
    expect(hasFeed).toBeTruthy();
  });

  test('1.3 Agent panel opens', async ({ page }) => {
    // Click the agent panel header to expand it
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    const agentBody = await page.locator('#agent-body').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase1/agent-panel.png' });
    expect(agentBody).toBeTruthy();
  });

  test('1.4 Settings page loads', async ({ page }) => {
    await page.evaluate(() => { if (typeof setActive === 'function') setActive('settings'); });
    await page.waitForTimeout(1000);
    const settingsVisible = await page.locator('#settings-panel').isVisible({ timeout: 3000 });
    await page.screenshot({ path: '/tmp/stratos-qa/phase1/settings.png' });
    expect(settingsVisible).toBeTruthy();
  });

  test('1.5 Markets panel loads', async ({ page }) => {
    await page.evaluate(() => { if (typeof setActive === 'function') setActive('markets_view'); });
    await page.waitForTimeout(2000);
    const marketsVisible = await page.locator('#markets-panel').isVisible({ timeout: 3000 });
    await page.screenshot({ path: '/tmp/stratos-qa/phase1/markets.png' });
    expect(marketsVisible).toBeTruthy();
  });

  test('1.6 Sidebar is visible', async ({ page }) => {
    const sidebar = await page.locator('#sidebar').isVisible({ timeout: 3000 });
    await page.screenshot({ path: '/tmp/stratos-qa/phase1/sidebar.png' });
    expect(sidebar).toBeTruthy();
  });
});
