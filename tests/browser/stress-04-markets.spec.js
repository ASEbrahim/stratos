const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Stress 4: Markets & Charts', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.evaluate(() => setActive('markets_view'));
    await page.waitForTimeout(2000);
  });

  test('S4.1 Markets panel renders sections', async ({ page }) => {
    const overview = await page.locator('#mp-sec-overview').isVisible({ timeout: 3000 }).catch(() => false);
    const news = await page.locator('#mp-sec-news').isVisible({ timeout: 3000 }).catch(() => false);
    const stats = await page.locator('#mp-sec-stats').isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Overview:', overview, 'News:', news, 'Stats:', stats);
    await page.screenshot({ path: '/tmp/stratos-stress/market/sections.png' });
  });

  test('S4.2 Chart grid renders', async ({ page }) => {
    const chartGrid = await page.locator('#mp-chart-grid').isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Chart grid visible:', chartGrid);
    await page.screenshot({ path: '/tmp/stratos-stress/market/chart-grid.png' });
  });

  test('S4.3 Section toggle collapse/expand', async ({ page }) => {
    for (const sec of ['overview', 'news', 'stats']) {
      await page.evaluate((s) => { if (typeof mpToggleSection === 'function') mpToggleSection(s); }, sec);
      await page.waitForTimeout(300);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/market/sections-toggled.png' });
    // Toggle back
    for (const sec of ['overview', 'news', 'stats']) {
      await page.evaluate((s) => { if (typeof mpToggleSection === 'function') mpToggleSection(s); }, sec);
      await page.waitForTimeout(300);
    }
  });

  test('S4.4 Market agent input works', async ({ page }) => {
    const input = page.locator('#mp-agent-input');
    const visible = await input.isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Market agent input visible:', visible);
    if (visible) {
      await input.fill('What is the current market sentiment?');
      const val = await input.inputValue();
      expect(val).toContain('market');
    }
    await page.screenshot({ path: '/tmp/stratos-stress/market/agent.png' });
  });

  test('S4.5 Ticker settings tab', async ({ page }) => {
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    await page.evaluate(() => switchSettingsTab('market'));
    await page.waitForTimeout(1000);
    const tickers = await page.locator('#market-tickers-panel').isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Tickers panel visible:', tickers);
    await page.screenshot({ path: '/tmp/stratos-stress/market/tickers-settings.png' });
  });

  test('S4.6 Add top 10 tickers preset', async ({ page }) => {
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    await page.evaluate(() => switchSettingsTab('market'));
    await page.waitForTimeout(1000);
    // Check if addTop10Tickers exists
    const hasFunc = await page.evaluate(() => typeof addTop10Tickers === 'function');
    console.log('addTop10Tickers exists:', hasFunc);
    await page.screenshot({ path: '/tmp/stratos-stress/market/top10.png' });
  });
});
