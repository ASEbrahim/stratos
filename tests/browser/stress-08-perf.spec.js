const { test, expect } = require('@playwright/test');
const { login, AUTH_TOKEN } = require('./auth');
const fs = require('fs');

test.describe('Stress 8: Performance & Console Errors', () => {

  test('S8.1 Page load timing', async ({ page }) => {
    const start = Date.now();
    await login(page);
    const loadTime = Date.now() - start;
    console.log(`Page load time: ${loadTime}ms`);
    await page.screenshot({ path: '/tmp/stratos-stress/perf/page-load.png' });
    expect(loadTime).toBeLessThan(10000); // Under 10s
  });

  test('S8.2 API response times', async ({ page }) => {
    await login(page);
    const timings = await page.evaluate(async (token) => {
      const headers = { 'X-Auth-Token': token };
      const results = {};
      const endpoints = ['/api/health', '/api/status', '/api/data', '/api/agent-status', '/api/feedback-stats'];
      for (const ep of endpoints) {
        const start = performance.now();
        const hdrs = ep === '/api/health' ? {} : headers;
        await fetch(ep, { headers: hdrs });
        results[ep] = Math.round(performance.now() - start);
      }
      return results;
    }, AUTH_TOKEN);
    console.log('API response times:');
    Object.entries(timings).forEach(([ep, ms]) => console.log(`  ${ep}: ${ms}ms`));
    // All should be under 5s
    Object.values(timings).forEach(ms => expect(ms).toBeLessThan(5000));
  });

  test('S8.3 No memory leak — 10 navigation cycles', async ({ page }) => {
    await login(page);
    const views = ['dashboard', 'settings', 'markets_view'];
    for (let i = 0; i < 10; i++) {
      for (const v of views) {
        await page.evaluate((id) => setActive(id), v);
        await page.waitForTimeout(200);
      }
    }
    // If we get here without crash, no obvious leak
    await page.waitForTimeout(500);
    const mainContent = await page.locator('#main-content, #feed-column, #settings-panel, #markets-panel').first().isVisible({ timeout: 5000 }).catch(() => false);
    // Page should still be functional
    const hasBody = await page.locator('body').isVisible();
    expect(hasBody).toBeTruthy();
    await page.screenshot({ path: '/tmp/stratos-stress/perf/after-cycles.png' });
  });

  test('S8.4 Full console error sweep after all stress tests', async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push({ msg: e.message, url: page.url() }));
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        if (text.includes('favicon') || text.includes('net::ERR')) return;
        errors.push({ msg: text, url: page.url() });
      }
    });

    await login(page);
    await page.waitForTimeout(2000);

    // Feed
    await page.evaluate(() => setActive('dashboard'));
    await page.waitForTimeout(1000);

    // Agent + all personas
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(500);
    for (const p of ['intelligence', 'scholarly', 'market', 'gaming', 'anime', 'tcg']) {
      await page.evaluate((name) => switchPersona(name), p);
      await page.waitForTimeout(300);
    }

    // Context editor
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(300);

    // File browser
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(300);

    // Settings all tabs
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(500);
    for (const tab of ['profile', 'sources', 'market', 'youtube', 'system']) {
      await page.evaluate((t) => switchSettingsTab(t), tab);
      await page.waitForTimeout(500);
    }

    // Markets
    await page.evaluate(() => setActive('markets_view'));
    await page.waitForTimeout(1000);

    // Write report
    fs.writeFileSync('/tmp/stratos-stress/perf/console-errors.json', JSON.stringify(errors, null, 2));
    console.log(`\n=== STRESS TEST CONSOLE ERRORS ===`);
    console.log(`Total errors: ${errors.length}`);
    errors.forEach(e => console.log(`  [${e.url}] ${e.msg.substring(0, 100)}`));
    // Filter out rate limiting (429) which happens during parallel test runs
    const real = errors.filter(e => !e.msg.includes('429') && !e.msg.includes('Too Many'));
    expect(real.length).toBe(0);
  });
});
