const { test, expect } = require('@playwright/test');
const { login } = require('./auth');
const fs = require('fs');

test.describe('Phase 8: Console Error Sweep', () => {

  test('8.1 Console error sweep across all views', async ({ page }) => {
    const errors = [];
    const warnings = [];

    page.on('pageerror', e => errors.push({ msg: e.message, url: page.url(), type: 'pageerror' }));
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Skip known non-critical errors
        if (text.includes('favicon') || text.includes('net::ERR')) return;
        errors.push({ msg: text, url: page.url(), type: 'console.error' });
      }
      if (msg.type() === 'warning') {
        warnings.push({ msg: msg.text(), url: page.url() });
      }
    });

    // Login
    await login(page);
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase8/01-feed.png' });

    // Open agent panel
    await page.evaluate(() => { if (typeof toggleAgentChat === 'function') toggleAgentChat(); });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase8/02-agent.png' });

    // Switch through personas
    for (const p of ['intelligence', 'scholarly', 'market', 'gaming', 'anime', 'tcg']) {
      await page.evaluate((name) => { if (typeof switchPersona === 'function') switchPersona(name); }, p);
      await page.waitForTimeout(500);
    }
    await page.screenshot({ path: '/tmp/stratos-qa/phase8/03-personas.png' });

    // Context editor
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase8/04-context-editor.png' });
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(500);

    // File browser
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase8/05-file-browser.png' });
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(500);

    // Settings - each tab
    await page.evaluate(() => { if (typeof setActive === 'function') setActive('settings'); });
    await page.waitForTimeout(1000);
    for (const tab of ['profile', 'sources', 'market', 'youtube', 'system']) {
      await page.evaluate((t) => { if (typeof switchSettingsTab === 'function') switchSettingsTab(t); }, tab);
      await page.waitForTimeout(1000);
      await page.screenshot({ path: `/tmp/stratos-qa/phase8/06-settings-${tab}.png` });
    }

    // Markets panel
    await page.evaluate(() => { if (typeof setActive === 'function') setActive('markets_view'); });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase8/07-markets.png' });

    // Write error report
    const report = {
      timestamp: new Date().toISOString(),
      total_errors: errors.length,
      total_warnings: warnings.length,
      errors,
      warnings: warnings.slice(0, 20), // Cap warnings
    };
    fs.writeFileSync('/tmp/stratos-qa/phase8/console-errors.json', JSON.stringify(report, null, 2));

    console.log(`\n=== CONSOLE ERROR SWEEP ===`);
    console.log(`Total errors: ${errors.length}`);
    console.log(`Total warnings: ${warnings.length}`);
    errors.forEach(e => console.log(`  [ERROR] ${e.msg.substring(0, 120)}`));

    // Categorize: critical errors that affect functionality
    const critical = errors.filter(e =>
      !e.msg.includes('_mpLoad') &&
      !e.msg.includes('is not defined') &&
      !e.msg.includes('favicon') &&
      !e.msg.includes('404')
    );
    console.log(`\nCritical errors: ${critical.length}`);
    critical.forEach(e => console.log(`  [CRITICAL] ${e.msg.substring(0, 120)}`));
  });
});
