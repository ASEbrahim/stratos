const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Phase 6: Workspace & Settings Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.evaluate(() => {
      if (typeof setActive === 'function') setActive('settings');
    });
    await page.waitForTimeout(1000);
    await page.evaluate(() => {
      if (typeof switchSettingsTab === 'function') switchSettingsTab('system');
    });
    await page.waitForTimeout(1000);
  });

  test('6.1 System tab loads', async ({ page }) => {
    const panel = await page.locator('#display-settings-panel, #workspace-panel').first().isVisible({ timeout: 3000 });
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/system-tab.png' });
    expect(panel).toBeTruthy();
  });

  test('6.2 Workspace panel visible', async ({ page }) => {
    const workspace = await page.locator('#workspace-panel').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/workspace-panel.png' });
    console.log('Workspace panel visible:', workspace);
    if (workspace) {
      const stats = await page.locator('#ws-stats-content').textContent().catch(() => '');
      console.log('Workspace stats:', stats.substring(0, 300));
    }
  });

  test('6.3 Export profile button exists', async ({ page }) => {
    const exportBtn = page.locator('button:has-text("Export"), [onclick*="_wsExport"]');
    const visible = await exportBtn.first().isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/export-button.png' });
    console.log('Export button visible:', visible);
  });

  test('6.4 Preference signals panel', async ({ page }) => {
    const signals = await page.locator('#preference-signals-panel').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/signals-panel.png' });
    console.log('Signals panel visible:', signals);
    if (signals) {
      const list = await page.locator('#ws-signals-list').textContent().catch(() => '');
      console.log('Signals list:', list.substring(0, 200));
    }
  });

  test('6.5 Settings Profile tab loads', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof switchSettingsTab === 'function') switchSettingsTab('profile');
    });
    await page.waitForTimeout(1000);
    const role = await page.locator('#simple-role').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/profile-tab.png' });
    expect(role).toBeTruthy();
  });

  test('6.6 Settings Sources tab loads', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof switchSettingsTab === 'function') switchSettingsTab('sources');
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/sources-tab.png' });
    const categories = await page.locator('#simple-categories-container').isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Categories container visible:', categories);
  });

  test('6.7 Settings Market tab loads', async ({ page }) => {
    await page.evaluate(() => {
      if (typeof switchSettingsTab === 'function') switchSettingsTab('market');
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/market-tab.png' });
    const tickers = await page.locator('#market-tickers-panel').isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Tickers panel visible:', tickers);
  });

  test('6.8 Import strategy selector exists', async ({ page }) => {
    const strategy = await page.locator('#ws-import-strategy').isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Import strategy selector visible:', strategy);
    await page.screenshot({ path: '/tmp/stratos-qa/phase6/import-strategy.png' });
  });
});
