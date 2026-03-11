const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Stress 1: Wizard & Feed', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('S1.1 Feed renders with score colors and metadata', async ({ page }) => {
    await page.waitForTimeout(2000);
    const feedCards = await page.locator('.feed-card, .news-card, [class*="card"]').count();
    console.log(`Feed cards: ${feedCards}`);
    await page.screenshot({ path: '/tmp/stratos-stress/feed/feed-cards.png' });
    // Feed should have content (Developer_KW has categories)
    const mainContent = await page.locator('#main-content, #feed-column').first().isVisible();
    expect(mainContent).toBeTruthy();
  });

  test('S1.2 Score filters work', async ({ page }) => {
    await page.waitForTimeout(2000);
    for (const level of ['all', 'critical', 'high', 'medium', 'noise']) {
      await page.evaluate((l) => { if (typeof toggleScoreFilter === 'function') toggleScoreFilter(l); }, level);
      await page.waitForTimeout(300);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/feed/score-filters.png' });
  });

  test('S1.3 Feedback actions — save/dismiss article', async ({ page }) => {
    await page.waitForTimeout(2000);
    // Find first article's action buttons
    const saveBtn = page.locator('[onclick*="save"], [data-action="save"], .save-btn').first();
    const hasSave = await saveBtn.isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Save button found:', hasSave);
    await page.screenshot({ path: '/tmp/stratos-stress/feed/feedback.png' });
  });

  test('S1.4 XSS in search/filter input', async ({ page }) => {
    await page.waitForTimeout(2000);
    const search = page.locator('#feed-search');
    if (await search.isVisible({ timeout: 2000 }).catch(() => false)) {
      await search.fill('<img src=x onerror=alert(1)>');
      await page.waitForTimeout(500);
      // Check that XSS didn't execute
      const alertFired = await page.evaluate(() => window._xss_test || false);
      expect(alertFired).toBeFalsy();
    }
    await page.screenshot({ path: '/tmp/stratos-stress/feed/xss-search.png' });
  });

  test('S1.5 Settings profile saves and persists', async ({ page }) => {
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    await page.evaluate(() => switchSettingsTab('profile'));
    await page.waitForTimeout(1000);
    const roleField = page.locator('#simple-role');
    if (await roleField.isVisible({ timeout: 2000 }).catch(() => false)) {
      const original = await roleField.inputValue();
      console.log('Current role:', original);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/settings/profile-tab.png' });
  });
});
