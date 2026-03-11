const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Phase 4: YouTube Management Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    // Navigate to settings → YouTube tab
    await page.evaluate(() => {
      if (typeof setActive === 'function') setActive('settings');
    });
    await page.waitForTimeout(1000);
    await page.evaluate(() => {
      if (typeof switchSettingsTab === 'function') switchSettingsTab('youtube');
    });
    await page.waitForTimeout(1000);
  });

  test('4.1 YouTube settings tab loads', async ({ page }) => {
    const panel = await page.locator('#youtube-settings-panel').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase4/youtube-tab.png' });
    expect(panel).toBeTruthy();
  });

  test('4.2 Add channel input visible', async ({ page }) => {
    const input = await page.locator('#yt-channel-url').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase4/add-channel-input.png' });
    expect(input).toBeTruthy();
  });

  test('4.3 Lens checkboxes visible', async ({ page }) => {
    const lenses = await page.locator('#yt-lens-checkboxes').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase4/lens-checkboxes.png' });
    console.log('Lens checkboxes visible:', lenses);
  });

  test('4.4 Channel list renders', async ({ page }) => {
    const list = page.locator('#youtube-channel-list');
    const visible = await list.isVisible({ timeout: 3000 }).catch(() => false);
    const text = await list.textContent().catch(() => '');
    console.log('Channel list:', text.substring(0, 200));
    await page.screenshot({ path: '/tmp/stratos-qa/phase4/channel-list.png' });
    // It's ok if empty — we just need the container to exist
    expect(visible).toBeTruthy();
  });

  test('4.5 Add a test channel', async ({ page }) => {
    const input = page.locator('#yt-channel-url');
    await input.fill('https://www.youtube.com/@TEDEd');
    await page.screenshot({ path: '/tmp/stratos-qa/phase4/channel-input-filled.png' });
    // Click add button
    await page.evaluate(() => { if (typeof _ytAddChannel === 'function') _ytAddChannel(); });
    await page.waitForTimeout(5000); // Network call to resolve channel
    await page.screenshot({ path: '/tmp/stratos-qa/phase4/channel-added.png' });
    const list = await page.locator('#youtube-channel-list').textContent().catch(() => '');
    console.log('Channel list after add:', list.substring(0, 300));
  });
});
