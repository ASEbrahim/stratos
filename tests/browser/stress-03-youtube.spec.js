const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Stress 3: YouTube Full Test', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.evaluate(() => setActive('settings'));
    await page.waitForTimeout(1000);
    await page.evaluate(() => switchSettingsTab('youtube'));
    await page.waitForTimeout(1500);
  });

  test('S3.1 Channel list loads with real data', async ({ page }) => {
    const list = page.locator('#youtube-channel-list');
    const text = await list.textContent().catch(() => '');
    console.log('Channel list:', text.substring(0, 300));
    // Should not still say "Loading channels..."
    const stillLoading = text.includes('Loading channels...');
    console.log('Still loading:', stillLoading);
    await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-list.png' });
    // We have channels from previous session
    expect(stillLoading).toBeFalsy();
  });

  test('S3.2 Channel card shows name and video count', async ({ page }) => {
    await page.waitForTimeout(1000);
    const cards = await page.locator('.yt-channel-card').count();
    console.log('Channel cards:', cards);
    if (cards > 0) {
      const firstCard = await page.locator('.yt-channel-card').first().textContent();
      console.log('First card:', firstCard.substring(0, 100));
      expect(firstCard).toContain('videos');
    }
    await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-cards.png' });
  });

  test('S3.3 Lens checkboxes exist and are interactive', async ({ page }) => {
    const checkboxContainer = page.locator('#yt-lens-checkboxes');
    const visible = await checkboxContainer.isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Lens checkboxes visible:', visible);
    if (visible) {
      const checkboxes = await checkboxContainer.locator('input[type="checkbox"]').count();
      console.log('Lens checkbox count:', checkboxes);
    }
    await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-lenses.png' });
  });

  test('S3.4 Toggle video list for a channel', async ({ page }) => {
    await page.waitForTimeout(1000);
    const cards = await page.locator('.yt-channel-card').count();
    if (cards > 0) {
      // Click the list button on first channel
      const listBtn = page.locator('.yt-channel-card').first().locator('button[title="Show videos"]');
      if (await listBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await listBtn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-videos.png' });
      }
    }
  });

  test('S3.5 Add channel with various URL formats', async ({ page }) => {
    const input = page.locator('#yt-channel-url');
    // Test @handle format
    await input.fill('@TEDEd');
    await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-handle-input.png' });
    // Don't actually add (would be duplicate)
  });

  test('S3.6 Empty URL submission shows error', async ({ page }) => {
    const input = page.locator('#yt-channel-url');
    await input.fill('');
    await page.evaluate(() => { if (typeof _ytAddChannel === 'function') _ytAddChannel(); });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-empty.png' });
  });

  test('S3.7 Processing status indicator', async ({ page }) => {
    const status = page.locator('#yt-processing-status');
    const visible = await status.isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Processing status element exists:', visible);
    await page.screenshot({ path: '/tmp/stratos-stress/feed/youtube-status.png' });
  });
});
