const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Phase 5: Games & Scenarios Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
  });

  test('5.1 Scenario bar appears for gaming persona', async ({ page }) => {
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('gaming'); });
    await page.waitForTimeout(1000);
    const bar = await page.locator('#games-scenario-bar').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase5/scenario-bar.png' });
    expect(bar).toBeTruthy();
  });

  test('5.2 Scenario bar hidden for non-gaming persona', async ({ page }) => {
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('intelligence'); });
    await page.waitForTimeout(1000);
    const bar = await page.locator('#games-scenario-bar').isVisible({ timeout: 2000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase5/no-scenario-bar.png' });
    expect(bar).toBeFalsy();
  });

  test('5.3 Create scenario', async ({ page }) => {
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('gaming'); });
    await page.waitForTimeout(1000);

    page.on('dialog', async dialog => {
      if (dialog.type() === 'prompt') {
        await dialog.accept('QA_Test_World');
      } else {
        await dialog.accept();
      }
    });

    await page.evaluate(() => { if (typeof _gamesCreateScenario === 'function') _gamesCreateScenario(); });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase5/scenario-created.png' });
    const content = await page.locator('#games-scenario-content, #games-scenario-bar').textContent().catch(() => '');
    console.log('Scenario bar content:', content.substring(0, 200));
  });

  test('5.4 Create second scenario and switch', async ({ page }) => {
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('gaming'); });
    await page.waitForTimeout(1000);

    page.on('dialog', async dialog => {
      if (dialog.type() === 'prompt') {
        await dialog.accept('QA_Second_World');
      } else {
        await dialog.accept();
      }
    });

    await page.evaluate(() => { if (typeof _gamesCreateScenario === 'function') _gamesCreateScenario(); });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/stratos-qa/phase5/two-scenarios.png' });

    // Check dropdown options
    const options = await page.locator('#games-scenario-content select option, #games-scenario-bar select option').count().catch(() => 0);
    console.log('Scenario dropdown options:', options);
  });

  test('5.5 Delete scenario', async ({ page }) => {
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('gaming'); });
    await page.waitForTimeout(1000);

    page.on('dialog', async dialog => {
      await dialog.accept(); // Confirm delete
    });

    await page.evaluate(() => {
      if (typeof _gamesDeleteScenario === 'function') _gamesDeleteScenario('QA_Second_World');
    });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: '/tmp/stratos-qa/phase5/scenario-deleted.png' });
  });
});
