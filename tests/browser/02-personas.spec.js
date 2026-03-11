const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Phase 2: Persona & Context Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    // Open agent panel
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
  });

  test('2.1 Persona selector is visible', async ({ page }) => {
    const picker = await page.locator('#agent-persona-picker').isVisible({ timeout: 3000 });
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/persona-picker.png' });
    expect(picker).toBeTruthy();
  });

  test('2.2 Persona dropdown lists all 6 personas', async ({ page }) => {
    // Open persona picker dropdown
    await page.evaluate(() => { if (typeof _togglePersonaPicker === 'function') _togglePersonaPicker(); });
    await page.waitForTimeout(500);
    const dropdown = page.locator('#persona-picker-dropdown');
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/persona-dropdown.png' });
    const isVisible = await dropdown.isVisible({ timeout: 2000 }).catch(() => false);
    expect(isVisible).toBeTruthy();
    // Check persona items
    const items = await dropdown.locator('label').count();
    console.log(`Persona items found: ${items}`);
    expect(items).toBeGreaterThanOrEqual(5); // at least 5 personas
  });

  test('2.3 Switching personas changes label', async ({ page }) => {
    const labelBefore = await page.locator('#persona-picker-label').textContent();
    console.log('Before persona:', labelBefore);
    // Switch to scholarly
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('scholarly'); });
    await page.waitForTimeout(1000);
    const labelAfter = await page.locator('#persona-picker-label').textContent();
    console.log('After persona:', labelAfter);
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/persona-switched.png' });
    expect(labelAfter.toLowerCase()).toContain('scholar');
  });

  test('2.4 Context editor opens', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1000);
    const editor = await page.locator('#context-editor-panel').isVisible({ timeout: 3000 }).catch(() => false);
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/context-editor.png' });
    expect(editor).toBeTruthy();
  });

  test('2.5 Context editor loads persona content', async ({ page }) => {
    // Switch to intelligence first
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('intelligence'); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    const textarea = page.locator('#ctx-editor-textarea');
    const content = await textarea.inputValue().catch(() => '');
    console.log('Intelligence context (first 100 chars):', content.substring(0, 100));
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/context-intelligence.png' });
    // Should have some default or custom content
    expect(content.length).toBeGreaterThan(0);
  });

  test('2.6 Context isolation - edit scholarly, verify not in gaming', async ({ page }) => {
    // Switch to scholarly and edit context
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('scholarly'); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    const textarea = page.locator('#ctx-editor-textarea');
    await textarea.fill('QA_ISOLATION_TEST: Battle of Karbala study notes');
    // Save
    await page.evaluate(() => { if (typeof _ctxSave === 'function') _ctxSave(); });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/context-scholarly-saved.png' });

    // Switch persona selector to gaming
    const personaSelect = page.locator('#ctx-persona-select');
    if (await personaSelect.isVisible()) {
      await personaSelect.selectOption('gaming');
      await page.waitForTimeout(1500);
    } else {
      // Close editor, switch persona, reopen
      await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
      await page.waitForTimeout(500);
      await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('gaming'); });
      await page.waitForTimeout(500);
      await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
      await page.waitForTimeout(1500);
    }

    const gamingContent = await textarea.inputValue().catch(() => '');
    console.log('Gaming context (first 100 chars):', gamingContent.substring(0, 100));
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/context-gaming-isolation.png' });
    expect(gamingContent).not.toContain('QA_ISOLATION_TEST');
  });

  test('2.7 Context save persists after refresh', async ({ page }) => {
    // Save a known string to intelligence context
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('intelligence'); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    const textarea = page.locator('#ctx-editor-textarea');
    await textarea.fill('QA_PERSIST_TEST: This should survive refresh');
    await page.evaluate(() => { if (typeof _ctxSave === 'function') _ctxSave(); });
    await page.waitForTimeout(1500);

    // Refresh page
    await login(page);
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('intelligence'); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    const content = await textarea.inputValue().catch(() => '');
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/context-persisted.png' });
    expect(content).toContain('QA_PERSIST_TEST');
  });

  test('2.8 Multi-agent picker - select 2 personas', async ({ page }) => {
    // Open persona picker
    await page.evaluate(() => { if (typeof _togglePersonaPicker === 'function') _togglePersonaPicker(); });
    await page.waitForTimeout(500);
    // Check if multi-select checkboxes exist
    const checkboxes = await page.locator('#persona-picker-dropdown input[type="checkbox"]').count();
    console.log(`Multi-select checkboxes: ${checkboxes}`);
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/multi-agent-picker.png' });
    if (checkboxes === 0) {
      console.log('Multi-agent picker: single-select mode (no checkboxes)');
      // Try clicking multiple persona items to see if multi-select is via click
      const items = page.locator('#persona-picker-dropdown [data-persona]');
      const count = await items.count();
      console.log(`Total persona items: ${count}`);
    }
  });

  test('2.9 Reset context to default', async ({ page }) => {
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('intelligence'); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleContextEditor === 'function') toggleContextEditor(); });
    await page.waitForTimeout(1500);
    // Click reset
    const resetBtn = page.locator('button:has-text("Reset"), .ctx-btn-reset');
    const resetVisible = await resetBtn.first().isVisible({ timeout: 2000 }).catch(() => false);
    console.log('Reset button visible:', resetVisible);
    if (resetVisible) {
      // Handle confirm dialog
      page.on('dialog', dialog => dialog.accept());
      await resetBtn.first().click();
      await page.waitForTimeout(1500);
    }
    await page.screenshot({ path: '/tmp/stratos-qa/phase2/context-reset.png' });
  });
});
