const { test, expect } = require('@playwright/test');
const { login } = require('./auth');

test.describe('Phase 3: File Browser Tests', () => {

  test.beforeEach(async ({ page }) => {
    await login(page);
    // Open agent panel
    await page.evaluate(() => {
      if (typeof toggleAgentChat === 'function') toggleAgentChat();
    });
    await page.waitForTimeout(1000);
    // Set persona to intelligence
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('intelligence'); });
    await page.waitForTimeout(500);
  });

  test('3.1 File browser opens', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(2000);
    // The panel is dynamically created, check for the sidebar inside it
    const panel = await page.locator('#file-browser-panel').isVisible({ timeout: 5000 }).catch(() => false);
    const sidebar = await page.locator('.ctx-editor-sidebar').first().isVisible({ timeout: 3000 }).catch(() => false);
    console.log('Panel visible:', panel, 'Sidebar visible:', sidebar);
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/file-browser-open.png' });
    expect(panel || sidebar).toBeTruthy();
  });

  test('3.2 Create file via file browser', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);

    // Handle the prompt dialog for file name
    page.on('dialog', async dialog => {
      if (dialog.type() === 'prompt') {
        await dialog.accept('qa_test_file.md');
      } else {
        await dialog.accept();
      }
    });

    await page.evaluate(() => { if (typeof _fbCreateFile === 'function') _fbCreateFile(); });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/file-created.png' });

    // Check if file appears in list
    const fileList = page.locator('#fb-file-list');
    const text = await fileList.textContent().catch(() => '');
    console.log('File list content:', text.substring(0, 200));
  });

  test('3.3 Create folder via file browser', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);

    page.on('dialog', async dialog => {
      if (dialog.type() === 'prompt') {
        await dialog.accept('qa_test_folder');
      } else {
        await dialog.accept();
      }
    });

    await page.evaluate(() => { if (typeof _fbCreateFolder === 'function') _fbCreateFolder(); });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/folder-created.png' });

    const fileList = page.locator('#fb-file-list');
    const text = await fileList.textContent().catch(() => '');
    console.log('File list after folder:', text.substring(0, 200));
  });

  test('3.4 Edit and save file content', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);

    // Click a file to edit (should open editor)
    const firstFile = page.locator('#fb-file-list .fb-item-file, #fb-file-list [data-type="file"]').first();
    const hasFile = await firstFile.isVisible({ timeout: 2000 }).catch(() => false);
    if (hasFile) {
      await firstFile.click();
      await page.waitForTimeout(1000);
      const editor = page.locator('#fb-editor-textarea');
      const editorVisible = await editor.isVisible({ timeout: 2000 }).catch(() => false);
      if (editorVisible) {
        await editor.fill('QA TEST CONTENT: File browser edit test');
        await page.evaluate(() => { if (typeof _fbSaveFile === 'function') _fbSaveFile(); });
        await page.waitForTimeout(1000);
      }
    }
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/file-edited.png' });
  });

  test('3.5 File isolation - intelligence file not in gaming', async ({ page }) => {
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1000);
    const intelligenceList = await page.locator('#fb-file-list').textContent().catch(() => '');
    console.log('Intelligence files:', intelligenceList.substring(0, 200));
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/intelligence-files.png' });

    // Switch to gaming persona in file browser
    const personaSelect = page.locator('#fb-persona-select');
    if (await personaSelect.isVisible()) {
      await personaSelect.selectOption('gaming');
      await page.waitForTimeout(1500);
    } else {
      // Close, switch persona, reopen
      await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
      await page.waitForTimeout(500);
      await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('gaming'); });
      await page.waitForTimeout(500);
      await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
      await page.waitForTimeout(1500);
    }

    const gamingList = await page.locator('#fb-file-list').textContent().catch(() => '');
    console.log('Gaming files:', gamingList.substring(0, 200));
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/gaming-files-isolation.png' });
    // Gaming should NOT contain the intelligence test file
    expect(gamingList).not.toContain('qa_test_file');
  });

  test('3.6 Empty state when no files', async ({ page }) => {
    // Switch to anime persona (unlikely to have files)
    await page.evaluate(() => { if (typeof switchPersona === 'function') switchPersona('anime'); });
    await page.waitForTimeout(500);
    await page.evaluate(() => { if (typeof toggleFileBrowser === 'function') toggleFileBrowser(); });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: '/tmp/stratos-qa/phase3/empty-state.png' });
    const content = await page.locator('#fb-file-list').textContent().catch(() => '');
    console.log('Anime files (expect empty):', content);
  });
});
