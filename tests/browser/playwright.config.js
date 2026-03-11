const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: '.',
  timeout: 30000,
  use: {
    baseURL: 'http://localhost:8080',
    browserName: 'chromium',
    screenshot: 'on',
    trace: 'retain-on-failure',
    viewport: { width: 1920, height: 1080 },
  },
  outputDir: '/tmp/stratos-qa/results',
  reporter: [['list'], ['json', { outputFile: '/tmp/stratos-qa/results.json' }]],
});
