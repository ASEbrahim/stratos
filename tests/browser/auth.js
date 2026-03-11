// Shared auth helper — injects token + profile for Developer_KW (profile_id=4)
const AUTH_TOKEN = '0e704d2baab3464b1014de26cbde6cd543c5d99a0a2bb2c68fc7783f0923c415';
const TEST_PROFILE = 'Developer_KW';
const TEST_PROFILE_ID = 4;

async function login(page) {
  // Inject auth token and profile into localStorage before navigating
  await page.goto('/');
  await page.evaluate(({ token, profile }) => {
    localStorage.setItem('stratos_auth_token', token);
    localStorage.setItem('stratos_active_profile', profile);
    localStorage.setItem('stratos_tour_never', 'true');
  }, { token: AUTH_TOKEN, profile: TEST_PROFILE });
  // Reload so the app picks up the injected auth
  await page.goto('/');
  await page.waitForTimeout(3000);
  // Dismiss any tour overlay that might appear
  await page.evaluate(() => {
    const overlay = document.getElementById('stratos-tour-overlay');
    if (overlay) overlay.remove();
    const welcome = document.getElementById('stratos-tour-welcome');
    if (welcome) welcome.remove();
  });
}

module.exports = { login, AUTH_TOKEN, TEST_PROFILE, TEST_PROFILE_ID };
