# StratOS Security Audit Report -- 2026-03-15

## Summary

- **Manual audit findings:** 11 total
- **Fixed:** 6 (all Critical + High)
- **Documented for later:** 5 (Medium + Low)

| Severity | Found | Fixed |
|----------|-------|-------|
| Critical | 2 | 2 |
| High | 3 | 3 |
| Medium | 4 | 1 (RSS SSRF) |
| Low | 2 | 0 |

---

## Critical Findings

### [FINDING-001] SSRF via /api/proxy -- Open Relay to Internal Services

- **Severity:** Critical
- **Status:** Fixed
- **Location:** `backend/routes/media.py:46-78`, `backend/server.py:144`

**Description:** The `/api/proxy` endpoint was **auth-exempt** (because `<img>` tags can't send custom headers) and had **zero URL validation**. Any external attacker could use it to:
- Fetch internal services: `http://127.0.0.1:11434/api/tags` (Ollama model list)
- Scan the internal network (10.x.x.x, 192.168.x.x, 172.16.x.x)
- Access cloud metadata endpoints (169.254.169.254)
- Use non-HTTP schemes: `file:///etc/passwd`, `gopher://`, `dict://`
- Bypass IP-based restrictions on third-party APIs

**Exploit (pre-fix):**
```bash
# Fetch Ollama model list through the proxy
curl "http://localhost:8080/api/proxy?url=http://127.0.0.1:11434/api/tags"

# Read local files via file:// scheme
curl "http://localhost:8080/api/proxy?url=file:///etc/passwd"

# Scan internal network
curl "http://localhost:8080/api/proxy?url=http://192.168.1.1/"
```

**Fix:** Created `backend/routes/url_validation.py` with:
- Scheme whitelist: only `http://` and `https://` allowed
- DNS pre-resolution: resolves hostname to IP *before* connecting (prevents DNS rebinding)
- Private IP blocklist: all RFC 1918, RFC 4193, loopback, link-local, multicast, IPv4-mapped IPv6
- Applied validation in the proxy handler before any fetch occurs

**Code changes:**
- New file: `backend/routes/url_validation.py` (85 lines) -- URL validation utility
- Modified: `backend/routes/media.py` -- added validation call before proxy fetch

**Verification:**
```bash
# These now return 403 "Blocked URL":
curl "http://localhost:8080/api/proxy?url=http://127.0.0.1:11434/api/tags"
curl "http://localhost:8080/api/proxy?url=file:///etc/passwd"
curl "http://localhost:8080/api/proxy?url=http://192.168.1.1/"

# External URLs still work:
curl "http://localhost:8080/api/proxy?url=https://example.com/image.jpg"
```

---

### [FINDING-002] API Keys and SMTP Password Hardcoded in config.yaml

- **Severity:** Critical
- **Status:** Fixed (redacted from config.yaml)
- **Location:** `backend/config.yaml:54-58, 67-70`

**Description:** The following secrets were committed to the git repository in `config.yaml`:
- Google API key: `AIzaSyAYmzOV-kl1m60gidsKAhXKiDM92F4qGmY`
- Google CSE ID: `52d68e3fc17d74e9a`
- Serper API key: `bc26c29537b924cf7e8022eaaf83c50cfdeb51be`
- Gmail SMTP app password: `ctko nnjl skkl kxxj`

These were already in `.env` (gitignored), and `main.py` loads `.env` on startup overriding config.yaml values. But having them in config.yaml means they're in the git history.

**Fix:**
- Replaced all secret values in `config.yaml` with empty strings `""`
- The application still works because `.env` provides the actual values at runtime

**Remaining action needed (manual):**
- Rotate ALL exposed keys: Google API key, Serper API key, Gmail app password
- The Anthropic API key in `.env` was never in config.yaml -- it's safe
- Consider using `git filter-branch` or BFG Repo Cleaner to purge secrets from git history (if repo is public)

---

## High Findings

### [FINDING-003] Password Reset Code -- No Brute-Force Protection

- **Severity:** High
- **Status:** Fixed
- **Location:** `backend/routes/auth.py:496-528`

**Description:** The `/api/auth/reset-password` endpoint accepted a 5-digit verification code (90,000 possibilities) but had no attempt counter. The only protection was global rate limiting (5 requests per 300 seconds), which:
1. Was global per-path (not per-IP), so an attacker's requests counted against ALL users
2. At 5 req/300s = 1 req/60s, exhausting 90K codes would take ~62 days -- slow but feasible for a targeted attack

**Exploit (pre-fix):**
```bash
# Brute-force reset code (rate limit was only 5/300s globally)
for i in $(seq 10000 99999); do
  curl -X POST http://localhost:8080/api/auth/reset-password \
    -d '{"email":"victim@test.com","code":"'$i'","new_password":"hacked123"}'
done
```

**Fix:**
- Added `reset_attempts` column to `users` table (migration 026)
- After 5 failed code attempts, the reset code is invalidated (user must request a new one)
- Attempt counter resets when a new code is issued or when reset succeeds
- User sees remaining attempts in error message

**Code changes:**
- `backend/routes/auth.py`: Added attempt tracking to `/api/auth/reset-password`
- `backend/migrations.py`: Added migration_026 for `reset_attempts` column

---

### [FINDING-004] OTP Login Code -- No Brute-Force Protection

- **Severity:** High
- **Status:** Fixed
- **Location:** `backend/routes/auth.py:579-653`

**Description:** The `/api/auth/otp-verify` endpoint had the same issue as password reset: 5-digit code with no attempt counter. The registration verification endpoint (`/api/auth/verify`) correctly had a 5-attempt limit, but OTP verify was missing it.

**Fix:**
- Added `otp_attempts` column to `users` table (migration 026)
- After 5 failed OTP attempts, the code is invalidated
- Attempt counter resets when a new OTP is issued or when verification succeeds

**Code changes:**
- `backend/routes/auth.py`: Added attempt tracking to `/api/auth/otp-verify`
- `backend/routes/auth.py`: Reset `otp_attempts` on new OTP request and successful verify
- `backend/migrations.py`: Added migration_026 for `otp_attempts` column

---

### [FINDING-005] Rate Limiting is Global Per-Path, Not Per-IP

- **Severity:** High
- **Status:** Fixed
- **Location:** `backend/auth.py:304-318`

**Description:** Rate limiting used the URL path as the sole bucket key. This meant:
1. **DoS vector:** An attacker sending 10 requests to `/api/auth/login` would lock out ALL users from logging in for 300 seconds
2. **Brute-force bypass:** Multiple attackers from different IPs share the same bucket, so they're collectively limited but individually unrestricted
3. Legitimate users are punished for attacker traffic

**Fix:**
- Rate limit buckets are now keyed by `(path, client_ip)` instead of just `path`
- Client IP is extracted from `X-Forwarded-For` header (for reverse proxy setups) or socket address
- Added periodic cleanup of stale rate limit buckets (every 10 minutes)
- Updated `server.py` to pass client IP from `_client_ip()` helper

**Code changes:**
- `backend/auth.py`: `rate_limited()` now accepts `client_ip` parameter, buckets keyed by path+IP
- `backend/server.py`: Added `_client_ip()` helper, passes IP to rate_limited() in do_GET and do_POST

---

## Medium Findings

### [FINDING-006] RSS Discovery SSRF -- Server Fetches Arbitrary URLs

- **Severity:** Medium
- **Status:** Fixed
- **Location:** `backend/routes/feeds.py:365-512`

**Description:** The `/api/discover-rss` endpoint fetches user-provided URLs server-side. While it requires authentication (unlike the proxy), an authenticated user could point it at internal services.

**Fix:** Added the same `url_validation.validate_url()` check used by the proxy.

**Code changes:** `backend/routes/feeds.py`: Added SSRF validation before URL fetch

---

### [FINDING-007] XSS in Agent Response Rendering

- **Severity:** Medium
- **Status:** Documented (fix requires frontend changes)
- **Location:** `frontend/agent.js:778`

**Description:** In `appendAgentMessage()`, the `content` variable is inserted into innerHTML via template literal without escaping:
```javascript
<div class="agent-response ...">${content}</div>
```
While `formatAgentText()` applies escaping in most code paths, this specific path uses raw content. Since content comes from the LLM backend (not directly from user input), exploitation requires either a compromised backend or a prompt injection attack that causes the LLM to output HTML.

**Recommended fix:** Always pass content through `escAgent()` or `formatAgentText()` before innerHTML assignment.

---

### [FINDING-008] XSS in Feed Embed Rendering

- **Severity:** Medium
- **Status:** Documented (fix requires frontend changes)
- **Location:** `frontend/feed.js:545, 600`

**Description:** YouTube and Twitch embed IDs are inserted into inline `onclick` handlers that set `innerHTML` with iframe elements. While `esc()` is applied to the embed IDs, the inline handler pattern (string-in-string-in-onclick) creates a complex escaping context that is fragile.

**Recommended fix:** Replace inline onclick handlers with event listeners and DOM-based iframe creation.

---

### [FINDING-009] CORS Wildcard

- **Severity:** Medium
- **Status:** Documented
- **Location:** `backend/server.py:86-87`

**Description:** Default CORS is `Access-Control-Allow-Origin: *`. Since auth uses custom headers (`X-Auth-Token`) rather than cookies, this doesn't directly enable credential theft. However, it allows any website to make API requests to the server (though they can't include the auth token unless they already have it).

The `cors_origins` config key already supports an allowlist -- just needs to be configured:
```yaml
system:
  cors_origins:
    - "http://localhost:8080"
    - "https://your-domain.com"
```

---

## Low Findings

### [FINDING-010] SQL F-Strings in Migrations

- **Severity:** Low
- **Status:** Documented
- **Location:** `backend/migrations.py:144, 255, 327`, `backend/routes/auth.py:917`

**Description:** Four locations use f-strings to inject table/column names into SQL. All use hardcoded Python lists (not user input), so there's no exploitable injection path. This is a code quality issue, not a vulnerability.

---

### [FINDING-011] SHA-256 Password Comparison Not Timing-Safe

- **Severity:** Low
- **Status:** Documented
- **Location:** `backend/routes/auth.py:41-43`

**Description:** The `_verify_password()` function uses `==` for SHA-256 hash comparison (legacy/fallback path). This is theoretically vulnerable to timing attacks but practically unexploitable because:
1. The primary path uses `bcrypt.checkpw()` which is constant-time
2. SHA-256 comparison requires network-level timing precision (nanoseconds)
3. The fallback only triggers for legacy users who haven't logged in since bcrypt was added

**Recommended fix:** Replace `==` with `hmac.compare_digest()` in the SHA-256 branches.

---

## Files Changed

| File | Change |
|------|--------|
| `backend/routes/url_validation.py` | **NEW** -- SSRF prevention utility (scheme, DNS, private IP checks) |
| `backend/routes/media.py` | Added SSRF validation to /api/proxy |
| `backend/routes/feeds.py` | Added SSRF validation to /api/discover-rss |
| `backend/routes/auth.py` | Added brute-force counters to reset-password and OTP-verify |
| `backend/auth.py` | Rate limiting now per-IP instead of global per-path |
| `backend/server.py` | Added _client_ip() helper, passes IP to rate limiter |
| `backend/migrations.py` | Migration 026: reset_attempts + otp_attempts columns |
| `backend/config.yaml` | Redacted hardcoded API keys and SMTP password |

## Recommended Follow-Up Actions

1. **Rotate all leaked secrets** (Google API key, Serper key, Gmail app password)
2. **Fix frontend XSS** (FINDING-007, FINDING-008) -- requires careful frontend work
3. **Configure CORS allowlist** in config.yaml for production deployment
4. **Add rate limiting to /api/proxy** -- currently no rate limit on the auth-exempt proxy
5. **Consider requiring auth for /api/proxy** and using a signed-URL approach for `<img>` tags
6. **Run Semgrep** for automated static analysis: `cd backend && semgrep --config auto . --severity ERROR --severity WARNING`
