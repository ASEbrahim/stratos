# Backend TODO (from Ahmad / Frontend)

## TTS API Endpoint Needed

**Route:** POST /api/tts
**Module:** routes/media.py (or routes/controls.py)
**Body:** {"text": "string to speak"}
**Response:** audio/wav binary
**Auth:** required (standard profile_id from session)
**Limits:** max 5,000 characters, 1 concurrent request per profile

Use processors/tts.py which already has the Piper integration.
If Piper has no voice model, download en_US-lessac-medium.

Ahmad's frontend handles failure gracefully — if the endpoint doesn't exist yet,
the speak button shows an error and resets. No crash.
