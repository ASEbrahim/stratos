"""
Shared route handler utilities.
JSON response helpers, auth validation, LLM output cleaning.
"""

import json
import gzip
import re


# ═══════════════════════════════════════════════════════════
# LLM OUTPUT CLEANING — shared across all inference endpoints
# ═══════════════════════════════════════════════════════════

# Phrases that indicate the model is reasoning internally, not answering the user.
# Checked case-insensitively against the start of a paragraph.
_REASONING_STARTS = re.compile(
    r'^(okay|ok|alright|so,?\s|let me|let\'s|i need to|i should|i\'ll|i will|'
    r'the user|i called|i used|now i|first,?\s|wait,?\s|looking at|check (?:if|the|for)|'
    r'hmm|maybe|i can see|i notice|it seems|it looks|based on|given that|'
    r'since the|before i|after |from the|according to|we need|we should|'
    r'i have to|now,?\s|right,?\s|thinking|to answer|i see that|'
    r'upon review|reviewing|analyzing|let me check|let me look|'
    r'the question|the answer|i\'m going|going to|the feed|the data|'
    r'there\'s a|there are|there is|'
    # Internal context references (model narrating what it sees in system prompt)
    r'the (?:market|historical|current|recent) (?:section|data|feed|context)|'
    r'the (?:MARKET|HISTORICAL|CURRENT|FEED|CATEGORY|DAILY|TOP)|'
    r'i don\'t see|i see |i found|nothing in|no (?:mention|data|info)|'
    r'avoid |skip |don\'t )',
    re.IGNORECASE
)

# Phrases that mark reasoning when they appear at the START of a paragraph
# anywhere in the response (not just the first paragraph).  Used for mid-response
# reasoning detection.
_MID_REASONING_RE = re.compile(
    r'^(?:wait,?\s|check (?:if|the|for)|i need to|i should|let me|'
    r'the user(?:\'s| said| asked| wants)|'
    r'avoid |skip |don\'t |do not |'
    r'hmm|maybe i|actually,?\s|'
    r'note:|note that|remember|'
    r'so (?:i should|i need|avoid|we should)|'
    r'looking (?:at|for|through)|'
    r'the (?:MARKET|HISTORICAL|CURRENT|FEED|CATEGORY|DAILY|TOP)\b)',
    re.IGNORECASE
)

# Transition phrases where reasoning ends and the answer begins.
_TRANSITION_RE = re.compile(
    r'(?:^|\n)(?:so,?\s*(?:the|here|in)|therefore|thus|hence|in summary|'
    r'to summarize|here(?:\'s| is| are) (?:the|my|a|an|what)|'
    r'final(?:ly|[\s:,])|in conclusion|to conclude|'
    r'the (?:key|main|important|notable|additional) (?:points?|takeaways?|insights?|things?|bullets?|items?|highlights?)|'
    r'(?:additional|the) (?:points?|bullets?|items?) (?:are|include)|'
    r'my (?:analysis|summary|assessment|recommendation|answer))',
    re.IGNORECASE | re.MULTILINE
)


def strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> blocks and bare </think> closers from Qwen3 output.

    Qwen3 sometimes omits the opening <think> tag but includes </think>,
    so the regex <think>.*?</think> doesn't match.  This handles both cases.
    """
    if not text:
        return text
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    return text


def _strip_mid_response_reasoning(text: str) -> str:
    """Detect and truncate reasoning that appears MID-response.

    The model sometimes starts with a clean answer but then transitions into
    internal reasoning partway through.  E.g.:

        "Here are the top trends:\n\n- Oil up 3%\n\nWait, the user said
        'skip generic advice.' So avoid saying 'monitor closely'..."

    This function splits on paragraph breaks and truncates at the first
    paragraph that looks like reasoning, keeping only the clean content before it.
    """
    if not text:
        return text

    parts = re.split(r'\n\n+', text)
    if len(parts) <= 1:
        return text

    # Keep paragraphs until we hit one that looks like mid-response reasoning
    kept = [parts[0]]  # First paragraph already passed the preamble check
    for part in parts[1:]:
        stripped = part.strip()
        if not stripped:
            continue
        if _MID_REASONING_RE.match(stripped):
            break  # Reasoning started — truncate here
        kept.append(part)

    result = '\n\n'.join(kept).strip()
    return result if len(result) > 30 else text  # Don't over-truncate


def strip_reasoning_preamble(text: str) -> str:
    """Strip untagged reasoning from Qwen3 MoE responses.

    Qwen3 sometimes narrates its thought process as plain text (without <think>
    tags).  This aggressively detects reasoning blocks and strips them.

    Strategy:
      1. If the text starts with reasoning: look for a transition phrase and
         extract content after it, or drop reasoning paragraphs from the front.
      2. ALWAYS scan for mid-response reasoning and truncate if found.
      3. Returns empty string if the entire text is reasoning (triggers retry).
    """
    if not text:
        return text

    starts_with_reasoning = _REASONING_STARTS.match(text.strip())

    if starts_with_reasoning:
        # Strategy 1: Look for a transition phrase and take everything after it.
        match = _TRANSITION_RE.search(text)
        if match:
            after = text[match.start():].strip()
            colon_idx = after.find(':')
            newline_idx = after.find('\n')
            if colon_idx != -1 and (newline_idx == -1 or colon_idx < newline_idx):
                candidate = after[colon_idx + 1:].strip()
            else:
                candidate = after
            if len(candidate) > 40:
                # Still check for mid-response reasoning in the extracted content
                return _strip_mid_response_reasoning(candidate)

        # Strategy 2: Drop reasoning paragraphs from the front.
        parts = re.split(r'\n\n+', text)
        if len(parts) < 2:
            return ""  # Single reasoning paragraph — trigger retry

        for i, part in enumerate(parts):
            stripped = part.strip()
            if not stripped:
                continue
            if (stripped.startswith(('**', '#', '- ', '1.', '1)', '* ', '•')) or
                    (i > 0 and not _REASONING_STARTS.match(stripped))):
                result = '\n\n'.join(parts[i:]).strip()
                if len(result) > 40:
                    return _strip_mid_response_reasoning(result)

        return ""  # All paragraphs are reasoning — trigger retry

    # Text doesn't start with reasoning — still check for mid-response reasoning.
    return _strip_mid_response_reasoning(text)


def extract_json(text: str) -> str:
    """Extract JSON from LLM output that may be wrapped in reasoning text.

    Strips everything before the first '{' or '[' and after the last '}' or ']'.
    Returns the extracted JSON string, or the original text if no JSON found.
    """
    if not text:
        return text
    # Try object first
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        return text[start:end + 1]
    # Try array
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def json_response(handler, data, status=200, compress=True):
    """Send a JSON response with optional gzip compression."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")

    if compress and len(body) > 1024 and 'gzip' in handler.headers.get('Accept-Encoding', ''):
        body = gzip.compress(body)
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Content-Length", str(len(body)))

    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler, message, status=400):
    """Send a JSON error response."""
    json_response(handler, {"error": str(message)}, status=status, compress=False)


def read_json_body(handler) -> dict:
    """Read and parse JSON from request body. Returns {} on empty/invalid body."""
    content_length = int(handler.headers.get('Content-Length', 0))
    if content_length == 0:
        return {}
    raw = handler.rfile.read(content_length)
    text = raw.decode('utf-8').strip()
    if not text:
        return {}
    return json.loads(text)


def sse_event(handler, data: dict):
    """Send a Server-Sent Event."""
    handler.wfile.write(f'data: {json.dumps(data)}\n\n'.encode())
    handler.wfile.flush()


def start_sse(handler):
    """Set up headers for SSE streaming."""
    handler.send_response(200)
    handler.send_header("Content-type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
