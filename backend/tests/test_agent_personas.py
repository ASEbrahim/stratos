"""
Agent Persona Test Suite — Sprint 1 Phase 5

Stress-tests each persona via real HTTP requests to /api/agent-chat.
Validates: response quality, tool invocation, streaming, error handling.

Usage:
    python3 tests/test_agent_personas.py [--host http://localhost:8080] [--token TOKEN]
"""

import json
import os
import sys
import time
import argparse
import requests
from typing import Dict, Any, List, Optional

# Ensure backend/ is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════

PERSONA_TESTS = {
    "intelligence": [
        {
            "name": "basic_greeting",
            "message": "Hello, what can you do?",
            "expect_tokens": True,
            "min_length": 30,
            "expect_tool": None,
        },
        {
            "name": "feed_context_use",
            "message": "What are the top 3 signals from my feed right now?",
            "expect_tokens": True,
            "min_length": 50,
            "expect_tool": None,  # Should use context, not tools
        },
        {
            "name": "search_feed_tool",
            "message": "Search my feed history for articles about oil prices",
            "expect_tokens": True,
            "min_length": 20,
            "expect_tool": "search_feed",
        },
        {
            "name": "category_list",
            "message": "What categories am I tracking?",
            "expect_tokens": True,
            "min_length": 20,
            "expect_tool": "manage_categories",
        },
    ],
    "market": [
        {
            "name": "market_greeting",
            "message": "How are markets doing?",
            "expect_tokens": True,
            "min_length": 30,
            "expect_tool": None,
        },
        {
            "name": "watchlist_check",
            "message": "Show me my watchlist",
            "expect_tokens": True,
            "min_length": 20,
            "expect_tool": "manage_watchlist",
        },
    ],
    "scholarly": [
        {
            "name": "scholarly_greeting",
            "message": "Tell me about the history of the Abbasid Caliphate",
            "expect_tokens": True,
            "min_length": 100,
            "expect_tool": None,
        },
        {
            "name": "scholarly_arabic",
            "message": "What does the word 'jihad' actually mean in Arabic?",
            "expect_tokens": True,
            "min_length": 50,
            "expect_tool": None,
        },
    ],
}


# ═══════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════

class AgentTestRunner:
    def __init__(self, host: str, token: str = ""):
        self.host = host.rstrip("/")
        self.token = token
        self.results: List[Dict[str, Any]] = []

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["X-Auth-Token"] = self.token
        return h

    def run_test(self, persona: str, test: dict) -> Dict[str, Any]:
        """Run a single test case against the agent."""
        name = f"{persona}/{test['name']}"
        result = {
            "name": name,
            "passed": False,
            "tokens_received": 0,
            "response_text": "",
            "tool_calls_seen": [],
            "errors": [],
            "elapsed_ms": 0,
        }

        try:
            start = time.time()
            resp = requests.post(
                f"{self.host}/api/agent-chat",
                headers=self._headers(),
                json={
                    "message": test["message"],
                    "history": [],
                    "persona": persona,
                },
                stream=True,
                timeout=120,
            )
            if resp.status_code != 200:
                result["errors"].append(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return result

            full_text = ""
            token_count = 0
            tool_calls = []

            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8", errors="replace")
                if text.startswith("data: "):
                    text = text[6:]
                try:
                    event = json.loads(text)
                    if "token" in event:
                        full_text += event["token"]
                        token_count += 1
                    if "status" in event:
                        # Tool call status like "🔍 web_search..."
                        status = event["status"]
                        for tool_name in ["web_search", "search_feed", "manage_watchlist",
                                          "manage_categories", "search_files", "read_document"]:
                            if tool_name in status:
                                tool_calls.append(tool_name)
                    if "error" in event:
                        result["errors"].append(event["error"])
                    if event.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

            elapsed = (time.time() - start) * 1000
            result["elapsed_ms"] = round(elapsed)
            result["tokens_received"] = token_count
            result["response_text"] = full_text[:500]
            result["tool_calls_seen"] = tool_calls

            # ── Assertions ──
            passed = True
            if test.get("expect_tokens") and token_count == 0:
                result["errors"].append("Expected tokens but got none")
                passed = False
            if test.get("min_length") and len(full_text) < test["min_length"]:
                result["errors"].append(
                    f"Response too short: {len(full_text)} < {test['min_length']}"
                )
                passed = False
            if test.get("expect_tool"):
                if test["expect_tool"] not in tool_calls:
                    result["errors"].append(
                        f"Expected tool '{test['expect_tool']}' but saw: {tool_calls}"
                    )
                    # Soft fail — tool use is probabilistic with LLMs
                    # Don't mark as failed, just log the warning
            if result["errors"] and not any("Expected tool" in e for e in result["errors"]):
                passed = False
            elif not result["errors"]:
                passed = True
            else:
                # Only tool expectation misses — still pass
                passed = len(full_text) >= test.get("min_length", 0) and token_count > 0

            result["passed"] = passed

        except requests.Timeout:
            result["errors"].append("Request timed out (120s)")
        except Exception as e:
            result["errors"].append(f"Exception: {e}")

        return result

    def run_all(self, personas: Optional[List[str]] = None):
        """Run all tests for specified personas (or all)."""
        test_personas = personas or list(PERSONA_TESTS.keys())
        total = sum(len(PERSONA_TESTS.get(p, [])) for p in test_personas)
        print(f"\n{'='*60}")
        print(f"  Agent Persona Test Suite — {total} tests")
        print(f"  Host: {self.host}")
        print(f"{'='*60}\n")

        passed = 0
        failed = 0
        warnings = 0

        for persona in test_personas:
            tests = PERSONA_TESTS.get(persona, [])
            if not tests:
                continue
            print(f"  ── {persona.upper()} ({len(tests)} tests) ──")

            for test in tests:
                name = f"{persona}/{test['name']}"
                sys.stdout.write(f"    {name} ... ")
                sys.stdout.flush()

                result = self.run_test(persona, test)
                self.results.append(result)

                if result["passed"]:
                    tool_warns = [e for e in result["errors"] if "Expected tool" in e]
                    if tool_warns:
                        print(f"PASS (warn: {tool_warns[0]}) [{result['elapsed_ms']}ms]")
                        warnings += 1
                    else:
                        print(f"PASS [{result['elapsed_ms']}ms, {result['tokens_received']} tokens]")
                    passed += 1
                else:
                    print(f"FAIL")
                    for err in result["errors"]:
                        print(f"      ✗ {err}")
                    if result["response_text"]:
                        preview = result["response_text"][:150].replace("\n", " ")
                        print(f"      Response: {preview}...")
                    failed += 1

            print()

        # ── Summary ──
        print(f"{'='*60}")
        print(f"  Results: {passed} passed, {failed} failed, {warnings} warnings")
        print(f"  Total time: {sum(r['elapsed_ms'] for r in self.results)/1000:.1f}s")
        print(f"{'='*60}\n")

        return failed == 0


# ═══════════════════════════════════════════════════════════
# STANDALONE TESTS (no server needed)
# ═══════════════════════════════════════════════════════════

def test_persona_imports():
    """Test persona module imports and configuration."""
    print("\n  ── UNIT TESTS (no server) ──")
    errors = []

    # Test imports
    try:
        from routes.personas import (
            list_personas, get_persona_config, build_persona_prompt,
            PERSONA_TOOLS, PERSONA_GREETINGS
        )
    except ImportError as e:
        errors.append(f"Import failed: {e}")
        return errors

    # All personas have tools defined
    personas = list_personas()
    names = [p["name"] for p in personas]
    for name in names:
        if name not in PERSONA_TOOLS:
            errors.append(f"Persona '{name}' missing from PERSONA_TOOLS")
        if name not in PERSONA_GREETINGS:
            errors.append(f"Persona '{name}' missing from PERSONA_GREETINGS")

    # Intelligence must have core tools
    intel_tools = PERSONA_TOOLS.get("intelligence", [])
    for t in ["web_search", "search_feed", "manage_watchlist", "manage_categories"]:
        if t not in intel_tools:
            errors.append(f"Intelligence missing tool: {t}")

    # Scholarly must have document tools
    scholar_tools = PERSONA_TOOLS.get("scholarly", [])
    for t in ["search_files", "read_document"]:
        if t not in scholar_tools:
            errors.append(f"Scholarly missing tool: {t}")

    # Prompt generation
    for name in ["intelligence", "market", "scholarly"]:
        prompt = build_persona_prompt(name, "Engineer", "Kuwait", ["AAPL"], "Tech (5 kw)", "web search available")
        if len(prompt) < 50:
            errors.append(f"{name} prompt too short: {len(prompt)}")
        # Word count check (~300 word limit for 9B model)
        word_count = len(prompt.split())
        if word_count > 350:
            errors.append(f"{name} prompt too long: {word_count} words (max ~300)")

    # Agent tool definitions
    from routes.agent import AGENT_TOOLS
    tool_names = [t["function"]["name"] for t in AGENT_TOOLS]
    for expected in ["web_search", "manage_watchlist", "search_feed", "manage_categories",
                     "search_files", "read_document"]:
        if expected not in tool_names:
            errors.append(f"AGENT_TOOLS missing: {expected}")

    for err in errors:
        print(f"    ✗ {err}")
    if not errors:
        print(f"    ✓ All {len(names)} personas configured correctly")
        print(f"    ✓ All {len(tool_names)} tools defined")
        print(f"    ✓ Prompt word counts within limits")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Agent Persona Test Suite")
    parser.add_argument("--host", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--token", default="", help="Auth token")
    parser.add_argument("--persona", nargs="*", help="Test specific persona(s)")
    parser.add_argument("--unit-only", action="store_true", help="Run unit tests only (no server)")
    args = parser.parse_args()

    # Always run unit tests first
    unit_errors = test_persona_imports()

    if args.unit_only:
        sys.exit(1 if unit_errors else 0)

    # Integration tests
    print()
    runner = AgentTestRunner(args.host, args.token)
    success = runner.run_all(args.persona)
    sys.exit(0 if success and not unit_errors else 1)


if __name__ == "__main__":
    main()
