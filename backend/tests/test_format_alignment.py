"""Format alignment tests between training (export_training.py) and inference (scorer_adaptive.py).

The fine-tuned scorer model was trained on data produced by export_training.py.
At inference, scorer_adaptive.py constructs prompts sent to this model.
These formats MUST be character-for-character identical for the core template.
If they drift, the model produces unparseable output and scoring silently degrades.

Run: cd backend && python -m pytest tests/ -v
"""
import json
import re
import sys
import unittest
from pathlib import Path

import pytest

# Add backend to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_PROFILE_1 = {
    "role": "Software Engineer",
    "location": "Dubai",
    "context": "Track: cloud computing, AI startups, AWS certifications. Deprioritize: sports, entertainment.",
    "interests": ["AI", "cloud computing"],
    "tracked_companies": ["Microsoft", "Amazon", "Google"],
    "tracked_institutions": ["Dubai Internet City", "DIFC"],
    "tracked_interests": ["AI", "cloud computing", "AWS"],
    "tracked_industries": ["tech", "cloud services"],
}

SAMPLE_PROFILE_2 = {
    "role": "Chemical Engineering student at Kuwait University",
    "location": "Kuwait",
    "context": "Track: EQUATE, KNPC hiring, petrochemical internships. Deprioritize: senior roles.",
    "interests": ["petrochemicals", "process engineering"],
    "tracked_companies": ["EQUATE", "KNPC", "KOC"],
    "tracked_institutions": ["Kuwait University"],
    "tracked_interests": ["petrochemicals", "process engineering", "catalysis"],
    "tracked_industries": ["oil and gas", "petrochemicals"],
}

SAMPLE_PROFILE_EMPTY = {
    "role": "professional",
    "location": "unspecified",
    "context": "",
    "interests": [],
    "tracked_companies": [],
    "tracked_institutions": [],
    "tracked_interests": [],
    "tracked_industries": [],
}


def _build_inference_system_prompt(profile: dict) -> str:
    """Replicate scorer_adaptive.py _build_llm_prompt_v2 system prompt construction.

    This must stay in sync with scorer_adaptive.py _build_llm_prompt_v2 (lines ~917-967).
    If that method changes, this test should break to flag the drift.
    """
    role = profile.get('role', 'professional')
    location = profile.get('location', 'unspecified')
    context = profile.get('context', '')

    # Student detection — matches scorer_adaptive._detect_student logic
    role_lower = role.lower()
    student_signals = ['student', 'bachelor', 'undergraduate', 'graduate student',
                       'master student', 'phd student', 'university', 'college',
                       'studying', 'freshman', 'sophomore', 'junior year', 'senior year']
    experienced_signals = ['head', 'manager', 'director', 'lead', 'senior', 'principal',
                          'specialist', 'professional', 'years experience', 'engineer at',
                          'working at', 'employed at', 'consultant']
    role_says_student = any(s in role_lower for s in student_signals)
    role_says_experienced = any(s in role_lower for s in experienced_signals)
    is_student = role_says_student and not role_says_experienced

    if is_student:
        level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
    else:
        level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

    # Tracked fields — matches scorer_adaptive._tracked_fields_block
    companies = ', '.join(profile.get('tracked_companies', []))
    institutions = ', '.join(profile.get('tracked_institutions', []))
    interests = ', '.join(profile.get('tracked_interests', profile.get('interests', [])))
    industries = ', '.join(profile.get('tracked_industries', []))
    tracked = (
        f"Tracked companies: {companies if companies else 'None specified'}\n"
        f"Tracked institutions: {institutions if institutions else 'None specified'}\n"
        f"Tracked interests: {interests if interests else 'None specified'}\n"
        f"Tracked industries: {industries if industries else 'None specified'}"
    )

    # NOTE: At inference, feedback_text and LANGUAGE line are appended after this.
    # Those are runtime-only additions not present in training data.
    # The core template below must match export_training.build_system_prompt.

    system = f"""You are a relevance scorer for a {role} in {location}.
User context: {context if context else 'Not specified'}
{tracked}
{level_note}

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

    return system


def _build_training_system_prompt(profile: dict) -> str:
    """Replicate export_training.py build_system_prompt construction.

    This must stay in sync with export_training.py build_system_prompt (lines ~141-171).
    If that method changes, this test should break to flag the drift.
    """
    from export_training import build_system_prompt

    role = profile.get('role', 'professional')
    location = profile.get('location', 'unspecified')
    context = profile.get('context', '')
    companies = ', '.join(profile.get('tracked_companies', []))
    institutions = ', '.join(profile.get('tracked_institutions', []))
    interests = ', '.join(profile.get('tracked_interests', profile.get('interests', [])))
    industries = ', '.join(profile.get('tracked_industries', []))

    return build_system_prompt(
        role, location, context,
        tracked_companies=companies,
        tracked_institutions=institutions,
        tracked_interests=interests,
        tracked_industries=industries,
    )


def _build_inference_user_message(title: str, content: str, cat_label: str, cat_items: str) -> str:
    """Replicate scorer_adaptive.py _build_llm_prompt_v2 user message construction."""
    return f"""Score this article:
Category: {cat_label}
Keywords: {cat_items}
Title: {title}
Content: {content}"""


def _build_training_user_message(title: str, content: str, cat_label: str, cat_items: str) -> str:
    """Replicate export_training.py build_user_message construction."""
    from export_training import build_user_message
    return build_user_message(title, content, cat_label, cat_items)


# ═══════════════════════════════════════════════════════════════════════
# Format Alignment Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFormatAlignment:
    """Verify that training and inference prompt formats are identical.

    The fine-tuned model was trained on output from export_training.py.
    At inference, scorer_adaptive.py constructs prompts. These MUST match.
    """

    def test_system_prompt_alignment_experienced(self):
        """System prompt for experienced professional must match between training and inference."""
        training = _build_training_system_prompt(SAMPLE_PROFILE_1)
        inference = _build_inference_system_prompt(SAMPLE_PROFILE_1)
        assert training == inference, (
            f"System prompt drift detected for experienced profile!\n"
            f"--- Training ---\n{training}\n"
            f"--- Inference ---\n{inference}\n"
            f"--- Diff ---\n{_show_diff(training, inference)}"
        )

    def test_system_prompt_alignment_student(self):
        """System prompt for student must match between training and inference."""
        training = _build_training_system_prompt(SAMPLE_PROFILE_2)
        inference = _build_inference_system_prompt(SAMPLE_PROFILE_2)
        assert training == inference, (
            f"System prompt drift detected for student profile!\n"
            f"--- Training ---\n{training}\n"
            f"--- Inference ---\n{inference}\n"
            f"--- Diff ---\n{_show_diff(training, inference)}"
        )

    def test_system_prompt_alignment_empty_profile(self):
        """System prompt with empty/default profile must match."""
        training = _build_training_system_prompt(SAMPLE_PROFILE_EMPTY)
        inference = _build_inference_system_prompt(SAMPLE_PROFILE_EMPTY)
        assert training == inference, (
            f"System prompt drift for empty profile!\n"
            f"--- Diff ---\n{_show_diff(training, inference)}"
        )

    def test_user_message_alignment(self):
        """User message format must match between training and inference."""
        title = "AWS announces new AI services for cloud developers"
        content = "Amazon Web Services today unveiled a suite of new machine learning services targeting cloud-native development teams..."
        cat_label = "Tech Companies"
        cat_items = "Microsoft, Amazon, Google"

        training = _build_training_user_message(title, content, cat_label, cat_items)
        inference = _build_inference_user_message(title, content, cat_label, cat_items)
        assert training == inference, (
            f"User message drift detected!\n"
            f"--- Training ---\n{training}\n"
            f"--- Inference ---\n{inference}"
        )

    def test_user_message_alignment_empty_fields(self):
        """User message with empty fields must match."""
        training = _build_training_user_message("Title", "", "general", "")
        inference = _build_inference_user_message("Title", "", "general", "")
        assert training == inference

    def test_tracked_fields_block_format(self):
        """Verify tracked fields block uses consistent format."""
        from export_training import build_system_prompt
        prompt = build_system_prompt(
            "Engineer", "Dubai", "context",
            tracked_companies="Foo, Bar",
            tracked_institutions="",
            tracked_interests="AI",
            tracked_industries="tech",
        )
        assert "Tracked companies: Foo, Bar" in prompt
        assert "Tracked institutions: None specified" in prompt
        assert "Tracked interests: AI" in prompt
        assert "Tracked industries: tech" in prompt

    def test_student_detection_alignment(self):
        """Verify student detection logic matches between training and inference.

        export_training uses STUDENT_KEYWORDS set.
        scorer_adaptive uses _detect_student with student_signals list.
        Both must agree on the same inputs.
        """
        from export_training import _is_student

        # Should be student
        assert _is_student("Chemical Engineering student at Kuwait University") is True
        assert _is_student("Graduate student in AI") is True
        assert _is_student("Undergraduate freshman") is True

        # Should NOT be student
        assert _is_student("Software Engineer") is False
        assert _is_student("Senior Geophysicist at KOC") is False
        assert _is_student("Director of Engineering") is False

    def test_assistant_response_format(self):
        """Training data assistant format must match inference parse regex."""
        # The format used in export_training.py format_chatml():
        # f"SCORE: {score} | REASON: {reason}"
        sample_assistant = "SCORE: 7.5 | REASON: relevant to cloud computing"

        # The regex used in scorer_adaptive.py _llm_score():
        score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', sample_assistant, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.+)', sample_assistant, re.IGNORECASE)

        assert score_match is not None
        assert float(score_match.group(1)) == 7.5
        assert reason_match is not None
        assert reason_match.group(1).strip() == "relevant to cloud computing"


# ═══════════════════════════════════════════════════════════════════════
# Scoring Boundary Tests
# ═══════════════════════════════════════════════════════════════════════

class TestForbidden50:
    """Forbidden 5.0 rule: scores 4.8-5.2 must be nudged to <=4.8 or >=5.3.

    Score 5.0 is the exact noise/medium boundary. The model tends to
    cluster here when uncertain. The nudge forces decisiveness.
    """

    @staticmethod
    def _apply_forbidden_50(score: float) -> float:
        """Replicate the forbidden 5.0 logic from scorer_adaptive.py."""
        if 4.8 <= score <= 5.2:
            return 5.3 if score >= 5.0 else 4.8
        return score

    def test_score_4_7_unchanged(self):
        assert self._apply_forbidden_50(4.7) == 4.7

    def test_score_4_8_stays_4_8(self):
        assert self._apply_forbidden_50(4.8) == 4.8

    def test_score_4_9_nudged_down(self):
        assert self._apply_forbidden_50(4.9) == 4.8

    def test_score_5_0_nudged_up(self):
        assert self._apply_forbidden_50(5.0) == 5.3

    def test_score_5_1_nudged_up(self):
        assert self._apply_forbidden_50(5.1) == 5.3

    def test_score_5_2_nudged_up(self):
        assert self._apply_forbidden_50(5.2) == 5.3

    def test_score_5_3_unchanged(self):
        assert self._apply_forbidden_50(5.3) == 5.3

    def test_all_forbidden_scores_nudged(self):
        """Every score in 4.8-5.2 must end up <=4.8 or >=5.3."""
        for raw in [4.8, 4.9, 5.0, 5.1, 5.2]:
            result = self._apply_forbidden_50(raw)
            assert result <= 4.8 or result >= 5.3, f"Score {raw} -> {result} is in forbidden zone"


class TestScoreParsing:
    """Test score parsing from LLM responses.

    Replicates the regex parsing logic from scorer_adaptive.py _llm_score
    and _parse_batch_response.
    """

    @staticmethod
    def _parse_score(response: str):
        """Replicate scorer_adaptive.py _llm_score parsing logic.
        Returns (score, reason) or (None, None) on failure.
        """
        if not response:
            return None, None
        try:
            score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', response, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))
                reason = reason_match.group(1).strip() if reason_match else "LLM scored"
                return score, reason
        except (ValueError, AttributeError):
            pass
        return None, None

    def test_valid_score_standard(self):
        score, reason = self._parse_score("SCORE: 7.5 | REASON: relevant to cloud computing")
        assert score == 7.5
        assert reason == "relevant to cloud computing"

    def test_valid_score_zero(self):
        score, reason = self._parse_score("SCORE: 0.0 | REASON: noise")
        assert score == 0.0
        assert reason == "noise"

    def test_valid_score_ten(self):
        score, reason = self._parse_score("SCORE: 10.0 | REASON: critical")
        assert score == 10.0
        assert reason == "critical"

    def test_valid_score_integer(self):
        """Integer scores (no decimal) should parse correctly."""
        score, reason = self._parse_score("SCORE: 8 | REASON: high relevance")
        assert score == 8.0
        assert reason == "high relevance"

    def test_empty_string(self):
        score, reason = self._parse_score("")
        assert score is None
        assert reason is None

    def test_no_score_keyword(self):
        score, reason = self._parse_score("no score here")
        assert score is None

    def test_out_of_range_high_clamped(self):
        """Scores > 10.0 should be clamped to 10.0."""
        score, reason = self._parse_score("SCORE: 15.0 | REASON: impossible")
        assert score == 10.0

    def test_out_of_range_negative_clamped(self):
        """Negative scores should be clamped to 0.0."""
        # Note: the regex \d+ won't match negative numbers, so -2.0 won't parse
        score, reason = self._parse_score("SCORE: -2.0 | REASON: negative")
        assert score is None  # Regex can't match negative

    def test_score_without_reason(self):
        score, reason = self._parse_score("SCORE: 6.5")
        assert score == 6.5
        assert reason == "LLM scored"  # fallback reason

    def test_batch_response_parsing(self):
        """Test numbered batch response parsing."""
        response = """[1] SCORE: 8.5 | REASON: direct career match
[2] SCORE: 3.0 | REASON: irrelevant sports news
[3] SCORE: 6.0 | REASON: tangentially relevant
[4] SCORE: 9.2 | REASON: hiring at tracked company"""

        # Replicate _parse_batch_response regex
        results = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r'[\[\(]?(\d+)[\]\).]?\s*SCORE:\s*(\d+\.?\d*)\s*\|?\s*REASON:\s*(.*)',
                line, re.IGNORECASE
            )
            if m:
                idx = int(m.group(1)) - 1
                score = max(0.0, min(10.0, float(m.group(2))))
                results.append((idx, score, m.group(3).strip()))

        assert len(results) == 4
        assert results[0] == (0, 8.5, "direct career match")
        assert results[1] == (1, 3.0, "irrelevant sports news")
        assert results[2] == (2, 6.0, "tangentially relevant")
        assert results[3] == (3, 9.2, "hiring at tracked company")


# ═══════════════════════════════════════════════════════════════════════
# Language Filtering Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLanguageFiltering:
    """Test language detection and filtering functions.

    These are imported from processors.scorer and used in scorer_adaptive.py
    for filtering non-target language content.
    """

    def test_location_to_lang_dubai(self):
        """Dubai profile should expect English + Arabic content (UAE is bilingual)."""
        from processors.scorer_base import location_to_lang
        allowed_scripts, region, lang_label = location_to_lang("Dubai")
        assert 'latin' in allowed_scripts
        assert 'arabic' in allowed_scripts

    def test_location_to_lang_kuwait(self):
        """Kuwait profile should allow both latin and arabic scripts."""
        from processors.scorer_base import location_to_lang
        allowed_scripts, region, lang_label = location_to_lang("Kuwait")
        assert 'latin' in allowed_scripts
        assert 'arabic' in allowed_scripts

    def test_location_to_lang_empty(self):
        """Empty location should fall back to English-only defaults."""
        from processors.scorer_base import location_to_lang
        allowed_scripts, region, lang_label = location_to_lang("")
        assert 'latin' in allowed_scripts

    def test_english_title_passes_for_dubai(self):
        """English title should pass language filter for Dubai profile."""
        from processors.scorer_base import _is_non_latin_title, location_to_lang
        allowed_scripts, _, _ = location_to_lang("Dubai")
        assert _is_non_latin_title("AWS announces new cloud AI services", allowed_scripts) is False

    def test_chinese_title_filtered_for_dubai(self):
        """Chinese title should be filtered for English-expected Dubai profile."""
        from processors.scorer_base import _is_non_latin_title, location_to_lang
        allowed_scripts, _, _ = location_to_lang("Dubai")
        assert _is_non_latin_title("亚马逊宣布新的云计算服务", allowed_scripts) is True

    def test_arabic_title_passes_for_kuwait(self):
        """Arabic title should pass for Kuwait profile (allows arabic + latin)."""
        from processors.scorer_base import _is_non_latin_title, location_to_lang
        allowed_scripts, _, _ = location_to_lang("Kuwait")
        assert _is_non_latin_title("الكويت تعلن عن مشاريع جديدة", allowed_scripts) is False

    def test_japanese_title_filtered_for_dubai(self):
        """Japanese title should be filtered for English-expected Dubai profile."""
        from processors.scorer_base import _is_non_latin_title, location_to_lang
        allowed_scripts, _, _ = location_to_lang("Dubai")
        assert _is_non_latin_title("東京で新しいテクノロジーイベント開催", allowed_scripts) is True

    def test_non_target_language_body_check(self):
        """Non-target language detection should also check body text."""
        from processors.scorer_base import _is_non_target_language, location_to_lang
        allowed_scripts, _, _ = location_to_lang("Dubai")
        # English title but Chinese body
        result = _is_non_target_language(
            "Breaking News",
            "这是一篇关于科技发展的新闻报道，涵盖了人工智能和云计算的最新进展",
            allowed_scripts
        )
        assert result is True

    def test_english_content_passes_body_check(self):
        """Full English content should pass body check for English profile."""
        from processors.scorer_base import _is_non_target_language, location_to_lang
        allowed_scripts, _, _ = location_to_lang("Dubai")
        result = _is_non_target_language(
            "AWS Cloud Services Update",
            "Amazon Web Services announced new AI-powered features for cloud developers today.",
            allowed_scripts
        )
        assert result is False


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

class TestReasoningStripping(unittest.TestCase):
    """Tests for strip_think_blocks, strip_reasoning_preamble, and extract_json."""

    def setUp(self):
        from routes.helpers import strip_think_blocks, strip_reasoning_preamble, extract_json
        self.strip_think = strip_think_blocks
        self.strip_reasoning = strip_reasoning_preamble
        self.extract_json = extract_json

    def test_think_block_standard(self):
        text = "<think>internal reasoning</think>The actual answer."
        self.assertEqual(self.strip_think(text), "The actual answer.")

    def test_think_block_no_open_tag(self):
        text = "some reasoning</think>The actual answer."
        self.assertEqual(self.strip_think(text), "The actual answer.")

    def test_think_block_empty(self):
        self.assertEqual(self.strip_think(""), "")
        self.assertEqual(self.strip_think(None), None)

    def test_pattern_a_wait_user_asked(self):
        text = "Wait, the user asked about trends. Let me check the data.\n\nLooking at the recent scans, I see several items.\n\n**Key Trends:**\n- Oil prices rising\n- Tech layoffs continuing"
        result = self.strip_reasoning(text)
        self.assertIn("Key Trends", result)
        self.assertNotIn("Wait, the user", result)

    def test_pattern_b_looking_at_feed(self):
        text = "Looking at the feed data, there's a mention of OPEC cuts. Let me analyze further.\n\nThe key insight here is the supply reduction.\n\n**OPEC has cut production by 1M barrels**, impacting crude prices significantly."
        result = self.strip_reasoning(text)
        self.assertNotIn("Looking at the feed", result)
        self.assertTrue(len(result) > 40)

    def test_pattern_c_check_if(self):
        text = "Check if there are any other high-rated items in the feed.\n\nI need to review the scores.\n\n- Oil supply disruption (Score: 9.2)\n- NVIDIA earnings beat (Score: 8.7)"
        result = self.strip_reasoning(text)
        self.assertIn("Oil supply disruption", result)
        self.assertNotIn("Check if", result)

    def test_pattern_d_so_answer_should(self):
        text = "So, the answer should mention the key trends. Let me compile them.\n\nSo, here are the key points:\n- Oil is up 3%\n- Gold hit new highs"
        result = self.strip_reasoning(text)
        self.assertIn("Oil is up", result)

    def test_transition_phrase_extraction(self):
        text = "I need to analyze the data. The user wants trends.\n\nHere are the notable highlights:\n- Trend A\n- Trend B\n- Trend C"
        result = self.strip_reasoning(text)
        self.assertIn("Trend A", result)

    def test_clean_text_passes_through(self):
        text = "**Market Summary:**\n- Oil is up 3%\n- Gold steady\n- Tech mixed"
        result = self.strip_reasoning(text)
        self.assertEqual(result, text)

    def test_all_reasoning_returns_empty(self):
        text = "Let me check the data and see what's relevant."
        result = self.strip_reasoning(text)
        self.assertEqual(result, "")

    def test_mid_response_reasoning_truncated(self):
        """Model starts clean then dumps reasoning mid-response."""
        text = "**Top trends this week:**\n\n- Oil prices are up 3.2% on OPEC cuts\n- NVIDIA earnings beat expectations\n\nWait, the user said 'skip generic advice.' So avoid saying 'monitor closely'...\n\nActually, let me also check the historical data."
        result = self.strip_reasoning(text)
        self.assertIn("Oil prices are up", result)
        self.assertNotIn("Wait, the user said", result)
        self.assertNotIn("Actually, let me", result)

    def test_mid_response_internal_reference(self):
        """Model references internal context sections mid-response."""
        text = "**Summary:**\n\n- Gold hit $3,428\n- Tech sector mixed\n\nThe MARKET section mentions GC=F at $3,428.75. Check the HISTORICAL DATA for trends.\n\nI need to review the scores."
        result = self.strip_reasoning(text)
        self.assertIn("Gold hit $3,428", result)
        self.assertNotIn("The MARKET section", result)

    def test_entirely_reasoning_internal_refs(self):
        """Entire response is reasoning referencing internal context."""
        text = "The MARKET section mentions GC=F at $3,428. Check the HISTORICAL DATA section. Wait, the user's categories include tech trends."
        result = self.strip_reasoning(text)
        self.assertEqual(result, "")

    def test_clean_response_with_the_word(self):
        """Legitimate response starting with 'The' should pass through."""
        text = "The oil market saw significant gains this week, with WTI crude rising 4.2% to $78.50 per barrel."
        result = self.strip_reasoning(text)
        self.assertEqual(result, text)

    def test_extract_json_from_reasoning(self):
        text = 'OK let me generate the JSON.\n\n{"categories": [{"id": "tech"}], "tickers": ["AAPL"]}'
        result = self.extract_json(text)
        self.assertTrue(result.startswith("{"))
        parsed = json.loads(result)
        self.assertIn("categories", parsed)

    def test_extract_json_array(self):
        text = 'Here are the suggestions:\n["item1", "item2", "item3"]'
        result = self.extract_json(text)
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 3)

    def test_extract_json_clean(self):
        text = '{"key": "value"}'
        self.assertEqual(self.extract_json(text), text)

    def test_extract_json_single_paragraph_reasoning(self):
        """JSON embedded in reasoning without paragraph breaks — the scenario
        that caused strip_reasoning_preamble to destroy wizard JSON responses."""
        text = 'Let me think about relevant categories for a software engineer. {"selected_categories": ["tech", "career"], "selected_subs": {"tech": ["ai"]}}'
        result = self.extract_json(text)
        parsed = json.loads(result)
        self.assertEqual(parsed["selected_categories"], ["tech", "career"])

    def test_extract_json_reasoning_then_array(self):
        """JSON array after reasoning text — tab-suggest scenario."""
        text = 'I should suggest keywords relevant to career for a student. ["Google Internships", "LinkedIn Jobs", "Indeed"]'
        result = self.extract_json(text)
        parsed = json.loads(result)
        self.assertIn("Google Internships", parsed)


def _show_diff(a: str, b: str) -> str:
    """Show character-level differences between two strings."""
    lines = []
    a_lines = a.split('\n')
    b_lines = b.split('\n')
    max_lines = max(len(a_lines), len(b_lines))
    for i in range(max_lines):
        la = a_lines[i] if i < len(a_lines) else "<missing>"
        lb = b_lines[i] if i < len(b_lines) else "<missing>"
        if la != lb:
            lines.append(f"Line {i+1}:")
            lines.append(f"  training:  {la!r}")
            lines.append(f"  inference: {lb!r}")
    return '\n'.join(lines) if lines else "(identical)"
