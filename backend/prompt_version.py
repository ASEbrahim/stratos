"""
Prompt Template Version Pinning
================================
Hashes the scoring prompt template at both training export and inference time.
Logs a WARNING if they drift apart — the #1 recurring failure mode in V2/V2.1.

Usage:
    from prompt_version import PROMPT_VERSION, check_prompt_alignment

    # At training export:
    check_prompt_alignment(system_prompt, user_prompt, context="export_training")

    # At inference:
    check_prompt_alignment(system_prompt, user_prompt, context="scorer_adaptive")
"""

import hashlib
import logging
import re

logger = logging.getLogger("PromptVersion")

# The canonical prompt template version.
# Update this ONLY when intentionally changing the prompt format.
# Format: "v{version}_{short_hash}"
PROMPT_VERSION = "v2.2_pending"  # Will be set after first alignment check


def _normalize_template(text: str) -> str:
    """Strip variable content to get the skeleton for hashing.

    Replaces dynamic values (role, location, article content, etc.) with
    placeholders so only the STRUCTURE is compared.
    """
    # Replace multi-line content blocks
    t = text

    # Replace profile fields
    t = re.sub(r'(?<=for a ).*?(?= in )', '{ROLE}', t)
    t = re.sub(r'(?<=in )[A-Z][\w, ]+(?=\.?\n)', '{LOCATION}', t)
    t = re.sub(r'(?<=User context: ).*', '{CONTEXT}', t)
    t = re.sub(r'(?<=Tracked companies: ).*', '{COMPANIES}', t)
    t = re.sub(r'(?<=Tracked institutions: ).*', '{INSTITUTIONS}', t)
    t = re.sub(r'(?<=Tracked interests: ).*', '{INTERESTS}', t)
    t = re.sub(r'(?<=Tracked industries: ).*', '{INDUSTRIES}', t)

    # Replace article fields
    t = re.sub(r'(?<=Category: ).*', '{CATEGORY}', t)
    t = re.sub(r'(?<=Keywords: ).*', '{KEYWORDS}', t)
    t = re.sub(r'(?<=Title: ).*', '{TITLE}', t)
    t = re.sub(r'(?<=Content: ).*', '{CONTENT}', t, flags=re.DOTALL)

    # Remove runtime-only lines (LANGUAGE, feedback)
    t = re.sub(r'LANGUAGE:.*\n?', '', t)
    # Remove feedback blocks (variable length, runtime-only)
    t = re.sub(r'Recent user feedback.*?(?=\n\n|\nScore each)', '', t, flags=re.DOTALL)

    # Normalize whitespace
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = t.strip()

    return t


def hash_template(system: str, user: str) -> str:
    """Compute a short hash of the prompt template skeleton."""
    normalized = _normalize_template(system + "\n---\n" + user)
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


# Store the first hash seen in this process for drift detection
_reference_hash = {}
_reference_context = {}


def check_prompt_alignment(system: str, user: str, context: str = "unknown") -> str:
    """Check if this prompt matches the canonical template.

    Call from both training export and inference. If they produce different
    hashes, log a WARNING.

    Args:
        system: System prompt text
        user: User prompt text
        context: Where this is being called from (e.g., "export_training", "scorer_adaptive")

    Returns:
        The template hash
    """
    h = hash_template(system, user)

    if "training" not in _reference_hash:
        if "training" in context or "export" in context or "prepare" in context:
            _reference_hash["training"] = h
            _reference_context["training"] = context
            logger.debug(f"Prompt template registered (training): {h} from {context}")

    if "inference" not in _reference_hash:
        if "scorer" in context or "inference" in context:
            _reference_hash["inference"] = h
            _reference_context["inference"] = context
            logger.debug(f"Prompt template registered (inference): {h} from {context}")

    # Check for drift between training and inference
    if "training" in _reference_hash and "inference" in _reference_hash:
        if _reference_hash["training"] != _reference_hash["inference"]:
            logger.warning(
                f"PROMPT TEMPLATE DRIFT DETECTED!\n"
                f"  Training hash:  {_reference_hash['training']} (from {_reference_context['training']})\n"
                f"  Inference hash: {_reference_hash['inference']} (from {_reference_context['inference']})\n"
                f"  This causes the model to see different prompts at inference vs training.\n"
                f"  Fix: align the templates in export_training.py and scorer_adaptive.py"
            )

    return h
