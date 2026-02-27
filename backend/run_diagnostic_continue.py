#!/usr/bin/env python3
"""Continue diagnostic loop for profiles 7-10."""

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("DIAGNOSTIC_CONT")

from run_diagnostic import (
    PROFILES, DIAGNOSTIC_DIR, create_profile_yaml,
    run_profile_scan, check_serper_credits,
    analyze_kuwait_contamination, analyze_adversarial_discrimination,
)

def main():
    logger.info("Continuing diagnostic for profiles 7-10")

    initial_credits = check_serper_credits()
    logger.info(f"Serper credits at start: {initial_credits}")

    # Load previous results
    results_path = DIAGNOSTIC_DIR / "all_results.json"
    if results_path.exists():
        with open(results_path) as f:
            prev_results_serialized = json.load(f)
        logger.info(f"Loaded previous results for profiles: {list(prev_results_serialized.keys())}")
    else:
        prev_results_serialized = {}

    # Initialize StratOS
    from main import StratOS
    strat = StratOS("config.yaml")

    all_results = {}

    # Load previous full outputs for analysis
    for pnum_str, data in prev_results_serialized.items():
        pnum = int(pnum_str)
        # Reload full articles from saved output
        output_path = DIAGNOSTIC_DIR / f"profile_{pnum}_output.json"
        if output_path.exists():
            with open(output_path) as f:
                full_output = json.load(f)
            data["articles"] = full_output.get("news", [])
        all_results[pnum] = data

    # Run remaining profiles
    for pnum in [7, 8, 9, 10]:
        credits = check_serper_credits()
        logger.info(f"\nSerper credits before profile {pnum}: {credits}")

        try:
            credits_int = int(credits) if str(credits).isdigit() else 9999
        except (ValueError, TypeError):
            credits_int = 9999

        if credits_int < 1600:
            logger.warning(f"Serper credits below 1600 ({credits}). Stopping.")
            break

        # Ensure profile YAML exists
        create_profile_yaml(pnum)

        result = run_profile_scan(pnum, strat)
        all_results[pnum] = result

        time.sleep(2)

    final_credits = check_serper_credits()
    logger.info(f"\nSerper credits at end: {final_credits}")

    # Save updated results
    serializable = {}
    for pnum, result in all_results.items():
        r = {k: v for k, v in result.items() if k != "articles"}
        serializable[str(pnum)] = r

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Run cross-profile analysis
    contamination = analyze_kuwait_contamination(all_results)
    shared_adversarial = analyze_adversarial_discrimination(all_results)

    analysis_path = DIAGNOSTIC_DIR / "cross_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "kuwait_contamination": {str(k): v for k, v in contamination.items()},
            "adversarial_shared_articles": shared_adversarial,
            "initial_credits": str(initial_credits),
            "final_credits": str(final_credits),
        }, f, indent=2)

    logger.info(f"\nAll {len(all_results)} profiles complete")
    logger.info(f"Credits: {initial_credits} -> {final_credits}")


if __name__ == "__main__":
    main()
