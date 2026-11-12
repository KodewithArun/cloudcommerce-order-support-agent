"""CLI runner executing conversation evaluation suite with rich formatted reports."""

from __future__ import annotations

import argparse
import sys

from app.eval.runner import ConversationEvaluator
from app.eval.scenarios import SCENARIOS

# ANSI color codes for clean terminal output
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_report(results: list, verbose: bool = False) -> int:
    """Print an ASCII summary table of the evaluation results."""
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    total_time_ms = sum(r.total_duration_ms for r in results)
    total_tokens = sum(r.total_tokens for r in results)

    print()
    print("=" * 86)
    print(f"{BOLD}                CloudCommerce Conversation Evaluation Suite{RESET}")
    print("=" * 86)
    print(
        f"{'#':<3} | {'Scenario Title':<38} | {'Turns':<5} | {'Refusal?':<8} | {'Result':<6} | {'Time (ms)':<9} | {'Tokens':<6}"
    )
    print("-" * 86)

    for idx, r in enumerate(results, start=1):
        has_refusal = any(t.is_refusal for t in r.scenario.turns)
        refusal_str = "YES" if has_refusal else "NO"
        status_str = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
        print(
            f"{idx:<3} | {r.scenario.title:<38} | {len(r.scenario.turns):<5} | {refusal_str:<8} | {status_str:<15} | {r.total_duration_ms:<9.1f} | {r.total_tokens:<6}"
        )

        if verbose or not r.passed:
            for turn_res in r.turn_results:
                mark = f"{GREEN}✓{RESET}" if turn_res.passed else f"{RED}✗{RESET}"
                print(f"      {mark} Turn {turn_res.turn_index}: \"{turn_res.user_message}\"")
                print(f"        -> \"{turn_res.agent_response}\"")
                if turn_res.failure_reasons:
                    for reason in turn_res.failure_reasons:
                        print(f"        {RED}Error: {reason}{RESET}")

    print("-" * 86)
    pass_rate = (passed_count / total * 100) if total > 0 else 0.0
    summary_color = GREEN if passed_count == total else RED
    print(
        f"{BOLD}Summary:{RESET} {summary_color}{passed_count}/{total} Passed ({pass_rate:.1f}%){RESET} | "
        f"Failed: {failed_count} | Total Time: {total_time_ms:.1f}ms | Total Tokens: {total_tokens}"
    )
    print("=" * 86)
    print()

    return 0 if failed_count == 0 else 1


def main() -> None:
    """Command-line entry point for cloudcommerce-eval."""
    parser = argparse.ArgumentParser(
        description="Run the CloudCommerce conversation evaluation suite."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against the live configured LLM provider instead of deterministic mock.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print turn-by-turn user messages and agent responses.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Filter to run a specific scenario by ID.",
    )

    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.scenario_id == args.scenario]
        if not scenarios:
            print(f"Error: No scenario found matching ID '{args.scenario}'")
            sys.exit(1)

    evaluator = ConversationEvaluator(live=args.live)
    results = evaluator.run_all(scenarios)
    exit_code = print_report(results, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

