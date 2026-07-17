"""scripts/cli.py — standalone local-testing wrapper around the Analytics
Agent's LangGraph, for exercising it without the HTTP server running.

Reads the dataset from app.config.DATASET_PATH (the .env-configured
reference dataset) instead of an upload. Each query calls
run_analytics_graph() fresh — same as a real HTTP request — so, unlike the
old CLI's AnalyticsAgent (constructed once, reused across an interactive
session), conversation memory does NOT carry over between turns here. That
statefulness only makes sense for a long-lived process serving one
conversation at a time; a per-request FastAPI service, and this CLI, use
the same fresh-per-call code path deliberately for consistency, trading
away multi-turn memory in exchange for one code path instead of two.
"""

import argparse
import sys
from pathlib import Path

# Run as `python scripts/cli.py`, so the interpreter puts scripts/ (not the
# Analytics-Agent repo root) on sys.path — add the root so `app.*` resolves
# regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.analytics_agent.graph import run_analytics_graph
from app.config import DATASET_PATH


def _ask(query: str, ml_readiness: float, llm_readiness: float, file_content: bytes) -> None:
    print(f"\n{'=' * 70}")
    print(f"Query: {query}")
    print(f"{'=' * 70}\n")
    result = run_analytics_graph(
        file_content=file_content,
        business_question=query,
        ml_readiness_score=ml_readiness,
        llm_readiness_score=llm_readiness,
    )
    print(result["response"])


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Insurance Analytics Agent — standalone local test CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py --query "Show Gross Written Premium for FY2025"
  python scripts/cli.py --query "Why did underwriting result decline in Q2 2025?"
  python scripts/cli.py --interactive
        """,
    )
    parser.add_argument("--query", type=str, help="Business question to answer")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive mode")
    parser.add_argument(
        "--ml-readiness", type=float, default=99.75,
        help="ML readiness score to simulate (default: 99.75)"
    )
    parser.add_argument(
        "--llm-readiness", type=float, default=99.75,
        help="LLM readiness score to simulate — gates Groq narration vs. the "
             "deterministic template formatter (default: 99.75)"
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH} — set DATASET_PATH in .env.")
        return
    file_content = DATASET_PATH.read_bytes()

    if args.interactive:
        print("\nEnterprise Insurance Analytics Agent (standalone CLI)")
        print("   Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    print("Goodbye.")
                    break
                if not query:
                    continue
                _ask(query, args.ml_readiness, args.llm_readiness, file_content)
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
    elif args.query:
        _ask(args.query, args.ml_readiness, args.llm_readiness, file_content)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
