"""Run one live end-to-end Gemini + Tavily research check."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from workflow.pipeline import run_research_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="What are common benefits and limitations of residential solar panels?",
    )
    args = parser.parse_args()

    missing = [key for key in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)]
    if missing:
        parser.error("Missing required settings: " + ", ".join(missing))

    progress = []
    result = run_research_agent(args.question, progress_callback=progress.append)
    if not result.get("plan") or not result.get("sources") or not result.get("report"):
        raise SystemExit("Live smoke check failed: incomplete research result.")
    if any(not source.get("url") or not source.get("content") for source in result["sources"]):
        raise SystemExit("Live smoke check failed: a source is missing its URL or text.")

    print(json.dumps({
        "status": "passed",
        "progress_steps": progress,
        "research_tasks": len(result["plan"]),
        "unique_sources": len(result["sources"]),
        "verified_claims": len(result["verification"].get("claims", [])),
        "report_characters": len(result["report"]),
    }, indent=2))


if __name__ == "__main__":
    main()
