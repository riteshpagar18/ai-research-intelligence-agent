"""Run the labeled claim-verification benchmark against Gemini."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from agents.verifier import verify_claim

CASES_FILE = ROOT / "evaluation" / "cases.json"
LABELS = ("supported", "partially_supported", "unsupported", "conflicting")
VERIFIER_PROMPT_VERSION = "conflict-aware-v2"


def score_results(cases, predictions):
    """Compute class-level metrics and quote-match coverage."""
    confusion = {label: {other: 0 for other in (*LABELS, "error")} for label in LABELS}
    valid_quote_count = 0
    eligible_evidence_count = 0
    unsupported_evidence_count = 0

    for case, prediction in zip(cases, predictions):
        expected = case["expected_status"]
        predicted = prediction.get("status", "error")
        if predicted not in LABELS:
            predicted = "error"
        confusion[expected][predicted] += 1
        has_quote = bool(prediction.get("evidence"))
        if expected in {"supported", "partially_supported", "conflicting"}:
            eligible_evidence_count += 1
            valid_quote_count += int(has_quote)
        elif expected == "unsupported" and has_quote:
            unsupported_evidence_count += 1

    total = len(cases)
    correct = sum(confusion[label][label] for label in LABELS)
    f1_scores = []
    per_class = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in LABELS if other != label)
        false_negative = sum(confusion[label][other] for other in (*LABELS, "error") if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }

    observed_labels = [
        label for label in LABELS if sum(confusion[label].values()) > 0
    ]
    observed_macro_f1 = (
        sum(per_class[label]["f1"] for label in observed_labels) / len(observed_labels)
        if observed_labels else 0.0
    )

    return {
        "cases": total,
        "status_accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(observed_macro_f1, 4),
        "macro_f1_all_labels": round(sum(f1_scores) / len(f1_scores), 4),
        "labels_present": observed_labels,
        "valid_quote_coverage_on_evidence_cases": round(
            valid_quote_count / eligible_evidence_count, 4
        ) if eligible_evidence_count else 0.0,
        "unsupported_cases_with_quotes": unsupported_evidence_count,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Send one Gemini request per case. This uses your API quota.",
    )
    parser.add_argument(
        "--score-file", type=Path,
        help="Recalculate metrics from a saved evaluation JSON without API calls.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N cases.")
    parser.add_argument("--case-id", help="Run one named case, such as CASE-21.")
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds between Gemini requests (default: 6).")
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON results.")
    args = parser.parse_args()

    if args.live and args.score_file:
        parser.error("Use either --live or --score-file, not both.")

    load_dotenv(ROOT / ".env")
    if args.score_file:
        saved = json.loads(args.score_file.read_text())
        records = saved.get("results", [])
        cases = [{"expected_status": item["expected_status"]} for item in records]
        predictions = [item["prediction"] for item in records]
        saved["summary"] = score_results(cases, predictions)
        rendered = json.dumps(saved, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        return

    cases = json.loads(CASES_FILE.read_text())
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            parser.error(f"Unknown benchmark case: {args.case_id}")
    elif args.limit:
        cases = cases[:args.limit]
    if not args.live:
        counts = {}
        for case in cases:
            label = case["expected_status"]
            counts[label] = counts.get(label, 0) + 1
        print(json.dumps({
            "benchmark": str(CASES_FILE.relative_to(ROOT)),
            "cases": len(cases),
            "expected_status_counts": counts,
            "message": "No API requests sent. Add --live to run the Gemini benchmark.",
        }, indent=2))
        return

    if not os.getenv("GEMINI_API_KEY"):
        parser.error("GEMINI_API_KEY is missing; add it to .env before a live run.")

    predictions = []
    details = []
    for index, case in enumerate(cases):
        try:
            prediction = verify_claim(case["question"], case["claim"], case["sources"])
        except Exception as exc:  # Keep the run auditable when an API call fails.
            prediction = {"status": "error", "evidence": [], "error": str(exc)}
        predictions.append(prediction)
        details.append({
            "id": case["id"],
            "expected_status": case["expected_status"],
            "prediction": prediction,
        })
        if index < len(cases) - 1 and args.delay > 0:
            time.sleep(args.delay)

    output = {
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "summary": score_results(cases, predictions),
        "results": details,
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
