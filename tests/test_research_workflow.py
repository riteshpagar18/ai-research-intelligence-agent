import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.genai.errors import APIError
from streamlit.testing.v1 import AppTest
from agents import verifier
from agents import gemini
from config import load_streamlit_secrets
from scripts.evaluate import score_results
from tools.pdf_export import create_report_pdf
from workflow import pipeline


class ResearchWorkflowTests(unittest.TestCase):
    def test_streamlit_host_secrets_fill_missing_environment_values(self):
        environ = {"GEMINI_MODEL": "keep-this-model"}
        streamlit_module = SimpleNamespace(secrets={
            "GEMINI_API_KEY": "host-gemini-key",
            "GEMINI_MODEL": "cloud-model",
            "TAVILY_API_KEY": "host-tavily-key",
        })
        load_streamlit_secrets(streamlit_module, environ)
        self.assertEqual(environ["GEMINI_API_KEY"], "host-gemini-key")
        self.assertEqual(environ["TAVILY_API_KEY"], "host-tavily-key")
        self.assertEqual(environ["GEMINI_MODEL"], "keep-this-model")

    def test_missing_streamlit_secrets_are_ok_for_local_runs(self):
        class NoSecrets:
            @property
            def secrets(self):
                raise FileNotFoundError("No local Streamlit secrets file")

        environ = {}
        load_streamlit_secrets(NoSecrets(), environ)
        self.assertEqual(environ, {})

    def test_streamlit_app_renders_without_errors(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "🔍 AI Research & Intelligence Agent")

    def test_verifier_keeps_only_exact_source_quotes(self):
        source = {
            "source_id": "S1",
            "title": "Example",
            "url": "https://example.com",
            "content": "The study surveyed 120 participants in 2025.",
        }
        payload = {
            "claims": [
                {
                    "claim": "The study surveyed 120 participants.",
                    "status": "supported",
                    "source_ids": ["S1"],
                    "evidence": [{
                        "source_id": "S1",
                        "quote": "The study surveyed 120 participants in 2025.",
                    }],
                },
                {
                    "claim": "The study proved a treatment was effective.",
                    "status": "supported",
                    "source_ids": ["S1"],
                    "evidence": [{"source_id": "S1", "quote": "Invented quote."}],
                },
            ],
            "limitations": [],
        }
        with patch.object(verifier, "generate_text", return_value=json.dumps(payload)):
            result = verifier.verify_sources("Question", [source])
        self.assertEqual(result["claims"][0]["status"], "supported")
        self.assertEqual(result["claims"][1]["status"], "unsupported")
        self.assertEqual(result["claims"][1]["evidence"], [])

    def test_malformed_verifier_json_fails_closed(self):
        with patch.object(verifier, "generate_text", return_value="not json"):
            result = verifier.verify_sources("Question", [{
                "title": "Example", "url": "https://example.com", "content": "Text."
            }])
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["raw_assessment"], "not json")

    def test_benchmark_claim_check_returns_validated_status_and_quote(self):
        source = {
            "source_id": "S1", "title": "Synthetic", "url": "https://example.invalid/source",
            "content": "The fictional sample contained 12 records.",
        }
        response = json.dumps({
            "status": "supported",
            "evidence": [{"source_id": "S1", "quote": "The fictional sample contained 12 records."}],
        })
        with patch.object(verifier, "generate_text", return_value=response):
            result = verifier.verify_claim("Synthetic question", "It contained 12 records.", [source])
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["source_ids"], ["S1"])
        self.assertEqual(len(result["evidence"]), 1)

    def test_conflicting_claim_keeps_quotes_from_both_sources(self):
        sources = [
            {"source_id": "S1", "title": "First", "url": "https://example.invalid/1",
             "content": "The fictional system capacity is 10 terabytes."},
            {"source_id": "S2", "title": "Second", "url": "https://example.invalid/2",
             "content": "The revised fictional system capacity is 12 terabytes."},
        ]
        response = json.dumps({
            "status": "conflicting",
            "evidence": [
                {"source_id": "S1", "quote": "The fictional system capacity is 10 terabytes."},
                {"source_id": "S2", "quote": "The revised fictional system capacity is 12 terabytes."},
            ],
        })
        with patch.object(verifier, "generate_text", return_value=response):
            result = verifier.verify_claim("Capacity?", "It is 10 terabytes.", sources)
        self.assertEqual(result["status"], "conflicting")
        self.assertEqual(result["source_ids"], ["S1", "S2"])

    def test_retryable_gemini_rate_limit_is_retried(self):
        class Models:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise APIError(429, {"message": "rate limited"})
                return type("Response", (), {"text": "ready"})()

        models = Models()
        client = type("Client", (), {"models": models})()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch.object(gemini.genai, "Client", return_value=client), \
             patch.object(gemini.time, "sleep"):
            self.assertEqual(gemini.generate_text("hello"), "ready")
        self.assertEqual(models.calls, 2)

    def test_langgraph_runs_all_research_steps(self):
        source = {"title": "Example", "url": "https://example.com", "content": "Evidence."}
        with patch.object(pipeline, "create_research_plan", return_value=["task", "task two"]), \
             patch.object(pipeline, "search_web", return_value=[source]), \
             patch.object(pipeline, "verify_sources", return_value={"claims": [], "limitations": []}), \
             patch.object(pipeline, "generate_report", return_value="# Report"):
            result = pipeline.run_research_agent("A detailed test question")
        self.assertEqual(result["plan"], ["task", "task two"])
        self.assertEqual(result["sources"][0]["source_id"], "S1")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["report"], "# Report")

    def test_empty_search_results_raise_clear_error(self):
        with patch.object(pipeline, "create_research_plan", return_value=["task"]), \
             patch.object(pipeline, "search_web", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No sources found"):
                pipeline.run_research_agent("A detailed test question")

    def test_pdf_export_returns_pdf_bytes_with_sources(self):
        pdf = create_report_pdf(
            "# Findings\nA supported result [S1].",
            [{"source_id": "S1", "title": "Example", "url": "https://example.com"}],
        )
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_evaluation_dataset_has_25_labeled_synthetic_cases(self):
        cases = json.loads((Path(__file__).parents[1] / "evaluation" / "cases.json").read_text())
        self.assertEqual(len(cases), 25)
        self.assertTrue(all(case["synthetic"] for case in cases))
        self.assertEqual({case["expected_status"] for case in cases}, {
            "supported", "partially_supported", "unsupported", "conflicting"
        })

    def test_evaluation_metrics_calculate_classification_and_quote_coverage(self):
        cases = [
            {"expected_status": "supported"},
            {"expected_status": "unsupported"},
        ]
        predictions = [
            {"status": "supported", "evidence": [{"quote": "exact"}]},
            {"status": "supported", "evidence": []},
        ]
        metrics = score_results(cases, predictions)
        self.assertEqual(metrics["status_accuracy"], 0.5)
        self.assertEqual(metrics["valid_quote_coverage_on_evidence_cases"], 1.0)
        self.assertEqual(metrics["unsupported_cases_with_quotes"], 0)

    def test_macro_f1_subset_uses_labels_present_in_sample(self):
        cases = [{"expected_status": "supported"} for _ in range(3)]
        predictions = [
            {"status": "supported", "evidence": [{"quote": "matched"}]}
            for _ in cases
        ]
        metrics = score_results(cases, predictions)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1_all_labels"], 0.25)
        self.assertEqual(metrics["labels_present"], ["supported"])

    def test_saved_evaluation_can_be_rescored_without_api_calls(self):
        root = Path(__file__).parents[1]
        saved = {
            "results": [
                {
                    "id": f"CASE-{index}",
                    "expected_status": "supported",
                    "prediction": {"status": "supported", "evidence": [{"quote": "matched"}]},
                }
                for index in range(3)
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "evaluation.json"
            file_path.write_text(json.dumps(saved))
            completed = subprocess.run(
                [sys.executable, "scripts/evaluate.py", "--score-file", str(file_path)],
                cwd=root, check=True, capture_output=True, text=True,
            )
        summary = json.loads(completed.stdout)["summary"]
        self.assertEqual(summary["status_accuracy"], 1.0)
        self.assertEqual(summary["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
