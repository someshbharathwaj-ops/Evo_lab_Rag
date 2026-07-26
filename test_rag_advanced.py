"""
Test suite for RAG advanced capabilities:
- Cross-Encoder Reranker (rag/reranker.py)
- LLM-as-a-Judge Answer Verification (rag/judge.py)
- Pipeline Orchestration & Feature Flags (rag/pipeline.py)
- FastAPI /query Endpoint Integration (main.py)
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from rag.judge import JudgeResult, _parse_judge_response, judge_answer
from rag.pipeline import NO_CONTEXT_ANSWER, PipelineResult, run_pipeline
from rag.reranker import rerank


class TestReranker(unittest.TestCase):
    def test_rerank_empty_chunks(self):
        result = rerank("test query", [], top_n=5, model_name="dummy-model")
        self.assertEqual(result, [])

    def test_rerank_missing_model_name_raises(self):
        chunks = [{"text": "sample text", "metadata": {}}]
        with self.assertRaises(ValueError):
            rerank("test query", chunks, top_n=5, model_name="")

    @patch("rag.reranker._get_cross_encoder")
    def test_rerank_sorting_order(self, mock_get_encoder):
        mock_model = MagicMock()
        # Suppose model predicts scores 0.2 for first chunk and 0.9 for second chunk
        mock_model.predict.return_value = MagicMock(tolist=lambda: [0.2, 0.9])
        mock_get_encoder.return_value = mock_model

        chunks = [
            {"chunk_id": "c1", "text": "Less relevant context"},
            {"chunk_id": "c2", "text": "Highly relevant context"},
        ]

        reranked = rerank("query", chunks, top_n=2, model_name="dummy/model")

        self.assertEqual(len(reranked), 2)
        # Should be sorted descending by rerank_score
        self.assertEqual(reranked[0]["chunk_id"], "c2")
        self.assertAlmostEqual(reranked[0]["rerank_score"], 0.9)
        self.assertEqual(reranked[1]["chunk_id"], "c1")
        self.assertAlmostEqual(reranked[1]["rerank_score"], 0.2)


class TestJudge(unittest.TestCase):
    def test_parse_valid_json(self):
        raw = json.dumps({
            "passed": True,
            "score": 0.92,
            "hallucination": False,
            "missing_information": [],
            "unsupported_claims": [],
            "feedback": "All facts grounded.",
            "revised_answer": ""
        })
        res = _parse_judge_response(raw)
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.92)
        self.assertFalse(res.hallucination)
        self.assertEqual(res.feedback, "All facts grounded.")

    def test_parse_json_in_markdown_fence(self):
        raw = """```json
{
  "passed": false,
  "score": 0.40,
  "hallucination": true,
  "missing_information": ["Missing definition of fitness"],
  "unsupported_claims": ["Evolution started in 1999"],
  "feedback": "Hallucination detected.",
  "revised_answer": "Fitness evaluates individuals."
}
```"""
        res = _parse_judge_response(raw)
        self.assertFalse(res.passed)
        self.assertEqual(res.score, 0.40)
        self.assertTrue(res.hallucination)
        self.assertIn("Missing definition of fitness", res.missing_information)
        self.assertIn("Evolution started in 1999", res.unsupported_claims)

    def test_parse_malformed_json_fallback(self):
        raw = "Not valid JSON response at all"
        res = _parse_judge_response(raw)
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.5)
        self.assertFalse(res.hallucination)
        self.assertIn("not valid JSON", res.feedback)

    @patch("rag.llm_client._get_client")
    def test_judge_answer_mock_llm(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "passed": True,
                "score": 0.95,
                "hallucination": False,
                "missing_information": [],
                "unsupported_claims": [],
                "feedback": "Grounded answer.",
                "revised_answer": ""
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = judge_answer("q", "ctx", "ans", judge_model="qwen3")
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.95)


class TestPipeline(unittest.TestCase):
    @patch("rag.pipeline.retrieve")
    @patch("rag.pipeline.call_llm")
    def test_pipeline_basic_flow(self, mock_call_llm, mock_retrieve):
        mock_retrieve.return_value = [
            {"text": "Sample context text.", "source": "doc1.pdf", "page": 2}
        ]
        mock_call_llm.return_value = "Generated answer text."

        res = run_pipeline("What is this?")
        self.assertIsInstance(res, PipelineResult)
        self.assertEqual(res.answer, "Generated answer text.")
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["source"], "doc1.pdf")
        self.assertEqual(res.sources[0]["page"], 2)

    @patch("rag.pipeline.retrieve")
    def test_pipeline_no_chunks(self, mock_retrieve):
        mock_retrieve.return_value = []
        res = run_pipeline("Unknown query")
        self.assertEqual(res.answer, NO_CONTEXT_ANSWER)
        self.assertEqual(res.sources, [])

    @patch("rag.pipeline.ENABLE_JUDGE", True)
    @patch("rag.pipeline.MAX_REGENERATE_ATTEMPTS", 2)
    @patch("rag.pipeline.retrieve")
    @patch("rag.pipeline.call_llm")
    @patch("rag.judge.judge_answer")
    def test_pipeline_judge_rejection_and_retry(self, mock_judge, mock_call_llm, mock_retrieve):
        mock_retrieve.return_value = [{"text": "Fact A", "source": "file.pdf", "page": 1}]
        mock_call_llm.side_effect = ["Initial wrong answer", "Corrected answer"]

        # First judge fails, second judge passes
        mock_judge.side_effect = [
            JudgeResult(passed=False, score=0.3, hallucination=True, feedback="Hallucinated fact B"),
            JudgeResult(passed=True, score=0.9, hallucination=False, feedback="Grounded now"),
        ]

        res = run_pipeline("Query")
        self.assertEqual(res.answer, "Corrected answer")
        self.assertEqual(res.attempts, 2)
        self.assertTrue(res.judge_used)
        self.assertTrue(res.judge_passed)


if __name__ == "__main__":
    unittest.main()
