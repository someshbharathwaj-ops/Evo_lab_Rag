"""Fast unit tests for the RAG orchestration (no model, database, or Ollama needed)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import importlib


class LoaderTests(unittest.TestCase):
    def test_clean_text_removes_pdf_control_bytes(self):
        from ingestion.loaders.loaders import clean_text

        cleaned = clean_text("Alpha\x00\x01\n\t Beta\x7f  Gamma")
        self.assertEqual(cleaned, "Alpha Beta Gamma")
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x01", cleaned)
        self.assertNotIn("\x7f", cleaned)


class PromptTests(unittest.TestCase):
    def test_context_is_labelled_and_deduplicated(self):
        from rag.prompts import build_context

        chunk = {
            "text": "Grounded fact.",
            "metadata": {"source": "/tmp/paper.pdf", "page": 2},
        }
        context = build_context([chunk, chunk])
        self.assertEqual(context.count("Grounded fact."), 1)
        self.assertIn("paper.pdf, page 2", context)

    def test_prompt_contains_question_and_context(self):
        from rag.prompts import build_rag_prompt

        prompt = build_rag_prompt("Why?", "Because.")
        self.assertIn("Why?", prompt)
        self.assertIn("Because.", prompt)
        self.assertIn("Formatting Rules", prompt)


class SplitterTests(unittest.TestCase):
    def test_chunk_ids_are_deterministic(self):
        from ingestion.splitters import splitters

        class Encoding:
            @staticmethod
            def encode(text):
                return [ord(character) for character in text]

            @staticmethod
            def decode(tokens):
                return "".join(chr(token) for token in tokens)

        tokenizer = MagicMock()
        tokenizer.encoding_for_model.return_value = Encoding()

        documents = [{"text": "one two three " * 30, "metadata": {"source": "x.pdf", "page": 1}}]
        with patch.object(splitters, "tiktoken", tokenizer):
            first = splitters.token_based_splitter(documents, chunk_size=20, chunk_overlap=5)
            second = splitters.token_based_splitter(documents, chunk_size=20, chunk_overlap=5)
        self.assertEqual(
            [chunk["chunk_id"] for chunk in first],
            [chunk["chunk_id"] for chunk in second],
        )


class PipelineTests(unittest.TestCase):
    def test_ingestion_embeds_and_uploads(self):
        pipeline = importlib.import_module("ingestion.pipeline")
        chunk = {"chunk_id": "id", "text": "text", "metadata": {}}
        with (
            patch.object(pipeline.os.path, "isfile", return_value=True),
            patch.object(pipeline, "load_pdf", return_value=[{"text": "text", "metadata": {}}]),
            patch.object(pipeline, "token_based_splitter", return_value=[chunk]),
            patch.object(pipeline, "embed_texts", return_value=[[0.1, 0.2]]) as embed,
            patch.object(pipeline, "insert_chunks", return_value=1) as insert,
            patch.object(pipeline, "_write_debug_chunks") as write,
        ):
            result = pipeline.run_ingestion("paper.pdf")
            embed.assert_called_once_with(["text"])
            self.assertEqual(insert.call_args.args[0][0]["embedding"], [0.1, 0.2])
            write.assert_called_once()
            self.assertNotIn("embedding", result[0])


class RagTests(unittest.TestCase):
    def test_rag_returns_generated_answer_only(self):
        pipeline = importlib.import_module("rag.rag_pipeline")
        with (
            patch.object(pipeline, "retrieve_context", return_value="private context") as retrieve,
            patch.object(pipeline, "call_llm", return_value="generated answer") as call_llm,
        ):
            self.assertEqual(pipeline.run_rag("question"), "generated answer")
            retrieve.assert_called_once()
            self.assertIn("private context", call_llm.call_args.args[0])

    def test_no_context_has_safe_answer(self):
        pipeline = importlib.import_module("rag.rag_pipeline")
        with patch.object(pipeline, "retrieve_context", return_value=""):
            self.assertEqual(pipeline.run_rag("question"), pipeline.NO_CONTEXT_ANSWER)


class OllamaClientTests(unittest.TestCase):
    def test_client_uses_ollama_credentials(self):
        from config import OLLAMA_BASE_URL

        llm_client = importlib.import_module("rag.llm_client")
        llm_client._get_client.cache_clear()
        with patch.object(llm_client, "openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            llm_client._get_client()
            
            expected_kwargs = {
                "base_url": OLLAMA_BASE_URL,
                "api_key": "ollama"
            }
            if "ngrok" in OLLAMA_BASE_URL:
                expected_kwargs["default_headers"] = {"ngrok-skip-browser-warning": "true"}
                expected_kwargs["timeout"] = 120.0
                expected_kwargs["max_retries"] = 2
                
            mock_openai.OpenAI.assert_called_once_with(**expected_kwargs)

    def test_clean_html_math_converts_subscripts_and_superscripts(self):
        from rag.llm_client import clean_html_math

        self.assertEqual(
            clean_html_math("where x<sub>i</sub> represents"),
            "where $x_{i}$ represents"
        )
        self.assertEqual(
            clean_html_math("where p<sub>i</sub> is the"),
            "where $p_{i}$ is the"
        )
        self.assertEqual(
            clean_html_math("f(x<sub>i</sub>) is the"),
            "$f(x_{i})$ is the"
        )
        self.assertEqual(
            clean_html_math("x<sup>2</sup> + y<sup>2</sup> = z<sup>2</sup>"),
            "$x^{2}$ + $y^{2}$ = $z^{2}$"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
