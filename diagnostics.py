"""Run isolated diagnostics for the RAG API deployment.

This script does not change the RAG pipeline. It checks each subsystem
separately and writes a JSON report for debugging.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "diagnostics_reports"
DEFAULT_QUESTION = "What is Evolutionary Algorithm?"


@dataclass
class CheckResult:
    name: str
    status: str
    duration_ms: int
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


def _duration_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _run_check(name: str, check: Callable[[], dict[str, Any]]) -> CheckResult:
    start = time.perf_counter()
    try:
        details = check()
        return CheckResult(
            name=name,
            status="ok",
            duration_ms=_duration_ms(start),
            details=details,
        )
    except Exception as exc:
        return CheckResult(
            name=name,
            status="failed",
            duration_ms=_duration_ms(start),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _url_json(url: str, timeout: int = 20) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        headers={"ngrok-skip-browser-warning": "true"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = body
            return response.status, parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body or exc.reason}") from exc


def check_python_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
    }


def check_required_imports() -> dict[str, Any]:
    modules = [
        "fastapi",
        "uvicorn",
        "psycopg",
        "openai",
        "sentence_transformers",
        "fitz",
        "tiktoken",
        "dotenv",
    ]
    imported: dict[str, str] = {}
    for module in modules:
        imported[module] = importlib.import_module(module).__name__
    return {"imports": imported}


def check_configuration() -> dict[str, Any]:
    import config

    return {
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.TOP_K,
        "score_threshold": config.SCORE_THRESHOLD,
        "database_target": "DATABASE_URL"
        if config.DATABASE_URL
        else f"{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}",
        "db_schema": config.DB_SCHEMA,
        "db_table": config.DB_TABLE,
    }


def check_database() -> dict[str, Any]:
    from ingestion.vectorstore.pgvector_store import _connection

    with _connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
    return {"postgres": "connected", "version": version}


def check_pgvector() -> dict[str, Any]:
    from ingestion.vectorstore.pgvector_store import _connection

    with _connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("pgvector extension is not installed")
    return {"pgvector": "available", "version": row[0]}


def check_vector_table() -> dict[str, Any]:
    from ingestion.vectorstore.pgvector_store import _connection, count_chunks

    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute AS a
            JOIN pg_class AS c ON c.oid = a.attrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relname = 'chunks'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
        row = cur.fetchone()
    return {
        "chunks": count_chunks(),
        "embedding_column_type": row[0] if row else None,
    }


def check_embedding_model() -> dict[str, Any]:
    from config import EMBEDDING_MODEL
    from ingestion.embeddings.embedding import embed_text, get_embedding_dimension

    dimension = get_embedding_dimension()
    sample = embed_text("diagnostic embedding probe")
    return {
        "model": EMBEDDING_MODEL,
        "dimension": dimension,
        "sample_vector_length": len(sample),
    }


def check_retriever(question: str) -> dict[str, Any]:
    from rag.retriever import retrieve, retrieve_context

    matches = retrieve(question, top_k=1)
    context = retrieve_context(question, top_k=1)
    return {
        "matches": len(matches),
        "top_similarity": matches[0].get("similarity") if matches else None,
        "context_chars": len(context),
    }


def check_prompt_builder(question: str) -> dict[str, Any]:
    from rag.prompts import build_rag_prompt
    from rag.retriever import retrieve_context

    context = retrieve_context(question, top_k=1)
    prompt = build_rag_prompt(question, context)
    return {
        "context_chars": len(context),
        "prompt_chars": len(prompt),
        "has_question": question in prompt,
    }


def check_ollama() -> dict[str, Any]:
    from config import LLM_MODEL, OLLAMA_BASE_URL

    tags_url = OLLAMA_BASE_URL.removesuffix("/v1") + "/api/tags"
    status, payload = _url_json(tags_url)
    models = [model.get("name", "") for model in payload.get("models", [])]
    has_model = any(name == LLM_MODEL or name.startswith(f"{LLM_MODEL}:") for name in models)
    if not has_model:
        raise RuntimeError(f"Ollama model '{LLM_MODEL}' was not found in {models}")
    return {"url": tags_url, "http_status": status, "model": LLM_MODEL, "models": models}


def check_rag_pipeline(question: str) -> dict[str, Any]:
    from rag.rag_pipeline import run_rag

    answer = run_rag(question)
    if not answer.strip():
        raise RuntimeError("RAG pipeline returned an empty answer")
    return {
        "answer_chars": len(answer),
        "answer_preview": answer[:300],
    }


def check_fastapi_import() -> dict[str, Any]:
    module = importlib.import_module("api.main")
    routes = sorted(route.path for route in module.app.routes)
    return {"routes": routes}


def check_fastapi_http(base_url: str, question: str) -> dict[str, Any]:
    root_status, root = _url_json(f"{base_url}/")
    health_status, health = _url_json(f"{base_url}/health", timeout=120)

    query_request = urllib.request.Request(
        f"{base_url}/v1/rag/query",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(query_request, timeout=240) as response:
        query_status = response.status
        query = json.loads(response.read().decode("utf-8"))

    return {
        "base_url": base_url,
        "root_status": root_status,
        "root": root,
        "health_status": health_status,
        "health": health,
        "query_status": query_status,
        "answer_chars": len(query.get("answer", "")),
        "answer_preview": query.get("answer", "")[:300],
    }


def build_checks(args: argparse.Namespace) -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    checks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("python_environment", check_python_environment),
        ("required_imports", check_required_imports),
        ("configuration", check_configuration),
        ("database", check_database),
        ("pgvector", check_pgvector),
        ("vector_table", check_vector_table),
        ("embedding_model", check_embedding_model),
        ("retriever", lambda: check_retriever(args.question)),
        ("prompt_builder", lambda: check_prompt_builder(args.question)),
        ("ollama", check_ollama),
        ("rag_pipeline", lambda: check_rag_pipeline(args.question)),
        ("fastapi_import", check_fastapi_import),
    ]
    if args.api_base_url:
        checks.append(
            ("fastapi_http", lambda: check_fastapi_http(args.api_base_url.rstrip("/"), args.question))
        )
    return checks


def write_report(results: list[CheckResult]) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"diagnostics_{timestamp}.json"
    failures = [result for result in results if result.status != "ok"]
    report = {
        "created_at": timestamp,
        "summary": {
            "total": len(results),
            "ok": len(results) - len(failures),
            "failed": len(failures),
        },
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG API diagnostics")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--api-base-url",
        help="Optional running FastAPI base URL, e.g. http://127.0.0.1:8000",
    )
    args = parser.parse_args()

    results = []
    for name, check in build_checks(args):
        print(f"[CHECK] {name}")
        result = _run_check(name, check)
        results.append(result)
        if result.status == "ok":
            print(f"[OK] {name} ({result.duration_ms} ms)")
        else:
            print(f"[FAILED] {name}: {result.error}")

    report_path = write_report(results)
    failures = [result for result in results if result.status != "ok"]
    print(f"\nReport written to: {report_path}")
    print(f"Passed: {len(results) - len(failures)} / {len(results)}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure.name}: {failure.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
