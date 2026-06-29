"""
============================================================
  Component Test Suite for Evo_lab_Rag
============================================================
Tests every component in isolation without modifying source files.
Run from the project root:
    python test_all_components.py
============================================================
"""
import os
import sys
import time
import traceback
from pathlib import Path

# -- Setup sys.path so all imports resolve correctly ----------
PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_OF_PROJECT = PROJECT_ROOT.parent

# Add both the project root (for `config`, `ingestion.*`, `rag.*`)
# and the parent (for `Evo_lab_Rag.config` used in pgvector_store.py)
for p in [str(PROJECT_ROOT), str(PARENT_OF_PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Test tracking --------------------------------------------
results = []
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
WARN = "[WARN]"


def run_test(name, func):
    """Run a test function and record the result."""
    print(f"\n{'-' * 60}")
    print(f"  TEST: {name}")
    print(f"{'-' * 60}")
    try:
        status, detail = func()
        results.append((name, status, detail))
        print(f"  Result: {status}")
        if detail:
            print(f"  Detail: {detail}")
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  Result: {FAIL}")
        print(f"  Error:  {e}")
        traceback.print_exc()


# ===========================================================
#  1. CONFIG
# ===========================================================
def test_config():
    from config import (
        LLM_MODEL, MAX_TOKENS, TEMPERATURE,
        EMBEDDING_MODEL, EMBEDDING_DIMENSION,
        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    )

    print(f"    LLM_MODEL         = {LLM_MODEL}")
    print(f"    MAX_TOKENS        = {MAX_TOKENS}")
    print(f"    TEMPERATURE       = {TEMPERATURE}")
    print(f"    EMBEDDING_MODEL   = {EMBEDDING_MODEL}")
    print(f"    EMBEDDING_DIMENSION = {EMBEDDING_DIMENSION}")
    print(f"    CHUNK_SIZE        = {CHUNK_SIZE}")
    print(f"    CHUNK_OVERLAP     = {CHUNK_OVERLAP}")
    print(f"    TOP_K             = {TOP_K}")
    print(f"    DB_HOST           = {DB_HOST}")
    print(f"    DB_PORT           = {DB_PORT}")
    print(f"    DB_NAME           = {DB_NAME}")
    print(f"    DB_USER           = {DB_USER}")
    print(f"    DB_PASSWORD       = {'*' * len(DB_PASSWORD) if DB_PASSWORD else '(empty)'}")

    # Validate types
    assert isinstance(MAX_TOKENS, int), f"MAX_TOKENS should be int, got {type(MAX_TOKENS)}"
    assert isinstance(TEMPERATURE, float), f"TEMPERATURE should be float, got {type(TEMPERATURE)}"
    assert isinstance(EMBEDDING_DIMENSION, int), f"EMBEDDING_DIMENSION should be int"
    assert isinstance(CHUNK_SIZE, int), f"CHUNK_SIZE should be int"
    assert isinstance(TOP_K, int), f"TOP_K should be int"

    # Check that .env values are loaded (not just defaults)
    if DB_HOST == "localhost":
        return WARN, "DB_HOST is 'localhost' -- .env may not be loaded"

    return PASS, f"All {14} config values loaded successfully"


# ===========================================================
#  2. LOADERS -- clean_text
# ===========================================================
def test_loader_clean_text():
    from ingestion.loaders.loaders import clean_text

    raw = "Hello\n  world\t\tthis   is\r\n  messy    text"
    cleaned = clean_text(raw)

    print(f"    Input:  {repr(raw)}")
    print(f"    Output: {repr(cleaned)}")

    assert "\n" not in cleaned, "Newlines should be removed"
    assert "\t" not in cleaned, "Tabs should be removed"
    assert "  " not in cleaned, "Multiple spaces should be collapsed"
    assert cleaned == "Hello world this is messy text"

    return PASS, "Text cleaning works correctly"


# ===========================================================
#  3. LOADERS -- load_pdf
# ===========================================================
def test_loader_pdf():
    from ingestion.loaders.loaders import load_pdf

    # Look for any PDF in data/raw/
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return SKIP, f"Directory not found: {raw_dir}"

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        return SKIP, "No PDF files found in data/raw/"

    pdf_path = str(pdf_files[0])
    print(f"    Loading: {pdf_path}")

    docs = load_pdf(pdf_path)
    print(f"    Pages loaded: {len(docs)}")

    assert isinstance(docs, list), "Should return a list"
    assert len(docs) > 0, "Should load at least one page"

    first = docs[0]
    assert "text" in first, "Each doc should have 'text' key"
    assert "metadata" in first, "Each doc should have 'metadata' key"
    assert "source" in first["metadata"], "Metadata should have 'source'"
    assert "page" in first["metadata"], "Metadata should have 'page'"
    print(f"    First page preview: {first['text'][:100]}...")

    return PASS, f"Loaded {len(docs)} pages from {Path(pdf_path).name}"


# ===========================================================
#  4. SPLITTERS -- token_based_splitter
# ===========================================================
def test_splitter():
    try:
        import tiktoken
    except ImportError:
        return SKIP, "tiktoken not installed"

    from ingestion.splitters.splitters import token_based_splitter

    # Create a fake document with enough text to produce multiple chunks
    fake_text = "Artificial intelligence is transforming the world. " * 200
    documents = [{
        "text": fake_text,
        "metadata": {"source": "test.pdf", "page": 1}
    }]

    chunks = token_based_splitter(documents, chunk_size=100, chunk_overlap=20)
    print(f"    Input text tokens: ~{len(fake_text.split())}")
    print(f"    Chunks created: {len(chunks)}")

    assert isinstance(chunks, list), "Should return a list"
    assert len(chunks) > 1, f"Should create multiple chunks, got {len(chunks)}"

    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk, "Chunk should have 'chunk_id'"
    assert "text" in first_chunk, "Chunk should have 'text'"
    assert "metadata" in first_chunk, "Chunk should have 'metadata'"
    assert "token_start" in first_chunk["metadata"], "Metadata should have 'token_start'"
    assert "token_end" in first_chunk["metadata"], "Metadata should have 'token_end'"
    assert first_chunk["metadata"]["source"] == "test.pdf", "Source metadata should be preserved"

    # Verify chunk_ids are unique
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids should be unique"

    print(f"    First chunk preview: {first_chunk['text'][:80]}...")

    return PASS, f"Created {len(chunks)} chunks with correct structure"


# ===========================================================
#  5. EMBEDDINGS -- embed_text & embed_texts
# ===========================================================
def test_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return SKIP, "sentence-transformers not installed"

    from ingestion.embeddings.embedding import embed_text, embed_texts
    from config import EMBEDDING_DIMENSION

    print("    Loading model (may take a moment on first run)...")
    t0 = time.time()

    # Single text
    vec = embed_text("What is machine learning?")
    elapsed = time.time() - t0
    print(f"    Single embed time: {elapsed:.2f}s")
    print(f"    Vector dimension:  {len(vec)}")
    print(f"    Expected dimension: {EMBEDDING_DIMENSION}")
    print(f"    First 5 values:    {vec[:5]}")

    assert isinstance(vec, list), "Should return a list"
    assert len(vec) == EMBEDDING_DIMENSION, f"Dimension mismatch: {len(vec)} != {EMBEDDING_DIMENSION}"
    assert all(isinstance(v, float) for v in vec), "All values should be floats"

    # Batch texts
    t0 = time.time()
    vecs = embed_texts(["Hello world", "Machine learning is great"])
    elapsed = time.time() - t0
    print(f"    Batch embed time:  {elapsed:.2f}s")
    print(f"    Batch size:        {len(vecs)}")

    assert isinstance(vecs, list), "Should return a list of lists"
    assert len(vecs) == 2, "Should return 2 embeddings"
    assert len(vecs[0]) == EMBEDDING_DIMENSION, "Each embedding should match EMBEDDING_DIMENSION"

    return PASS, f"Embeddings work -- dimension={len(vec)}"


# ===========================================================
#  6. DATABASE CONNECTION (pgvector_store)
# ===========================================================
def test_db_connection():
    try:
        import psycopg
    except ImportError:
        return SKIP, "psycopg not installed"

    from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

    print(f"    Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}...")

    try:
        conn = psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            autocommit=True,
            connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"    PostgreSQL: {version[:60]}...")

            # Check pgvector extension
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            if row:
                print(f"    pgvector extension: installed")
            else:
                print(f"    pgvector extension: NOT installed (CREATE EXTENSION vector needed)")

            # Check if chunks table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'chunks'
                )
            """)
            table_exists = cur.fetchone()[0]
            print(f"    'chunks' table: {'exists' if table_exists else 'does not exist'}")

            if table_exists:
                cur.execute("SELECT COUNT(*) FROM chunks")
                count = cur.fetchone()[0]
                print(f"    Rows in chunks: {count}")

        conn.close()
        return PASS, f"Connected to {DB_HOST}/{DB_NAME}"

    except psycopg.OperationalError as e:
        return FAIL, f"Connection failed: {e}"


# ===========================================================
#  7. PGVECTOR STORE -- import check
# ===========================================================
def test_pgvector_store_import():
    """Test that pgvector_store.py can be imported without errors."""
    try:
        from ingestion.vectorstore.pgvector_store import (
            create_table, insert_chunks, similarity_search
        )
        print(f"    create_table:      {create_table}")
        print(f"    insert_chunks:     {insert_chunks}")
        print(f"    similarity_search: {similarity_search}")
        return PASS, "All pgvector_store functions imported"
    except ImportError as e:
        # Check the specific issue
        err_msg = str(e)
        if "Evo_lab_Rag.config" in err_msg:
            return FAIL, (
                f"{e}\n"
                "    -> Fix: In pgvector_store.py, change:\n"
                "        from Evo_lab_Rag.config import ...\n"
                "      to:\n"
                "        from config import ..."
            )
        return FAIL, str(e)


# ===========================================================
#  8. RETRIEVER -- import check
# ===========================================================
def test_retriever_import():
    try:
        from rag.retriever import retrieve
        print(f"    retrieve: {retrieve}")

        import inspect
        sig = inspect.signature(retrieve)
        print(f"    Signature: retrieve{sig}")

        return PASS, "Retriever imported successfully"
    except ImportError as e:
        return FAIL, str(e)


# ===========================================================
#  9. LLM CLIENT -- import & config check
# ===========================================================
def test_llm_client_import():
    try:
        from openai import OpenAI
    except ImportError:
        return SKIP, "openai not installed (pip install openai)"

    try:
        from rag.llm_client import call_llm
        print(f"    call_llm: {call_llm}")

        from config import LLM_MODEL, OPENROUTER_API_KEY
        print(f"    Model: {LLM_MODEL}")
        print(f"    API Key: {'set (' + OPENROUTER_API_KEY[:12] + '...)' if OPENROUTER_API_KEY else 'NOT SET'}")

        if not OPENROUTER_API_KEY:
            return WARN, "Imported OK, but OPENROUTER_API_KEY is empty in .env"

        return PASS, f"LLM client imported -- model={LLM_MODEL}"
    except ImportError as e:
        return FAIL, str(e)


# ===========================================================
#  10. LLM CLIENT -- live call (OpenRouter API)
# ===========================================================
def test_llm_live_call():
    try:
        from openai import OpenAI
    except ImportError:
        return SKIP, "openai not installed"

    from config import OPENROUTER_API_KEY, LLM_MODEL
    if not OPENROUTER_API_KEY:
        return SKIP, "OPENROUTER_API_KEY not set in .env"

    print(f"    Calling OpenRouter with model: {LLM_MODEL}...")
    try:
        from rag.llm_client import call_llm
        text = call_llm("Say 'hello' and nothing else.")
        print(f"    Response: {text}")

        if text.startswith("LLM error:"):
            return FAIL, text

        return PASS, f"OpenRouter responded: '{text[:50]}'"

    except Exception as e:
        return FAIL, str(e)


# ===========================================================
#  11. PROMPTS
# ===========================================================
def test_prompts():
    from rag.prompts import RAG_PROMPT

    # Verify template placeholders exist
    assert "{context}" in RAG_PROMPT, "RAG_PROMPT should have {context} placeholder"
    assert "{question}" in RAG_PROMPT, "RAG_PROMPT should have {question} placeholder"

    # Test formatting works
    formatted = RAG_PROMPT.format(
        context="Test context here.",
        question="What is AI?"
    )
    print(f"    Template length: {len(RAG_PROMPT)} chars")
    print(f"    Formatted preview: {formatted[:80].strip()}...")

    assert "Test context here." in formatted
    assert "What is AI?" in formatted

    return PASS, "Prompt template is valid and formattable"


# ===========================================================
#  12. RAG PIPELINE -- import check
# ===========================================================
def test_rag_pipeline_import():
    try:
        from rag.rag_pipeline import run_rag
        print(f"    run_rag: {run_rag}")

        import inspect
        sig = inspect.signature(run_rag)
        print(f"    Signature: run_rag{sig}")

        return PASS, "RAG pipeline imported successfully"
    except ImportError as e:
        return FAIL, str(e)


# ===========================================================
#  13. INGESTION PIPELINE -- import check
# ===========================================================
def test_ingestion_pipeline_import():
    try:
        from ingestion.pipeline import run_ingestion
        print(f"    run_ingestion: {run_ingestion}")

        import inspect
        sig = inspect.signature(run_ingestion)
        print(f"    Signature: run_ingestion{sig}")

        return PASS, "Ingestion pipeline imported successfully"
    except ImportError as e:
        return FAIL, str(e)


# ===========================================================
#  14. MISSING __init__.py CHECK
# ===========================================================
def test_init_files():
    expected_dirs = [
        "ingestion",
        "ingestion/loaders",
        "ingestion/splitters",
        "ingestion/embeddings",
        "ingestion/vectorstore",
        "rag",
    ]

    missing = []
    present = []

    for d in expected_dirs:
        init_path = PROJECT_ROOT / d / "__init__.py"
        if init_path.exists():
            present.append(d)
        else:
            missing.append(d)

    for d in present:
        print(f"    [OK]   {d}/__init__.py")
    for d in missing:
        print(f"    [MISS] {d}/__init__.py  (MISSING)")

    if missing:
        return WARN, (
            f"{len(missing)} __init__.py files missing: {', '.join(missing)}\n"
            "    -> Imports work here because sys.path is set up,\n"
            "      but may fail in production without these files."
        )

    return PASS, f"All {len(expected_dirs)} __init__.py files present"


# ===========================================================
#  MAIN -- Run all tests
# ===========================================================
def main():
    print("=" * 60)
    print("  Evo_lab_Rag -- Component Test Suite")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 60)

    tests = [
        ("1.  Config Loading",                test_config),
        ("2.  Loader -- clean_text()",         test_loader_clean_text),
        ("3.  Loader -- load_pdf()",           test_loader_pdf),
        ("4.  Splitter -- token_based",        test_splitter),
        ("5.  Embeddings -- embed_text/texts", test_embeddings),
        ("6.  Database Connection",           test_db_connection),
        ("7.  PGVector Store -- import",       test_pgvector_store_import),
        ("8.  Retriever -- import",            test_retriever_import),
        ("9.  LLM Client -- import & config",  test_llm_client_import),
        ("10. LLM Client -- live call",        test_llm_live_call),
        ("11. Prompts -- template",            test_prompts),
        ("12. RAG Pipeline -- import",         test_rag_pipeline_import),
        ("13. Ingestion Pipeline -- import",   test_ingestion_pipeline_import),
        ("14. Missing __init__.py check",     test_init_files),
    ]

    for name, func in tests:
        run_test(name, func)

    # -- Summary ------------------------------------------
    print("\n")
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    warned = sum(1 for _, s, _ in results if s == WARN)

    for name, status, detail in results:
        # Truncate detail for summary
        short = detail.split("\n")[0] if detail else ""
        if len(short) > 55:
            short = short[:52] + "..."
        print(f"  {status}  {name:<36} {short}")

    print(f"\n  Total: {len(results)} | "
          f"Passed: {passed} | "
          f"Failed: {failed} | "
          f"Warnings: {warned} | "
          f"Skipped: {skipped}")

    if failed > 0:
        print(f"\n  >> {failed} test(s) FAILED -- see details above")
    else:
        print(f"\n  >> All tests passed!")

    print("=" * 60)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
