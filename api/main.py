"""Re-export the FastAPI app so callers can use either `main:app` or `api.main:app`."""

from main import app  # noqa: F401  – re-exported for Render / uvicorn entrypoint

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
