# Frontend API Handoff

## Local backend URL

Run the API backend locally:

```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000
```

Local base URL:

```text
http://localhost:8000
```

## Expose through ngrok

In a second terminal:

```powershell
ngrok http 8000
```

ngrok will show a forwarding URL like:

```text
https://abc123.ngrok-free.app
```

Give this base URL to the frontend developer:

```text
https://abc123.ngrok-free.app
```

## Authentication

Frontend requests must send:

```text
X-API-Key: <frontend api key>
```

The backend reads valid keys from:

```text
API_KEYS=frontend-key,backend-key
```

## Health check

```http
GET /health
```

Example:

```powershell
Invoke-RestMethod https://abc123.ngrok-free.app/health
```

## RAG query endpoint

```http
POST /v1/rag/query
```

Request body:

```json
{
  "question": "What is this document about?",
  "top_k": 5,
  "score_threshold": 0.0,
  "metadata_filter": null
}
```

Response body:

```json
{
  "answer": "Generated answer from the RAG backend.",
  "model": "qwen3",
  "top_k": 5
}
```

## Frontend fetch example

```javascript
const response = await fetch("https://abc123.ngrok-free.app/v1/rag/query", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "frontend-key"
  },
  body: JSON.stringify({
    question: "What does the document conclude?",
    top_k: 5,
    score_threshold: 0.0,
    metadata_filter: null
  })
});

const data = await response.json();
console.log(data.answer);
```

## Notes

- Keep Ollama running before starting the API backend.
- Keep ngrok running while the frontend developer tests.
- The ngrok URL changes each time unless a reserved ngrok domain is configured.
- Do not share database credentials or Ollama internals with the frontend.
